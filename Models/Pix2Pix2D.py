import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
import time
from Tools.losses import generator_loss, discriminator_loss
import os
#from IPython import display
import numpy as np
from batchup import data_source
from tqdm import tqdm



OUTPUT_CHANNELS = 1


def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result


def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result




def Generator(input_shape):
  inputs = tf.keras.layers.Input(shape=input_shape)

  down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
    downsample(128, 4),  # (batch_size, 64, 64, 128)
    downsample(256, 4),  # (batch_size, 32, 32, 256)
    downsample(512, 4),  # (batch_size, 16, 16, 512)
    downsample(512, 4),  # (batch_size, 8, 8, 512)
    downsample(512, 4),  # (batch_size, 4, 4, 512)
    downsample(512, 4),  # (batch_size, 2, 2, 512)
    downsample(512, 4),  # (batch_size, 1, 1, 512)
  ]

  up_stack = [
    upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
    upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
    upsample(512, 4),  # (batch_size, 16, 16, 1024)
    upsample(256, 4),  # (batch_size, 32, 32, 512)
    upsample(128, 4),  # (batch_size, 64, 64, 256)
    upsample(64, 4),  # (batch_size, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh')  # (batch_size, 256, 256, 3)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)



def Discriminator(input_shape):
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=input_shape, name='input_image')
  tar = tf.keras.layers.Input(shape=input_shape, name='target_image')

  x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

  down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
  down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
  down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)




class Pix2Pix():
  def __init__(self, args):
    self.args = args
    # default input shape = (256, 256, 3)
    self.generator = Generator(args['input_shape'])
    self.discriminator = Discriminator(args['input_shape'])

    # default learning rate for both generator and discriminatore = 2e-4
    self.generator_optimizer = tf.keras.optimizers.Adam(args['lr_G'], beta_1=0.5)
    self.discriminator_optimizer = tf.keras.optimizers.Adam(args['lr_D'], beta_1=0.5)
    
    log_dir = args['log_dir']
    self.summary_writer = tf.summary.create_file_writer(
      log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

  @tf.function
  def train_step(self, input_image, target, step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      gen_output = self.generator(input_image, training=True)

      disc_real_output = self.discriminator([input_image, target], training=True)
      disc_generated_output = self.discriminator([input_image, gen_output], training=True)

      target = tf.cast(target, tf.float32)   # added to prevent type mismatch error

      gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
      disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            self.generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                self.discriminator.trainable_variables)

    self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                            self.generator.trainable_variables))
    self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                self.discriminator.trainable_variables))

    with self.summary_writer.as_default():
      tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
      tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
      tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
      tf.summary.scalar('disc_loss', disc_loss, step=step//1000)

      return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss



  def fit(self, train_data, test_data):
    example_us, example_mri = next(iter(train_data))
    example_input = np.expand_dims(example_us, axis=0)
    example_target = np.expand_dims(example_mri, axis=0)
    max_epochs = self.args['max_epochs']
    start = time.time()
    for epoch in range(max_epochs):
      for step, (input_image, target) in tqdm(enumerate(train_data.batch(self.args['batch_size']))):
        gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = self.train_step(input_image, target, step)

        # save a sample generated image
        if step % self.args['save_image_freq'] == 0:
          sample_generated_image_path = os.path.join(self.args['sample_generated_images_dir'], f'SampleGenerated_epoch{epoch+1}_step{step}')
          generate_images(self.generator, example_input, example_target, sample_generated_image_path)

        # saved models
        if step % self.args['save_weights_freq'] == 0:
          self.generator.save_weights(os.path.join(self.args['last_models_dir'], 'Generator.h5'))
          self.discriminator.save_weights(os.path.join(self.args['last_models_dir'], 'Discriminator.h5'))
      
      #display.clear_output(wait=True)

      print(f'Epoch: {epoch+1}/{max_epochs} --- gen_total_loss: {float(gen_total_loss):.3f}  --- disc_loss: {float(disc_loss):.3f}  --- Time: {time.time()-start:.2f} sec')
      start = time.time()

      

      
      
      

def generate_images(model, test_input, tar, save_to):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15, 15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # Getting the pixel values in the [0, 1] range to plot.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.savefig(save_to)














if __name__ == '__main__':
    generator = Generator()
    generator.summary()
    #tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)



