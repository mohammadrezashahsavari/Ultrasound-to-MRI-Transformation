import keras
import matplotlib.pyplot as plt
import numpy as np
import os


class GANMonitor(keras.callbacks.Callback):
    """A callback to generate and save images after each epoch"""

    def __init__(self, test_dataset, save_images_to):
        example_us, example_mri = next(iter(test_dataset))
        self.example_us = np.expand_dims(example_us, axis=0)
        self.example_mri = np.expand_dims(example_mri, axis=0)
        self.save_images_to = save_images_to

    def on_epoch_end(self, epoch, logs=None):
        pred_mri = self.model.gen_G(self.example_us)
        plt.figure(figsize=(15, 15))

        display_list = [self.example_us, self.example_mri, pred_mri]
        title = ['Input Image', 'Ground Truth', 'Predicted Image']

        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.title(title[i])
            # Getting the pixel values in the [0, 1] range to plot.
            plt.imshow(np.squeeze(display_list[i]) * 0.5 + 0.5)
            plt.axis('off')

        save_to = os.path.join(self.save_images_to, f'Genrated_epoch{epoch}')
        plt.savefig(save_to)
        plt.close()


'''
class GANMonitor(keras.callbacks.Callback):
    """A callback to generate and save images after each epoch"""

    def __init__(self, num_img=4):
        self.num_img = num_img

    def on_epoch_end(self, epoch, logs=None):
        _, ax = plt.subplots(4, 2, figsize=(12, 12))
        for i, img in enumerate(test_horses.take(self.num_img)):
            prediction = self.model.gen_G(img)[0].numpy()
            prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
            img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)

            ax[i, 0].imshow(img)
            ax[i, 1].imshow(prediction)
            ax[i, 0].set_title("Input image")
            ax[i, 1].set_title("Translated image")
            ax[i, 0].axis("off")
            ax[i, 1].axis("off")

            prediction = keras.utils.array_to_img(prediction)
            prediction.save(
                "generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch + 1)
            )
        plt.show()
        plt.close()

'''