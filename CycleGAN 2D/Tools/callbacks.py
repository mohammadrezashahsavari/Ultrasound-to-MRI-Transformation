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

