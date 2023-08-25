import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
import math


def Mean_absolute_error(gt_image, pred_image):
    # reshaping images for easier calculation
    gt_image = gt_image.reshape((-1, 1))
    pred_image = pred_image.reshape((-1, 1))
    return mean_absolute_error(gt_image, pred_image)

def Mean_absolute_percentage_error(gt_image, pred_image):
    # reshaping images for easier calculation
    gt_image = gt_image.reshape((-1, 1))
    pred_image = pred_image.reshape((-1, 1))
    return mean_absolute_percentage_error(gt_image, pred_image)

def Root_mean_squared_error(gt_image, pred_image):
    # reshaping images for easier calculation
    gt_image = gt_image.reshape((-1, 1))
    pred_image = pred_image.reshape((-1, 1))
    return math.sqrt(mean_squared_error(gt_image, pred_image))

def Structural_similarity(gt_image, pred_image):
    return structural_similarity(gt_image, pred_image, channel_axis=-1, gaussian_weights=True)

def Peak_signal_noise_ratio(gt_image, pred_image):
    return peak_signal_noise_ratio(gt_image, pred_image)



if __name__ == '__main__':
    a = np.array([[1, 1, 1], [1, 10000, 1]])
    b = np.array([[2, 6, 2], [2, 1, 6]])
    print(Mean_absolute_error(a, b))
    print(Mean_absolute_percentage_error(a, b))
    print(structural_similarity(a, b))


