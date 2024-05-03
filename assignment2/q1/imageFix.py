import cv2
import matplotlib.pyplot as plt
import numpy as np


def brightness_and_contrast_stretching(image, a, b):
    x0 = image.min()
    x1 = image.max()
    y0 = max(a * x0 + b, 0)
    y1 = min(a * x1 + b, 255)
    stretched_image = ((1 - (image - x0) / (x1 - x0)) * y0 + ((image - x0) / (x1 - x0) * y1)).astype('uint8')
    return stretched_image


def gama_correction(image, gamma):
    image = np.power(image / 255.0, gamma) * 255
    return np.clip(image, 0, 255).astype(np.uint8)


def histogram_equalization(image):
    return cv2.equalizeHist(image)


def apply_fix(image, id):
    if id == 1:
        return histogram_equalization(image)
    elif id == 2:
        return gama_correction(image, 0.6)
    else:
        return brightness_and_contrast_stretching(image, 0.80, -15)


for i in range(1, 4):
    if i == 1:
        path = f'{i}.png'
    else:
        path = f'{i}.jpg'
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    fixed_image = apply_fix(image, i)
    plt.imsave(f'{i}_fixed.jpg', fixed_image, cmap='gray', vmin=0, vmax=255)
