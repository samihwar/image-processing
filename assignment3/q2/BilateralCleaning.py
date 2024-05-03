import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def clean_Gaussian_noise_bilateral(im, radius, stdSpatial, stdIntensity):
    im = im.astype(np.float64)
    # Define the size of the image
    height, width = im.shape

    # Initialize the clean image with zeros
    cleanIm = np.zeros_like(im, dtype=np.float64)

    # Generate indices for spatial distances
    x, y = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1))

    # Calculate spatial Gaussian mask
    gs = np.exp(-(x ** 2 + y ** 2) / (2 * stdSpatial ** 2))

    # Iterate over each pixel in the image
    for i in range(height):
        for j in range(width):
            # Define the local window
            window = im[max(0, i - radius): min(height, i + radius + 1), max(0, j - radius): min(width, j + radius + 1)]

            # Calculate intensity difference
            intensity_diff = window - im[i, j]

            # Calculate intensity Gaussian mask
            gi = np.exp(-(intensity_diff ** 2) / (2 * stdIntensity ** 2))

            # Calculate weighted sum
            weighted_sum = np.sum(gi * gs[:window.shape[0], :window.shape[1]] * window)

            # Calculate the normalization factor
            normalization_factor = np.sum(gi * gs[:window.shape[0], :window.shape[1]])

            # Update the pixel value in the clean image
            cleanIm[i, j] = weighted_sum / normalization_factor

    cleanIm = np.clip(cleanIm, 0, 255).astype(np.uint8)

    return cleanIm


image_a = cv2.imread('balls.jpg', cv2.IMREAD_GRAYSCALE)
clear_image_a = clean_Gaussian_noise_bilateral(image_a, 30, 15, 10)
output_path = os.path.join(os.getcwd(), "edited_balls.jpg")
cv2.imwrite(output_path, clear_image_a)

image_b = cv2.imread('taj.jpg', cv2.IMREAD_GRAYSCALE)
clear_image_b = clean_Gaussian_noise_bilateral(image_b, 10, 25, 35)
output_path = os.path.join(os.getcwd(), "edited_taj.jpg")
cv2.imwrite(output_path, clear_image_b)

image_c = cv2.imread('NoisyGrayImage.png', cv2.IMREAD_GRAYSCALE)
clear_image_c = clean_Gaussian_noise_bilateral(image_c, 10, 20, 80)
output_path = os.path.join(os.getcwd(), "edited_NoisyGrayImage.png")
cv2.imwrite(output_path, clear_image_c)
