import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import cv2
import matplotlib.pyplot as plt

image_path = 'zebra.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

H, W = image.shape

zebra_array = np.array(image)
gray_array = zebra_array

plt.figure(figsize=(10, 10))

plt.subplot(321)
plt.title('Original Grayscale Image')
plt.imshow(image, cmap='gray')

fourier_transform_shifted = fftshift(fft2(gray_array))

plt.subplot(322)
plt.title('Fourier Spectrum')
plt.imshow(np.log(1 + np.abs(fourier_transform_shifted)), cmap='gray')

padded_fourier_transform = np.pad(fourier_transform_shifted, ((H // 2, H // 2), (W // 2, W // 2)), mode='constant')

plt.subplot(323)
plt.title('Fourier Spectrum Zero Padding')
plt.imshow(np.log(1 + np.abs(padded_fourier_transform)), cmap='gray')

larger_image = np.abs(ifft2(ifftshift(padded_fourier_transform)))
larger_image *= 4
larger_image = larger_image.astype(np.uint8)
larger_image = (1 + np.abs(larger_image))

plt.subplot(324)
plt.title('Two Times Larger Grayscale Image')
plt.imshow(larger_image, cmap='gray')


def add_zeros_between(matrix):
    rows, cols = matrix.shape
    new_rows = rows * 2 - 1
    new_cols = cols * 2 - 1

    new_matrix = np.zeros((new_rows, new_cols), dtype=matrix.dtype)

    for i in range(rows):
        for j in range(cols):
            new_matrix[i * 2, j * 2] = matrix[i, j]

    return new_matrix


result_matrix = add_zeros_between(fourier_transform_shifted)

plt.subplot(325)
plt.title('Fourier Spectrum Four Copies')
plt.imshow(np.log(1 + np.abs(result_matrix)), cmap='gray')

four_times_image = np.abs(ifft2(ifftshift(result_matrix)))
four_times_image *= 4
four_times_image = np.uint8(four_times_image)

plt.subplot(326)
plt.title('Four Copies Grayscale Image')
plt.imshow(four_times_image, cmap='gray')
plt.show()
# plt.savefig('zebra_scaled.png')
