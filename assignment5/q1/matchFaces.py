import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import warnings

from scipy.ndimage import maximum_filter

warnings.filterwarnings("ignore")


def scale_down(image, resize_ratio):
    rows, cols = image.shape
    preserved_rows = int(rows * resize_ratio)
    preserved_cols = int(cols * resize_ratio)

    # calculate the center
    center_row = rows // 2
    center_col = cols // 2

    # Compute the boundaries of the preserved region
    start_row = center_row - preserved_rows // 2
    end_row = start_row + preserved_rows
    start_col = center_col - preserved_cols // 2
    end_col = start_col + preserved_cols

    # Compute the Fourier transform of the image
    image_fft = fftshift(fft2(image))

    # Extract the central region in the Fourier domain
    image_fft_preserved = np.zeros((end_row - start_row, end_col - start_col))
    image_fft_preserved = image_fft[start_row:end_row, start_col:end_col]
    # Compute the inverse Fourier transform to obtain the scaled down image
    scaled_image = np.abs(ifft2(ifftshift(image_fft_preserved)))

    return scaled_image


def scale_up(image, resize_ratio):
    rows, cols = image.shape
    preserved_rows = int(rows * resize_ratio)
    preserved_cols = int(cols * resize_ratio)

    # Calculate the difference in dimensions
    add_rows = preserved_rows - rows
    add_cols = preserved_cols - cols

    # Compute the Fourier transform of the image
    image_fft = fftshift(fft2(image))

    # Add zero padding, adjusting to ensure exact dimensions
    padded_fourier_transform = np.pad(image_fft, ((add_rows // 2, add_rows - add_rows // 2),
                                                  (add_cols // 2, add_cols - add_cols // 2)), mode='constant')
    larger_image = np.abs(ifft2(ifftshift(padded_fourier_transform)))
    return larger_image


def ncc_2d(image, pattern):
    # Get dimensions of image and pattern
    image_height, image_width = image.shape
    pattern_height, pattern_width = pattern.shape

    # Compute mean of pattern
    pattern_mean = np.mean(pattern)

    # Compute correlation between image and pattern
    cross_correlation = np.zeros_like(image, dtype=float)
    for i in range(image_height - pattern_height + 1):
        for j in range(image_width - pattern_width + 1):
            window = np.array(image[i:i + pattern_height, j:j + pattern_width])
            window_mean = np.mean(window)
            numerator = np.sum((window - window_mean) * (pattern - pattern_mean))
            denominator = np.sqrt(np.sum((window - window_mean) ** 2) * np.sum((pattern - pattern_mean) ** 2))
            if denominator == 0:
                cross_correlation[i, j] = 0
            else:
                cross_correlation[i, j] = numerator / denominator

    return cross_correlation


def display(image, pattern):
    plt.subplot(2, 3, 1)
    plt.title('Image')
    plt.imshow(image, cmap='gray')

    plt.subplot(2, 3, 3)
    plt.title('Pattern')
    plt.imshow(pattern, cmap='gray', aspect='equal')

    ncc = ncc_2d(image, pattern)

    plt.subplot(2, 3, 5)
    plt.title('Normalized Cross-Correlation Heatmap')
    plt.imshow(ncc ** 2, cmap='coolwarm', vmin=0, vmax=1, aspect='auto')

    cbar = plt.colorbar()
    cbar.set_label('NCC Values')

    plt.show()


def draw_matches(image, matches, pattern_size):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for point in matches:
        y, x = point
        top_left = (int(x - pattern_size[1] / 2), int(y - pattern_size[0] / 2))
        bottom_right = (int(x + pattern_size[1] / 2), int(y + pattern_size[0] / 2))
        cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 1)

    plt.imshow(image, cmap='gray')
    plt.show()

    cv2.imwrite(f"{CURR_IMAGE}_result.jpg", image)


CURR_IMAGE = "students"

image = cv2.imread(f'{CURR_IMAGE}.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
imagefl = image - cv2.GaussianBlur(image, (21, 21), 8) + 128

pattern = cv2.imread('template.jpg')
pattern = cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)
pattern = pattern - cv2.GaussianBlur(pattern, (23, 23), 4) + 128
# ############ DEMO #############
# display(image, pattern)

# ############ Students #############

# no false 2 not detected#
# image_scaling = 1.4
# pattern_scaling = 0.79
# threshold = 0.37

# 1 false 1 not detected
# image_scaling = 1.4
# pattern_scaling = 0.78
# threshold = 0.36

# all visible 2 false##
image_scaling = 1.4
pattern_scaling = 0.77
threshold = 0.355
image_scaled = scale_up(imagefl, image_scaling)
patten_scaled = scale_down(pattern, pattern_scaling)

display(image_scaled, patten_scaled)

ncc = ncc_2d(image_scaled, patten_scaled)
ncc[ncc != maximum_filter(ncc, size=(40, 40))] = 0
real_matches = np.argwhere(ncc > threshold)

# ####### DONT CHANGE THE NEXT TWO LINES #########
real_matches[:, 0] += patten_scaled.shape[0] // 2  # if pattern was not scaled, replace this with "pattern"
real_matches[:, 1] += patten_scaled.shape[1] // 2  # if pattern was not scaled, replace this with "pattern"
# If you chose to scale the original image, make sure to scale back the matches in the inverse resize ratio.

draw_matches(image, real_matches / image_scaling, patten_scaled.shape)
############ Crew #############
CURR_IMAGE = "thecrew"

image = cv2.imread(f'{CURR_IMAGE}.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

pattern = cv2.imread('template.jpg')
pattern = cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)

# 2 missing
# image_scaling = 1.52
# pattern_scaling = 0.36
# threshold = 0.44

# 1 missing 3 false
image_scaling = 1.7
pattern_scaling = 0.37
threshold = 0.4

image_scaled = scale_up(image, image_scaling)
patten_scaled = scale_down(pattern, pattern_scaling)

display(image_scaled, patten_scaled)

ncc = ncc_2d(image_scaled, patten_scaled)
ncc[ncc != maximum_filter(ncc, size=(40, 15))] = 0
real_matches = np.argwhere(ncc > threshold)

######### DONT CHANGE THE NEXT TWO LINES #########
real_matches[:, 0] += patten_scaled.shape[0] // 2  # if pattern was not scaled, replace this with "pattern"
real_matches[:, 1] += patten_scaled.shape[1] // 2  # if pattern was not scaled, replace this with "pattern"

# If you chose to scale the original image, make sure to scale back the matches in the inverse resize ratio.

draw_matches(image, real_matches / image_scaling, patten_scaled.shape)
