import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, median_filter


def clean_baby(im):
    clean_im = cv2.medianBlur(im, ksize=3)
    first = np.float32([[5, 19],
                        [111, 19],
                        [5, 131],
                        [111, 131]])

    second = np.float32([[77, 162],
                         [147, 117],
                         [132, 244],
                         [245, 160]])

    third = np.float32([[181, 4],
                        [249, 70],
                        [120, 50],
                        [176, 121]])

    final = np.float32([[0, 0],
                        [255, 0],
                        [0, 255],
                        [255, 255]])

    trans1 = cv2.getPerspectiveTransform(first, final)
    clean_im1 = cv2.warpPerspective(clean_im, trans1, (256, 256), flags=cv2.INTER_CUBIC)

    trans2 = cv2.getPerspectiveTransform(second, final)
    clean_im2 = cv2.warpPerspective(clean_im, trans2, (256, 256), flags=cv2.INTER_CUBIC)

    trans3 = cv2.getPerspectiveTransform(third, final)
    clean_im3 = cv2.warpPerspective(clean_im, trans3, (256, 256), flags=cv2.INTER_CUBIC)

    image_arr = [clean_im3, clean_im2, clean_im1]
    image_arr = np.array(image_arr)
    clean_im = np.mean(image_arr, axis=0)
    # cv2.imwrite('edited_baby.jpg', clean_im)
    return clean_im


def clean_windmill(im):
    # Fourier transform of the image
    f_transform = np.fft.fft2(im)
    f_shift = np.fft.fftshift(f_transform)

    f_shift[124][100] = 0
    f_shift[132][156] = 0

    f_shift = np.fft.ifftshift(f_shift)
    reconstructed_image = np.fft.ifft2(f_shift)
    # cv2.imwrite('edited_windmill.jpg', np.abs(reconstructed_image))
    return np.abs(reconstructed_image)


def clean_watermelon(im):
    sharpening_kernel = np.array(
        [[0, -1, 0],
         [-1, 4, -1],
         [0, -1, 0]], dtype=np.float32) * 3

    # add the image itself
    sharpening_kernel[1, 1] += 1

    # cv2.imwrite('edited_watermelon.jpg', cv2.filter2D(im, -1, sharpening_kernel))
    return cv2.filter2D(im, -1, sharpening_kernel)


def clean_umbrella(im):
    kernel_value = np.zeros([256, 256])
    kernel_value[0][0] = 0.5
    kernel_value[4][79] = 0.5
    kernel_forieh = np.fft.fft2(kernel_value)
    # so we don`t get high values and don`t divide by 0
    kernel_forieh[abs(kernel_forieh) < 0.0000009] = 1

    im_forieh = np.fft.fft2(im)
    fft_clean_im = im_forieh / kernel_forieh
    clean_im = np.abs(np.fft.ifft2(fft_clean_im))
    # cv2.imwrite('edited__umbrella.jpg', clean_im)
    return clean_im


def clean_USAflag(im):
    # so we keep the left square
    clean_im = im.copy()

    clean_im = median_filter(clean_im, [1, 50])
    clean_im = gaussian_filter(clean_im, sigma=(0, 7), order=0)

    # add the square back
    clean_im[:90, :142] = im[:90, :142]
    # cv2.imwrite('edited_USAflag.jpg', clean_im)
    return clean_im


def clean_house(im):
    kernel_value = np.zeros([191, 191])
    # set the first 10 pixels of the first rwo to 0.1
    kernel_value[0][0:10] = 0.1
    kfft = np.fft.fft2(kernel_value)
    kfft[abs(kfft) < 0.01] = 1
    tran_im = np.fft.fft2(im)
    clean_im = abs(np.fft.ifft2(tran_im / kfft))
    # cv2.imwrite('edited_house.jpg', clean_im)
    return clean_im


def clean_bears(im):
    # cv2.imwrite('edited_bear.jpg', brightness_and_contrast_stretching(im, 3.6, -100))
    return brightness_and_contrast_stretching(im, 3.6, -100)


# from the previous assignment
def brightness_and_contrast_stretching(image, a, b):
    x0 = image.min()
    x1 = image.max()
    y0 = max(a * x0 + b, 0)
    y1 = min(a * x1 + b, 255)
    stretched_image = ((1 - (image - x0) / (x1 - x0)) * y0 + ((image - x0) / (x1 - x0) * y1)).astype('uint8')
    return stretched_image
