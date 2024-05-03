import cv2
import numpy as np
import os

# from matplotlib import pyplot as plt


def calc_mse(src_image, target):
    print(((src_image - target) ** 2).mean())


def gimme_image_and_transform(matrix, sol_file):
    image = cv2.imread("1.jpg", cv2.IMREAD_GRAYSCALE)
    # Convert the PIL Image to a NumPy array
    src_image = np.array(image)
    transformed_image = cv2.filter2D(src_image, -1, matrix, borderType=cv2.BORDER_CONSTANT)
    output_path = os.path.join(os.getcwd(), sol_file)
    cv2.imwrite(output_path, transformed_image)
    # cv2.imshow(sol_file, transformed_image)
    # cv2.waitKey()
    return transformed_image


if __name__ == '__main__':
    # here we do average on each row
    sol_file = "edited_image_1.jpg"
    output_path = os.path.join(os.getcwd(), sol_file)
    image = cv2.imread("1.jpg", cv2.IMREAD_GRAYSCALE)
    # a kernel with only the middle row containing 1's
    kernel = np.zeros_like(image, dtype=np.float32)
    middle_row = kernel.shape[0] // 2
    kernel[middle_row, :] = 1
    # Normalize the kernel
    kernel /= np.sum(kernel)
    transformed_image = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_WRAP)
    cv2.imwrite(output_path, transformed_image)
    calc_mse(cv2.imread("image_1.jpg", cv2.IMREAD_GRAYSCALE), transformed_image)

    # GaussianBlur
    sol_file = "edited_image_2.jpg"
    output_path = os.path.join(os.getcwd(), sol_file)
    image = cv2.imread("1.jpg", cv2.IMREAD_GRAYSCALE)
    src_image = np.array(image)
    transformed_image = cv2.GaussianBlur(src_image, (11, 11), 10, borderType=cv2.BORDER_WRAP)
    cv2.imwrite(output_path, transformed_image)
    calc_mse(cv2.imread("image_2.jpg", cv2.IMREAD_GRAYSCALE), transformed_image)

    # MedianBlur
    sol_file = "edited_image_3.jpg"
    output_path = os.path.join(os.getcwd(), sol_file)
    image = cv2.imread("1.jpg", cv2.IMREAD_GRAYSCALE)
    median_filtered_image = cv2.medianBlur(image, 11)
    cv2.imwrite(output_path, median_filtered_image)
    calc_mse(cv2.imread("image_3.jpg", cv2.IMREAD_GRAYSCALE), median_filtered_image)

    # get average from the 15 pixels around y-axis
    sol_file = "edited_image_4.jpg"
    _kernel = np.ones((15, 1)) / 15
    image = cv2.imread("1.jpg", cv2.IMREAD_GRAYSCALE)
    src_image = np.array(image)
    border_size = _kernel.shape[0] // 2
    padded_image = cv2.copyMakeBorder(src_image, border_size, border_size, border_size, border_size, cv2.BORDER_WRAP)
    transformed_image = cv2.filter2D(padded_image, -1, _kernel)
    transformed_image = transformed_image[border_size:-border_size, border_size:-border_size]

    output_path = os.path.join(os.getcwd(), sol_file)
    cv2.imwrite(output_path, transformed_image)
    calc_mse(cv2.imread("image_4.jpg", cv2.IMREAD_GRAYSCALE), transformed_image)

    # the Bsharp image when in the middle of the process when want to maek an image sharper using Gussian blur
    sol_file = "edited_image_5.jpg"
    output_path = os.path.join(os.getcwd(), sol_file)
    image = cv2.imread("1.jpg", cv2.IMREAD_GRAYSCALE)
    hpf = image - cv2.GaussianBlur(image, (17, 17), 4) + 127
    cv2.imwrite(output_path, hpf)
    calc_mse(cv2.imread("image_5.jpg", cv2.IMREAD_GRAYSCALE), hpf)

    # Laplacian filter
    sol_file = "edited_image_6.jpg"  # a little bit dark
    # Laplacian filter kernel
    laplacian_kernel = np.array(
        [[-0.2, -0.2, -0.2],
         [0, 0, 0],
         [0.2, 0.2, 0.2]], dtype=np.float32) * 1.7
    transformed_image = gimme_image_and_transform(laplacian_kernel, sol_file)
    calc_mse(cv2.imread("image_6.jpg", cv2.IMREAD_GRAYSCALE), transformed_image)

    # moving the image up half the way
    sol_file = "edited_image_7.jpg"
    output_path = os.path.join(os.getcwd(), sol_file)
    image = cv2.imread("1.jpg", cv2.IMREAD_GRAYSCALE)
    kernel_size = image.shape[0]
    kernel = np.zeros((kernel_size, 1), dtype=np.float32)
    kernel[0, 0] = 1
    transformed_image = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_WRAP)
    cv2.imwrite(output_path, transformed_image)
    calc_mse(cv2.imread("image_7.jpg", cv2.IMREAD_GRAYSCALE), transformed_image)

    # just converting to grayscale
    sol_file = "edited_image_8.jpg"
    # nothing
    id_kernel = np.array(
        [[0, 0, 0],
         [0, 1, 0],
         [0, 0, 0]], dtype=np.float32)
    transformed_image = gimme_image_and_transform(id_kernel, sol_file)
    calc_mse(cv2.imread("image_8.jpg", cv2.IMREAD_GRAYSCALE), transformed_image)

    # sharpening
    sol_file = "edited_image_9.jpg"
    # sharpening
    sharpening_kernel = np.array(
        [[-1, -1, -1],
         [-1, 12, -1],
         [-1, -1, -1]], dtype=np.float32) / 4
    transformed_image = gimme_image_and_transform(sharpening_kernel, sol_file)
    calc_mse(cv2.imread("image_9.jpg", cv2.IMREAD_GRAYSCALE), transformed_image)
