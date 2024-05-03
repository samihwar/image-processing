import cv2
import numpy as np
import matplotlib.pyplot as plt


def my_pyrDown(image):
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    # Downsample by taking every 2nd pixel in each row and column
    downsampled = blurred[::2, ::2]

    return downsampled


def my_pyrUp(image):
    return cv2.resize(image, (2 * image.shape[1], 2 * image.shape[0]))


def get_gaussian_pyramid(image, levels):
    pyramid = [image.astype(np.float32)]
    for _ in range(levels - 1):
        image = my_pyrDown(image)
        pyramid.append(image.astype(np.float32))
    return pyramid


def get_laplacian_pyramid(image, levels):
    gaussian_pyramid = get_gaussian_pyramid(image, levels)
    laplacian_pyramid = []
    for i in range(levels - 1):
        expanded = my_pyrUp(gaussian_pyramid[i + 1])
        expanded = expanded[:gaussian_pyramid[i].shape[0], :gaussian_pyramid[i].shape[1]]  # Adjust size
        laplacian = cv2.subtract(gaussian_pyramid[i], expanded)
        laplacian_pyramid.append(laplacian)
    laplacian_pyramid.append(gaussian_pyramid[-1])
    return laplacian_pyramid


def blend_pyramids(levels, laplacian_pyramid_apple, laplacian_pyramid_orange):
    blended_pyramid = []
    for i in range(levels):
        mask = np.zeros_like(laplacian_pyramid_apple[i])
        cols = mask.shape[1]
        mask[:, :cols // 2 - int((levels-i)*1.5)] = 1
        for c in range(cols // 2 - int((levels-i)*1.5), cols // 2 + int((levels-i)*1.5)+1):
            mask[:, c] = 0.9 - 0.9 * (c-(cols // 2 - int((levels-i)*1.5))) / (2 * (levels-i))
        blended_level = laplacian_pyramid_orange[i] * (1-mask) + laplacian_pyramid_apple[i] * mask
        blended_pyramid.append(blended_level)
    return blended_pyramid


def restore_from_pyramid(pyramid):
    restored_image = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):
        expanded = my_pyrUp(restored_image)
        expanded = expanded[:pyramid[i].shape[0], :pyramid[i].shape[1]]
        restored_image = cv2.add(expanded, pyramid[i])
    return np.clip(restored_image, 0, 255).astype(np.uint8)


def validate_operation(img):
    levels = 6
    pyr = get_laplacian_pyramid(img, levels)
    img_restored = restore_from_pyramid(pyr)
    plt.title(f"MSE is {np.mean((img_restored - img) ** 2)}")
    plt.imshow(img_restored, cmap='gray')
    plt.show()


apple = cv2.imread('apple.jpg', cv2.IMREAD_GRAYSCALE)
orange = cv2.imread('orange.jpg', cv2.IMREAD_GRAYSCALE)

validate_operation(apple)
validate_operation(orange)

levels = 6
laplacian_pyramid_apple = get_laplacian_pyramid(apple, levels)
laplacian_pyramid_orange = get_laplacian_pyramid(orange, levels)

blended_pyramid = blend_pyramids(levels, laplacian_pyramid_orange, laplacian_pyramid_apple)

restored_image = restore_from_pyramid(blended_pyramid)

# Calculate MSE for the blended image compared to the original apple and orange images
mse_with_apple = np.mean((restored_image - apple) ** 2)
mse_with_orange = np.mean((restored_image - orange) ** 2)

print("MSE with original apple image:", mse_with_apple)
print("MSE with original orange image:", mse_with_orange)

plt.imshow(restored_image, cmap='gray')
plt.axis('off')
plt.show()

cv2.imwrite("result.jpg", restored_image)
