import os

import numpy as np
import cv2

noisy_image = cv2.imread('broken.jpg', 0)

deionised_image = cv2.medianBlur(noisy_image, 5)  # Kernel size 5x5

# now we will aplay sharpening
# defining the kernel for sharpening
kernel_sharpening = np.array([[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]])

# # Apply the sharpening kernel to the deionised image
sharpened_image = cv2.filter2D(deionised_image, -1, kernel_sharpening)
output_path = os.path.join(os.getcwd(), "1_edited_broken.png")
cv2.imwrite(output_path, sharpened_image)

# Load the noisy images
noisy_images = np.load('noised_images.npy')

# Denoise each noisy image using median filtering
deionised_image = [cv2.medianBlur(img, 5) for img in noisy_images]
deionised_image = [cv2.filter2D(img, -1, kernel_sharpening) for img in deionised_image]

# Average the deionised images
cleaned_image = np.mean(deionised_image, axis=0).astype(np.uint8)

output_path = os.path.join(os.getcwd(), "200_edited_broken.png")
cv2.imwrite(output_path, cleaned_image)
