import cv2
import numpy
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import warnings

warnings.filterwarnings("ignore")


def calculate_auc(x, y):
    # Custom function to calculate AUC using the trapezoidal rule
    area = 0.5 * np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]))
    return area


# Input: numpy array of images and number of gray levels to quantize the images down to
# Output: numpy array of images, each with only n_colors gray levels
def quantization(imgs_arr, n_colors=4):
    img_size = imgs_arr[0].shape
    res = []

    for img in imgs_arr:
        X = img.reshape(img_size[0] * img_size[1], 1)
        km = KMeans(n_clusters=n_colors)
        km.fit(X)

        img_compressed = km.cluster_centers_[km.labels_]
        img_compressed = np.clip(img_compressed.astype('uint8'), 0, 255)

        res.append(img_compressed.reshape(img_size[0], img_size[1]))

    return np.array(res)


# Input: A path to a folder and formats of images to read
# Output: numpy array of grayscale versions of images read from input folder, and also a list of their names
def read_dir(folder, formats=(".jpg", ".png")):
    image_arrays = []
    lst = [file for file in os.listdir(folder) if file.endswith(formats)]
    for filename in lst:
        file_path = os.path.join(folder, filename)
        image = cv2.imread(file_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_arrays.append(gray_image)
    return np.array(image_arrays), lst


# Input: image object (as numpy array) and the index of the wanted bin (between 0 to 9)
# Output: the height of the idx bin in pixels
def get_bar_height(image, idx):
    # Assuming the image is of the same pixel proportions as images supplied in this exercise, the following values
    # will work
    x_pos = 70 + 40 * idx
    y_pos = 274
    while image[y_pos, x_pos] == 0:
        y_pos -= 1
    return 274 - y_pos


def compare_hist(src_image, target):
    target_his = cv2.calcHist([target], [0], None, [256], [0, 256]).flatten()
    target_sumhis = np.cumsum(target_his)  # cumulative histogram

    (window) = np.lib.stride_tricks.sliding_window_view(src_image, target.shape)  # creating the window
    # (src_image.shape[0] - target.shape[0]) - 202, = 115
    # (src_image.shape[0] - target.shape[0]) - 200) = 113
    for x in range(113, 115):  # just the highest number
        for y in range(70):  # here we want to search the image from the left to 70 pxls to the right
            src_his = cv2.calcHist([window[x, y]], [0], None, [256], [0, 256]).flatten()
            src_sumhis = np.cumsum(src_his)  # cumulative histogram

            # here we want to calculate the difference between the two hists
            absolute_difference = np.sum(np.abs(src_sumhis - target_sumhis))
            if absolute_difference < 260:
                return True
    return False


# Sections a, b
images, names = read_dir('data')
numbers, _ = read_dir('numbers')
# cv2.imshow("",images[0])
# histr = cv2.calcHist([images[0]],[0],None,[256],[0,256])
# plt.plot(histr)
# plt.show()
# cv2.waitKey(0)

max_student_num = np.empty(7)  # to save the max students number in each image
thresholded_images = []
students_per_bin = np.zeros((7, 10))

for im in range(7):
    for i in range(10):
        if compare_hist(images[im], numbers[i]):
            max_student_num[im] = i

# quantization all the images
quan_imges = quantization(np.array(images), n_colors=3)

# and then threshold
for img in range(7):
    max_num = (max_student_num[img])
    thresholded_images.append(cv2.threshold(quan_imges[img], 240, 255, cv2.THRESH_BINARY)[1])
    for bar in range(10):
        students_per_bin[img][bar] = round(max_num * get_bar_height(thresholded_images[img], bar) / 156)
# printing...
for id in range(7):
    print("Histogram " + names[id] + " gave ", end="")
    for col in range(9):
        print(str(int(students_per_bin[id][col])) + ",", end="")
    print(str(int(students_per_bin[id][col])))
    # print(f'Histogram {names[id]} gave {students_per_bin[id]}')
