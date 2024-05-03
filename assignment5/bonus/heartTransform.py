import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def parametric_x(t):
	return 14.5 * np.sin(t)**3
# a + r * 14.5 * np.sin(np.deg2rad(t)) ** 3


def parametric_y(t):
	return 0.5 * np.cos(4 * t) + 2 * np.cos(3 * t) + 4 * np.cos(2 * t) - 13 * np.cos(t)
# b + r * (0.5 * np.cos(4 * np.deg2rad(t)) + 2 * np.cos(3 * np.deg2rad(t)) + 4 * np.cos(2 * np.deg2rad(t)) - 13 * np.cos(np.deg2rad(t)))


'''
This is an example implementation for if we did a circle's transform.

def parametric_x(t):
	return np.cos(t)

def parametric_y(t):
	return np.sin(t)
	
'''


def find_hough_shape(image, edge_image, r_min, r_max, bin_threshold):
	img_height, img_width = image.shape[:2]

	# Note that cos and sin work with radians
	thetas = np.arange(0, 360, step=2)
	rs = np.arange(r_min, r_max, 0.5)

	# Calculate Cos(theta) and Sin(theta), it will be required later
	cos_thetas = parametric_y(np.deg2rad(thetas))
	sin_thetas = parametric_x(np.deg2rad(thetas))

	# Quantize thetas and radii
	shape_candidates = [(r, theta) for r in rs for theta in thetas]

	# Hough Accumulator. We are using defaultdict instead of standard dict as this will initialize for keys which are not already present in the dictionary instead of throwing an exception.
	accumulator = defaultdict(int)

	edge_points =np.argwhere(edge_image > 0)
	for point in edge_points:
		y, x = point
		for r, theta in shape_candidates:
			theta_index = theta//2
			x_center = x - int(r * sin_thetas[theta_index])
			y_center = y - int(r * cos_thetas[theta_index])
			# Don't forget - the points should be integers!

			# vote for current candidate
			accumulator[(x_center, y_center, r)] += 1

	# Output image with detected lines drawn
	output_img = image.copy()
	# Output list of detected circles. A single circle would be a tuple of (x,y,r,threshold)
	# (x,y,r,theta,threshold)
	out_shapes = []

	# Sort the accumulator based on the votes for the candidate circles
	num_thetas = len(thetas)
	for candidate_shape, votes in sorted(accumulator.items(), key=lambda i: -i[1]):
		x, y, r = candidate_shape
		current_vote_percentage = votes / num_thetas
		if current_vote_percentage > bin_threshold:
			out_shapes.append((x, y, r, current_vote_percentage))

		# Your code goes here
		# If the current_vote_percentage is larger than bin_threshold then we want that shape. append x,y,r and the precentage to the out_shapes.


	# DO NOT EDIT
	pixel_threshold = 10
	postprocess_shapes = []
	for x, y, r, v in out_shapes:
		# Exclude shapes that are too close of each other
		# Remove nearby duplicate circles based on postprocess_shapes
		if all(abs(x - xc) > pixel_threshold or abs(y - yc) > pixel_threshold > pixel_threshold and abs(r - rc) > pixel_threshold for xc, yc, rc, v in postprocess_shapes):
			postprocess_shapes.append((x, y, r, v))
	out_shapes = postprocess_shapes


	# Draw shortlisted hearts on the output image
	for x, y, r, v in out_shapes:

		# We found heart at (x,y) with radius r.
		# Generate all the x1 and y1 points of that heart.
		# theta between 0 and 2pi
		t = np.linspace(0, 2 * np.pi, 100)
		x1 = x + r * parametric_x(t)
		y1 = y + r * parametric_y(t)

		colors = [
			(255, 0, 0),  # 'b' (blue)
			(0, 255, 0),  # 'g' (green)
			(255, 255, 0),  # 'c' (cyan)
			(255, 0, 255),  # 'm' (magenta)
			(0, 255, 255),  # 'y' (yellow)
			(0, 0, 0),  # 'k' (black)
		]

		color_chars = ['b', 'g', 'c', 'm', 'y', 'k']

		id = np.random.randint(len(colors))
		color1 = colors[id]
		color2 = color_chars[id]

		plt.plot(x1, y1, markersize=1.5, color=color2)
		output_img = cv2.circle(output_img, (x, y), 1, color1, -1)
		print(x, y, r, v)

	return output_img


# IMAGE_NAME = "simple"
#
# image = cv2.imread(f'{IMAGE_NAME}.jpg')
# edge_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# min_edge_threshold, max_edge_threshold = 100, 200
# edge_image = cv2.Canny(edge_image, min_edge_threshold, max_edge_threshold)
#
# r_min = 6.95
# r_max = 7
#
# bin_threshold = 0.15
#
# if edge_image is not None:
#
# 	print ("Attempting to detect Hough hearts...")
# 	results_img = find_hough_shape(image, edge_image, r_min, r_max, bin_threshold)
#
# 	if results_img is not None:
# 		plt.imshow(cv2.cvtColor(results_img, cv2.COLOR_BGR2RGB))
# 		plt.show()
# 		cv2.imwrite(f'{IMAGE_NAME}_detected.jpg', results_img)
# 	else:
# 		print ("Error in input image!")
#
# 	print ("Hough hearts detection complete!")
#
# IMAGE_NAME = "med"
#
# image = cv2.imread(f'{IMAGE_NAME}.jpg')
# edge_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# min_edge_threshold, max_edge_threshold = 100, 200
# edge_image = cv2.Canny(edge_image, min_edge_threshold, max_edge_threshold)
#
# r_min = 2.1
# r_max = 5.1
# bin_threshold = 0.29
#
# if edge_image is not None:
#
# 	print ("Attempting to detect Hough hearts...")
# 	results_img = find_hough_shape(image, edge_image, r_min, r_max, bin_threshold)
#
# 	if results_img is not None:
# 		plt.imshow(cv2.cvtColor(results_img, cv2.COLOR_BGR2RGB))
# 		plt.show()
# 		cv2.imwrite(f'{IMAGE_NAME}_detected.jpg', results_img)
# 	else:
# 		print ("Error in input image!")
#
# 	print ("Hough hearts detection complete!")
#
# print("hello world")
#
IMAGE_NAME = "hard"

image = cv2.imread(f'{IMAGE_NAME}.jpg')
edge_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

min_edge_threshold, max_edge_threshold = 100, 200
edge_image = cv2.Canny(edge_image, min_edge_threshold, max_edge_threshold)

r_min = 2.8
r_max = 11.5

bin_threshold = 0.2447

if edge_image is not None:

	print ("Attempting to detect Hough hearts...")
	results_img = find_hough_shape(image, edge_image, r_min, r_max, bin_threshold)

	if results_img is not None:
		plt.imshow(cv2.cvtColor(results_img, cv2.COLOR_BGR2RGB))
		plt.show()
		cv2.imwrite(f'{IMAGE_NAME}_detected.jpg', results_img)
	else:
		print ("Error in input image!")

	print ("Hough hearts detection complete!")