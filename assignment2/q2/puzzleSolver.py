import cv2
import numpy as np
import os
import shutil
import sys


# matches is of (3|4 X 2 X 2) size. Each row is a match - pair of (kp1,kp2) where kpi = (x,y)
def get_transform(matches, is_affine):
    src_points, dst_points = matches[:, 0], matches[:, 1]
    if is_affine:
        # Affine transformation
        transform_matrix, _ = cv2.estimateAffine2D(dst_points, src_points)
    else:
        # Projective transformation (homography)
        transform_matrix, _ = cv2.findHomography(dst_points, src_points, cv2.RANSAC)

    return transform_matrix


def stitch(img1, img2):
    # Convert images to grayscale
    gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Find non-black pixels in the second image
    colored_mask = (gray_img2 != 0)

    # Copy non-black pixels from the second image to the first image
    img1[colored_mask] = img2[colored_mask]

    return img1


# Output size is (w,h)
def inverse_transform_target_image(target_img, original_transform, output_size):
    # Get the height and width of the first image
    first_image_height, first_image_width = output_size
    if transform_matrix.shape == (2, 3):
        inverse_transformed_image = cv2.warpAffine(target_img, original_transform,
                                                   (first_image_width, first_image_height)
                                                   , flags=2, borderMode=cv2.BORDER_TRANSPARENT)
    else:
        output_size = (first_image_width, first_image_height)
        inverse_transformed_image = cv2.warpPerspective(target_img, original_transform, output_size, flags=2,
                                                        borderMode=cv2.BORDER_TRANSPARENT)
    return inverse_transformed_image


# returns list of pieces file names
def prepare_puzzle(puzzle_dir):
    edited = os.path.join(puzzle_dir, 'abs_pieces')
    if os.path.exists(edited):
        shutil.rmtree(edited)
    os.mkdir(edited)

    affine = 4 - int("affine" in puzzle_dir)

    matches_data = os.path.join(puzzle_dir, 'matches.txt')
    n_images = len(os.listdir(os.path.join(puzzle_dir, 'pieces')))

    matches = np.loadtxt(matches_data, dtype=np.int64).reshape(n_images - 1, affine, 2, 2)

    return matches, affine == 3, n_images


if __name__ == '__main__':
    lst = ['puzzle_affine_1', 'puzzle_affine_2', 'puzzle_homography_1']
    # lst = ['puzzle_homography_1']

    for puzzle_dir in lst:
        print(f'Starting {puzzle_dir}')

        puzzle = os.path.join('puzzles', puzzle_dir)

        pieces_pth = os.path.join(puzzle, 'pieces')
        edited = os.path.join(puzzle, 'abs_pieces')

        matches, is_affine, n_images = prepare_puzzle(puzzle)

        # here we get the pictures and the names from the pieces_path
        imgNames = []
        images = []
        for file_name in os.listdir(pieces_pth):
            if file_name.endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(pieces_pth, file_name)
                image = cv2.imread(file_path)
                images.append(image)
                imgNames.append(file_name)

        # here we put the first image in the abs
        sol_file = str(imgNames[0])[0:-4] + "_relative.jpeg"
        cv2.imwrite(os.path.join(edited, sol_file), images[0])

        # here we want to update the final puzzle so we start as the first one
        final_puzzle = images[0]

        for i in range(len(imgNames) - 1):
            transform_matrix = get_transform(matches[i], is_affine)

            transformed_image = inverse_transform_target_image(images[i + 1], transform_matrix, images[0].shape[:2])
            sol_file = str(imgNames[i + 1])[0:-4] + "_relative.jpeg"
            cv2.imwrite(os.path.join(edited, sol_file), transformed_image)

            final_puzzle = stitch(final_puzzle, transformed_image)
        sol_file = f'solution.jpg'
        cv2.imwrite(os.path.join(puzzle, sol_file), final_puzzle)
