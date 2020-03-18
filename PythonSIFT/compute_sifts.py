import numpy as np
import cv2
import pysift
from matplotlib import pyplot as plt
from functools import cmp_to_key

from pysift import generateBaseImage, computeNumberOfOctaves, generateGaussianKernels, generateGaussianImages, \
    generateDoGImages, findScaleSpaceExtrema, removeDuplicateKeypoints, convertKeypointsToInputImageSize, generateDescriptors, \
    computeKeypointsWithOrientations, isPixelAnExtremum, localizeExtremumViaQuadraticFit, compareKeypoints


# default params
sigma = 1.6
num_intervals = 3
assumed_blur = 0.5
image_border_width = 5
contrast_threshold = 0.04


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


# query image
image = cv2.imread('box.png', 0)


# Compute SIFT keypoints and descriptors

image = image.astype('float32')
base_image = generateBaseImage(image, sigma, assumed_blur)
num_octaves = computeNumberOfOctaves(base_image.shape)
gaussian_kernels = generateGaussianKernels(sigma, num_intervals)
gaussian_images = generateGaussianImages(base_image, num_octaves, gaussian_kernels)
dog_images = generateDoGImages(gaussian_images)


# keypoints = findScaleSpaceExtrema(gaussian_images, dog_images, num_intervals, sigma, image_border_width)

print('Finding scale-space extrema...')
threshold = np.floor(0.5 * contrast_threshold / num_intervals * 255)  # from OpenCV implementation
keypoints = []

for octave_index, dog_images_in_octave in enumerate(dog_images):
    for image_index, (first_image, second_image, third_image) in enumerate(
            zip(dog_images_in_octave, dog_images_in_octave[1:], dog_images_in_octave[2:])):
        # (i, j) is the center of the 3x3 array
        for i in range(image_border_width, first_image.shape[0] - image_border_width):
            for j in range(image_border_width, first_image.shape[1] - image_border_width):
                if isPixelAnExtremum(first_image[i - 1:i + 2, j - 1:j + 2], second_image[i - 1:i + 2, j - 1:j + 2],
                                     third_image[i - 1:i + 2, j - 1:j + 2], threshold):
                    localization_result = localizeExtremumViaQuadraticFit(i, j, image_index + 1, octave_index,
                                                                          num_intervals, dog_images_in_octave, sigma,
                                                                          contrast_threshold, image_border_width)
                    if localization_result is not None:
                        keypoint, localized_image_index = localization_result
                        keypoints_with_orientations = computeKeypointsWithOrientations(keypoint, octave_index,
                                                                                       gaussian_images[octave_index][
                                                                                           localized_image_index])
                        for keypoint_with_orientation in keypoints_with_orientations:
                            keypoints.append(keypoint_with_orientation)


# keypoints = removeDuplicateKeypoints(keypoints)

print('Removing duplicate keypoints...')
if len(keypoints) < 2:
    print("Less than 2 keypoints!")

keypoints.sort(key=cmp_to_key(compareKeypoints))
unique_keypoints = [keypoints[0]]

for next_keypoint in keypoints[1:]:
    last_unique_keypoint = unique_keypoints[-1]
    if last_unique_keypoint.pt[0] != next_keypoint.pt[0] or \
       last_unique_keypoint.pt[1] != next_keypoint.pt[1] or \
       last_unique_keypoint.size != next_keypoint.size or \
       last_unique_keypoint.angle != next_keypoint.angle:
        unique_keypoints.append(next_keypoint)
keypoints = unique_keypoints


# keypoints = convertKeypointsToInputImageSize(keypoints)

print('Converting keypoints to input image size...')
converted_keypoints = []
for keypoint in keypoints:
    keypoint.pt = tuple(0.5 * np.array(keypoint.pt))
    keypoint.size *= 0.5
    keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
    converted_keypoints.append(keypoint)
keypoints = converted_keypoints


print('Computing desctiptors...')
descriptors = generateDescriptors(keypoints, gaussian_images)


print(keypoints, descriptors)