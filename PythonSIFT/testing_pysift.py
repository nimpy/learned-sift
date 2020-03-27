import numpy as np
import cv2
from matplotlib import pyplot as plt
from functools import cmp_to_key
import pickle
import datetime
from scipy import ndimage
import math
import imageio

from pysift import generateBaseImage, computeNumberOfOctaves, generateGaussianKernels, generateGaussianImages, \
    generateDoGImages, findScaleSpaceExtrema, removeDuplicateKeypoints, convertKeypointsToInputImageSize, generateDescriptors, \
    computeKeypointsWithOrientations, isPixelAnExtremum, localizeExtremumViaQuadraticFit, compareKeypoints, unpackOctave

from testing_pysift_utils import rotate_image_without_resize, pickle_keypoint_with_descriptor, unpickle_keypoint_with_descriptor


# default params
sigma = 1.6
num_intervals = 3
assumed_blur = 0.5
image_border_width = 5
contrast_threshold = 0.04


# query image
image = cv2.imread('box.png', 0)
use_pickled = True

# Compute SIFT keypoints and descriptors

image = image.astype('float32')
base_image = generateBaseImage(image, sigma, assumed_blur)
num_octaves = computeNumberOfOctaves(base_image.shape)
gaussian_kernels = generateGaussianKernels(sigma, num_intervals)
gaussian_images = generateGaussianImages(base_image, num_octaves, gaussian_kernels)
dog_images = generateDoGImages(gaussian_images)

if not use_pickled:

    ######
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

    ######
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

    ######
    # keypoints = convertKeypointsToInputImageSize(keypoints)

    print('Converting keypoints to input image size...')
    converted_keypoints = []
    for keypoint in keypoints:
        keypoint.pt = tuple(0.5 * np.array(keypoint.pt))
        keypoint.size *= 0.5
        keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
        converted_keypoints.append(keypoint)
    keypoints = converted_keypoints

    ######
    print('Generating desctiptors...')
    descriptors = generateDescriptors(keypoints, gaussian_images)


    # print(keypoints, descriptors)
    #
    for i, keypoint in enumerate(keypoints):
        print(i)
        # print("size", keypoint.size)
        # print("angle", keypoint.angle)
        # print(descriptors[i])
        o, l, s = unpackOctave(keypoint.octave)
        print(s)


    keypoint_index = 292
    keypoint = keypoints[keypoint_index]
    descriptor = descriptors[keypoint_index]

    pickle_keypoint_with_descriptor(keypoint, descriptor, keypoint_index)

else:
    keypoint_index = 292
    keypoint, descriptor = unpickle_keypoint_with_descriptor('zimnica/pickled_keypoint' + str(keypoint_index) + '_with_descriptor20200323_163005.pickle')
    print(keypoint.angle, keypoint.size)








# image_rotated = rotate_image_with_resize(image, keypoint.angle)
# plt.imshow(image_rotated, cmap="gray")

# TODO keep in mind that in opencv the order is reversed!!
patch_centre_x = int(keypoint.pt[1])
patch_centre_y = int(keypoint.pt[0])
patch_diameter = int(2 * math.floor(keypoint.size / 2) + 1)  # rounding it to the nearest odd number
patch_radius = (patch_diameter - 1) // 2

print(patch_centre_x, patch_centre_y, patch_diameter)

patch = image[patch_centre_x - patch_diameter: patch_centre_x + patch_diameter, patch_centre_y - patch_diameter: patch_centre_y + patch_diameter]  # using the diameter and not the radius to get a larger patch

# just printing
print("image shape", image.shape)
print("patch shape", patch.shape)
print("patch diameter", patch_diameter)
print("patch radius", patch_radius)
# TODO this could go out of the border and create an error!
print(patch_centre_x - patch_diameter, patch_centre_x + patch_diameter)
print(patch_centre_y - patch_diameter, patch_centre_y + patch_diameter)
plt.imshow(patch, cmap="gray")
plt.show()













######
# # rotate the patch (we're not there yet)
# patch_rotated = rotate_image_without_resize(patch, 360 - keypoint.angle)
# plt.imshow(patch_rotated, cmap="gray")
# plt.show()
# print(patch_rotated.shape)
#
# patch_rotated_cropped = patch_rotated[patch_rotated.shape[0] // 2 - patch_radius: patch_rotated.shape[0] // 2 + patch_radius + 1,
#                                       patch_rotated.shape[1] // 2 - patch_radius: patch_rotated.shape[1] // 2 + patch_radius + 1]
#
# plt.imshow(patch_rotated_cropped, cmap="gray")
# plt.show()
# print(patch_rotated_cropped.shape)
#
# imageio.imsave('images/patch' + str(keypoint_index) + '_rotated_cropped.png', patch_rotated_cropped)
# print(descriptor)
#
#
#
# plt.show(block=True)
# plt.interactive(False)


print('=====================')



