import numpy as np
import cv2
import pysift
from matplotlib import pyplot as plt
from functools import cmp_to_key
import pickle
import datetime

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


def pickle_keypoint_with_descriptor(keypoint, descriptor):
    temp = (keypoint.pt, keypoint.size, keypoint.angle, keypoint.response, keypoint.octave, keypoint.class_id, descriptor)

    pickle_file_path = 'zimnica/pickled_keypoint_with_descriptor' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.pickle'
    try:
        pickle.dump(temp, open(pickle_file_path, "wb"))
    except Exception as e:
        print("Problem while trying to pickle: ", str(e))


def unpickle_keypoint_with_descriptor(pickle_file_path):
    try:
        # point, size, angle, response, keypoint.octave, keypoint.class_id, descriptor = pickle.load(open(pickle_file_path, "rb"))
        temp = pickle.load(open(pickle_file_path, "rb"))
        keypoint = cv2.KeyPoint(x=temp[0][0], y=temp[0][1], _size=temp[1], _angle=temp[2], _response=temp[3], _octave=temp[4], _class_id=temp[5])
        descriptor = temp[6]
        return keypoint, descriptor

    except Exception as e:
        print("Problem while trying to unpickle: ", str(e))
        return None


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

    keypoint = keypoints[292]
    descriptor = descriptors[292]

    pickle_keypoint_with_descriptor(keypoint, descriptor)

else:
    keypoint, descriptor = unpickle_keypoint_with_descriptor('zimnica/pickled_keypoint_with_descriptor20200318_141401.pickle')


print(keypoint.angle, keypoint.size)



# for every image from the folder:
# # convert it to gray
# # find all the keypoints
# # for each keypoint:


image_rotated = rotate_image(image, keypoint.angle)
plt.imshow(image_rotated)
# rotate the image around it by its angle
# extract and save as a patch
# calculate its SIFT and save as file
# make sure that if it’s calculated like that the rotation of the keypoint is 0


