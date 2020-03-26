import numpy as np
import cv2
import pysift
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


# default params
sigma = 1.6
num_intervals = 3
assumed_blur = 0.5
image_border_width = 5
contrast_threshold = 0.04


def rotate_image_without_resize(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def rotate_image_with_resize(image, angleInDegrees):
    h, w = image.shape[:2]
    img_c = (w / 2, h / 2)

    rot = cv2.getRotationMatrix2D(img_c, angleInDegrees, 1)

    rad = math.radians(angleInDegrees)
    sin = math.sin(rad)
    cos = math.cos(rad)
    b_w = int((h * abs(sin)) + (w * abs(cos)))
    b_h = int((h * abs(cos)) + (w * abs(sin)))

    rot[0, 2] += ((b_w / 2) - img_c[0])
    rot[1, 2] += ((b_h / 2) - img_c[1])

    outImg = cv2.warpAffine(image, rot, (b_w, b_h), flags=cv2.INTER_LINEAR)
    return outImg


def rotate_image_around_keypoint(image, keypoint):
    image_center = tuple(np.array(keypoint.pt, dtype=np.int32))
    rot_mat = cv2.getRotationMatrix2D(image_center, keypoint.angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def pickle_keypoint_with_descriptor(keypoint, descriptor, keypoint_index):
    temp = (keypoint.pt, keypoint.size, keypoint.angle, keypoint.response, keypoint.octave, keypoint.class_id, descriptor)

    pickle_file_path = 'zimnica/pickled_keypoint' + str(keypoint_index) + '_with_descriptor' + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + '.pickle'
    try:
        pickle.dump(temp, open(pickle_file_path, "wb"))
    except Exception as e:
        print("Problem while trying to pickle: ", str(e))


def unpickle_keypoint_with_descriptor(pickle_file_path):
    try:
        # point, size, angle, response, keypoint.octave, keypoint.class_id, descriptor = pickle.load(open(pickle_file_path, "rb"))
        temp = pickle.load(open(pickle_file_path, "rb"))
        keypoint = cv2.KeyPoint(x=temp[0][0], y=temp[0][1], _size=temp[1], _angle=temp[2], _response=temp[3], _octave=temp[4], _class_id=temp[5])
        print(temp[0][0], temp[0][1], temp[1], temp[2], temp[3], temp[4], temp[5])
        print(temp[6])
        descriptor = temp[6]
        return keypoint, descriptor

    except Exception as e:
        print("Problem while trying to unpickle: ", str(e))
        return None


# query image
image = cv2.imread('box.png', 0)
use_pickled = False

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

        o1, l1, s1 = unpackOctave(keypoint)

        keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)

        o2, l2, s2 = unpackOctave(keypoint)

        # print(o1, o2) # assert o1 == o2 + 1
        # print(l1, l2) # assert l1 == l2
        # print(s1, s2)  # assert s1 * 2 == s2
        # print()

        assert o1 == o2 + 1, "WTF"
        assert l1 == l2, "WTF"
        assert s1 * 2 == s2, "WTF"

        converted_keypoints.append(keypoint)
    keypoints = converted_keypoints


    print('Computing desctiptors...')
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





# for every image from the folder:
# # convert it to gray
# # find all the keypoints
# # for each keypoint:


# rotate the image around it by its angle


# image_rotated = rotate_image_with_resize(image, keypoint.angle)
# plt.imshow(image_rotated, cmap="gray")

# TODO keep in mind that in opencv the order is reversed!!
patch_centre_x = int(keypoint.pt[1])
patch_centre_y = int(keypoint.pt[0])
patch_radius = int(math.ceil(keypoint.size / 2))
patch_diameter = patch_radius * 2

print(patch_centre_x, patch_centre_y, patch_diameter)


patch = image[patch_centre_x - patch_diameter: patch_centre_x + patch_diameter, patch_centre_y - patch_diameter: patch_centre_y + patch_diameter]
print(image.shape)
print(patch_centre_x - patch_diameter)  # TODO this could go out of the border and create an error!
print(patch_centre_x + patch_diameter)
print(patch_centre_y - patch_diameter)
print(patch_centre_y + patch_diameter)
plt.imshow(patch, cmap="gray")
plt.show()

patch_rotated = rotate_image_without_resize(patch, 360 - keypoint.angle)
plt.imshow(patch_rotated, cmap="gray")
plt.show()
print(patch_rotated.shape)

patch_rotated_cropped = patch_rotated[patch_rotated.shape[0] // 2 - patch_radius: patch_rotated.shape[0] // 2 + patch_radius + 1,
                                      patch_rotated.shape[1] // 2 - patch_radius: patch_rotated.shape[1] // 2 + patch_radius + 1]

plt.imshow(patch_rotated_cropped, cmap="gray")
plt.show()
print(patch_rotated_cropped.shape)

imageio.imsave('images/patch' + str(keypoint_index) + '_rotated_cropped.png', patch_rotated_cropped)
print(descriptor)


# extract and save as a patch
# calculate its SIFT and save as file
# make sure that if it’s calculated like that the rotation of the keypoint is 0

plt.show(block=True)
plt.interactive(False)


print('=====================')



# create a keypoint of proper size, angle, etc
# calculate sift and compare the values



#compare the sift from this descriptor and the other one