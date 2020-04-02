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

from testing_pysift_utils import rotate_image_without_resize, pickle_keypoint_with_descriptor, unpickle_keypoint_with_descriptor, \
    get_gaussian_images_from_image


# default params
sigma = 1.6
num_intervals = 3
assumed_blur = 0.5
image_border_width = 5
contrast_threshold = 0.04



image = cv2.imread('box.png', 0)
keypoint_index = 528  # 587 here is 459 for sift; 528 here is 584 for sift
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
        print("id", i)
        print(keypoint.pt[0], keypoint.pt[1], keypoint.size, keypoint.angle, keypoint.octave)
        print(descriptors[i])
        print()


    keypoint = keypoints[keypoint_index]
    descriptor = descriptors[keypoint_index]

    pickle_keypoint_with_descriptor(keypoint, descriptor, keypoint_index)

else:

    keypoint, descriptor = unpickle_keypoint_with_descriptor('zimnica/pickled_keypoint' + str(keypoint_index) + '_with_descriptor20200402_122717.pickle')
    print(keypoint.angle, keypoint.size)
    print()

print('=====================')
print('\n'*10)





print('=====================')
# print('\n'*10)

# try to understand the difference between this descriptor and ...
descriptors = generateDescriptors([keypoint], gaussian_images)
descr_cvsift = descriptors[0]
print(image.shape)
print(descr_cvsift)



# image_rotated = rotate_image_with_resize(image, keypoint.angle)
# plt.imshow(image_rotated, cmap="gray")

# TODO keep in mind that in opencv the order is reversed!!
patch_centre_x = int(keypoint.pt[1])
patch_centre_y = int(keypoint.pt[0])
patch_diameter = int(2 * math.floor(keypoint.size / 2) + 1)  # rounding it to the nearest odd number
patch_radius = (patch_diameter - 1) // 2
# TODO adjust this afterwards, for the moment it's because it makes a difference on half_width variable, which in turn makes a difference on end descriptor.
#  For this particular keypoint, the global maximum is when the diameter is 3.2 times the keypoint size, because if it's bigger, the patch goes out of the image
#  (well not really, just gets cut off)
patch_diameter *= 5  # 3.2
patch_diameter = int(patch_diameter)

# TODO check the +1 part
patch = image[max(patch_centre_x - patch_diameter, 0): min(patch_centre_x + patch_diameter + 1, image.shape[0]),
              max(patch_centre_y - patch_diameter, 0): min(patch_centre_y + patch_diameter + 1, image.shape[1])]  # using the diameter and not the radius to get a larger patch
print(patch.shape)

# just printing
# print("image shape", image.shape)
# print("patch shape", patch.shape)
# print("patch diameter", patch_diameter)
# print("patch radius", patch_radius)
# # TODO this could go out of the border and create an error!
# print(patch_centre_x - patch_diameter, patch_centre_x + patch_diameter)
# print(patch_centre_y - patch_diameter, patch_centre_y + patch_diameter)
# plt.imshow(patch, cmap="gray")
# plt.show()
#




if patch_centre_x - patch_diameter >= 0 and patch_centre_x + patch_diameter + 1 < image.shape[0]:
    keypoint_from_patch_x_coord = patch.shape[0] // 2
else:
    if patch_centre_x - patch_diameter >= 0:
        keypoint_from_patch_x_coord = patch.shape[0] // 2
    else:
        keypoint_from_patch_x_coord = patch_centre_x

if patch_centre_y - patch_diameter >= 0 and patch_centre_y + patch_diameter + 1 < image.shape[1]:
    keypoint_from_patch_y_coord = patch.shape[1] // 2
else:
    if patch_centre_y - patch_diameter >= 0:
        keypoint_from_patch_y_coord = patch.shape[1] // 2
    else:
        keypoint_from_patch_y_coord = patch_centre_y
print(keypoint_from_patch_x_coord, keypoint_from_patch_y_coord)


keypoint_from_patch = cv2.KeyPoint(keypoint_from_patch_y_coord, keypoint_from_patch_x_coord, _size=keypoint.size, _angle=keypoint.angle, _octave=keypoint.octave)
keypoints_from_patch = [keypoint_from_patch]

# patch_gaussian_images = get_gaussian_images_from_image(patch)
# descriptors = generateDescriptors(keypoints_from_patch, patch_gaussian_images)
# print(descriptors)

###


octave_index, layer_index, _ = unpackOctave(keypoint)

gaussian_images_cropped = gaussian_images.copy()
gaussian_images_cropped_octave_layer = gaussian_images[octave_index + 1, layer_index][max(patch_centre_x - patch_diameter, 0): min(patch_centre_x + patch_diameter + 1, image.shape[0]),
                                                                        max(patch_centre_y - patch_diameter, 0): min(patch_centre_y + patch_diameter + 1, image.shape[1])]
gaussian_images_cropped[octave_index + 1, layer_index] = gaussian_images_cropped_octave_layer
descriptors = generateDescriptors(keypoints_from_patch, gaussian_images_cropped)
# ... and this descriptor
descr_cropped_gauss = descriptors[0]
print(descr_cropped_gauss)

print(np.corrcoef(descr_cvsift, descr_cropped_gauss))

print("check that the keypoint centre is correct")
print(image[patch_centre_x - 2: patch_centre_x + 2, patch_centre_y - 2: patch_centre_y + 2])
print(patch[keypoint_from_patch_x_coord - 2: keypoint_from_patch_x_coord + 2, keypoint_from_patch_y_coord - 2: keypoint_from_patch_y_coord + 2])

plt.imshow(image, cmap='gray')
plt.scatter([patch_centre_y], [patch_centre_x])
plt.show()

plt.imshow(patch, cmap='gray')
plt.scatter([keypoint_from_patch_y_coord], [keypoint_from_patch_x_coord])

# TODO next: work in progress -- figure out why these points aren't in the same spot!!!


#their shape is the same...
# print(gaussian_images[octave_index + 1, layer_index])
# print(gaussian_images_cropped[octave_index + 1, layer_index])
# print(np.array_equal(gaussian_images[octave_index + 1, layer_index], gaussian_images_cropped[octave_index + 1, layer_index]))

# print(np.setdiff1d(gaussian_images, gaussian_images_cropped))

# for i in range(gaussian_images.shape[0]):
#     for j in range(gaussian_images.shape[1]):
#         print(gaussian_images[i, j].shape)
#         print(gaussian_images_cropped[i, j].shape)
#         assert gaussian_images[i, j].shape == gaussian_images_cropped[i, j].shape, "wut "
#         assert np.array_equal(gaussian_images[i, j], gaussian_images_cropped[i, j]), "why"
#         print()


# for i in range()


# (1) gde je razlika i (2) zasto je uopste ta razlika bitna????


######

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
plt.show(block=True)
plt.interactive(False)





