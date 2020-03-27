"""
    Should be compatible with both Python 2 and 3
    (first couple of methods used with Jupyter Notebook files that use Python 2 (conda environment 'cv',
    methods from rotate_image_with_resize to be used in testing_pysift.py)
"""
import numpy as np
import cv2 as cv
import pickle, datetime
import math


def unpack_octave(keypoint):
    """Compute octave, layer, and scale from a keypoint
    """
    octave = keypoint.octave & 255
    layer = (keypoint.octave >> 8) & 255
    if octave >= 128:
        octave = octave | -128
    scale = 1 / np.float32(1 << octave) if octave >= 0 else np.float32(1 << -octave)  # 1/2^octave
    return octave, layer, scale


def unpack_octave_from_kp_octave(keypoint_octave):
    """Compute octave, layer, and scale from a keypoint
    """
    octave = keypoint_octave & 255
    layer = (keypoint_octave >> 8) & 255
    if octave >= 128:
        octave = octave | -128
    scale = 1 / np.float32(1 << octave) if octave >= 0 else np.float32(1 << -octave)  # 1/2^octave
    return octave, layer, scale


def pack_octave(octave, layer):
    """Compute keypoint.octave from octave and layer
    """
    if octave < 0:
        octave += 256
    keypoint_octave = layer << 8
    keypoint_octave = keypoint_octave | octave
    return keypoint_octave


def keypoint_octave_simplified(keypoint_octave):
    """Keep only the two least significant bytes and discard the third one,
       which is not used for calculating the SIFT descriptor
    """
    return keypoint_octave & 65535


def check_packing_and_unpacking(keypoint_octave):
    octave, layer, _ = unpack_octave_from_kp_octave(keypoint_octave)
    keypoint_octave_new = pack_octave(octave, layer)
    octave_new, layer_new, _ = unpack_octave_from_kp_octave(keypoint_octave_new)
    keypoint_octave_new_new = pack_octave(octave_new, layer_new)
    assert keypoint_octave_new == keypoint_octave_new_new, "A bad thing happened!"
    assert keypoint_octave_new_new == keypoint_octave_simplified(keypoint_octave), "A bad thing happened!"
    # print keypoint_octave, keypoint_octave_new, keypoint_octave_new_new, keypoint_octave_simplified(keypoint_octave)
    # print octave, octave_new
    # print layer, layer_new
    # print


def rotate_image_without_resize(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
    return result


def rotate_image_with_resize(image, angleInDegrees):
    h, w = image.shape[:2]
    img_c = (w / 2, h / 2)

    rot = cv.getRotationMatrix2D(img_c, angleInDegrees, 1)

    rad = math.radians(angleInDegrees)
    sin = math.sin(rad)
    cos = math.cos(rad)
    b_w = int((h * abs(sin)) + (w * abs(cos)))
    b_h = int((h * abs(cos)) + (w * abs(sin)))

    rot[0, 2] += ((b_w / 2) - img_c[0])
    rot[1, 2] += ((b_h / 2) - img_c[1])

    outImg = cv.warpAffine(image, rot, (b_w, b_h), flags=cv.INTER_LINEAR)
    return outImg


def rotate_image_around_keypoint(image, keypoint):
    image_center = tuple(np.array(keypoint.pt, dtype=np.int32))
    rot_mat = cv.getRotationMatrix2D(image_center, keypoint.angle, 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
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
        keypoint = cv.KeyPoint(x=temp[0][0], y=temp[0][1], _size=temp[1], _angle=temp[2], _response=temp[3], _octave=temp[4], _class_id=temp[5])
        print(temp[0][0], temp[0][1], temp[1], temp[2], temp[3], temp[4], temp[5])
        print(temp[6])
        descriptor = temp[6]
        return keypoint, descriptor

    except Exception as e:
        print("Problem while trying to unpickle: ", str(e))
        return None

