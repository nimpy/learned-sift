"""
    Written in Python 2, to be used with Jupyter Notebook files that use Python 2 (conda environment 'cv')
"""
import numpy as np
import cv2 as cv

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

