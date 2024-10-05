import random

import matplotlib.pyplot as plt
import scipy
import skimage
import torch
import numpy as np
import torch.nn.functional as F
# from line_profiler import profile
from numba import prange, njit

from dataset_utils.dataset_preprocessing_utils import normalise


# @profile
@njit(parallel=True)
def meshgrid(x, y):
    # https://stackoverflow.com/questions/70613681/numba-compatible-numpy-meshgrid
    xx = np.empty(shape=(x.size, y.size), dtype=x.dtype)
    yy = np.empty(shape=(x.size, y.size), dtype=y.dtype)
    for j in prange(y.size):
        xx[:, j] = j
        for k in prange(x.size):
            xx[k, j] = k  # change to x[k] if indexing xy
            yy[k, j] = j  # change to y[j] if indexing xy
    return yy, xx


def create_radial_mask(landmarks, img_size, pixel_size, device="cpu", radius=4, min_radius=0, do_normalise=True):
    """
    Create a radial mask around the landmarks
    """
    h, w = img_size
    yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device),
                            indexing='ij')
    mask = torch.zeros((landmarks.shape[0], h, w), device=device)
    for i, landmark in enumerate(landmarks):
        if landmark[0] < 0 or landmark[1] < 0 or landmark[0] >= h or landmark[1] >= w:
            continue
        x, y = landmark[1], landmark[0]

        distances = torch.sqrt(((yy - y) * pixel_size[0]) ** 2 + ((xx - x) * pixel_size[1]) ** 2)

        # mask[i] = torch.where((min_radius <= distances) & (distances <= radius))
        mask[i] = torch.where((min_radius <= distances) & (distances <= radius), 1, 0)

    # return torch.nonzero(mask, as_tuple=True)

    return normalise(mask) if do_normalise else mask


def create_radial_mask_batch(landmarks, img_size, pixel_size, device="cpu", radius=4, min_radius=0, do_normalise=True):
    """
    Create a radial mask around the landmarks
    """
    h, w = img_size
    mask = torch.zeros((landmarks.shape[0], landmarks.shape[1], h, w), device=device)
    for i in range(landmarks.shape[0]):
        mask[i] = create_radial_mask(landmarks[i], img_size, pixel_size[i], device, radius, min_radius, do_normalise)
    return mask


def generate_offset_maps(landmarks, img_size, mask, device="cpu"):
    """
    Create offset maps around the landmarks
    """
    h, w = img_size
    yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device),
                            indexing='ij')
    y_offset = torch.zeros((landmarks.shape[0], h, w), device=device)
    x_offset = torch.zeros((landmarks.shape[0], h, w), device=device)
    for i, landmark in enumerate(landmarks):
        if landmark[0] < 0 or landmark[1] < 0 or landmark[0] >= h or landmark[1] >= w:
            continue
        x, y = landmark[1], landmark[0]
        x_offset[i] = torch.where(mask[i] > 0, (xx - x), 0)
        y_offset[i] = torch.where(mask[i] > 0, (yy - y), 0)
    # normalise the offsets to be between -1 and 1
    return x_offset / x_offset.max(), y_offset / y_offset.max()


# @profile
@njit(parallel=True, fastmath=True)
def generate_combined_offset_map_numpy(landmarks, img_size, pixel_size, radius=4):
    h, w = img_size
    yy, xx = meshgrid(np.arange(h), np.arange(w))

    yy = yy.reshape(1, h, w)
    xx = xx.reshape(1, h, w)

    # Calculate the maximum possible distance once
    # max_dist = np.sqrt((h * pixel_size[0]) ** 2 + (w * pixel_size[1]) ** 2)

    n_landmarks = landmarks.shape[0]
    combined_offsets = np.zeros((n_landmarks, h, w), dtype=np.float32)

    for i in prange(n_landmarks):
        y_offsets = (yy - landmarks[i, 0]) * pixel_size[0]
        x_offsets = (xx - landmarks[i, 1]) * pixel_size[1]

        dists = np.sqrt(x_offsets ** 2 + y_offsets ** 2)
        combined_offsets[i] = np.where(dists <= radius, dists, 0) / radius

    return combined_offsets


# @profile
# @njit(parallel=True, fastmath=True)
# TODO: Fix the parallelisation
def create_landmark_image(landmarks, img_size, do_normalisation=False, use_gaussian=False, sigma=None):
    """Convert coordinates to image with landmarks as neighbourhoods around the coordinates"""
    c, d = landmarks.shape
    h, w = img_size
    landmark_img = np.zeros((c, h, w), dtype=np.float32)
    if sigma is None:
        sigma = np.ones(c) * 1.0

    for k in prange(c):
        # round the landmark to the nearest integer
        x, y = landmarks[k, 1], landmarks[k, 0]
        if x < 0 or y < 0 or y >= h or x >= w:
            continue
        landmark_img[k, y, x] = 1.0
        if use_gaussian and sigma[k] > 0:
            scipy.ndimage.gaussian_filter(landmark_img[k], sigma=sigma[k], output=landmark_img[k])
            # landmark_img[k] = skimage.filters.gaussian(landmark_img[k], sigma=sigma[k])

    if do_normalisation:
        return (2 * landmark_img) - 1
    return landmark_img

    # return landmark_img
