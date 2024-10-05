import math
import random

import matplotlib.pyplot as plt
import scipy
import torch
import numpy as np

from dataset_utils.dataset_preprocessing_utils import normalise


def create_landmark_image_multi(landmarks, img_size, dtype=torch.float32, eps_window_size=0, max_value=1,
                                do_normalisation=True, device="cpu"):
    """Convert coordinates to image with landmarks as neighbourhoods around the coordinates"""
    num_annotators, c, d = landmarks.shape

    # annotator, channel, dim
    h, w = img_size
    landmark_img = torch.zeros((c, h, w), dtype=dtype, device=device)
    yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device),
                            indexing='ij')
    if type(landmarks) == np.ndarray:
        landmarks = np.round(landmarks).astype(int)
    else:
        landmarks = torch.round(landmarks).long()
    max_value = max_value / num_annotators
    for a, annotator in enumerate(landmarks):
        for k, landmark in enumerate(annotator):
            if landmark[0] < 0 or landmark[1] < 0 or landmark[0] >= h or landmark[1] >= w:
                continue
            # round the landmark to the nearest integer
            x, y = landmark[1], landmark[0]
            if eps_window_size <= 1:
                landmark_img[k, y, x] += max_value
            else:
                # Create a mask for the circle around the landmark
                distances = torch.abs(yy - y) + torch.abs(xx - x)
                circle_mask = distances <= eps_window_size
                # Calculate values within the circle
                values = torch.where(circle_mask, max_value - (distances / eps_window_size * max_value), 0)

                # Update the landmark image
                landmark_img[k] += torch.max(landmark_img[k], values)

    return normalise(landmark_img) if do_normalisation else landmark_img


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


def create_radial_mask_batch(landmarks, img_size, pixel_size, radius=4, min_radius=0, do_normalise=True):
    """
    Create a radial mask around the landmarks
    """
    h, w = img_size
    mask = torch.zeros((landmarks.shape[0], landmarks.shape[1], h, w), device=landmarks.device)
    for i in range(landmarks.shape[0]):
        mask[i] = create_radial_mask(landmarks[i], img_size, pixel_size[i], landmarks.device, radius, min_radius,
                                     do_normalise)
    return mask


def generate_offset_maps(landmarks, img_size, mask):
    """
    Create offset maps around the landmarks
    """
    h, w = img_size
    yy, xx = torch.meshgrid(torch.arange(h, device=landmarks.device), torch.arange(w, device=landmarks.device),
                            indexing='ij')
    y_offset = torch.zeros((landmarks.shape[0], h, w), device=landmarks.device)
    x_offset = torch.zeros((landmarks.shape[0], h, w), device=landmarks.device)
    for i, landmark in enumerate(landmarks):
        if landmark[0] < 0 or landmark[1] < 0 or landmark[0] >= h or landmark[1] >= w:
            continue
        x, y = landmark[1], landmark[0]
        x_offset[i] = torch.where(mask[i] > 0, (xx - x), 0)
        y_offset[i] = torch.where(mask[i] > 0, (yy - y), 0)
    # normalise the offsets to be between -1 and 1
    return x_offset / x_offset.max(), y_offset / y_offset.max()


def generate_combined_offset_map(landmarks, img_size, pixel_size, radius=4):
    h, w = img_size
    yy, xx = torch.meshgrid(torch.arange(h, device=landmarks.device), torch.arange(w, device=landmarks.device),
                            indexing='ij')

    yy = yy.reshape(1, h, w)
    xx = xx.reshape(1, h, w)

    # Calculate the maximum possible distance once
    # max_dist = np.sqrt((h * pixel_size[0]) ** 2 + (w * pixel_size[1]) ** 2)

    n_landmarks = landmarks.shape[0]
    combined_offsets = torch.zeros((n_landmarks, h, w), device=landmarks.device)

    for i in range(n_landmarks):
        y_offsets = pixel_size[0] * (yy - landmarks[i, 0])
        x_offsets = pixel_size[1] * (xx - landmarks[i, 1])

        dists = torch.sqrt(x_offsets ** 2 + y_offsets ** 2)
        combined_offsets[i] = torch.where(dists <= radius, dists, 0) / radius

    return combined_offsets



@torch.compile
def create_landmark_image_no_batch(landmarks, img_size, dtype=torch.float32, do_normalisation=True, device="cpu"):
    """Convert coordinates to image with landmarks as neighbourhoods around the coordinates"""
    c, d = landmarks.shape
    h, w = img_size
    landmark_img = torch.zeros((c, h, w), dtype=dtype, device=device)
    landmarks = landmarks.round()
    if type(landmarks) == np.ndarray:
        landmarks = landmarks.astype(int)
    else:
        landmarks = landmarks.long()
    for k, landmark in enumerate(landmarks):
        if landmark[0] < 0 or landmark[1] < 0 or landmark[0] >= h or landmark[1] >= w:
            continue
        # round the landmark to the nearest integer
        x, y = landmark[1], landmark[0]

        landmark_img[k, y, x] = 1

    return normalise(landmark_img) if do_normalisation else landmark_img


def create_landmark_image(landmarks, img_size, eps_window_size=0, max_value=1,
                          do_normalisation=True, device="cpu"):
    """Convert coordinates to image with landmarks as neighbourhoods around the coordinates"""

    b, c, d = landmarks.shape
    h, w = img_size

    dtype = torch.float32

    output_image = torch.zeros((b, c, h, w), dtype=dtype, device=device)
    for i in range(b):
        output_image[i] = create_landmark_image_no_batch(landmarks[i], img_size, dtype, eps_window_size, max_value,
                                                         do_normalisation, device)
    return output_image

