# Copyright © 2025107441785
# This code is protected by copyright law.

# Regarding DINOv2
# We hereby declare that we have only modified line 58 of the source code in `dinov2/hub/backbone.py`, and the modified content is as follows:
# state_dict = torch.hub.load_state_dict_from_url(url=url,model_dir="/NSFF-main/dinov2/weight", map_location="cpu")

# Patent Declaration
# Our method is patented (2025107441785) and protected by copyright law. This patent primarily targets software or computer devices developed based on our method, and its content is completely independent of DINOv2.
# This patent covers only one method for generating image detection. While this method is inspired by DINOv2, it does not include the DINOv2 algorithm, technology, or implementation details.

# License Notice
# Our license (GPL-3.0) applies only to our own method and does not involve the DINOv2 repository.
# You are free to:
# View, download, and use our code for personal study, research, or evaluation.
# Modify the source code and distribute your modified versions, but must retain this statement.

# Without express written permission, you may not:
# Use our code for any commercial purpose, including but not limited to sale, rental, or provision as part of a commercial product.
# Provide technical support or services based on our code to third parties in any form for compensation.

# This code is provided "as is," and the authors assume no responsibility.

import numpy as np
from skimage.util import random_noise
from scipy import fftpack
import cv2
import torch
import torch.nn as nn
from PIL import Image
import random


def calculate_grayscale_fluctuation(patch):
    patch = patch.astype(np.float32)
    padded = np.pad(patch, pad_width=1, mode='reflect')
    total_fluct = 0.0
    center = padded[1:-1, 1:-1]
    total_fluct += np.abs(padded[0:-2, 0:-2] - center)
    total_fluct += np.abs(padded[0:-2, 1:-1] - center)
    total_fluct += np.abs(padded[0:-2, 2:] - center)
    total_fluct += np.abs(padded[1:-1, 0:-2] - center)
    total_fluct += np.abs(padded[1:-1, 2:] - center)
    total_fluct += np.abs(padded[2:, 0:-2] - center)
    total_fluct += np.abs(padded[2:, 1:-1] - center)
    total_fluct += np.abs(padded[2:, 2:] - center)

    return float(np.sum(total_fluct))

# patches sampling
def extract_patches_and_calculate_total_grayscale_fluctuation(image, patch_size=32, gap_factor=6):
    height, width, channels = image.shape
    num_patches = int((height / patch_size) * (width / patch_size) * gap_factor)
    max_i = height - patch_size
    max_j = width - patch_size
    total_possible = (max_i + 1) * (max_j + 1)
    num_patches = min(num_patches, total_possible)

    all_indices = np.arange(total_possible)
    selected = np.random.choice(all_indices, size=num_patches, replace=False)
    indices = [(idx // (max_j + 1), idx % (max_j + 1)) for idx in selected]

    patches = []
    lbp_statistics = []
    for i, j in indices:
        patch = image[i:i + patch_size, j:j + patch_size]
        total_stat = sum(calculate_grayscale_fluctuation(patch[:, :, channel]) for channel in range(channels))
        patches.append(patch)
        lbp_statistics.append(total_stat)

    sorted_indices = np.argsort(lbp_statistics)
    patches_sorted = [patches[i] for i in sorted_indices]
    return patches_sorted


# reconstruct patches
def reconstruct_image_from_portions(patches_sorted, patch_size=32, new_shape=(384, 384)):
    num_reconstruct_patches = int(new_shape[0]/patch_size) * int(new_shape[1]/patch_size)
    poor_texture_patches = patches_sorted[:num_reconstruct_patches]

    def _reconstruct_image(patches, new_shape):
        reconstructed_image = np.zeros((new_shape[0], new_shape[1], 3), dtype=np.uint8)
        idx = 0
        for i in range(0, new_shape[0], patch_size):
            for j in range(0, new_shape[1], patch_size):
                reconstructed_image[i:i + patch_size, j:j + patch_size, :] = patches[idx]
                idx += 1
                if idx >= len(patches):
                    return reconstructed_image
        return reconstructed_image

    poor_texture_img = _reconstruct_image(poor_texture_patches, new_shape)
    return poor_texture_img


# GHPF
def apply_GHPF(image, D0=30):
    img_channels = [image[:, :, 0], image[:, :, 1], image[:, :, 2]]
    GHPF_channels = []
    rows, cols = image.shape[:2]
    crow, ccol = rows // 2, cols // 2

    y = np.arange(rows) - crow
    x = np.arange(cols) - ccol
    X, Y = np.meshgrid(x, y)
    D = np.sqrt(X ** 2 + Y ** 2)
    mask = 1 - np.exp(- (D ** 2) / (2 * D0 ** 2))

    for c in img_channels:
        dft = fftpack.fft2(c)
        dft_shift = fftpack.fftshift(dft)
        dft_shift = dft_shift * mask
        f_ishift = fftpack.ifftshift(dft_shift)
        channel_back = np.abs(fftpack.ifft2(f_ishift))
        GHPF_channels.append(np.uint8(np.clip(channel_back, 0, 255)))
    GPHF_img = cv2.merge(GHPF_channels)

    return GPHF_img


def NFE_GHPF(image, D0 = None):
    if D0 is None:
        D0 = random.uniform(20, 40)
    image = np.array(image)
    patches_sorted = extract_patches_and_calculate_total_grayscale_fluctuation(image, patch_size=32)
    poor_texture_img = reconstruct_image_from_portions(patches_sorted)
    GHPF_poor_texture_img = apply_GHPF(poor_texture_img, D0)   # 高斯高通滤波器
    image = Image.fromarray(GHPF_poor_texture_img)
    return image
