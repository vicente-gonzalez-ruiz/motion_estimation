import numpy as np
from scipy.ndimage import gaussian_filter

def get_pyramid(volume, num_levels, sigma=1.0):
    pyramid = [volume]
    for _ in range(num_levels):
        smoothed_volume = gaussian_filter(pyramid[-1], sigma=sigma)
        downsampled_volume = smoothed_volume[::2, ::2, ::2]  # Downsampling by taking every second element
        pyramid.append(downsampled_volume)
    return pyramid

def expand_level(volume, sigma=1.0):
    expanded_volume = np.zeros((volume.shape[0]*2, volume.shape[1]*2, volume.shape[2]*2), dtype=volume.dtype)
    expanded_volume[::2, ::2, ::2] = volume  # Upsampling by doubling the dimensions
    smoothed_volume = gaussian_filter(expanded_volume, sigma=sigma)
    return smoothed_volume
