import numpy as np
from scipy.ndimage import gaussian_filter
import logging

DOWN_SCALE = 2 # Only integers
NUM_LEVELS = 3

class Gaussian_Pyramid:

    def __init__(self, logging_level=logging.INFO):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging_level)

    def get_pyramid(self, volume, num_levels, down_scale=DOWN_SCALE):
        yield volume
        sigma = 2 * down_scale / 6.0
        #pyramid = [volume]
        for _ in range(num_levels):
            #smoothed_volume = gaussian_filter(pyramid[-1], sigma=sigma)
            volume = gaussian_filter(volume, sigma=sigma)
            volume = volume[::down_scale, ::down_scale, ::down_scale]
            #pyramid.append(downsampled_volume)
            yield volume
        #return pyramid

    def expand_level(self, volume, down_scale=DOWN_SCALE):
        expanded_volume = np.zeros((volume.shape[0]*down_scale,
                                    volume.shape[1]*down_scale,
                                    volume.shape[2]*down_scale),
                                   dtype=volume.dtype)
        expanded_volume[::down_scale, ::down_scale, ::down_scale] = volume
        sigma = 2 * down_scale / 6.0
        smoothed_volume = gaussian_filter(expanded_volume, sigma=sigma)
        return smoothed_volume
