import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import zoom
import logging
import inspect

DOWN_SCALE = 2 # Only integers
NUM_LEVELS = 3

class Gaussian_Pyramid:

    def __init__(self, logger):
        self.logger = logger

    def get_pyramid(self, volume, num_levels, down_scale=DOWN_SCALE):

        if self.logger.level <= logging.INFO:
            print(f"\nFunction: {inspect.currentframe().f_code.co_name}")
            args, _, _, values = inspect.getargvalues(inspect.currentframe())
            for arg in args:
                if isinstance(values[arg], np.ndarray):
                    print(f"{arg}.shape: {values[arg].shape}", end=' ')
                    print(f"{np.min(values[arg])} {np.average(values[arg])} {np.max(values[arg])}")
                else:
                    print(f"{arg}: {values[arg]}")

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

        if self.logger.level <= logging.INFO:
            print(f"\nFunction: {inspect.currentframe().f_code.co_name}")
            args, _, _, values = inspect.getargvalues(inspect.currentframe())
            for arg in args:
                if isinstance(values[arg], np.ndarray):
                    print(f"{arg}.shape: {values[arg].shape}", end=' ')
                    print(f"{np.min(values[arg])} {np.average(values[arg])} {np.max(values[arg])}")
                else:
                    print(f"{arg}: {values[arg]}")

        #expanded_volume = np.zeros((volume.shape[0]*down_scale,
        #                            volume.shape[1]*down_scale,
        #                            volume.shape[2]*down_scale),
        #                           dtype=volume.dtype)
        #expanded_volume[::down_scale, ::down_scale, ::down_scale] = volume
        #sigma = 2 * down_scale / 6.0
        #smoothed_volume = gaussian_filter(expanded_volume, sigma=sigma)
        #return smoothed_volume
        zoom_factors = (down_scale, down_scale, down_scale)
        expanded_vol = zoom(volume, zoom_factors, order=1)  # order=1 for linear interpolation
        return expanded_vol
