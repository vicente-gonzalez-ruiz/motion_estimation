import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import logging

DOWN_SCALE = 2 # Only integers
NUM_LEVELS = 3

class Gaussian_Pyramid:

    def __init__(self, logging_level=logging.INFO):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging_level)

    def filter(self, signal, sigma):
        # Create a 1D Gaussian kernel
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
    
        kernel = np.exp(-(np.arange(kernel_size) - (kernel_size - 1) / 2)**2 / (2 * sigma**2))
        kernel /= np.sum(kernel)
    
        # Apply Gaussian smoothing to the signal
        smoothed_signal = np.convolve(signal, kernel, mode='same')
    
        return smoothed_signal

    def downsample(self, signal, down_scale=DOWN_SCALE):
        # Downsampling by taking every second (in general DOWN_SCALE-nd) element
        return signal[::down_scale]

    def get_pyramid(self, signal, num_levels=NUM_LEVELS, down_scale=DOWN_SCALE):
        yield signal # Return the original signal, also
        sigma = 2 * down_scale / 6.0
        for level in range(num_levels):
            signal = self.filter(signal, sigma)
            signal = self.downsample(signal, down_scale)
            yield signal

    '''
    def _expand_level(level, sigma=1.0):
        expanded_level = np.zeros(2 * len(level), dtype=level.dtype)[:, None]
        expanded_level[::2] = level
        expanded_level = np.squeeze(expanded_level)
        expanded_level = filter(expanded_level, sigma)*2
        return expanded_level[:, None]
    '''

    def expand_level(self, signal, down_scale=DOWN_SCALE):
        #print(level.shape)
        x = np.arange(len(signal))
        xnew = np.linspace(0, len(signal), num=len(x)*down_scale)
        expanded_signal = np.interp(xnew, x, signal)
        return expanded_signal
