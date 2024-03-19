import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def filter(signal, sigma):
    # Create a 1D Gaussian kernel
    kernel_size = int(6 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = np.exp(-(np.arange(kernel_size) - (kernel_size - 1) / 2)**2 / (2 * sigma**2))
    kernel /= np.sum(kernel)

    # Apply Gaussian smoothing to the signal
    smoothed_signal = np.convolve(signal, kernel, mode='same')

    return smoothed_signal

def downsample(signal):
    # Downsampling by taking every second element
    return signal[::2]

def get_pyramid(signal, num_levels, sigma=1.0):
    yield signal

    for level in range(num_levels):
        signal = filter(signal, sigma)
        signal = downsample(signal)
        yield signal

def _expand_level(level, sigma=1.0):
    expanded_level = np.zeros(2 * len(level), dtype=level.dtype)[:, None]
    expanded_level[::2] = level
    expanded_level = np.squeeze(expanded_level)
    expanded_level = filter(expanded_level, sigma)*2
    return expanded_level[:, None]

def expand_level(level):
    #print(level.shape)
    x = np.arange(len(level))
    xnew = np.linspace(0, len(level), num=len(x)*2)
    expanded_level = np.interp(xnew, x, level)
    return expanded_level
