# 1D pixel remapping

import logging
import numpy as np
from scipy.interpolate import interp1d

class Projection():
    
    def __init__(
        self,
        logging_level=logging.INFO
    ):
        self.logging_level = logging_level

    def remap(self,
              signal,
              flow,
              interpolation_mode='linear',
              extension_mode='edge'):
        """
        Project a 1D signal using optical flow.
    
        Parameters:
        - signal: 1D numpy array, the reference signal.
        - flow: 1D numpy array, the displacement field.
        - interpolation_mode: Interpolation mode for 1D interpolation, e.g., 'linear', 'nearest'.
        - extension_mode: Extension mode for values outside the signal range, e.g., 'edge', 'constant'.
    
        Returns:
        - projection: 1D numpy array, the projected signal.
        """

        # Generate coordinates for interpolation
        x_coords = np.arange(len(signal))
        projected_x_coords = x_coords - np.squeeze(flow)
    
        # Perform 1D interpolation
        interp_func = interp1d(x_coords, signal, kind=interpolation_mode, fill_value='extrapolate', bounds_error=False)
        projection = interp_func(projected_x_coords)
        
        return projection

    def add_coordinates(flow, target):
        return flow + np.moveaxis(np.indices(target.shape), 0, -1)

'''
def project(logger, signal, flow, interpolation_mode='linear', extension_mode='edge'):

    logger.debug(f"len(signal)={len(signal)}")
    logger.debug(f"len(flow)={len(flow)}")
    logger.debug(f"interpolation_mode={interpolation_mode}")
    logger.debug(f"extension_mode={extension_mode}")
    
    # Generate coordinates for interpolation
    x_coords = np.arange(len(signal))
    projected_x_coords = x_coords - np.squeeze(flow)

    # Perform 1D interpolation
    interp_func = interp1d(x_coords, signal, kind=interpolation_mode, fill_value='extrapolate', bounds_error=False)
    projection = interp_func(projected_x_coords)

    return projection

'''
