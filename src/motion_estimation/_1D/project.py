import logging
logger = logging.getLogger(__name__)
#logging.basicConfig(format="[%(filename)s:%(lineno)s %(funcName)s()] %(message)s")
#logger.setLevel(logging.CRITICAL)
#logger.setLevel(logging.ERROR)
logger.setLevel(logging.WARNING)
#logger.setLevel(logging.INFO)
#logger.setLevel(logging.DEBUG)

import numpy as np
from scipy.interpolate import interp1d

def project(signal, flow, interpolation_mode='linear', extension_mode='edge'):
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

    logger.info(f"len(signal)={len(signal)}")
    logger.info(f"len(flow)={len(flow)}")
    logger.info(f"interpolation_mode={interpolation_mode}")
    logger.info(f"extension_mode={extension_mode}")
    
    # Generate coordinates for interpolation
    x_coords = np.arange(len(signal))
    projected_x_coords = x_coords + flow

    # Perform 1D interpolation
    interp_func = interp1d(x_coords, signal, kind=interpolation_mode, fill_value='extrapolate', bounds_error=False)
    projection = interp_func(projected_x_coords)

    return projection


