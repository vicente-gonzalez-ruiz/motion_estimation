'''Farneback's optical flow algorithm (3D) using opticalflow3d. See https://github.com/yongxb/OpticalFlow3d.'''

import opticalflow3D # pip install opticalflow3d
from numba.core.errors import NumbaPerformanceWarning
import warnings; warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
import numpy as np
import logging
import inspect

# Polynomial expansion
SPATIAL_SIZE = 9    # Side of the Gaussian applicability window used
                    # during the polynomial expansion. Applicability (that is, the relative importance of the points in the neighborhood) size should match the scale of the structures we wnat to estimate orientation for (page 77). However, small applicabilities are more sensitive to noise.
SIGMA_K = 0.15      # Scaling factor used to calculate the standard
                    # deviation of the Gaussian applicability. The
                    # formula to calculate the standard deviation is
                    # sigma = sigma_k*(spatial_size - 1).

# OF estimation
FILTER_TYPE = "box" # Shape of the filer used to average the flow. It
                    # can be "box" or "gaussian".
FILTER_SIZE = 21    # Size of the filter used to average the G and
                    # matrices (see Eqs. 4.7 and 4.27 of the thesis).
PYRAMID_LEVELS = 3  # Number of pyramid layers
ITERATIONS = 5      # Number of iterations at each pyramid level
PYRAMID_SCALE = 0.5

class OF_Estimation():
    
    def __init__(self, logging_level=logging.INFO):
        #self.logger = logging.getLogger(__name__)
        #self.logger.setLevel(logging_level)
        self.logging_level = logging_level

    def pyramid_get_flow(
        self,
        target, reference,
        pyramid_levels=PYRAMID_LEVELS,
        spatial_size=SPATIAL_SIZE,
        iterations=ITERATIONS,
        sigma_k=SIGMA_K,
        filter_type=FILTER_TYPE,
        filter_size=FILTER_SIZE,
        presmoothing=None,
        block_size=(256, 256, 256),
        overlap=(8, 8, 8),
        threads_per_block=(8, 8, 8)
    ):

        '''
        for attr, value in vars(self).items():
            self.logger.debug(f"{attr}: {value}")
        '''

        if self.logging_level < logging.INFO:
            print(f"\nFunction: {inspect.currentframe().f_code.co_name}")
            args, _, _, values = inspect.getargvalues(inspect.currentframe())
            for arg in args:
                if isinstance(values[arg], np.ndarray):
                    print(f"{arg}.shape: {values[arg].shape}", end=' ')
                    print(f"{np.min(values[arg])} {np.average(values[arg])} {np.max(values[arg])}")
                else:
                    print(f"{arg}: {values[arg]}")

        farneback = opticalflow3D.Farneback3D(
            iters=iterations,
            num_levels=pyramid_levels,
            scale=PYRAMID_SCALE,
            spatial_size=spatial_size,
            sigma_k=sigma_k,
            filter_type=filter_type,
            filter_size=filter_size,
            presmoothing=presmoothing,
            device_id=0)

        flow_z, flow_y, flow_x, output_confidence = farneback.calculate_flow(
            image1=reference,
            image2=target,
            start_point=(0, 0, 0),
            total_vol=(reference.shape[0], reference.shape[1], reference.shape[2]),
            sub_volume=block_size,
            overlap=overlap,
            threadsperblock=threads_per_block)

        return flow_z, flow_y, flow_x
