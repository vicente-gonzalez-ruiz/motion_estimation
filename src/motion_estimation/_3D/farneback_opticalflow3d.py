'''Farneback's optical flow algorithm (3D) using opticalflow3d. See https://github.com/yongxb/OpticalFlow3d.'''

import opticalflow3D # pip install opticalflow3d
from numba.core.errors import NumbaPerformanceWarning
import warnings; warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
import numpy as np
import logging
import inspect

PYRAMID_LEVELS = 3
WINDOW_SIDE = 5
ITERATIONS = 5
N_POLY = 11
#POLY_SIGMA = 1.2
PYR_SCALE = 0.5

class OF_Estimation():
    
    def __init__(self, logging_level=logging.INFO):
        #self.logger = logging.getLogger(__name__)
        #self.logger.setLevel(logging_level)
        self.logging_level = logging_level

    def pyramid_get_flow(
        self,
        target,
        reference,
        flow=None,
        pyramid_levels=PYRAMID_LEVELS, # Number of pyramid layers
        window_side=WINDOW_SIDE,       # Control the sie of the 3D gaussian 
        iterations=ITERATIONS,         # Number of iterations at each pyramid level
        N_poly=N_POLY,                 # Control the size of the 3D gaussian kernel used to compute the polynomial expansion
        block_size=(256, 256, 256),
        overlap=(64, 64, 64),
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
            scale=0.5,
            spatial_size=N_poly, sigma_k=1.0, # Gaussian applicability (that is, the relative importance of the points in the neighborhood) with standard_deviation = sigma_k * (spatial_size -1) used for the polinomial expansion. Should match the scale of the structures we wnat to estimate orientation for (page 77). However, small applicabilities are more sensitive to noise.
            filter_type="gaussian", filter_size=window_side, # Averaging window size to generate smooth OF. 
            presmoothing=None,
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