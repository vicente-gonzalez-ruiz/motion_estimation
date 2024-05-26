'''Farneback's optical flow algorithm (3D) using well-known libraries. See https://github.com/ericPrince/optical-flow'''

import opticalflow3D
from numba.core.errors import NumbaPerformanceWarning
import warnings; warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
import numpy as np
import logging

PYRAMID_LEVELS = 3
WINDOW_SIDE = 5
ITERATIONS = 5
N_POLY = 11
#POLY_SIGMA = 1.2
PYR_SCALE = 0.5

class Farneback_Estimator(logging.Logger):
    
    def __init__(
        self,      
        logging_level=logging.INFO
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging_level)

    def pyramid_get_flow(
        self,
        target,
        reference,
        flow=None,
        pyramid_levels=PYRAMID_LEVELS, # Number of pyramid layers
        window_side=WINDOW_SIDE,       # Applicability window side
        iterations=ITERATIONS,         # Number of iterations at each pyramid level
        N_poly=N_POLY,                 # Standard deviation of the Gaussian basis used in the polynomial expansion
        block_size=(256, 256, 256),
        overlap=(64, 64, 64),
        threads_per_block=(8, 8, 8)
    ):

        for attr, value in vars(self).items():
            self.logger.debug(f"{attr}: {value}")

        farneback = opticalflow3D.Farneback3D(
            iters=iterations,
            num_levels=pyramid_levels,
            scale=0.5,
            spatial_size=window_side,
            sigma_k=1.0,
            filter_type="gaussian",
            filter_size=N_poly,
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