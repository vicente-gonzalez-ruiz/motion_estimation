'''Farmeback's optical flow algorithm (2D). See
https://docs.opencv.org/3.4/dc/d6b/group__video__track.html#ga5d10ebbd59fe09c5f650289ec0ece5af'''

#import logging
import time

import cv2
import numpy as np

PYRAMID_LEVELS = 3
WINDOW_SIDE = 5
ITERATIONS = 5
N_POLY = 5
POLY_SIGMA = 1.2
PYR_SCALE = 0.5

class OF_Estimation():
    
    def __init__(
        self,
        logger
        #logging_level=logging.INFO
    ):
        #self.logger = logging.getLogger(__name__)
        #self.flags = 0
        #self.logger.setLevel(logging_level)
        self.logger = logger
        
        for attr, value in vars(self).items():
            self.logger.debug(f"{attr}: {value}")

    def pyramid_get_flow(
        self,
        target,
        reference,
        flow=None,
        pyramid_levels=PYRAMID_LEVELS, # Number of pyramid layers
        window_side=WINDOW_SIDE,       # Applicability window side
        iterations=ITERATIONS,         # Number of iterations at each pyramid level
        N_poly=N_POLY,                 # Order of the polynomial expansion
        sigma_poly=POLY_SIGMA,         # Standard deviation of the Gaussian basis used in the polynomial expansion
        flags=0                        # cv2.OPTFLOW_USE_INITIAL_FLOW | cv2.OPTFLOW_FARNEBACK_GAUSSIAN
    ):

        for name, value in locals().items():
            self.logger.debug(f"{name}: {value}")

        #sigma_poly = (N_poly - 1)/4 # Standard deviation of the Gaussian basis used in the polynomial expansion
        #self.logger.info(f"sigma_poly={sigma_poly}")
        #print(target.shape, reference.shape, flow, target.dtype, reference.dtype)
        flow = cv2.calcOpticalFlowFarneback(
            prev=target,
            next=reference,
            flow=flow,
            pyr_scale=0.5,
            levels=pyramid_levels,
            winsize=window_side,
            iterations=iterations,
            poly_n=N_poly,
            poly_sigma=sigma_poly,
            flags=flags)

        return flow

class Estimator_in_GPU(OF_Estimation):

    def __init__(self,
            levels=PYRAMID_LEVELS,
            #pyr_scale=PYR_SCALE,
            fast_pyramids=False,
            win_side=WINDOW_SIDE,
            num_iterations=ITERATIONS,
            sigma_poly=POLY_SIGMA,
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW | cv2.OPTFLOW_FARNEBACK_GAUSSIAN):
        super().__init(levels,
                       #pyr_scale=PRY_SCALE,
                       fast_pyramids,
                       win_side,
                       num_iterations,
                       sigma_poly,
                       flags)
        
        self.flower = cv2.cuda_FarnebackOpticalFlow.create(
            numLevels=self.levels,
            pyrScale=PYR_SCALE, #self.pyr_scale,
            fastPyramids=self.fast_pyramids,
            winSize=self.window_side,
            numIters=self.num_iterations,
            polyN=POLY_N,
            polySigma=self.poly_sigma,
            flags=self.flags)

        if self.logger.getEffectiveLevel() <= logging.INFO:
            self.transference_time = 0

    def get_times(self):
        return self.running_time, self.transference_time

    def get_flow(self,
            target,
            reference,
            flow):
        '''The returned flow express the positions of the pixels of target in
respect of the pixels of reference. In other words, if we project
target using the flow, get should get reference.

        '''
        
        if self.logger.getEffectiveLevel() <= logging.INFO:
            time_0 = time.perf_counter()
            
        GPU_target = cv2.cuda_GpuMat()
        GPU_target.upload(target)
        GPU_reference = cv2.cuda_GpuMat()
        GPU_reference.upload(reference)
        GPU_prev_flow = cv2.cuda_GpuMat()
        GPU_prev_flow.upload(flow)
        
        if self.logger.getEffectiveLevel() <= logging.INFO:
            last_transference_time += (time.perf_counter() - time_0)
            self.transference_time += last_transference_time
                 
        if self.logger.getEffectiveLevel() <= logging.INFO:
            time_0 = time.perf_counter()

        GPU_flow = cv2.cuda.FarnebackOpticalFlow.calc(
            self.flower,
            I0=GPU_target,
            I1=GPU_reference,
            flow=GPU_prev_flow)    
    
        if self.logger.getEffectiveLevel() <= logging.INFO:
            time_1 = time.perf_counter()
            last_running_time = time_1 - time_0
            self.running_time += last_running_time

        if self.logger.getEffectiveLevel() <= logging.INFO:
            time_0 = time.perf_counter()

        flow = GPU_flow.download()
    
        if self.logger.getEffectiveLevel() <= logging.INFO:
            last_transference_time += (time.perf_counter() - time_0)
            self.transference_time += self.last_transference_time

        self.logger.debug(f"avg_OF={np.average(np.abs(flow)):4.2f}")

        return flow
