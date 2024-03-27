'''Farmeback's optical flow algorithm (2D). See
https://docs.opencv.org/3.4/dc/d6b/group__video__track.html#ga5d10ebbd59fe09c5f650289ec0ece5af'''

import logging
#logger = logging.getLogger(__name__)
#logging.basicConfig(format="[%(filename)s:%(lineno)s %(funcName)s()] %(message)s")
#logger.setLevel(logging.CRITICAL)
#logger.setLevel(logging.ERROR)
#logger.setLevel(logging.WARNING)
#logger.setLevel(logging.INFO)
#logger.setLevel(logging.DEBUG)
import time

import cv2
import numpy as np

PYRAMID_LEVELS = 3
WINDOW_SIDE = 5
NUM_ITERATIONS = 5
N_POLY = 5
POLY_SIGMA = 1.2
PYR_SCALE = 0.5

class Estimator_in_CPU():
    
    def __init__(self,
                logger):
        self.logger = logger
        '''
        self.pyr_levels = pyr_levels
        self.win_side = win_side
        self.num_iters = num_iters
        self.sigma_poly = sigma_poly
        self.flags = flags
        for attr, value in vars(self).items():
            self.logger.info(f"{attr}: {value}")
        '''

        if logger.getEffectiveLevel() <= logging.INFO:
            self.running_time = 0
            self.total_running_time = 0

    def get_times(self):
        return self.running_time

    def pyramid_get_flow(
        self,
        target,
        reference,
        flow=None,
        pyramid_levels=PYRAMID_LEVELS, # Number of pyramid layers
        window_side=WINDOW_SIDE, # Applicability window side
        num_iterations=NUM_ITERATIONS, # Number of iterations at each pyramid level
        N_poly=N_POLY, # Standard deviation of the Gaussian basis used in the polynomial expansion
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW | cv2.OPTFLOW_FARNEBACK_GAUSSIAN):

        if self.logger.getEffectiveLevel() <= logging.INFO:
            time_0 = time.perf_counter()

        sigma_poly = (N_poly - 1)/4

        flow = cv2.calcOpticalFlowFarneback(
            prev=target,
            next=reference,
            flow=flow,
            pyr_scale=PYR_SCALE, #self.pyr_scale,
            levels=pyramid_levels,
            winsize=window_side,
            iterations=num_iterations,
            poly_n=N_poly,
            poly_sigma=sigma_poly,
            flags=flags)

        if self.logger.getEffectiveLevel() <= logging.INFO:
            time_1 = time.perf_counter()
            last_running_time = time_1 - time_0
            self.total_running_time += last_running_time

        self.logger.debug(f"avg_OF={np.average(np.abs(flow)):4.2f}")

        return flow

class Estimator_in_GPU(Estimator_in_CPU):

    def __init__(self,
            levels=PYRAMID_LEVELS,
            #pyr_scale=PYR_SCALE,
            fast_pyramids=False,
            win_side=WINDOW_SIDE,
            num_iterations=NUM_ITERATIONS,
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

        if logger.getEffectiveLevel() <= logging.INFO:
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
        
        if logger.getEffectiveLevel() <= logging.INFO:
            time_0 = time.perf_counter()
            
        GPU_target = cv2.cuda_GpuMat()
        GPU_target.upload(target)
        GPU_reference = cv2.cuda_GpuMat()
        GPU_reference.upload(reference)
        GPU_prev_flow = cv2.cuda_GpuMat()
        GPU_prev_flow.upload(flow)
        
        if logger.getEffectiveLevel() <= logging.INFO:
            last_transference_time += (time.perf_counter() - time_0)
            self.transference_time += last_transference_time
                 
        if logger.getEffectiveLevel() <= logging.INFO:
            time_0 = time.perf_counter()

        GPU_flow = cv2.cuda.FarnebackOpticalFlow.calc(
            self.flower,
            I0=GPU_target,
            I1=GPU_reference,
            flow=GPU_prev_flow)    
    
        if logger.getEffectiveLevel() <= logging.INFO:
            time_1 = time.perf_counter()
            last_running_time = time_1 - time_0
            self.running_time += last_running_time

        if logger.getEffectiveLevel() <= logging.INFO:
            time_0 = time.perf_counter()

        flow = GPU_flow.download()
    
        if logger.getEffectiveLevel() <= logging.INFO:
            last_transference_time += (time.perf_counter() - time_0)
            self.transference_time += self.last_transference_time

        logger.debug(f"avg_OF={np.average(np.abs(flow)):4.2f}")

        return flow
