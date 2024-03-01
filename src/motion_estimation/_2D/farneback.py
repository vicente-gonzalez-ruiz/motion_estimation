'''motion_estimation/farneback.py. See
https://docs.opencv.org/3.4/dc/d6b/group__video__track.html#ga5d10ebbd59fe09c5f650289ec0ece5af'''

import logging
logger = logging.getLogger(__name__)
#logging.basicConfig(format="[%(filename)s:%(lineno)s %(funcName)s()] %(message)s")
#logger.setLevel(logging.CRITICAL)
#logger.setLevel(logging.ERROR)
logger.setLevel(logging.WARNING)
#logger.setLevel(logging.INFO)
#logger.setLevel(logging.DEBUG)

import cv2
import numpy as np

LEVELS = 3
WINDOW_SIDE = 5
ITERS = 5
POLY_N = 5
POLY_SIGMA = 1.2
PYR_SCALE = 0.5

class Estimator_in_CPU():
    
    def __init__(self,
            levels=LEVELS, # Number of pyramid layers
            pyr_scale=PYR_SCALE, # Pyramid slope. Multiply by 2^levels the searching area if the OFE
            fast_pyramids=False, # CUDA specific
            win_side=WINDOW_SIDE, # Applicability window side
            iters=ITERS, # Number of iterations at each pyramid level
            poly_n=POLY_N, # Size of the pixel neighborhood used to find the polynomial expansion in each pixel
            poly_sigma=POLY_SIGMA, # Standard deviation of the Gaussian basis used in the polynomial expansion
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW | cv2.OPTFLOW_FARNEBACK_GAUSSIAN):
        
        logger.info(f"pyr_scale={pyr_scale} levels={levels} winsize={win_side} iterations={iters} poly_n={poly_n} poly_sigma={poly_sigma} flags={flags}")
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.win_side = win_side
        self.iters = iters
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        self.flags = flags

        if logger.getEffectiveLevel() <= logging.INFO:
            self.running_time = 0

    def get_times(self):
        return self.running_time

    def get_flow(self,
            target,
            reference,
            prev_flow):

        if logger.getEffectiveLevel() <= logging.INFO:
            time_0 = time.perf_counter()

        flow = cv2.calcOpticalFlowFarneback(
            prev=target,
            next=reference,
            flow=prev_flow,
            pyr_scale=self.pyr_scale,
            levels=self.levels,
            winsize=self.win_side,
            iterations=self.iters,
            poly_n=self.poly_n,
            poly_sigma=self.poly_sigma,
            flags=self.flags)

        if logger.getEffectiveLevel() <= logging.INFO:
            time_1 = time.perf_counter()
            last_running_time = time_1 - time_0
            self.total_running_time += last_running_time

        logger.debug(f"avg_OF={np.average(np.abs(flow)):4.2f}")

        return flow

class Estimator_in_GPU(Estimator_in_CPU):

    def __init__(self,
            levels=LEVELS,
            pyr_scale=PYR_SCALE,
            fast_pyramids=False,
            win_side=WINDOW_SIDE,
            iters=ITERS,
            poly_n=POLY_N,
            poly_sigma=POLY_SIGMA,
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW | cv2.OPTFLOW_FARNEBACK_GAUSSIAN):
        super().__init(levels,
                       pyr_scale,
                       fast_pyramids,
                       win_side,
                       iters,
                       poly_n,
                       poly_sigma,
                       flags)
        
        self.flower = cv2.cuda_FarnebackOpticalFlow.create(
            numLevels=self.levels,
            pyrScale=self.pyr_scale,
            fastPyramids=self.fast_pyramids,
            winSize=self.window_side,
            numIters=self.iters,
            polyN=self.poly_n,
            polySigma=self.poly_sigma,
            flags=self.flags)

        if logger.getEffectiveLevel() <= logging.INFO:
            self.transference_time = 0

    def get_times(self):
        return self.running_time, self.transference_time

    def get_flow(selff,
            target,
            reference,
            prev_flow):
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
        GPU_prev_flow.upload(prev_flow)
        
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