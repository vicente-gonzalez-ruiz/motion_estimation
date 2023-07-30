''' motion_estimation/farneback.py '''

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

#OFCA_EXTENSION_MODE = cv2.BORDER_REPLICATE
LEVELS = 3
WINDOW_SIDE = 5
ITERS = 5
POLY_N = 5
POLY_SIGMA = 1.2
SIGMA = 2.0
PYR_SCALE = 0.5

# projection(next, flow) ~ prev
def get_flow(
        reference,
        target,
        prev_flow=None,
        pyr_scale=0.5,
        levels=LEVELS,
        winsize=WINDOW_SIDE,
        iterations=ITERS,
        poly_n:float=POLY_N,
        poly_sigma:float=POLY_SIGMA,
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW |  cv2.OPTFLOW_FARNEBACK_GAUSSIAN):
    logger.info(f"pyr_scale={pyr_scale} levels={levels} winsize={winsize} iterations={iterations} poly_n={poly_n} poly_sigma={poly_sigma}")
    # projection(next, flow) ~ prev
    flow = cv2.calcOpticalFlowFarneback(
        prev=target,
        next=reference,
        flow=prev_flow,
        pyr_scale=pyr_scale,
        levels=levels,
        winsize=winsize,
        iterations=iterations,
        poly_n=poly_n,
        poly_sigma=poly_sigma,
        flags=flags)
    #flow = cv2.calcOpticalFlowFarneback(target, reference, flow=prev_flow, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags=cv2.OPTFLOW_USE_INITIAL_FLOW | cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    return flow
