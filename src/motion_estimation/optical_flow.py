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

# Number of levels of the gaussian pyramid used in the Farneback's
# optical flow computation algorith (OFCA). This value controls the
# search area size.
OF_LEVELS = 3
print("OFCA: default number of levels =", OF_LEVELS)

# Window (squared) side used in the Farneback's OFCA. This value controls the
# coherence of the OF.
OF_WINDOW_SIDE = 33
print(f"OFCA: default window size = {OF_WINDOW_SIDE}x{OF_WINDOW_SIDE}")

# Number of iterations of the Farneback's OFCA. This value controls
# the accuracy of the OF.
OF_ITERS = 3
print(f"OFCA: default number of iterations =", OF_ITERS)

# Signal extension mode used in the OFCA. See https://docs.opencv.org/3.4/d2/de8/group__core__array.html
#ofca_extension_mode = cv2.BORDER_CONSTANT
#ofca_extension_mode = cv2.BORDER_WRAP
#ofca_extension_mode = cv2.BORDER_DEFAULT
ofca_extension_mode = cv2.BORDER_REPLICATE
#ofca_extension_mode = cv2.BORDER_REFLECT
#ofca_extension_mode = cv2.BORDER_REFLECT_101
#ofca_extension_mode = cv2.BORDER_TRANSPARENT
#ofca_extension_mode = cv2.BORDER_REFLECT101
#ofca_extension_mode = BORDER_ISOLATED
print("OFCA: extension mode =", ofca_extension_mode)

#POLY_N = 5
#POLY_SIGMA = 1.1
POLY_N = 7
POLY_SIGMA = 1.5
print("OFCA: default poly_n", POLY_N)
print("OFCA: default poly_sigma", POLY_SIGMA)

def Farneback_ME(predicted:np.ndarray,
                 reference:np.ndarray,
                 initial_MVs:np.ndarray=None,
                 pyr_scale=0.5,
                 levels:int=OF_LEVELS,
                 wside:int=OF_WINDOW_SIDE,
                 iters:int=OF_ITERS,
                 poly_n:float=POLY_N,
                 poly_sigma:float=POLY_SIGMA) -> np.ndarray:
    logger.info(f"estimate: pyr_scale={pyr_scale} levels={levels} wside={wside} iters={iters} poly_n={poly_n} poly_sigma={poly_sigma}")
    MVs = cv2.calcOpticalFlowFarneback(
        prev=predicted,
        next=reference,
        flow=initial_MVs,
        pyr_scale=pyr_scale,
        levels=levels,
        winsize=wside,
        iterations=iters,
        poly_n=5,
        poly_sigma=1.2,
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW | cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    return MVs
