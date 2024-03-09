'''Farneback's optical flow algorithm (1D), but using the 2D version provided by OpenCV'''

import numpy as np
import scipy
from functools import partial
import skimage.transform
import logging
#logger = logging.getLogger(__name__)
logging.basicConfig(format="[%(filename)s:%(lineno)s %(funcName)s()] %(message)s")
#logger.setLevel(logging.CRITICAL)
#logger.setLevel(logging.ERROR)
#logger.setLevel(logging.WARNING)
#logger.setLevel(logging.INFO)
#logger.setLevel(logging.DEBUG)
import motion_estimation
from motion_estimation._2D.farneback import Estimator_in_CPU as Estimator

class Farneback(Estimator):

    def __init__(self, pyr_levels=3, win_side=5, num_iters=3, poly_n=5, poly_sigma=1.0, verbosity=logging.INFO):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(verbosity)
        self.win_side = win_side
        super().__init__(pyr_levels=pyr_levels, win_side=win_side, iters=num_iters, poly_n=poly_n, poly_sigma=poly_sigma)

    def get_flow(self, reference_slice, target_slice, flow):
        flow_slice = self.get_flow(reference=reference_slice, target=target_slice, flow=flow)
        flow = flow_slice[(self.w + 1) >> 1, :]
        return flow
