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
from . import polinomial_expansion
from . import pyramid_gaussian

class Farneback(motion_estimation._2D.farneback.Estimator_in_CPU):

    def __init__(self, l=3, w=5, num_iters=3, poly_n=5, poly_sigma=1.0, verbosity=logging.INFO):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(verbosity)
        self.w = w
        super().__init__(levels=l, pyr_scale=0.5, win_side=w, iters=num_iters, poly_n, poly_sigma, verbosity)

    def get_flow(self, reference_slice, target_slice, prev_flow=None):
        flow_slice = self.get_flow(reference=reference_slice, target=target_slice, prev_flow)
        flow = flow_slice[(self.w + 1) >> 1, :]
        return flow
