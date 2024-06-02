# 2D pixel remapping

import logging
import cv2
import numpy as np

class Slice_Projection():
    
    def __init__(
        self,
        logging_level=logging.INFO
    ):
        self.logging_level = logging_level

    def remap(self,
              slice,
              flow,
              interpolation_mode=cv2.INTER_LINEAR,
              extension_mode=cv2.BORDER_REPLICATE):
        height, width = flow.shape[:2]
        map_x = np.tile(np.arange(width), (height, 1))
        map_y = np.swapaxes(np.tile(np.arange(height), (width, 1)), 0, 1)
        map_xy = (flow + np.dstack((map_x, map_y))).astype('float32')
        projection = cv2.remap(
            slice,
            map_xy,
            None,
            interpolation=interpolation_mode,
            borderMode=extension_mode)
        return projection

    def add_coordinates(flow, target):
        return flow + np.moveaxis(np.indices(target.shape), 0, -1)


