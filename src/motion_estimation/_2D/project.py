# 2D pixel remapping

import cv2
import numpy as np
import logging
import inspect

class Projection():
    
    def __init__(
        self,
        logger
        #logging_level=logging.INFO
    ):
        self.logger = logger
        #self.logging_level = logging_level

    def remap(self,
              image,
              flow,
              interpolation_mode=cv2.INTER_LINEAR,
              extension_mode=cv2.BORDER_REPLICATE):

        if self.logger.getEffectiveLevel() <= logging.DEBUG:
            print(f"\nFunction: {inspect.currentframe().f_code.co_name}")
            args, _, _, values = inspect.getargvalues(inspect.currentframe())
            for arg in args:
                if isinstance(values[arg], np.ndarray):
                    print(f"{arg}.shape: {values[arg].shape}", end=' ')
                    print(f"{np.min(values[arg])} {np.average(values[arg])} {np.max(values[arg])}")
                else:
                    print(f"{arg}: {values[arg]}")

        height, width = flow.shape[:2]
        map_x = np.tile(np.arange(width), (height, 1))
        map_y = np.swapaxes(np.tile(np.arange(height), (width, 1)), 0, 1)
        map_xy = (flow + np.dstack((map_x, map_y))).astype('float32')
        projection = cv2.remap(
            image,
            map_xy,
            None,
            interpolation=interpolation_mode,
            borderMode=extension_mode)
        return projection

    def add_coordinates(flow, target):

        if self.logger.getEffectiveLevel() <= logging.DEBUG:
            print(f"\nFunction: {inspect.currentframe().f_code.co_name}")
            args, _, _, values = inspect.getargvalues(inspect.currentframe())
            for arg in args:
                if isinstance(values[arg], np.ndarray):
                    print(f"{arg}.shape: {values[arg].shape}", end=' ')
                    print(f"{np.min(values[arg])} {np.average(values[arg])} {np.max(values[arg])}")
                else:
                    print(f"{arg}: {values[arg]}")

        return flow + np.moveaxis(np.indices(target.shape), 0, -1)


