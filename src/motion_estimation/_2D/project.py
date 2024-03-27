import logging
#logger = logging.getLogger(__name__)
#logging.basicConfig(format="[%(filename)s:%(lineno)s %(funcName)s()] %(message)s")
#logger.setLevel(logging.CRITICAL)
#logger.setLevel(logging.ERROR)
#logger.setLevel(logging.WARNING)
#logger.setLevel(logging.INFO)
#logger.setLevel(logging.DEBUG)

import cv2
import numpy as np

def project(logger,
            image,
            flow,
            interpolation_mode=cv2.INTER_LINEAR,
            extension_mode=cv2.BORDER_REPLICATE):
    
    logger.debug(f"reference.shape={image.shape}")
    
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
    return flow + np.moveaxis(np.indices(target.shape), 0, -1)

