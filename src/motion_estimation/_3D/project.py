# 3D pixel remapping

import logging
import numpy as np
from scipy.ndimage import map_coordinates

class Projection():
    
    def __init__(
        self,
        logging_level=logging.INFO
    ):
        self.logging_level = logging_level

    def remap(self,
              vol,
              flow,
              interpolation_mode='linear',
              extension_mode='nearest'):

        height, width, depth = flow.shape[:3]
        map_x, map_y, map_z = np.indices((height, width, depth))
        map_x = map_x.astype('float32')
        map_y = map_y.astype('float32')
        map_z = map_z.astype('float32')
        
        # Apply flow displacement to coordinates
        map_x_shifted = map_x + flow[..., 0]
        map_y_shifted = map_y + flow[..., 1]
        map_z_shifted = map_z + flow[..., 2]
    
        # Clip shifted coordinates to stay within bounds
        map_x_shifted = np.clip(map_x_shifted, 0, width - 1)
        map_y_shifted = np.clip(map_y_shifted, 0, height - 1)
        map_z_shifted = np.clip(map_z_shifted, 0, depth - 1)
    
        # Perform interpolation
        projection = map_coordinates(
            vol,
            [map_x_shifted, map_y_shifted, map_z_shifted],
            order=1, mode=extension_mode)
        
        return projection

    # Untested
    def add_coordinates(flow, target):
        return flow + np.moveaxis(np.indices(target.shape), 0, -1)

'''
def project(logger,
            vol,
            flow,
            interpolation_mode='linear',
            extension_mode='nearest'):
    
    logger.debug(f"vol.shape={vol.shape}, flow.shape={flow.shape}")
    
    height, width, depth = flow.shape[:3]
    map_x, map_y, map_z = np.indices((height, width, depth))
    map_x = map_x.astype('float32')
    map_y = map_y.astype('float32')
    map_z = map_z.astype('float32')
    
    # Apply flow displacement to coordinates
    map_x_shifted = map_x + flow[..., 0]
    map_y_shifted = map_y + flow[..., 1]
    map_z_shifted = map_z + flow[..., 2]

    # Clip shifted coordinates to stay within bounds
    map_x_shifted = np.clip(map_x_shifted, 0, width - 1)
    map_y_shifted = np.clip(map_y_shifted, 0, height - 1)
    map_z_shifted = np.clip(map_z_shifted, 0, depth - 1)

    # Perform interpolation
    projection = map_coordinates(
        vol,
        [map_x_shifted, map_y_shifted, map_z_shifted],
        order=1, mode=extension_mode)

    return projection
'''

