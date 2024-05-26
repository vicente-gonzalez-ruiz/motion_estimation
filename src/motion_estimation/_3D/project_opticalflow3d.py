import logging
import numpy as np
import opticalflow3D

class Volume_Projection(logging.Logger):
    
    def __init__(
        self,
        logging_level=logging.INFO
    ):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging_level)

    def remap(self, volume, flow):
        projection = opticalflow3D.helpers.generate_inverse_image(
            image=volume,
            vx=flow[2],
            vy=flow[1],
            vz=flow[0],
            use_gpu=False)
        return projection

    def add_coordinates(motion, target):
        return motion + np.moveaxis(np.indices(target.shape), 0, -1)