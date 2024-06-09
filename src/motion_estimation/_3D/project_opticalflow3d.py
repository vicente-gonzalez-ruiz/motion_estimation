import numpy as np
import opticalflow3D
import inspect
import logging

class Volume_Projection():
    
    def __init__(
        self,
        logging_level=logging.INFO
    ):
        #self.logger = logging.getLogger(__name__)
        #self.logger.setLevel(logging_level)
        self.logging_level = logging_level

    def remap(self, volume, flow, use_gpu=True):

        if self.logging_level < logging.INFO:
            print(f"\nFunction: {inspect.currentframe().f_code.co_name}")
            args, _, _, values = inspect.getargvalues(inspect.currentframe())
            for arg in args:
                if isinstance(values[arg], np.ndarray):
                    print(f"{arg}.shape: {values[arg].shape}", end=' ')
                    print(f"{np.min(values[arg])} {np.average(values[arg])} {np.max(values[arg])}")
                else:
                    try:
                        print(f"({type(arg)}) {arg}: {str(values[arg])[:50]} ...")
                    except TypeError:
                        print(f"({type(arg)}) {arg}: {values[arg]}")
        
        projection = opticalflow3D.helpers.generate_inverse_image(
            image=volume,
            vx=flow[2],
            vy=flow[1],
            vz=flow[0],
            use_gpu=use_gpu)
        return projection

    def add_coordinates(motion, target):
        return motion + np.moveaxis(np.indices(target.shape), 0, -1)