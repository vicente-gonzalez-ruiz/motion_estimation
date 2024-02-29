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
import matplotlib
import matplotlib.pyplot as plt

def colorize(MVs):
    hsv = np.zeros((MVs.shape[0], MVs.shape[1], 3), dtype=np.uint8)
    hsv[...,1] = 255
    mag, ang = cv2.cartToPolar(MVs[...,0], MVs[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb

def show_vectors(flow, dpi=150, title=None):
    #plt.figure.set_dpi(200)
    plt.figure(dpi=dpi)
    plt.quiver(flow[..., 0][::-1], flow[..., 1])
    plt.title(title, fontsize=10)
    plt.show()

def project(
        image,
        flow,
        interpolation_mode=cv2.INTER_LINEAR,
        extension_mode=cv2.BORDER_REPLICATE):
    
    logger.info(f"reference.shape={image.shape}")
    
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

