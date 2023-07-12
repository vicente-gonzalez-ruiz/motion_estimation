''' motion_estimation/predict.py '''

import cv2
import numpy as np

# Signal extension mode used in the OFCA. See https://docs.opencv.org/3.4/d2/de8/group__core__array.html
#ofca_extension_mode = cv2.BORDER_CONSTANT
#ofca_extension_mode = cv2.BORDER_WRAP
#ofca_extension_mode = cv2.BORDER_DEFAULT
extension_mode = cv2.BORDER_REPLICATE
#ofca_extension_mode = cv2.BORDER_REFLECT
#ofca_extension_mode = cv2.BORDER_REFLECT_101
#ofca_extension_mode = cv2.BORDER_TRANSPARENT
#ofca_extension_mode = cv2.BORDER_REFLECT101
#ofca_extension_mode = BORDER_ISOLATED
print("extension mode =", extension_mode)

def make(reference: np.ndarray, MVs: np.ndarray) -> np.ndarray:
    height, width = MVs.shape[:2]
    map_x = np.tile(np.arange(width), (height, 1))
    map_y = np.swapaxes(np.tile(np.arange(height), (width, 1)), 0, 1)
    map_xy = (MVs + np.dstack((map_x, map_y))).astype('float32')
    #map_xy = (np.rint(MVs) + np.dstack((map_x, map_y)).astype(np.float32)) # OJO RINT
    return cv2.remap(reference, map_xy, None, interpolation=cv2.INTER_LINEAR, borderMode=extension_mode)
    #return cv2.remap(reference, map_xy, None, interpolation=cv2.INTER_NEAREST, borderMode=ofca_extension_mode)
    
    #return cv2.remap(reference, cv2.convertMaps(map_x, map_y, dstmap1type=cv2.CV_16SC2), interpolation=cv2.INTER_LINEAR, borderMode=ofca_extension_mode)
    #return cv2.remap(reference, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=ofca_extension_mode)
