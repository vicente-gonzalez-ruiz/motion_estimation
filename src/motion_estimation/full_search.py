'''Full-search motion-estimation.'''

import cv2
import numpy as np

def block_ME(P, R, block_side=16, max_abs_motion=8):
    
    def local_search(by, bx):
        errors_by_search_area = np.empty((2*max_abs_motion + 1, 2*max_abs_motion + 1))
        for ry in range(-max_abs_motion, max_abs_motion + 1):
            for rx in range(-max_abs_motion, max_abs_motion + 1):
                R_block = extended_R[by*block_side + ry + max_abs_motion:
                                    (by + 1)*block_side + ry + max_abs_motion,
                                    bx*block_side + rx + max_abs_motion:
                                    (bx + 1)*block_side + rx + max_abs_motion]
                #show_frame(R_block, f"R ({by} {bx} {ry} {rx} {by*block_side + ry + max_abs_motion}:{(by + 1)*block_side + ry + max_abs_motion}, {bx*block_side + rx + max_abs_motion}:{(bx + 1)*block_side + rx + max_abs_motion})")
                P_block = P[by*block_side : (by + 1)*block_side, bx*block_side : (bx + 1)*block_side]
                #show_frame(P_block, f"P ({by*block_side}:{(by + 1)*block_side},{bx*block_side}:{(bx + 1)*block_side})")
                #errors_in_search_area = np.abs(R_block - P_block)
                #error_by_block = np.sum(errors_in_search_area)
                error = R_block.astype(np.float32) - P_block
                errors_in_search_area = error*error
                error_by_block = np.average(errors_in_search_area)
                #show_frame(errors_in_search_area, f"by={by} bx={bx} ry={ry} rx={rx} error={error_by_block}")
                errors_by_search_area[ry + max_abs_motion, rx + max_abs_motion] = error_by_block
                #show_frame(errors_by_search_area, "errors")
        mv_index = np.argmin(errors_by_search_area)
        if errors_by_search_area[max_abs_motion, max_abs_motion] == errors_by_search_area[0, 0]:
            MV_y, MV_x = 0, 0
        else:
            MV_y = mv_index // (2*max_abs_motion + 1) - max_abs_motion
            MV_x = mv_index  % (2*max_abs_motion + 1) - max_abs_motion
        #print("index=", mv_index, "y=", MV_y, "x=", MV_x)
        #print(errors_by_search_area.astype(np.int))
        return MV_y, MV_x

    assert max_abs_motion > 0
    extended_R = cv2.copyMakeBorder(R, max_abs_motion, max_abs_motion, max_abs_motion, max_abs_motion, cv2.BORDER_REPLICATE) 
    extended_R[max_abs_motion : R.shape[0] + max_abs_motion,
               max_abs_motion : R.shape[1] + max_abs_motion] = R
    #show_frame(extended_R, "extended R")
    blocks_in_y = P.shape[0]//block_side
    blocks_in_x = P.shape[1]//block_side
    MVs = np.zeros((blocks_in_y, blocks_in_x, 2), dtype=np.int8)
    #print(blocks_in_y, blocks_in_x)
    for by in range(blocks_in_y):
        for bx in range(blocks_in_x):
            MV_y, MV_x = local_search(by, bx)
            MVs[by, bx] = (MV_x, MV_y)
    return MVs

def dense_ME(predicted, reference, search_range=32, overlapping_area_side=17):
    extended_reference = np.zeros((reference.shape[0] + search_range, reference.shape[1] + search_range))
    assert overlapping_area_side % 2 != 0 # This a requirement of cv.GaussianBLur
    extended_reference[search_range//2:reference.shape[0]+search_range//2,
                       search_range//2:reference.shape[1]+search_range//2] = reference
    flow = np.zeros((predicted.shape[0], predicted.shape[1], 2), dtype=np.int8)
    min_error = np.full((predicted.shape[0], predicted.shape[1]), 255, dtype=np.uint8)
    for y in range(search_range):
        print(f"{y}/{search_range-1}", end='\r')
        for x in range(search_range):
            error = extended_reference[y : predicted.shape[0] + y,
                                       x : predicted.shape[1] + x] - predicted
            A_error = abs(error) # Ojo probar MSE
            blur_A_error = cv2.GaussianBlur(a_error, (overlapping_area_side, overlapping_area_side), 0).astype(np.int)
            which_min = blur_A_error <= min_error
            flow[:,:,0] = np.where(which_min, x - search_range//2, flow[:,:,0])
            flow[:,:,1] = np.where(which_min, y - search_range//2, flow[:,:,1])
            min_error = np.minimum(min_error, blur_A_error)
    return flow.astype(np.float)
