'''Farneback's optical flow algorithm (2D). See https://github.com/ericPrince/optical-flow'''

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

class Farneback(polinomial_expansion.Polinomial_Expansion):

    def __init__(self, verbosity=logging.INFO):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(verbosity)
        super().__init__()

    def get_flow(self,
        f1, f2, sigma_poly, c1, c2, sigma_flow, num_iters=3, flow=None, model="constant", mu=None
    ):
        """
        Calculates optical flow using only one level of the algorithm described by Gunnar Farneback
    
        Parameters
        ----------
        f1
            First image
        f2
            Second image
        sigma_poly
            Polynomial expansion applicability Gaussian kernel sigma
        c1
            Certainty of first image
        c2
            Certainty of second image
        sigma_flow
            Applicability window Gaussian kernel sigma for polynomial matching
        num_iters
            Number of iterations to run (defaults to 1)
        flow: (optional)
            Initial displacement field
        p: (optional)
            Initial global displacement model parameters
        model: ['constant', 'affine', 'eight_param']
            Optical flow parametrization to use
        mu: (optional)
            Weighting term for usage of global parametrization. Defaults to
            using value recommended in Farneback's thesis
    
        Returns
        -------
        flow
            Optical flow field. d[i, j] is the (y, x) displacement for pixel (i, j)
        """
    
        # TODO: add initial warp parameters as optional input?
    
        # Calculate the polynomial expansion at each point in the images
        A1, B1, C1 = self.poly_expand(f1, c1, sigma_poly)
        A2, B2, C2 = self.poly_expand(f2, c2, sigma_poly)
    
        # Pixel coordinates of each point in the images
        x = np.stack(
            np.broadcast_arrays(np.arange(f1.shape[0])[:, None], np.arange(f1.shape[1])),
            axis=-1,
        ).astype(int)
    
        # Initialize displacement field
        if flow is None:
            flow = np.zeros(list(f1.shape) + [2])
    
        # Set up applicability convolution window
        n_flow = int(4 * sigma_flow + 1)
        xw = np.arange(-n_flow, n_flow + 1)
        w = np.exp(-(xw**2) / (2 * sigma_flow**2))
    
        # Evaluate warp parametrization model at pixel coordinates
        if model == "constant":
            S = np.eye(2)
    
        elif model in ("affine", "eight_param"):
            S = np.empty(list(x.shape) + [6 if model == "affine" else 8])
    
            S[..., 0, 0] = 1
            S[..., 0, 1] = x[..., 0]
            S[..., 0, 2] = x[..., 1]
            S[..., 0, 3] = 0
            S[..., 0, 4] = 0
            S[..., 0, 5] = 0
    
            S[..., 1, 0] = 0
            S[..., 1, 1] = 0
            S[..., 1, 2] = 0
            S[..., 1, 3] = 1
            S[..., 1, 4] = x[..., 0]
            S[..., 1, 5] = x[..., 1]
    
            if model == "eight_param":
                S[..., 0, 6] = x[..., 0] ** 2
                S[..., 0, 7] = x[..., 0] * x[..., 1]
    
                S[..., 1, 6] = x[..., 0] * x[..., 1]
                S[..., 1, 7] = x[..., 1] ** 2
    
        else:
            raise ValueError("Invalid parametrization model")
    
        S_T = S.swapaxes(-1, -2)
    
        # Iterate convolutions to estimate the optical flow
        for _ in range(num_iters):
            # Set flow~ as displacement field fit to nearest pixel (and constrain to not
            # being off image). Note we are setting certainty to 0 for points that
            # would have been off-image had we not constrained them
            flow_ = flow.astype(int)
            x_ = x + flow_
    
            # x_ = np.maximum(np.minimum(x_, np.array(f1.shape) - 1), 0)
    
            # Constrain d~ to be on-image, and find points that would have
            # been off-image
            x_2 = np.maximum(np.minimum(x_, np.array(f1.shape) - 1), 0)
            off_f = np.any(x_ != x_2, axis=-1)
            x_ = x_2
    
            # Set certainty to 0 for off-image points
            c_ = c1[x_[..., 0], x_[..., 1]]
            c_[off_f] = 0
    
            # Calculate A and delB for each point, according to paper
            A = (A1 + A2[x_[..., 0], x_[..., 1]]) / 2
    
            A *= c_[
                ..., None, None
            ]  # recommendation in paper: add in certainty by applying to A and delB
    
            delB = -1 / 2 * (B2[x_[..., 0], x_[..., 1]] - B1) + (A @ flow_[..., None])[..., 0]
            delB *= c_[
                ..., None
            ]  # recommendation in paper: add in certainty by applying to A and delB
    
            # Pre-calculate quantities recommended by paper
            A_T = A.swapaxes(-1, -2)
            ATA = S_T @ A_T @ A @ S
            ATb = (S_T @ A_T @ delB[..., None])[..., 0]
            # btb = delB.swapaxes(-1, -2) @ delB
    
            # If mu is 0, it means the global/average parametrized warp should not be
            # calculated, and the parametrization should apply to the local calculations
            if mu == 0:
                # Apply separable cross-correlation to calculate linear equation
                # for each pixel: G*d = h
                G = scipy.ndimage.correlate1d(ATA, w, axis=0, mode="constant", cval=0)
                G = scipy.ndimage.correlate1d(G, w, axis=1, mode="constant", cval=0)
    
                h = scipy.ndimage.correlate1d(ATb, w, axis=0, mode="constant", cval=0)
                h = scipy.ndimage.correlate1d(h, w, axis=1, mode="constant", cval=0)
    
                flow = (S @ np.linalg.solve(G, h)[..., None])[..., 0]
    
            # if mu is not 0, it should be used to regularize the least squares problem
            # and "force" the background warp onto uncertain pixels
            else:
                # Calculate global parametrized warp
                G_avg = np.mean(ATA, axis=(0, 1))
                h_avg = np.mean(ATb, axis=(0, 1))
                p_avg = np.linalg.solve(G_avg, h_avg)
                flow_avg = (S @ p_avg[..., None])[..., 0]
    
                # Default value for mu is to set mu to 1/2 the trace of G_avg
                if mu is None:
                    mu = 1 / 2 * np.trace(G_avg)
    
                # Apply separable cross-correlation to calculate linear equation
                G = scipy.ndimage.correlate1d(A_T @ A, w, axis=0, mode="constant", cval=0)
                G = scipy.ndimage.correlate1d(G, w, axis=1, mode="constant", cval=0)
    
                h = scipy.ndimage.correlate1d(
                    (A_T @ delB[..., None])[..., 0], w, axis=0, mode="constant", cval=0
                )
                h = scipy.ndimage.correlate1d(h, w, axis=1, mode="constant", cval=0)
    
                # Refine estimate of displacement field
                flow = np.linalg.solve(G + mu * np.eye(2), h + mu * flow_avg)
    
        # TODO: return global displacement parameters and/or global displacement if mu != 0
    
        return flow

    def pyramid_get_flow(self, target, reference, flow=None, pyr_levels=3, sigma_poly=4.0, sigma_flow=4.0, num_iters=3): # target and reference double's
        self.logger.info(f"pyr_levels={pyr_levels}")
        self.logger.info(f"sigma_poly={sigma_poly}")
        # c1 = np.ones_like(target)
        # c2 = np.ones_like(reference)
    
        # certainties for images - certainty is decreased for pixels near the edge
        # of the image, as recommended by Farneback
        c1 = np.minimum(
            1, 1 / 5 * np.minimum(np.arange(target.shape[0])[:, None], np.arange(target.shape[1]))
        )
        c1 = np.minimum(
            c1,
            1
            / 5
            * np.minimum(
                target.shape[0] - 1 - np.arange(target.shape[0])[:, None],
                target.shape[1] - 1 - np.arange(target.shape[1]),
            ),
        )
        c2 = c1
    
        # ---------------------------------------------------------------
        # calculate optical flow with this algorithm
        # ---------------------------------------------------------------
    
        #n_pyr = 4
    
        # # Configuration using perspective warp regularization
        # # to clean edges
        # opts = dict(
        #     sigma_poly=4.0,
        #     sigma_flow=4.0,
        #     num_iter=3,
        #     model='eight_param',
        #     mu=None,
        # )

        # Configuration using no regularization model
        opts = dict(
            #sigma_poly=4.0,
            #sigma_flow=4.0,
            #num_iter=3,
            model="constant",
            mu=0,
        )
    
        # optical flow field
        #d = prev_flow
    
        # calculate optical flow using pyramids
        # note: reversed(...) because we start with the smallest pyramid
        for pyr1, pyr2, c1_, c2_ in reversed(
            list(
                zip(
                    *list(
                        map(
                            partial(skimage.transform.pyramid_gaussian, max_layer=pyr_levels),
                            [target, reference, c1, c2],
                        )
                    )
                )
            )
        ):
            if flow is not None:
                # TODO: account for shapes not quite matching
                #d = skimage.transform.pyramid_expand(d, multichannel=True)
                flow = skimage.transform.pyramid_expand(flow, channel_axis=2)
                flow = flow[: pyr1.shape[0], : pyr2.shape[1]]
    
            flow = self.get_flow(pyr1, pyr2, c1=c1_, c2=c2_, flow=flow, sigma_poly=sigma_poly, sigma_flow=sigma_flow, num_iters=num_iters, **opts)
    
        #xw = d + np.moveaxis(np.indices(target.shape), 0, -1)
        #return xw
        return flow[..., [1, 0]] # (x, y) notation

