'''Farneback's optical flow algorithm (2D). See https://github.com/ericPrince/optical-flow'''

import numpy as np
import logging
#logger = logging.getLogger(__name__)
logging.basicConfig(format="[%(filename)s:%(lineno)s %(funcName)s()] %(message)s")
#logger.setLevel(logging.CRITICAL)
#logger.setLevel(logging.ERROR)
#logger.setLevel(logging.WARNING)
#logger.setLevel(logging.INFO)
#logger.setLevel(logging.DEBUG)
import polinomial_expansion
import scipy

class Farneback(polinomial_expansion.Polinomial_Expansion):

    def __init__(self, verbosity=logging.INFO):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(verbosity)
        super().__init__()

    def estimate_2d(self,
        f1, f2, sigma, c1, c2, sigma_flow, num_iter=1, d=None, model="constant", mu=None
    ):
        """
        Calculates optical flow with an algorithm described by Gunnar Farneback
    
        Parameters
        ----------
        f1
            First image
        f2
            Second image
        sigma
            Polynomial expansion applicability Gaussian kernel sigma
        c1
            Certainty of first image
        c2
            Certainty of second image
        sigma_flow
            Applicability window Gaussian kernel sigma for polynomial matching
        num_iter
            Number of iterations to run (defaults to 1)
        d: (optional)
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
        d
            Optical flow field. d[i, j] is the (y, x) displacement for pixel (i, j)
        """
    
        # TODO: add initial warp parameters as optional input?
    
        # Calculate the polynomial expansion at each point in the images
        A1, B1, C1 = self.poly_expand(f1, c1, sigma)
        A2, B2, C2 = self.poly_expand(f2, c2, sigma)
    
        # Pixel coordinates of each point in the images
        x = np.stack(
            np.broadcast_arrays(np.arange(f1.shape[0])[:, None], np.arange(f1.shape[1])),
            axis=-1,
        ).astype(int)
        print(f1.shape)
        print(np.arange(f1.shape[0]))
        print(np.arange(f1.shape[0])[:, None])
        print(np.broadcast_arrays(np.arange(f1.shape[0])[:, None], np.arange(f1.shape[1])))
        print(x)
    
        # Initialize displacement field
        if d is None:
            d = np.zeros(list(f1.shape) + [2])
    
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
        for _ in range(num_iter):
            # Set d~ as displacement field fit to nearest pixel (and constrain to not
            # being off image). Note we are setting certainty to 0 for points that
            # would have been off-image had we not constrained them
            d_ = d.astype(int)
            x_ = x + d_
    
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
            print(A1, A2[x_[..., 0], x_[..., 1]])
    
            A *= c_[
                ..., None, None
            ]  # recommendation in paper: add in certainty by applying to A and delB
    
            delB = -1 / 2 * (B2[x_[..., 0], x_[..., 1]] - B1) + (A @ d_[..., None])[..., 0]
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
    
                d = (S @ np.linalg.solve(G, h)[..., None])[..., 0]
    
            # if mu is not 0, it should be used to regularize the least squares problem
            # and "force" the background warp onto uncertain pixels
            else:
                # Calculate global parametrized warp
                G_avg = np.mean(ATA, axis=(0, 1))
                h_avg = np.mean(ATb, axis=(0, 1))
                p_avg = np.linalg.solve(G_avg, h_avg)
                d_avg = (S @ p_avg[..., None])[..., 0]
    
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
                d = np.linalg.solve(G + mu * np.eye(2), h + mu * d_avg)
    
        # TODO: return global displacement parameters and/or global displacement if mu != 0
    
        return d
