'''Farneback's optical flow algorithm (1D). See https://github.com/ericPrince/optical-flow'''

import numpy as np
import scipy
from functools import partial
import skimage.transform
import logging
#from polinomial_expansion import Polinomial_Expansion
#from pyramid_gaussian import Gaussian_Pyramid
from . import polinomial_expansion
from . import pyramid_gaussian

PYRAMID_LEVELS = 1
ITERATIONS = 1
WINDOW_LENGTH = 17
N_POLY = 17
DOWN_SCALE = 2 # Only integers

class OF_Estimation(polinomial_expansion.Polinomial_Expansion, pyramid_gaussian.Gaussian_Pyramid):

    def __init__(self, logging_level=logging.INFO):
        self.logging_level = logging_level
        #logging.basicConfig(format="[%(filename)s:%(lineno)s %(funcName)s()] %(message)s")
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.WARNING)
        #self.PE = polinomial_expansion.Polinomial_Expansion(logging_level)
        #self.gaussian_pyramid = pyramid_gaussian.Gaussian_Pyramid()

    def flow_iterative(
        self,
        f1, f2, c1, c2,
        flow=None,
        sigma=4.0,
        sigma_flow=4.0,
        iterations=ITERATIONS,
        model="constant",
        mu=None
    ):

        """
        Calculates optical flow using only one level of the algorithm described by Gunnar Farneback
    
        Parameters
        ----------
        f1
            First signal
        f2
            Second signal
        sigma
            Polynomial expansion applicability Gaussian kernel sigma
        c1
            Certainty of first signal
        c2
            Certainty of second signal
        sigma_flow
            Applicability window Gaussian kernel sigma for polynomial matching
        iterations
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
            Optical flow field. flow[i] is the x displacement for sample i
        """

        if self.logging_level <= logging.INFO:
            print(f"\nFunction: {inspect.currentframe().f_code.co_name}")
            args, _, _, values = inspect.getargvalues(inspect.currentframe())
            for arg in args:
                if isinstance(values[arg], np.ndarray):
                    print(f"{arg}.shape: {values[arg].shape}", end=' ')
                    print(f"{np.min(values[arg])} {np.average(values[arg])} {np.max(values[arg])}")
                else:
                    print(f"{arg}: {values[arg]}")
                    
        # TODO: add initial warp parameters as optional input?

        # Calculate the polynomial expansion at each sample in the signals
        A1, B1, C1 = self.poly_expand(f1, c1, sigma)
        A2, B2, C2 = self.poly_expand(f2, c2, sigma)

        # Sample indexes in the signals
        x = np.arange(f1.shape[0])[:, None].astype(int)
        #print(x)

        # Initialize the displacements field
        if flow is None:
            flow = np.zeros(list(f1.shape) + [1])
    
        # Set up applicability convolution window
        #sigma_flow = (w/2 - 1)/4
        n_flow = int(4 * sigma_flow + 1)
        xw = np.arange(-n_flow, n_flow + 1)
        app_conv_win = np.exp(-(xw**2) / (2 * sigma_flow**2))
    
        # Evaluate warp parametrization model at pixel coordinates
        if model == "constant":
            S = np.eye(1)
    
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
    
        S_T = S.swapaxes(-1, -2) # Without effect in 1D

        # Iterate convolutions to estimate the optical flow
        for _ in range(iterations):
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
            c_ = c1[x_[..., 0]]
            c_[off_f] = 0
    
            # Calculate A and delB for each point, according to paper
            A = (A1 + A2[x_[..., 0]]) / 2  # Eq. 7.12 (see also Fig. 7.8)
            #print(A1, A2)
            A *= c_[
                ..., None, None
            ]  # recommendation in paper: add in certainty by applying to A and delB
    
            #print(A.shape, d_[..., None].shape)
            delB = -1 / 2 * (B2[x_[..., 0]] - B1) + (A @ flow_[..., None])[..., 0]
            delB *= c_[
                ..., None
            ]  # recommendation in paper: add in certainty by applying to A and delB
    
            # Pre-calculate quantities recommended by paper
            A_T = A.swapaxes(-1, -2) # Without effect in 1D
            ATA = S_T @ A_T @ A @ S # G(x) in the thesis (see Fig. 7.8)
            ATb = (S_T @ A_T @ delB[..., None])[..., 0] # h(x) in the thesis (see Fig. 7.8)
            # btb = delB.swapaxes(-1, -2) @ delB
    
            # If mu is 0, it means the global/average parametrized warp should not be
            # calculated, and the parametrization should apply to the local calculations
            if mu == 0:
                # Apply separable cross-correlation to calculate linear equation
                # for each pixel: G*d = h
                G = scipy.ndimage.correlate1d(ATA, app_conv_win, axis=0, mode="constant", cval=0)
                #G = scipy.ndimage.correlate1d(G, w, axis=1, mode="constant", cval=0)
    
                h = scipy.ndimage.correlate1d(ATb, app_conv_win, axis=0, mode="constant", cval=0)
                #h = scipy.ndimage.correlate1d(h, w, axis=1, mode="constant", cval=0)
    
                flow = (S @ np.linalg.solve(G, h)[..., None])[..., 0]
    
            # if mu is not 0, it should be used to regularize the least squares problem
            # and "force" the background warp onto uncertain pixels
            else:
                # Calculate global parametrized warp
                G_avg = np.mean(ATA, axis=(0))
                h_avg = np.mean(ATb, axis=(0))
                p_avg = np.linalg.solve(G_avg, h_avg)
                flow_avg = (S @ p_avg[..., None])[..., 0]
    
                # Default value for mu is to set mu to 1/2 the trace of G_avg
                if mu is None:
                    mu = 1 / 2 * np.trace(G_avg)
    
                # Apply separable cross-correlation to calculate linear equation
                G = scipy.ndimage.correlate1d(A_T @ A, app_conv_win, axis=0, mode="constant", cval=0)
                #G = scipy.ndimage.correlate1d(G, w, axis=1, mode="constant", cval=0)
    
                h = scipy.ndimage.correlate1d(
                    (A_T @ delB[..., None])[..., 0], app_conv_win, axis=0, mode="constant", cval=0
                )
                #h = scipy.ndimage.correlate1d(h, w, axis=1, mode="constant", cval=0)
    
                # Refine estimate of displacement field
                flow = np.linalg.solve(G + mu * np.eye(1), h + mu * flow_avg)
    
        # TODO: return global displacement parameters and/or global displacement if mu != 0
    
        return flow

    def get_flow_iteration(
        self,
        f1, f2, c1, c2,
        flow=None,
        N_poly=N_POLY,
        window_length=WINDOW_LENGTH,
        iterations=ITERATIONS,
        model="constant",
        mu=None
    ):

       if self.logging_level <= logging.INFO:
            print(f"\nFunction: {inspect.currentframe().f_code.co_name}")
            args, _, _, values = inspect.getargvalues(inspect.currentframe())
            for arg in args:
                if isinstance(values[arg], np.ndarray):
                    print(f"{arg}.shape: {values[arg].shape}", end=' ')
                    print(f"{np.min(values[arg])} {np.average(values[arg])} {np.max(values[arg])}")
                else:
                    print(f"{arg}: {values[arg]}")

        sigma = (N_poly - 1)/4
        sigma_flow = (window_length - 1)/4
        return self.flow_iterative(
            f1=f1, f2=f2, c1=c1, c2=c2,
            flow=flow, sigma=sigma, sigma_flow=sigma_flow,
            iterations=iterations,
            model=model,
            mu=mu)

    def pyramid_get_flow(
        self,
        target,
        reference,
        flow=None,
        pyramid_levels=PYRAMID_LEVELS, 
        down_scale=DOWN_SCALE,
        window_length=WINDOW_LENGTH,
        iterations=ITERATIONS,
        N_poly=N_POLY,
        model="constant",
        mu=None): # target and reference double's

        if self.logging_level <= logging.INFO:
            print(f"\nFunction: {inspect.currentframe().f_code.co_name}")
            args, _, _, values = inspect.getargvalues(inspect.currentframe())
            for arg in args:
                if isinstance(values[arg], np.ndarray):
                    print(f"{arg}.shape: {values[arg].shape}", end=' ')
                    print(f"{np.min(values[arg])} {np.average(values[arg])} {np.max(values[arg])}")
                else:
                    print(f"{arg}: {values[arg]}")

        c1 = np.ones_like(reference)
        c2 = np.ones_like(target)
        #c2 = c1
    
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
                            partial(self.get_pyramid,
                                    num_levels=pyramid_levels,
                                    down_scale=down_scale),
                            [reference, target, c1, c2],
                        )
                    )
                )
            )
        ):
            if flow is not None:
                # TODO: account for shapes not quite matching
                #d = skimage.transform.pyramid_expand(d, multichannel=True)
                flow = self.expand_level(np.squeeze(flow))[:, None]
                flow = flow[: pyr1.shape[0]]

            flow = self.get_flow_iteration(
                f1=pyr1, f2=pyr2, c1=c1_, c2=c2_,
                flow=flow,
                N_poly=N_poly,
                window_length=window_length,
                iterations=iterations,
                **opts)

        #xw = d + np.moveaxis(np.indices(target.shape), 0, -1)
        #return xw
        return flow
