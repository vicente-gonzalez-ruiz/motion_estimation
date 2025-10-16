from __future__ import annotations

'''3D Farneback's optical flow estimation. See https://github.com/ericPrince/optical-flow.'''

import numpy as np
import scipy
from functools import partial
import skimage.transform
from . import polinomial_expansion
from . import pyramid_gaussian
import logging
import inspect

from typing import TYPE_CHECKING, Any, Literal, TypedDict

import numpy as np
from scipy.ndimage import correlate1d

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["FlowIterative3DOptions", "__version__", "flow_iterative_3d", "poly_exp_3d"]


__version__ = "1.1.0" # Version for 3D implementation


PYRAMID_LEVELS = 3 # Only integers
WINDOW_SIDE = 5 # Only integers
ITERATIONS = 7
N_POLY = 7
DOWN_SCALE = 2 # Only integers

class OF_Estimation(polinomial_expansion.Polinomial_Expansion, pyramid_gaussian.Gaussian_Pyramid):

    def __init__(self, logger):
        polinomial_expansion.Polinomial_Expansion.__init__(self, logger)
        pyramid_gaussian.Gaussian_Pyramid.__init__(self, logger)

    
    def flow_iterative[DType: np.floating[Any]](self,  # noqa: PLR0913, PLR0915
        f1: NDArray[DType],
        f2: NDArray[DType],
        c1: NDArray[DType],
        c2: NDArray[DType],
        *,
        sigma_poly: float,
        sigma_flow: float,
        num_iter: int = 1,
        d: NDArray[DType] | None = None,
        model: Literal["constant", "affine"] = "constant",
        mu: float | None = None,
    ) -> NDArray[DType]:
        """
        Calculates 3D optical flow using the iterative Farneback algorithm.
    
        Parameters
        ----------
        f1, f2
            First and second 3D volumes.
        c1, c2
            Certainty maps for the first and second volumes.
        sigma_poly
            Sigma for the polynomial expansion kernel.
        sigma_flow
            Sigma for the flow integration window.
        num_iter
            Number of iterations to perform.
        d
            Initial displacement field (optional).
        model
            Motion model to use ('constant' or 'affine').
        mu
            Weighting factor for the global motion model. If None, it's auto-determined.
    
        Returns
        -------
        d
            3D optical flow field. d[z, y, x] is the (dz, dy, dx) displacement.
        """
        dtype = f1.dtype
    
        # Calculate polynomial expansions for both volumes
        A1, B1, _ = self.poly_expand(f1, c1, sigma_poly)
        A2, B2, _ = self.poly_expand(f2, c2, sigma_poly)
    
        # Create a 3D grid of voxel coordinates
        coords = np.indices(f1.shape, dtype=int).transpose(1, 2, 3, 0) # z, y, x
    
        # Initialize displacement field if not provided
        if d is None:
            d = np.zeros((*f1.shape, 3), dtype=dtype)
    
        # Applicability window for flow calculation
        n_flow = int(4 * sigma_flow + 1)
        xw = np.arange(-n_flow, n_flow + 1, dtype=dtype)
        w = np.exp(-(xw**2) / (2 * sigma_flow**2))
    
        # --- Setup motion model parametrization matrix S ---
        if model == "constant":
            S = np.eye(3, dtype=dtype)
            num_params = 3
        elif model == "affine":
            num_params = 12  # 3 translation + 9 for matrix
            S = np.zeros((*f1.shape, 3, num_params), dtype=dtype)
            z, y, x = coords[..., 0], coords[..., 1], coords[..., 2]
            # dz = p0 + p1*z + p2*y + p3*x
            S[..., 0, 0] = 1; S[..., 0, 1] = z; S[..., 0, 2] = y; S[..., 0, 3] = x
            # dy = p4 + p5*z + p6*y + p7*x
            S[..., 1, 4] = 1; S[..., 1, 5] = z; S[..., 1, 6] = y; S[..., 1, 7] = x
            # dx = p8 + p9*z + p10*y + p11*x
            S[..., 2, 8] = 1; S[..., 2, 9] = z; S[..., 2, 10] = y; S[..., 2, 11] = x
        else:
            msg = "Invalid parametrization model for 3D"
            raise ValueError(msg)
    
        S_T = S.swapaxes(-1, -2)
    
        for _ in range(num_iter):
            # Discretize displacement field for neighborhood lookup
            d_ = d.astype(int)
            coords_ = coords + d_
    
            # Constrain coordinates to be within the volume bounds
            vol_shape = np.array(f1.shape)
            coords_2 = np.maximum(np.minimum(coords_, vol_shape - 1), 0)
            off_volume = np.any(coords_ != coords_2, axis=-1)
            coords_ = coords_2
    
            # Get certainty from warped coordinates, zeroing out-of-bounds points
            c_ = c2[coords_[..., 0], coords_[..., 1], coords_[..., 2]]
            c_[off_volume] = 0
    
            # Calculate A and ΔB from the paper's equations
            A = (A1 + A2[coords_[..., 0], coords_[..., 1], coords_[..., 2]]) / 2
            A *= c_[..., None, None]  # Apply certainty
    
            delB = -0.5 * (B2[coords_[..., 0], coords_[..., 1], coords_[..., 2]] - B1) + (A @ d_[..., None])[..., 0]
            delB *= c_[..., None] # Apply certainty
    
            # Pre-calculate components for solving the linear system
            A_T = A.swapaxes(-1, -2)
            ATA = S_T @ A_T @ A @ S
            ATb = (S_T @ A_T @ delB[..., None])[..., 0]
    
            if mu == 0:
                # Local model only
                G = correlate1d(ATA, w, axis=0, mode="constant", cval=0)
                G = correlate1d(G, w, axis=1, mode="constant", cval=0)
                G = correlate1d(G, w, axis=2, mode="constant", cval=0)
    
                h = correlate1d(ATb, w, axis=0, mode="constant", cval=0)
                h = correlate1d(h, w, axis=1, mode="constant", cval=0)
                h = correlate1d(h, w, axis=2, mode="constant", cval=0)
    
                p = np.linalg.solve(G, h[..., None])[..., 0].astype(dtype)
                d = (S @ p[..., None])[..., 0]
    
            else:
                # Global model with local refinement
                G_avg = np.mean(ATA, axis=(0, 1, 2))
                h_avg = np.mean(ATb, axis=(0, 1, 2))
                p_avg = np.linalg.solve(G_avg, h_avg[..., None])[..., 0]
                d_avg = (S @ p_avg[..., None])[..., 0]
    
                if mu is None:
                    mu = 0.5 * np.trace(G_avg)
    
                G_local = A_T @ A
                h_local = (A_T @ delB[..., None])[..., 0]
    
                G = correlate1d(G_local, w, axis=0, mode="constant", cval=0)
                G = correlate1d(G, w, axis=1, mode="constant", cval=0)
                G = correlate1d(G, w, axis=2, mode="constant", cval=0)
    
                h = correlate1d(h_local, w, axis=0, mode="constant", cval=0)
                h = correlate1d(h, w, axis=1, mode="constant", cval=0)
                h = correlate1d(h, w, axis=2, mode="constant", cval=0)
    
                d = np.linalg.solve(G + mu * np.eye(3), (h + mu * d_avg)[..., None])[..., 0]
    
        return d


    
    def flow_iterative_old(
        self,
        f1, f2, c1, c2,
        flow=None,
        sigma=1.0,
        sigma_flow=1.0,
        iterations=ITERATIONS,
        model="constant",
        mu=None
    ):
        """
        Calculates optical flow using only one level of the algorithm described by Gunnar Farneback
    
        Parameters
        ----------
        f1
            First vol
        f2
            Second vol
        sigma
            Polynomial expansion applicability Gaussian kernel sigma
        c1
            Certainty of first vol
        c2
            Certainty of second vol
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
            Optical flow field. flow[i, j, k] is the (z, y, x) displacement for pixel (i, j, k)
        """

        if self.logger.level <= logging.INFO:
            print(f"\nFunction: {inspect.currentframe().f_code.co_name}")
            args, _, _, values = inspect.getargvalues(inspect.currentframe())
            for arg in args:
                if isinstance(values[arg], np.ndarray):
                    print(f"{arg}.shape: {values[arg].shape}", end=' ')
                    print(f"{np.min(values[arg])} {np.average(values[arg])} {np.max(values[arg])}")
                else:
                    print(f"{arg}: {values[arg]}")
    
        # TODO: add initial warp parameters as optional input?
    
        # Calculate the polynomial expansion at each volxel in the volumes
        A1, B1, C1 = self.poly_expand(f1, c1, sigma)
        A2, B2, C2 = self.poly_expand(f2, c2, sigma)
    
        # Voxel coordinates of each point in the vols
        x = np.stack(
            np.meshgrid(
                np.arange(f1.shape[0]),
                np.arange(f1.shape[1]),
                np.arange(f1.shape[2]),
                indexing='ij'
            ),
            axis=-1
        ).astype(int)
    
        # Initialize the displacements field
        if flow is None:
            flow = np.zeros(list(f1.shape) + [3])
    
        # Set up applicability convolution window
        n_flow = int(4 * sigma_flow + 1)
        xw = np.arange(-n_flow, n_flow + 1)
        app_conv_win = np.exp(-(xw**2) / (2 * sigma_flow**2))
    
        # Evaluate warp parametrization model at pixel coordinates
        if model == "constant":
            S = np.eye(3)
    
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
        for _ in range(iterations):
            # Set flow~ as displacement field fit to nearest voxel (and constrain to not
            # being off vol). Note we are setting certainty to 0 for points that
            # would have been off-vol had we not constrained them
            flow_ = flow.astype(int)
            x_ = x + flow_
    
            # x_ = np.maximum(np.minimum(x_, np.array(f1.shape) - 1), 0)
    
            # Constrain d~ to be on-vol, and find points that would have
            # been off-vol
            x_2 = np.maximum(np.minimum(x_, np.array(f1.shape) - 1), 0)
            off_f = np.any(x_ != x_2, axis=-1)
            x_ = x_2
    
            # Set certainty to 0 for off-vol points
            c_ = c1[x_[..., 0], x_[..., 1], x_[..., 2]]
            c_[off_f] = 0
    
            # Calculate A and delB for each point, according to paper
            A = (A1 + A2[x_[..., 0], x_[..., 1], x_[..., 2]]) / 2  # Eq. 7.12 (see also Fig. 7.8)
    
            A *= c_[
                ..., None, None
            ]  # recommendation in paper: add in certainty by applying to A and delB
    
            delB = -1 / 2 * (B2[x_[..., 0], x_[..., 1], x_[..., 2]] - B1) + (A @ flow_[..., None])[..., 0]
            delB *= c_[
                ..., None
            ]  # recommendation in paper: add in certainty by applying to A and delB
    
            # Pre-calculate quantities recommended by paper
            A_T = A.swapaxes(-1, -2)
            ATA = S_T @ A_T @ A @ S # G(x) in the thesis (see Fig. 7.8)
            ATb = (S_T @ A_T @ delB[..., None])[..., 0] # h(x) in the thesis (see Fig. 7.8)
            # btb = delB.swapaxes(-1, -2) @ delB
    
            # If mu is 0, it means the global/average parametrized warp should not be
            # calculated, and the parametrization should apply to the local calculations
            if mu == 0:
                # Apply separable cross-correlation to calculate linear equation
                # for each voxel: G*d = h
                G = scipy.ndimage.correlate1d(ATA, app_conv_win, axis=0, mode="constant", cval=0)
                G = scipy.ndimage.correlate1d(G, app_conv_win, axis=1, mode="constant", cval=0)
                G = scipy.ndimage.correlate1d(G, app_conv_win, axis=2, mode="constant", cval=0)
    
                h = scipy.ndimage.correlate1d(ATb, app_conv_win, axis=0, mode="constant", cval=0)
                h = scipy.ndimage.correlate1d(h, app_conv_win, axis=1, mode="constant", cval=0)
                h = scipy.ndimage.correlate1d(h, app_conv_win, axis=2, mode="constant", cval=0)
                try:
                    flow = (S @ np.linalg.solve(G, h)[..., None])[..., 0]
                except np.linalg.LinAlgError:
                    print("G.shape:", G.shape)
                    print("h.shape:", h.shape)
                    _G = G.reshape(G.shape[0] * G.shape[1] * G.shape[2] , G.shape[3] * G.shape[4])
                    _h = h.reshape(h.shape[0] * h.shape[1] * h.shape[2] , h.shape[3])
                    __ = np.linalg.lstsq(_G, _h, rcond=None)[0]
                    #_ = __.reshape(list()) OJO!!!
                    _flow = (S @ __[..., None])[..., 0]
                    flow = _flow
    
            # if mu is not 0, it should be used to regularize the least squares problem
            # and "force" the background warp onto uncertain pixels
            else:
                # Calculate global parametrized warp
                G_avg = np.mean(ATA, axis=(0, 1, 2))
                h_avg = np.mean(ATb, axis=(0, 1, 2))
                p_avg = np.linalg.solve(G_avg, h_avg)
                flow_avg = (S @ p_avg[..., None])[..., 0]
    
                # Default value for mu is to set mu to 1/2 the trace of G_avg
                if mu is None:
                    mu = 1 / 2 * np.trace(G_avg)
    
                # Apply separable cross-correlation to calculate linear equation
                G = scipy.ndimage.correlate1d(A_T @ A, app_conv_win, axis=0, mode="constant", cval=0)
                G = scipy.ndimage.correlate1d(G, app_conv_win, axis=1, mode="constant", cval=0)
                G = scipy.ndimage.correlate1d(G, app_conv_win, axis=2, mode="constant", cval=0)
    
                h = scipy.ndimage.correlate1d(
                    (A_T @ delB[..., None])[..., 0], app_conv_win, axis=0, mode="constant", cval=0
                )
                h = scipy.ndimage.correlate1d(h, app_conv_win, axis=1, mode="constant", cval=0)
                h = scipy.ndimage.correlate1d(h, app_conv_win, axis=2, mode="constant", cval=0)
    
                # Refine estimate of displacement field
                flow = np.linalg.solve(G + mu * np.eye(3), h + mu * flow_avg)
    
        # TODO: return global displacement parameters and/or global displacement if mu != 0
    
        return flow

    def get_flow_iteration(
        self,
        f1, f2, c1, c2,
        d=None, # flow
        N_poly=N_POLY,
        window_side=WINDOW_SIDE,
        iterations=ITERATIONS,
        model="constant",
        mu=None):

        if self.logger.level <= logging.INFO:
            print(f"\nFunction: {inspect.currentframe().f_code.co_name}")
            args, _, _, values = inspect.getargvalues(inspect.currentframe())
            for arg in args:
                if isinstance(values[arg], np.ndarray):
                    print(f"{arg}.shape: {values[arg].shape}", end=' ')
                    print(f"{np.min(values[arg])} {np.average(values[arg])} {np.max(values[arg])}")
                else:
                    print(f"{arg}: {values[arg]}")

        sigma = (N_poly - 1)/4
        sigma_flow = (window_side - 1)/4
        return self.flow_iterative(
            f1=f1, f2=f2, c1=c1, c2=c2,
            d=d, sigma_poly=sigma, sigma_flow=sigma_flow,
            num_iter=iterations,
            model=model,
            mu=mu)

    def pyramid_get_flow(
        self,
        target,
        reference,
        flow=None,
        pyramid_levels=PYRAMID_LEVELS,
        down_scale=DOWN_SCALE,
        window_side=WINDOW_SIDE,
        iterations=ITERATIONS,
        N_poly=N_POLY,
        model="constant",
        mu=None): # target and reference double's

        if self.logger.level <= logging.INFO:
            print(f"\nFunction: {inspect.currentframe().f_code.co_name}")
            args, _, _, values = inspect.getargvalues(inspect.currentframe())
            for arg in args:
                if isinstance(values[arg], np.ndarray):
                    print(f"{arg}.shape: {values[arg].shape}", end=' ')
                    print(f"{np.min(values[arg])} {np.average(values[arg])} {np.max(values[arg])}")
                else:
                    print(f"{arg}: {values[arg]}")

        # c1 = np.ones_like(target)
        # c2 = np.ones_like(reference)
    
        # certainties for images - certainty is decreased for pixels near the edge
        # of the volume, as recommended by Farneback

        c1 = np.minimum(
            1, 1/5 * np.minimum(
                np.minimum(
                    np.arange(target.shape[0])[:, None, None],
                    np.arange(target.shape[1])[None, :, None]
                ),
                np.arange(target.shape[2])[None, None, :]
            )
        )
        
        c1 = np.minimum(
            c1,
            1/5 * np.minimum(
                np.minimum(
                    target.shape[0] - 1 - np.arange(target.shape[0])[:, None, None],
                    target.shape[1] - 1 - np.arange(target.shape[1])[None, :, None]
                ),
                target.shape[2] - 1 - np.arange(target.shape[2])[None, None, :]
            )
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
                expanded_Z_flow = 2*self.expand_level(flow[..., 0])[: pyr1.shape[0], : pyr1.shape[1], : pyr1.shape[2]]
                expanded_Y_flow = 2*self.expand_level(flow[..., 1])[: pyr1.shape[0], : pyr1.shape[1], : pyr1.shape[2]]
                expanded_X_flow = 2*self.expand_level(flow[..., 2])[: pyr1.shape[0], : pyr1.shape[1], : pyr1.shape[2]]
                print("pyr1.shape:", pyr1.shape)
                print("flow.shape:", flow.shape)
                print("expanded_Z_flow.shape:", expanded_Z_flow.shape)
                flow = np.empty(shape=(pyr1.shape[0], pyr1.shape[1], pyr1.shape[2], 3))
                print("flow[..., 0].shape=", flow[..., 0].shape)
                flow[..., 0] = expanded_Z_flow
                flow[..., 1] = expanded_Y_flow
                flow[..., 2] = expanded_X_flow
                #flow = self.expand_level(flow)
                #flow = flow[: pyr1.shape[0], : pyr1.shape[1], : pyr1.shape[2]]

            flow = self.get_flow_iteration(
                f1=pyr1, f2=pyr2, c1=c1_, c2=c2_,
                d=flow, # flow
                N_poly=N_poly,
                window_side=window_side,
                iterations=iterations,
                **opts)
    
        #xw = d + np.moveaxis(np.indices(target.shape), 0, -1)
        #return xw
        return flow[..., [2, 1, 0]] # (x, y, z) notation

