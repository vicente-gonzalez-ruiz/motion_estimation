from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypedDict

import numpy as np
from scipy.ndimage import correlate1d

import logging
import inspect

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["FlowIterativeOptions", "__version__", "flow_iterative", "poly_exp_1d"]


__version__ = "1.1.0"


class FlowIterativeOptions(TypedDict, total=False):
    """Options for flow_iterative"""

    poly_sigma: float
    """Standard deviation of Gaussian used for polynomial expansion"""
    flow_sigma: float
    """Standard deviation of Gaussian used for integration"""
    iterations: int
    """Number of iterations to run"""
    w_update: bool
    """Update displacement field with flow from previous iteration"""
    border: Literal["constant", "reflect"]

class Polynomial_Expansion():

    def __init__(
        self,
        logger
        #logging_level=logging.INFO
    ):
        #self.logging_level = logging_level
        self.logger = logger

    def poly_expand[DType: np.floating[Any]](self,
            f: NDArray[DType], c: NDArray[DType], sigma: float
    ) -> tuple[NDArray[DType], NDArray[DType], NDArray[DType]]:
        """
        Calculates the local polynomial expansion of a 1D signal.

        This is a 1D adaptation of the 2D method described by Farneback.
        It approximates the signal `f` in the neighborhood of each point `p`
        with a quadratic polynomial: f(x) ~ A*(x-p)^2 + B*(x-p) + C

        Parameters
        ----------
        f
            Input 1D signal.
        c
            Certainty of the signal at each point.
        sigma
            Standard deviation of the applicability Gaussian kernel, which defines
            the size of the neighborhood.

        Returns
        -------
        A
            Quadratic term of the polynomial expansion (1D array).
        B
            Linear term of the polynomial expansion (1D array).
        C
            Constant term of the polynomial expansion (1D array).
        """
        dtype = f.dtype

        # 1. Calculate the 1D applicability kernel (a Gaussian)
        n = int(4 * sigma + 1)
        x = np.arange(-n, n + 1, dtype=dtype)
        a = np.exp(-(x**2) / (2 * sigma**2), dtype=dtype)  # a: applicability kernel

        # 2. Define the basis vectors for the polynomial: [1, x, x^2]
        # This corresponds to the coefficients [C, B, A]
        b = np.stack(
            [
                np.ones_like(x),
                x,
                x**2,
            ],
            axis=-1,
        )

        # 3. Pre-calculate product of certainty and signal
        cf = c * f

        # 4. Set up the linear system G*r = v to solve for the polynomial coefficients r
        # G is a matrix and v is a vector for each point in the signal.
        # r is the parameter vector [C, B, A]^T
        G = np.empty((*f.shape, b.shape[-1], b.shape[-1]), dtype=dtype)
        v = np.empty((*f.shape, b.shape[-1]), dtype=dtype)

        # 5. Use correlation to efficiently calculate G and v for every point.
        # Pre-calculate quantities as recommended in the Farneback paper.
        ab = a[:, np.newaxis] * b
        abb = ab[:, :, np.newaxis] * b[:, np.newaxis, :]

        # Calculate G and v for each point with 1D cross-correlation
        for i in range(b.shape[-1]):
            for j in range(b.shape[-1]):
                G[..., i, j] = correlate1d(
                    c, abb[..., i, j], axis=0, mode="constant", cval=0
                )
                v[..., i] = correlate1d(cf, ab[..., i], axis=0, mode="constant", cval=0)

        # 6. Solve the system G*r = v for r at each point in the signal
        # Add a small identity matrix to G for stability to handle ill-conditioned cases
        G += np.eye(G.shape[-1], dtype=dtype) * 1e-6

        # Initialize r with the correct shape
        r = np.empty_like(v)

        # Solve the linear system for each point
        for i in range(f.shape[0]):
            r[i, :] = np.linalg.solve(G[i, ...], v[i, :])


        # 7. Extract coefficients A, B, and C from the parameter vector r
        C = r[..., 0]  # Constant term
        B = r[..., 1]  # Linear term
        A = r[..., 2]  # Quadratic term

        return A, B, C

    def expand(self, f, c, window_length):

        if self.logger.getEffectiveLevel() <= logging.INFO:
        #if self.logging_level <= logging.INFO:
            print(f"\nFunction: {inspect.currentframe().f_code.co_name}")
            args, _, _, values = inspect.getargvalues(inspect.currentframe())
            for arg in args:
                if isinstance(values[arg], np.ndarray):
                    print(f"{arg}.shape: {values[arg].shape}", end=' ')
                    print(f"{np.min(values[arg])} {np.average(values[arg])} {np.max(values[arg])}")
                else:
                    print(f"{arg}: {values[arg]}")

        sigma = (window_length - 1)/4
        return self.poly_expand(f, c, sigma)
