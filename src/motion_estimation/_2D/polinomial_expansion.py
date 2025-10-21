'''Farneback's polinomial expansion (2D). See https://github.com/ericPrince/optical-flow'''

import numpy as np
import scipy
import logging
import inspect

class Polinomial_Expansion():

    def __init__(self, logger):
        self.logger = logger

    def poly_expand(self, f, c, sigma):
        """
        Calculates the local polynomial expansion of a 2D signal, as described by Farneback
        Uses separable normalized correlation
        
        $f ~ x^T A x + B^T x + C$
        
        If f[i, j] and c[i, j] are the signal value and certainty of pixel (i, j) then
        A[i, j] is a 2x2 array representing the quadratic term of the polynomial, B[i, j]
        is a 2-element array representing the linear term, and C[i, j] is a scalar
        representing the constant term.
        
        Parameters
        ----------
        f
            Input signal
        c
            Certainty of signal
        sigma
            Standard deviation of applicability Gaussian kernel
        Returns
        -------
        A
            Quadratic term of polynomial expansion
        B
            Linear term of polynomial expansion
        C
            Constant term of polynomial expansion
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

        # Calculate applicability kernel (1D because it is separable)
        poly_n = int(4 * sigma + 1)
        x = np.arange(-poly_n, poly_n + 1, dtype=np.int32)
        a = np.exp(-(x**2) / (2 * sigma**2))  # a: applicability kernel [n]
    
        # b: calculate b from the paper. Calculate separately for X and Y dimensions
        # [n, 6]
        bx = np.stack(
            [np.ones(a.shape), x, np.ones(a.shape), x**2, np.ones(a.shape), x], axis=-1
        )
        by = np.stack(
            [np.ones(a.shape), np.ones(a.shape), x, np.ones(a.shape), x**2, x, ], axis=-1,
        )
    
        # Pre-calculate product of certainty and signal
        cf = c * f
    
        # G and v are used to calculate "r" from the paper: v = G*r (see Eq. 4.9 of the thesis)
        # r is the parametrization of the 2nd order polynomial for f
        G = np.empty(list(f.shape) + [bx.shape[-1]] * 2)
        v = np.empty(list(f.shape) + [bx.shape[-1]])
    
        # Apply separable cross-correlations
    
        # Pre-calculate quantities recommended in paper (Eq. 4.6 in the thesis)
        ab = np.einsum("i,ij->ij", a, bx) # ab[i,j] = bx[i,j]*a[i] (multiply each row of "bx" by the corresponding element of "a")
        abb = np.einsum("ij,ik->ijk", ab, bx) # abb[i,j,k] = ab[i,j]*bx[j,k]
        
        # Calculate G and v for each pixel with cross-correlation (axis 0)
        #print("bx.shape[-1]", bx.shape[-1])
        for i in range(bx.shape[-1]):
            for j in range(bx.shape[-1]):
                #print("G[..., i, j].shape", G[..., i, j].shape)
                G[..., i, j] = scipy.ndimage.correlate1d(
                    c, abb[..., i, j], axis=0, mode="constant", cval=0
                )
    
            v[..., i] = scipy.ndimage.correlate1d(
                cf, ab[..., i], axis=0, mode="constant", cval=0
            )
    
        # Pre-calculate quantities recommended in paper
        ab = np.einsum("i,ij->ij", a, by)
        abb = np.einsum("ij,ik->ijk", ab, by)
    
        # Calculate G and v for each pixel with cross-correlation (axis 1)
        for i in range(bx.shape[-1]):
            for j in range(bx.shape[-1]):
                G[..., i, j] = scipy.ndimage.correlate1d(
                    G[..., i, j], abb[..., i, j], axis=1, mode="constant", cval=0
                )
    
            v[..., i] = scipy.ndimage.correlate1d(
                v[..., i], ab[..., i], axis=1, mode="constant", cval=0
            )
    
        # Solve r for each pixel (eq. 4.8 of the thesis)
        print("G.shape", G.shape)
        print("v.shape", v.shape)
        r = np.linalg.solve(G, v)

        # Basis (see eq. 4.2 of the thesis):
        # 1 x 1 x^2   1   x 
        # 1 1 y   1 y^2   y
        # -----------------
        # 1 x y x^2 y^2 xy
    
        # Quadratic term
        A = np.empty(list(f.shape) + [2, 2])
        A[..., 0, 0] = r[..., 3]
        A[..., 0, 1] = r[..., 5] / 2
        A[..., 1, 0] = A[..., 0, 1]
        A[..., 1, 1] = r[..., 4]
    
        # Linear term
        B = np.empty(list(f.shape) + [2])
        B[..., 0] = r[..., 1]
        B[..., 1] = r[..., 2]
    
        # constant term
        C = r[..., 0]
    
        return A, B, C

    def expand(self, f, c, window_side):

        if self.logger.level <= logging.INFO:
            print(f"\nFunction: {inspect.currentframe().f_code.co_name}")
            args, _, _, values = inspect.getargvalues(inspect.currentframe())
            for arg in args:
                if isinstance(values[arg], np.ndarray):
                    print(f"{arg}.shape: {values[arg].shape}", end=' ')
                    print(f"{np.min(values[arg])} {np.average(values[arg])} {np.max(values[arg])}")
                else:
                    print(f"{arg}: {values[arg]}")

        sigma = (window_side - 1)/4
        return self.poly_expand(f, c, sigma)