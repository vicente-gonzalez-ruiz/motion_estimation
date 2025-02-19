'''Farneback's polynomial expansion (1D). See https://github.com/ericPrince/optical-flow'''

import numpy as np
import scipy
import logging
import inspect

class Polinomial_Expansion():

    def __init__(
        self,
        logger
        #logging_level=logging.INFO
    ):
        #self.logging_level = logging_level
        self.logger = logger

    def poly_expand(self, f, c, sigma):
    #def poly_expand(self, f, c, poly_n):
        """
        Calculates the local polynomial expansion of a 1D signal.
        
        $f ~ x^T A x + B^T x + C$
        
        If f[i] and c[i] are the signal value and certainty of sample i then
        A[i] is a 1x1 array representing the quadratic term of the polynomial, B[i]
        is a 1-element array representing the linear term, and C[i] is a scalar
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
                    
        # Kernel applicability (gaussian)
        poly_n = int(4 * sigma + 1)
        #sigma = (poly_n/2 - 1)/4
        x = np.arange(-poly_n, poly_n + 1, dtype=np.int32)
        a = np.exp(-(x**2) / (2 * sigma**2))
        #x = np.arange(poly_n, dtype=np.int32)
        #a = np.ones(poly_n)
        #print("poly_n=", poly_n, "sigma=", sigma, a)

        # b: calculate b from the paper.
        b = np.stack([np.ones(a.shape), x, x**2], axis=-1)
    
        # Pre-calculate product of certainty and signal
        cf = c * f
    
        # G and v are used to calculate "r" from the paper: v = G*r
        # r is the parametrization of the 2nd order polynomial for f
        G = np.empty(list(f.shape) + [b.shape[-1]] * 2)
        v = np.empty(list(f.shape) + [b.shape[-1]])

        # Apply cross-correlation
    
        # Pre-calculate quantities recommended in paper
        ab = np.einsum("i,ij->ij", a, b) # ab[i] = b[i]*a[i]
        abb = np.einsum("ij,ik->ijk", ab, b) # abb[i,j] = ab[i]*b[j]
    
        # Calculate G and v for each sample with cross-correlation
        for i in range(b.shape[-1]):
            for j in range(b.shape[-1]):
                #print("G[..., i, j].shape", G[..., i, j].shape)
                G[..., i, j] = scipy.ndimage.correlate1d(
                    c, abb[..., i, j], axis=0, mode="constant", cval=0
                )
    
            v[..., i] = scipy.ndimage.correlate1d(
                cf, ab[..., i], axis=0, mode="constant", cval=0
            )
    
        # Solve r for each pixel
        r = np.linalg.solve(G, v)
    
        # Quadratic term
        A = np.empty(list(f.shape) + [1, 1])
        A[..., 0, 0] = r[..., 2]
    
        # Linear term
        B = np.empty(list(f.shape) + [1])
        B[..., 0] = r[..., 1]
    
        # constant term
        C = r[..., 0]
    
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