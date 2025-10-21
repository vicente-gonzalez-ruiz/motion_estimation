'''Farneback's polynomial expansion (1D).

Farneback models a small neighbourhood of samples by a quadratic polynomial,

p(x) = ax^2 + bx + c.

If a signal patch is translated by d, the polynomial coefficients change as

p(x-d) = ax^2 + (b-2ad)x + (c + ad^2 - bd).

So, if you have the fitted coefficients (a_1, b_1, c_1) for patch 1
and (a_2, b_2, c_2) for patch 2 that is a translated version, you can recover the translation (when curvature $a$ is nonzero) with

d \approx \frac{b_1 - b_2){2 a_\text{avg}}

and

a_\text{avg} = \frac{a_1 + a_2}{2}.

See https://github.com/ericPrince/optical-flow

'''


import numpy as np
import scipy
import logging
import inspect

class Polynomial_Expansion(): # Eric Prince

    def __init__(
        self,
        logger
        #logging_level=logging.INFO
    ):
        #self.logging_level = logging_level
        self.logger = logger

    def _generate_gaussian_kernel(self, sigma):
        poly_n = int(4 * sigma + 1)
        #sigma = (poly_n/2 - 1)/4
        x = np.arange(-poly_n, poly_n + 1, dtype=np.int32)
        a = np.exp(-(x**2) / (2 * sigma**2))
        #x = np.arange(poly_n, dtype=np.int32)
        #a = np.ones(poly_n)
        #print("poly_n=", poly_n, "sigma=", sigma, a)
        return a, x

    def poly_expand(self, f, c, sigma):
    #def poly_expand(self, f, c, poly_n):
        """Calculates the local polynomial expansion of a 1D signal.
        
        $f(x) ~ x^T A x + B^T x + C$

        Basis functions:

        $\{1, x, x^2\}$

        In 1D (see Eq. 4.4),

        $(x) A (x) + B^T (x) + c = r_1 + r_2 x + r_3 x^2$

        and therefore, $r_1 = C$, $r_2 = B^T = B$ and $r_3^2 = A$. 
        
        If f[i] and c[i] are the signal value and certainty of sample
        i then A[i] is a 1x1 array representing the quadratic term of
        the polynomial, B[i] is a 1-element array representing the
        linear term, and C[i] is a scalar representing the constant
        term.
        
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

        if self.logger.level <= logging.INFO:
            print(f"\nFunction: {inspect.currentframe().f_code.co_name}")
            args, _, _, values = inspect.getargvalues(inspect.currentframe())
            for arg in args:
                if isinstance(values[arg], np.ndarray):
                    print(f"{arg}.shape: {values[arg].shape}", end=' ')
                    print(f"{np.min(values[arg])} {np.average(values[arg])} {np.max(values[arg])}")
                else:
                    print(f"{arg}: {values[arg]}")
                    
        # Kernel applicability (gaussian)
        a, x = self._generate_gaussian_kernel(sigma)
        
        # b: calculate b from the paper.
        b = np.stack([np.ones(x.shape), x, x**2], axis=-1)
    
        # Pre-calculate product of certainty and signal
        cf = c * f
    
        # G and v are used to calculate "r" from the paper: v = G*r
        # r is the parametrization of the 2nd order polynomial for f
        G = np.empty(list(f.shape) + [b.shape[-1]] * 2)
        v = np.empty(list(f.shape) + [b.shape[-1]])

        # Apply cross-correlation
    
        # Pre-calculate quantities recommended in paper
        #ab = np.einsum("i,ij->ij", a, b) # ab[i] = b[i]*a[i]
        #abb = np.einsum("ij,ik->ijk", ab, b) # abb[i,j] = ab[i]*b[j]
        ab = a[:, np.newaxis] * b
        abb = ab[:, :, np.newaxis] * b[:, np.newaxis, :]
        
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
        # Add a small identity matrix to G for stability to handle ill-conditioned cases
        #G += np.eye(G.shape[-1], dtype=dtype) * 1e-6
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
