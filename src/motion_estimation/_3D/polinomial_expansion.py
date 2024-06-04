'''Farneback's polinomial expansion (3D). See https://github.com/ericPrince/optical-flow'''

import numpy as np
import scipy
import logging
import inspect
#from scipy.linalg import lstsq
#from scipy.optimize import least_squares

class Polinomial_Expansion():

    def __init__(self, logging_level=logging.INFO):
        self.logging_level = logging_level
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging_level)
        
    def poly_expand(self, f, c, sigma):
        """
        Calculates the local polynomial expansion of a 3D signal.
        
        $f ~ x^T A x + B^T x + C$
        
        If f[i, j. k] and c[i, j, k] are the signal value and certainty of voxel (i, j, k) then
        A[i, j, k] is a 3x3 array representing the quadratic term of the polynomial, B[i, j, k]
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
    
        # b: calculate b from the paper. Calculate separately for X, Y and Z dimensions
        # [n, 6]
        def _1(a):
            return np.ones(a.shape)
        bx = np.stack([_1(a),     x, _1(a), _1(a),  x**2, _1(a), _1(a),    x,    x, _1(a)], axis=-1)
        by = np.stack([_1(a), _1(a),     x, _1(a), _1(a),  x**2, _1(a),    x, _1(a),    x], axis=-1)
        bz = np.stack([_1(a), _1(a), _1(a),     x, _1(a), _1(a), x**2, _1(a),     x,    x], axis=-1)

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
        
        # Calculate G and v for each voxel with cross-correlation (axis 0)
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
    
        # Calculate G and v for each voxel with cross-correlation (axis 1)
        for i in range(bx.shape[-1]):
            for j in range(bx.shape[-1]):
                G[..., i, j] = scipy.ndimage.correlate1d(
                    G[..., i, j], abb[..., i, j], axis=1, mode="constant", cval=0
                )
    
            v[..., i] = scipy.ndimage.correlate1d(
                v[..., i], ab[..., i], axis=1, mode="constant", cval=0
            )

        # Pre-calculate quantities recommended in paper
        ab = np.einsum("i,ij->ij", a, bz)
        abb = np.einsum("ij,ik->ijk", ab, bz)
    
        # Calculate G and v for each voxel with cross-correlation (axis 1)
        for i in range(bx.shape[-1]):
            for j in range(bx.shape[-1]):
                G[..., i, j] = scipy.ndimage.correlate1d(
                    G[..., i, j], abb[..., i, j], axis=1, mode="constant", cval=0
                )
    
            v[..., i] = scipy.ndimage.correlate1d(
                v[..., i], ab[..., i], axis=1, mode="constant", cval=0
            )
    
        # Solve r for each voxel (eq. 4.8 of the thesis)
        print("G.shape", G.shape)
        print("v.shape", v.shape)
        try:
            r = np.linalg.solve(G, v)
        except np.linalg.LinAlgError:
            _G = G.reshape(G.shape[0] * G.shape[1], G.shape[2] * G.shape[3] * G.shape[4])
            _v = v.reshape(v.shape[0] * v.shape[1], v.shape[2] * v.shape[3])
            _r = np.linalg.lstsq(_G, _v, rcond=None)[0]
            r = _r.reshape(list(f.shape) + [10])

        # Basis (see eq. 4.2 of the thesis):
        # 1 x 1 1 x^2   1   1  x  x  1
        # 1 1 y 1   1 y^2   1  y  1  y
        # 1 1 1 z   1   1 z^2  1  z  z
        # ----------------------------
        # 1 x y z x^2 y^2 z^2 xy xz yz
        
        # (See eq. 4.4 of the thesis)
        #
        # (x y z) A (x y z)^T + b^T(x y z)^T + c =
        # r1 + r_2x + r_3y + r_3z + r_5x^2 + r_6y^2 + r_yz^2 + r_8xy + r_9xz + r_10yz
        #
        # where
        #
        # c = r_1
        #
        # b = (r_1 r_2 r_3)^T
        #
        #     /   r_5  r_8/2  r_9/2 \
        # A = | r_8/2    r_6 r_10/2 |
        #     \ r_9/2 r_10/2    r_7 /
        #
        #
        #        /  r_5  r_8/2  r_9/2 \ / x \   /  x^2r_5  xyr_8/2  xzr_9/2 \
        # (x y z)| r_8/2    r_6 r_10/2 || y | = | xyr_8/2   y^2r_6 yzr_10/2 |
        #        \ r_9/2 r_10/2    r_7 /\ z /   \ xzr_9/2 yzr_10/2   z^2r_7 /
     
        # Quadratic term
        A = np.empty(list(f.shape) + [3, 3])
        A[..., 0, 0] = r[..., 4]
        A[..., 0, 1] = r[..., 7]/2
        A[..., 0, 2] = r[..., 8]/2
        A[..., 1, 0] = r[..., 7]/2
        A[..., 1, 1] = r[..., 5]
        A[..., 1, 2] = r[..., 9]/2
        A[..., 2, 0] = r[..., 8]/2
        A[..., 2, 1] = r[..., 9]/2
        A[..., 2, 2] = r[..., 6]
    
        # Linear term
        B = np.empty(list(f.shape) + [3])
        B[..., 0] = r[..., 1]
        B[..., 1] = r[..., 2]
        B[..., 2] = r[..., 3]
    
        # constant term
        C = r[..., 0]
    
        return A, B, C

    def expand(self, f, c, window_side):

        if self.logging_level <= logging.INFO:
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