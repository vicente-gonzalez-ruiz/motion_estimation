'''* Local quadratic model in 3D

For a local region of a 3D scalar field 
$f(x)$ with $x=(x,y,z)^T$,
we approximate it by a second-order polynomial around a local origin (say 
$x_0=0$):

$f(x)≈x^T A x + b^T x + c$

where

A is a symmetric 3×3 matrix (6 unique coefficients),

b is a 3×1 vector (linear term),

c is a scalar (constant term).

Expanding this:

f(x,y,z)=a_{xx} x^2 + a_{yy} y^2 + a_{zz} z^2 + 2a_{xy} x  + 2a_{xz} xz + 2a_{yz} yz + b_x x + b_y y + b_z z + c

* If the signal is shifted by vector  $d = (d_x, d_y, d_z)^T$:

$f′(x)=f(x−d)$

then the new polynomial coefficients are related by:

A' = A
b' = b - aAd
c' = c + d^TA d - b^T d

So given two local expansions (A_1,b_1,c_1) and (A_2,b_2,c_2), we can estimate the displacement vector $d$ by:

$2A_text{avg} d = b_1 - b2$

and therefore,

$d = \frac{1}{2}A^{-1}_\text{avg}(b_1 - b_2)$

where

$A_\text{avg} = \frac{A_1 + A_1}{2}$.

* Fitting the coefficients by weighted least squares

For each voxel (center position), we fit the polynomial coefficients
by minimizing:

$\min_{A,b,c}\sum_uw(u)(f(x_0+u)-(u^TAu ç b^tu + x))^2$

where $w(u)$ is a Gaussian weighting function in 3D.

Because $A$ is symmetric, we only solve for 10 unknowns:

$[a_{xx}​,a_{yy}​,a_{zz}​,a_{xy}​,a_{xz}​,a_{yz}​,b_x​,b_y​,b_z​,c]$

We can write this as a linear system 
$Xθ=f$, weighted by $w$.

* Notes & remarks

The matrix $A$ encodes curvatures (local structure tensor-like info).
If $A$ is nearly singular (flat region), the displacement estimate is
unreliable.

For full dense optical flow or motion field, you would:

a) Fit $A,b,c$ for every voxel (efficiently via separable convolutions in x/y/z).
b) Then compute 
$d(x)$ everywhere by solving the local 3×3 system.

The same equations generalize to higher dimensions: always 
$b′=b−2Ad$.
'''

import numpy as np

def gaussian_kernel_3d(radius, sigma):
    """Create a normalized 3D Gaussian kernel."""
    ax = np.arange(-radius, radius+1)
    xx, yy, zz = np.meshgrid(ax, ax, ax, indexing='ij')
    w = np.exp(-0.5 * (xx**2 + yy**2 + zz**2) / sigma**2)
    w /= w.sum()
    return xx, yy, zz, w

def fit_quadratic_wls_3d(volume, center, radius=2, sigma=1.0):
    """
    Fit a 3D quadratic polynomial around `center` in `volume`.
    Returns A (3x3 symmetric), b (3,), c (scalar)
    """
    cx, cy, cz = center
    sx, sy, sz = volume.shape
    xx, yy, zz, w = gaussian_kernel_3d(radius, sigma)
    # Extract local patch
    xmin, xmax = max(0, cx-radius), min(sx, cx+radius+1)
    ymin, ymax = max(0, cy-radius), min(sy, cy+radius+1)
    zmin, zmax = max(0, cz-radius), min(sz, cz+radius+1)
    patch = volume[xmin:xmax, ymin:ymax, zmin:zmax]
    # Corresponding kernel slice
    kx0 = radius - (cx - xmin)
    kx1 = kx0 + patch.shape[0]
    ky0 = radius - (cy - ymin)
    ky1 = ky0 + patch.shape[1]
    kz0 = radius - (cz - zmin)
    kz1 = kz0 + patch.shape[2]
    wx = w[kx0:kx1, ky0:ky1, kz0:kz1]
    xx = xx[kx0:kx1, ky0:ky1, kz0:kz1]
    yy = yy[kx0:kx1, ky0:ky1, kz0:kz1]
    zz = zz[kx0:kx1, ky0:ky1, kz0:kz1]

    # Flatten data
    X1 = xx**2
    X2 = yy**2
    X3 = zz**2
    X4 = xx*yy
    X5 = xx*zz
    X6 = yy*zz
    X7 = xx
    X8 = yy
    X9 = zz
    X10 = np.ones_like(xx)

    X = np.stack([X1,X2,X3,X4,X5,X6,X7,X8,X9,X10], axis=-1).reshape(-1,10)
    y = patch.reshape(-1)
    wv = wx.reshape(-1)
    W = np.diag(wv)

    XT_W = X.T * wv
    G = XT_W @ X
    rhs = XT_W @ y
    eps = 1e-10
    coeffs = np.linalg.solve(G + eps*np.eye(10), rhs)

    a_xx,a_yy,a_zz,a_xy,a_xz,a_yz,bx,by,bz,c = coeffs
    A = np.array([
        [a_xx, a_xy, a_xz],
        [a_xy, a_yy, a_yz],
        [a_xz, a_yz, a_zz]
    ])
    b = np.array([bx, by, bz])
    return A, b, c

def estimate_shift_3d(A1,b1,c1, A2,b2,c2, eps=1e-8):
    """
    Estimate 3D translation between two polynomial expansions.
    Returns displacement vector d.
    """
    A_avg = 0.5*(A1 + A2)
    try:
        d = 0.5 * np.linalg.solve(A_avg + eps*np.eye(3), (b1 - b2))
    except np.linalg.LinAlgError:
        d = np.zeros(3)
    return d

# --- Example ---
if __name__ == "__main__":
    # Synthetic test: a quadratic function shifted in 3D
    gx, gy, gz = np.mgrid[-4:5, -4:5, -4:5]
    true_d = np.array([0.6, -0.8, 1.2])
    f1 = 0.4*gx**2 + 0.5*gy**2 + 0.6*gz**2 + 0.3*gx*gy - 0.2*gx*gz + 0.1*gy*gz + 0.8*gx - 0.5*gy + 0.3*gz + 1.0
    f2 = 0.4*(gx-true_d[0])**2 + 0.5*(gy-true_d[1])**2 + 0.6*(gz-true_d[2])**2 \
       + 0.3*(gx-true_d[0])*(gy-true_d[1]) - 0.2*(gx-true_d[0])*(gz-true_d[2]) + 0.1*(gy-true_d[1])*(gz-true_d[2]) \
       + 0.8*(gx-true_d[0]) - 0.5*(gy-true_d[1]) + 0.3*(gz-true_d[2]) + 1.0

    A1,b1,c1 = fit_quadratic_wls_3d(f1, center=(4,4,4), radius=3, sigma=2.0)
    A2,b2,c2 = fit_quadratic_wls_3d(f2, center=(4,4,4), radius=3, sigma=2.0)
    d_est = estimate_shift_3d(A1,b1,c1, A2,b2,c2)
    print("True shift:", true_d)
    print("Estimated shift:", d_est)
