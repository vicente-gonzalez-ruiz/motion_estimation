'''

title{3D Polynomial Expansion (Farnebäck-style) for Optical Flow Estimation}
\author{Translated to \LaTeX{}}
\date{}
\maketitle

\section{Mathematical Background}

We generalize the Farnebäck local polynomial expansion to 3D signals (volumes or video cubes).

\subsection{Local quadratic model in 3D}

For a local region of a 3D scalar field \( f(\mathbf{x}) \), with \( \mathbf{x} = (x, y, z)^T \),
we approximate it by a second-order polynomial around a local origin (say \( \mathbf{x}_0 = 0 \)):

\[
f(\mathbf{x}) \approx \mathbf{x}^T A \mathbf{x} + \mathbf{b}^T \mathbf{x} + c
\]

where
\begin{itemize}
    \item \( A \) is a symmetric \( 3\times3 \) matrix (6 unique coefficients),
    \item \( \mathbf{b} \) is a 3-vector (linear term),
    \item \( c \) is a scalar (constant term).
\end{itemize}

Expanding this gives
\[
f(x, y, z) = a_{xx}x^2 + a_{yy}y^2 + a_{zz}z^2 + 2a_{xy}xy + 2a_{xz}xz + 2a_{yz}yz + b_x x + b_y y + b_z z + c.
\]

\subsection{Effect of translation}

If the signal is shifted by a vector \( \mathbf{d} = (d_x, d_y, d_z)^T \):

\[
f'(\mathbf{x}) = f(\mathbf{x} - \mathbf{d}),
\]

then the new polynomial coefficients are related by
\[
\begin{aligned}
A' &= A, \\
\mathbf{b}' &= \mathbf{b} - 2A\mathbf{d}, \\
c' &= c + \mathbf{d}^T A \mathbf{d} - \mathbf{b}^T \mathbf{d}.
\end{aligned}
\]

Thus, given two local expansions \( (A_1, \mathbf{b}_1, c_1) \) and \( (A_2, \mathbf{b}_2, c_2) \),
the local translation vector \( \mathbf{d} \) is obtained as

\[
2A_{\text{avg}} \mathbf{d} = \mathbf{b}_1 - \mathbf{b}_2
\quad \Rightarrow \quad
\boxed{\mathbf{d} = \tfrac{1}{2} A_{\text{avg}}^{-1} (\mathbf{b}_1 - \mathbf{b}_2)}
\]
where \( A_{\text{avg}} = \tfrac{1}{2}(A_1 + A_2) \).

\subsection{Fitting coefficients by weighted least squares}

For each voxel (center position), fit the polynomial coefficients by minimizing

\[
\min_{A, \mathbf{b}, c} \sum_{\mathbf{u}} w(\mathbf{u})
\big(f(\mathbf{x}_0 + \mathbf{u}) - (\mathbf{u}^T A \mathbf{u} + \mathbf{b}^T \mathbf{u} + c)\big)^2,
\]

where \( w(\mathbf{u}) \) is a 3D Gaussian weighting function.

Because \( A \) is symmetric, there are 10 unknowns:
\[
[a_{xx}, a_{yy}, a_{zz}, a_{xy}, a_{xz}, a_{yz}, b_x, b_y, b_z, c].
\]
We can write this as a linear system \( X\theta = f \) (weighted by \( w \)).

\section{Convolution-based Polynomial Expansion (Farnebäck style)}

\subsection{Polynomial model}

Define the 10 basis functions \( B_i(\mathbf{u}) \):
\[
\begin{aligned}
B_1 &= x^2, \quad B_2 = y^2, \quad B_3 = z^2,\\
B_4 &= xy, \quad B_5 = xz, \quad B_6 = yz,\\
B_7 &= x, \quad B_8 = y, \quad B_9 = z, \quad B_{10} = 1.
\end{aligned}
\]

For each voxel position \( \mathbf{x} \), the weighted least-squares normal equations are:
\[
G \theta = r(\mathbf{x}),
\]
where
\[
G_{ij} = \sum_{\mathbf{u}} w(\mathbf{u}) B_i(\mathbf{u}) B_j(\mathbf{u}),
\quad
r_i(\mathbf{x}) = \sum_{\mathbf{u}} w(\mathbf{u}) B_i(\mathbf{u}) f(\mathbf{x} + \mathbf{u}).
\]
Thus,
\[
\theta(\mathbf{x}) = G^{-1} r(\mathbf{x}).
\]

The \( r_i(\mathbf{x}) \) can be obtained by convolving \( f \) with
kernels \( k_i(\mathbf{u}) = w(\mathbf{u}) B_i(\mathbf{u}) \).

\subsection{Recovering the coefficients}

From \(\theta = [a_{xx}, a_{yy}, a_{zz}, a_{xy}, a_{xz}, a_{yz}, b_x, b_y, b_z, c]^T\),
we extract
\[
A = 
\begin{pmatrix}
a_{xx} & a_{xy} & a_{xz} \\
a_{xy} & a_{yy} & a_{yz} \\
a_{xz} & a_{yz} & a_{zz}
\end{pmatrix}, 
\quad
\mathbf{b} = 
\begin{pmatrix}
b_x \\ b_y \\ b_z
\end{pmatrix}, 
\quad
c.
\]

\subsection{Translation estimation}

Given two local expansions \((A_1,\mathbf{b}_1,c_1)\) and \((A_2,\mathbf{b}_2,c_2)\),
the translation vector is:
\[
\mathbf{d} = \tfrac{1}{2} A_{\text{avg}}^{-1} (\mathbf{b}_1 - \mathbf{b}_2),
\quad
A_{\text{avg}} = \tfrac{1}{2}(A_1 + A_2).
\]

\section{Notes}

\begin{itemize}
    \item The Gram matrix \( G \) depends only on the Gaussian kernel and is inverted once.
    \item The method is mathematically identical to weighted least squares at every voxel.
    \item When the curvature matrix \( A_{\text{avg}} \) is near-singular (flat regions), the displacement estimate is unreliable.
    \item The output coefficient field \( \theta \) has 10 channels per voxel.
\end{itemize}

'''

import numpy as np

def gaussian_kernel_3d(radius, sigma):
    """Return grid (x,y,z) and normalized 3D Gaussian kernel w(u)."""
    ax = np.arange(-radius, radius+1, dtype=float)
    x, y, z = np.meshgrid(ax, ax, ax, indexing='ij')
    w = np.exp(-0.5 * (x**2 + y**2 + z**2) / (sigma**2))
    w /= w.sum()
    return x, y, z, w

# Convolution helper: try scipy.ndimage first, else use FFT
def convolve3d(volume, kernel, mode='reflect'):
    """
    Convolve volume with kernel (same output shape).
    mode applies to scipy.ndimage.convolve; for FFT fallback edges are handled by zero-padding.
    """
    try:
        from scipy.ndimage import convolve
        return convolve(volume, kernel, mode=mode)
    except Exception:
        # FFT-based convolution via numpy. Uses zero-padding, then crops to 'same'.
        from numpy.fft import fftn, ifftn
        vol_shape = np.array(volume.shape)
        kern_shape = np.array(kernel.shape)
        out_shape = vol_shape + kern_shape - 1
        # compute FFTs
        F_vol = fftn(volume, out_shape)
        F_k = fftn(kernel, out_shape)
        conv = np.real(ifftn(F_vol * F_k))
        # crop center to original volume shape (same)
        start = (kern_shape - 1) // 2
        end = start + vol_shape
        sx, sy, sz = start
        ex, ey, ez = end
        return conv[sx:ex, sy:ey, sz:ez]

def build_basis_kernels(radius, sigma):
    """Build the 10 kernel-weighted basis kernels k_i(u)=w(u)*B_i(u)."""
    x, y, z, w = gaussian_kernel_3d(radius, sigma)
    # bases
    B = []
    B.append(x**2)    # B1
    B.append(y**2)    # B2
    B.append(z**2)    # B3
    B.append(x*y)    # B4
    B.append(x*z)    # B5
    B.append(y*z)    # B6
    B.append(x)      # B7
    B.append(y)      # B8
    B.append(z)      # B9
    B.append(np.ones_like(x))  # B10
    kernels = [w * Bi for Bi in B]   # k_i = w*B_i
    return np.stack(kernels, axis=0), (x,y,z,w)

def compute_gram_matrix_from_kernel(x,y,z,w):
    """
    Compute the 10x10 Gram matrix G_{ij} = sum_u w(u) B_i(u) B_j(u)
    using the basis definitions consistent with build_basis_kernels.
    """
    # Flatten arrays
    flat_w = w.ravel()
    flat_x = x.ravel()
    flat_y = y.ravel()
    flat_z = z.ravel()
    # construct basis matrix (Npoints x 10)
    Bmat = np.stack([
        flat_x**2,
        flat_y**2,
        flat_z**2,
        flat_x*flat_y,
        flat_x*flat_z,
        flat_y*flat_z,
        flat_x,
        flat_y,
        flat_z,
        np.ones_like(flat_x)
    ], axis=1)  # shape (N,10)
    # weighting: multiply each row by sqrt(w) then compute B^T B weighted
    Wdiag = flat_w
    # compute G = B^T diag(w) B efficiently:
    G = (Bmat.T * Wdiag) @ Bmat
    return G

def fit_quadratic_convolutional_3d(volume, radius=3, sigma=1.5, reg=1e-8):
    """
    Fit quadratic polynomial for every voxel using convolutional method.
    Returns theta_vol with shape (10, *volume.shape) where theta are the 10 coeffs per voxel.
    Coeff ordering: [a_xx,a_yy,a_zz,a_xy,a_xz,a_yz,bx,by,bz,c]
    """
    kernels, (x,y,z,w) = build_basis_kernels(radius, sigma)
    # convolve volume with each kernel to get r_i arrays
    r_list = []
    for i in range(kernels.shape[0]):
        r = convolve3d(volume, kernels[i])
        r_list.append(r)
    # stack into shape (10, X, Y, Z)
    r_stack = np.stack(r_list, axis=0)
    # compute Gram matrix G and inverse
    G = compute_gram_matrix_from_kernel(x,y,z,w)
    # regularize and invert
    G_reg = G + reg * np.eye(10)
    G_inv = np.linalg.inv(G_reg)
    # reshape r_stack to (10, N) where N = voxels, then theta = G_inv @ r_stack
    shape = volume.shape
    Nvox = shape[0]*shape[1]*shape[2]
    r_mat = r_stack.reshape(10, Nvox)  # (10, N)
    theta_mat = G_inv @ r_mat          # (10, N)
    theta_vol = theta_mat.reshape(10, *shape)
    return theta_vol  # coefficients per voxel

def coeffs_to_A_b_c(theta_voxel):
    """
    Convert a theta vector of length 10 (a_xx,a_yy,a_zz,a_xy,a_xz,a_yz,bx,by,bz,c)
    into A (3x3), b (3,), c scalar.
    """
    a_xx, a_yy, a_zz, a_xy, a_xz, a_yz, bx, by, bz, c = theta_voxel
    A = np.array([[a_xx, a_xy, a_xz],
                  [a_xy, a_yy, a_yz],
                  [a_xz, a_yz, a_zz]])
    b = np.array([bx, by, bz])
    return A, b, c

def estimate_displacement_field(theta1, theta2, eps=1e-6):
    """
    Given theta volumes (10, X, Y, Z) for two volumes, compute displacement field d(x).
    Returns dxyz arrays of shape (3, X, Y, Z).
    """
    shape = theta1.shape[1:]
    Nvox = shape[0]*shape[1]*shape[2]
    # reshape to (10, N)
    t1 = theta1.reshape(10, -1)
    t2 = theta2.reshape(10, -1)
    # build A and b for each voxel
    # indices for theta: 0..9 as: a_xx,a_yy,a_zz,a_xy,a_xz,a_yz,bx,by,bz,c
    A_fields = np.zeros((3,3,Nvox), dtype=float)
    b_fields = np.zeros((3,Nvox), dtype=float)
    # assign
    A_fields[0,0] = t1[0]  # a_xx
    A_fields[1,1] = t1[1]  # a_yy
    A_fields[2,2] = t1[2]  # a_zz
    A_fields[0,1] = A_fields[1,0] = t1[3]  # a_xy
    A_fields[0,2] = A_fields[2,0] = t1[4]  # a_xz
    A_fields[1,2] = A_fields[2,1] = t1[5]  # a_yz
    b_fields[0] = t1[6]
    b_fields[1] = t1[7]
    b_fields[2] = t1[8]
    # but we need A1,A2 and b1,b2 -> easier to compute per-voxel using theta1 and theta2
    def get_A_b(t):
        A = np.zeros((3,3,t.shape[1]), dtype=float)
        A[0,0] = t[0]
        A[1,1] = t[1]
        A[2,2] = t[2]
        A[0,1] = A[1,0] = t[3]
        A[0,2] = A[2,0] = t[4]
        A[1,2] = A[2,1] = t[5]
        b = np.vstack([t[6], t[7], t[8]])
        return A, b

    A1, b1 = get_A_b(t1)
    A2, b2 = get_A_b(t2)
    # compute A_avg and solve 2*A_avg * d = (b1 - b2)  => d = 0.5 * A_avg^{-1} (b1-b2)
    d_fields = np.zeros((3, Nvox), dtype=float)
    for i in range(Nvox):
        Aavg = 0.5*(A1[:,:,i] + A2[:,:,i])
        rhs = (b1[:,i] - b2[:,i])
        # solve with regularization if near-singular
        try:
            d = 0.5 * np.linalg.solve(Aavg + eps*np.eye(3), rhs)
        except np.linalg.LinAlgError:
            # fallback: least-squares
            d = 0.5 * np.linalg.lstsq(Aavg + eps*np.eye(3), rhs, rcond=None)[0]
        d_fields[:,i] = d
    d_vol = d_fields.reshape(3, *shape)
    return d_vol

# Example usage (synthetic test)
if __name__ == "__main__":
    # small synthetic volume
    R = 6
    gx, gy, gz = np.mgrid[-R:R+1, -R:R+1, -R:R+1].astype(float)
    # define a quadratic volume f1
    f1 = (0.3*gx**2 + 0.2*gy**2 + 0.4*gz**2
          + 0.1*gx*gy - 0.05*gx*gz + 0.07*gy*gz
          + 0.8*gx - 0.6*gy + 0.3*gz + 2.0)
    # shift vector (subvoxel)
    true_d = np.array([0.65, -0.4, 0.95])
    # produce shifted volume by sampling (simple trilinear interpolation)
    try:
        from scipy.ndimage import map_coordinates
        coords = np.stack([gx - true_d[0], gy - true_d[1], gz - true_d[2]], axis=0)
        f2 = map_coordinates(f1, coords, order=3, mode='mirror')
    except Exception:
        # fallback: nearest-neighbor shift (coarse)
        s = np.round(true_d).astype(int)
        f2 = np.roll(np.roll(np.roll(f1, -s[0], axis=0), -s[1], axis=1), -s[2], axis=2)

    # compute coefficients via convolutional method
    radius = 5
    sigma = 2.0
    theta1 = fit_quadratic_convolutional_3d(f1, radius=radius, sigma=sigma)
    theta2 = fit_quadratic_convolutional_3d(f2, radius=radius, sigma=sigma)

    # estimate displacement field
    dvol = estimate_displacement_field(theta1, theta2)
    # inspect center voxel estimate
    center = tuple(s//2 for s in f1.shape)
    est_center = dvol[:, center[0], center[1], center[2]]
    print("true shift:", true_d)
    print("estimated (center):", est_center)
