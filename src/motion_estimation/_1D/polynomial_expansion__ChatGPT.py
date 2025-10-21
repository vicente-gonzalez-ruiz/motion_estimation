import numpy as np

def gaussian_kernel_1d(radius, sigma):
    xs = np.arange(-radius, radius+1, dtype=float)
    w = np.exp(-0.5 * (xs/sigma)**2)
    w /= w.sum()
    return xs, w

def fit_quadratic_wls(signal, center_idx, radius, sigma):
    """
    Weighted least squares quadratic fit in a window around center_idx.
    Returns (a,b,c) for p(x)=a x^2 + b x + c where x is offset from center.
    """
    N = len(signal)
    left = max(0, center_idx - radius)
    right = min(N, center_idx + radius + 1)
    xs = np.arange(left, right) - center_idx
    ys = signal[left:right].astype(float)
    _, w_full = gaussian_kernel_1d(radius, sigma)
    # select corresponding part of kernel (because near boundaries kernel truncated)
    k_center = radius  # index of zero offset in kernel
    k_left = k_center - (center_idx - left)
    k_right = k_center + (right - center_idx)
    w = w_full[int(k_left):int(k_right)]
    # Design matrix: columns x^2, x, 1
    X = np.vstack([xs**2, xs, np.ones_like(xs)]).T
    W = np.diag(w)
    # normal equations: (X^T W X) p = X^T W y
    XT_W = X.T * w  # broadcasting: each row scaled by weights
    G = XT_W @ X
    rhs = XT_W @ ys
    # Solve with small regularization for stability:
    eps = 1e-10
    try:
        coeffs = np.linalg.solve(G + eps*np.eye(3), rhs)
    except np.linalg.LinAlgError:
        coeffs = np.linalg.lstsq(G + eps*np.eye(3), rhs, rcond=None)[0]
    a, b, c = coeffs[0], coeffs[1], coeffs[2]
    return a, b, c

def fit_quadratic_convolution(signal, radius, sigma):
    """
    Efficient fit using precomputed kernel moments and convolutions.
    Kernel assumed symmetric about zero (Gaussian), so moment m1=m3=0.
    Returns arrays a, b, c (same length as signal). Edges handled by 'same' conv.
    """
    xs, w = gaussian_kernel_1d(radius, sigma)  # xs centered at 0
    # precompute kernel-weighted basis functions:
    wk0 = w                 # w(x)
    wk1 = w * xs            # w(x) * x
    wk2 = w * (xs**2)       # w(x) * x^2

    # convolve (use np.convolve with mode='same')
    y = signal.astype(float)
    t0 = np.convolve(y, wk0[::-1], mode='same')   # sum w * y
    t1 = np.convolve(y, wk1[::-1], mode='same')   # sum w*x*y
    t2 = np.convolve(y, wk2[::-1], mode='same')   # sum w*x^2*y

    # kernel moments (same for all centers because kernel symmetric and fixed)
    m0 = wk0.sum()
    m2 = (wk2).sum()    # = sum w*x^2
    m4 = np.sum(w * (xs**4))

    # Solve normal equations exploiting symmetry (m1=m3=0)
    # G = [[m4, 0, m2],
    #      [0 , m2, 0 ],
    #      [m2, 0, m0]]
    # second row gives: m2 * b = t1 => b = t1 / m2
    small = 1e-12
    b = t1 / (m2 + small)

    # the remaining 2x2 system for [a, c]:
    # [m4 m2] [a] = [t2]
    # [m2 m0] [c]   [t0]
    det = m4*m0 - m2*m2
    if abs(det) < 1e-15:
        # degenerate kernel (shouldn't happen for Gaussian); fall back to zeros
        a = np.zeros_like(y)
        c = t0 / (m0 + small)
    else:
        a = ( t2 * m0 - t0 * m2 ) / det
        c = (-t2 * m2 + t0 * m4 ) / det

    return a, b, c

def estimate_shift_from_coeffs(a1,b1,c1, a2,b2,c2, eps=1e-8):
    """
    Given quadratic coefficients for two patches (p1 and p2),
    estimate translation d such that p2(x) ~ p1(x - d).
    Returns d (float) or None if degenerate.
    Use average curvature for stability.
    """
    a_avg = 0.5*(a1 + a2)
    denom = 2.0 * a_avg
    if abs(denom) < eps:
        # curvature too small -> ambiguous shift for near-linear patches
        return None
    d = (b1 - b2) / denom
    return d

# small usage example
if __name__ == "__main__":
    # create synthetic signal that is a shifted quadratic patch
    x = np.linspace(-5, 5, 201)
    patch = 0.5*x**2 + 0.8*x + 1.2
    true_shift = 1.7
    # create two signals: s1 is patch, s2 is patch shifted by true_shift (non-integer)
    from scipy.interpolate import interp1d
    f = interp1d(x, patch, kind='cubic', fill_value='extrapolate')
    s1 = patch
    s2 = f(x - true_shift)

    # fit coefficients (convolution approach)
    radius = 7
    sigma = 3.0
    a1,b1,c1 = fit_quadratic_convolution(s1, radius, sigma)
    a2,b2,c2 = fit_quadratic_convolution(s2, radius, sigma)

    # pick center index for local estimate
    idx = len(x)//2
    d_est = estimate_shift_from_coeffs(a1[idx], b1[idx], c1[idx], a2[idx], b2[idx], c2[idx])
    print("true_shift:", true_shift, "estimated:", d_est)
