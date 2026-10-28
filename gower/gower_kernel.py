"""Gower distance kernels.

gower_rows is the fast numba version used by the build.
gower_reference is a slow, plain Python version used only by the tests.
"""
import numpy as np
import numba


@numba.njit(parallel=True, cache=True)
def gower_rows(xn_a, xc_a, xn, xc, inv_range, out, shared):
    """Gower distance between every row of block A and every row of the full data.

    xn_a, xn : float32 numeric features, NaN = missing
    xc_a, xc : int16 categorical codes, -1 = missing
    inv_range: float32 1 / range of each numeric feature (ranges from the full data)
    out      : float32 (len(A), N) distances, filled in place
    shared   : int16 (len(A), N) number of features both rows have, filled in place

    A pair with no shared features gets NaN so the validation step catches it.
    """
    n_a = xn_a.shape[0]
    n = xn.shape[0]
    p_num = xn.shape[1]
    p_cat = xc.shape[1]
    for i in numba.prange(n_a):
        for j in range(n):
            s = 0.0  # float64 accumulator
            w = 0
            # numeric part: |x_i - x_j| / range, skipped where either value is NaN
            for k in range(p_num):
                a = xn_a[i, k]
                b = xn[j, k]
                if a == a and b == b:
                    s += abs(a - b) * inv_range[k]
                    w += 1
            # categorical part: 0 if same category, 1 if different, skipped where either is -1
            for k in range(p_cat):
                a = xc_a[i, k]
                b = xc[j, k]
                if a >= 0 and b >= 0:
                    w += 1
                    if a != b:
                        s += 1.0
            if w > 0:
                out[i, j] = s / w
            else:
                out[i, j] = np.nan
            shared[i, j] = w


def gower_reference(xn, xc, ranges, i, j):
    """Gower distance for one pair, written straight from the definition. Returns (d, W)."""
    s = 0.0
    w = 0
    for k in range(xn.shape[1]):
        a, b = float(xn[i, k]), float(xn[j, k])
        if not (np.isnan(a) or np.isnan(b)):
            s += abs(a - b) / float(ranges[k])
            w += 1
    for k in range(xc.shape[1]):
        a, b = int(xc[i, k]), int(xc[j, k])
        if a != -1 and b != -1:
            s += float(a != b)
            w += 1
    return (s / w if w > 0 else float("nan")), w
