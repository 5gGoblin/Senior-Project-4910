"""Building blocks for subsampled persistent homology on the Gower matrix.

A persistence measure here is a pair (P, w): P is a (k, 2) array of
(birth, death) points and w is a length-k array of weights. A single diagram
has weight 1 on every point. The mean persistence measure of B diagrams pools
all their points with weight 1/B (Cao & Monod, arXiv:2204.09155, Sec. 3).
"""
import numba
import numpy as np
import ot
from ripser import ripser


def open_matrix(path):
    """Open gower.npy as a read-only memmap."""
    D = np.load(path, mmap_mode="r")
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"{path} is not a square matrix: {D.shape}")
    return D


def draw_indices(rng, N, n):
    """n distinct row indices, sorted so the slice reads the memmap in order."""
    return np.sort(rng.choice(N, size=n, replace=False))


def subsample_diagrams(D, idx, maxdim=1):
    """Vietoris-Rips H0..H<maxdim> of the vehicles in idx.

    Returns a list of (k, 2) arrays. The one infinite H0 bar is dropped.
    Ripser leaves out zero-length bars (identical vehicles), so H0 can have
    fewer than n - 1 points.
    """
    sub = np.asarray(D[np.ix_(idx, idx)], dtype=np.float64)
    dgms = ripser(sub, distance_matrix=True, maxdim=maxdim)["dgms"]
    out = []
    for k, dg in enumerate(dgms):
        if k == 0:
            dg = dg[np.isfinite(dg[:, 1])]
        elif not np.all(np.isfinite(dg)):
            # cannot happen without a thresh; fail loudly rather than drop points
            raise RuntimeError(f"H{k} has infinite deaths")
        out.append(np.ascontiguousarray(dg, dtype=np.float64))
    return out


def mean_measure(dgms):
    """Pool B diagrams into a measure with weight 1/B per point."""
    B = len(dgms)
    if B == 0:
        raise ValueError("no diagrams")
    P = np.vstack([np.asarray(d, dtype=np.float64).reshape(-1, 2) for d in dgms])
    return P, np.full(len(P), 1.0 / B)


def to_grid(P, w, h):
    """Bin a measure onto square cells of side h; each cell's mass sits at its centre.

    Moving every point by at most h / sqrt(2) changes OT_2 by at most
    h / sqrt(2) times the square root of the total mass (triangle inequality),
    so the same h must be used for every comparison and reported.
    """
    if len(P) == 0:
        return np.zeros((0, 2)), np.zeros(0)
    cells = np.floor(P / h).astype(np.int64)
    uniq, inv = np.unique(cells, axis=0, return_inverse=True)
    return (uniq + 0.5) * h, np.bincount(inv.ravel(), weights=w, minlength=len(uniq))


def diag_cost(P):
    """Squared Euclidean distance from (b, d) to the diagonal: (d - b)^2 / 2."""
    return (P[:, 1] - P[:, 0]) ** 2 / 2


def total_persistence(P, w):
    """OT_2^2 between a measure and the empty measure."""
    return float(np.sum(w * diag_cost(P)))


def ot22(P, wp, Q, wq):
    """OT_2^2 between two persistence measures, ground cost squared Euclidean.

    Partial transport with the diagonal is turned into balanced transport by
    giving each side one extra node for the diagonal that can absorb all of
    the other side's mass (diagonal-to-diagonal costs 0).
    """
    k, m = len(P), len(Q)
    if k == 0 and m == 0:
        return 0.0
    if k == 0:
        return total_persistence(Q, wq)
    if m == 0:
        return total_persistence(P, wp)
    M = np.zeros((k + 1, m + 1))
    M[:k, :m] = ot.dist(P, Q)  # squared Euclidean by default
    M[:k, m] = diag_cost(P)
    M[k, :m] = diag_cost(Q)
    a = np.append(wp, wq.sum())
    b = np.append(wq, wp.sum())
    # both sides then carry wp.sum() + wq.sum(); normalise to avoid rounding complaints
    s = a.sum()
    return float(ot.emd2(a / s, b / s, M, numItermax=10_000_000)) * s


@numba.njit(cache=True)
def _prim(D):
    N = D.shape[0]
    in_tree = np.zeros(N, dtype=np.bool_)
    best = np.full(N, np.inf)
    out = np.empty(N - 1)
    v = 0
    in_tree[0] = True
    for k in range(N - 1):
        u, bu = -1, np.inf
        for j in range(N):
            if not in_tree[j]:
                dj = D[v, j]
                if dj < best[j]:
                    best[j] = dj
                if best[j] < bu:
                    u, bu = j, best[j]
        if u < 0:
            raise ValueError("graph is disconnected (NaN distances?)")
        out[k] = bu
        in_tree[u] = True
        v = u
    return out


def mst_h0_deaths(D):
    """Exact H0 death times of the Rips filtration: the N - 1 MST edge lengths, sorted.

    Prim's algorithm on the dense matrix, O(N^2), reads each row once. Zero
    lengths (identical vehicles) are kept; Ripser drops them.
    Do not use scipy's minimum_spanning_tree here: it treats 0 as "no edge".
    """
    return np.sort(_prim(D))
