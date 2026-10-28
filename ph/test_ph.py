"""pytest tests for the subsampled PH code."""
import itertools
import json
import os

import numpy as np
import pytest
from ripser import ripser
from scipy.spatial.distance import cdist

from ph_core import (diag_cost, mean_measure, mst_h0_deaths, ot22, subsample_diagrams,
                     to_grid, total_persistence)
from run_subsampled_ph import run


def brute_w22(A, B):
    """W_2^2 between two small diagrams by trying every partial matching."""
    best = np.inf
    k, m = len(A), len(B)
    for r in range(min(k, m) + 1):
        for ai in itertools.combinations(range(k), r):
            for bj in itertools.permutations(range(m), r):
                cost = sum(np.sum((A[i] - B[j]) ** 2) for i, j in zip(ai, bj))
                cost += sum(diag_cost(A[[i]])[0] for i in range(k) if i not in ai)
                cost += sum(diag_cost(B[[j]])[0] for j in range(m) if j not in bj)
                best = min(best, cost)
    return best


def circle_matrix(N=300, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 2 * np.pi, N)
    X = np.column_stack([np.cos(t), np.sin(t)]) + rng.normal(0, noise, (N, 2))
    D = cdist(X, X)
    return D / D.max()


def test_ot22_basic_cases():
    P = np.array([[0.1, 0.5]])
    w = np.ones(1)
    empty = np.zeros((0, 2))
    assert ot22(P, w, empty, np.zeros(0)) == pytest.approx(0.08)
    assert total_persistence(P, w) == pytest.approx(0.08)
    assert ot22(P, w, P, w) == pytest.approx(0.0, abs=1e-12)
    Q = np.array([[0.12, 0.52]])
    assert ot22(P, w, Q, w) == pytest.approx(0.02 ** 2 * 2)


@pytest.mark.parametrize("seed", range(5))
def test_ot22_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    def dgm(k):
        b = rng.uniform(0, 0.5, k)
        return np.column_stack([b, b + rng.uniform(0, 0.3, k)])
    A, B = dgm(3), dgm(2)
    assert ot22(A, np.ones(3), B, np.ones(2)) == pytest.approx(brute_w22(A, B), rel=1e-6)
    assert ot22(A, np.ones(3), B, np.ones(2)) == pytest.approx(ot22(B, np.ones(2), A, np.ones(3)), rel=1e-6)


def test_mean_measure_and_grid_keep_mass():
    rng = np.random.default_rng(0)
    dgms = [rng.uniform(0, 1, (k, 2)) for k in (3, 5, 0, 4)]
    P, w = mean_measure(dgms)
    assert len(P) == 12 and w.sum() == pytest.approx(3.0)
    G, gw = to_grid(P, w, 0.1)
    assert gw.sum() == pytest.approx(3.0)
    assert np.all(np.abs(G - P[np.argmin(cdist(G, P), axis=1)]) <= 0.05 + 1e-12)


def test_mst_matches_ripser_h0():
    D = circle_matrix(200)
    exact = mst_h0_deaths(D)
    rip = ripser(D, distance_matrix=True, maxdim=0)["dgms"][0][:, 1]
    rip = np.sort(rip[np.isfinite(rip)])
    np.testing.assert_allclose(exact, rip, rtol=1e-6)


def test_mst_keeps_zero_edges():
    D = circle_matrix(50)
    D = np.vstack([D, D[:1]])
    D = np.hstack([D, D[:, :1]])
    D[-1, -1] = 0.0  # vehicle 50 is a copy of vehicle 0
    deaths = mst_h0_deaths(D)
    assert len(deaths) == 50 and deaths[0] == 0.0


def test_subsample_diagrams_drops_infinite_bar():
    D = circle_matrix(100)
    h0, h1 = subsample_diagrams(D, np.arange(0, 100, 2))
    assert np.all(np.isfinite(h0)) and len(h0) == 49
    assert len(h1) >= 1 and (h1[:, 1] - h1[:, 0]).max() > 0.3  # the circle


def test_run_finds_circle_and_resumes(tmp_path):
    D = circle_matrix(400).astype(np.float32)
    path = str(tmp_path / "gower.npy")
    np.save(path, D)
    out = str(tmp_path / "run")
    s1 = run(path, 80, out, seed=3, b_min=4, patience=2, tau=0.5, batch=3, jobs=1, verbose=False)
    assert s1["stop_B_per_chain"] >= 4 and s1["diagrams_total"] == 2 * s1["diagrams_per_chain"]
    z = np.load(os.path.join(out, "mean_measure.npz"))
    P, w = z["h1_points"], z["h1_weights"]
    big = (P[:, 1] - P[:, 0]) > 0.3
    assert w[big].sum() == pytest.approx(1.0, abs=0.15)  # about one big loop per subsample
    # rerun with the same settings: every subsample loads from disk, same answer
    s2 = run(path, 80, out, seed=3, b_min=4, patience=2, tau=0.5, batch=3, jobs=1, verbose=False)
    z2 = np.load(os.path.join(out, "mean_measure.npz"))
    assert s2["diagrams_total"] == s1["diagrams_total"]
    np.testing.assert_array_equal(z2["h1_points"], z["h1_points"])
    # a different batch size draws the same subsamples (seeded by chain and index)
    out3 = str(tmp_path / "run3")
    run(path, 80, out3, seed=3, b_min=4, patience=2, tau=0.5, batch=1, jobs=1, verbose=False)
    a = np.load(os.path.join(out, "diagrams", "chain1_0002.npz"))
    b = np.load(os.path.join(out3, "diagrams", "chain1_0002.npz"))
    np.testing.assert_array_equal(a["idx"], b["idx"])
    np.testing.assert_array_equal(a["h1"], b["h1"])
    with pytest.raises(RuntimeError):
        run(path, 81, out, seed=3, jobs=1, verbose=False)


def test_plots_run(tmp_path):
    import plot_mean_measure as pm
    D = circle_matrix(300).astype(np.float32)
    path = str(tmp_path / "gower.npy")
    np.save(path, D)
    out = str(tmp_path / "run")
    run(path, 60, out, b_min=3, patience=1, tau=1.0, jobs=1, verbose=False)
    z, cfg = pm.load_run(out)
    pm.h1_heatmap(z, str(tmp_path / "h1.png"))
    pm.h0_hist(z, str(tmp_path / "h0.png"), exact=mst_h0_deaths(D.astype(np.float64)))
    pm.convergence(out, cfg, str(tmp_path / "conv.png"))
    pm.compare_h1([(z, cfg), (z, cfg)], str(tmp_path / "cmp.png"))
    assert all(os.path.getsize(tmp_path / f) > 0 for f in ("h1.png", "h0.png", "conv.png", "cmp.png"))
