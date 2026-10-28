"""Subsampled persistent homology (Cao & Monod, arXiv:2204.09155) on the Gower matrix.

Usage:
    python run_subsampled_ph.py ../gower_output/gower.npy --n 2000 --out-dir ../ph_output/n2000

Two independent chains of subsamples are drawn. Each subsample is n distinct
vehicles; its n x n block is sliced from the memmap and passed to Ripser
(Vietoris-Rips, H0 and H1). After each new pair of diagrams the two chains'
H1 mean measures are compared:

    r_B = OT_2^2(mean_1, mean_2) / (0.5 * (T(mean_1) + T(mean_2)))

where T is total persistence (OT_2^2 to the empty measure). The run stops
once both chains have at least --b-min diagrams and r_B < --tau for
--patience values of B in a row. The paper's consecutive-change rule
(Q* < 1e-3) is computed on chain 1 and logged for comparison only.

The final mean persistence measure pools all diagrams from both chains.
Each subsample's diagrams are saved as they finish, so rerunning the same
command after a crash resumes. Subsample i of chain c always uses the random
generator seeded with (seed, c, i), so results do not depend on --jobs or
--batch.
"""
import argparse
import json
import os
import platform
import time
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from ph_core import (draw_indices, mean_measure, open_matrix, ot22, subsample_diagrams,
                     to_grid, total_persistence)

PAPER_Q_THRESHOLD = 1e-3


def subsample_path(out_dir, c, i):
    return os.path.join(out_dir, "diagrams", f"chain{c}_{i:04d}.npz")


def compute_subsample(matrix_path, n, seed, c, i, out_dir):
    """Draw subsample i of chain c, run Ripser, save, and return the diagrams."""
    path = subsample_path(out_dir, c, i)
    if os.path.exists(path):
        z = np.load(path)
        return z["h0"], z["h1"], float(z["seconds"])
    D = open_matrix(matrix_path)
    rng = np.random.default_rng([seed, c, i])
    idx = draw_indices(rng, D.shape[0], n)
    t = time.time()
    h0, h1 = subsample_diagrams(D, idx, maxdim=1)
    sec = time.time() - t
    tmp = path[:-4] + ".tmp.npz"
    np.savez(tmp, idx=idx, h0=h0, h1=h1, seconds=sec)
    os.replace(tmp, path)
    return h0, h1, sec


def relative_gap(m1, m2):
    T = 0.5 * (total_persistence(*m1) + total_persistence(*m2))
    return ot22(*m1, *m2) / T if T > 0 else 0.0, T


def run(matrix_path, n, out_dir, seed=0, tau=0.01, b_min=20, patience=5, b_max=200,
        batch=None, grid=0.002, jobs=-1, verbose=True):
    os.makedirs(os.path.join(out_dir, "diagrams"), exist_ok=True)
    D = open_matrix(matrix_path)
    N = D.shape[0]
    if not 2 <= n <= N:
        raise ValueError(f"n = {n} must be between 2 and N = {N}")
    del D

    settings = {"matrix": os.path.abspath(matrix_path), "N": int(N), "n": n, "seed": seed,
                "tau": tau, "b_min": b_min, "patience": patience, "b_max": b_max, "grid": grid}
    cfg_path = os.path.join(out_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            old = json.load(f)["settings"]
        if any(old.get(k) != settings[k] for k in ("N", "n", "seed")):
            raise RuntimeError(f"{out_dir} holds a run with different N, n or seed; use a new --out-dir")

    if batch is None:
        batch = max(1, (os.cpu_count() or 2) // 2)
    pool = Parallel(n_jobs=jobs)
    chains = [[], []]          # lists of (h0, h1)
    seconds = []
    hist = []
    calm = 0
    stop_B = None
    prev_mean1, prev_err = None, None
    t0 = time.time()

    while stop_B is None and len(chains[0]) < b_max:
        start = len(chains[0])
        stop = min(start + batch, b_max)
        jobs_ = [(c, i) for i in range(start, stop) for c in (0, 1)]
        res = pool(delayed(compute_subsample)(matrix_path, n, seed, c, i, out_dir) for c, i in jobs_)
        for (c, i), (h0, h1, sec) in zip(jobs_, res):
            chains[c].append((h0, h1))
            seconds.append(sec)

        # evaluate the rules at every B in this batch, as if diagrams arrived one pair at a time
        for B in range(start + 1, stop + 1):
            g1 = [to_grid(*mean_measure([d[k] for d in chains[0][:B]]), grid) for k in (0, 1)]
            g2 = [to_grid(*mean_measure([d[k] for d in chains[1][:B]]), grid) for k in (0, 1)]
            r1, T1 = relative_gap(g1[1], g2[1])
            r0, T0 = relative_gap(g1[0], g2[0])

            # paper's rule on chain 1: Err_B = OT(mean_{B}, mean_{B-1}), Q* = |Err_B / Err_{B-1} - 1|
            err = ot22(*g1[1], *prev_mean1) if prev_mean1 is not None else np.nan
            q = abs(err / prev_err - 1) if prev_err not in (None, 0) and np.isfinite(prev_err) else np.nan
            prev_mean1, prev_err = g1[1], err

            calm = calm + 1 if r1 < tau else 0
            hist.append({"B": B, "r_h1": r1, "T_h1": T1, "r_h0": r0, "T_h0": T0,
                         "paper_err_h1": err, "paper_q_h1": q, "calm": calm})
            if stop_B is None and B >= b_min and calm >= patience:
                stop_B = B
        if verbose:
            last = hist[-1]
            print(f"  B = {last['B']:3d} per chain  r_H1 = {last['r_h1']:.4f}  r_H0 = {last['r_h0']:.4f}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)

    B_done = len(chains[0])
    history = pd.DataFrame(hist)
    history.to_csv(os.path.join(out_dir, "history.csv"), index=False)

    # pooled mean persistence measure over both chains, ungridded
    all_dgms = chains[0] + chains[1]
    h0P, h0w = mean_measure([d[0] for d in all_dgms])
    h1P, h1w = mean_measure([d[1] for d in all_dgms])
    np.savez(os.path.join(out_dir, "mean_measure.npz"), h0_points=h0P, h0_weights=h0w,
             h1_points=h1P, h1_weights=h1w, n=n, N=N, diagrams=len(all_dgms))

    q = history["paper_q_h1"].to_numpy()
    hit = np.nonzero(q < PAPER_Q_THRESHOLD)[0]
    summary = {
        "stopped": stop_B is not None,
        "stop_B_per_chain": stop_B,
        "diagrams_per_chain": B_done,
        "diagrams_total": 2 * B_done,
        "final_r_h1": float(history["r_h1"].iloc[-1]),
        "final_r_h0": float(history["r_h0"].iloc[-1]),
        "paper_rule_first_B": int(history["B"].iloc[hit[0]]) if len(hit) else None,
        "ripser_seconds_mean": float(np.mean(seconds)),
        "ripser_seconds_max": float(np.max(seconds)),
        "wall_seconds": time.time() - t0,
        "h1_points_per_subsample": float(h1w.sum()),
        "h0_points_per_subsample": float(h0w.sum()),
    }
    import ripser
    import ot
    with open(cfg_path, "w") as f:
        json.dump({"settings": settings, "summary": summary,
                   "versions": {"python": platform.python_version(), "numpy": np.__version__,
                                "ripser": ripser.__version__, "pot": ot.__version__},
                   "created": datetime.now().isoformat(timespec="seconds")}, f, indent=1)
    if verbose:
        print(json.dumps(summary, indent=1))
        if stop_B is None:
            print(f"WARNING: stopping rule not met by B_max = {b_max}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("matrix", help="gower.npy from build_gower.py")
    ap.add_argument("--n", type=int, required=True, help="vehicles per subsample (pick with timing_test.py)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.01, help="threshold for r_B")
    ap.add_argument("--b-min", type=int, default=20, help="minimum diagrams per chain")
    ap.add_argument("--patience", type=int, default=5, help="consecutive B with r_B < tau")
    ap.add_argument("--b-max", type=int, default=200, help="give up after this many per chain")
    ap.add_argument("--batch", type=int, default=None, help="subsamples per chain per parallel round")
    ap.add_argument("--grid", type=float, default=0.002, help="cell size for the OT comparisons")
    ap.add_argument("--jobs", type=int, default=-1, help="parallel Ripser processes (-1 = all cores)")
    args = ap.parse_args()
    run(args.matrix, args.n, args.out_dir, seed=args.seed, tau=args.tau, b_min=args.b_min,
        patience=args.patience, b_max=args.b_max, batch=args.batch, grid=args.grid, jobs=args.jobs)


if __name__ == "__main__":
    main()
