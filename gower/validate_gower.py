"""Checks on a finished Gower matrix, read one block at a time so the full file never loads.

Usage:
    python validate_gower.py ../gower_output
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from gower_kernel import gower_reference


def validate(out_dir, block=1000, n_random=1000, n_pairs_contrib=20000, seed=0, verbose=True):
    D = np.load(os.path.join(out_dir, "gower.npy"), mmap_mode="r")
    x_num = np.load(os.path.join(out_dir, "x_num.npy"))
    x_cat = np.load(os.path.join(out_dir, "x_cat.npy"))
    ranges = pd.read_csv(os.path.join(out_dir, "ranges.csv"))["range"].to_numpy(np.float32)
    with open(os.path.join(out_dir, "config.json")) as f:
        config = json.load(f)
    n = D.shape[0]
    bounds = [(s, min(s + block, n)) for s in range(0, n, block)]

    # symmetry, range, NaN and diagonal, block pair by block pair
    max_asym = 0.0
    n_nan = 0
    d_min, d_max = np.inf, -np.inf
    diag_ok = True
    for ia, (a0, a1) in enumerate(bounds):
        rows = np.asarray(D[a0:a1])
        n_nan += int(np.isnan(rows).sum())
        d_min = min(d_min, float(np.nanmin(rows)))
        d_max = max(d_max, float(np.nanmax(rows)))
        diag_ok &= bool(np.all(rows[np.arange(a1 - a0), np.arange(a0, a1)] == 0))
        for b0, b1 in bounds[ia:]:
            diff = np.abs(rows[:, b0:b1] - np.asarray(D[b0:b1, a0:a1]).T)
            max_asym = max(max_asym, float(np.nanmax(diff)))

    # random cells recomputed from the definition
    rng = np.random.default_rng(seed)
    ii = rng.integers(0, n, n_random)
    jj = rng.integers(0, n, n_random)
    ref = np.array([gower_reference(x_num, x_cat, ranges, i, j)[0] if i != j else 0.0
                    for i, j in zip(ii, jj)])
    max_ref_err = float(np.nanmax(np.abs(np.asarray(D[ii, jj], dtype=np.float64) - ref)))

    # mean contribution of each feature over random pairs; near 0 = range stretched by an outlier
    pi = rng.integers(0, n, n_pairs_contrib)
    pj = rng.integers(0, n, n_pairs_contrib)
    with np.errstate(invalid="ignore"):
        s_num = np.abs(x_num[pi] - x_num[pj]) / ranges
    ca, cb = x_cat[pi], x_cat[pj]
    s_cat = np.where((ca >= 0) & (cb >= 0), (ca != cb).astype(np.float32), np.nan)
    names = config["numeric_features"] + config["categorical_features"]
    contrib = pd.DataFrame({
        "feature": names,
        "type": ["numeric"] * x_num.shape[1] + ["categorical"] * x_cat.shape[1],
        "mean_difference": np.concatenate([np.nanmean(s_num, 0), np.nanmean(s_cat, 0)]),
        "share_of_pairs_present": np.concatenate([(~np.isnan(s_num)).mean(0), (~np.isnan(s_cat)).mean(0)]),
    }).sort_values("mean_difference")
    contrib.to_csv(os.path.join(out_dir, "feature_contribution.csv"), index=False)

    result = {
        "symmetric": max_asym <= 1e-6, "max_asymmetry": max_asym,
        "zero_diagonal": diag_ok,
        "in_0_1": d_min >= 0 and d_max <= 1, "min": d_min, "max": d_max,
        "nan_count": n_nan,
        "random_cells_checked": n_random, "max_error_vs_reference": max_ref_err,
        "matches_reference": max_ref_err <= 1e-5,
    }
    result["passed"] = bool(result["symmetric"] and diag_ok and result["in_0_1"]
                            and n_nan == 0 and result["matches_reference"])
    with open(os.path.join(out_dir, "validation.json"), "w") as f:
        json.dump(result, f, indent=1)
    if verbose:
        print(json.dumps(result, indent=1))
        print("lowest mean per-feature differences (check for outlier-stretched ranges):")
        print(contrib.head(5).to_string(index=False))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--block", type=int, default=1000)
    args = ap.parse_args()
    validate(args.out_dir, block=args.block)
