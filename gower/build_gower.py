"""Build the full N x N Gower distance matrix in row blocks, with checkpoints.

Usage:
    python build_gower.py ../data/cars_clean.csv --out-dir ../gower_output

Each block computes 1,000 rows against all N rows with the numba kernel and
writes them into gower.npy (a float32 memmap on disk). A block is logged in
done_blocks.txt only after it is flushed to disk, so rerunning the same
command after a crash skips finished blocks and redoes the rest.
"""
import argparse
import glob
import json
import os
import subprocess
import time
from datetime import datetime

import numba
import numpy as np
import pandas as pd

from gower_kernel import gower_rows
from gower_prep import DEFAULT_EXCLUDE, fingerprint, load_table, prepare, save_inputs

MATRIX = "gower.npy"
LOG = "done_blocks.txt"
STATS_DIR = "w_stats"


def block_bounds(n, block):
    return [(s, min(s + block, n)) for s in range(0, n, block)]


def read_log(log_path, header):
    """Return the set of finished block indices, or raise if the log is from different inputs."""
    if not os.path.exists(log_path):
        return None
    with open(log_path) as f:
        lines = f.read().splitlines()
    if not lines or lines[0] != header:
        raise RuntimeError(
            f"{log_path} was written for different inputs or block size.\n"
            f"  log:     {lines[0] if lines else '(empty)'}\n  current: {header}\n"
            "Delete the output folder (or use a new --out-dir) to start over.")
    return {int(line.split()[0]) for line in lines[1:] if line.strip()}


def build(prep, out_dir, block=1000, floor=20, hard_floor=15, sample_per_block=2000,
          max_blocks=None, seed=0, verbose=True):
    """Compute every unfinished block. Returns True when all blocks are done.

    max_blocks stops early after that many blocks (used by the resume test).
    """
    x_num, x_cat, inv_range = prep["x_num"], prep["x_cat"], prep["inv_range"]
    n = x_num.shape[0]
    p = x_num.shape[1] + x_cat.shape[1]
    os.makedirs(os.path.join(out_dir, STATS_DIR), exist_ok=True)
    matrix_path = os.path.join(out_dir, MATRIX)
    log_path = os.path.join(out_dir, LOG)

    header = f"# fingerprint {fingerprint(prep)} n {n} block {block} floor {floor} hard_floor {hard_floor}"
    done = read_log(log_path, header)
    if done is None or not os.path.exists(matrix_path):
        if done:
            raise RuntimeError(f"{log_path} lists finished blocks but {matrix_path} is missing.")
        D = np.lib.format.open_memmap(matrix_path, mode="w+", dtype=np.float32, shape=(n, n))
        with open(log_path, "w") as f:
            f.write(header + "\n")
        done = set()
    else:
        D = np.load(matrix_path, mmap_mode="r+")
        if D.shape != (n, n) or D.dtype != np.float32:
            raise RuntimeError(f"{matrix_path} has shape {D.shape}, expected {(n, n)}")

    # small copies of the inputs so validation can recompute cells later
    np.save(os.path.join(out_dir, "x_num.npy"), x_num)
    np.save(os.path.join(out_dir, "x_cat.npy"), x_cat)

    blocks = block_bounds(n, block)
    todo = [b for b in range(len(blocks)) if b not in done]
    if verbose:
        print(f"{n} rows, {p} features, {len(blocks)} blocks of {block} rows, {len(todo)} to do")

    out = np.empty((block, n), dtype=np.float32)
    shared = np.empty((block, n), dtype=np.int16)
    t0 = time.time()
    for count, b in enumerate(todo):
        if max_blocks is not None and count >= max_blocks:
            return False
        a0, a1 = blocks[b]
        m = a1 - a0
        o, s = out[:m], shared[:m]
        gower_rows(x_num[a0:a1], x_cat[a0:a1], x_num, x_cat, inv_range, o, s)

        local = np.arange(m)
        o[local, a0 + local] = 0.0  # zero diagonal

        # data first, then the log line; a crash in between only means redoing this block
        D[a0:a1, :] = o
        D.flush()

        # low-overlap summaries; self-pairs are excluded by setting them to p
        s[local, a0 + local] = p
        hist = np.bincount(s.ravel(), minlength=p + 1)[: p + 1].astype(np.int64)
        hist[p] -= m
        row_min_w = s.min(axis=1)
        row_below = (s < floor).sum(axis=1)
        ii, jj = np.nonzero(s < hard_floor)
        keep = (a0 + ii) < jj  # each unordered pair once
        low_pairs = np.column_stack([a0 + ii[keep], jj[keep], s[ii[keep], jj[keep]]])
        rng = np.random.default_rng(seed + b)
        ri = rng.integers(0, m, sample_per_block)
        rj = rng.integers(0, n, sample_per_block)
        off = (a0 + ri) != rj
        sample = o[ri[off], rj[off]]
        np.savez(os.path.join(out_dir, STATS_DIR, f"block_{b:04d}.npz"), hist=hist,
                 row_min_w=row_min_w, row_below=row_below, low_pairs=low_pairs, sample=sample)

        with open(log_path, "a") as f:
            f.write(f"{b} {a0} {a1} {int(row_min_w.min())} {int(row_below.sum())}\n")

        if verbose:
            el = time.time() - t0
            eta = el / (count + 1) * (len(todo) - count - 1)
            print(f"  block {b + 1}/{len(blocks)} rows {a0}-{a1}  {el:.0f}s elapsed, ~{eta:.0f}s left", flush=True)
    del D
    return True


def summarize(prep, out_dir, floor=20, hard_floor=15):
    """Combine the per-block stats into the low-overlap report and the lookup table."""
    n = prep["x_num"].shape[0]
    p = prep["x_num"].shape[1] + prep["x_cat"].shape[1]
    files = sorted(glob.glob(os.path.join(out_dir, STATS_DIR, "block_*.npz")))
    hist = np.zeros(p + 1, dtype=np.int64)
    row_min_w, row_below, low_pairs, samples = [], [], [], []
    for path in files:
        z = np.load(path)
        hist += z["hist"]
        row_min_w.append(z["row_min_w"])
        row_below.append(z["row_below"])
        low_pairs.append(z["low_pairs"])
        samples.append(z["sample"])
    row_min_w = np.concatenate(row_min_w)
    if len(row_min_w) != n:
        raise RuntimeError("block stats are incomplete; rerun the build")

    # the matrix is symmetric, so every unordered pair was counted twice
    hist //= 2
    pd.DataFrame({"shared_features": np.arange(p + 1), "pairs": hist}).to_csv(
        os.path.join(out_dir, "w_histogram.csv"), index=False)
    pd.DataFrame(np.concatenate(low_pairs), columns=["row_i", "row_j", "shared_features"]).to_csv(
        os.path.join(out_dir, "low_overlap_pairs.csv"), index=False)

    lookup = prep["lookup"].copy()
    lookup["min_shared_features"] = row_min_w
    lookup["partners_below_floor"] = np.concatenate(row_below)
    lookup.to_csv(os.path.join(out_dir, "id_lookup.csv"), index=False)

    total = hist.sum()
    sample = np.concatenate(samples)
    report = {
        "features": p,
        "pairs": int(total),
        "min_shared_features": int(np.nonzero(hist)[0].min()),
        "pairs_below_floor": {"floor": floor, "pairs": int(hist[:floor].sum()),
                              "share": float(hist[:floor].sum() / total)},
        "pairs_below_hard_floor": {"floor": hard_floor, "pairs": int(hist[:hard_floor].sum())},
        "distance_sample_quantiles": dict(zip(
            ["0", "0.01", "0.25", "0.5", "0.75", "0.99", "1"],
            [float(v) for v in np.nanquantile(sample, [0, .01, .25, .5, .75, .99, 1])])),
    }
    with open(os.path.join(out_dir, "low_overlap_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    return report


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
                                       cwd=os.path.dirname(os.path.abspath(__file__))).decode().strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="cleaned CSV or parquet file")
    ap.add_argument("--out-dir", default="gower_output")
    ap.add_argument("--block", type=int, default=1000, help="rows per block")
    ap.add_argument("--floor", type=int, default=20, help="report pairs with fewer shared features than this")
    ap.add_argument("--hard-floor", type=int, default=15, help="list pairs with fewer shared features than this")
    ap.add_argument("--exclude", default=",".join(DEFAULT_EXCLUDE),
                    help="comma-separated identifier/metadata columns kept out of the distance")
    ap.add_argument("--categorical", default="",
                    help="comma-separated numeric-looking columns to treat as categorical")
    ap.add_argument("--sort-by", default="id_trim", help="column that fixes the row order ('' to keep file order)")
    ap.add_argument("--max-missing", type=float, default=None,
                    help="drop features missing in more than this share of rows (default: drop none)")
    ap.add_argument("--threads", type=int, default=None, help="numba threads (default: all cores)")
    ap.add_argument("--skip-validate", action="store_true")
    args = ap.parse_args()

    if args.threads:
        numba.set_num_threads(args.threads)

    df = load_table(args.input)
    sort_by = args.sort_by if args.sort_by and args.sort_by in df.columns else None
    prep = prepare(df, exclude=[c for c in args.exclude.split(",") if c],
                   force_categorical=[c for c in args.categorical.split(",") if c],
                   sort_by=sort_by, max_missing=args.max_missing)
    config = save_inputs(prep, args.out_dir, {
        "input": os.path.abspath(args.input),
        "block": args.block, "floor": args.floor, "hard_floor": args.hard_floor,
        "sort_by": sort_by, "git_commit": git_commit(),
        "created": datetime.now().isoformat(timespec="seconds"),
    })
    print(f"numeric features ({len(prep['num_cols'])}): {', '.join(prep['num_cols'])}")
    print(f"categorical features ({len(prep['cat_cols'])}): {', '.join(prep['cat_cols'])}")
    if prep["dropped"]:
        print(f"dropped: {prep['dropped']}")
    gb = config["n_rows"] ** 2 * 4 / 1e9
    print(f"matrix will be {gb:.1f} GB at {os.path.abspath(args.out_dir)}")

    if not build(prep, args.out_dir, block=args.block, floor=args.floor, hard_floor=args.hard_floor):
        return
    report = summarize(prep, args.out_dir, floor=args.floor, hard_floor=args.hard_floor)
    print(json.dumps(report, indent=1))

    if not args.skip_validate:
        from validate_gower import validate
        validate(args.out_dir, block=args.block)


if __name__ == "__main__":
    main()
