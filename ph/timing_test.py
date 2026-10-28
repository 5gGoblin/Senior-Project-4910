"""Time one Ripser call (H0 and H1) at several subsample sizes and suggest n.

Usage:
    python timing_test.py ../gower_output/gower.npy --sizes 500,1000,2000,3000 --reps 2 \
        --budget-min 60 --diagrams 60

Each size is run in a fresh process so peak memory is measured per size.
The suggested n is the largest tested size for which
    diagrams * mean_seconds / parallel_jobs <= budget,
with parallel_jobs = min(cores, memory_GB * 0.7 / peak_GB). The default of 60
diagrams assumes the two-chain rule stops around 30 per chain; it is a guess.
"""
import argparse
import json
import multiprocessing as mp
import os
import resource
import sys
import time

import numpy as np


def _time_one(matrix_path, n, seed, q):
    from ph_core import draw_indices, open_matrix, subsample_diagrams
    D = open_matrix(matrix_path)
    idx = draw_indices(np.random.default_rng(seed), D.shape[0], n)
    t = time.time()
    h0, h1 = subsample_diagrams(D, idx)
    sec = time.time() - t
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = peak / 2**20 if sys.platform == "darwin" else peak / 1024  # bytes on macOS, KB on Linux
    q.put({"n": n, "seed": seed, "seconds": sec, "peak_mb": peak_mb, "h0": len(h0), "h1": len(h1)})


def total_memory_gb():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except (ValueError, OSError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("matrix")
    ap.add_argument("--sizes", default="500,1000,2000,3000")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--budget-min", type=float, default=60, help="wall-clock minutes for the full run")
    ap.add_argument("--diagrams", type=int, default=60, help="expected total diagrams (both chains)")
    ap.add_argument("--out", default="timing.json")
    args = ap.parse_args()

    ctx = mp.get_context("spawn")
    rows = []
    for n in [int(s) for s in args.sizes.split(",")]:
        for r in range(args.reps):
            q = ctx.Queue()
            p = ctx.Process(target=_time_one, args=(args.matrix, n, 1000 + r, q))
            p.start()
            row = q.get()
            p.join()
            rows.append(row)
            print(f"n = {n:5d}  {row['seconds']:7.1f} s  peak {row['peak_mb']:6.0f} MB  "
                  f"H0 {row['h0']}  H1 {row['h1']}", flush=True)

    cores = os.cpu_count() or 1
    mem = total_memory_gb()
    print(f"\n{cores} cores, {mem:.0f} GB memory" if mem else f"\n{cores} cores")
    best = None
    table = []
    for n in sorted({r["n"] for r in rows}):
        rs = [r for r in rows if r["n"] == n]
        sec = float(np.mean([r["seconds"] for r in rs]))
        peak = max(r["peak_mb"] for r in rs) / 1000
        jobs = cores if mem is None else max(1, min(cores, int(mem * 0.7 / peak)))
        minutes = args.diagrams * sec / jobs / 60
        table.append({"n": n, "mean_seconds": sec, "peak_gb": peak, "parallel_jobs": jobs,
                      "est_minutes": minutes})
        print(f"n = {n:5d}: ~{minutes:6.1f} min for {args.diagrams} diagrams on {jobs} parallel jobs")
        if minutes <= args.budget_min:
            best = n
    print(f"\nsuggested n (largest tested within {args.budget_min:g} min): {best}")
    with open(args.out, "w") as f:
        json.dump({"runs": rows, "estimates": table, "suggested_n": best,
                   "budget_min": args.budget_min, "diagrams": args.diagrams}, f, indent=1)


if __name__ == "__main__":
    main()
