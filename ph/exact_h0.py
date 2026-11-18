"""Exact H0 of the Rips filtration on all N vehicles, from the minimum spanning tree.

Usage:
    python exact_h0.py ../gower_output/gower.npy --out ../ph_output/exact_h0.npy \
        --edges ../ph_output/mst_edges.npz

The H0 death times of a Vietoris-Rips filtration are the N - 1 edge lengths
of a minimum spanning tree (the single-linkage merge heights). This needs no
subsampling. It reads the whole matrix once, so the first run is limited by
disk speed.

With --edges it also saves the MST edges (row indices u, v and length), which
label_auc.py uses to ask whether H0 merges between vehicles of different
makes or body types happen later than merges within one.
"""
import argparse
import time

import numpy as np

from ph_core import mst_h0_edges, open_matrix


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("matrix")
    ap.add_argument("--out", default="exact_h0.npy")
    ap.add_argument("--edges", default=None, help="also save MST edges (u, v, length) to this .npz")
    args = ap.parse_args()
    D = open_matrix(args.matrix)
    t = time.time()
    u, v, deaths = mst_h0_edges(np.asarray(D))
    np.save(args.out, deaths)
    if args.edges:
        np.savez(args.edges, u=u, v=v, length=deaths)
    q = np.quantile(deaths, [0, 0.1, 0.5, 0.9, 0.99, 1])
    print(f"{len(deaths)} death times in {time.time() - t:.0f} s; zero-length: {(deaths == 0).sum()}")
    print("quantiles 0/10/50/90/99/100%: " + " ".join(f"{v:.4f}" for v in q))


if __name__ == "__main__":
    main()
