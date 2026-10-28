"""Exact H0 of the Rips filtration on all N vehicles, from the minimum spanning tree.

Usage:
    python exact_h0.py ../gower_output/gower.npy --out ../ph_output/exact_h0.npy

The H0 death times of a Vietoris-Rips filtration are the N - 1 edge lengths
of a minimum spanning tree (the single-linkage merge heights). This needs no
subsampling. It reads the whole matrix once, so the first run is limited by
disk speed.
"""
import argparse
import time

import numpy as np

from ph_core import mst_h0_deaths, open_matrix


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("matrix")
    ap.add_argument("--out", default="exact_h0.npy")
    args = ap.parse_args()
    D = open_matrix(args.matrix)
    t = time.time()
    deaths = mst_h0_deaths(np.asarray(D))
    np.save(args.out, deaths)
    q = np.quantile(deaths, [0, 0.1, 0.5, 0.9, 0.99, 1])
    print(f"{len(deaths)} death times in {time.time() - t:.0f} s; zero-length: {(deaths == 0).sum()}")
    print("quantiles 0/10/50/90/99/100%: " + " ".join(f"{v:.4f}" for v in q))


if __name__ == "__main__":
    main()
