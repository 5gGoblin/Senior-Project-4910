"""Figures for a subsampled PH run.

Usage:
    python plot_mean_measure.py ../ph_output/n2000 --exact-h0 ../ph_output/exact_h0.npy
    python plot_mean_measure.py ../ph_output/n2000 --compare ../ph_output/n1000 ../ph_output/n4000

Writes into the run folder:
    mean_h1_heatmap.png   H1 mean persistence measure, (birth, persistence) coordinates
    mean_h0_hist.png      H0 death times: mean count per subsample, and (with --exact-h0)
                          the share of bars per bin next to the exact full-data H0
    convergence.png       r_B for H1 and H0 against B
    compare_h1.png        (with --compare) H1 persistence tail for each run
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed order
INK, MUTED, GRID = "#1f1f1e", "#6b6b67", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "font.size": 10, "axes.titlesize": 11,
    "legend.frameon": False, "lines.linewidth": 2,
})


def load_run(run_dir):
    z = np.load(os.path.join(run_dir, "mean_measure.npz"))
    with open(os.path.join(run_dir, "config.json")) as f:
        cfg = json.load(f)
    return z, cfg


def label(z):
    return f"n = {int(z['n']):,}, {int(z['diagrams'])} subsamples, N = {int(z['N']):,}"


def h1_heatmap(z, out, bin_b=0.004, bin_p=0.002):
    P, w = z["h1_points"], z["h1_weights"]
    birth, pers = P[:, 0], P[:, 1] - P[:, 0]
    bmax = np.quantile(birth, 0.999) + bin_b
    pmax = pers.max() + bin_p
    H, xe, ye = np.histogram2d(birth, pers, bins=[np.arange(0, bmax + bin_b, bin_b),
                                                  np.arange(0, pmax + bin_p, bin_p)], weights=w)
    H = np.ma.masked_less_equal(H, 0)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.grid(False)
    m = ax.pcolormesh(xe, ye, H.T, cmap=SEQ, norm=LogNorm(vmin=H.min(), vmax=H.max()), shading="flat")
    cb = fig.colorbar(m, ax=ax, pad=0.02)
    cb.set_label("mean H1 points per subsample in cell")
    cb.outline.set_visible(False)
    ax.set_xlabel("birth (Gower distance)")
    ax.set_ylabel("persistence = death - birth")
    ax.set_title(f"H1 mean persistence measure\n{label(z)}; cells {bin_b} x {bin_p}", loc="left")
    off = birth > bmax
    if off.any():
        ax.text(0.99, 0.98, f"{w[off].sum():.2f} points/subsample with birth > {bmax:.2f} not shown",
                transform=ax.transAxes, ha="right", va="top", color=MUTED, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def h0_hist(z, out, exact=None, bin_d=0.002):
    d, w = z["h0_points"][:, 1], z["h0_weights"]
    hi = max(d.max(), exact.max() if exact is not None else 0) + bin_d
    edges = np.arange(0, hi + bin_d, bin_d)
    ncols = 2 if exact is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4), squeeze=False)
    ax = axes[0, 0]
    cnt, _ = np.histogram(d, bins=edges, weights=w)
    ax.bar(edges[:-1], cnt, width=bin_d * 0.9, align="edge", color=CAT[0])
    ax.set_xlabel("death (Gower distance)")
    ax.set_ylabel("mean H0 bars per subsample in bin")
    ax.set_title(f"H0 mean persistence measure\n{label(z)}", loc="left")
    if exact is not None:
        n_zero = int((exact == 0).sum())
        exact = exact[exact > 0]
        ax = axes[0, 1]
        sub_share, _ = np.histogram(d, bins=edges, weights=w / w.sum())
        ex_share, _ = np.histogram(exact, bins=edges)
        ex_share = ex_share / len(exact)
        x = edges[:-1] + bin_d / 2
        ax.plot(x, sub_share, color=CAT[0], label=f"subsamples, n = {int(z['n']):,}")
        ax.plot(x, ex_share, color=CAT[1], label=f"exact (MST), N = {int(z['N']):,}")
        for share, c, txt in ((sub_share, CAT[0], f"median {np.median(d):.3f}"),
                              (ex_share, CAT[1], f"median {np.median(exact):.3f}")):
            k = int(np.argmax(share))
            ax.annotate(txt, (x[k], share[k]), xytext=(6, 4), textcoords="offset points",
                        color=INK, fontsize=8)
        ax.set_xlabel("death (Gower distance)")
        ax.set_ylabel("share of H0 bars in bin")
        ax.set_title(f"Subsample H0 vs exact full-data H0\n{n_zero} zero-length exact bars left out,"
                     " as Ripser does", loc="left")
        ax.legend(loc="upper right")
        ax.set_xlim(0, np.quantile(np.concatenate([d, exact]), 0.995))
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def convergence(run_dir, cfg, out):
    import pandas as pd
    h = pd.read_csv(os.path.join(run_dir, "history.csv"))
    s = cfg["settings"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.loglog(h["B"], h["r_h1"], color=CAT[0], label="H1 (stopping rule)")
    ax.loglog(h["B"], h["r_h0"], color=CAT[1], label="H0 (logged only)")
    ref = h["r_h1"].iloc[0] * h["B"].iloc[0] ** 0.5 / np.sqrt(h["B"])
    ax.loglog(h["B"], ref, color=MUTED, lw=1, ls=":", label="slope -1/2")
    ax.axhline(s["tau"], color=MUTED, lw=1, ls="--")
    ax.text(0.45, s["tau"], f"tau = {s['tau']}", color=MUTED, fontsize=8, ha="left", va="top",
            transform=ax.get_yaxis_transform())
    stop = cfg["summary"]["stop_B_per_chain"]
    if stop:
        ax.axvline(stop, color=MUTED, lw=1)
        ax.text(stop * 0.97, 0.97, f"stop at B = {stop}", color=MUTED, fontsize=8, ha="right", va="top",
                rotation=90, transform=ax.get_xaxis_transform())
    ax.set_xlabel("B (diagrams per chain)")
    ax.set_ylabel("r_B = OT2^2(chain 1, chain 2) / total persistence")
    ax.set_title(f"Two-chain agreement, n = {s['n']:,}, grid {s['grid']}", loc="left")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def compare_h1(runs, out):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    runs = sorted(runs, key=lambda r: int(r[0]["n"]))
    for (z, _), c in zip(runs, CAT):
        P, w = z["h1_points"], z["h1_weights"]
        pers = P[:, 1] - P[:, 0]
        order = np.argsort(pers)[::-1]
        ax.semilogy(pers[order], np.cumsum(w[order]), color=c, label=f"n = {int(z['n']):,}")
    ax.set_xlabel("persistence x")
    ax.set_ylabel("mean H1 points per subsample\nwith persistence > x")
    ax.set_title("H1 persistence tail at different subsample sizes", loc="left")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--exact-h0", default=None, help="exact_h0.npy from exact_h0.py")
    ap.add_argument("--compare", nargs="*", default=[], help="other run folders (e.g. n/2)")
    args = ap.parse_args()
    z, cfg = load_run(args.run_dir)
    exact = np.load(args.exact_h0) if args.exact_h0 else None
    h1_heatmap(z, os.path.join(args.run_dir, "mean_h1_heatmap.png"))
    h0_hist(z, os.path.join(args.run_dir, "mean_h0_hist.png"), exact)
    convergence(args.run_dir, cfg, os.path.join(args.run_dir, "convergence.png"))
    if args.compare:
        runs = [(z, cfg)] + [load_run(d) for d in args.compare]
        compare_h1(runs, os.path.join(args.run_dir, "compare_h1.png"))
    print(f"figures written to {args.run_dir}")


if __name__ == "__main__":
    main()
