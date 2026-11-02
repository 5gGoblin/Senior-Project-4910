"""Side-by-side comparison of two subsampled PH runs built on different Gower matrices.

Usage:
    python compare_runs.py \
        --run ../ph_output/n4000 --exact ../ph_output/exact_h0.npy --name "specs only" \
        --run ../ph_output_labels/n4000 --exact ../ph_output_labels/exact_h0.npy --name "specs + labels" \
        --out-dir ../ph_output_compare

Writes into --out-dir:
    compare_h1_heatmaps.png   the two H1 mean measures on one shared colour scale
    compare_h1_tail.png       mean H1 points per subsample with persistence > x
    compare_h0.png            exact (MST) H0 deaths and subsample H0 deaths, both runs
    compare_summary.json      the numbers quoted in the figures

Both runs should use the same N, n and seed. Then subsample i of chain c holds
the same vehicles in both runs, and the difference comes only from the distance.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from plot_mean_measure import CAT, INK, MUTED, SEQ, load_run


def h1_stats(z):
    P, w = z["h1_points"], z["h1_weights"]
    birth, pers = P[:, 0], P[:, 1] - P[:, 0]
    out = {"h1_points_per_subsample": float(w.sum()),
           "h1_median_birth": float(np.median(birth)),
           "h1_max_persistence": float(pers.max())}
    for t in (0.02, 0.04, 0.06, 0.08):
        out[f"h1_per_subsample_pers_gt_{t}"] = float(w[pers > t].sum())
    return out


def h0_stats(z, exact):
    d = z["h0_points"][:, 1]
    ex = exact[exact > 0]
    return {"h0_subsample_median": float(np.median(d)),
            "h0_exact_median_nonzero": float(np.median(ex)),
            "h0_exact_zero_edges": int((exact == 0).sum()),
            "h0_exact_quantiles_10_50_90_99": [float(v) for v in np.quantile(ex, [.1, .5, .9, .99])],
            "h0_exact_max": float(exact.max())}


def heatmaps(runs, names, out, bin_b=0.004, bin_p=0.002):
    pts = [(z["h1_points"][:, 0], z["h1_points"][:, 1] - z["h1_points"][:, 0], z["h1_weights"])
           for z, _ in runs]
    bmax = max(np.quantile(b, 0.999) for b, _, _ in pts) + bin_b
    pmax = max(p.max() for _, p, _ in pts) + bin_p
    xe = np.arange(0, bmax + bin_b, bin_b)
    ye = np.arange(0, pmax + bin_p, bin_p)
    Hs = [np.histogram2d(b, p, bins=[xe, ye], weights=w)[0] for b, p, w in pts]
    vmin = min(H[H > 0].min() for H in Hs)
    vmax = max(H.max() for H in Hs)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True, sharey=True)
    for ax, H, name, (z, _) in zip(axes, Hs, names, runs):
        ax.grid(False)
        m = ax.pcolormesh(xe, ye, np.ma.masked_less_equal(H, 0).T, cmap=SEQ,
                          norm=LogNorm(vmin=vmin, vmax=vmax), shading="flat")
        ax.set_xlabel("birth (Gower distance)")
        ax.set_title(f"H1, {name}\nn = {int(z['n']):,}, {int(z['diagrams'])} subsamples", loc="left")
    axes[0].set_ylabel("persistence = death - birth")
    cb = fig.colorbar(m, ax=axes, pad=0.02)
    cb.set_label("mean H1 points per subsample in cell")
    cb.outline.set_visible(False)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def tail(runs, names, out):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for (z, _), name, c in zip(runs, names, CAT):
        P, w = z["h1_points"], z["h1_weights"]
        pers = P[:, 1] - P[:, 0]
        order = np.argsort(pers)[::-1]
        ax.semilogy(pers[order], np.cumsum(w[order]), color=c, label=name)
    ax.set_xlabel("persistence x")
    ax.set_ylabel("mean H1 points per subsample\nwith persistence > x")
    ax.set_title(f"H1 persistence tail, n = {int(runs[0][0]['n']):,}", loc="left")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def h0(runs, exacts, names, out, bin_d=0.002):
    subs = [z["h0_points"][:, 1] for z, _ in runs]
    exs = [e[e > 0] for e in exacts]
    hi = np.quantile(np.concatenate(subs + exs), 0.995)
    edges = np.arange(0, hi + bin_d, bin_d)
    x = edges[:-1] + bin_d / 2
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, data, title in ((axes[0], exs, "Exact H0 (MST on all vehicles), zero-length edges left out"),
                            (axes[1], subs, f"Subsample H0, n = {int(runs[0][0]['n']):,}")):
        for d, name, c in zip(data, names, CAT):
            share, _ = np.histogram(d, bins=edges)
            ax.plot(x, share / len(d), color=c, label=f"{name}, median {np.median(d):.4f}")
            ax.axvline(np.median(d), color=c, lw=1, ls="--")
        ax.set_xlabel("death (Gower distance)")
        ax.set_ylabel("share of H0 bars in bin")
        ax.set_title(title, loc="left")
        ax.legend(loc="upper right")
        ax.set_xlim(0, hi)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help="run folder (give twice)")
    ap.add_argument("--exact", action="append", required=True, help="exact_h0.npy for each run, same order")
    ap.add_argument("--name", action="append", required=True, help="label for each run, same order")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    if not len(args.run) == len(args.exact) == len(args.name) == 2:
        ap.error("give --run, --exact and --name exactly twice each")
    os.makedirs(args.out_dir, exist_ok=True)
    runs = [load_run(r) for r in args.run]
    exacts = [np.load(e) for e in args.exact]
    for k in ("N", "n", "seed"):
        a, b = (cfg["settings"][k] for _, cfg in runs)
        if a != b:
            print(f"WARNING: runs differ in {k} ({a} vs {b}); the comparison is not paired")

    heatmaps(runs, args.name, os.path.join(args.out_dir, "compare_h1_heatmaps.png"))
    tail(runs, args.name, os.path.join(args.out_dir, "compare_h1_tail.png"))
    h0(runs, exacts, args.name, os.path.join(args.out_dir, "compare_h0.png"))

    summary = {}
    for (z, cfg), ex, name, path in zip(runs, exacts, args.name, args.run):
        summary[name] = {"run": os.path.abspath(path), "n": int(z["n"]), "N": int(z["N"]),
                         "diagrams": int(z["diagrams"]),
                         "stop_B_per_chain": cfg["summary"]["stop_B_per_chain"],
                         **h1_stats(z), **h0_stats(z, ex)}
    with open(os.path.join(args.out_dir, "compare_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    print(f"figures written to {args.out_dir}")


if __name__ == "__main__":
    main()
