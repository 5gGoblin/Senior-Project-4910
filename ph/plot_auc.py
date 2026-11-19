"""AUC figure for one label, comparing two Gower matrices.

Usage:
    python plot_auc.py --label Body_type \
        --name "specs only" --edges ../ph_output/mst_edges.npz --queries ../ph_output/label_auc.npz \
        --name "specs + labels" --edges ../ph_output_labels/mst_edges.npz --queries ../ph_output_labels/label_auc.npz \
        --lookup ../gower_output/id_lookup.csv --labels ../data/cars_clean_identifiers.csv \
        --out ../ph_output_compare/auc_body_type.png

Left panel: ROC curves of the H0 merge AUC (MST edge length vs "edge joins two
different labels"). Right panel: distribution of the per-query retrieval AUC
(the .npz written by label_auc.py), with the mean marked. See label_auc.py.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from label_auc import h0_edge_auc, load_labels
from plot_mean_measure import CAT, MUTED


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="Body_type")
    ap.add_argument("--name", action="append", required=True)
    ap.add_argument("--edges", action="append", required=True)
    ap.add_argument("--queries", action="append", required=True, help="label_auc.npz for each run")
    ap.add_argument("--lookup", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if not len(args.name) == len(args.edges) == len(args.queries):
        ap.error("give --name, --edges and --queries once per run")
    lab = load_labels(args.lookup, args.labels, [args.label])[args.label]
    pretty = args.label.replace("_", " ").lower()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    bins = np.linspace(0, 1, 41)
    for name, ep, qp, c in zip(args.name, args.edges, args.queries, CAT):
        res, fpr, tpr = h0_edge_auc(dict(np.load(ep)), lab)
        ax1.plot(fpr, tpr, color=c, label=f"{name}: AUC {res['auc']:.3f}")
        q = np.load(qp)[args.label]
        ax2.hist(q, bins=bins, color=c, alpha=0.45, label=f"{name}: mean {q.mean():.3f}")
        ax2.axvline(q.mean(), color=c, lw=1.5, ls="--")
    ax1.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=":", label="no information (0.5)")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel(f"false positive rate (within-{pretty} merges)")
    ax1.set_ylabel(f"true positive rate (cross-{pretty} merges)")
    ax1.set_title(f"H0 merge AUC, {pretty}\nexact MST edges, ranked by merge distance", loc="left")
    ax1.legend(loc="lower right")
    ax2.axvline(0.5, color=MUTED, lw=1, ls=":")
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("retrieval AUC of one query vehicle")
    ax2.set_ylabel("query vehicles")
    ax2.set_title(f"Retrieval AUC, {pretty}\n{len(q):,} random query vehicles per matrix", loc="left")
    ax2.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
