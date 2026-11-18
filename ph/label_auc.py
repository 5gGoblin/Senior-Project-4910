"""ROC AUC of the Gower distance against assigned labels (make, body type).

Usage:
    python label_auc.py --matrix ../gower_output/gower.npy --edges ../ph_output/mst_edges.npz \
        --lookup ../gower_output/id_lookup.csv --labels ../data/cars_clean_identifiers.csv \
        --out ../ph_output/label_auc.json

Two AUCs per label column. Both are 0.5 when the distance carries no
information about the label and 1.0 when it separates the label perfectly.

H0 merge AUC (uses the MST edges from exact_h0.py --edges)
    Every MST edge is one H0 merge, at its length. Call an edge "cross" if its
    two vehicles have different labels. AUC = P(a random cross edge is longer
    than a random within edge), ties counted as 1/2. It asks whether the
    vehicles merge within a label before they merge across labels.

Retrieval AUC (uses the full matrix)
    For a query vehicle, rank every other vehicle by distance. AUC = P(a random
    vehicle with the same label is closer than a random vehicle with a
    different label), ties 1/2. This is computed for --queries random query
    vehicles and averaged. It replaces "how many of the 10 nearest share the
    label" with a measure over the whole ranking.

Vehicles with a missing label are left out (edges or pairs that touch one).
If a label column is also a feature of the distance, its AUC is inflated by
construction and should only be read as a reference.
"""
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def auc(scores, y):
    """ROC AUC with y = 1 as the positive class (Mann-Whitney U / (n1 n0), ties 1/2)."""
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y, dtype=bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(scores)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def roc_curve(scores, y):
    """False and true positive rates for every threshold, predicting y = 1 when score >= threshold."""
    scores = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y, dtype=bool)
    order = np.argsort(-scores, kind="stable")
    s, yy = scores[order], y[order]
    last = np.r_[np.nonzero(np.diff(s))[0], len(s) - 1]  # last index of each distinct score
    tp = np.cumsum(yy)[last]
    fp = np.cumsum(~yy)[last]
    return np.r_[0, fp / max(fp[-1], 1)], np.r_[0, tp / max(tp[-1], 1)]


def load_labels(lookup_csv, labels_csv, cols):
    """Label columns in matrix row order, matched on id_trim. Missing values become None."""
    lookup = pd.read_csv(lookup_csv, usecols=["row", "id_trim"]).sort_values("row")
    lab = pd.read_csv(labels_csv, usecols=["id_trim"] + list(cols))
    m = lookup.merge(lab, on="id_trim", how="left", validate="1:1")
    if not (m["row"].to_numpy() == np.arange(len(m))).all():
        raise RuntimeError("id_lookup.csv rows are not 0..N-1")
    return {c: m[c].astype(object).where(m[c].notna(), None).to_numpy() for c in cols}


def h0_edge_auc(edges, labels):
    """AUC of MST edge length for 'edge joins two different labels'. Returns auc, fpr, tpr, counts."""
    u, v, length = edges["u"], edges["v"], edges["length"]
    a, b = labels[u], labels[v]
    ok = np.array([x is not None and y is not None for x, y in zip(a, b)])
    cross = (a != b) & ok
    s, y = length[ok], cross[ok]
    fpr, tpr = roc_curve(s, y)
    return {"auc": auc(s, y), "edges": int(ok.sum()), "cross_edges": int(y.sum()),
            "cross_share": float(y.mean())}, fpr, tpr


def retrieval_auc(D, labels, queries, seed=0):
    """Mean over query vehicles of P(same-label vehicle closer than different-label vehicle)."""
    N = D.shape[0]
    known = np.array([x is not None for x in labels])
    codes = pd.factorize(pd.Series(labels).where(known))[0]  # -1 for missing
    rng = np.random.default_rng(seed)
    q = np.sort(rng.choice(np.nonzero(known)[0], size=min(queries, int(known.sum())), replace=False))
    out = []
    for start in range(0, len(q), 200):
        rows = q[start:start + 200]
        block = np.asarray(D[rows], dtype=np.float64)
        for i, r in enumerate(rows):
            keep = known.copy()
            keep[r] = False
            same = codes[keep] == codes[r]
            # closer = better, so score is -distance
            out.append(auc(-block[i, keep], same))
    out = np.array(out)
    out = out[np.isfinite(out)]  # a query whose label is unique has no positives
    return {"auc_mean": float(out.mean()), "auc_median": float(np.median(out)),
            "auc_q25": float(np.quantile(out, 0.25)), "auc_q75": float(np.quantile(out, 0.75)),
            "queries": int(len(out))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", required=True, help="gower.npy")
    ap.add_argument("--edges", required=True, help="mst_edges.npz from exact_h0.py --edges")
    ap.add_argument("--lookup", required=True, help="id_lookup.csv written next to gower.npy")
    ap.add_argument("--labels", required=True, help="CSV with id_trim and the label columns")
    ap.add_argument("--cols", default="Make,Body_type")
    ap.add_argument("--queries", type=int, default=2000, help="query vehicles for the retrieval AUC")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="JSON file for the results")
    args = ap.parse_args()
    cols = [c for c in args.cols.split(",") if c]
    labels = load_labels(args.lookup, args.labels, cols)
    edges = dict(np.load(args.edges))
    D = np.load(args.matrix, mmap_mode="r")
    res = {}
    for c in cols:
        h0, _, _ = h0_edge_auc(edges, labels[c])
        res[c] = {"h0_merge": h0, "retrieval": retrieval_auc(D, labels[c], args.queries, args.seed)}
        print(c, json.dumps(res[c]))
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
