"""Turn the cleaned table into the arrays the kernel needs.

Everything here runs once on the full dataset, so every block uses the same
ranges and the same category codes.
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd

# identifiers and metadata: kept in the lookup table, never used in the distance
DEFAULT_EXCLUDE = [
    "Unnamed: 0", "id_trim", "Make", "Modle", "Generation", "Series", "Trim",
    "label", "pct_filled", "Body_type", "car_class", "country_of_origin",
    "emission_standards", "Year_from", "Year_to",
]


def load_table(path):
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def prepare(df, exclude=DEFAULT_EXCLUDE, force_categorical=(), sort_by=None, max_missing=None):
    """Split the table into numeric and categorical feature arrays.

    Returns a dict with x_num, x_cat, ranges, inv_range, num_cols, cat_cols,
    categories, lookup and dropped (columns removed and why).
    """
    if sort_by is not None:
        df = df.sort_values(sort_by, kind="stable")
    # row r of the matrix is row r of the lookup table from here on
    df = df.reset_index(drop=True)

    exclude = [c for c in exclude if c in df.columns]
    lookup = df[exclude].copy()
    lookup.insert(0, "row", np.arange(len(df)))
    feats = df.drop(columns=exclude)

    dropped = {}
    if max_missing is not None:
        miss = feats.isna().mean()
        for c in miss.index[miss > max_missing]:
            dropped[c] = f"missing {miss[c]:.1%}"
        feats = feats.drop(columns=list(dropped))

    # select_dtypes catches int64 as well as float64
    num_cols = [c for c in feats.select_dtypes("number").columns if c not in force_categorical]
    cat_cols = [c for c in feats.columns if c not in num_cols]

    x_num = feats[num_cols].to_numpy(dtype=np.float32)
    with np.errstate(all="ignore"):
        ranges = np.nanmax(x_num, axis=0) - np.nanmin(x_num, axis=0)
    # a constant or empty column adds no information, so drop it instead of setting range = 1
    keep = np.isfinite(ranges) & (ranges > 0)
    for c in np.array(num_cols)[~keep]:
        dropped[c] = "constant or empty"
    num_cols = [c for c, k in zip(num_cols, keep) if k]
    x_num = np.ascontiguousarray(x_num[:, keep])
    ranges = ranges[keep].astype(np.float32)

    # codes built on the full data; pandas gives -1 for missing
    categories = {}
    codes = []
    for c in list(cat_cols):
        cat = feats[c].astype("category")
        if len(cat.cat.categories) < 2:
            dropped[c] = "constant or empty"
            cat_cols.remove(c)
            continue
        if len(cat.cat.categories) > np.iinfo(np.int16).max:
            raise ValueError(f"{c} has too many categories for int16")
        categories[c] = [str(v) for v in cat.cat.categories]
        codes.append(cat.cat.codes.to_numpy())
    if codes:
        x_cat = np.ascontiguousarray(np.column_stack(codes).astype(np.int16))
    else:
        x_cat = np.empty((len(df), 0), dtype=np.int16)

    return {
        "x_num": x_num,
        "x_cat": x_cat,
        "ranges": ranges,
        "inv_range": (1.0 / ranges).astype(np.float32),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "categories": categories,
        "lookup": lookup,
        "dropped": dropped,
    }


def fingerprint(prep):
    """Hash of everything that changes the distances. A resume is only allowed if it matches."""
    h = hashlib.sha256()
    h.update(json.dumps([prep["num_cols"], prep["cat_cols"], prep["categories"]]).encode())
    for key in ("x_num", "x_cat", "ranges"):
        h.update(np.ascontiguousarray(prep[key]).tobytes())
    return h.hexdigest()[:16]


def save_inputs(prep, out_dir, extra_config=None):
    """Write the files needed to reproduce the matrix."""
    os.makedirs(out_dir, exist_ok=True)
    pd.Series(prep["ranges"], index=prep["num_cols"], name="range").to_csv(
        os.path.join(out_dir, "ranges.csv"), index_label="feature")
    with open(os.path.join(out_dir, "categories.json"), "w") as f:
        json.dump(prep["categories"], f, indent=1)
    config = {
        "n_rows": int(prep["x_num"].shape[0]),
        "numeric_features": prep["num_cols"],
        "categorical_features": prep["cat_cols"],
        "dropped": prep["dropped"],
        "weights": "equal",
        "fingerprint": fingerprint(prep),
    }
    config.update(extra_config or {})
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=1)
    return config
