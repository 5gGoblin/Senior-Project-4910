"""Tests for the Gower build. Run with:  pytest test_gower.py"""
import os

import numpy as np
import pandas as pd
import pytest

from build_gower import build, summarize
from gower_kernel import gower_reference, gower_rows
from gower_prep import fingerprint, prepare


def full_matrix(prep):
    n = prep["x_num"].shape[0]
    out = np.empty((n, n), np.float32)
    shared = np.empty((n, n), np.int16)
    gower_rows(prep["x_num"], prep["x_cat"], prep["x_num"], prep["x_cat"], prep["inv_range"], out, shared)
    return out, shared


def synthetic(n=2000, seed=0, missing=0.15):
    """Fake vehicles: 6 numeric (some int64) + 3 categorical features, with missing values."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "id_trim": np.arange(n),
        "Make": rng.choice(["A", "B", "C"], n),
        "length_mm": rng.integers(3500, 5500, n),
        "curb_weight_kg": rng.normal(1600, 300, n),
        "engine_hp": rng.normal(200, 70, n),
        "number_of_seats": rng.integers(2, 9, n),
        "max_speed_km_per_h": rng.normal(210, 30, n),
        "payload_kg": rng.normal(500, 100, n),
        "drive_wheels": rng.choice(["FWD", "RWD", "AWD"], n),
        "engine_type": rng.choice(["Gasoline", "Diesel", "Hybrid", "Electric"], n),
        "transmission": rng.choice(["Manual", "Automatic", "CVT"], n),
    })
    df["length_mm"] = df["length_mm"].astype("Int64")  # nullable int, like a cleaned export
    for c in df.columns[2:]:
        df.loc[rng.random(n) < missing, c] = None
    return df


def test_hand_computed_example():
    df = pd.DataFrame({
        "x": [0.0, 1.0, np.nan, 4.0],
        "y": [10.0, np.nan, 30.0, 20.0],
        "c": pd.Series(["a", "b", None, "a"], dtype=object),
    })
    prep = prepare(df, exclude=[])
    D, W = full_matrix(prep)
    # ranges: x = 4, y = 20
    expected = {(0, 1): 0.625, (0, 2): 1.0, (0, 3): 0.5, (1, 3): 0.875, (2, 3): 0.5}
    for (i, j), d in expected.items():
        assert D[i, j] == pytest.approx(d, abs=1e-6)
        assert D[j, i] == pytest.approx(d, abs=1e-6)
    # rows 1 and 2 share no features: W = 0 must give NaN, not 0
    assert W[1, 2] == 0
    assert np.isnan(D[1, 2])
    assert W[0, 3] == 3 and W[0, 1] == 2


def test_column_typing():
    prep = prepare(synthetic(200), exclude=["id_trim", "Make"])
    # int64 and nullable Int64 columns must be numeric, not dropped
    assert "number_of_seats" in prep["num_cols"]
    assert "length_mm" in prep["num_cols"]
    assert prep["cat_cols"] == ["drive_wheels", "engine_type", "transmission"]
    assert "Make" not in prep["num_cols"] + prep["cat_cols"]
    # missing categories are -1
    assert (prep["x_cat"] == -1).any()


def test_constant_columns_dropped():
    df = synthetic(100)
    df["const_num"] = 5.0
    df["const_cat"] = "same"
    prep = prepare(df, exclude=["id_trim", "Make"])
    assert "const_num" in prep["dropped"] and "const_cat" in prep["dropped"]


def test_ranges_come_from_full_data():
    df = synthetic(500)
    prep = prepare(df, exclude=["id_trim", "Make"])
    k = prep["num_cols"].index("engine_hp")
    assert prep["ranges"][k] == pytest.approx(df["engine_hp"].max() - df["engine_hp"].min(), rel=1e-6)


def test_kernel_matches_reference():
    prep = prepare(synthetic(300, seed=1), exclude=["id_trim", "Make"])
    D, W = full_matrix(prep)
    rng = np.random.default_rng(2)
    for i, j in rng.integers(0, 300, (400, 2)):
        d, w = gower_reference(prep["x_num"], prep["x_cat"], prep["ranges"], i, j)
        assert W[i, j] == w
        assert D[i, j] == pytest.approx(d, abs=1e-6)
    assert np.array_equal(D, D.T)
    assert np.nanmin(D) >= 0 and np.nanmax(D) <= 1


def test_blocks_match_one_shot(tmp_path):
    prep = prepare(synthetic(2000), exclude=["id_trim", "Make"])
    D_full, _ = full_matrix(prep)
    np.fill_diagonal(D_full, 0.0)
    assert build(prep, str(tmp_path), block=300, verbose=False)  # uneven last block
    D = np.load(tmp_path / "gower.npy")
    assert np.array_equal(D, D_full, equal_nan=True)  # a few W = 0 pairs are NaN in both


def test_resume_gives_identical_file(tmp_path):
    prep = prepare(synthetic(1500), exclude=["id_trim", "Make"])
    clean, resumed = tmp_path / "clean", tmp_path / "resumed"
    assert build(prep, str(clean), block=200, verbose=False)
    assert not build(prep, str(resumed), block=200, max_blocks=3, verbose=False)  # "crash" after 3 blocks
    assert build(prep, str(resumed), block=200, verbose=False)
    with open(clean / "gower.npy", "rb") as a, open(resumed / "gower.npy", "rb") as b:
        assert a.read() == b.read()
    assert summarize(prep, str(clean)) == summarize(prep, str(resumed))


def test_resume_refuses_changed_inputs(tmp_path):
    df = synthetic(500)
    prep = prepare(df, exclude=["id_trim", "Make"])
    assert not build(prep, str(tmp_path), block=100, max_blocks=2, verbose=False)
    df.loc[0, "engine_hp"] = 9999.0  # changes the range of engine_hp
    prep2 = prepare(df, exclude=["id_trim", "Make"])
    assert fingerprint(prep) != fingerprint(prep2)
    with pytest.raises(RuntimeError, match="different inputs"):
        build(prep2, str(tmp_path), block=100, verbose=False)


def test_low_overlap_report(tmp_path):
    prep = prepare(synthetic(600, missing=0.4), exclude=["id_trim", "Make"])
    _, W = full_matrix(prep)
    p = prep["x_num"].shape[1] + prep["x_cat"].shape[1]
    assert build(prep, str(tmp_path), block=128, floor=6, hard_floor=4, verbose=False)
    report = summarize(prep, str(tmp_path), floor=6, hard_floor=4)
    iu = np.triu_indices(600, k=1)
    assert report["pairs"] == len(iu[0])
    assert report["pairs_below_floor"]["pairs"] == int((W[iu] < 6).sum())
    pairs = pd.read_csv(tmp_path / "low_overlap_pairs.csv")
    assert len(pairs) == int((W[iu] < 4).sum())
    assert (pairs["row_i"] < pairs["row_j"]).all()
    lookup = pd.read_csv(tmp_path / "id_lookup.csv")
    np.fill_diagonal(W, p)
    assert np.array_equal(lookup["min_shared_features"], W.min(axis=1))
