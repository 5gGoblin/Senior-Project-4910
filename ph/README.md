# Subsampled persistent homology

Vietoris–Rips persistent homology (H0 and H1) of the vehicles, computed on
random subsamples of the precomputed Gower matrix and averaged into a mean
persistence measure, following Cao and Monod, "Approximating Persistent
Homology for Large Datasets" (arXiv:2204.09155). The Fréchet mean is not
implemented yet.

## Files

| File | What it does |
| --- | --- |
| `ph_core.py` | subsample slicing + Ripser, mean measure, grid binning, OT₂² between measures, exact H0 by MST |
| `timing_test.py` | times one Ripser call at several n and suggests the largest n that fits a time budget |
| `run_subsampled_ph.py` | the main run: two chains of subsamples, stopping rule, mean measure; resumable |
| `exact_h0.py` | exact H0 death times for all N vehicles (minimum spanning tree) |
| `plot_mean_measure.py` | H1 heatmap, H0 histogram vs exact H0, convergence curve, n vs n/2 comparison |
| `test_ph.py` | pytest tests (OT against brute force, MST against Ripser, resume, plots) |
| `NOTES.md` | method choices, where they differ from the paper, first results on cars_step5 |
| `results_cars_step5/` | figures, `history.csv` and `config.json` from the cars_step5 runs (n = 1,000, 2,000, 4,000) |

## Setup

    pip install -r requirements.txt
    pytest test_ph.py

Needs the matrix from `../gower/build_gower.py` (`gower_output/gower.npy`).

## Run

From this folder:

    # 1. pick n (each size runs in its own process; prints a suggestion)
    python timing_test.py ../gower_output/gower.npy --sizes 1000,2000,3000,4000 --budget-min 60

    # 2. main run and the n/2 bias check
    python run_subsampled_ph.py ../gower_output/gower.npy --n 2000 --out-dir ../ph_output/n2000
    python run_subsampled_ph.py ../gower_output/gower.npy --n 1000 --out-dir ../ph_output/n1000

    # 3. exact H0 on all vehicles
    python exact_h0.py ../gower_output/gower.npy --out ../ph_output/exact_h0.npy

    # 4. figures (written into the run folder)
    python plot_mean_measure.py ../ph_output/n2000 --exact-h0 ../ph_output/exact_h0.npy \
        --compare ../ph_output/n1000

- Each subsample's diagrams are saved to `diagrams/` as soon as they are done.
  Rerunning the same command resumes. Rerunning with a smaller `--tau` or larger
  `--b-max` extends a finished run; changing `--n` or `--seed` needs a new `--out-dir`.
- Subsample *i* of chain *c* is drawn with a generator seeded by `(seed, c, i)`,
  so results do not depend on `--jobs` or `--batch`.
- Memory: one Ripser process peaks around 1.1 GB at n = 2,000 and up to
  4.4 GB at n = 5,000. Lower `--jobs` if the machine runs out of memory.
- Keep `ph_output/` out of Git (it holds one `.npz` per subsample).

## Outputs (in `--out-dir`)

| File | Contents |
| --- | --- |
| `mean_measure.npz` | pooled H0 and H1 points (birth, death) with weight 1/(2B) each; `n`, `N`, `diagrams` |
| `history.csv` | per B: `r_h1` (stopping statistic), `r_h0`, total persistence, the paper's `Err_B` and `Q*` |
| `config.json` | settings, summary (stop B, timings, points per subsample), library versions |
| `diagrams/chain{c}_{i}.npz` | one subsample: row indices `idx`, `h0`, `h1`, Ripser seconds |
| `*.png` | figures from `plot_mean_measure.py` |

Row indices refer to rows of `gower_output/id_lookup.csv`.
