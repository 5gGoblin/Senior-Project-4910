# Gower distance matrix

Builds the full N x N Gower distance matrix for the cleaned vehicle data as a
float32 `.npy` file on disk, 1,000 rows at a time.

## Files

| File | What it does |
| --- | --- |
| `gower_kernel.py` | numba kernel (`gower_rows`) and a slow reference version for tests |
| `gower_prep.py` | splits the table into numeric and categorical arrays, full-data ranges, category codes (missing = -1) |
| `build_gower.py` | block loop, checkpoints, low-overlap report; the script you run |
| `validate_gower.py` | checks a finished matrix one block at a time |
| `test_gower.py` | pytest tests |

## Setup

    pip install -r requirements.txt
    pytest test_gower.py

## Run

From this folder:

    python build_gower.py ../data/<cleaned file>.csv --out-dir ../gower_output

- Identifier and metadata columns (`id_trim`, `Make`, `Modle`, `Generation`, `Series`,
  `Trim`, `Body_type`, `car_class`, `country_of_origin`, `emission_standards`,
  `Year_from`, `Year_to`, ...) are left out of the distance and kept in
  `id_lookup.csv`. Change the list with `--exclude a,b,c`.
- Every other numeric column (int or float) is a numeric feature; everything
  else is categorical. Force a numeric-looking column to categorical with
  `--categorical col1,col2`.
- Rows are sorted by `id_trim`. Row r of the matrix is row r of `id_lookup.csv`.
- If the run stops, rerun the same command. Finished blocks are skipped. If the
  input data changed, the script refuses to resume; delete the output folder.
- The output folder holds a 3.6 to 4 GB file. Keep it out of Git
  (add `gower_output/` to `.gitignore`).

## Outputs (in `--out-dir`)

| File | Contents |
| --- | --- |
| `gower.npy` | the matrix; open with `np.load(path, mmap_mode="r")` |
| `id_lookup.csv` | row index, identifiers, metadata, `min_shared_features`, `partners_below_floor` |
| `config.json` | feature lists, dropped columns, input path, git commit, date, fingerprint |
| `ranges.csv`, `categories.json` | numeric ranges and category code tables |
| `w_histogram.csv` | number of pairs for each count of shared features |
| `low_overlap_pairs.csv` | pairs with fewer than `--hard-floor` (15) shared features |
| `low_overlap_report.json` | summary of the above plus distance quantiles |
| `validation.json` | symmetry, zero diagonal, range, NaN, and 1,000 cells recomputed |
| `feature_contribution.csv` | mean per-feature difference; values near 0 point to outlier-stretched ranges |
| `done_blocks.txt`, `w_stats/`, `x_num.npy`, `x_cat.npy` | checkpoint log and working files |
