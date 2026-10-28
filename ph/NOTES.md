# Subsampled PH: method notes and first results

October 25, 2026. Results are on `cars_step5.csv` (31,645 vehicles, 41
features). They need to be rerun on the final cleaned data.

## What the code does

1. Draw a subsample of n distinct vehicles uniformly at random.
2. Slice the n × n block from `gower.npy` and run Ripser
   (`distance_matrix=True`, `maxdim=1`, no threshold). Drop the infinite H0 bar.
3. Do this for two independent chains of subsamples. After each new pair of
   diagrams, compare the two chains' H1 mean measures (stopping rule below).
4. Pool every diagram from both chains into one mean persistence measure: each
   point keeps its (birth, death) and gets weight 1/(2B).
5. Plot H1 as a weighted 2D histogram in (birth, persistence) coordinates and H0
   as a weighted histogram of death times.

p = 2 is used in every comparison between measures: OT₂² with squared
Euclidean ground cost, and (d − b)²/2 as the cost of sending a point to the
diagonal. The mean measure itself does not depend on p.

## Where this differs from the paper

| Item | Paper | Here | Reason |
| --- | --- | --- | --- |
| Sampling within a subsample | i.i.d. (with replacement) | without replacement | Repeats only add zero-distance pairs. With n = 2,000 and N = 31,645 about 63 repeats would be expected; the effect is small. |
| Stopping rule | add one subsample at a time; stop when Q* = \|Err_B / Err_{B−1} − 1\| < 10⁻³ | two chains; stop when r_B < τ for 5 B in a row, B ≥ 20 per chain | Err_B shrinks like 1/B from the averaging weight alone, so it does not measure convergence. See below. |
| H0 | subsampled | exact, from the minimum spanning tree of all N vehicles; the subsampled H0 is kept as a bias check | H0 of a Rips filtration is the MST edge lengths. |
| OT computations | exact on raw points | both measures binned to a 0.002 grid first | Keeps the transport problem small. Moving each point by at most 0.0014 bounds the change in OT₂. |
| Distance | Euclidean point clouds in the theory | Gower, non-Euclidean | The theory does not cover this case. See "Uncertain" below. |

The stopping statistic is

    r_B = OT₂²(mean of chain 1, mean of chain 2) / (½ (T₁ + T₂))

where T is total persistence, OT₂² to the empty measure. It measures how much
two independent estimates disagree, as a fraction of the size of the measure.
It only measures variance. Bias (dependence on n) is checked by running
another n.

The paper's Q* is computed on chain 1 and saved in `history.csv` so the two
rules can be compared.

## First results (cars_step5)

### Timing

One Ripser call (H0 + H1) on this machine (4 cores, 15 GB):

| n | seconds | peak memory |
| --- | --- | --- |
| 1,000 | 1.0–1.3 | 0.6 GB |
| 2,000 | 5–7 | 1.1 GB |
| 3,000 | 18 | 1.6 GB |
| 4,000 | 36 (mean of 40) | 3.1 GB |
| 5,000 | 54 | 4.4 GB |

Peak memory is the whole Python process, measured in a fresh process for n ≤ 3,000
and as a running maximum for 4,000 and 5,000, so those two are upper bounds.
Cost grows roughly like n^2.3 over this range. This is much faster than the
paper's Euclidean timings. The full n = 4,000 run (40 diagrams, 4 parallel
jobs) took 7.5 minutes. n is not fixed yet; n = 4,000 or 5,000 looks affordable
on a laptop if run overnight. The timing should be repeated on the machine we
use for the final run.

### Stopping rule

| run | τ | stopped at B per chain | total diagrams | final r_B (H1) | paper rule met? |
| --- | --- | --- | --- | --- | --- |
| n = 1,000 | 0.05 | 20 | 40 | 0.016 | no |
| n = 2,000 | 0.05 | 20 | 40 | 0.014 | no |
| n = 2,000 | 0.01 | 52 | 104 | 0.009 | no (Q* never below 10⁻³ in 52 steps) |
| n = 4,000 | 0.05 | 20 | 40 | 0.007 | no |

- r_B (H1) falls close to B^(−1/2), as expected for a variance term.
- With τ = 0.05 the rule never binds: r_B is below 0.05 from B = 2, so the
  minimum B = 20 decides. The default is now τ = 0.01. Like 0.05, this value is a
  choice, not derived from anything.
- The paper's Q* moved between 0.002 and 0.70 with no trend over 52 steps.
  Err_B fell from 0.005 to 0.0002, close to 1/B. This is consistent with the argument that
  Err_B is driven by the averaging weight.
- Larger n gives lower r_B at the same B (0.007 at n = 4,000 vs 0.014 at
  n = 2,000), so fewer subsamples are needed at larger n.

### H1 mean persistence measure

See `mean_h1_heatmap.png` (n = 2,000, 104 subsamples).

- The measure is one connected region near the diagonal. Births are mostly
  0.03–0.15, persistence mostly below 0.04. There is no separate group of points
  far from the diagonal.
- About 1,830 H1 points per subsample. The largest persistence in any
  subsample was about 0.10.
- Mean points per subsample with persistence above a level, by n:

  | n | > 0.04 | > 0.06 | > 0.08 | max persistence |
  | --- | --- | --- | --- | --- |
  | 1,000 | 13.9 | 1.30 | 0.20 | 0.094 |
  | 2,000 | 22.9 | 2.12 | 0.33 | 0.104 |
  | 4,000 | 38.0 | 3.60 | 0.53 | 0.096 |

  The count above each level grows with n, at about n^0.7, while the maximum
  persistence stays near 0.1. The median birth moves down with n (0.092, 0.082,
  0.073), which is expected because denser samples connect at shorter distances.
- Reading: at these sample sizes the data shows many small loops at all scales
  and no small set of dominant loops whose count stays fixed as n grows. This
  does not rule out loops that involve rare vehicles, since uniform subsamples
  rarely include them.

### H0: subsample vs exact

See `mean_h0_hist.png`. The exact MST on all 31,645 vehicles takes 2 seconds
once the matrix is in memory. It has 1,212 zero-length edges (vehicles with
identical specs on the shared features).

| | median H0 death |
| --- | --- |
| exact, N = 31,645 (zeros left out) | 0.011 |
| subsamples, n = 4,000 | 0.036 |
| subsamples, n = 2,000 | 0.046 |
| subsamples, n = 1,000 | 0.057 |

Subsample H0 deaths are 3 to 5 times the full-data values. This confirms the
earlier proposal: DBSCAN ε candidates for Mapper should come from the exact MST
deaths or k-nearest-neighbor distances on the full matrix, not from subsample H0.

A log-log fit of median death against n gives a slope of −0.34 across the three
subsample sizes and −0.49 when the full data is included, which would put the
intrinsic dimension b roughly between 2 and 3. This is a heuristic estimate and
should not be quoted as a measured dimension.

## Uncertain

- **Triangle inequality.** In 10⁶ random triples from the matrix, 0.1% violated
  the triangle inequality, by up to 0.156. These come from pairs that share few
  non-missing features. Ripser still runs, but the stability results behind the
  paper's bias bound assume a metric.
- **τ and B_min** are choices. Report them with the r_B curve.
- **Grid size 0.002** is coarse relative to the many H1 points with persistence
  below 0.002. Those points have almost no OT₂² cost, so this should not change
  the stopping decision much, but it has not been tested with a finer grid.
- **Counts are per subsample.** The heatmap says how many features an n-vehicle
  subsample has, not how many the full dataset has.

## Not done yet

- Fréchet mean of the H1 diagrams (GUDHI).
- Rerun everything on the final cleaned CSV, with n from a timing test on the
  machine used for the final run.
