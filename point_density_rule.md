# Point count: how many points each glacier needs

Sequel to `convergence_summary.md`, which ends with the wind-fallback ice-mask
fix. Everything here uses post-fix runs.

## Goal

Pick the point count to run each glacier at while calibrating `wind_factor`
and `kp`, with a stated tolerance, and justify it well enough to defend.

## What the sweep actually showed

Post-fix sweep, five glaciers, 58 runs (`_ct` tags). Mass balance barely
depends on mesh spacing: the full-sweep spread is only 0.018 to 0.052 m
w.e./yr, and three of the five glaciers never leave 2 cm/yr at any resolution
tried, down to 19 points on Lemon Creek.

That makes the question "find the spacing where the answer stops
changing" degenerate. For Gulkana, Kennicott and Lemon Creek the entire
sweep spread is smaller than the 2 cm tolerance, so every resolution passes
trivially and `recommend_n.py` collapses to the coarsest run in the sweep.
`plot_convergence_diff.py` has the same failure in picture form, and worse: it
defines "converged" as the mean of a stretch of points selected for having
spread below the tolerance, then plots deviation from that mean against a band
drawn at the same tolerance. Points in the stretch cannot land outside the
band. The band restates the selection rule rather than testing anything.

## Why "it stopped changing" was the wrong criterion

Each mesh is a different set of locations, so each run is a fresh draw. The
error is random, not monotone in spacing. Wolverine reads +3.4 cm at 45
points, +0.9 at 65, +1.6 at 166 -- that is the dice, not resolution.

Checking every resolution we would have called "converged" against the model
below, they all sat in a 58 to 88 percent pass band. Gulkana "converging" at
48 points was a 71 percent chance of passing that came up heads. Lemon Creek
at 19 points was 74 percent. Those were coin flips reported as results.

## The relationship we are using

The error of the area-weighted glacier mean is random with standard deviation

    s(N) = C * sigma / sqrt(N)

    sigma  = standard deviation of per-point annual mass balance across the
             glacier [cm w.e./yr]
    N      = number of points
    C      = 0.0603

Fit to 51 (glacier, resolution) error measurements, using each sweep's three
finest runs as the reference and excluding them from the fit.

Validation:

- the fitted exponent came out 0.457, i.e. the square root, which is the
  textbook sampling law rather than something tuned
- rms of the normalized error is flat across the low-N and high-N halves of
  the data, confirming the exponent
- mean of the normalized error is ~0, so it is unbiased
- residual scatter is 0.90 in log space against a floor of 1.11, which is what
  a *perfect* model of `s` would leave when applied to single draws. There is
  no systematic misfit left to find, which is also why no alternative
  normalization separated from the others.

C = 1/16.6 is the payoff from laying points out evenly instead of randomly:
the mesh delivers the accuracy a random sample of about 275x more points
would. Random sampling on these glaciers really does follow `sigma/sqrt(N)`
(checked directly, matched within 10 percent on all five), so the constant is
the stratification gain and nothing else.

## Turning that into a point count

`s` is a typical size, not a guarantee. The errors are approximately normal,
so 95 percent land within 1.96 s. Requiring `1.96 * s(N) <= tol`:

    N = (1.96 * C * sigma / tol)^2

At tol = 3 cm/yr and C = 0.0603 this is

    N = (sigma / 25.4)^2

Note `tol` enters squared, so halving the tolerance quadruples the points.

Why 3 cm/yr: the sampling error is a fixed offset for a given mesh, not fresh
noise each year. Same points, same elevations, same bias at every timestep, so
it does not average down over a long run -- it accumulates linearly. 3 cm/yr
is 0.6 m w.e. over 20 years, comfortably under the Hugonnet floor. 5 cm/yr
would be 1.0 m w.e. and sit right on it.

## Why spread (sigma), and not spacing or area

Spacing does not transfer: the same spacing buys very different accuracy on
different glaciers. Area does not transfer either; Kahiltna is 29x Gulkana's
area but what sets its requirement is its spread. Spread is the quantity the
sampling error actually depends on, and it is the only thing the calibration
parameters can change, so the rule self-corrects when they move.

Tried and rejected:

- elevation-detrended spread. Theoretically the right scale, since a mesh
  captures the elevation gradient for free, but unusable: it does not plateau
  with resolution (Wolverine reads 20 at 2000 m and 73 at 70 m, a 3.65x
  underestimate from a cheap coarse run) and it depends on the detrending
  method (Kahiltna is 126 with quantile bins, 246 with a cubic, and the fitted
  slope spread goes from 1.3x to 2.8x with that choice)
- Kish effective sample size, `1/sum(w^2)`, to account for uneven Voronoi
  weights. Correct in principle -- weights are genuinely uneven at coarse
  spacing, `N_eff/N` runs 0.47 to 0.99 -- but it improved the fit by 2 percent
- area-weighted elevation standard deviation, as a DEM-only proxy needing no
  simulation. Scatter was as good, but it over-resolves by 2.3x overall
  because spread/elev-sd runs 0.43 (Kennicott, continental) to 1.29
  (Wolverine, maritime). Usable as a no-pilot fallback with a climate class,
  not as the primary

Plain spread plateaus acceptably (coarse/finest runs 0.76 to 1.07) and has no
free choices in it.

## Parameter dependence

45 runs, a 3x3 grid in (`wind_factor`, `kp`) over {1, 2.5, 5}, one spacing per
glacier (`_pg_wf*_kp*` tags). The centre cell repeats the sweep's parameters
as a control and reproduced the already-measured spread to within 0.2 percent
on all five glaciers.

- Spread moves a lot with parameters. Gulkana goes 96.7 at wf=1 to 233 at
  wf=2.5 to 413.8 at wf=5, scaling as wf^0.90. So a point count cannot be
  carried between parameter sets; the spread has to be re-measured. It is
  cheap to re-measure, being a standard deviation over hundreds of points
  rather than a single-draw error.
- C itself does not drift with wind factor. Elevation explains 96 percent of
  the variance at wf=1, 2.5 and 5 alike (R2 0.959, 0.965, 0.962) and the
  residual fraction is flat near 19 percent. Wind scales the field while
  preserving its elevation structure, so the mesh keeps its advantage. This
  was the main worry about the rule transferring, and it does not happen.
- `kp` matters, and on Kahiltna it dominates wind factor. At wf=2.5, taking kp
  from 1 to 5 moves the spread 412 to 658 (+60 percent); at kp=5, taking wf
  from 1 to 5 moves it 603 to 624 (+3 percent).
- The worst cell is not at a corner of the box for four of the five glaciers.
  Sampling corners only would understate the worst case by 3 to 8 percent.
  Wolverine and Lemon Creek are also non-monotone in wind factor (Wolverine at
  kp=1 reads 143, 327, 237 as wf goes 1, 2.5, 5) against a ~3 percent
  measurement noise floor, so that structure is real. Sample the interior.

## Final numbers

3 cm/yr at 95 percent confidence, each glacier sized at its worst cell in the
parameter box, so one fixed mesh holds everywhere we calibrate.

| glacier | max spread | N | h |
| --- | --- | --- | --- |
| kahiltna | 658.2 | 672 | ~2020 m |
| kennicott | 630.1 | 616 | ~1330 m |
| gulkana | 411.3 | 263 | ~450 m |
| wolverine | 385.6 | 231 | ~420 m |
| lemon_creek | 282.3 | 124 | ~480 m |

Total 1906 points. At 99 percent it would be 3292.

N is the target, not h. At these coarse spacings the points-per-spacing
relation is unstable -- `N*h^2/A` runs from 2.2 at fine spacing to about 11 at
coarse -- so confirm the spacing with `mesh_preview.py`, which costs no
simulation time.

## Limits

- C is fit from five Alaskan glaciers over one 3-year period. Parameter
  transfer was tested on Gulkana only.
- The eight error points from the wf=1 and wf=5 sweeps pool to C = 0.080
  against 0.0559 at the centre. That is 1.2 sigma on eight points and
  non-monotone in wind factor, so it reads as noise; part of it is also the
  single finest run being used as the reference there, which inflates measured
  errors by up to 16 percent at the fine end. Refitting on all 51 points gives
  the 0.0603 used above.
- Assumes errors are normal and independent between meshes. The retrodiction
  above is consistent with that but does not prove it.
- The grid used one spacing per glacier, so it maps spread across the
  parameter box but cannot re-fit C there.
- Spread was measured at spacings where it is at least 95 percent of its
  asymptote. Within a grid every cell is at the same spacing, so any residual
  plateau bias cancels when comparing cells.

## Files

Code:

- `point_density_convergence.py` -- added `--kp`, which did not exist (kp was
  pinned to the baseline). kp now also appears in the run output path;
  without it, two runs differing only in kp write to the same directory and
  the existing cleanup deletes the earlier one.

Results in `project/point_density_results/`:

- `{glacier}_mesh_convergence_ct.csv` -- the five-glacier sweep, 58 runs, at
  wf=2.5 kp=2.5. `recommend_n.merge_if_needed` only merges when the combined
  file is absent, so a split sweep merged before all parts land will silently
  keep a partial file.
- `gulkana_mesh_convergence_wf1.csv`, `_wf5.csv` -- error-vs-N at the ends of
  the wind factor range
- `{glacier}_mesh_convergence_pg_wf{WF}_kp{KP}.csv` -- the parameter grid

Batch: `batch_jobs/param_grid.sh`

Caution: `Output/point_density_test/` holds 51 `ct` directories dated 09-15
which are pre-wind-fix, at spacings the resweep did not repeat. The `ct` tag
does not distinguish them from the post-fix runs. Only the 09-16 directories
back the numbers here. `plot_convergence_diff.py` has a related trap: its
`SUFFIX_PRIORITY` falls back silently to an older results file when the one
it wants is missing, reporting the substitution only in a column of its
printout.
