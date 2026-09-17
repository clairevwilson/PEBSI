# Point density: how many points does each glacier need?

## The question

Glacier-wide mass balance is computed from N independent 1-D columns placed on
a patch-conforming mesh. We need to choose N per glacier, tight enough that
the answer is trustworthy and loose enough to calibrate `wind_factor` and `kp`
without wasting a GPU-month.

The question went through three framings. The first two were wrong and the
third is what this document is about:

1. "At what N does mass balance stop changing?" -- chased a drift that turned
   out to be a bug, not a resolution effect.
2. "What is the true converged mass balance, and where in the sweep do we
   reach it?" -- fit an asymptote and read off N. Superseded: with bugs fixed, what remains is not a drift to extrapolate but random sampling
   scatter, and the exponential was fitting noise.
3. "How many points does each glacier need for the error to stay inside a
   stated tolerance, with stated confidence, anywhere in the calibration
   parameter box?" -- answerable, and answered below.

## What we tried and ruled out

Mass balance kept changing out to the largest N tried (~6k Gulkana, ~11k
Kennicott), and coarse meshes were consistently more negative than fine ones.
Consistently, not randomly, which indicated this was a bug rather than noise.

Ruled out first: that the Voronoi cell-averaging of inputs was broken (it
preserves area-weighted mean elevation to 0.19 m across the sweep), and that
coarse meshes sampled the hypsometry badly (applying the fine mesh's own
balance-vs-elevation relation to each coarse mesh's own points and weights
reproduced the converged answer). So the points were in the right places, and
a point at a given elevation was getting a different answer depending on
resolution. Binning by elevation put the whole discrepancy below 1500 m, in
the ablation zone.

Regressing modeled balance against every per-point input, coarse vs fine,
ablation zone only: wind speed-up explained ~90% of the gap. Root cause was
`get_wind_fields` in `terrain.py`, which averaged each point's wind speed-up
over every finite pixel in its Voronoi cell with no ice mask -- 87% of the
windmapper grid is off-glacier, reaching 6.6 km past the margin, and
`load_dem_info` already masked to ice for exactly this reason. Coarse meshes
have bigger cells, so they pull in more distant terrain; the tongue is the
most sheltered part of the domain, so averaging drags its speed-up up, giving
more melt precisely in the ablation zone and shrinking as cells get smaller.

Adding the ice mask took Gulkana's h=1200 ablation-zone wind bias from +14%
to +0.5%, and dropped full-sweep spread from 0.436 to 0.163 m w.e. on Gulkana
and 0.683 to 0.190 on Kennicott.

A smaller input-resolution effect remains, roughly half of the leftover drift,
from slope and aspect being taken off the 30 m DEM and then cell-averaged --
the same mechanism as wind but through solar geometry. `dem_smooth_m` exists
to pin terrain resolution independent of mesh spacing but is off by default
and no length has been chosen; slope variance decays smoothly with no knee, so
there is no natural scale to pick.

## Two error sources, with different mathematics

**Input-resolution bias.** Systematic, monotone in h, discretization-like. The
wind bug was this, and the residual slope/aspect effect still is. This is the
only component for which "refining converges on the answer" is the right
mental model.

**Sampling error.** Each point is an independent column; there is no flux
between them. Glacier-wide balance is therefore a weighted survey estimate of
a spatial field, not the solution of a coupled system. Meshes at different h
are not nested -- the h=900 point set is not a refinement of the h=1200 set
but a different draw of locations -- so the error is a random variable whose
distribution depends on N, not a function of h. Convergence holds in mean
square, not for any individual run, and a sequence of single runs at
decreasing h is not a converging sequence but a set of near-independent draws.

With the wind bug fixed, sampling error dominates. Everything below is about
it.

## The sampling model

The estimator is the area-weighted mean

    B = sum_i w_i b_i

with weights `w_i` (Voronoi cell area, clipped to the outline, summing to 1)
and `b_i` the point's annual mass balance. Define

    σ = standard deviation of b_i across the glacier [cm w.e./yr]

σ is a standard deviation, not a range. It is large: Gulkana's terminus melts
at -700 cm/yr while its accumulation area gains +209, giving σ = 235 cm/yr at
wf=2.5.

The model is that the error of B is random with standard deviation

    s(N) = C σ / sqrt(N)

fit to 51 (glacier, resolution) error measurements from the five-glacier
post-fix sweep, taking each sweep's three finest runs as the reference and
excluding those three from the fit. Fitting `s = C σ N^-p` with both free:

    p = 0.457      C = 0.0603

### Why the square root is measured, not assumed

- p came out 0.457 from the fit, consistent with the 0.5 of survey sampling.
- The rms normalized error is flat between the low-N and high-N halves of the
  data (0.0526 and 0.0588), which is the same statement independently.
- Control: drawing *random* point subsets instead of meshes gives error
  σ/sqrt(N) with no prefactor, matching within 10% on all five glaciers. So
  the sampling framework is verified separately from the mesh, and C is
  attributable entirely to the mesh layout.
- The mean normalized error is +0.019, i.e. unbiased.
- Residual scatter is 0.90 in log space. A *perfect* model of s, applied to
  single draws of a normal variable, still leaves log-scatter of pi/sqrt(8) =
  1.11. Being at or below that floor means there is no systematic misfit left
  to extract, which is also why no alternative normalization separated from
  the others (below).

### What C is

C = 0.0603 = 1/16.6. Random sampling gives σ/sqrt(N) and the mesh gives
C σ/sqrt(N), so the mesh reaches the accuracy of a random sample of 1/C^2, or
about 275 times more points. That gain is the even, area-weighted layout
capturing the elevation gradient for free: elevation explains 96% of per-point
variance, and any space-filling mesh covers the elevation range in roughly
area-proportional numbers without being told to.

## Error bounds

s(N) is a standard deviation, not a bound, so it needs a distribution. B is a
weighted average over hundreds to thousands of points, so its error is
approximately normal by the central limit theorem, giving

    P(|error| <= z s(N)) = 2 Phi(z) - 1

with z = 1.96 for 95% and z = 2.576 for 99%. Requiring `z s(N) <= tol`:

    N = (z C σ / tol)^2

Three properties matter in practice:

- tol enters squared. Halving the tolerance quadruples N.
- σ enters squared. This is what makes the parameter box expensive.
- The bound is per mesh, not per year. The error is a fixed offset for a given
  mesh: same points, same elevations, same bias at every timestep. It does not
  average down over a long run, it accumulates linearly. So a 3 cm/yr bound is
  0.6 m w.e. over 20 years, and if the error were instead independent year to
  year you would get a sqrt(20) reduction and could afford a looser tolerance.

### The bounds are calibrated

The check that matters is whether the stated probabilities match what the
sweep did. Taking the resolutions that looked anomalous:

| glacier | N | s(N) | P(pass 2 cm) | observed | |
| --- | --- | --- | --- | --- | --- |
| wolverine | 45 | 2.46 | 58% | +3.39 | fail |
| wolverine | 93 | 1.71 | 76% | +2.14 | fail |
| wolverine | 166 | 1.28 | 88% | +1.57 | pass |
| kennicott | 206 | 1.53 | 81% | +1.96 | pass |
| kahiltna | 310 | 1.68 | 77% | +2.88 | fail |
| gulkana | 48 | 1.90 | 71% | +0.77 | pass |
| lemon_creek | 19 | 1.77 | 74% | +0.49 | pass |

Wolverine reading +3.4 at 45 points, +0.9 at 65 and +1.6 at 166 is not a
resolution effect; at 45 points it had a 42% chance of missing. Every
resolution that a "where does it stop changing" reading would have called
converged sits in a 58 to 88 percent pass band, which is why that framing
could not work: a single run landing inside tolerance is a coin flip, not a
measurement of convergence.

## Choice of tolerance

3 cm/yr, which is 0.6 m w.e. over 20 years and sits under the Hugonnet
minimum error. 5 cm/yr aggregates to 1.0 m and would sit on it. 2 cm/yr costs
2.2 times the points (4288 vs 1906 across the five glaciers) for precision the
observations cannot use.

## Why σ, and not some other normalization

Eight models were fit: σ, elevation-detrended σ (quantile bins and cubic), and
area-weighted elevation standard deviation, each over N and over Kish
effective sample size. All landed at log-scatter 0.88 to 0.98 against the 1.11
floor, i.e. statistically indistinguishable, because the single-draw noise
dominates. The choice therefore rests on which quantity is measurable and
stable, not on fit quality:

- **Elevation-detrended σ** is theoretically the right scale, since the mesh
  captures the elevation gradient for free and only the residual has to be
  averaged down. Unusable in practice: it does not plateau with resolution
  (Wolverine reads 20 at h=2000 and 73 at h=70, so a cheap coarse run
  underestimates it 3.65x and hands back too coarse a mesh), and it depends on
  the detrending method (Kahiltna is 126 with quantile bins, 246 with a cubic).
- **Kish effective sample size**, `N_eff = 1/sum(w_i^2)`, correctly handles
  the uneven Voronoi weights, which are genuinely uneven at coarse spacing
  (`N_eff/N` runs 0.47 to 0.99). It improved the fit by 2%.
- **Elevation standard deviation** needs no simulation at all, straight from
  the DEM, and fit as well as anything. But it over-resolves by 2.3x overall,
  because σ/elev_sd runs 0.43 (Kennicott, continental) to 1.29 (Wolverine,
  maritime). Usable as a no-pilot fallback with a climate class, not as the
  primary.
- **Plain σ** plateaus acceptably (coarse/finest runs 0.76 to 1.07), has no
  free choices in it, and pushes the whole stratification question into a
  single fitted constant.

## Does it survive calibration?

45 runs, a 3x3 grid in (`wind_factor`, `kp`) over {1, 2.5, 5}, one spacing per
glacier. The centre cell repeats the sweep's parameters as a control and
reproduced the already-measured σ to within 0.2% on all five glaciers.

- **σ moves a lot.** Gulkana goes 96.7 at wf=1, 233 at wf=2.5, 413.8 at wf=5,
  scaling as wf^0.90. A point count cannot be carried between parameter sets;
  σ has to be re-measured. That is cheap, because σ is a standard deviation
  over hundreds of points rather than a single-draw error, so one run per
  parameter set suffices where a sweep would not.
- **C does not drift with wind factor.** Elevation R^2 is 0.959, 0.965, 0.962
  at wf = 1, 2.5, 5, and the residual fraction is flat at 0.203, 0.186, 0.194.
  Wind scales the field while preserving its elevation structure, so the mesh
  keeps its advantage. This was the main threat to the rule transferring, and
  it does not materialize.
- **kp matters, and dominates wind factor on Kahiltna.** At wf=2.5, taking kp
  from 1 to 5 moves σ from 412 to 658 (+60%); at kp=5, taking wf from 1 to 5
  moves it from 603 to 624 (+3%).
- **The worst cell is not at a corner** of the box for four of the five
  glaciers. Sampling corners only understates the worst case by 3 to 8
  percent. Wolverine and Lemon Creek are also non-monotone in wind factor
  (Wolverine at kp=1 reads 143, 327, 237 as wf goes 1, 2.5, 5) against a ~3
  percent measurement noise floor, so that structure is real.

So each glacier is sized at its own worst cell over the box.

## The equation we use

    N = (1.96 C σ_max / tol)^2        C = 0.0603

    at tol = 3 cm/yr:   N = (σ_max / 25.4)^2

where σ_max is the largest σ over the calibration parameter box. At 3 cm/yr
and 95% confidence:

| glacier | σ_max | N | h |
| --- | --- | --- | --- |
| kahiltna | 658.2 | 672 | ~2020 m |
| kennicott | 630.1 | 616 | ~1330 m |
| gulkana | 411.3 | 263 | ~450 m |
| wolverine | 385.6 | 231 | ~420 m |
| lemon_creek | 282.3 | 124 | ~480 m |

Total 1906 points. At 99% confidence it is 3292.

N is the target, not h. The points-per-spacing relation is not a constant:
`N h^2 / A` runs from 2.2 at fine spacing to about 11 at coarse, because coarse
meshes spend points on the outline. Get h from `mesh_preview.py`, which costs
no simulation time, rather than from a formula.

For a glacier not in this set: one run at moderate spacing gives σ, then apply
the equation. σ is within ~5% of its asymptote by h = 700 to 850 m on the
glaciers where it matters.

## Limits

- C is fit from five Alaskan glaciers over a single 3-year period. Parameter
  transfer was tested on Gulkana only.
- The eight error measurements from the wf=1 and wf=5 sweeps pool to C = 0.080
  against 0.0559 at the centre. That is 1.2 sigma on eight points and
  non-monotone in wind factor, so it reads as noise rather than a trend; part
  of it is also those tests using a single finest run as the reference, which
  inflates measured errors by up to 16% at the fine end. Refitting on all 51
  points gives the 0.0603 used above.
- Assumes errors are normal and independent between meshes. The retrodiction
  table is consistent with that but does not prove it.
- The grid used one spacing per glacier, so it maps σ across the parameter box
  but cannot re-fit C there.
- The residual slope/aspect input-resolution bias is not represented in this
  framework at all. It is systematic, so it does not shrink with N, and the
  tolerance above does not cover it.
- σ was measured at spacings where it is at least 95% of its asymptote. Within
  a grid every cell shares a spacing, so any residual bias cancels when
  comparing cells.

## Files

Code:

- `pebsi/io/terrain.py` -- ice mask in `get_wind_fields` (the fix); shading
  cell-averaged instead of nearest-pixel; optional DEM smoothing in
  `load_dem_info`
- `pebsi/physics/energybalance.py` -- direct-beam shortwave scales by sunlit
  fraction instead of a hard shade gate
- `pebsi/defaults.py` -- `dem_smooth_m`, default 0
- `simulation.py` -- float64 config no longer silently downgraded to float32
- `point_density_convergence.py` -- `--tag` so parallel jobs do not overwrite
  each other's results; `--kp`, which did not exist (kp was pinned to the
  baseline). kp now also appears in the run output path; without it two runs
  differing only in kp write to the same directory and the existing cleanup
  deletes the earlier one.

Results in `project/point_density_results/`:

- `{glacier}_mesh_convergence_ct.csv` -- five-glacier post-fix sweep, 58 runs,
  wf=2.5 kp=2.5
- `{glacier}_mesh_convergence_prewindfix.csv` -- pre-fix, kept for comparison
- `gulkana_mesh_convergence_wf1.csv`, `_wf5.csv` -- error vs N at the ends of
  the wind factor range
- `{glacier}_mesh_convergence_pg_wf{WF}_kp{KP}.csv` -- the parameter grid

Batch: `batch_jobs/mesh_convergence.sh`, `batch_jobs/param_grid.sh`

Three traps worth knowing about:

- `Output/point_density_test/` holds 51 `ct` directories dated 09-15 that are
  pre-wind-fix, at spacings the resweep did not repeat. The `ct` tag does not
  distinguish them from post-fix runs; only the 09-16 directories back the
  numbers here.
- `recommend_n.merge_if_needed` merges split sweep parts only when the combined
  file is absent, so a merge run before all parts land leaves a partial file
  that will never be corrected.
- `plot_convergence_diff.py` defines "converged" as the mean of a stretch of
  points selected for having spread below the tolerance, then plots deviation
  from that mean against a band drawn at the same tolerance -- points in the
  stretch cannot fall outside the band. Its `SUFFIX_PRIORITY` also falls back
  silently to an older results file, reporting the substitution only in one
  column of its printout.

`recommend_n.py` and `plot_convergence_diff.py` both still implement framing 2
above and will keep returning the coarsest run in a sweep whenever the sweep's
spread is smaller than the tolerance.
