# Point density: how many points does each glacier need?

## The question

Glacier-wide mass balance is computed from N independent points on a
patch-conforming mesh. We need to choose N per glacier, tight enough to
trust and loose enough to calibrate `wind_factor` and `kp` without wasting a
GPU-month.

We went through three framings of the question. The first two didn't work:

1. "At what N does mass balance stop changing?" — chased a drift that turned
   out to be a data-sampling bug, not a resolution effect.
2. "What's the true converged value, and where does the sweep reach it?" —
   once the bug was fixed, what was left wasn't a drift to extrapolate but
   random scatter, so fitting a curve to it was fitting noise.
3. "How many points does each glacier need for the error to stay inside a
   stated tolerance, with stated confidence, anywhere in the calibration
   parameter box?" — this is what the rest of the doc answers.

## What we tried

Coarse meshes were consistently more negative than fine ones, well past any
reasonable resolution. That consistency (not randomness) said bug, not noise.
Fix: generate the mesh, use the Voronoi cells to crop every per-point input
to the glacier's true area, and avoid missing-data artifacts by filling any
point that lands on a gap in the ice albedo or wind speedup data with its
nearest valid ice pixel. Once that was in place, the systematic drift
disappeared and what remained was random point-to-point scatter.

## Two kinds of error, two kinds of math

**Input bias** — systematic, gets smaller as the mesh refines. This is what
the fix above removed. "Finer mesh converges on the truth" is the right
mental model for this one.

**Sampling scatter** — each point is an independent column with no
interaction between points, so glacier-wide balance is a weighted survey
estimate of a spatial field, not the solution of a coupled system. A mesh at
one spacing is not a refinement of a mesh at another spacing — it's a
different, mostly-independent set of locations. So the error at a given N is
a random variable, not a fixed function of spacing, and it only converges in
the statistical sense (its average size shrinks), not run by run.

With the fix in, sampling scatter is what's left, and it's what the rest of
this document is about.

## The model

Let σ be the standard deviation of annual point mass balance across a
glacier, in cm w.e./yr. It's on the order of 100 cm as a result of negative
net mass balance in the ablation area and positive net mass balance in the
accumulation area.

s(N) is the typical size of the error between an N-point mesh's answer and
the glacier's true mass balance — how much the answer would bounce around if
you regenerated the mesh at that N many times. It behaves as:

    s(N) = C · σ / √N

### How this was fit

We don't have access to the true mass balance, so we built a stand-in for it
per glacier: the average of that glacier's three finest sweep runs (three
different meshes near the fine end). Every other resolution's error was
measured against that stand-in, giving 51 (glacier, resolution)
measurements across the five-glacier sweep. Those three finest runs
themselves were excluded from the fit — they define the reference, so they
can't also be a measurement of it. Fitting `s = C · σ / √N` to the 51 points
gives:

    C = 0.0603

### Is the reference good enough?

Averaging three finest runs only cancels *random* mesh-to-mesh error, and
only as well as the model's own math says it should: using the fitted
equation to estimate the reference's own leftover noise gives 0.11 to 0.17
cm w.e./yr on all five glaciers — under 6% of the 3 cm/yr tolerance we use
below. That's a self-consistency check, not independent proof, since it uses
the same equation being validated.

What it can't rule out is a *systematic* bias shared by all three finest
runs alike — averaging doesn't cancel that. One such bias is known and still
open: the residual slope/aspect resolution effect noted above, where terrain
roughness still changes somewhat with mesh spacing. If any of that leaked
into the finest runs, it would sit in the reference too, undetected by this
check.

### Why this form is trustworthy

- Fitting `s = C · σ · N⁻ᵖ` with both C and p free returns p ≈ 0.46 — the
  square root, the textbook rate for averaging down a random error. That's
  measured and agrees with theory.
- Drawing points at random instead of using the mesh gives error ≈ σ/√N with
  no prefactor, matching the sweep to within 10% on all five glaciers. So the
  √N law is confirmed independent of the mesh, and C is cleanly attributable
  to the mesh's layout.
- The leftover scatter in the fit is close to the theoretical floor for a
  perfect model applied to single random draws — there's no further pattern
  left to explain.

C = 1/16.6. In plain terms: laying points out evenly across the glacier
(instead of picking them at random) buys about the same accuracy as a random
sample 275 times larger, because elevation alone already explains ~96% of the
scatter and an even mesh captures the elevation gradient automatically.

## Turning that into a point count

s(N) is a typical error size, not a guarantee — individual meshes are random
draws. Assuming the error is close to normally distributed (it's a weighted
average over hundreds of points, so this is a safe assumption), 95% of runs
land within 1.96·s(N), giving:

    N = (1.96 · C · σ / tol)²

Two things worth knowing about this:

- **tol enters squared.** Halving the tolerance quadruples N.
- **The error doesn't average out over a run.** It's a fixed offset for a
  given mesh — same points, same bias, every year — so it accumulates
  linearly, not with the square root of years. That's why the tolerance choice below is in
  cm/yr but is really about the multi-year total.

### Sanity check

We checked this against the sweep directly: taking the resolutions that
"looked" converged by the old drift-based standard and asking the model what
probability of passing 2 cm/yr it assigns them — they land in a 58–88% pass
band. In other words, those apparent convergences were mostly coin flips
landing heads, not real convergence. That's the direct evidence that framing
1 and 2 above couldn't have worked, and that the probability-based framing is 
the right one.

## Choosing the tolerance

We're using **3 cm/yr**, which is 0.6 m w.e. over 20 years — comfortably
under the Hugonnet measurement floor for all five glaciers in this expeirment
(minimum 0.99 m w.e. error over 20 years at Kennicott). 
5 cm/yr (1.0 m over 20 years) would sit too close to the Hugonnet margins. 
2 cm/yr would cost 2.2x more points for precision the observations can't 
use anyway.

## Why σ, and not spacing or area

Neither spacing nor area transfers between glaciers: the same spacing buys
very different accuracy on different glaciers, and area alone doesn't predict
how much a glacier needs (Kahiltna is 29x Gulkana's area, but what actually
sets its point requirement is its σ). σ is the quantity that the sampling
error is actually driven by, and it's the only thing the calibration
parameters can change — so the rule adjusts itself automatically when
`wind_factor` or `kp` move.

We also tried a few refinements to σ and none earned their complexity:
detrending against elevation first is the theoretically "purer" signal, but
it doesn't stabilize with mesh resolution (a cheap coarse run underestimates
it several-fold) and it depends on how you detrend. Weighting by effective
sample size (to account for uneven point weights at coarse spacing) helped
by only ~2%. A DEM-only elevation-spread proxy needs no simulation at all but
over- or under-resolves by more than 2x depending on the glacier's climate.
Plain σ, measured directly from a real run, is simple and stable enough to
build the rule on.

## Does it survive calibration?

We ran a 3x3 grid over `wind_factor` and `kp` at {1, 2.5, 5} each, one
spacing per glacier, to see whether σ (and therefore the point count) holds
up across the range we're calibrating over.

- **σ moves a lot with the parameters.** On Gulkana it goes from about 97 at
  wf=1 to 233 at wf=2.5 to 414 at wf=5. So the point count from one parameter
  set can't just be reused at another — σ has to be re-measured. That's
  cheap: it only takes one run per parameter combination, not a whole sweep.
- **C itself doesn't drift with wind factor** — elevation explains about the
  same fraction of the scatter (~96%) at every wind factor we tried, so the
  mesh's advantage holds up. This was our main worry going in, and it didn't
  materialize.
- **kp matters, and on Kahiltna it matters more than wind factor.** Moving kp
  from 1 to 5 raised σ by 60%; moving wind factor over the same range only
  raised it 3%.
- **The worst case usually isn't at a corner of the calibration box.** For
  four of five glaciers the highest σ showed up in the interior of the grid,
  not at an extreme combination. So we sized each glacier off its actual
  worst cell, not off the corners.

## The equation we're using

    N = (1.96 · C · σ_max / tol)²          C = 0.0603, tol = 3 cm/yr

where σ_max is the largest σ measured anywhere in the calibration box for
that glacier. This simplifies to:

    N = (σ_max / 25.4)²

| glacier | σ_max (cm/yr) | N | h |
| --- | --- | --- | --- |
| kahiltna | 658 | 672 | ~2020 m |
| kennicott | 630 | 616 | ~1330 m |
| gulkana | 411 | 263 | ~450 m |
| wolverine | 386 | 231 | ~420 m |
| lemon_creek | 282 | 124 | ~480 m |

**Total: 1906 points.** (3292 at 99% confidence instead of 95%, if we want
extra margin.)

N is the number to trust; h is a starting point to confirm with a mesh
preview before committing, since points-per-spacing isn't a fixed ratio —
coarse meshes spend more of their points along the outline than fine ones do.

## Limits

- Fit from five Alaskan glaciers over one 3-year period; parameter transfer
  was stress-tested on Gulkana specifically, not all five.
- Assumes the error is normally distributed and independent between meshes.
  Consistent with everything we checked, not separately proven.
- The calibration grid used one spacing per glacier, so it maps how σ moves
  across the parameter box but doesn't re-derive C at every point in it.

## Files

Results: `project/point_density_results/{glacier}_mesh_convergence_ct.csv`
(five-glacier post-fix sweep) and `{glacier}_mesh_convergence_pg_wf{WF}_kp{KP}.csv`
(the parameter grid).

Batch scripts: `batch_jobs/mesh_convergence.sh`, `batch_jobs/param_grid.sh`.

Everything that generated and debugged this (the convergence sweeps,
`check_*`/`debug_*` scripts, old result CSVs, old plots) has been deleted
from the working tree. It's all still in git history if it's ever needed
again.
