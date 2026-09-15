# Mesh convergence: how many points does mass balance need?

## Goal

Figure out at what point count (N) glacier-wide mass balance stops changing,
for Gulkana and Kennicott, using the patch-conforming mesh.

## What we saw

Running the mesh sweep (point_density_convergence.py) on both glaciers, mass
balance kept changing all the way out to the largest N we tried (~6k on
Gulkana, ~11k on Kennicott). Coarse meshes were consistently MORE NEGATIVE
than fine ones. That's a bigger N than expected for glaciers this size, so we
went looking for why.

## Ruling things out

- Area-weighting (elevation, slope, aspect, ice albedo, wind) is done by
  averaging every input over each point's Voronoi cell, not just sampling the
  pixel under the point. Checked that this preserves the area-weighted mean
  elevation to within 0.19 m across the whole sweep -- so the averaging
  itself isn't broken.
- Checked whether the coarse mesh was sampling the glacier's elevation
  distribution wrong (bad hypsometry). It wasn't -- applying the FINE mesh's
  own balance-vs-elevation relationship to each COARSE mesh's own points and
  weights reproduced the converged answer almost exactly. So the coarse mesh
  points are in the right places. The problem had to be that a point at a
  given elevation was getting a different (wrong) answer depending on mesh
  resolution, not that the wrong elevations were being sampled.
- That pointed at some per-point input being resolution-dependent. Binned
  balance by elevation and found the discrepancy was entirely below 1500 m
  (the ablation zone) -- above that, coarse and fine agreed.

## What we found

Ran a regression of modeled balance against every per-point input (elevation,
slope, aspect, ice albedo, sky-view, sunlit fraction, wind speed-up), coarse
mesh vs fine mesh, ablation zone only. Wind speed-up alone explained ~90% of
the gap.

Root cause: `get_wind_fields` in terrain.py averages each point's wind
speed-up over every finite pixel in its Voronoi cell, with no mask to
restrict that to on-glacier pixels. Checked the windmapper grid directly --
87% of it is off-glacier terrain, reaching up to 6.6 km past the margin.
`load_dem_info` (elevation/slope/aspect) already masks to ice for this exact
reason; the wind loader never got the same treatment.

Coarse meshes have bigger Voronoi cells, so they pull in more of that distant
off-glacier terrain when averaging. The glacier's tongue is the most
sheltered part of the whole domain, so any averaging over a wider area drags
its wind speed-up UP toward the surrounding mean. More wind -> more melt --
right where the ablation zone is, right why coarse meshes came out more
negative, and right why it shrinks as N grows (smaller cells = less
off-glacier contamination).

## Fix

Added an ice mask to `get_wind_fields`, same as `load_dem_info` already does.
Confirmed on Gulkana: at h=1200 the coarse-mesh ablation-zone wind bias went
from +14% to +0.5% (near zero), and mass balance moved from -3.98 to -3.66
m w.e. at that same resolution.

## Reran full sweeps with the fix

Full-sweep spread (max - min across all N tested) dropped:
- Gulkana: 0.436 -> 0.163 m w.e.
- Kennicott: 0.683 -> 0.190 m w.e.

## Where convergence actually sits now

Fit an exponential to mass balance vs N on the post-fix sweeps to find where
it flattens:

| Glacier   | Converged value | N for +/-0.02 m w.e./yr | N for +/-0.01 |
|-----------|-----------------|--------------------------|----------------|
| Gulkana   | -1.174 m w.e./yr | ~800                    | ~1,500         |
| Kennicott | +0.733 m w.e./yr | ~5,700                  | ~9,400         |

Gulkana's fit is solid (two independent halves of the sweep agree to 0.002).
Kennicott's is less certain (the two halves disagree by 0.008, about the
same size as what's left to converge) -- would want to extend that sweep
finer to pin it down for real.

## What's left over (not fixed, just found)

There's still a smaller residual drift after the wind fix. Traced part of it
(roughly half) to slope/aspect being computed straight off the 30 m DEM and
then cell-averaged -- so the terrain roughness the model sees also changes
with point spacing, same mechanism as wind but through solar geometry
instead of turbulent flux. Added an optional DEM-smoothing parameter
(`dem_smooth_m`, defaults to 0 = off) that would let you pin terrain
resolution independent of mesh spacing, but haven't turned it on or picked a
length -- there's no natural scale in the DEM to pick from (checked; slope
variance decays smoothly with no noise-vs-terrain knee). Left as future work.

## Files changed

- `pebsi/io/terrain.py` -- ice mask added to `get_wind_fields` (the fix);
  shading now cell-averaged instead of nearest-pixel; optional DEM smoothing
  in `load_dem_info`
- `pebsi/physics/energybalance.py` -- direct-beam shortwave now scales by
  sunlit fraction instead of a hard on/off shade gate (needed for the
  shading change above)
- `pebsi/defaults.py` -- new `dem_smooth_m` parameter, default 0
- `simulation.py` -- fixed float64 config being silently downgraded to
  float32 (jax_enable_x64 was being set after some modules already imported)
- `point_density_convergence.py` -- added `--tag` so two jobs can sweep the
  same glacier in parallel without overwriting each other's results file

Old (pre-fix) sweep results saved at
`project/point_density_results/{glacier}_mesh_convergence_prewindfix.csv`.
