"""
Scores every (kp, wind_factor) combination in the loss_surface run and
saves the four glacier-wide log-losses to an .npz, summed across every
calibration glacier -- the same convention AD_optimize's own total uses.

The run tiled the combined mesh (every glacier at once) once per
combination, so combination c occupies points [c*n_base:(c+1)*n_base];
each block still holds every glacier's points together, told apart by
the 'rgiid' field the output carries. Each glacier's slice is scored
against its own observations the way AD_optimize scores it --
March-referenced albedo deltas, and mass balance compared only over the
period the Hugonnet data actually covers, area-weighted by ds.weight
rather than a flat mean across points -- and the per-glacier losses are
summed into that combination's total.

Slow (it reads ~2 GB of daily output per glacier), so it is kept apart
from plot_loss_surface.py -- run this once, then iterate on the figure.

@author: clairevwilson
"""
import os
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from project.glacierwide_loss import Albedo, SnowlineMelt, MassBalance
from project.parameters import translate_rgi

OUT_ZARR = '/ocean/projects/ees260009p/cwilson4/Output/AD_optimize_1/output.zarr'
GRID_FN = '/ocean/projects/ees260009p/cwilson4/Output/AD_optimize/loss_surface_grid.npz'
LOSS_FN = '/ocean/projects/ees260009p/cwilson4/Output/AD_optimize_1/loss_surface_losses.npz'

ALBEDO_SIGMA = 0.0777

METRICS = ('albedo', 'mb', 'snow', 'melt')


def main():
    grid = np.load(GRID_FN, allow_pickle=True)
    kp_values = grid['kp']
    wf_values = grid['wind_factor']
    n_base = int(grid['n_base'])
    n_combos = int(grid['n_combos'])
    glaciers = [str(g) for g in grid['glaciers']]
    start, end = str(grid['start']), '2025-03-20' # str(grid['end'])
    rgi_ids = {g: translate_rgi[g]['6'] for g in glaciers}

    print(f'{len(glaciers)} glaciers x {n_combos} combos, {n_base} points '
          f'per combo, {start} to {end}', flush=True)

    ds = xr.open_zarr(OUT_ZARR)

    # measured sides are identical for every combination (all replicas sit
    # at the same coordinates), so the observation loaders are built once
    # per glacier
    obs = {}
    for g in glaciers:
        albedo = Albedo(g, use='s2')
        snowmelt = SnowlineMelt(g)
        mb = MassBalance(g, dates=(start, end))
        print(f'  {g:<12} MB measured {mb.meas:+.3f} +/- {mb.sigma:.3f} '
              f'm w.e. over {mb.matched_start.date()} to '
              f'{mb.matched_end.date()}', flush=True)
        obs[g] = dict(albedo=albedo, snowmelt=snowmelt, mb=mb)

    losses = {k: np.full(n_combos, np.nan) for k in METRICS}
    losses['total'] = np.full(n_combos, np.nan)
    per_glacier = {g: {k: np.full(n_combos, np.nan) for k in METRICS}
                   for g in glaciers}

    for c in range(n_combos):
        block = ds.isel(point=slice(c * n_base, (c + 1) * n_base)).load()
        rgiid = block.rgiid.values

        totals = {k: 0.0 for k in METRICS}
        for g in glaciers:
            sub = block.isel(point=(rgiid == rgi_ids[g]))
            o = obs[g]

            o['albedo'].get_model_albedo(sub)
            o['albedo'].get_deltas(method='march_mean')
            a_loss = o['albedo'].log_loss(sigma=ALBEDO_SIGMA)

            o['snowmelt'].get_model_snow(sub)
            s_loss, m_loss = o['snowmelt'].bernoulli_loss()

            # area-weighted by ds.weight rather than a flat mean across
            # points, and truncated to the period the observation covers
            # o['mb'].get_model_mb(sub)
            # mb_loss = o['mb'].log_loss(mod=o['mb'].mod)
            mb_loss = 0

            vals = dict(albedo=a_loss, snow=s_loss, melt=m_loss, mb=mb_loss)
            for k, v in vals.items():
                per_glacier[g][k][c] = float(v)
                totals[k] += float(v)

        for k in METRICS:
            losses[k][c] = totals[k]
        losses['total'][c] = sum(totals.values())

        if c % 10 == 0 or c == n_combos - 1:
            print(f'  [{c + 1}/{n_combos}] kp={grid["kp_of_point"][c * n_base]:.1f} '
                  f'wf={grid["wind_factor_of_point"][c * n_base]:.1f} '
                  f'total={losses["total"][c]:.3f}', flush=True)

    # combos were built kp-major, so index c = i_kp * n_wf + j_wf
    shape = (len(kp_values), len(wf_values))
    grids = {k: v.reshape(shape) for k, v in losses.items()}
    per_glacier_grids = {f'{g}_{k}': v.reshape(shape)
                         for g, kv in per_glacier.items() for k, v in kv.items()}

    np.savez(LOSS_FN, kp=kp_values, wind_factor=wf_values,
             glaciers=np.array(glaciers), start=np.array(start),
             end=np.array(end), n_base=n_base, **grids, **per_glacier_grids)
    print(f'\nWrote {LOSS_FN}', flush=True)

    best = np.unravel_index(np.nanargmin(grids['total']), shape)
    print(f'Best total {grids["total"][best]:.3f} at '
          f'kp={kp_values[best[0]]:.1f}, wind_factor={wf_values[best[1]]:.1f}',
          flush=True)


if __name__ == '__main__':
    main()
