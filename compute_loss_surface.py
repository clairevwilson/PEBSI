"""
Scores every (kp, wind_factor) combination in the loss_surface run and
saves the four glacier-wide log-losses to an .npz.

The run tiled the adaptive mesh once per combination, so combination c
occupies points [c*n_base:(c+1)*n_base]; each block is pulled out and
scored the way AD_optimize scores it -- March-referenced albedo deltas,
and mass balance compared only over the period the Hugonnet data
actually covers rather than the full model window.

Slow (it reads ~2 GB of daily output), so it is kept apart from
plot_loss_surface.py -- run this once, then iterate on the figure.

@author: clairevwilson
"""
import os
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from project.glacierwide_loss import Albedo, SnowlineMelt, MassBalance

OUT_ZARR = '/ocean/projects/ees260009p/cwilson4/Output/AD_optimize_0/output.zarr'
GRID_FN = '/ocean/projects/ees260009p/cwilson4/Output/AD_optimize/loss_surface_grid.npz'
LOSS_FN = '/ocean/projects/ees260009p/cwilson4/Output/AD_optimize/loss_surface_losses.npz'

ALBEDO_SIGMA = 0.0777

METRICS = ('albedo', 'mb', 'snow', 'melt')


def main():
    grid = np.load(GRID_FN, allow_pickle=True)
    kp_values = grid['kp']
    wf_values = grid['wind_factor']
    n_base = int(grid['n_base'])
    n_combos = int(grid['n_combos'])
    glacier = str(grid['glacier'])
    start, end = str(grid['start']), str(grid['end'])

    print(f'{glacier}: {n_base} points x {n_combos} combos, {start} to {end}',
          flush=True)

    ds = xr.open_zarr(OUT_ZARR)

    # measured sides are identical for every combination (all replicas sit at
    # the same coordinates), so the observation loaders are built once
    albedo = Albedo(glacier, use='s2')
    snowmelt = SnowlineMelt(glacier)
    mb = MassBalance(glacier, dates=(start, end))
    print(f'MB measured {mb.meas:+.3f} +/- {mb.sigma:.3f} m w.e. over '
          f'{mb.matched_start.date()} to {mb.matched_end.date()}', flush=True)

    losses = {k: np.full(n_combos, np.nan) for k in METRICS}
    losses['total'] = np.full(n_combos, np.nan)

    for c in range(n_combos):
        sub = ds.isel(point=slice(c * n_base, (c + 1) * n_base)).load()

        albedo.get_model_albedo(sub)
        albedo.get_deltas(method='march_mean')
        losses['albedo'][c] = albedo.log_loss(sigma=ALBEDO_SIGMA)

        snowmelt.get_model_snow(sub)
        losses['snow'][c], losses['melt'][c] = snowmelt.bernoulli_loss()

        # the model side is truncated to the period the observation actually
        # covers, not the full model window
        mod = (sub.sel(time=slice(mb.matched_start, mb.matched_end))
               .mass_balance.sum(dim='time').mean(dim='point').values)
        losses['mb'][c] = mb.log_loss(mod=float(mod))

        losses['total'][c] = sum(losses[k][c] for k in METRICS)

        if c % 10 == 0 or c == n_combos - 1:
            print(f'  [{c + 1}/{n_combos}] kp={grid["kp_of_point"][c * n_base]:.1f} '
                  f'wf={grid["wind_factor_of_point"][c * n_base]:.1f} '
                  f'total={losses["total"][c]:.3f}', flush=True)

    # combos were built kp-major, so index c = i_kp * n_wf + j_wf
    shape = (len(kp_values), len(wf_values))
    grids = {k: v.reshape(shape) for k, v in losses.items()}

    np.savez(LOSS_FN, kp=kp_values, wind_factor=wf_values,
             glacier=np.array(glacier), start=np.array(start),
             end=np.array(end), n_base=n_base, **grids)
    print(f'\nWrote {LOSS_FN}', flush=True)

    best = np.unravel_index(np.nanargmin(grids['total']), shape)
    print(f'Best total {grids["total"][best]:.3f} at '
          f'kp={kp_values[best[0]]:.1f}, wind_factor={wf_values[best[1]]:.1f}',
          flush=True)


if __name__ == '__main__':
    main()
