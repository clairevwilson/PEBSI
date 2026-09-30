"""
Builds best.zarr: the loss_surface run's points, subset down to each
glacier's own best (kp, wind_factor) combination.

This is independent per glacier -- compute_loss_surface.py's
own 'best' is the combo that minimizes the *summed* total across every
calibration glacier (the calibration objective), which need not be any
single glacier's individual best. Reads loss_surface_losses.npz for each
glacier's own albedo/mb/snow/melt grids, sums them into that glacier's
total, and takes the argmin. That combo's points for the glacier are
pulled out of output.zarr (full variables, full time range) and
concatenated across glaciers into best.zarr.

@author: clairevwilson
"""
import numpy as np
import xarray as xr

from project.parameters import translate_rgi

ROOT = '/ocean/projects/ees260009p/cwilson4/Output/loss_surface_1/'
OUT_ZARR = ROOT + 'output.zarr'
LOSS_FN = ROOT + 'loss_surface_losses.npz'
BEST_FN = ROOT + 'best.zarr'

METRICS = ('albedo', 'mb', 'snow', 'melt')


def main():
    grids = np.load(LOSS_FN, allow_pickle=True)
    kp_values = grids['kp']
    wf_values = grids['wind_factor']
    n_base = int(grids['n_base'])
    glaciers = [str(g) for g in grids['glaciers']]
    shape = (len(kp_values), len(wf_values))

    ds_full = xr.open_zarr(OUT_ZARR, consolidated=False)
    rgiid_all = ds_full['rgiid'].values

    best_slices = []
    for g in glaciers:
        # combos were built kp-major, so index c = i_kp * n_wf + j_wf
        total_g = sum(grids[f'{g}_{k}'] for k in METRICS)
        i, j = np.unravel_index(np.nanargmin(total_g), shape)
        c = i * len(wf_values) + j
        kp, wf = float(kp_values[i]), float(wf_values[j])
        print(f'{g:<12} best kp={kp:.2f} wind_factor={wf:.2f} '
              f'(combo {c}) total={total_g[i, j]:.3f}', flush=True)

        combo_slice = slice(c * n_base, (c + 1) * n_base)
        rgi_id = translate_rgi[g]['6']
        point_mask = rgiid_all[combo_slice] == rgi_id
        combo_points = np.arange(combo_slice.start, combo_slice.stop)[point_mask]

        sub = ds_full.isel(point=combo_points)
        sub = sub.assign_coords(
            kp=('point', np.full(sub.sizes['point'], kp)),
            wind_factor=('point', np.full(sub.sizes['point'], wf)))
        best_slices.append(sub)

    best_ds = xr.concat(best_slices, dim='point')
    best_ds = best_ds.chunk({'point': -1})

    for var in best_ds.variables:
        best_ds[var].encoding.pop('chunks', None)

    best_ds.to_zarr(BEST_FN, mode='w')
    print(f'\nWrote {BEST_FN}', flush=True)


if __name__ == '__main__':
    main()
