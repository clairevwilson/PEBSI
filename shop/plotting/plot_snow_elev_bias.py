"""
Checks whether the wet extent misfit at each glacier's best
(kp, wind_factor) from the loss surface is organized by elevation.

For each glacier the best combination is the argmin of the sum of its
per-glacier loss grids (the same x plot_loss_surface.py draws). That
combination's points are pulled from the run's output.zarr and matched
against the SAR wet mask exactly as compute_loss_surface.py does, then
wet frequency is compared by elevation band, model vs observed.

@author: clairevwilson
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from project.glacierwide_loss import Wet
from project.parameters import translate_rgi

RUN_DIR = '/ocean/projects/ees260009p/cwilson4/Output/loss_surface_1/'
METRICS = ('albedo', 'mb', 'wet')
BAND = 100


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', default=RUN_DIR)
    return p.parse_args()


def band_stats(elev, mod, meas, valid, edges):
    idx = np.digitize(elev, edges) - 1
    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        va = valid[:, m]
        if not va.any():
            continue
        mo, me = mod[:, m][va], meas[:, m][va]
        rows.append(dict(z=0.5 * (edges[b] + edges[b + 1]), n=int(m.sum()),
                         obs=me.mean(), mod=mo.mean(),
                         false_pos=(mo & ~me).mean(), false_neg=(~mo & me).mean()))
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    grid = np.load(os.path.join(args.run_dir, 'loss_surface_grid.npz'), allow_pickle=True)
    loss = np.load(os.path.join(args.run_dir, 'loss_surface_losses.npz'), allow_pickle=True)
    kp_values, wf_values = loss['kp'], loss['wind_factor']
    n_base = int(grid['n_base'])
    glaciers = [str(g) for g in loss['glaciers']]

    ds = xr.open_zarr(os.path.join(args.run_dir, 'output.zarr'), consolidated=False)

    fig, axes = plt.subplots(1, len(glaciers), figsize=(4 * len(glaciers), 4), squeeze=False)
    for col, g in enumerate(glaciers):
        tot = sum(loss[f'{g}_{k}'] for k in METRICS)
        i, j = np.unravel_index(np.nanargmin(tot), tot.shape)
        c = i * len(wf_values) + j
        block = ds.isel(point=slice(c * n_base, (c + 1) * n_base))
        rgiid = block['rgiid'].values
        gmask = rgiid == translate_rgi[g]['6']

        wet = Wet(g)
        tmin = wet.ds_meas_wet.time.values.min()
        tmax = wet.ds_meas_wet.time.values.max()
        pad = np.timedelta64(5, 'D')
        sub = (block[['lon', 'lat', 'elev', 'total_water']]
               .isel(point=gmask).sel(time=slice(tmin - pad, tmax + pad)).load())
        wet.get_model_wet(sub)
        elev = sub['elev'].values
        valid = ~np.isnan(wet.meas_wet)
        meas_wet = wet.meas_wet == 1
        edges = np.arange(np.floor(elev.min() / BAND) * BAND, elev.max() + BAND, BAND)

        stats = band_stats(elev, wet.mod_wet, meas_wet, valid, edges)

        print(f'\n===== {g}: kp={kp_values[i]} wf={wf_values[j]}, {gmask.sum()} points, '
              f'{elev.min():.0f}-{elev.max():.0f} m, {len(wet.time)} SAR dates', flush=True)
        print(f'  wet: obs freq {meas_wet[valid].mean():.2f}, model freq {wet.mod_wet[valid].mean():.2f}, '
              f'model-wet/obs-dry {(wet.mod_wet & ~meas_wet)[valid].mean():.2f}, '
              f'model-dry/obs-wet {(~wet.mod_wet & meas_wet)[valid].mean():.2f}')
        print('  by band (wet freq obs/mod):')
        for r in stats.to_dict('records'):
            print(f'    {r["z"]:5.0f} m n={r["n"]:4d}  wet {r["obs"]:.2f}/{r["mod"]:.2f}')

        ax = axes[0, col]
        ax.plot(stats.obs, stats.z, 'k-o', ms=3, label='SAR')
        ax.plot(stats['mod'], stats.z, 'C0-o', ms=3, label='model')
        ax.set_title(f'{g}\nkp={kp_values[i]} wf={wf_values[j]}')
        ax.set_xlabel('wet frequency')
        ax.set_ylabel('elevation [m]')
        ax.set_xlim(0, 1)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig_fn = os.path.join(args.run_dir, 'wet_elev_bias.png')
    fig.savefig(fig_fn, dpi=140)
    print(f'\nWrote {fig_fn}', flush=True)


if __name__ == '__main__':
    main()
