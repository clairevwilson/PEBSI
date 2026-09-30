"""
Checks whether the snow and melt extent misfit at each glacier's best
(kp, wind_factor) from the loss surface is organized by elevation.

For each glacier the best combination is the argmin of the sum of its
four per-glacier loss grids (the same x plot_loss_surface.py draws).
That combination's points are pulled from the run's output.zarr and
matched against the SAR snow / melting masks exactly as
compute_loss_surface.py does, then:
  - snow and melt frequency by elevation band, model vs observed
  - per-date snowline, fit as the elevation threshold that best
    separates snow (above) from not-snow (below), model minus observed

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

from project.glacierwide_loss import SnowlineMelt
from project.parameters import translate_rgi

RUN_DIR = '/ocean/projects/ees260009p/cwilson4/Output/loss_surface_1/'
METRICS = ('albedo', 'mb', 'snow', 'melt')
BAND = 100


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', default=RUN_DIR)
    return p.parse_args()


def step_threshold(elev, is_snow):
    """
    Elevation h minimizing the misclassification of 'snow iff elev > h'.
    Returns nan when the best split sits at either end (all snow or no
    snow), since there is no snowline on the glacier that day.
    """
    order = np.argsort(elev)
    e, s = elev[order], is_snow[order].astype(float)
    # errors if the split sits just below index k: snow below + bare above
    snow_below = np.concatenate([[0], np.cumsum(s)])
    bare_above = np.concatenate([[0], np.cumsum((1 - s)[::-1])])[::-1]
    k = np.argmin(snow_below + bare_above)
    if k == 0 or k == len(e):
        return np.nan
    return 0.5 * (e[k - 1] + e[k])


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

    fig, axes = plt.subplots(3, len(glaciers), figsize=(4 * len(glaciers), 11))
    for col, g in enumerate(glaciers):
        tot = sum(loss[f'{g}_{k}'] for k in METRICS)
        i, j = np.unravel_index(np.nanargmin(tot), tot.shape)
        c = i * len(wf_values) + j
        block = ds.isel(point=slice(c * n_base, (c + 1) * n_base))
        rgiid = block['rgiid'].values
        gmask = rgiid == translate_rgi[g]['6']

        sm = SnowlineMelt(g)
        tmin = min(sm.ds_meas_snow.time.values.min(), sm.ds_meas_melt.time.values.min())
        tmax = max(sm.ds_meas_snow.time.values.max(), sm.ds_meas_melt.time.values.max())
        pad = np.timedelta64(5, 'D')
        sub = (block[['lon', 'lat', 'elev', 'surftype', 'total_water']]
               .isel(point=gmask).sel(time=slice(tmin - pad, tmax + pad)).load())
        sm.get_model_snow(sub)
        elev = sub['elev'].values
        valid_snow = ~np.isnan(sm.meas_snow)
        valid_melt = ~np.isnan(sm.meas_melt)
        meas_snow = sm.meas_snow == 1
        meas_melt = sm.meas_melt == 1
        edges = np.arange(np.floor(elev.min() / BAND) * BAND, elev.max() + BAND, BAND)

        snow = band_stats(elev, sm.mod_snow, meas_snow, valid_snow, edges)
        melt = band_stats(elev, sm.mod_melt, meas_melt, valid_melt, edges)

        # snowline on each date where both model and SAR have one on the glacier
        month = pd.DatetimeIndex(sm.time).month
        # both snowlines use only the points SAR retrieved on that date
        sl = np.array([[step_threshold(elev[v], sm.mod_snow[t][v]), step_threshold(elev[v], meas_snow[t][v])]
                       for t, v in enumerate(valid_snow)])
        ok = ~np.isnan(sl).any(axis=1)
        diff = sl[ok, 0] - sl[ok, 1]
        summer = np.isin(month[ok], [6, 7, 8, 9])

        print(f'\n===== {g}: kp={kp_values[i]} wf={wf_values[j]}, {gmask.sum()} points, '
              f'{elev.min():.0f}-{elev.max():.0f} m, {len(sm.time)} SAR dates', flush=True)
        vs, vm = valid_snow, valid_melt
        print(f'  snow: obs freq {meas_snow[vs].mean():.2f}, model freq {sm.mod_snow[vs].mean():.2f}, '
              f'model-snow/obs-bare {(sm.mod_snow & ~meas_snow)[vs].mean():.2f}, '
              f'model-bare/obs-snow {(~sm.mod_snow & meas_snow)[vs].mean():.2f}')
        print(f'  melt: obs freq {meas_melt[vm].mean():.2f}, model freq {sm.mod_melt[vm].mean():.2f}, '
              f'model-melt/obs-dry {(sm.mod_melt & ~meas_melt)[vm].mean():.2f}, '
              f'model-dry/obs-melt {(~sm.mod_melt & meas_melt)[vm].mean():.2f}')
        print(f'  snowline (model - obs): {ok.sum()} dates with both, '
              f'median {np.median(diff):+.0f} m, IQR {np.percentile(diff, 25):+.0f} to '
              f'{np.percentile(diff, 75):+.0f} m; Jun-Sep median '
              f'{np.median(diff[summer]) if summer.any() else np.nan:+.0f} m (n={summer.sum()})')
        print(f'  dates with snowline in obs only {(~np.isnan(sl[:, 1]) & np.isnan(sl[:, 0])).sum()}, '
              f'model only {(np.isnan(sl[:, 1]) & ~np.isnan(sl[:, 0])).sum()}')
        print('  by band (snow freq obs/mod, melt freq obs/mod):')
        for s, m in zip(snow.to_dict('records'), melt.to_dict('records')):
            print(f'    {s["z"]:5.0f} m n={s["n"]:4d}  snow {s["obs"]:.2f}/{s["mod"]:.2f}  '
                  f'melt {m["obs"]:.2f}/{m["mod"]:.2f}')

        ax = axes[0, col]
        ax.plot(snow.obs, snow.z, 'k-o', ms=3, label='SAR')
        ax.plot(snow['mod'], snow.z, 'C0-o', ms=3, label='model')
        ax.set_title(f'{g}\nkp={kp_values[i]} wf={wf_values[j]}')
        ax.set_xlabel('snow-covered frequency')
        ax.set_xlim(0, 1)
        ax = axes[1, col]
        ax.plot(melt.obs, melt.z, 'k-o', ms=3, label='SAR')
        ax.plot(melt['mod'], melt.z, 'C3-o', ms=3, label='model')
        ax.set_xlabel('melting frequency')
        ax.set_xlim(0, 1)
        ax = axes[2, col]
        t = pd.DatetimeIndex(sm.time)
        ax.plot(t, sl[:, 1], 'k.', ms=4, label='SAR')
        ax.plot(t, sl[:, 0], 'C0.', ms=4, label='model')
        ax.set_ylabel('snowline [m]')
        ax.tick_params(axis='x', labelrotation=45)
        for r in (0, 1):
            axes[r, col].set_ylabel('elevation [m]')
    for r in range(3):
        axes[r, 0].legend(fontsize=8)
    fig.tight_layout()
    fig_fn = os.path.join(args.run_dir, 'snow_elev_bias.png')
    fig.savefig(fig_fn, dpi=140)
    print(f'\nWrote {fig_fn}', flush=True)


if __name__ == '__main__':
    main()
