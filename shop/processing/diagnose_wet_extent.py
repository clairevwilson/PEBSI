"""
Breaks the wet extent Brier score from a loss_surface run into
interpretable pieces for one glacier at a few (kp, wind_factor)
combinations: how often the SAR sees wet vs how often the model says
wet, the false-wet and missed-wet rates, and how that splits by month
and elevation.

Defaults compare the combination minimizing the summed loss (the X on
plot_loss_surface.py's figures) with the one minimizing wet loss alone.

@author: clairevwilson
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from project.glacierwide_loss import Wet
from project.parameters import translate_rgi

RUN_DIR = '/ocean/projects/ees260009p/cwilson4/Output/loss_surface_2/'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('glaciers', nargs='+')
    p.add_argument('--run-dir', default=RUN_DIR)
    return p.parse_args()


def score(wet, elev):
    meas = wet.meas_wet
    ok = ~np.isnan(meas)
    obs = meas.astype(bool) & ok
    mod = wet.mod_wet
    p = wet.p_wet
    f_obs = obs[ok].mean()

    out = dict(
        n=int(ok.sum()),
        obs_wet=f_obs,
        mod_wet=mod[ok].mean(),
        accuracy=(mod == obs)[ok].mean(),
        false_wet=(mod & ~obs)[ok].sum() / (~obs & ok).sum(),
        missed_wet=(~mod & obs)[ok].sum() / obs.sum(),
        brier=np.nanmean((p - meas)**2),
        brier_const=f_obs * (1 - f_obs))

    month = pd.DatetimeIndex(wet.time).month
    rows = []
    for m in np.unique(month):
        sel = (month == m)[:, None] & ok
        rows.append(dict(month=m, n=int(sel.sum()), obs_wet=obs[sel].mean(),
                         mod_wet=mod[sel].mean(),
                         brier_share=np.sum((p - meas)[sel]**2) / np.sum((p - meas)[ok]**2)))
    by_month = pd.DataFrame(rows).set_index('month')

    edges = np.quantile(elev, [0, 1/3, 2/3, 1])
    band = np.clip(np.searchsorted(edges, elev, side='right') - 1, 0, 2)
    rows = []
    for b in range(3):
        sel = (band == b)[None, :] & ok
        rows.append(dict(elev=f'{edges[b]:.0f}-{edges[b + 1]:.0f}',
                         obs_wet=obs[sel].mean(), mod_wet=mod[sel].mean(),
                         false_wet=(mod & ~obs)[sel].sum() / (~obs & sel).sum(),
                         missed_wet=(~mod & obs)[sel].sum() / (obs & sel).sum()))
    by_elev = pd.DataFrame(rows).set_index('elev')
    return out, by_month, by_elev


def main():
    args = parse_args()
    grid = np.load(os.path.join(args.run_dir, 'loss_surface_grid.npz'), allow_pickle=True)
    losses = np.load(os.path.join(args.run_dir, 'loss_surface_losses.npz'), allow_pickle=True)
    kp_values, wf_values = losses['kp'], losses['wind_factor']
    n_base = int(grid['n_base'])
    ds = xr.open_zarr(os.path.join(args.run_dir, 'output.zarr'), consolidated=False)
    rgiid_base = ds['rgiid'].values[:n_base]

    pd.set_option('display.float_format', '{:.3f}'.format)
    pd.set_option('display.width', 200)
    for g in args.glaciers:
        total = sum(losses[f'{g}_{k}'] for k in ('albedo', 'mb', 'wet'))
        combos = {'summed best (X)': np.unravel_index(np.nanargmin(total), total.shape),
                  'wet best': np.unravel_index(np.nanargmin(losses[f'{g}_wet']), total.shape)}
        gidx = np.flatnonzero(rgiid_base == translate_rgi[g]['6'])
        wet = Wet(g)
        tmin = wet.ds_meas_wet.time.values.min() - np.timedelta64(5, 'D')
        tmax = wet.ds_meas_wet.time.values.max() + np.timedelta64(5, 'D')

        print(f'\n===== {g} ({len(gidx)} points) =====')
        for name, (i, j) in combos.items():
            c = i * len(wf_values) + j
            sub = (ds[['lon', 'lat', 'elev', 'total_water']]
                   .isel(point=c * n_base + gidx)
                   .sel(time=slice(tmin, tmax)).load())
            wet.get_model_wet(sub)
            out, by_month, by_elev = score(wet, sub['elev'].values)
            print(f'\n--- {name}: kp={kp_values[i]} wf={wf_values[j]} ---')
            print('  '.join(f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}'
                            for k, v in out.items()))
            print(by_month.to_string())
            print(by_elev.to_string())


if __name__ == '__main__':
    main()
