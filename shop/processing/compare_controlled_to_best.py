"""
Compares a controlled-experiment run against the loss surface's
best.zarr over the years both cover, glacier by glacier, to see where
their mass balance and albedo part ways.

best.zarr is daily (mass_balance summed, albedo averaged over each day);
the controlled run is hourly, so it is resampled the same way first.
Points are matched by (lat, lon) within each glacier, since the two runs
share the adaptive mesh but not necessarily its ordering.

Writes a summary table to stdout and a figure beside the controlled run.

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
from project.glacierwide_loss import MassBalance
from project.parameters import translate_names

BEST_FN = '/ocean/projects/ees260009p/cwilson4/Output/loss_surface_2/best.zarr'
CTRL_DIR = '/ocean/projects/ees260009p/cwilson4/Output/controlled/base_0/'
SUMMER = [6, 7, 8]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--best-fn', default=BEST_FN)
    p.add_argument('--ctrl-dir', default=CTRL_DIR)
    return p.parse_args()


def match_points(best, ctrl, gid):
    """
    Indices into each dataset's points for one glacier, in the same order.
    """
    idx = {}
    for name, ds in (('best', best), ('ctrl', ctrl)):
        pts = np.where(ds['rgiid'].values.astype(str) == gid)[0]
        key = np.round(ds['lat'].values[pts], 6) * 1e4 + np.round(ds['lon'].values[pts], 6)
        idx[name] = pts[np.argsort(key)]
    for v in ('lat', 'lon'):
        np.testing.assert_allclose(best[v].values[idx['best']], ctrl[v].values[idx['ctrl']],
                                   atol=1e-6, err_msg=f'{gid} meshes differ')
    return idx['best'], idx['ctrl']


def main():
    args = parse_args()
    best = xr.open_zarr(args.best_fn, consolidated=False)
    ctrl = xr.open_zarr(os.path.join(args.ctrl_dir, 'output.zarr'), consolidated=False)

    # whole days both runs cover
    start = max(best.time.values[0], ctrl.time.values[0])
    end = pd.Timestamp(min(best.time.values[-1], ctrl.time.values[-1])).floor('D') - pd.Timedelta(hours=1)
    print(f'Overlap {pd.Timestamp(start)} to {end}', flush=True)

    # everything is read once up front; the controlled run is hourly and
    # chunked along time, so per-glacier reads would re-decompress it all
    ctrl_time = pd.DatetimeIndex(ctrl.time.values)
    i0, i1 = ctrl_time.get_loc(pd.Timestamp(start)), ctrl_time.get_loc(end) + 1
    n_days = (i1 - i0) // 24
    days = pd.date_range(start, periods=n_days, freq='D')
    n_years = n_days / 365.25
    ctrl_years = (ctrl_time[-1] - ctrl_time[0]).days / 365.25

    print('Loading controlled run...', flush=True)
    ctrl_mb = ctrl['mass_balance'].values
    ctrl_alb = ctrl['albedo'].isel(time=slice(i0, i1)).values
    ctrl_full_pt = ctrl_mb.sum(axis=0)
    c_mb = ctrl_mb[i0:i1].reshape(n_days, 24, -1).sum(axis=1)
    c_alb = ctrl_alb.reshape(n_days, 24, -1).mean(axis=1)
    del ctrl_mb, ctrl_alb

    print('Loading best.zarr...', flush=True)
    b_mb = best['mass_balance'].sel(time=days).values
    b_alb = best['albedo'].sel(time=days).values
    best_2020_pt = best['mass_balance'].sel(time=slice('2000-01-01', '2019-12-31')).sum('time').values
    summer = days.month.isin(SUMMER)

    def daily(a):
        return xr.DataArray(a, dims=('time', 'point'), coords={'time': days})

    gids = list(np.unique(best['rgiid'].values.astype(str)))
    rows = []
    series = {}
    for gid in gids:
        g = translate_names[gid]
        ib, ic = match_points(best, ctrl, gid)
        w = best['weight'].values[ib]
        np.testing.assert_allclose(w, ctrl['weight'].values[ic], rtol=1e-6)

        # glacier-wide (weights sum to 1 per glacier)
        mb_b = daily(b_mb[:, ib] @ w[:, None])[:, 0]
        mb_c = daily(c_mb[:, ic] @ w[:, None])[:, 0]
        alb_b = daily(b_alb[:, ib] @ w[:, None])[:, 0]
        alb_c = daily(c_alb[:, ic] @ w[:, None])[:, 0]

        # per-point rate difference, to see where on the glacier it comes from
        elev = best['elev'].values[ib]
        d_pt = (c_mb[:, ic] - b_mb[:, ib]).sum(axis=0) / n_years
        low = elev <= np.median(elev)

        obs = MassBalance(g)
        ctrl_full = ctrl_full_pt[ic] @ w
        best_2020 = best_2020_pt[ib] @ w

        rows.append(dict(
            glacier=g,
            hugonnet=obs.meas,
            best_2000_2020=float(best_2020) / 20,
            best_overlap=float(mb_b.sum()) / n_years,
            ctrl_overlap=float(mb_c.sum()) / n_years,
            ctrl_full_run=float(ctrl_full) / ctrl_years,
            d_mb_low=float(np.average(d_pt[low], weights=w[low])),
            d_mb_high=float(np.average(d_pt[~low], weights=w[~low])),
            alb_summer_best=float(alb_b[summer].mean()),
            alb_summer_ctrl=float(alb_c[summer].mean()),
            alb_max_daily_diff=float(np.abs(alb_c - alb_b).max()),
        ))
        series[g] = dict(
            mb_b=mb_b.groupby('time.year').sum(), mb_c=mb_c.groupby('time.year').sum(),
            alb_b=alb_b[summer].groupby('time.year').mean(),
            alb_c=alb_c[summer].groupby('time.year').mean(),
            elev=elev, d_pt=d_pt, hugonnet=obs.meas)
        print(f'  {g} done', flush=True)

    df = pd.DataFrame(rows).set_index('glacier')
    with pd.option_context('display.width', 250, 'display.max_columns', None, 'display.float_format', '{:.3f}'.format):
        print('\nMass balance rates [m w.e. yr-1]; d_mb = ctrl - best per point, '
              'area-weighted, lower/upper half of glacier by elevation; '
              'summer = JJA glacier-wide mean albedo')
        print(df)

    # full calendar years only for the annual bars/lines
    years = np.arange(pd.Timestamp(start).year, end.year)
    fig, axes = plt.subplots(3, len(gids), figsize=(3 * len(gids), 8), sharex='row')
    for j, (g, s) in enumerate(series.items()):
        ax = axes[0, j]
        ax.plot(years, s['mb_b'].sel(year=years), color='0.5', lw=2, label='loss_surface_2 best')
        ax.plot(years, s['mb_c'].sel(year=years), color='k', lw=2, label='controlled base')
        ax.axhline(s['hugonnet'], color='k', ls=':', lw=1, label='Hugonnet 2000-2020')
        ax.set_title(g)
        ax = axes[1, j]
        ax.plot(years, s['alb_b'].sel(year=years), color='0.5', lw=2)
        ax.plot(years, s['alb_c'].sel(year=years), color='k', lw=2)
        ax = axes[2, j]
        ax.scatter(s['d_pt'], s['elev'], s=8, color='k')
        ax.axvline(0, color='0.7', lw=1)
    axes[0, 0].set_ylabel('Annual MB (m w.e.)')
    axes[1, 0].set_ylabel('JJA albedo')
    axes[2, 0].set_ylabel('Elevation (m)')
    for ax in axes[2]:
        ax.set_xlabel('MB rate ctrl - best\n(m w.e. yr$^{-1}$)')
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig_fn = os.path.join(args.ctrl_dir, 'compare_to_best.png')
    fig.savefig(fig_fn, dpi=150)
    print(f'\nWrote {fig_fn}', flush=True)


if __name__ == '__main__':
    main()
