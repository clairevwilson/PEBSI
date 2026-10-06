"""
Uses an hourly-output Gulkana run (one kp, wind_factor pair) to test
whether the daily-minimum total_water written by the daily runs is why
the model misses late-summer SAR wetness, and whether those misses are
snow that refreezes overnight.

  - checks the daily minimum of the hourly run against the matching
    combination in the loss surface run
  - scores SAR wet extent against total_water reduced four ways: daily
    minimum (what the daily output stores), daily mean, daily maximum,
    and the hour of each pass (ascending ~ASC_HOUR UTC, descending
    ~DESC_HOUR UTC; Sentinel-1 is a dawn-dusk orbit, ~18:00 / ~06:00
    local solar time at Gulkana)
  - for Jul-Sep SAR-wet observations the daily minimum misses: surface
    type, snow depth, daily maximum water, and that day's refreeze and
    melt

@author: clairevwilson
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from project.glacierwide_loss import Wet
from project.parameters import translate_rgi

GLACIER = 'gulkana'
RUN_GLOB = '/ocean/projects/ees260009p/cwilson4/Output/gulkana_hourly*/'
LOSS_SURFACE_DIR = '/ocean/projects/ees260009p/cwilson4/Output/loss_surface_0/'
ANALYSIS_START = '2015-10-01'
ASC_HOUR = 4
DESC_HOUR = 16
LATE_MONTHS = (7, 8, 9)
SURFTYPES = {0: 'snow', 1: 'firn', 2: 'ice'}
DEPTH_BINS = [0, 0.01, 0.1, 0.25, 0.5, 1, np.inf]
VARS = ['lon', 'lat', 'total_water', 'refreeze', 'melt', 'surftype', 'snowdepth', 'airtemp']


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', default=None,
                   help=f'hourly run directory (default: newest match of {RUN_GLOB})')
    p.add_argument('--fig-dir', default=None, help='default: --run-dir')
    return p.parse_args()


def daily_ds(ds, water):
    """
    Daily dataset carrying `water` as total_water, so Wet.get_model_wet
    (whose daily-min resample is then a no-op) scores that reduction.
    """
    return xr.Dataset({'total_water': water}, coords={'lon': ds.lon, 'lat': ds.lat})


def at_hour(ds, hour):
    """total_water at one UTC hour each day, relabelled to the day."""
    w = ds['total_water'].sel(time=ds.time.dt.hour == hour)
    return w.assign_coords(time=w.time.dt.floor('D'))


def score(wets):
    """
    Brier and Jul-Sep miss / false-wet rates pooled over one or more Wet
    objects that already ran get_model_wet.
    """
    se, n, missed, sar_wet, false_wet, sar_dry = 0.0, 0, 0, 0, 0, 0
    for wet in wets:
        y = wet.meas_wet
        valid = ~np.isnan(y)
        se += np.nansum((wet.p_wet - y)**2)
        n += valid.sum()
        late = np.isin(pd.to_datetime(wet.time).month, LATE_MONTHS)[:, None] & valid
        missed += (late & (y == 1) & ~wet.mod_wet).sum()
        sar_wet += (late & (y == 1)).sum()
        false_wet += (late & (y == 0) & wet.mod_wet).sum()
        sar_dry += (late & (y == 0)).sum()
    return dict(brier=se / n, missed=missed / sar_wet, false_wet=false_wet / sar_dry)


def weekly_wet(wets):
    rows = []
    for wet in wets:
        week = (pd.to_datetime(wet.time).dayofyear.values - 1) // 7
        y = wet.meas_wet
        rows.append(pd.DataFrame(dict(week=week, obs=np.nanmean(y, axis=1),
                                      mod=np.nanmean(np.where(np.isnan(y), np.nan, wet.p_wet), axis=1))))
    return pd.concat(rows).groupby('week').mean()


def check_against_loss_surface(daily_min, lat, lon):
    """
    Max |difference| between this run's daily minimum and the loss surface
    run's stored daily total_water for the same kp, wind_factor and points.
    """
    grid = np.load(os.path.join(LOSS_SURFACE_DIR, 'loss_surface_grid.npz'), allow_pickle=True)
    n_base = int(grid['n_base'])
    i_kp = int(np.argmin(np.abs(grid['kp'] - KP)))
    i_wf = int(np.argmin(np.abs(grid['wind_factor'] - WF)))
    c = i_kp * len(grid['wind_factor']) + i_wf
    ds = xr.open_zarr(os.path.join(LOSS_SURFACE_DIR, 'output.zarr'), consolidated=False)
    block = ds[['lon', 'lat', 'rgiid', 'total_water']].isel(point=slice(c * n_base, (c + 1) * n_base))
    block = block.isel(point=block.rgiid.values == translate_rgi[GLACIER]['6'])
    if block.sizes['point'] != len(lat):
        return f'point counts differ ({block.sizes["point"]} vs {len(lat)}), not compared'
    order_ls = np.lexsort((block.lon.values, block.lat.values))
    order_h = np.lexsort((lon, lat))
    ref = block['total_water'].sel(time=daily_min.time).values[:, order_ls]
    diff = np.nanmax(np.abs(ref - daily_min.values[:, order_h]))
    return f'max |daily min - loss surface daily total_water| = {diff:.3g} kg m-2'


def main():
    global KP, WF
    args = parse_args()
    run_dir = args.run_dir or sorted(glob.glob(RUN_GLOB), key=os.path.getmtime)[-1]
    fig_dir = args.fig_dir or run_dir
    grid = np.load(glob.glob(os.path.join(run_dir, 'loss_surface_grid*.npz'))[0], allow_pickle=True)
    KP, WF = float(grid['kp'][0]), float(grid['wind_factor'][0])
    print(f'run {run_dir} | kp {KP}, wf {WF}', flush=True)

    ds = (xr.open_zarr(os.path.join(run_dir, 'output.zarr'), consolidated=False)[VARS]
          .sel(time=slice(ANALYSIS_START, None)).load())
    water = ds['total_water']
    daily = water.resample(time='1D')
    reductions = {'daily min': daily.min(), 'daily mean': daily.mean(), 'daily max': daily.max()}

    print(check_against_loss_surface(reductions['daily min'], ds.lat.values, ds.lon.values), flush=True)

    results = {}
    for name, w in reductions.items():
        wet = Wet(GLACIER)
        wet.get_model_wet(daily_ds(ds, w))
        results[name] = [wet]
    pass_wets = []
    for direction, hour in (('ascending', ASC_HOUR), ('descending', DESC_HOUR)):
        wet = Wet(GLACIER, direction=direction)
        wet.get_model_wet(daily_ds(ds, at_hour(ds, hour)))
        pass_wets.append(wet)
    results['pass hour'] = pass_wets

    threshold = results['daily min'][0].water_threshold * 1000
    print(f'\nthreshold {threshold:g} mm | scoring from {ANALYSIS_START}', flush=True)
    for name, wets in results.items():
        s = score(wets)
        print(f'  {name:<11} Brier {s["brier"]:.3f} | Jul-Sep SAR-wet missed {s["missed"]:.0%} | '
              f'Jul-Sep SAR-dry called wet {s["false_wet"]:.0%}', flush=True)

    wet = results['daily min'][0]
    days = pd.DatetimeIndex(wet.time).floor('D')
    def on_days(da, reduce):
        return getattr(da.resample(time='1D'), reduce)().sel(time=days).values

    day_max = on_days(water, 'max')
    day_refreeze = on_days(ds['refreeze'], 'sum') * 1000
    day_melt = on_days(ds['melt'], 'sum') * 1000
    day_depth = on_days(ds['snowdepth'], 'mean')
    day_surftype = on_days(ds['surftype'], 'last')
    day_tmin = on_days(ds['airtemp'], 'min')

    y = wet.meas_wet
    late = np.isin(pd.to_datetime(wet.time).month, LATE_MONTHS)[:, None] & ~np.isnan(y)
    groups = {'hit': late & (y == 1) & wet.mod_wet,
              'missed': late & (y == 1) & ~wet.mod_wet,
              'missed, min < 1 mm': late & (y == 1) & ~wet.mod_wet & (reductions['daily min'].sel(time=days).values < 1)}
    depth_labels = [f'{lo:g}-{hi:g} m' if np.isfinite(hi) else f'>{lo:g} m'
                    for lo, hi in zip(DEPTH_BINS[:-1], DEPTH_BINS[1:])]

    print('\nJul-Sep SAR-wet observations, scored on the daily minimum:', flush=True)
    for name, m in groups.items():
        surf = ', '.join(f'{v} {np.mean(day_surftype[m] == k):.0%}' for k, v in SURFTYPES.items())
        depth = np.histogram(day_depth[m], bins=DEPTH_BINS)[0] / max(m.sum(), 1)
        print(f'  {name} (n {m.sum()})', flush=True)
        print(f'    surface: {surf}', flush=True)
        print(f'    snow depth: ' + ', '.join(f'{lab} {f:.0%}' for lab, f in zip(depth_labels, depth)), flush=True)
        print(f'    daily max water median {np.median(day_max[m]):.1f} mm, '
              f'>= threshold {np.mean(day_max[m] >= threshold):.0%}', flush=True)
        print(f'    daily refreeze median {np.median(day_refreeze[m]):.1f} mm w.e., '
              f'> 0.1 mm on {np.mean(day_refreeze[m] > 0.1):.0%} of days | '
              f'daily melt median {np.median(day_melt[m]):.1f} mm w.e. | '
              f'min air temp median {np.median(day_tmin[m]):.1f} C', flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    ax = axes[0]
    wk = weekly_wet(results['daily min'])
    ax.plot(wk.index * 7 + 4, wk['obs'], 'k-', label='SAR')
    for (name, wets), c in zip(results.items(), ['C0', 'C1', 'C2', 'C3']):
        wk = weekly_wet(wets)
        ax.plot(wk.index * 7 + 4, wk['mod'], color=c, label=f'model, {name}')
    ax.set_ylim(0, 1)
    ax.set_xlabel('Day of year')
    ax.set_ylabel('Wet fraction')
    ax.set_title(f'{GLACIER}, kp {KP}, wf {WF}, threshold {threshold:g} mm', fontsize=10)
    ax.legend(fontsize=7)

    ax = axes[1]
    m = groups['missed']
    bins = np.linspace(0, 60, 61)
    ax.hist(np.clip(reductions['daily min'].sel(time=days).values[m], 0, 60), bins=bins,
            histtype='step', lw=1.5, color='C0', label='daily min')
    ax.hist(np.clip(day_max[m], 0, 60), bins=bins, histtype='step', lw=1.5, color='C2', label='daily max')
    ax.axvline(threshold, color='gray', ls=':', lw=1)
    ax.set_xlabel('total_water [mm w.e.]')
    ax.set_ylabel('Count')
    ax.set_title('Jul-Sep SAR wet, missed on daily min', fontsize=10)
    ax.legend(fontsize=7)

    ax = axes[2]
    for k, v in SURFTYPES.items():
        mk = m & (day_surftype == k)
        if mk.any():
            ax.scatter(day_depth[mk], day_refreeze[mk], s=3, alpha=0.3, label=v)
    ax.set_xlabel('Snow depth [m]')
    ax.set_ylabel('Daily refreeze [mm w.e.]')
    ax.set_title('Missed: depth vs refreeze', fontsize=10)
    ax.legend(fontsize=7, markerscale=4)

    fig.tight_layout()
    fig_fn = os.path.join(fig_dir, f'{GLACIER}_hourly_wet.png')
    fig.savefig(fig_fn, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'\nWrote {fig_fn}', flush=True)


if __name__ == '__main__':
    main()
