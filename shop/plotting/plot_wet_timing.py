"""
Breaks down the wet extent Brier score of a loss_surface run to see
whether the misfit comes from melt onset (spring, dry -> wet) or freeze
onset (fall, wet -> dry).

Each glacier is scored at its own best (kp, wind_factor) from the
compute_loss_surface.py losses, and at the combination where the wet
term alone is best. For each, the Brier score is split by:
  - month, into false wet (model wet, SAR dry) and missed wet
    (model dry, SAR wet)
  - half of the melt season: each point-year (Mar-Nov) is split at the
    midpoint of its observed wet period (mid-July if never observed
    wet), so errors land with the transition they are closest to
  - timing: first and last wet acquisition per point-year, model minus
    observed, sampled at the SAR acquisition times only

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from project.glacierwide_loss import Wet
from project.parameters import translate_rgi

ROOT = '/ocean/projects/ees260009p/cwilson4/Output/loss_surface_0/'
METRICS = ('albedo', 'mb', 'wet')
SEASON_MONTHS = range(3, 12)
DEFAULT_MID_DOY = 196
SEASON_NAMES = {0: 'winter', 1: 'melt onset half', 2: 'freeze onset half'}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', default=ROOT)
    p.add_argument('--loss-fn', default=None,
                   help='default: loss_surface_losses_crps.npz in --run-dir')
    p.add_argument('--direction', default=None, choices=['ascending', 'descending'],
                   help='score only this SAR pass direction (default: both)')
    p.add_argument('--fig-dir', default=None,
                   help='default: wet_timing/ in --run-dir')
    p.add_argument('--wet-scale', type=float, default=0.01,
                   help='Wet wet_scale [m w.e.] used for p_wet')
    return p.parse_args()


def season_labels(time, meas):
    """
    0 = winter (Dec-Feb), 1 = melt onset half, 2 = freeze onset half,
    per (time, point), split at each point-year's observed wet midpoint.
    """
    dates = pd.to_datetime(time)
    doy = dates.dayofyear.values
    labels = np.zeros(meas.shape, dtype=int)
    for year in np.unique(dates.year):
        idx = np.where((dates.year == year) & dates.month.isin(SEASON_MONTHS))[0]
        if len(idx) == 0:
            continue
        wet = meas[idx] == 1
        doy_t = doy[idx][:, None]
        first = np.where(wet, doy_t, np.inf).min(axis=0)
        last = np.where(wet, doy_t, -np.inf).max(axis=0)
        mid = np.where(np.isfinite(first), (first + last) / 2, DEFAULT_MID_DOY)
        labels[idx] = np.where(doy_t <= mid[None, :], 1, 2)
    return labels


def timing_errors(time, mod, meas):
    """
    Model minus observed day of year of the first and last wet
    acquisition, per point-year where both are wet at least once,
    using only acquisitions with a valid observation.
    """
    dates = pd.to_datetime(time)
    doy = dates.dayofyear.values
    onset, freeze = [], []
    for year in np.unique(dates.year):
        idx = np.where((dates.year == year) & dates.month.isin(SEASON_MONTHS))[0]
        if len(idx) == 0:
            continue
        valid = ~np.isnan(meas[idx])
        obs_wet = (meas[idx] == 1) & valid
        mod_wet = mod[idx] & valid
        doy_t = doy[idx][:, None]
        both = obs_wet.any(axis=0) & mod_wet.any(axis=0)
        if not both.any():
            continue
        first_obs = np.where(obs_wet, doy_t, np.inf).min(axis=0)
        first_mod = np.where(mod_wet, doy_t, np.inf).min(axis=0)
        last_obs = np.where(obs_wet, doy_t, -np.inf).max(axis=0)
        last_mod = np.where(mod_wet, doy_t, -np.inf).max(axis=0)
        onset.append((first_mod - first_obs)[both])
        freeze.append((last_mod - last_obs)[both])
    return np.concatenate(onset), np.concatenate(freeze)


def diagnose(wet):
    p = wet.p_wet
    y = wet.meas_wet
    valid = ~np.isnan(y) & ~np.isnan(p)
    false_wet = np.where(valid, (1 - y) * p**2, 0)
    missed_wet = np.where(valid, y * (1 - p)**2, 0)
    total = false_wet.sum() + missed_wet.sum()
    n = valid.sum()

    labels = season_labels(wet.time, y)
    by_season = {}
    for k, name in SEASON_NAMES.items():
        m = labels == k
        by_season[name] = dict(
            share=(false_wet[m].sum() + missed_wet[m].sum()) / total,
            false_wet=false_wet[m].sum() / total,
            missed_wet=missed_wet[m].sum() / total,
            brier=(false_wet[m].sum() + missed_wet[m].sum()) / max((valid & m).sum(), 1))

    months = pd.to_datetime(wet.time).month.values
    by_month = np.array([[false_wet[months == mo].sum() / total,
                          missed_wet[months == mo].sum() / total] for mo in range(1, 13)])

    onset, freeze = timing_errors(wet.time, wet.mod_wet, y)

    doy = pd.to_datetime(wet.time).dayofyear.values
    week = (doy - 1) // 7
    frac_obs = np.nanmean(y, axis=1)
    frac_mod = np.nanmean(np.where(np.isnan(y), np.nan, p), axis=1)
    weekly = pd.DataFrame(dict(week=week, obs=frac_obs, mod=frac_mod)).groupby('week').mean()

    return dict(brier=total / n, by_season=by_season, by_month=by_month,
                onset=onset, freeze=freeze, weekly=weekly)


def pick_combos(losses, glacier):
    """
    (label, (i_kp, i_wf)) for the glacier's best combined point and the
    point where the wet term alone is best.
    """
    combined = sum(losses[f'{glacier}_{k}'] for k in METRICS)
    return [('best combined', np.unravel_index(np.nanargmin(combined), combined.shape)),
            ('best wet only', np.unravel_index(np.nanargmin(losses[f'{glacier}_wet']), combined.shape))]


def load_combo(ds, c, n_base, rgi, variables=('total_water',)):
    block = ds[['lon', 'lat', 'rgiid', *variables]].isel(point=slice(c * n_base, (c + 1) * n_base))
    gmask = block.rgiid.values == rgi
    return block.isel(point=gmask).load()


def report(glacier, label, kp, wf, d):
    print(f'\n{glacier} | {label} (kp {kp}, wf {wf}) | Brier {d["brier"]:.3f}', flush=True)
    for name, s in d['by_season'].items():
        print(f'  {name:<18} share {s["share"]:.0%} (false wet {s["false_wet"]:.0%}, '
              f'missed wet {s["missed_wet"]:.0%}), Brier {s["brier"]:.3f}', flush=True)
    for name, e in (('onset', d['onset']), ('freeze', d['freeze'])):
        q25, q50, q75 = np.percentile(e, [25, 50, 75])
        print(f'  {name} timing, model - obs [days]: median {q50:+.0f} '
              f'(IQR {q25:+.0f} to {q75:+.0f}), n = {len(e)}', flush=True)


def plot_glacier(glacier, results, fig_fn):
    fig, axes = plt.subplots(len(results), 2, figsize=(11, 3.2 * len(results)), squeeze=False)
    for row, (label, kp, wf, d) in enumerate(results):
        ax = axes[row, 0]
        wk = d['weekly']
        ax.plot(wk.index * 7 + 4, wk['obs'], 'k-', label='SAR wet fraction')
        ax.plot(wk.index * 7 + 4, wk['mod'], 'C0-', label='modeled p_wet')
        ax.set_xlim(1, 366)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Day of year')
        ax.set_ylabel('Mean over points and years')
        ax.set_title(f'{label}: kp {kp}, wf {wf}', fontsize=10)
        ax.legend(fontsize=8, loc='upper left')

        ax = axes[row, 1]
        mo = np.arange(1, 13)
        ax.bar(mo, d['by_month'][:, 0], color='C3', label='false wet (model wet, SAR dry)')
        ax.bar(mo, d['by_month'][:, 1], bottom=d['by_month'][:, 0], color='C0',
               label='missed wet (model dry, SAR wet)')
        ax.set_xticks(mo)
        ax.set_xlabel('Month')
        ax.set_ylabel('Share of squared error')
        ax.set_title(f'Brier {d["brier"]:.3f}', fontsize=10)
        ax.legend(fontsize=8, loc='upper left')
    fig.suptitle(glacier, fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_fn, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {fig_fn}', flush=True)


def main():
    args = parse_args()
    loss_fn = args.loss_fn or os.path.join(args.run_dir, 'loss_surface_losses_crps.npz')
    losses = np.load(loss_fn, allow_pickle=True)
    kp_values = losses['kp']
    wf_values = losses['wind_factor']
    n_base = int(losses['n_base'])
    ds = xr.open_zarr(os.path.join(args.run_dir, 'output.zarr'), consolidated=False)

    fig_dir = args.fig_dir or os.path.join(args.run_dir, 'wet_timing')
    os.makedirs(fig_dir, exist_ok=True)

    for glacier in [str(g) for g in losses['glaciers']]:
        picks = pick_combos(losses, glacier)

        wet = Wet(glacier, direction=args.direction, wet_scale=args.wet_scale)
        rgi = translate_rgi[glacier]['6']
        results = []
        for label, (i_kp, i_wf) in picks:
            c = i_kp * len(wf_values) + i_wf
            wet.get_model_wet(load_combo(ds, c, n_base, rgi))
            d = diagnose(wet)
            report(glacier, label, kp_values[i_kp], wf_values[i_wf], d)
            results.append((label, kp_values[i_kp], wf_values[i_wf], d))

        plot_glacier(glacier, results, os.path.join(fig_dir, f'wet_timing_{glacier}.png'))


if __name__ == '__main__':
    main()
