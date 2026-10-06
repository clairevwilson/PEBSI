"""
Uses Sentinel-2 albedo to check whether late-summer SAR wet observations
that the model misses sit on bare ice (an observation or definition
issue: SAR calls the ice wet, but the model's total_water on ice is ~0)
or on snow (a model issue: the model melts out to ice too early).

For each glacier, at its best combined (kp, wind_factor) and at the
point where the wet term alone is best:
  - weekly glacier-mean albedo, S2 vs model, sampled where S2 is valid
  - weekly SAR wet fraction vs modeled p_wet
  - for Jul-Sep SAR-wet observations, split into model wet (hit) and
    model dry (missed): the model's surface type and albedo, and the S2
    albedo at the nearest S2 acquisition within MATCH_DAYS. ALBEDO_ICE is
    the mean observed ice albedo the model uses, not an upper bound for
    ice, so S2 and model albedo are compared as distributions rather than
    classified with a threshold.
  - modeled total_water [mm w.e.] at those observations, for hits and
    for misses on snow and on firn, to see whether misses sit just under
    the wet threshold or are effectively dry

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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from project.glacierwide_loss import Albedo, Wet
from project.parameters import translate_rgi
from plot_wet_timing import ROOT, load_combo, pick_combos

LATE_MONTHS = (7, 8, 9)
MATCH_DAYS = 5
ALBEDO_ICE = 0.47
SURFTYPES = {0: 'snow', 1: 'firn', 2: 'ice'}
WATER_BINS = [0, 1, 5, 10, 20, 30, np.inf]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', default=ROOT)
    p.add_argument('--loss-fn', default=None,
                   help='default: loss_surface_losses_crps.npz in --run-dir')
    p.add_argument('--direction', default=None, choices=['ascending', 'descending'],
                   help='score only this SAR pass direction (default: both)')
    p.add_argument('--fig-dir', default=None,
                   help='default: wet_albedo/ in --run-dir')
    p.add_argument('--glaciers', nargs='+', default=None,
                   help='default: every glacier in the loss file')
    return p.parse_args()


def match_s2(wet_time, s2_time, s2_values):
    """
    S2 albedo (time, point) at each wet acquisition, from the nearest S2
    acquisition within MATCH_DAYS; NaN where none is close enough.
    """
    wt = pd.DataFrame({'t': pd.to_datetime(wet_time), 'i_wet': np.arange(len(wet_time))})
    st = pd.DataFrame({'t': pd.to_datetime(s2_time), 'i_s2': np.arange(len(s2_time))})
    m = pd.merge_asof(wt.sort_values('t'), st.sort_values('t'), on='t',
                      direction='nearest', tolerance=pd.Timedelta(days=MATCH_DAYS))
    m = m.sort_values('i_wet')
    out = np.full((len(wet_time), s2_values.shape[1]), np.nan)
    has = m['i_s2'].notna().values
    out[has] = s2_values[m['i_s2'].values[has].astype(int)]
    return out


def weekly(time, values):
    week = (pd.to_datetime(time).dayofyear.values - 1) // 7
    return pd.Series(np.nanmean(values, axis=1), index=week).groupby(level=0).mean()


def diagnose(sub, glacier, direction=None):
    wet = Wet(glacier, direction=direction)
    wet.get_model_wet(sub)
    alb = Albedo(glacier, use='s2')
    alb.get_model_albedo(sub)
    s2_time = alb.ds_meas_matched.time.values

    surftype = sub['surftype'].sel(time=wet.time, method='nearest').values
    mod_albedo = sub['albedo'].sel(time=wet.time, method='nearest').values
    water = sub['total_water'].sel(time=wet.time, method='nearest').values
    s2_at_wet = match_s2(wet.time, s2_time, alb.meas)

    y = wet.meas_wet
    months = pd.to_datetime(wet.time).month.values
    late = np.isin(months, LATE_MONTHS)[:, None] & ~np.isnan(y)
    groups = {'hit': late & (y == 1) & wet.mod_wet,
              'missed': late & (y == 1) & ~wet.mod_wet,
              'SAR dry': late & (y == 0)}

    stats = {}
    for name, m in groups.items():
        has_s2 = m & ~np.isnan(s2_at_wet)
        s2 = s2_at_wet[has_s2]
        stats[name] = dict(
            n=int(m.sum()),
            surftype={v: float(np.mean(surftype[m] == k)) if m.any() else np.nan
                      for k, v in SURFTYPES.items()},
            n_s2=len(s2),
            s2_quartiles=np.percentile(s2, [25, 50, 75]) if len(s2) else np.full(3, np.nan),
            mod_quartiles=(np.percentile(mod_albedo[has_s2], [25, 50, 75])
                           if len(s2) else np.full(3, np.nan)),
            s2_values=s2,
            mod_values=mod_albedo[has_s2])

    water_groups = {'hit': groups['hit'],
                    'missed, snow': groups['missed'] & (surftype == 0),
                    'missed, firn': groups['missed'] & (surftype == 1)}
    water_stats = {}
    for name, m in water_groups.items():
        w = water[m]
        water_stats[name] = dict(
            n=len(w),
            bins=np.histogram(w, bins=WATER_BINS)[0] / max(len(w), 1),
            median=float(np.median(w)) if len(w) else np.nan,
            values=w)

    obs_valid = ~np.isnan(alb.meas)
    return dict(
        stats=stats,
        water_stats=water_stats,
        threshold_mm=wet.water_threshold * 1000,
        albedo_obs=weekly(s2_time, alb.meas),
        albedo_mod=weekly(s2_time, np.where(obs_valid, alb.mod, np.nan)),
        wet_obs=weekly(wet.time, y),
        wet_mod=weekly(wet.time, np.where(np.isnan(y), np.nan, wet.p_wet)))


def report(glacier, label, kp, wf, albedo_loss, d):
    print(f'\n{glacier} | {label} (kp {kp}, wf {wf}) | albedo CRPS/sigma {albedo_loss:.3f}', flush=True)
    for name, s in d['stats'].items():
        st = ', '.join(f'{v} {s["surftype"][v]:.0%}' for v in SURFTYPES.values())
        s2q = '/'.join(f'{q:.2f}' for q in s['s2_quartiles'])
        modq = '/'.join(f'{q:.2f}' for q in s['mod_quartiles'])
        print(f'  Jul-Sep {name:<8} n {s["n"]:>6} | model surface: {st} | '
              f'albedo q25/med/q75 S2 {s2q}, model {modq} (n {s["n_s2"]} with S2)', flush=True)
    labels = [f'{lo:g}-{hi:g}' if np.isfinite(hi) else f'>{lo:g}'
              for lo, hi in zip(WATER_BINS[:-1], WATER_BINS[1:])]
    print(f'  total_water [mm] at Jul-Sep SAR-wet obs (threshold {d["threshold_mm"]:g} mm):', flush=True)
    for name, w in d['water_stats'].items():
        b = ', '.join(f'{lab} {f:.0%}' for lab, f in zip(labels, w['bins']))
        print(f'    {name:<13} n {w["n"]:>6} | median {w["median"]:.1f} | {b}', flush=True)


def plot_glacier(glacier, results, fig_fn):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    colors = ['C0', 'C1']

    ax = axes[0]
    ax.plot(results[0][3]['albedo_obs'].index * 7 + 4, results[0][3]['albedo_obs'], 'k-', label='S2')
    for (label, kp, wf, d), c in zip(results, colors):
        ax.plot(d['albedo_mod'].index * 7 + 4, d['albedo_mod'], color=c,
                label=f'model, {label} (kp {kp}, wf {wf})')
    ax.axhline(ALBEDO_ICE, color='gray', ls=':', lw=1, label='model ice albedo (mean)')
    ax.set_xlabel('Day of year')
    ax.set_ylabel('Glacier-mean albedo')
    ax.set_title('Albedo (where S2 is valid)', fontsize=10)
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.plot(results[0][3]['wet_obs'].index * 7 + 4, results[0][3]['wet_obs'], 'k-', label='SAR')
    for (label, kp, wf, d), c in zip(results, colors):
        ax.plot(d['wet_mod'].index * 7 + 4, d['wet_mod'], color=c, label=f'model, {label}')
    ax.set_ylim(0, 1)
    ax.set_xlabel('Day of year')
    ax.set_ylabel('Wet fraction')
    ax.set_title('Wet extent', fontsize=10)
    ax.legend(fontsize=7)

    ax = axes[2]
    stats = results[0][3]['stats']
    bins = np.linspace(0, 1, 41)
    for name, c in (('hit', 'C2'), ('missed', 'C3'), ('SAR dry', 'gray')):
        label = f'SAR wet, model {name}' if name != 'SAR dry' else 'SAR dry'
        if len(stats[name]['s2_values']):
            ax.hist(stats[name]['s2_values'], bins=bins, density=True, histtype='step',
                    lw=1.5, color=c, label=f'{label}: S2')
            ax.hist(stats[name]['mod_values'], bins=bins, density=True, histtype='step',
                    lw=1.5, ls='--', color=c, label=f'{label}: model')
    ax.axvline(ALBEDO_ICE, color='gray', ls=':', lw=1)
    ax.set_xlabel('Albedo at matched acquisition')
    ax.set_ylabel('Density')
    ax.set_title(f'Jul-Sep, {results[0][0]}', fontsize=10)
    ax.legend(fontsize=7)

    ax = axes[3]
    water_stats = results[0][3]['water_stats']
    bins = np.linspace(0, 60, 61)
    for name, c in (('hit', 'C2'), ('missed, snow', 'C3'), ('missed, firn', 'C4')):
        v = water_stats[name]['values']
        if len(v):
            ax.hist(np.clip(v, 0, 60), bins=bins, density=True, histtype='step', lw=1.5,
                    color=c, label=f'SAR wet, model {name} (n {len(v)})')
    ax.axvline(results[0][3]['threshold_mm'], color='gray', ls=':', lw=1, label='wet threshold')
    ax.set_xlabel('Modeled total_water [mm w.e.]')
    ax.set_ylabel('Density')
    ax.set_title(f'Jul-Sep, {results[0][0]}', fontsize=10)
    ax.legend(fontsize=7)

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

    fig_dir = args.fig_dir or os.path.join(args.run_dir, 'wet_albedo')
    os.makedirs(fig_dir, exist_ok=True)

    glaciers = args.glaciers or [str(g) for g in losses['glaciers']]
    for glacier in glaciers:
        rgi = translate_rgi[glacier]['6']
        results = []
        for label, (i_kp, i_wf) in pick_combos(losses, glacier):
            c = i_kp * len(wf_values) + i_wf
            sub = load_combo(ds, c, n_base, rgi, variables=('total_water', 'albedo', 'surftype'))
            d = diagnose(sub, glacier, args.direction)
            albedo_loss = losses[f'{glacier}_albedo'][i_kp, i_wf]
            report(glacier, label, kp_values[i_kp], wf_values[i_wf], albedo_loss, d)
            results.append((label, kp_values[i_kp], wf_values[i_wf], d))

        plot_glacier(glacier, results, os.path.join(fig_dir, f'wet_albedo_{glacier}.png'))


if __name__ == '__main__':
    main()
