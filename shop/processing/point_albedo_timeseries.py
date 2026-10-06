"""
Compares modeled and Sentinel-2 albedo through one melt season in a
glacier's accumulation area, alongside the surface properties the
albedo emulator sees.

Accumulation-area points are those in the glacier's upper elevation
third that the model keeps snow-covered every day from June through
mid-September. The single point shown is the one of those with the most
Sentinel-2 retrievals; the accumulation-area mean over all of them is
printed for every Sentinel-2 date too.

Needs a run with daily 'layers', 'surface' and 'MB' output.

@author: clairevwilson
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from project.glacierwide_loss import Albedo
from project.parameters import translate_rgi

RUN_DIR = '/ocean/projects/ees260009p/cwilson4/Output/grainsize_check/base_0/'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('glacier')
    p.add_argument('--run-dir', default=RUN_DIR)
    p.add_argument('--year', type=int, default=2019)
    return p.parse_args()


def main():
    args = parse_args()
    yr = args.year
    ds = xr.open_zarr(os.path.join(args.run_dir, 'output.zarr'), consolidated=False)
    pts = np.where(ds['rgiid'].values.astype(str) == translate_rgi[args.glacier]['6'])[0]
    sub = ds.isel(point=pts).sel(time=slice(f'{yr}-03-01', f'{yr}-09-30'))

    top = sub[['layergrainsize', 'layerdensity', 'layerheight', 'layerage',
               'layerdust', 'layerBC', 'layerOC', 'layertype']].isel(layer=0)
    model = xr.merge([top, sub[['albedo', 'surftype', 'accumulation', 'melt', 'snowdepth']]]).compute()
    elev = sub['elev'].values

    # accumulation-area points
    summer = model.sel(time=slice(f'{yr}-06-01', f'{yr}-09-15'))
    always_snow = (summer.surftype.values == 0).all(axis=0)
    upper = elev >= np.quantile(elev, 2 / 3)
    acc = np.where(always_snow & upper)[0]
    print(f'{args.glacier} {yr}: {acc.size} accumulation-area points '
          f'(upper third, snow every day Jun 1 - Sep 15)', flush=True)

    # Sentinel-2 at those points
    alb = Albedo(args.glacier, use='s2')
    x, y = Transformer.from_crs('EPSG:4326', alb.crs, always_xy=True).transform(
        sub.lon.values[acc], sub.lat.values[acc])
    s2 = alb.ds_meas['albedo'].sel(x=xr.DataArray(x, dims='point'),
                                   y=xr.DataArray(y, dims='point'), method='nearest')
    s2 = s2.sel(time=slice(f'{yr}-03-01', f'{yr}-09-30')).drop_duplicates('time').compute()
    s2_days = pd.DatetimeIndex(s2.time.values).normalize()
    mod_on_s2 = model.isel(point=acc).sel(time=s2_days)

    # accumulation-area mean on each Sentinel-2 date
    obs = s2.values
    mod = mod_on_s2.albedo.values
    both = np.isfinite(obs) & np.isfinite(mod)
    rows = []
    for i, d in enumerate(s2_days):
        m = both[i]
        if m.sum() < 0.25 * acc.size:
            continue
        rows.append(dict(date=str(d.date()), n=int(m.sum()),
                         obs=obs[i, m].mean(), mod=mod[i, m].mean(),
                         mod_minus_obs=(mod[i, m] - obs[i, m]).mean(),
                         grain=mod_on_s2.layergrainsize.values[i, m].mean(),
                         dust_ppm=mod_on_s2.layerdust.values[i, m].mean(),
                         age_days=mod_on_s2.layerage.values[i, m].mean()))
    with pd.option_context('display.width', 200, 'display.max_rows', None,
                           'display.max_columns', None, 'display.float_format', '{:.3f}'.format):
        print('\nAccumulation-area mean on Sentinel-2 dates (>= 25% of points retrieved)')
        print(pd.DataFrame(rows).set_index('date'))

    # the single point with the most retrievals
    k = int(np.argmax(np.isfinite(obs).sum(axis=0)))
    p = acc[k]
    pm = model.isel(point=p)
    obs_p = pd.Series(obs[:, k], index=s2_days).dropna()
    print(f'\nPoint {p}: {elev[p]:.0f} m, {obs_p.size} Sentinel-2 retrievals', flush=True)
    t = pd.DataFrame({
        'obs': obs_p,
        'mod': pm.albedo.sel(time=obs_p.index).values,
        'grain_um': pm.layergrainsize.sel(time=obs_p.index).values,
        'density': pm.layerdensity.sel(time=obs_p.index).values,
        'top_cm': pm.layerheight.sel(time=obs_p.index).values * 100,
        'age_days': pm.layerage.sel(time=obs_p.index).values,
        'dust_ppm': pm.layerdust.sel(time=obs_p.index).values,
        'BC_ppb': pm.layerBC.sel(time=obs_p.index).values,
        'snow_7d_mwe': pm.accumulation.rolling(time=7).sum().sel(time=obs_p.index).values,
    })
    with pd.option_context('display.width', 200, 'display.max_rows', None,
                           'display.max_columns', None, 'display.float_format', '{:.3f}'.format):
        print(t)

    fig, axes = plt.subplots(4, 1, figsize=(8, 9), sharex=True)
    axes[0].plot(pm.time, pm.albedo, color='k', lw=1.5, label='model (daily)')
    axes[0].scatter(obs_p.index, obs_p.values, color='tab:red', s=18, zorder=3, label='Sentinel-2')
    axes[0].set_ylabel('Albedo')
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(pm.time, pm.layergrainsize, color='k', lw=1.5)
    axes[1].set_ylabel('Surface grain\nsize (um)')
    axes[2].plot(pm.time, pm.layerdust, color='k', lw=1.5)
    axes[2].set_ylabel('Surface dust\n(ppm)')
    axes[3].bar(pm.time, pm.accumulation * 1000, color='k', width=1)
    axes[3].set_ylabel('Snowfall\n(mm w.e. d$^{-1}$)')
    axes[0].set_title(f'{args.glacier} accumulation-area point, {elev[p]:.0f} m, {yr}')
    fig.tight_layout()
    fig_fn = os.path.join(args.run_dir, f'point_albedo_{args.glacier}_{yr}.png')
    fig.savefig(fig_fn, dpi=150)
    print(f'\nWrote {fig_fn}', flush=True)


if __name__ == '__main__':
    main()
