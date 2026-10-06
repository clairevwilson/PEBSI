"""
For a glacier's accumulation-area points (upper elevation third, kept
snow-covered by the model every day Jun 1 - Sep 15 of each year),
compares Sentinel-2 and modeled albedo by month in every year of a
run, to show whether a summer offset is one year or every year.

@author: clairevwilson
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from project.glacierwide_loss import Albedo
from project.parameters import translate_rgi

BEST_FN = '/ocean/projects/ees260009p/cwilson4/Output/loss_surface_0/best.zarr'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('glacier')
    p.add_argument('--best-fn', default=BEST_FN)
    return p.parse_args()


def main():
    args = parse_args()
    ds = xr.open_zarr(args.best_fn, consolidated=False)
    pts = np.where(ds['rgiid'].values.astype(str) == translate_rgi[args.glacier]['6'])[0]
    sub = ds[['albedo', 'surftype', 'lon', 'lat', 'elev']].isel(point=pts)
    elev = sub.elev.values
    upper = elev >= np.quantile(elev, 2 / 3)

    alb = Albedo(args.glacier, use='s2')
    x, y = Transformer.from_crs('EPSG:4326', alb.crs, always_xy=True).transform(sub.lon.values, sub.lat.values)
    s2 = alb.ds_meas['albedo'].sel(x=xr.DataArray(x, dims='point'),
                                   y=xr.DataArray(y, dims='point'), method='nearest').drop_duplicates('time')
    s2 = s2.sel(time=s2.time.dt.month.isin(range(3, 10))).compute()
    days = pd.DatetimeIndex(s2.time.values).normalize()
    days_in = days <= pd.Timestamp(sub.time.values[-1])
    s2 = s2.isel(time=np.where(days_in)[0])
    days = days[days_in]
    mod = sub.albedo.sel(time=days).values
    obs = s2.values

    rows = []
    for yr in np.unique(days.year):
        summer = sub.surftype.sel(time=slice(f'{yr}-06-01', f'{yr}-09-15')).values
        acc = upper & (summer == 0).all(axis=0)
        for m in range(3, 10):
            i = np.where((days.year == yr) & (days.month == m))[0]
            o, md = obs[i][:, acc], mod[i][:, acc]
            ok = np.isfinite(o) & np.isfinite(md) & (o <= 1)
            if ok.sum() < 5:
                continue
            rows.append(dict(year=yr, month=m, points=int(acc.sum()), n=int(ok.sum()),
                             obs=o[ok].mean(), mod=md[ok].mean(), mod_minus_obs=(md[ok] - o[ok]).mean(),
                             obs_above_1=np.mean(obs[i][:, acc][np.isfinite(obs[i][:, acc])] > 1)))
    with pd.option_context('display.width', 200, 'display.max_rows', None,
                           'display.max_columns', None, 'display.float_format', '{:.3f}'.format):
        print(f'{args.glacier}: accumulation area (upper third, model snow all summer), '
              f'Sentinel-2 vs model albedo by month; retrievals > 1 excluded from the means')
        print(pd.DataFrame(rows).set_index(['year', 'month']))


if __name__ == '__main__':
    main()
