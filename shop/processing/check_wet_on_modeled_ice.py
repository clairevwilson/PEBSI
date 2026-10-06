"""
Checks what the real surface is at pixel-dates where the model has
bare ice but SAR flags the pixel wet: for each such pixel-date, the
Sentinel-2 albedo at the same pixel within a few days. Bright
(snow-like) albedo means the model melted out too early there; dark
(ice-like) albedo means SAR is flagging wet bare ice.

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
from project.glacierwide_loss import Albedo, Wet
from project.parameters import translate_names

BEST_FN = '/ocean/projects/ees260009p/cwilson4/Output/loss_surface_0/best.zarr'
ICE_DARK = 0.4
SNOW_BRIGHT = 0.5


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--best-fn', default=BEST_FN)
    p.add_argument('--window-days', type=float, default=3,
                   help='max days between SAR and Sentinel-2 dates (default %(default)s)')
    return p.parse_args()


def main():
    args = parse_args()
    ds = xr.open_zarr(args.best_fn, consolidated=False)
    rgiid = ds['rgiid'].values.astype(str)
    window = pd.Timedelta(days=args.window_days)

    print(f'Model ice & SAR wet pixel-dates, with Sentinel-2 albedo within {args.window_days:g} days')
    for gid in pd.unique(rgiid):
        g = translate_names[gid]
        sub = ds.isel(point=np.where(rgiid == gid)[0])

        wet = Wet(g)
        wet.get_model_wet(sub)
        surf = sub['surftype'].reindex(time=wet.time, method='nearest').values
        target = (surf == 2) & (wet.meas_wet == 1)

        alb = Albedo(g, use='s2')
        x, y = Transformer.from_crs('EPSG:4326', alb.crs, always_xy=True).transform(
            sub.lon.values, sub.lat.values)
        s2 = alb.ds_meas['albedo'].sel(x=xr.DataArray(x, dims='point'),
                                       y=xr.DataArray(y, dims='point'), method='nearest')
        s2 = s2.drop_duplicates('time')
        s2_times = pd.DatetimeIndex(s2.time.values)
        s2_vals = s2.values

        matched = []
        for ti, t in enumerate(pd.DatetimeIndex(wet.time)):
            pts = np.where(target[ti])[0]
            if not pts.size:
                continue
            near = np.where(np.abs(s2_times - t) <= window)[0]
            if not near.size:
                continue
            vals = s2_vals[near][:, pts]
            closest = np.argsort(np.abs(s2_times[near] - t))
            for p_i in range(len(pts)):
                v = vals[closest, p_i]
                v = v[np.isfinite(v)]
                if v.size:
                    matched.append(v[0])
        matched = np.array(matched)
        n_target = int(target.sum())
        if not matched.size:
            print(f'  {g:<12} {n_target} pixel-dates, none with a Sentinel-2 match')
            continue
        print(f'  {g:<12} {n_target} pixel-dates, {matched.size} with a Sentinel-2 match: '
              f'snow-bright (>= {SNOW_BRIGHT}) {np.mean(matched >= SNOW_BRIGHT):.0%}, '
              f'in between {np.mean((matched >= ICE_DARK) & (matched < SNOW_BRIGHT)):.0%}, '
              f'ice-dark (< {ICE_DARK}) {np.mean(matched < ICE_DARK):.0%}, '
              f'median albedo {np.median(matched):.2f}', flush=True)


if __name__ == '__main__':
    main()
