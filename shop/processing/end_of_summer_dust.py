"""
Finds the dust_factor that gives a target surface dust concentration in
snow at the end of summer.

For each run (one dust_factor each), takes the surface layer on the
end-of-summer date at every point still snow-covered then -- that
year's accumulation area -- and area-weights its dust concentration
per glacier. Surface dust scales close to linearly with dust_factor,
so the factor that reaches the target is interpolated in log-log space
across the runs.

@author: clairevwilson
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from project.parameters import translate_names

OUTPUT = '/ocean/projects/ees260009p/cwilson4/Output/'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run_dirs', nargs='*',
                   help=f'run directories (default: {OUTPUT}dust_tune_*/base_*)')
    p.add_argument('--day', default='08-31', help='MM-DD treated as end of summer')
    p.add_argument('--target', type=float, default=200, help='target dust [ppm]')
    return p.parse_args()


def summarize(run_dir, day):
    with open(os.path.join(run_dir, 'config.yaml')) as f:
        dust_factor = float(yaml.safe_load(f)['dust_factor'])
    ds = xr.open_zarr(os.path.join(run_dir, 'output.zarr'), consolidated=False)
    times = pd.DatetimeIndex(ds.time.values)
    dates = times[times.strftime('%m-%d') == day]
    top = ds[['layerdust', 'layertype']].isel(layer=0).sel(time=dates).compute()
    rgiid = ds['rgiid'].values.astype(str)
    weight = ds['weight'].values

    rows = []
    for gid in np.unique(rgiid):
        pts = rgiid == gid
        for d in dates:
            t = top.sel(time=d).isel(point=pts)
            snow = t.layertype.values == 0
            dust = t.layerdust.values[snow]
            w = weight[pts][snow]
            rows.append(dict(
                glacier=translate_names[gid], dust_factor=dust_factor, year=d.year,
                snow_area=w.sum(),
                dust_mean=np.average(dust, weights=w) if snow.any() else np.nan,
                dust_median=np.median(dust) if snow.any() else np.nan))
    return rows


def main():
    args = parse_args()
    run_dirs = args.run_dirs or sorted(glob.glob(os.path.join(OUTPUT, 'dust_tune_*', 'base_*')))
    df = pd.DataFrame([r for d in run_dirs for r in summarize(d, args.day)])

    with pd.option_context('display.width', 200, 'display.max_rows', None,
                           'display.float_format', '{:.2f}'.format):
        print(f'Surface dust [ppm] in snow-covered area on {args.day}')
        print(df.set_index(['glacier', 'dust_factor', 'year']))

        # mean across years, then interpolate log(dust) vs log(factor)
        mean = df.groupby(['glacier', 'dust_factor'])['dust_mean'].mean().unstack('dust_factor')
        print('\nMean across years [ppm]')
        print(mean)

    print(f'\ndust_factor reaching {args.target:g} ppm:')
    needed = {}
    for g, row in mean.iterrows():
        row = row.dropna()
        if row.empty:
            print(f'  {g:<12} no snow-covered area on {args.day} in any run')
            continue
        f = np.exp(np.interp(np.log(args.target), np.log(row.values), np.log(row.index.values),
                             left=np.nan, right=np.nan))
        needed[g] = f
        note = '' if np.isfinite(f) else '  (target outside the tested range)'
        print(f'  {g:<12} {f:.0f}{note}')
    vals = np.array([v for v in needed.values() if np.isfinite(v)])
    if vals.size:
        print(f'  median across glaciers: {np.median(vals):.0f}')


if __name__ == '__main__':
    main()
