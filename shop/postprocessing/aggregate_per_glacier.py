"""
Collapses a distributed run's point output to one glacier-wide series
per glacier, written as per_glacier.nc beside the run's output.zarr.

Each glacier's weights (ds.weight) already sum to 1 over its own points,
so the glacier-wide value is just the weighted sum across them. The
time axis is left at the run's own output frequency; resample from the
small file rather than the raw output.

By default every run directory under Output/controlled/ that holds an
output.zarr is processed.

@author: clairevwilson
"""
import argparse
import glob
import os
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from project.parameters import translate_names

ROOT = '/ocean/projects/ees260009p/cwilson4/Output/controlled/'
OUT_NAME = 'per_glacier.nc'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run_dirs', nargs='*',
                   help=f'run directories (default: every one in {ROOT})')
    p.add_argument('--vars', nargs='+', default=['mass_balance'],
                   help='(time, point) variables to aggregate (default %(default)s)')
    return p.parse_args()


def aggregate(run_dir, var_names):
    ds = xr.open_zarr(os.path.join(run_dir, 'output.zarr'), consolidated=False)
    rgiid = ds['rgiid'].values.astype(str)
    weight = ds['weight'].values
    gids = list(np.unique(rgiid))

    per_glacier = []
    with xr.set_options(keep_attrs=True):
        for gid in gids:
            mask = rgiid == gid
            w = xr.DataArray(weight[mask], dims='point')
            # skipna=False so a NaN point shows up instead of silently
            # dropping out of the glacier-wide value
            per_glacier.append((ds[var_names].isel(point=mask) * w)
                               .sum('point', skipna=False))

    glacier = xr.DataArray([translate_names[g] for g in gids], dims='glacier', name='glacier')
    agg = xr.concat(per_glacier, dim=glacier)
    agg = agg.assign_coords(rgiid=('glacier', gids))
    agg.attrs = dict(ds.attrs, source=os.path.join(run_dir, 'output.zarr'),
                     aggregation='area-weighted sum over points (ds.weight)')
    return agg.compute()


def main():
    args = parse_args()
    run_dirs = args.run_dirs or sorted(os.path.dirname(fn) for fn in
                                       glob.glob(os.path.join(ROOT, '*', 'output.zarr')))
    for run_dir in run_dirs:
        print(f'Aggregating {run_dir}...', flush=True)
        agg = aggregate(run_dir, args.vars)
        out_fn = os.path.join(run_dir, OUT_NAME)
        agg.to_netcdf(out_fn)
        print(f'Wrote {out_fn}', flush=True)
        for g in agg.glacier.values:
            if 'mass_balance' in agg:
                total = float(agg['mass_balance'].sel(glacier=g).sum())
                print(f'  {g:<12} cumulative mass_balance {total:.2f} m w.e.', flush=True)


if __name__ == '__main__':
    main()
