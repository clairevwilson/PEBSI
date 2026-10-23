"""
Collapses a distributed run's point output to small files written
beside the run's output.zarr:

per_glacier.nc
    One glacier-wide series per glacier. Each glacier's weights
    (ds.weight) already sum to 1 over its own points, so the glacier-wide
    value is just the weighted sum across them. The time axis is left at
    the run's own output frequency; resample from the small file rather
    than the raw output. The weighted mean lat/lon of each glacier is
    stored as a coordinate for mapping.

per_point_annual.nc
    Water-year (Oct-Sep) sums of a few variables at every point, labeled
    by the year the water year ends in, with elev/weight/rgiid kept as
    point coordinates for elevation profiles. The first and last water
    years are partial if the run does not start/end on Oct 1.

By default every run directory under Output/controlled/ that holds an
output.zarr is processed.

@author: clairevwilson
"""
import argparse
import glob
import os
import sys

import dask
import numpy as np
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from project.parameters import translate_names

ROOT = '/ocean/projects/ees260009p/cwilson4/Output/controlled/'
OUT_NAME = 'per_glacier.nc'
POINT_OUT_NAME = 'per_point_annual.nc'
GLACIER_VARS = ['mass_balance', 'melt', 'refreeze', 'airtemp',
                'shortwave_in', 'shortwave_net', 'longwave_net',
                'sensible_heat', 'latent_heat', 'ground_heat', 'rain_heat',
                'melt_energy']
POINT_VARS = ['mass_balance', 'melt', 'shortwave_in', 'shortwave_net']


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run_dirs', nargs='*',
                   help=f'run directories (default: every one in {ROOT})')
    p.add_argument('--vars', nargs='+', default=GLACIER_VARS,
                   help='(time, point) variables to aggregate per glacier (default %(default)s)')
    p.add_argument('--point-vars', nargs='+', default=POINT_VARS,
                   help='(time, point) variables to sum per water year at each point (default %(default)s)')
    return p.parse_args()


def aggregate_glaciers(ds, var_names):
    rgiid = ds['rgiid'].values.astype(str)
    weight = ds['weight'].values
    gids = list(np.unique(rgiid))

    per_glacier = []
    lats, lons = [], []
    with xr.set_options(keep_attrs=True):
        for gid in gids:
            mask = rgiid == gid
            w = xr.DataArray(weight[mask], dims='point')
            # skipna=False so a NaN point shows up instead of silently
            # dropping out of the glacier-wide value
            per_glacier.append((ds[var_names].isel(point=mask) * w)
                               .sum('point', skipna=False))
            lats.append(float((ds['lat'].values[mask] * weight[mask]).sum()))
            lons.append(float((ds['lon'].values[mask] * weight[mask]).sum()))

    glacier = xr.DataArray([translate_names[g] for g in gids], dims='glacier', name='glacier')
    agg = xr.concat(per_glacier, dim=glacier)
    return agg.assign_coords(rgiid=('glacier', gids), lat=('glacier', lats),
                             lon=('glacier', lons))


def aggregate_points(ds, var_names):
    with xr.set_options(keep_attrs=True):
        annual = ds[var_names].resample(time='YS-OCT').sum(skipna=False)
    annual = annual.rename(time='water_year')
    annual['water_year'] = annual['water_year'].dt.year + 1
    return annual.assign_coords({v: ds[v].astype(str) if v == 'rgiid' else ds[v]
                                 for v in ['rgiid', 'elev', 'weight', 'lat', 'lon']})


def aggregate(run_dir, var_names, point_var_names):
    ds = xr.open_zarr(os.path.join(run_dir, 'output.zarr'), consolidated=False)
    agg = aggregate_glaciers(ds, var_names)
    annual = aggregate_points(ds, point_var_names)
    source = os.path.join(run_dir, 'output.zarr')
    agg.attrs = dict(ds.attrs, source=source,
                     aggregation='area-weighted sum over points (ds.weight)')
    annual.attrs = dict(ds.attrs, source=source,
                        aggregation='water-year (Oct-Sep) sum at each point')
    # one compute so each zarr chunk is read once for both outputs
    return dask.compute(agg, annual)


def main():
    args = parse_args()
    run_dirs = args.run_dirs or sorted(os.path.dirname(fn) for fn in
                                       glob.glob(os.path.join(ROOT, '*', 'output.zarr')))
    for run_dir in run_dirs:
        print(f'Aggregating {run_dir}...', flush=True)
        agg, annual = aggregate(run_dir, args.vars, args.point_vars)
        for ds, name in [(agg, OUT_NAME), (annual, POINT_OUT_NAME)]:
            out_fn = os.path.join(run_dir, name)
            # write then swap in, since an open notebook's HDF5 lock on
            # the old file blocks overwriting it in place
            ds.to_netcdf(out_fn + '.tmp')
            os.replace(out_fn + '.tmp', out_fn)
            print(f'Wrote {out_fn}', flush=True)
        for g in agg.glacier.values:
            if 'mass_balance' in agg:
                total = float(agg['mass_balance'].sel(glacier=g).sum())
                print(f'  {g:<12} cumulative mass_balance {total:.2f} m w.e.', flush=True)


if __name__ == '__main__':
    main()
