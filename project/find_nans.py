"""
Finds where a stored PEBSI run first goes non-finite: for every output
variable, which points ever do, the first day each one does, and where
those points are (glacier, elevation, slope, aspect).
"""
import numpy as np
import pandas as pd
import xarray as xr

from project.parameters import GLACIERS, translate_rgi

OUTPUT_FN = '/ocean/projects/ees260009p/cwilson4/Output/nan_hunt_28k_0/output.zarr'

# how many of the earliest-failing points to describe
N_SHOW = 15


def main():
    ds = xr.open_zarr(OUTPUT_FN, consolidated=False)
    times = pd.to_datetime(ds['time'].values)
    rgiid = np.asarray(ds['rgiid'].values, dtype=str)
    name_of = {translate_rgi[g]['6']: g for g in GLACIERS}

    first_bad = {}
    for var in ds.data_vars:
        if 'time' not in ds[var].dims or ds[var].dtype.kind != 'f':
            continue
        bad = ~np.isfinite(ds[var].transpose('time', 'point').values)
        ever = bad.any(axis=0)
        if not ever.any():
            continue
        first = np.where(ever, bad.argmax(axis=0), -1)
        first_bad[var] = first
        n_bad_steps = bad.sum()
        print(f'{var:<16} {ever.sum():>6} points non-finite, {n_bad_steps} point-days, '
              f'earliest {times[first[ever].min()].date()}')

    if not first_bad:
        print('no non-finite values in any stored variable: the NaN is in the loss, not the model')
        return

    # the earliest day any variable goes bad, per point
    stacked = np.stack([np.where(f >= 0, f, np.iinfo(np.int64).max) for f in first_bad.values()])
    earliest = stacked.min(axis=0)
    culprit = np.array(list(first_bad))[stacked.argmin(axis=0)]
    failing = np.where(earliest < np.iinfo(np.int64).max)[0]
    order = failing[np.argsort(earliest[failing])]

    print(f'\n{len(failing)} points go non-finite in at least one variable')
    for gid in np.unique(rgiid[failing]):
        print(f'  {name_of.get(gid, gid):<12} {(rgiid[failing] == gid).sum()} of {(rgiid == gid).sum()} points')

    attrs = {k: np.asarray(ds[k].values) for k in ('elev', 'lat', 'lon') if k in ds}
    print(f'\nearliest {N_SHOW} failures:')
    print(f'  {"point":>6} {"glacier":<12} {"first day":<12} {"first var":<16}'
          + ''.join(f'{k:>10}' for k in attrs))
    for p in order[:N_SHOW]:
        print(f'  {p:>6} {name_of.get(rgiid[p], rgiid[p]):<12} '
              f'{str(times[earliest[p]].date()):<12} {culprit[p]:<16}'
              + ''.join(f'{v[p]:>10.3f}' for v in attrs.values()))

    # the values just before the first failure at the earliest point
    p, d = order[0], earliest[order[0]]
    print(f'\npoint {p}, days {times[max(d - 3, 0)].date()} to {times[d].date()}:')
    window = ds.isel(point=p, time=slice(max(d - 3, 0), d + 1))
    for var in ds.data_vars:
        if 'time' in ds[var].dims and ds[var].dtype.kind == 'f':
            print(f'  {var:<16} ' + ' '.join(f'{v:>11.4g}' for v in window[var].values))


if __name__ == '__main__':
    main()
