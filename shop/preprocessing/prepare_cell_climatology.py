"""
Prepares a netCDF file of every MERRA-2 cell's own CDF
for each bias-corrected variable, which quantile delta
mapping uses to place each value within its cell's
climatology (see Climate.bias_correct_qm).

@author: clairevwilson
"""
import os
import sys
import argparse
from types import SimpleNamespace

import numpy as np
import xarray as xr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# roughly the years covered by the station records behind the quantile mapping
PERIOD = ('2000-01-01', '2024-12-31')
sys.path.insert(0, ROOT)

from pebsi import defaults
from pebsi.io.climate import Climate
from project.parameters import host, HOST_PATHS

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--region', type=int, default=defaults.rgi_region)
p.add_argument('--climate-fp', default=HOST_PATHS[host]['climate_fp'])
p.add_argument('--vars', nargs='+', default=None,
               help='variables to process (default every group in bias_fn)')
args = p.parse_args()

params = SimpleNamespace(**{k: getattr(defaults, k) for k in dir(defaults) if not k.startswith('__')})
params.climate_fp = args.climate_fp
bias_fn = os.path.join(ROOT, params.bias_fn)
region = 'reg' + str(args.region).zfill(2)
out_fn = params.climate_fp + params.bias_cell_fn.format(r=region)

# Climate supplies the file names and unit conversions the model uses
cl = Climate.__new__(Climate)
cl.params = params
cl.get_vardict()

with xr.open_dataset(bias_fn, group='temp') as ds:
    quantiles = ds['quantile'].values
if args.vars is None:
    import netCDF4
    with netCDF4.Dataset(bias_fn) as nc:
        args.vars = list(nc.groups)


def open_var(v):
    fn = cl.GCM_fp + cl.var_dict[v]['fn'].format(r=region)
    ds = xr.open_zarr(fn, consolidated=False) if 'zarr' in fn else xr.open_dataset(fn)
    da = cl.check_units(v, ds[cl.var_dict[v]['vn']])
    return da.sel(time=slice(*PERIOD))


for i, var in enumerate(args.vars):
    if var == 'wind':
        u, v = open_var('uwind'), open_var('vwind')
        da = u
    else:
        da = open_var(var)
    lat, lon = da[cl.lat_vn].values, da[cl.lon_vn].values
    cdf = np.full((len(lat), len(lon), len(quantiles)), np.nan)

    # load one zarr chunk block of cells at a time
    lat_step, lon_step = da.chunks[da.dims.index(cl.lat_vn)][0], da.chunks[da.dims.index(cl.lon_vn)][0]
    for la in range(0, len(lat), lat_step):
        for lo in range(0, len(lon), lon_step):
            block = {cl.lat_vn: slice(la, la + lat_step), cl.lon_vn: slice(lo, lo + lon_step)}
            if var == 'wind':
                values = np.hypot(u.isel(block).values, v.isel(block).values)
            else:
                values = da.isel(block).transpose('time', cl.lat_vn, cl.lon_vn).values
            # SWin CDFs only hold daytime values, as in the station CDFs
            if var == 'SWin':
                values = np.where(values > 100, values, np.nan)
            cdf[block[cl.lat_vn], block[cl.lon_vn]] = np.moveaxis(np.nanquantile(values, quantiles, axis=0), 0, -1)
            print(f'{var} lat {la} lon {lo} done', flush=True)

    ds_out = xr.Dataset({'cdf': ((cl.lat_vn, cl.lon_vn, 'quantile'), cdf)},
                        coords={cl.lat_vn: lat, cl.lon_vn: lon, 'quantile': quantiles},
                        attrs={'period': ' to '.join(PERIOD)})
    ds_out.to_netcdf(out_fn, group=var, mode='w' if i == 0 else 'a')
    print(f'Wrote {var} to {out_fn}', flush=True)
