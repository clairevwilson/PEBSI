"""
Summarizes end-of-summer surface-layer black carbon by glacier and
elevation band, for comparison with measured BC in snow.

@author: clairevwilson
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from project.parameters import translate_names

RUN_DIR = '/ocean/projects/ees260009p/cwilson4/Output/bc_check/base_0/'
SURFACE_NAMES = {0: 'snow', 1: 'firn', 2: 'ice'}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', default=RUN_DIR)
    p.add_argument('--day', default='08-31', help='MM-DD treated as end of summer')
    p.add_argument('--n-bands', type=int, default=4, help='elevation bands per glacier')
    return p.parse_args()


def main():
    args = parse_args()
    ds = xr.open_zarr(os.path.join(args.run_dir, 'output.zarr'), consolidated=False)
    times = pd.DatetimeIndex(ds.time.values)
    dates = times[times.strftime('%m-%d') == args.day]
    print(f'End-of-summer dates: {[str(d.date()) for d in dates]}', flush=True)

    top = ds[['layerBC', 'layerdensity', 'layerheight', 'layertype']].isel(layer=0).sel(time=dates).compute()
    rgiid = ds['rgiid'].values.astype(str)
    elev = ds['elev'].values
    weight = ds['weight'].values

    rows = []
    for gid in np.unique(rgiid):
        g = translate_names[gid]
        pts = np.where(rgiid == gid)[0]
        edges = np.quantile(elev[pts], np.linspace(0, 1, args.n_bands + 1))
        band = np.clip(np.digitize(elev[pts], edges[1:-1]), 0, args.n_bands - 1)
        for d in dates:
            t = top.sel(time=d).isel(point=pts)
            bc = t.layerBC.values
            for b in range(args.n_bands):
                m = band == b
                stype = t.layertype.values[m].astype(int)
                rows.append(dict(
                    glacier=g, year=d.year,
                    elev=f'{edges[b]:.0f}-{edges[b + 1]:.0f}',
                    n=int(m.sum()),
                    surface='/'.join(f'{SURFACE_NAMES[s]} {np.mean(stype == s):.0%}'
                                     for s in np.unique(stype)),
                    top_cm=float(np.median(t.layerheight.values[m]) * 100),
                    density=float(np.median(t.layerdensity.values[m])),
                    BC_median=float(np.median(bc[m])),
                    BC_p10=float(np.percentile(bc[m], 10)),
                    BC_p90=float(np.percentile(bc[m], 90)),
                ))
            snow = t.layertype.values < 2
            if snow.any():
                w = weight[pts][snow]
                print(f'{g:<12} {d.year} area-weighted snow/firn surface BC: '
                      f'{np.average(bc[snow], weights=w):.1f} ppb, '
                      f'{w.sum():.0%} of area', flush=True)

    df = pd.DataFrame(rows).set_index(['glacier', 'year', 'elev'])
    with pd.option_context('display.width', 250, 'display.max_columns', None,
                           'display.max_rows', None, 'display.float_format', '{:.1f}'.format):
        print('\nSurface (top) layer at end of summer; BC in ppb')
        print(df)


if __name__ == '__main__':
    main()
