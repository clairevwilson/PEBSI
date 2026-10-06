"""
Summarizes surface-layer snow grain size through summer, per glacier:
on a set of dates, the spread of surface grain size over snow-covered
points, the share at the cap (grainsize_rfz), and the surface layer's
liquid water mass fraction, which drives wet grain growth.

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

RUN_DIR = '/ocean/projects/ees260009p/cwilson4/Output/grainsize_check/base_0/'
CAP = 1500


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', default=RUN_DIR)
    p.add_argument('--days', nargs='+',
                   default=['06-01', '06-15', '07-01', '07-15', '08-01', '08-15', '09-01', '09-15'],
                   help='MM-DD dates to summarize')
    return p.parse_args()


def main():
    args = parse_args()
    ds = xr.open_zarr(os.path.join(args.run_dir, 'output.zarr'), consolidated=False)
    times = pd.DatetimeIndex(ds.time.values)
    dates = times[times.strftime('%m-%d').isin(args.days)]
    top = ds[['layergrainsize', 'layertype', 'layerwater', 'layerdensity', 'layerheight']] \
        .isel(layer=0).compute()
    rgiid = ds['rgiid'].values.astype(str)

    rows = []
    for gid in np.unique(rgiid):
        pts = rgiid == gid
        t = top.isel(point=pts)
        snow_all = t.layertype.values == 0
        grain = np.where(snow_all, t.layergrainsize.values, np.nan)
        at_cap = grain >= CAP - 10
        # first day from May on that at least half the snow-covered points sit at the cap
        with np.errstate(invalid='ignore'):
            share = np.nanmean(np.where(snow_all, at_cap, np.nan), axis=1)
        half = (share >= 0.5) & (times.month >= 5) & (times.month <= 9)
        first_half = str(times[np.argmax(half)].date()) if half.any() else 'never'
        for d in dates:
            i = times.get_loc(d)
            snow = snow_all[i]
            if not snow.any():
                continue
            g = grain[i][snow]
            water = t.layerwater.values[i][snow]
            solid = (t.layerdensity.values[i] * t.layerheight.values[i])[snow]
            rows.append(dict(
                glacier=translate_names[gid], date=str(d.date()),
                snow_points=int(snow.sum()),
                grain_p10=np.percentile(g, 10), grain_median=np.median(g),
                grain_p90=np.percentile(g, 90), at_cap=np.mean(g >= CAP - 10),
                water_frac_median=np.median(water / (water + solid))))
        print(f'{translate_names[gid]:<12} first day >=50% of snow surface at {CAP} um: {first_half}',
              flush=True)

    with pd.option_context('display.width', 200, 'display.max_rows', None, 'display.max_columns', None,
                           'display.float_format', '{:.3f}'.format):
        print('\nSurface layer on snow-covered points; grain size in um')
        print(pd.DataFrame(rows).set_index(['glacier', 'date']))


if __name__ == '__main__':
    main()
