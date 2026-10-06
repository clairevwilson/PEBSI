"""
Compares Sentinel-2 albedo with on-glacier station albedo, and the
model at the nearest point, at Gulkana's radiometer station.

Station daily albedo is the ratio of summed reflected to summed
incoming shortwave over the day's hours with SWin above a floor, so
low-sun hours carry little weight. The Sentinel-2 value is the pixel
containing the station on each retrieval date.

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

AWS_FP = '/ocean/projects/ees260009p/cwilson4/climate_data/AWS/Processed/'
STATIONS = {
    'B': dict(fn=AWS_FP + 'gulkanaB2025_albedo.csv', time_col='TIMESTAMP'),
}
BEST_FN = '/ocean/projects/ees260009p/cwilson4/Output/loss_surface_0/best.zarr'
META_FN = '/jet/home/cwilson4/PEBSI/data/glacier_metadata.csv'
SWIN_FLOOR = 100 * 3600     # J m-2 per hour, i.e. 100 W m-2
SURFACE_NAMES = {0: 'snow', 1: 'firn', 2: 'ice'}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--best-fn', default=BEST_FN)
    return p.parse_args()


def station_daily_albedo(fn, time_col):
    df = pd.read_csv(fn, index_col=time_col, parse_dates=True)
    sw_in, sw_out = df['SWin'], df['SWout'].abs()
    day = sw_in > SWIN_FLOOR
    daily = sw_out[day].resample('1D').sum() / sw_in[day].resample('1D').sum()
    hours = day.resample('1D').sum()
    return daily[hours >= 4]


def main():
    args = parse_args()
    meta = pd.read_csv(META_FN, dtype={'rgiid': str})
    ds = xr.open_zarr(args.best_fn, consolidated=False)
    gul = np.where(ds['rgiid'].values.astype(str) == '01.00570')[0]
    lat, lon = ds.lat.values[gul], ds.lon.values[gul]

    alb = Albedo('gulkana', use='s2')
    proj = Transformer.from_crs('EPSG:4326', alb.crs, always_xy=True)

    for site, cfg in STATIONS.items():
        row = meta[(meta.rgiid == '01.00570') & (meta.site == site)].iloc[0]
        aws = station_daily_albedo(cfg['fn'], cfg['time_col'])

        x, y = proj.transform(row.lon, row.lat)
        s2 = alb.ds_meas['albedo'].sel(x=x, y=y, method='nearest').drop_duplicates('time').to_series()
        s2.index = s2.index.normalize()
        s2 = s2.dropna()

        p = gul[int(np.argmin(np.hypot(lat - row.lat, (lon - row.lon) * np.cos(np.deg2rad(row.lat)))))]
        model = ds[['albedo', 'surftype']].isel(point=p).to_dataframe()[['albedo', 'surftype']]
        model.index = model.index.normalize()

        dates = s2.index.intersection(aws.index)
        t = pd.DataFrame({'aws': aws.reindex(dates), 's2': s2.reindex(dates),
                          'model': model['albedo'].reindex(dates),
                          'model_surface': model['surftype'].reindex(dates).map(SURFACE_NAMES)})
        print(f'\nGulkana {site} ({row.elevation:.0f} m), station {aws.index[0].date()} to '
              f'{aws.index[-1].date()}; model point at {float(ds.elev.values[p]):.0f} m', flush=True)
        with pd.option_context('display.width', 200, 'display.max_rows', None,
                               'display.float_format', '{:.3f}'.format):
            print(t)
        ok = t[['aws', 's2']].notnull().all(axis=1) & (t.s2 <= 1)
        print(f'  {int(ok.sum())} dates: S2 - station mean {(t.s2 - t.aws)[ok].mean():+.3f}, '
              f'MAE {(t.s2 - t.aws)[ok].abs().mean():.3f}; model - station mean '
              f'{(t.model - t.aws)[ok].mean():+.3f}, MAE {(t.model - t.aws)[ok].abs().mean():.3f}', flush=True)


if __name__ == '__main__':
    main()
