"""
Checks whether Sentinel-2 albedo on spring snow depends on terrain,
which snow albedo itself should not, but an imperfect illumination
correction would.

1. Sentinel-2 alone: on April-May retrievals, every glacier pixel, the
   mean albedo and share of retrievals above 1 by aspect and slope.
2. Against the model: at each model point the model keeps snow-covered,
   the mean April-May model minus Sentinel-2 offset, by aspect, slope
   and elevation, and a least-squares fit of the offset on elevation,
   slope and northness so terrain can be separated from elevation.

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
from project.glacierwide_loss import Albedo, DEM
from project.parameters import translate_rgi

BEST_FN = '/ocean/projects/ees260009p/cwilson4/Output/loss_surface_0/best.zarr'
ASPECTS = {'N': (315, 45), 'E': (45, 135), 'S': (135, 225), 'W': (225, 315)}
SLOPE_BINS = [0, 5, 10, 15, 20, 90]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('glacier')
    p.add_argument('--best-fn', default=BEST_FN)
    p.add_argument('--months', type=int, nargs='+', default=[4, 5])
    return p.parse_args()


def aspect_class(aspect):
    out = np.full(aspect.shape, '', dtype=object)
    for name, (lo, hi) in ASPECTS.items():
        m = ((aspect >= lo) | (aspect < hi)) if lo > hi else ((aspect >= lo) & (aspect < hi))
        out[m] = name
    return out


def slope_aspect(dem):
    z = dem.values.astype(float)
    dx = float(abs(dem.x[1] - dem.x[0]))
    dy = float(abs(dem.y[1] - dem.y[0]))
    dzdy, dzdx = np.gradient(z, dy, dx)
    # y decreases down the rows of a north-up raster, so flip its sign for north
    dzdn = -dzdy if float(dem.y[1] - dem.y[0]) < 0 else dzdy
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdn)))
    # direction the slope faces (downhill), clockwise from north
    aspect = (np.degrees(np.arctan2(-dzdx, -dzdn)) + 360) % 360
    return slope, aspect


def main():
    args = parse_args()
    alb = Albedo(args.glacier, use='s2')
    s2 = alb.ds_meas['albedo'].rio.write_crs(alb.crs)
    s2 = s2.sel(time=s2.time.dt.month.isin(args.months))

    dem = DEM(args.glacier).dem.squeeze().rio.reproject_match(s2.isel(time=0))
    dem = dem.where(dem > -1000)
    slope, aspect = slope_aspect(dem)
    on_glacier = np.isfinite(s2.values).any(axis=0) & np.isfinite(slope)

    # 1. Sentinel-2 alone, per pixel
    vals = s2.values
    pix = pd.DataFrame({
        'mean_albedo': np.nanmean(np.where(vals <= 1, vals, np.nan), axis=0)[on_glacier],
        'share_above_1': (np.nansum(vals > 1, axis=0) / np.sum(np.isfinite(vals), axis=0))[on_glacier],
        'slope_bin': pd.cut(slope[on_glacier], SLOPE_BINS),
        'aspect': aspect_class(aspect[on_glacier]),
        'elev': dem.values[on_glacier]})
    with pd.option_context('display.width', 200, 'display.float_format', '{:.3f}'.format):
        print(f'{args.glacier}: Sentinel-2 months {args.months}, {on_glacier.sum()} glacier pixels')
        print('\nSentinel-2 by aspect')
        print(pix.groupby('aspect')[['mean_albedo', 'share_above_1', 'elev']].agg(['mean', 'size']).iloc[:, [0, 2, 4, 5]])
        print('\nSentinel-2 by slope')
        print(pix.groupby('slope_bin', observed=True)[['mean_albedo', 'share_above_1', 'elev']].mean())

    # 2. against the model at its points
    ds = xr.open_zarr(args.best_fn, consolidated=False)
    pts = np.where(ds['rgiid'].values.astype(str) == translate_rgi[args.glacier]['6'])[0]
    sub = ds[['albedo', 'surftype', 'lon', 'lat', 'elev']].isel(point=pts)
    x, y = Transformer.from_crs('EPSG:4326', alb.crs, always_xy=True).transform(sub.lon.values, sub.lat.values)
    xs, ys = xr.DataArray(x, dims='point'), xr.DataArray(y, dims='point')
    s2_pts = s2.sel(x=xs, y=ys, method='nearest').drop_duplicates('time')
    days = pd.DatetimeIndex(s2_pts.time.values).normalize()
    keep = days <= pd.Timestamp(sub.time.values[-1])
    s2_pts, days = s2_pts.isel(time=np.where(keep)[0]), days[keep]
    mod = sub.albedo.sel(time=days).values
    snow = sub.surftype.sel(time=days).values == 0
    obs = s2_pts.values
    ok = snow & np.isfinite(obs) & (obs <= 1) & np.isfinite(mod)
    offset = np.where(ok, mod - obs, np.nan)

    slope_da = xr.DataArray(slope, dims=('y', 'x'), coords={'y': dem.y, 'x': dem.x})
    aspect_da = xr.DataArray(aspect, dims=('y', 'x'), coords={'y': dem.y, 'x': dem.x})
    pt = pd.DataFrame({
        'offset': np.nanmean(offset, axis=0),
        'n': ok.sum(axis=0),
        'share_above_1': np.nansum(obs > 1, axis=0) / np.maximum(np.sum(np.isfinite(obs), axis=0), 1),
        'elev': sub.elev.values,
        'slope': slope_da.sel(x=xs, y=ys, method='nearest').values,
        'aspect_deg': aspect_da.sel(x=xs, y=ys, method='nearest').values})
    pt = pt[pt.n >= 5].copy()
    pt['aspect'] = aspect_class(pt.aspect_deg.values)
    pt['slope_bin'] = pd.cut(pt.slope, SLOPE_BINS)
    pt['elev_third'] = pd.qcut(pt.elev, 3)

    with pd.option_context('display.width', 200, 'display.float_format', '{:.3f}'.format):
        print(f'\nModel minus Sentinel-2 on modeled snow, months {args.months}, {len(pt)} model points')
        for col in ('aspect', 'slope_bin', 'elev_third'):
            print(f'\nby {col}')
            print(pt.groupby(col, observed=True)[['offset', 'share_above_1', 'elev', 'slope']]
                  .agg(['mean']).droplevel(1, axis=1).assign(points=pt.groupby(col, observed=True).size()))

    # offset ~ elevation (per km) + slope (per 10 deg) + northness
    X = np.column_stack([np.ones(len(pt)), (pt.elev - pt.elev.mean()) / 1000, pt.slope / 10,
                         np.cos(np.radians(pt.aspect_deg))])
    coef, *_ = np.linalg.lstsq(X, pt.offset.values, rcond=None)
    resid = pt.offset.values - X @ coef
    r2 = 1 - resid.var() / pt.offset.var()
    print(f'\nFit: offset = {coef[0]:.3f} + {coef[1]:+.3f} per km elevation '
          f'+ {coef[2]:+.3f} per 10 deg slope + {coef[3]:+.3f} x northness;  R2 = {r2:.2f}')


if __name__ == '__main__':
    main()
