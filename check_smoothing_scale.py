"""
Looks for a defensible smoothing length.

Slope taken from neighbouring 30 m COP30 pixels is partly DEM error --
stereo noise and gap-filling -- rather than terrain. Noise variance dies
off quickly as the DEM is smoothed; real terrain variance dies off
slowly. The scale where the decay rate changes separates them, and is
the shortest length at which slope is measuring the glacier instead of
the DEM.

Also tracks individual points through the smoothing, to show what it
does to the slope and aspect a point is actually run with.

    python check_smoothing_scale.py
"""
import os

import geopandas as gpd
import numpy as np
import rioxarray as rxr
import shapely
from scipy.ndimage import gaussian_filter

from host_paths import HOST_PATHS, host
from pebsi.io import mesh
from pebsi.io.terrain import Terrain
from project.glacierwide_loss import translate_rgi

LENGTHS = [0, 60, 90, 120, 150, 200, 250, 300, 400, 500, 700, 1000]
EXAMPLE_H = 300.0
p = HOST_PATHS[host]


def load(glacier):
    gid = translate_rgi[glacier]['6']
    region = gid.split('.')[0]
    names = [f.split('.')[0] for f in os.listdir(p['rgi_fp'])
             if f.startswith(region)]
    gdf = gpd.read_file(f"{p['rgi_fp']}../{names[0]}/{names[0]}.shp")
    poly, crs = mesh.glacier_polygon(gdf, gid)
    g = gdf.loc[gdf['RGIId'] == 'RGI60-' + gid]

    dem = rxr.open_rasterio(p['cop30_vrt_path']).squeeze().drop_vars('band')
    b = g.to_crs(dem.rio.crs).total_bounds
    dem = dem.rio.clip_box(b[0] - .02, b[1] - .02, b[2] + .02, b[3] + .02)
    dem = dem.rio.reproject(crs)
    if dem.rio.nodata is not None:
        dem = dem.where((dem != dem.rio.nodata) & np.isfinite(dem))
    return poly, dem


def derivatives(dem, L):
    """Slope and aspect after smoothing the DEM to length L."""
    rx, ry = dem.rio.resolution()
    z = dem.values
    if L > 0:
        sigma = L / (2.355 * abs(rx))
        ok = np.isfinite(z)
        tot = gaussian_filter(np.where(ok, z, 0.0), sigma, mode='nearest')
        wt = gaussian_filter(ok.astype(float), sigma, mode='nearest')
        z = np.where(wt > 1e-6, tot / wt, np.nan)
    dy, dx = np.gradient(z, ry, rx)
    slope = np.rad2deg(np.arctan(np.sqrt(dx ** 2 + dy ** 2)))
    aspect = np.rad2deg((np.arctan2(-dy, -dx) + 2 * np.pi) % (2 * np.pi)) % 360
    return slope, aspect


for glacier in ('gulkana', 'kennicott'):
    poly, dem = load(glacier)
    rx, _ = dem.rio.resolution()
    gx, gy = np.meshgrid(dem.x.values, dem.y.values)
    ice = shapely.contains_xy(poly, gx, gy) & np.isfinite(dem.values)

    print('=' * 78)
    print(f'{glacier.upper()}   DEM {abs(rx):.1f} m, {ice.sum()} on-ice pixels')
    print('=' * 78)
    print(f'{"L (m)":>7} {"L/px":>6} {"slope mean":>11} {"slope std":>10} '
          f'{"var left":>9} {"d(std)/dlnL":>12}')

    stds, means = [], []
    for L in LENGTHS:
        s, _ = derivatives(dem, L)
        v = s[ice]
        v = v[np.isfinite(v)]
        stds.append(v.std())
        means.append(v.mean())
    stds = np.array(stds)
    base = stds[0] ** 2

    for i, L in enumerate(LENGTHS):
        if i == 0:
            rate = np.nan
        else:
            lo = LENGTHS[i - 1] if LENGTHS[i - 1] > 0 else abs(rx)
            rate = (stds[i] - stds[i - 1]) / (np.log(L) - np.log(lo))
        rate_s = f'{rate:>12.3f}' if np.isfinite(rate) else f'{"-":>12}'
        print(f'{L:>7} {L / abs(rx):>6.1f} {means[i]:>11.3f} {stds[i]:>10.3f} '
              f'{100 * stds[i] ** 2 / base:>8.1f}% {rate_s}')

    # what the smoothing does to individual points on a real mesh
    xs, ys, w, _ = mesh.mesh_polygon(poly, EXAMPLE_H)
    pts = np.column_stack([xs, ys])
    px = np.column_stack([gx[ice], gy[ice]])

    tracks = {}
    for L in (0, 150, 250, 500):
        s, a = derivatives(dem, L)
        sv, _ = Terrain.cell_mean(pts, px, s[ice])
        av, _ = Terrain.cell_mean(pts, px, a[ice], circular=True)
        tracks[L] = (sv, av)

    s0 = tracks[0][0]
    good = np.isfinite(s0)
    order = np.argsort(np.where(good, s0, np.nan))
    valid = order[: good.sum()]
    picks = {'flattest': valid[0], 'median': valid[len(valid) // 2],
             'steepest': valid[-1]}

    print()
    print(f'  individual points on the h={EXAMPLE_H:.0f} mesh:')
    print(f'  {"point":>10} ' + ' '.join(f'{f"L={L}":>16}' for L in tracks))
    for name, idx in picks.items():
        row = f'  {name:>10} '
        for L in tracks:
            sv, av = tracks[L]
            row += f'{sv[idx]:>7.1f}deg/{av[idx]:>4.0f}  '
        print(row)
    print(f'  (slope deg / aspect deg for the same point at each L)')
    print()
