"""
Slope and aspect are cell-averaged from a 30 m DEM, so the variance the
model sees grows as the cells shrink toward the DEM's own pixels. If the
sweep is still recovering variance at its finest spacing, it is still
converging on the DEM grid rather than on a physical limit.

Reports the area-weighted spread of cell-mean slope at each h against
the raw pixel spread.
"""
import os

import geopandas as gpd
import numpy as np
import rioxarray as rxr
import shapely
from scipy.spatial import cKDTree

from host_paths import HOST_PATHS, host
from pebsi.io import mesh
from pebsi.io.terrain import Terrain
from project.glacierwide_loss import translate_rgi

SWEEPS = {
    'gulkana':   [1200, 600, 300, 150, 100, 80],
    'kennicott': [2000, 1200, 600, 375, 250],
}

p = HOST_PATHS[host]
for glacier, sweep in SWEEPS.items():
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
    rx, ry = dem.rio.resolution()
    dyg, dxg = np.gradient(dem.values, ry, rx)
    slope = np.rad2deg(np.arctan(np.sqrt(dxg ** 2 + dyg ** 2)))
    gx, gy = np.meshgrid(dem.x.values, dem.y.values)
    ok = shapely.contains_xy(poly, gx, gy) & np.isfinite(slope)
    px = np.column_stack([gx[ok], gy[ok]])
    pv = slope[ok]

    print('=' * 70)
    print(f'{glacier.upper()}  {poly.area / 1e6:.1f} km2, DEM {abs(rx):.1f} m, '
          f'{ok.sum()} pixels')
    print(f'raw pixel slope: mean {pv.mean():.3f} deg, std {pv.std():.3f} deg')
    print('=' * 70)
    print(f'{"h":>7} {"N":>7} {"h/DEM":>7} {"wtd mean":>10} {"wtd std":>9} '
          f'{"var captured":>14}')
    for h in sweep:
        xs, ys, w, _ = mesh.mesh_polygon(poly, float(h))
        pts = np.column_stack([xs, ys])
        v, cnt = Terrain.cell_mean(pts, px, pv)
        empty = ~np.isfinite(v)
        if empty.any():
            v[empty] = pv[cKDTree(px).query(pts[empty], workers=-1)[1]]
        m = np.average(v, weights=w)
        s = np.sqrt(np.average((v - m) ** 2, weights=w))
        print(f'{h:>7} {len(xs):>7} {h / abs(rx):>7.1f} {m:>10.3f} {s:>9.3f} '
              f'{100 * s ** 2 / pv.var():>13.1f}%')
    print()
