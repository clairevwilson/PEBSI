"""
Two tests of why glacier-wide mass balance drifts as the mesh is refined.

A. Cell-averaging the DEM preserves the area-weighted mean elevation but
   destroys the elevation spread inside each cell. Reports both against h,
   so the preserved mean can be checked and the collapsing spread measured.

B. The bias from that collapse goes as b''(z) * within-cell variance, so
   its sign is the sign of the curvature of the modelled balance profile.
   Fits b(z) from the finest saved run. Convex is the only sign consistent
   with coarse meshes coming out more negative; concave falsifies it.

    python debug_convergence.py

@author: clairevwilson
"""
import glob
import os

import geopandas as gpd
import numpy as np
import rioxarray as rxr
import shapely
import xarray as xr
from pyproj import Transformer
from scipy.spatial import cKDTree

from host_paths import host, HOST_PATHS
from pebsi.io import mesh

GID = '01.00570'
REGION = '01_rgi60_Alaska'
OUTDIR = '/ocean/projects/ees260009p/cwilson4/Output/point_density_test/'
SPACINGS = [1200, 900, 700, 500, 400, 300, 200, 150, 100, 80]
FINEST = 80.0

paths = HOST_PATHS[host]

rgi_gdf = gpd.read_file(paths['rgi_fp'] + f'../{REGION}/{REGION}.shp')
polygon, metric_crs = mesh.glacier_polygon(rgi_gdf, GID)

print('Loading DEM...', flush=True)
dem = rxr.open_rasterio(paths['cop30_vrt_path']).squeeze().drop_vars('band')
bounds = rgi_gdf.loc[rgi_gdf['RGIId'] == 'RGI60-' + GID].to_crs(dem.rio.crs).total_bounds
pad = 0.02
dem = dem.rio.clip_box(bounds[0] - pad, bounds[1] - pad,
                       bounds[2] + pad, bounds[3] + pad)
dem = dem.rio.reproject(metric_crs)
nodata = dem.rio.nodata
if nodata is not None:
    dem = dem.where((dem != nodata) & np.isfinite(dem))

grid_x, grid_y = np.meshgrid(dem.x.values, dem.y.values)
elev_grid = dem.values
on_ice = shapely.contains_xy(polygon, grid_x, grid_y) & np.isfinite(elev_grid)

px = np.column_stack([grid_x[on_ice], grid_y[on_ice]])
pz = elev_grid[on_ice]
print(f'DEM: {np.abs(np.diff(dem.x.values)).mean():.1f} m, '
      f'{on_ice.sum()} on-ice pixels', flush=True)
print(f'true area-mean elevation      {pz.mean():8.2f} m')
print(f'true elevation std            {pz.std():8.2f} m')
print(f'true elevation range          {pz.min():.0f} to {pz.max():.0f} m')

print()
print('=' * 78)
print('A. WITHIN-CELL ELEVATION SPREAD vs h')
print('=' * 78)
print(f'{"h":>6} {"N":>6} {"wtd mean z":>11} {"err vs true":>12} '
      f'{"wtd within-cell std":>20} {"max wt ratio":>13}')

for h in SPACINGS:
    xs, ys, weights, diag = mesh.mesh_polygon(polygon, float(h))
    pts = np.column_stack([xs, ys])

    owner = cKDTree(pts).query(px, workers=-1)[1]
    count = np.bincount(owner, minlength=len(pts))
    total = np.bincount(owner, weights=pz, minlength=len(pts))
    cell_mean = total / np.maximum(count, 1)

    # within-cell variance, from E[z^2] - E[z]^2 per cell
    total_sq = np.bincount(owner, weights=pz ** 2, minlength=len(pts))
    cell_var = np.maximum(total_sq / np.maximum(count, 1) - cell_mean ** 2, 0.0)

    ok = count > 0
    wtd_mean = np.average(cell_mean[ok], weights=weights[ok])
    wtd_std = np.average(np.sqrt(cell_var[ok]), weights=weights[ok])

    print(f'{h:>6} {len(xs):>6} {wtd_mean:>11.2f} {wtd_mean - pz.mean():>+12.2f} '
          f'{wtd_std:>20.2f} {diag["max_weight_ratio"]:>13.1f}')

print()
print('The mean column should stay flat -- cell-averaging preserves it.')
print('The spread column is what collapses as h shrinks, and what the')
print('bias term is proportional to (squared).')

print()
print('=' * 78)
print(f'B. CURVATURE OF b(z), from the h={FINEST:.0f} run')
print('=' * 78)

run = sorted(glob.glob(os.path.join(OUTDIR, f'gulkana_h{FINEST}_wf2.5_*')))
assert run, f'no saved run for h={FINEST}'
ds = xr.open_zarr(os.path.join(run[-1], 'output.zarr'))
mb = ds.mass_balance.sum(dim='time').compute().values
print(f'{len(mb)} points from {run[-1]}')

xs, ys, weights, _ = mesh.mesh_polygon(polygon, FINEST)
assert len(xs) == len(mb), f'mesh gives {len(xs)} points, run has {len(mb)}'

# same elevation the model used: mean over the point's own cell
owner = cKDTree(np.column_stack([xs, ys])).query(px, workers=-1)[1]
count = np.bincount(owner, minlength=len(xs))
z = np.bincount(owner, weights=pz, minlength=len(xs)) / np.maximum(count, 1)

ok = count > 0
z, mb, w = z[ok], mb[ok], weights[ok]
order = np.argsort(z)
z, mb, w = z[order], mb[order], w[order]

print()
print(f'{"elev band":>16} {"n":>5} {"mean b":>10} {"db/dz":>12}')
edges = np.percentile(z, np.linspace(0, 100, 11))
centres, means = [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (z >= lo) & (z <= hi)
    if m.sum() < 2:
        continue
    centres.append(z[m].mean())
    means.append(np.average(mb[m], weights=w[m]))
centres, means = np.array(centres), np.array(means)
grads = np.gradient(means, centres)
for c, mn, g in zip(centres, means, grads):
    print(f'{c:>16.0f} {"":>5} {mn:>10.3f} {g:>12.5f}')

coef = np.polyfit(z, mb, 2)
print()
print(f'quadratic fit b(z) = {coef[0]:.3e} z^2 + {coef[1]:.3e} z + {coef[2]:.3f}')
curv = 2 * coef[0]
print(f'curvature b\'\'(z) = {curv:+.3e} m w.e. per m^2')
print(f'  -> {"CONVEX" if curv > 0 else "CONCAVE"}')
print()
if curv > 0:
    print('CONVEX: collapsing within-cell spread makes coarse meshes too')
    print('negative, which is the direction the sweep actually drifts.')
else:
    print('CONCAVE: this predicts coarse meshes come out too POSITIVE,')
    print('which is the opposite of the sweep. Mechanism is falsified.')

# predicted bias at each h, from 0.5 * b'' * within-cell variance
print()
print(f'{"h":>6} {"wtd var":>12} {"predicted bias":>16}')
for h in SPACINGS:
    xs_h, ys_h, w_h, _ = mesh.mesh_polygon(polygon, float(h))
    own = cKDTree(np.column_stack([xs_h, ys_h])).query(px, workers=-1)[1]
    cnt = np.bincount(own, minlength=len(xs_h))
    tot = np.bincount(own, weights=pz, minlength=len(xs_h))
    cm = tot / np.maximum(cnt, 1)
    tsq = np.bincount(own, weights=pz ** 2, minlength=len(xs_h))
    cv = np.maximum(tsq / np.maximum(cnt, 1) - cm ** 2, 0.0)
    m = cnt > 0
    wvar = np.average(cv[m], weights=w_h[m])
    print(f'{h:>6} {wvar:>12.1f} {0.5 * curv * wvar:>+16.4f}')

print()
print('Predicted bias is what cell-averaging costs at each h, relative to')
print('resolving the sub-cell elevation range. Compare its span across the')
print('sweep against the ~0.44 m w.e. the sweep actually drifts.')

ds.close()
