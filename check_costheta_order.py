"""
Tests whether cos_theta being computed from cell-averaged slope/aspect
(current order) instead of averaged after being computed per-pixel
(correct order) accounts for part of the residual mesh-convergence drift.

cos_theta = cos(beta)cos(zen) + sin(beta)sin(zen)cos(az - alpha)

is nonlinear and non-monotonic in slope (beta) and aspect (alpha), so
cos_theta(mean(beta), mean(alpha)) != mean(cos_theta(beta, alpha)) over a
cell straddling two different facets. energybalance.py computes it from
point_attrs.slope/aspect, which are already cell-averaged (terrain.py) --
confirmed by reading the code before running this.

Builds a per-pixel, daylight-weighted time-mean cos_theta at native DEM
resolution, then compares two ways of getting a per-point value:

  current   cell-mean slope/aspect -> cos_theta from the means
  correct   cos_theta per pixel -> cell-mean of cos_theta

at a coarse (h=1200) and a reference (h=300) mesh, ablation zone only,
matching the DEM-smoothing test's 0.0244 h=300->h=1200 gap. Solar
position is pulled from the shading zarr, which already stores it as
glacier-uniform (no x/y dims) rather than per-pixel.

Sensitivity of b to cos_theta is fit from the saved h=300 run so the
gap can be converted to an MB-equivalent number, comparable to the
0.0133 DEM-smoothing result.

    python check_costheta_order.py

@author: clairevwilson
"""
import glob
import os

import geopandas as gpd
import numpy as np
import rioxarray as rxr
import shapely
import xarray as xr
from scipy.spatial import cKDTree

from host_paths import HOST_PATHS, host
from pebsi.io import mesh
from pebsi.io.terrain import Terrain
from point_density_convergence import OUTDIR
from project.glacierwide_loss import translate_rgi

GLACIER = 'gulkana'
COARSE, REFERENCE = 1200.0, 300.0
ABLATION_MAX = 1500.0
YEARS = 3.0

p = HOST_PATHS[host]
gid = translate_rgi[GLACIER]['6']
region = gid.split('.')[0]
names = [f.split('.')[0] for f in os.listdir(p['rgi_fp']) if f.startswith(region)]
gdf = gpd.read_file(f"{p['rgi_fp']}../{names[0]}/{names[0]}.shp")
poly, crs = mesh.glacier_polygon(gdf, gid)
glacier = gdf.loc[gdf['RGIId'] == 'RGI60-' + gid]

print('loading DEM...', flush=True)
dem = rxr.open_rasterio(p['cop30_vrt_path']).squeeze().drop_vars('band')
b = glacier.to_crs(dem.rio.crs).total_bounds
dem = dem.rio.clip_box(b[0] - .02, b[1] - .02, b[2] + .02, b[3] + .02)
dem = dem.rio.reproject(crs)
if dem.rio.nodata is not None:
    dem = dem.where((dem != dem.rio.nodata) & np.isfinite(dem))
rx, ry = dem.rio.resolution()
dyg, dxg = np.gradient(dem.values, ry, rx)
slope_grid = np.arctan(np.sqrt(dxg ** 2 + dyg ** 2))                       # rad
aspect_grid = np.arctan2(-dyg, -dxg) % (2 * np.pi)                         # rad
gx, gy = np.meshgrid(dem.x.values, dem.y.values)
ok = shapely.contains_xy(poly, gx, gy) & np.isfinite(dem.values)
px = np.column_stack([gx[ok], gy[ok]])
pz, pslope, paspect = dem.values[ok], slope_grid[ok], aspect_grid[ok]
print(f'  {ok.sum()} on-ice pixels', flush=True)

print('loading solar position from the shading zarr...', flush=True)
ds = xr.open_zarr(p['shading_fp'] + f'{gid}_shadows.zarr')
zen = ds['solar_zenith'].values     # rad, (time,), glacier-uniform
az = ds['solar_azimuth'].values     # rad, (time,)
ds.close()
daylight = zen < (np.pi / 2)
zen, az = zen[daylight], az[daylight]
weight_t = np.clip(np.cos(zen), 0, None)     # potential-insolation weight
weight_t /= weight_t.sum()
print(f'  {daylight.sum()} daylight hours of {len(daylight)}', flush=True)


def cos_theta(slope, aspect, zen, az):
    """slope, aspect: (P,); zen, az: (T,) -> (P, T)"""
    return (np.cos(zen)[None, :] * np.cos(slope)[:, None]
            + np.sin(zen)[None, :] * np.sin(slope)[:, None]
            * np.cos(az[None, :] - aspect[:, None]))


print('computing per-pixel time-mean cos_theta (correct order)...', flush=True)
ct_pixel = cos_theta(pslope, paspect, zen, az) @ weight_t   # (P,)

print()
meshes = {}
for h in (COARSE, REFERENCE):
    xs, ys, w, _ = mesh.mesh_polygon(poly, h)
    pts = np.column_stack([xs, ys])

    z_pt, _ = Terrain.cell_mean(pts, px, pz)
    slope_pt, _ = Terrain.cell_mean(pts, px, np.rad2deg(pslope))
    aspect_pt, _ = Terrain.cell_mean(pts, px, np.rad2deg(paspect), circular=True)
    empty = ~np.isfinite(z_pt)
    if empty.any():
        nn = cKDTree(px).query(pts[empty], workers=-1)[1]
        z_pt[empty] = pz[nn]
        slope_pt[empty] = np.rad2deg(pslope)[nn]
        aspect_pt[empty] = np.rad2deg(paspect)[nn]

    # current order: cos_theta from the model's own cell-mean slope/aspect
    ct_current = cos_theta(np.deg2rad(slope_pt), np.deg2rad(aspect_pt),
                           zen, az) @ weight_t

    # correct order: cos_theta computed per pixel, then cell-averaged
    ct_correct, _ = Terrain.cell_mean(pts, px, ct_pixel)
    if empty.any():
        ct_correct[empty] = ct_pixel[nn]

    meshes[h] = dict(pts=pts, w=w, z=z_pt, current=ct_current,
                     correct=ct_correct)
    print(f'h={h:.0f}: {len(w)} points', flush=True)

print()
print('=' * 78)
print(f'{GLACIER.upper()}   ablation zone (elev < {ABLATION_MAX:.0f} m)   '
      f'daylight-weighted mean cos_theta')
print('=' * 78)

fpts, fw, fz = (meshes[REFERENCE][k] for k in ('pts', 'w', 'z'))
cpts = meshes[COARSE]['pts']
abl = fz < ABLATION_MAX
owner = cKDTree(cpts).query(fpts[abl], workers=-1)[1]
area = fw[abl].sum()
print(f'{fw[abl].sum():.4f} of area, using h={REFERENCE:.0f} as the fine reference')
print()
print(f'{"":>10} {"current (model)":>16} {"correct":>10} {"gap closed":>12}')
for label, key in (('coarse', 'current'), ('coarse', 'correct')):
    pass
cur_coarse = meshes[COARSE]['current'][owner]
cor_coarse = meshes[COARSE]['correct'][owner]
cur_fine = meshes[REFERENCE]['current'][abl]
cor_fine = meshes[REFERENCE]['correct'][abl]

cm_cur = np.average(cur_coarse, weights=fw[abl])
cm_cor = np.average(cor_coarse, weights=fw[abl])
fm_cur = np.average(cur_fine, weights=fw[abl])
fm_cor = np.average(cor_fine, weights=fw[abl])

gap_current = cm_cur - fm_cur
gap_correct = cm_cor - fm_cor
print(f'{"coarse":>10} {cm_cur:>16.4f} {cm_cor:>10.4f}')
print(f'{"reference":>10} {fm_cur:>16.4f} {fm_cor:>10.4f}')
print(f'{"gap":>10} {gap_current:>+16.4f} {gap_correct:>+10.4f} '
      f'{100 * (1 - abs(gap_correct) / abs(gap_current)):>11.1f}%')

print()
print('=' * 78)
print('SENSITIVITY OF b TO cos_theta, from the saved h=300 run')
print('=' * 78)
run = sorted(glob.glob(os.path.join(OUTDIR, f'{GLACIER}_h{REFERENCE}_wf2.5_*')))
rds = xr.open_zarr(os.path.join(run[-1], 'output.zarr'))
mb = rds.mass_balance.sum(dim='time').compute().values
rds.close()
assert len(mb) == len(fw), f'run has {len(mb)} points, mesh gives {len(fw)}'

# elevation removed first, same approach as the earlier attribution script
A = np.column_stack([fz[abl], np.ones(abl.sum())])
resid = mb[abl] - A @ np.linalg.lstsq(
    A * np.sqrt(fw[abl])[:, None], mb[abl] * np.sqrt(fw[abl]), rcond=None)[0]
sens = np.polyfit(cur_fine, resid, 1, w=np.sqrt(fw[abl]))[0]
print(f'db/d(cos_theta) = {sens:.4f} m w.e. (total, {YEARS:.0f} yr) per unit cos_theta')

mb_effect_current = sens * gap_current * area / YEARS
mb_effect_correct = sens * gap_correct * area / YEARS
print()
print(f'MB effect of the coarse-vs-reference cos_theta gap, current order: '
      f'{mb_effect_current:+.4f} m w.e./yr')
print(f'MB effect with the correct order:                                 '
      f'{mb_effect_correct:+.4f} m w.e./yr')
print(f'implied share of the 0.0244 m w.e./yr h=300->h=1200 residual: '
      f'{100 * abs(mb_effect_current - mb_effect_correct) / 0.0244:.1f}%')
print()
print('Compare against the DEM-smoothing result: 0.0133 of 0.0244 explained')
print('by terrain roughness broadly. This isolates just the ordering of the')
print('cos_theta calculation, holding slope/aspect variance itself fixed.')
