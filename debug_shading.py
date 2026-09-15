"""
Checks whether the Voronoi-cell averaging of the shading fields in
Terrain.load_shading actually fires for gulkana at h=300, and whether it
produces anything different from the nearest-pixel sampling it replaced.

Skips the model entirely: opens the shading zarr, meshes the glacier, and
runs the same logic load_shading does.

    python debug_shading.py
"""
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr
from pyproj import Transformer
from scipy.spatial import cKDTree

from host_paths import host, HOST_PATHS
from pebsi.io import mesh
from pebsi.io.terrain import Terrain

GID = '01.00570'
SPACING = 300.0

paths = HOST_PATHS[host]
shade_fn = paths['shading_fp'] + f'{GID}_shadows.zarr'

print('=' * 70)
print('SHADING ZARR')
print('=' * 70)
ds = xr.open_zarr(shade_fn)
ds = ds.rio.set_spatial_dims(x_dim='x', y_dim='y')
ds = ds.rio.write_crs(ds['spatial_ref'].attrs['crs_wkt'])
print(f'dims          {dict(ds.sizes)}')
print(f'crs           {ds.rio.crs}')
res_x = float(np.abs(np.diff(ds.x.values)).mean())
res_y = float(np.abs(np.diff(ds.y.values)).mean())
print(f'resolution    {res_x:.1f} x {res_y:.1f} m')
print(f'x range       {ds.x.values.min():.0f} to {ds.x.values.max():.0f}')
print(f'y range       {ds.y.values.min():.0f} to {ds.y.values.max():.0f}')
print(f'shadow_mask   dims={ds["shadow_mask"].dims} dtype={ds["shadow_mask"].dtype}')
print(f'sky_view      dims={ds["sky_view_factor"].dims}')
print(f'solar_azimuth dims={ds["solar_azimuth"].dims}')

print()
print('=' * 70)
print('MESH POINTS')
print('=' * 70)
region_name = '01_rgi60_Alaska'
rgi_gdf = gpd.read_file(paths['rgi_fp'] + f'../{region_name}/{region_name}.shp')
polygon, metric_crs = mesh.glacier_polygon(rgi_gdf, GID)
xs, ys, weights, diagnostics = mesh.mesh_polygon(polygon, SPACING)
lons, lats = mesh.to_latlon(xs, ys, metric_crs)
print(f'n_points      {len(xs)}')
print(f'mesh crs      {metric_crs.to_string()[:60]}')
print(f'spacing       {diagnostics["spacing"]:.0f} m')

# exactly what load_shading does to place the points on the shading grid
transformer = Transformer.from_crs('EPSG:4326', ds.rio.crs, always_xy=True)
x_pts, y_pts = transformer.transform(lons, lats)
points = np.column_stack([x_pts, y_pts])
print(f'points x      {points[:, 0].min():.0f} to {points[:, 0].max():.0f}')
print(f'points y      {points[:, 1].min():.0f} to {points[:, 1].max():.0f}')

print()
print('=' * 70)
print('ON-ICE PIXEL MASK  (decides averaging vs nearest-pixel fallback)')
print('=' * 70)
ice = (rgi_gdf.loc[rgi_gdf['RGIId'] == 'RGI60-' + GID]
       .to_crs(ds.rio.crs).union_all())
grid_x, grid_y = np.meshgrid(ds.x.values, ds.y.values)
on_ice = shapely.contains_xy(ice, grid_x, grid_y)
print(f'grid shape    {grid_x.shape}  ({grid_x.size} pixels)')
print(f'on_ice.any()  {on_ice.any()}   <-- False means it fell back')
print(f'on-ice px     {on_ice.sum()} of {on_ice.size} ({100 * on_ice.mean():.1f}%)')
print(f'px per point  {on_ice.sum() / len(points):.1f}')

if not on_ice.any():
    print('\nFELL BACK to nearest-pixel. Averaging never ran.')
    raise SystemExit

print()
print('=' * 70)
print('AVERAGED vs NEAREST-PIXEL')
print('=' * 70)
pixels = np.column_stack([grid_x[on_ice], grid_y[on_ice]])

# how many on-ice pixels each point actually owns
owner = cKDTree(points).query(pixels, workers=-1)[1]
count = np.bincount(owner, minlength=len(points))
print(f'pixels owned  min={count.min()} median={np.median(count):.0f} max={count.max()}')
print(f'empty cells   {(count == 0).sum()} of {len(points)}')

shadow_pix = ds['shadow_mask'].values[:, on_ice].astype(np.float64)
print(f'shadow_pix    {shadow_pix.shape}  (time, on-ice pixels)')

averaged = Terrain.cell_mean_timeseries(points, pixels, shadow_pix)
print(f'averaged      {averaged.shape}  (points, time)')

# what the old code would have produced
target_x = xr.DataArray(x_pts, dims='points')
target_y = xr.DataArray(y_pts, dims='points')
nearest = (ds['shadow_mask']
           .sel(y=target_y, x=target_x, method='nearest')
           .transpose('points', 'time').values.astype(np.float64))
print(f'nearest       {nearest.shape}')

interior = (averaged > 1e-9) & (averaged < 1 - 1e-9)
print()
print(f'averaged: fraction of entries strictly between 0 and 1 = {interior.mean():.4f}')
print('  (0.0 means averaging changed nothing -- every cell fully lit or fully shaded)')
print(f'averaged  mean={averaged.mean():.6f}  nearest mean={nearest.mean():.6f}')
print(f'mean |averaged - nearest| = {np.abs(averaged - nearest).mean():.6f}')
print(f'entries differing by >0.01 = {(np.abs(averaged - nearest) > 0.01).mean():.4f}')

# only daylight hours matter for melt, so check those separately
sun_up = ds['solar_zenith'].values < (np.pi / 2)
print()
print(f'daylight steps {sun_up.sum()} of {len(sun_up)}')
if sun_up.any():
    a_day = averaged[:, sun_up]
    n_day = nearest[:, sun_up]
    interior_day = (a_day > 1e-9) & (a_day < 1 - 1e-9)
    print(f'daylight: fraction strictly between 0 and 1 = {interior_day.mean():.4f}')
    print(f'daylight: mean sunlit  averaged={a_day.mean():.6f}  nearest={n_day.mean():.6f}')
    print(f'daylight: mean |diff| = {np.abs(a_day - n_day).mean():.6f}')
    print(f'daylight: per-point mean sunlit, averaged vs nearest:')
    pa, pn = a_day.mean(axis=1), n_day.mean(axis=1)
    print(f'   averaged  min={pa.min():.4f} median={np.median(pa):.4f} max={pa.max():.4f}')
    print(f'   nearest   min={pn.min():.4f} median={np.median(pn):.4f} max={pn.max():.4f}')
    print(f'   area-weighted sunlit: averaged={np.average(pa, weights=weights):.6f} '
          f'nearest={np.average(pn, weights=weights):.6f}')

print()
print('=' * 70)
print('SKY VIEW FACTOR')
print('=' * 70)
svf_avg, _ = Terrain.cell_mean(points, pixels, ds['sky_view_factor'].values[on_ice])
svf_near = (ds['sky_view_factor']
            .sel(y=target_y, x=target_x, method='nearest').values)
print(f'averaged  mean={np.nanmean(svf_avg):.6f}')
print(f'nearest   mean={np.nanmean(svf_near):.6f}')
print(f'mean |diff| = {np.nanmean(np.abs(svf_avg - svf_near)):.6f}')

ds.close()
