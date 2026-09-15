"""
Attributes the coarse-mesh balance drift to individual model inputs.

Fixes three problems with debug_inputs.py:

  - the ice albedo pixel grid was left in the raster's own CRS while the
    points were in the metric CRS, so the albedo row dropped out entirely
  - empty cells were masked out instead of falling back to the pixel the
    point sits on, which is what get_ice_albedo and get_wind_fields do,
    biasing every sparse raster at fine resolution
  - the ablation zone was selected on cell-mean elevation, so coarse
    cells straddling the cut dragged in higher ground

Here each fine point is mapped into the coarse cell that contains it, so
the two meshes are compared at the same places, and the ablation zone is
defined by fine-resolution elevation.

Then the finest run's own balances give the sensitivity of b to each
input, and sensitivity times the coarse-to-fine shift in that input says
how much of the drift each one accounts for.

    python debug_attribution.py

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
from pebsi.io.terrain import Terrain

GID = '01.00570'
REGION = '01_rgi60_Alaska'
OUTDIR = '/ocean/projects/ees260009p/cwilson4/Output/point_density_test/'
COARSE, FINE = 1200.0, 80.0
ABLATION_MAX = 1500.0

paths = HOST_PATHS[host]
rgi_gdf = gpd.read_file(paths['rgi_fp'] + f'../{REGION}/{REGION}.shp')
polygon, metric_crs = mesh.glacier_polygon(rgi_gdf, GID)
glacier = rgi_gdf.loc[rgi_gdf['RGIId'] == 'RGI60-' + GID]


def to_metric(x, y, src_crs):
    """Puts a raster's pixel centres into the mesh's metric CRS."""
    tr = Transformer.from_crs(src_crs, metric_crs, always_xy=True)
    return tr.transform(x, y)


def cell_mean(pts, px, vals, circular=False):
    """cell_mean with the nearest-pixel fallback the model uses."""
    m, _ = Terrain.cell_mean(pts, px, vals, circular=circular)
    empty = ~np.isfinite(m)
    if empty.any():
        m[empty] = vals[cKDTree(px).query(pts[empty], workers=-1)[1]]
    return m


print('loading DEM...', flush=True)
dem = rxr.open_rasterio(paths['cop30_vrt_path']).squeeze().drop_vars('band')
b = glacier.to_crs(dem.rio.crs).total_bounds
dem = dem.rio.clip_box(b[0] - 0.02, b[1] - 0.02, b[2] + 0.02, b[3] + 0.02)
dem = dem.rio.reproject(metric_crs)
if dem.rio.nodata is not None:
    dem = dem.where((dem != dem.rio.nodata) & np.isfinite(dem))
rx, ry = dem.rio.resolution()
dyg, dxg = np.gradient(dem.values, ry, rx)
slope_grid = np.rad2deg(np.arctan(np.sqrt(dxg ** 2 + dyg ** 2)))
aspect_grid = np.rad2deg((np.arctan2(-dyg, -dxg) + 2 * np.pi) % (2 * np.pi)) % 360
gx, gy = np.meshgrid(dem.x.values, dem.y.values)
ok = shapely.contains_xy(polygon, gx, gy) & np.isfinite(dem.values) & np.isfinite(slope_grid)
dem_px = np.column_stack([gx[ok], gy[ok]])
dem_z, dem_slope, dem_aspect = dem.values[ok], slope_grid[ok], aspect_grid[ok]
print(f'  {ok.sum()} pixels', flush=True)

print('loading ice albedo...', flush=True)
alb = rxr.open_rasterio(paths['ice_albedo_fn'].format(gid=GID)).squeeze().drop_vars('band')
av = alb.values.astype(float)
if alb.rio.nodata is not None:
    av = np.where(av == alb.rio.nodata, np.nan, av)
agx, agy = np.meshgrid(alb.x.values, alb.y.values)
agx, agy = to_metric(agx, agy, alb.rio.crs)          # the CRS fix
aok = np.isfinite(av)
alb_px, alb_v = np.column_stack([agx[aok], agy[aok]]), av[aok]
print(f'  {aok.sum()} pixels', flush=True)

print('loading shading...', flush=True)
ds = xr.open_zarr(paths['shading_fp'] + f'{GID}_shadows.zarr')
ds = ds.rio.set_spatial_dims(x_dim='x', y_dim='y')
ds = ds.rio.write_crs(ds['spatial_ref'].attrs['crs_wkt'])
sgx, sgy = np.meshgrid(ds.x.values, ds.y.values)
sgx, sgy = to_metric(sgx, sgy, ds.rio.crs)
sok = shapely.contains_xy(polygon, sgx, sgy)
shade_px = np.column_stack([sgx[sok], sgy[sok]])
svf_v = ds['sky_view_factor'].values[sok]
sun_up = ds['solar_zenith'].values < (np.pi / 2)
sunlit_v = ds['shadow_mask'].values[sun_up][:, sok].mean(axis=0)
ds.close()
print(f'  {sok.sum()} pixels', flush=True)

print('loading wind...', flush=True)
wds = xr.open_dataset(paths['windmap_fn'].format(gid=GID))
wlon, wlat = np.meshgrid(wds['lon'].values, wds['lat'].values)
wgx, wgy = to_metric(wlon, wlat, 'EPSG:4326')
sp = wds['spdup'].transpose('y', 'x', 'direction').values
wok = np.isfinite(sp).all(axis=2)
wind_px, wind_v = np.column_stack([wgx[wok], wgy[wok]]), sp[wok].mean(axis=1)
wds.close()
print(f'  {wok.sum()} cells', flush=True)

FIELDS = ['elev', 'slope', 'southness', 'ice_albedo', 'sky_view', 'sunlit',
          'wind_spdup']


def inputs_for(h):
    xs, ys, w, _ = mesh.mesh_polygon(polygon, h)
    pts = np.column_stack([xs, ys])
    aspect = cell_mean(pts, dem_px, dem_aspect, circular=True)
    return pts, w, {
        'elev':       cell_mean(pts, dem_px, dem_z),
        'slope':      cell_mean(pts, dem_px, dem_slope),
        'southness': -np.cos(np.deg2rad(aspect)),
        'ice_albedo': cell_mean(pts, alb_px, alb_v),
        'sky_view':   cell_mean(pts, shade_px, svf_v),
        'sunlit':     cell_mean(pts, shade_px, sunlit_v),
        'wind_spdup': cell_mean(pts, wind_px, wind_v),
    }


cpts, cw, cin = inputs_for(COARSE)
fpts, fw, fin = inputs_for(FINE)
print(f'\ncoarse {len(cw)} points, fine {len(fw)} points')

# every fine point takes the value its containing coarse cell was given
owner = cKDTree(cpts).query(fpts, workers=-1)[1]
abl = fin['elev'] < ABLATION_MAX
print(f'ablation zone: {abl.sum()} fine points, '
      f'{fw[abl].sum():.4f} of area')

print()
print('=' * 84)
print(f'SAME PLACES, ABLATION ZONE (fine elev < {ABLATION_MAX:.0f} m)')
print('=' * 84)
print(f'{"input":>12} {"coarse":>11} {"fine":>11} {"shift":>10} '
      f'{"coarse std":>12} {"fine std":>10} {"std ratio":>10}')
shift = {}
for f in FIELDS:
    cv, fv, w = cin[f][owner][abl], fin[f][abl], fw[abl]
    cm = np.average(cv, weights=w)
    fm = np.average(fv, weights=w)
    cs = np.sqrt(np.average((cv - cm) ** 2, weights=w))
    fs = np.sqrt(np.average((fv - fm) ** 2, weights=w))
    shift[f] = cm - fm
    print(f'{f:>12} {cm:>11.4f} {fm:>11.4f} {cm - fm:>+10.4f} '
          f'{cs:>12.4f} {fs:>10.4f} {cs / fs if fs > 0 else np.nan:>10.3f}')

print()
print('=' * 84)
print('SENSITIVITY OF b TO EACH INPUT, AND ATTRIBUTION')
print('=' * 84)

run = sorted(glob.glob(os.path.join(OUTDIR, f'gulkana_h{FINE}_wf2.5_*')))
rds = xr.open_zarr(os.path.join(run[-1], 'output.zarr'))
mb = rds.mass_balance.sum(dim='time').compute().values
rds.close()
assert len(mb) == len(fw), f'run has {len(mb)} points, mesh gives {len(fw)}'

# weighted least squares of b on the inputs, ablation zone only
X = np.column_stack([fin[f][abl] for f in FIELDS] + [np.ones(abl.sum())])
y = mb[abl]
sw = np.sqrt(fw[abl])
coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)

pred = X @ coef
ss_res = np.average((y - pred) ** 2, weights=fw[abl])
ss_tot = np.average((y - np.average(y, weights=fw[abl])) ** 2, weights=fw[abl])
print(f'weighted R^2 of the fit = {1 - ss_res / ss_tot:.4f}')
print(f'ablation zone area fraction = {fw[abl].sum():.4f}')
print()
print(f'{"input":>12} {"db/dinput":>14} {"shift":>11} {"db in zone":>12} '
      f'{"MB contrib":>12}')
area = fw[abl].sum()
total = 0.0
for f, c in zip(FIELDS, coef[:-1]):
    contrib = c * shift[f] * area
    total += contrib
    print(f'{f:>12} {c:>14.4f} {shift[f]:>+11.4f} {c * shift[f]:>+12.4f} '
          f'{contrib:>+12.4f}')
print(f'{"TOTAL":>12} {"":>14} {"":>11} {"":>12} {total:>+12.4f}')

print()
print('MB contrib is each input\'s share of the coarse-mesh drift, from its')
print('own sensitivity times how far the coarse mesh moves it. The drift to')
print(f'explain at h={COARSE:.0f} is about -0.46 m w.e.')
print()
print('This attributes only the shift in the MEAN of each input. Any drift')
print('left over is the compression of their spread acting through')
print('curvature, which a linear fit cannot see.')
