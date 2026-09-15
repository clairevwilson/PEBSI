"""
Compares every per-point model input between a coarse and a fine mesh.

Points are independent and each input is a mean-preserving cell average,
so the balance at a given elevation can only move with resolution if the
response is nonlinear in an input whose spread is being compressed.

Cell-averaging preserves each input's area-weighted mean but collapses
its spread, by a factor that grows with cell size. This reports both,
glacier-wide and over the ablation zone alone, where the whole drift
lives.

  mean differs    the input itself is biased by resolution
  spread differs  the input is unbiased, and any effect on balance has
                  to come through curvature of the response to it

    python debug_inputs.py

@author: clairevwilson
"""
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
COARSE, FINE = 1200.0, 80.0
ABLATION_MAX = 1500.0

paths = HOST_PATHS[host]
rgi_gdf = gpd.read_file(paths['rgi_fp'] + f'../{REGION}/{REGION}.shp')
polygon, metric_crs = mesh.glacier_polygon(rgi_gdf, GID)
glacier = rgi_gdf.loc[rgi_gdf['RGIId'] == 'RGI60-' + GID]

# ---------------------------------------------------------------- DEM
print('loading DEM...', flush=True)
dem = rxr.open_rasterio(paths['cop30_vrt_path']).squeeze().drop_vars('band')
b = glacier.to_crs(dem.rio.crs).total_bounds
dem = dem.rio.clip_box(b[0] - 0.02, b[1] - 0.02, b[2] + 0.02, b[3] + 0.02)
dem = dem.rio.reproject(metric_crs)
nodata = dem.rio.nodata
if nodata is not None:
    dem = dem.where((dem != nodata) & np.isfinite(dem))

res_x, res_y = dem.rio.resolution()
dy_g, dx_g = np.gradient(dem.values, res_y, res_x)
slope_grid = np.rad2deg(np.arctan(np.sqrt(dx_g ** 2 + dy_g ** 2)))
aspect_grid = np.rad2deg((np.arctan2(-dy_g, -dx_g) + 2 * np.pi) % (2 * np.pi)) % 360

gx, gy = np.meshgrid(dem.x.values, dem.y.values)
ice = shapely.contains_xy(polygon, gx, gy)
usable = ice & np.isfinite(dem.values) & np.isfinite(slope_grid)
dem_px = np.column_stack([gx[usable], gy[usable]])
dem_z = dem.values[usable]
dem_slope = slope_grid[usable]
dem_aspect = aspect_grid[usable]
print(f'  {usable.sum()} on-ice DEM pixels at {abs(res_x):.1f} m', flush=True)

# -------------------------------------------------------------- albedo
print('loading ice albedo...', flush=True)
alb = rxr.open_rasterio(paths['ice_albedo_fn'].format(gid=GID)).squeeze().drop_vars('band')
alb_vals = alb.values.astype(float)
if alb.rio.nodata is not None:
    alb_vals = np.where(alb_vals == alb.rio.nodata, np.nan, alb_vals)
ax_g, ay_g = np.meshgrid(alb.x.values, alb.y.values)
alb_ok = np.isfinite(alb_vals)
alb_px = np.column_stack([ax_g[alb_ok], ay_g[alb_ok]])
alb_z = alb_vals[alb_ok]
print(f'  {alb_ok.sum()} albedo pixels, mean {alb_z.mean():.4f}, '
      f'std {alb_z.std():.4f}', flush=True)

# ------------------------------------------------------------- shading
print('loading shading...', flush=True)
ds = xr.open_zarr(paths['shading_fp'] + f'{GID}_shadows.zarr')
ds = ds.rio.set_spatial_dims(x_dim='x', y_dim='y')
ds = ds.rio.write_crs(ds['spatial_ref'].attrs['crs_wkt'])
sx_g, sy_g = np.meshgrid(ds.x.values, ds.y.values)
shade_ice = shapely.contains_xy(
    glacier.to_crs(ds.rio.crs).union_all(), sx_g, sy_g)
shade_px = np.column_stack([sx_g[shade_ice], sy_g[shade_ice]])
svf_px = ds['sky_view_factor'].values[shade_ice]
sun_up = ds['solar_zenith'].values < (np.pi / 2)
# time-mean sunlit fraction per pixel, daylight only
sunlit_px = ds['shadow_mask'].values[sun_up][:, shade_ice].mean(axis=0)
print(f'  {shade_ice.sum()} shading pixels, {sun_up.sum()} daylight steps',
      flush=True)

# ---------------------------------------------------------------- wind
print('loading wind...', flush=True)
wds = xr.open_dataset(paths['windmap_fn'].format(gid=GID))
wlon, wlat = np.meshgrid(wds['lon'].values, wds['lat'].values)
wtr = Transformer.from_crs('EPSG:4326', metric_crs, always_xy=True)
wx_g, wy_g = wtr.transform(wlon, wlat)
spdup = wds['spdup'].transpose('y', 'x', 'direction').values
wind_ok = np.isfinite(spdup).all(axis=2)
wind_px = np.column_stack([wx_g[wind_ok], wy_g[wind_ok]])
wind_z = spdup[wind_ok].mean(axis=1)   # direction-mean speed-up
wds.close()
print(f'  {wind_ok.sum()} wind cells', flush=True)


def inputs_for(h):
    """Every per-point input, built the way Terrain builds them."""
    xs, ys, w, _ = mesh.mesh_polygon(polygon, h)
    pts = np.column_stack([xs, ys])

    out = {'weight': w}
    out['elev'], _ = Terrain.cell_mean(pts, dem_px, dem_z)
    out['slope'], _ = Terrain.cell_mean(pts, dem_px, dem_slope)
    out['aspect'], _ = Terrain.cell_mean(pts, dem_px, dem_aspect, circular=True)
    out['ice_albedo'], _ = Terrain.cell_mean(pts, alb_px, alb_z)

    # shading grid is in its own CRS
    str_ = Transformer.from_crs(metric_crs, ds.rio.crs, always_xy=True)
    sxp, syp = str_.transform(xs, ys)
    spts = np.column_stack([sxp, syp])
    out['sky_view'], _ = Terrain.cell_mean(spts, shade_px, svf_px)
    out['sunlit'], _ = Terrain.cell_mean(spts, shade_px, sunlit_px)

    out['wind_spdup'], _ = Terrain.cell_mean(pts, wind_px, wind_z)
    return out


print()
coarse = inputs_for(COARSE)
fine = inputs_for(FINE)
print(f'coarse h={COARSE:.0f}: {len(coarse["weight"])} points')
print(f'fine   h={FINE:.0f}: {len(fine["weight"])} points')

FIELDS = ['elev', 'slope', 'aspect', 'ice_albedo', 'sky_view', 'sunlit',
          'wind_spdup']


def report(title, mask_fn):
    print()
    print('=' * 86)
    print(title)
    print('=' * 86)
    print(f'{"input":>12} {"coarse mean":>13} {"fine mean":>12} {"d mean":>10} '
          f'{"coarse std":>12} {"fine std":>10} {"std ratio":>11}')
    for f in FIELDS:
        cv, cw = coarse[f], coarse['weight']
        fv, fw = fine[f], fine['weight']
        cm_ = mask_fn(coarse) & np.isfinite(cv)
        fm_ = mask_fn(fine) & np.isfinite(fv)
        if cm_.sum() < 2 or fm_.sum() < 2:
            continue
        cmean = np.average(cv[cm_], weights=cw[cm_])
        fmean = np.average(fv[fm_], weights=fw[fm_])
        cstd = np.sqrt(np.average((cv[cm_] - cmean) ** 2, weights=cw[cm_]))
        fstd = np.sqrt(np.average((fv[fm_] - fmean) ** 2, weights=fw[fm_]))
        ratio = cstd / fstd if fstd > 0 else np.nan
        print(f'{f:>12} {cmean:>13.4f} {fmean:>12.4f} {cmean - fmean:>+10.4f} '
              f'{cstd:>12.4f} {fstd:>10.4f} {ratio:>11.3f}')


report('ALL POINTS', lambda d: np.ones(len(d['weight']), dtype=bool))
report(f'ABLATION ZONE ONLY (cell-mean elev < {ABLATION_MAX:.0f} m)',
       lambda d: d['elev'] < ABLATION_MAX)

for label, d in (('coarse', coarse), ('fine', fine)):
    m = d['elev'] < ABLATION_MAX
    print(f'{label:>7}: {m.sum()} points below {ABLATION_MAX:.0f} m, '
          f'carrying {d["weight"][m].sum():.4f} of area')

print()
print('A nonzero "d mean" in the ablation zone is a resolution-biased input')
print('and a direct cause. A "std ratio" well below 1 with d mean ~0 is an')
print('unbiased input whose spread is being compressed -- that can only move')
print('the balance through curvature of the response to it.')

ds.close()
