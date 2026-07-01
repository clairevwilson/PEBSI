"""
Two-panel plot for the ablation season (April 15 – September 15).

Left:  fractional hours in shadow (from shadow zarr)
Right: mean direct shortwave radiation [W/m²] (kriged from big_gulkana_1 outputs)
"""
import glob
import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from pyproj import CRS, Transformer
from shapely.geometry import mapping
import rasterio
import rasterio.features
import rasterio.transform

output_dir = '../Output/big_gulkana_1/'
shadow_fp  = '../data/shading/01.00570_shadows.zarr'
rgi_fp     = '/Users/cvw/local/RGI/rgi60/01_rgi60_Alaska/01_rgi60_Alaska.shp'

# ===================== LOAD SIMULATION POINTS =====================
fns = sorted(glob.glob(f'{output_dir}/*.zarr'))
assert len(fns) > 0, f'No zarr files found in {output_dir}'

lats, lons, vals_sw, rgi_ids = [], [], [], set()

for fn in fns:
    ds = xr.open_zarr(fn, consolidated=False)

    season = (
        ((ds.time.dt.month == 4) & (ds.time.dt.day >= 15)) |
        (ds.time.dt.month.isin([5, 6, 7, 8])) |
        ((ds.time.dt.month == 9) & (ds.time.dt.day <= 15))
    )
    sw = ds['shortwave_direct'].where(season, drop=True)
    vals_sw.append(float(sw.mean('time').values))

    lats.append(float(ds.attrs['lat']))
    lons.append(float(ds.attrs['lon']))
    rgi_ids.add(ds.attrs['id'])
    ds.close()

lats   = np.array(lats)
lons   = np.array(lons)
vals_sw = np.array(vals_sw)

# ===================== CRS + GLACIER OUTLINE =====================
clon = float(np.mean(lons))
clat = float(np.mean(lats))
metric_crs = CRS(f'+proj=laea +lat_0={clat:.2f} +lon_0={clon:.2f} +datum=WGS84 +units=m')

to_metric = Transformer.from_crs('EPSG:4326', metric_crs, always_xy=True)
xs, ys = to_metric.transform(lons, lats)

gdf     = gpd.read_file(rgi_fp)
glacier = gdf[gdf['RGIId'].isin(['RGI60-' + i for i in rgi_ids])].to_crs(metric_crs)

# ===================== KRIGING GRID =====================
bounds = glacier.total_bounds
res    = 200
grid_x = np.arange(bounds[0], bounds[2], res)
grid_y = np.arange(bounds[1], bounds[3], res)
xx, yy = np.meshgrid(grid_x, grid_y)
pts    = np.column_stack([xs, ys])
query  = np.column_stack([xx.ravel(), yy.ravel()])

shapes        = [(mapping(geom), 1) for geom in glacier.geometry]
sw_transform  = rasterio.transform.from_bounds(*bounds, len(grid_x), len(grid_y))
glacier_mask  = np.flipud(
    rasterio.features.rasterize(shapes, out_shape=(len(grid_y), len(grid_x)), transform=sw_transform)
)

rbf    = RBFInterpolator(pts, vals_sw, kernel='thin_plate_spline')
z_sw   = rbf(query).reshape(xx.shape)
z_sw   = np.where(glacier_mask == 1, z_sw, np.nan)

# ===================== SHADOW FRACTION =====================
shad_ds = xr.open_zarr(shadow_fp, consolidated=False)
shad_ds = -1 * (shad_ds - 1)

season_shad = (
    ((shad_ds.time.dt.month == 4) & (shad_ds.time.dt.day >= 15)) |
    (shad_ds.time.dt.month.isin([5, 6, 7, 8])) |
    ((shad_ds.time.dt.month == 9) & (shad_ds.time.dt.day <= 15))
)

frac_shadow = (
    shad_ds['shadow_mask']
    .where(season_shad, drop=True)
    .mean('time')
    .values
    .astype(np.float32)
)

# reproject shadow fraction to metric_crs
sr_attrs = dict(shad_ds['spatial_ref'].attrs)
if 'crs_wkt' in sr_attrs:
    shadow_crs = CRS.from_wkt(sr_attrs['crs_wkt'])
elif 'proj4' in sr_attrs:
    shadow_crs = CRS.from_proj4(sr_attrs['proj4'])
else:
    shadow_crs = metric_crs

shad_x = shad_ds.x.values
shad_y = shad_ds.y.values
xx_shad, yy_shad = np.meshgrid(shad_x, shad_y)

glacier_shad = glacier.to_crs(shadow_crs)

# ===================== PLOT =====================
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# ---- left: shadow fraction ----
ax = axes[0]
im_shad = ax.pcolormesh(xx_shad, yy_shad, frac_shadow,
                         cmap='Blues', vmin=0.4, vmax=0.6, shading='auto')
glacier_shad.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.0)
fig.colorbar(im_shad, ax=ax, label='Fraction of hours in shadow', fraction=0.03, pad=0.03)
ax.set_title('Shading')
ax.set_xlabel('Easting [m]')
ax.set_ylabel('Northing [m]')
ax.set_aspect('equal')

# ---- right: mean direct SW ----
ax = axes[1]
vmin = np.nanpercentile(vals_sw, 1)
vmax = np.nanpercentile(vals_sw, 95)
im_sw = ax.pcolormesh(xx, yy, z_sw, cmap='YlOrRd',
                       vmin=vmin, vmax=vmax, shading='auto')
glacier.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.0)
ax.scatter(xs, ys, c=vals_sw, cmap='YlOrRd', vmin=vmin, vmax=vmax,
           s=15, edgecolors='k', linewidths=0.3, zorder=5)
fig.colorbar(im_sw, ax=ax, label='Mean direct SW [W m$^{-2}$]', fraction=0.03, pad=0.03)
ax.set_title('Mean direct shortwave')
ax.set_xlabel('Easting [m]')
ax.set_ylabel('Northing [m]')
ax.set_aspect('equal')

for ax in axes:
    ax.set_xlim(-3300, 3300)
    ax.set_ylim(-3100, 2300)

rgi_label = ', '.join(sorted(rgi_ids))
fig.suptitle(f'Gulkana Glacier - ablation season (April 15 - September 15)', fontsize=13)

plt.tight_layout()
out_path = f'{output_dir}/shading_sw_map.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'saved to {out_path}')
