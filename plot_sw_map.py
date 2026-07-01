"""
2×2 plot of incoming shortwave radiation and terrain context.

Row 0 (top):    shadow map  |  DEM aspect
Row 1 (bottom): instantaneous SW (W/m2, April 20 08:00 local)  |  annual SW (J/m2)

Usage: python plot_sw_map.py
"""
import glob
import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import RBFInterpolator
from pyproj import CRS, Transformer
from shapely.geometry import mapping
import rasterio
import rasterio.features
import rasterio.transform
from rasterio.warp import reproject as rio_reproject, Resampling, calculate_default_transform

output_dir  = '../Output/big_gulkana_sim/'
shadow_fp   = '../data/shading/01.00570_shadows.zarr'
dem_fp      = '../DEMs/gulkana_dem.tif'
rgi_fp      = '/Users/cvw/local/RGI/rgi60/01_rgi60_Alaska/01_rgi60_Alaska.shp'

INSTANT_MONTH, INSTANT_DAY, INSTANT_HOUR = 4, 20, 17  # UTC hour matching 08:00 local

# ===================== LOAD SW POINT DATA =====================
fns = sorted(glob.glob(f'{output_dir}/*.zarr'))
assert len(fns) > 0, f'No zarr files found in {output_dir}'

lats, lons, vals_instant, vals_annual, rgi_ids = [], [], [], [], set()

for fn in fns:
    ds = xr.open_zarr(fn, consolidated=False)

    t_mask = (
        (ds.time.dt.month == INSTANT_MONTH) &
        (ds.time.dt.day   == INSTANT_DAY) &
        (ds.time.dt.hour  == INSTANT_HOUR)
    )
    sw_slice = ds['shortwave_in'].where(t_mask, drop=True)
    vals_instant.append(float(sw_slice.values[0]) if sw_slice.size > 0 else 0.0)
    vals_annual.append(float(ds['shortwave_in'].sum('time').values) * 1e-6)  # W→MJ/m²

    lats.append(float(ds.attrs['lat']))
    lons.append(float(ds.attrs['lon']))
    rgi_ids.add(ds.attrs['id'])
    ds.close()

lats        = np.array(lats)
lons        = np.array(lons)
vals_instant = np.array(vals_instant)
vals_annual  = np.array(vals_annual)

# ===================== SW CRS + KRIGING GRID =====================
clon = float(np.mean(lons))
clat = float(np.mean(lats))
metric_crs = CRS(f'+proj=laea +lat_0={clat:.2f} +lon_0={clon:.2f} +datum=WGS84 +units=m')

to_metric = Transformer.from_crs('EPSG:4326', metric_crs, always_xy=True)
xs, ys = to_metric.transform(lons, lats)

gdf = gpd.read_file(rgi_fp)
glacier = gdf[gdf['RGIId'].isin(['RGI60-' + i for i in rgi_ids])].to_crs(metric_crs)

bounds  = glacier.total_bounds
res_sw  = 200  # meters
grid_x  = np.arange(bounds[0], bounds[2], res_sw)
grid_y  = np.arange(bounds[1], bounds[3], res_sw)
xx, yy  = np.meshgrid(grid_x, grid_y)
pts     = np.column_stack([xs, ys])
query   = np.column_stack([xx.ravel(), yy.ravel()])

shapes       = [(mapping(geom), 1) for geom in glacier.geometry]
sw_transform = rasterio.transform.from_bounds(*bounds, len(grid_x), len(grid_y))
glacier_mask = np.flipud(
    rasterio.features.rasterize(shapes, out_shape=(len(grid_y), len(grid_x)), transform=sw_transform)
)


def krige(values):
    rbf = RBFInterpolator(pts, values, kernel='thin_plate_spline')
    z = rbf(query).reshape(xx.shape)
    return np.where(glacier_mask == 1, z, np.nan)


z_instant = krige(vals_instant)
z_annual  = krige(vals_annual)

# ===================== SHADOW RASTER =====================
shad_ds = xr.open_zarr(shadow_fp, consolidated=False)

# read CRS from spatial_ref CF variable
sr_attrs = dict(shad_ds['spatial_ref'].attrs)
if 'crs_wkt' in sr_attrs:
    shadow_crs = CRS.from_wkt(sr_attrs['crs_wkt'])
elif 'proj4' in sr_attrs:
    shadow_crs = CRS.from_proj4(sr_attrs['proj4'])
else:
    shadow_crs = metric_crs  # fallback: assume same projection

# select the target hour
shad_t = shad_ds.sel(
    time=f'2000-{INSTANT_MONTH:02d}-{INSTANT_DAY:02d}T{INSTANT_HOUR:02d}:00',
    method='nearest'
)
shadow_grid = -1 * (shad_t['shadow_mask'].values.astype(float) - 1)  # (y, x)
xx_shad, yy_shad = np.meshgrid(shad_ds.x.values, shad_ds.y.values)

glacier_shad = glacier.to_crs(shadow_crs)

# ===================== DEM ASPECT (reprojected to metric_crs) =====================
with rasterio.open(dem_fp) as src:
    dst_transform, dst_w, dst_h = calculate_default_transform(
        src.crs, metric_crs, src.width, src.height, *src.bounds
    )
    dem = np.full((dst_h, dst_w), np.nan, dtype=np.float64)
    rio_reproject(
        source=src.read(1).astype(np.float64),
        destination=dem,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=dst_transform,
        dst_crs=metric_crs,
        resampling=Resampling.bilinear,
        src_nodata=src.nodata,
        dst_nodata=np.nan,
    )

res_x = dst_transform.a
res_y = -dst_transform.e  # positive pixel height
left  = dst_transform.c
top   = dst_transform.f
x_dem = left + res_x * (np.arange(dst_w) + 0.5)
y_dem = top  - res_y * (np.arange(dst_h) + 0.5)

# rows increase southward → northward slope = -grad_row
grad_row, grad_col = np.gradient(dem, res_y, res_x)
aspect = np.degrees(np.arctan2(-grad_col, grad_row))
aspect = (aspect + 360) % 360
aspect[np.isnan(dem)] = np.nan

xx_dem, yy_dem = np.meshgrid(x_dem, y_dem)

# ===================== PLOT =====================
fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))

# ---- top-left: shadow mask ----
ax = axes[0, 0]
shadow_cmap = mcolors.ListedColormap(['lightyellow', 'darkgray'])
im_shad = ax.pcolormesh(xx_shad, yy_shad, shadow_grid,
                         cmap=shadow_cmap, vmin=0, vmax=1, shading='auto')
glacier_shad.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.0)
fig.colorbar(im_shad, ax=ax, ticks=[0.25, 0.75], fraction=0.03, pad=0.03).set_ticklabels(['sunlit', 'shadow'])
ax.set_title(f'Shadow mask — April 20, {INSTANT_HOUR:02d}:00 UTC')
ax.set_xlabel('Easting [m]')
ax.set_ylabel('Northing [m]')
ax.set_aspect('equal')

# ---- top-right: DEM aspect ----
ax = axes[0, 1]
im_asp = ax.pcolormesh(xx_dem, yy_dem, aspect,
                        cmap='twilight_r', vmin=0, vmax=360, shading='auto')
glacier.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.0)
cbar_asp = fig.colorbar(im_asp, ax=ax, label='Aspect [°]', fraction=0.03, pad=0.03)
cbar_asp.set_ticks([0, 90, 180, 270, 360])
cbar_asp.set_ticklabels(['N', 'E', 'S', 'W', 'N'])
ax.set_title('Terrain aspect')
ax.set_xlabel('Easting [m]')
ax.set_ylabel('Northing [m]')
ax.set_aspect('equal')

# ---- bottom panels: kriged SW ----
sw_panels = [
    dict(ax=axes[1, 0], z=z_instant, vals=vals_instant,
         label='Incoming SW [W m$^{-2}$]',
         title=f'Instantaneous SW — April 20, {INSTANT_HOUR:02d}:00 UTC'),
    dict(ax=axes[1, 1], z=z_annual, vals=vals_annual,
         label='Incoming SW [MJ m$^{-2}$]',
         title='Cumulative annual SW'),
]

for p in sw_panels:
    ax   = p['ax']
    vmin = np.nanpercentile(p['vals'], 1)
    vmax = np.nanpercentile(p['vals'], 95)

    im = ax.pcolormesh(xx, yy, p['z'], cmap='YlOrRd',
                       vmin=vmin, vmax=vmax, shading='auto')
    glacier.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.0)
    # ax.scatter(xs, ys, c=p['vals'], cmap='YlOrRd',
    #            vmin=vmin, vmax=vmax,
    #            s=10, edgecolors=None, linewidths=0.3, zorder=5)
    fig.colorbar(im, ax=ax, label=p['label'], fraction=0.03, pad=0.03)
    ax.set_title(p['title'])
    ax.set_xlabel('Easting [m]')
    ax.set_ylabel('Northing [m]')
    ax.set_aspect('equal')

for ax in axes.flatten():
    ax.set_xlim(-3500, 3500)
    ax.set_ylim(-3200, 2400)
    ax.set_xticks((-2500, 0, 2500))
    ax.set_yticks((-2000, 0, 2000))

rgi_label = ', '.join(sorted(rgi_ids))
fig.suptitle(f'Gulkana Glacier — {len(fns)} simulation points ({rgi_label})', fontsize=13)

plt.tight_layout()
plt.subplots_adjust(hspace=0.3)
out_path = f'{output_dir}/shortwave_map.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.show()
print(f'saved to {out_path}')
