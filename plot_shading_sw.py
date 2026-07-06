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
import matplotlib.patheffects as pe
from scipy.interpolate import RBFInterpolator
from pyproj import CRS, Transformer
from shapely.geometry import mapping
import rasterio
import rasterio.features
import rasterio.transform
from rasterio.warp import reproject as rio_reproject, Resampling, calculate_default_transform

output_dir = '../Output/big_gulkana_1/'
shadow_fp  = '../data/shading/01.00570_shadows.zarr'
dem_fp     = '../DEMs/gulkana_dem.tif'
rgi_fp     = '/Users/cvw/local/RGI/rgi60/01_rgi60_Alaska/01_rgi60_Alaska.shp'

CONTOUR_INTERVAL = 100    # elevation contour spacing in metres (used when CONTOUR_SOURCE='elevation')
CONTOUR_SOURCE   = 'value'  # 'elevation'  →  DEM isolines
                                # 'value'      →  isolines of each panel's own data
N_VALUE_CONTOURS = 10            # number of contour lines when CONTOUR_SOURCE='value'
SCALEBAR_M       = 1000  # scale bar length in metres

# ===================== HELPERS =====================
stroke = [pe.withStroke(linewidth=2, foreground='white')]

def add_map_decorations(ax, length_m, x_frac=0.08, y_frac=0.06, color='black'):
    """Draw a scale bar and north arrow side-by-side at the bottom-left."""
    xl, xr = ax.get_xlim()
    yb, yt = ax.get_ylim()
    span   = xr - xl
    vspan  = yt - yb

    # scale bar
    x0    = xl + x_frac * span
    y0    = yb + y_frac * vspan
    bar_h = 0.006 * vspan
    ax.fill_between([x0, x0 + length_m], [y0, y0], [y0 + bar_h, y0 + bar_h],
                    color=color, linewidth=0, zorder=6)
    lbl = f'{length_m/1000:.0f} km' if length_m >= 1000 else f'{length_m:.0f} m'
    ax.text(x0 + length_m / 2, y0 + bar_h * 2.5, lbl,
            ha='center', va='bottom', color=color,
            path_effects=stroke, zorder=7)

    # north arrow — placed just to the right of the scale bar with a small gap
    gap   = 0.015 * span
    x_arr = x0 + length_m + gap + 0.015 * span
    dy    = 0.10 * vspan
    ax.annotate('', xy=(x_arr, y0 + dy), xytext=(x_arr, y0),
                arrowprops=dict(arrowstyle='-|>', color=color, lw=1.5,
                                mutation_scale=10),
                zorder=7)
    ax.text(x_arr, y0 + dy * 1.1, 'N', ha='center', va='bottom',
            color=color,
            path_effects=stroke, zorder=7)


def reproject_dem(dem_fp, dst_crs):
    with rasterio.open(dem_fp) as src:
        dst_tf, dst_w, dst_h = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        dem_r = np.full((dst_h, dst_w), np.nan, dtype=np.float64)
        rio_reproject(source=src.read(1).astype(np.float64),
                      destination=dem_r,
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=dst_tf, dst_crs=dst_crs,
                      resampling=Resampling.bilinear,
                      src_nodata=src.nodata, dst_nodata=np.nan)
    res_x =  dst_tf.a
    res_y = -dst_tf.e
    x_arr = dst_tf.c + res_x * (np.arange(dst_w) + 0.5)
    y_arr = dst_tf.f - res_y * (np.arange(dst_h) + 0.5)
    return dem_r, x_arr, y_arr


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

lats    = np.array(lats)
lons    = np.array(lons)
vals_sw = np.array(vals_sw)

# ===================== CRS + GLACIER OUTLINE =====================
clon = float(np.mean(lons))
clat = float(np.mean(lats))
metric_crs = CRS(f'+proj=laea +lat_0={clat:.2f} +lon_0={clon:.2f} +datum=WGS84 +units=m')

to_metric = Transformer.from_crs('EPSG:4326', metric_crs, always_xy=True)
xs, ys = to_metric.transform(lons, lats)

gdf     = gpd.read_file(rgi_fp)
glacier = gdf[gdf['RGIId'].isin(['RGI60-' + i for i in rgi_ids])].to_crs(metric_crs)

# ===================== KRIGING GRID + SW =====================
bounds = glacier.total_bounds
res    = 200
grid_x = np.arange(bounds[0], bounds[2], res)
grid_y = np.arange(bounds[1], bounds[3], res)
xx, yy = np.meshgrid(grid_x, grid_y)
pts    = np.column_stack([xs, ys])
query  = np.column_stack([xx.ravel(), yy.ravel()])

shapes       = [(mapping(geom), 1) for geom in glacier.geometry]
sw_transform = rasterio.transform.from_bounds(*bounds, len(grid_x), len(grid_y))
glacier_mask = np.flipud(
    rasterio.features.rasterize(shapes, out_shape=(len(grid_y), len(grid_x)), transform=sw_transform)
)

rbf   = RBFInterpolator(pts, vals_sw, kernel='thin_plate_spline')
z_sw  = rbf(query).reshape(xx.shape)
z_sw  = np.where(glacier_mask == 1, z_sw, np.nan)

# ===================== DEM CONTOURS (metric_crs for right panel) =====================
dem_m, x_dem_m, y_dem_m = reproject_dem(dem_fp, metric_crs)
xx_dem_m, yy_dem_m = np.meshgrid(x_dem_m, y_dem_m)
elev_min = np.nanmin(dem_m)
elev_max = np.nanmax(dem_m)
contour_levels = np.arange(
    np.ceil(elev_min / CONTOUR_INTERVAL) * CONTOUR_INTERVAL,
    np.floor(elev_max / CONTOUR_INTERVAL) * CONTOUR_INTERVAL + 1,
    CONTOUR_INTERVAL
)

# resample SW onto the DEM grid (same resolution / extent as the shadow panel)
dem_query     = np.column_stack([xx_dem_m.ravel(), yy_dem_m.ravel()])
z_sw_dem      = rbf(dem_query).reshape(xx_dem_m.shape)
px = x_dem_m[1] - x_dem_m[0]   # per-pixel x step (positive)
py = y_dem_m[0] - y_dem_m[1]   # per-pixel y step (positive; y decreases with row)
dem_glac_mask = rasterio.features.rasterize(
    shapes,
    out_shape=dem_m.shape,
    transform=rasterio.transform.from_bounds(
        x_dem_m[0]  - px / 2,
        y_dem_m[-1] - py / 2,
        x_dem_m[-1] + px / 2,
        y_dem_m[0]  + py / 2,
        len(x_dem_m), len(y_dem_m),
    )
)
z_sw_dem = np.where(dem_glac_mask == 1, z_sw_dem, np.nan)

# ===================== SHADOW FRACTION =====================
shad_ds = xr.open_zarr(shadow_fp, consolidated=False)

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

sr_attrs = dict(shad_ds['spatial_ref'].attrs)
if 'crs_wkt' in sr_attrs:
    shadow_crs = CRS.from_wkt(sr_attrs['crs_wkt'])
elif 'proj4' in sr_attrs:
    shadow_crs = CRS.from_proj4(sr_attrs['proj4'])
else:
    shadow_crs = metric_crs

shad_x = shad_ds.x.values
shad_y = shad_ds.y.values

# mask frac_shadow to glacier outline
glacier_shad = glacier.to_crs(shadow_crs)
dx = float(shad_x[1] - shad_x[0])
dy = float(shad_y[0] - shad_y[1])   # positive (y decreases with row)
shad_src_tf = rasterio.transform.from_origin(
    float(shad_x[0]) - dx / 2,
    float(shad_y[0]) + dy / 2,
    dx, dy,
)
shad_glac_mask = rasterio.features.rasterize(
    [(mapping(geom), 1) for geom in glacier_shad.geometry],
    out_shape=frac_shadow.shape,
    transform=shad_src_tf,
)
frac_shadow = np.where(shad_glac_mask == 1, frac_shadow, np.nan)

xx_shad, yy_shad = np.meshgrid(shad_x, shad_y)

# DEM contours for left panel (shadow CRS)
dem_s, x_dem_s, y_dem_s = reproject_dem(dem_fp, shadow_crs)
xx_dem_s, yy_dem_s = np.meshgrid(x_dem_s, y_dem_s)

# ===================== PLOT =====================
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# ---- left: shadow fraction ----
ax = axes[0]
im_shad = ax.pcolormesh(xx_shad, yy_shad, frac_shadow,
                         cmap='Blues_r', vmin=0.4, vmax=0.6, shading='auto')
if CONTOUR_SOURCE == 'elevation':
    ax.contour(xx_dem_s, yy_dem_s, dem_s, levels=contour_levels,
               colors='gray', linewidths=0.8, alpha=0.7, zorder=3)
else:
    ax.contour(xx_shad, yy_shad, frac_shadow, N_VALUE_CONTOURS,
               colors='k', linewidths=0.6, alpha=1, zorder=3)
glacier_shad.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.0, zorder=4)
cbar = fig.colorbar(im_shad, ax=ax, label='Fraction of hours in direct sun',
                    fraction=0.03, pad=0.03)
cbar.set_ticks(np.linspace(0.4, 0.6, 3))
ax.set_title('Shading')
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
ax.set_xlabel('')
ax.set_ylabel('')
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.8)
    spine.set_color('black')

# ---- right: mean direct SW ----
ax = axes[1]
vmin = np.nanpercentile(vals_sw, 1)
vmax = np.nanpercentile(vals_sw, 95)
im_sw = ax.pcolormesh(xx_dem_m, yy_dem_m, z_sw_dem, cmap='YlOrRd',
                       vmin=vmin, vmax=vmax, shading='auto')
if CONTOUR_SOURCE == 'elevation':
    ax.contour(xx_dem_m, yy_dem_m, dem_m, levels=contour_levels,
               colors='gray', linewidths=0.4, alpha=0.7, zorder=3)
else:
    ax.contour(xx_dem_m, yy_dem_m, z_sw_dem, N_VALUE_CONTOURS,
               colors='k', linewidths=0.6, alpha=1, zorder=3)
glacier.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.0, zorder=4)
ax.scatter(xs, ys, c=vals_sw, cmap='YlOrRd', vmin=vmin, vmax=vmax,
           s=15, edgecolors='k', linewidths=0.3, zorder=5)
cbar = fig.colorbar(im_sw, ax=ax, label='Mean direct SW [W m$^{-2}$]',
                    fraction=0.03, pad=0.03)
cbar.set_ticks(np.linspace(vmin, vmax, 4).round(0))
ax.set_title('Mean direct shortwave')
ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
ax.set_xlabel('')
ax.set_ylabel('')
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(0.8)
    spine.set_color('black')

# apply shared axis limits and add map decorations
xlims = (-3300, 3300)
ylims = (-3100, 2300)
for ax in axes:
    ax.set_xlim(*xlims)
    ax.set_ylim(*ylims)
    add_map_decorations(ax, SCALEBAR_M)

rgi_label = ', '.join(sorted(rgi_ids))
fig.suptitle('Gulkana Glacier — ablation season (April 15 - September 15)', fontsize=13, y=0.95)

plt.tight_layout()
out_path = f'{output_dir}/shading_sw_map.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.show()
print(f'saved to {out_path}')
