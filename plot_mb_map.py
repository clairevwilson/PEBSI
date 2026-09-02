"""
Plot annual mass balance as a kriged heatmap from a PEBSI output.zarr.
Usage: python plot_mb_map.py <output_directory>
"""
import sys
import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from pyproj import CRS, Transformer
from shapely.geometry import mapping
import rasterio.features

output_dir = '/ocean/projects/ees260009p/cwilson4/Output/test_point_setup_2/' # 
rgi_fp = '/ocean/projects/ees260009p/cwilson4/RGI/rgi60/01_rgi60_Alaska/01_rgi60_Alaska.shp' # /ocean/projects/ees260009p/cwilson4/
plot_var = 'albedo'
cm = 'Greys_r'

# ===================== LOAD DATA =====================
# ds_map = xr.open_zarr(f'{output_dir.replace("now", "w")}/output.zarr', consolidated=False)
# ds_nomap = xr.open_zarr(f'{output_dir}/output.zarr', consolidated=False)

# ds = ds_map 

ds = xr.open_zarr(f'{output_dir}/output.zarr', consolidated=False)

n_years = len(np.unique(ds.time.dt.year.values))

rgi_ids = ['01.00570'] # [np.unique(ds['rgiid'].values)[0]]
points = ds.point.values[ds['rgiid'].values == rgi_ids[0]]
ds = ds.sel(point=points)

# if plot_var == 'mass_balance':
#     vals = (ds[plot_var] - ds_nomap[plot_var]).sum('time').values / max(n_years, 1)
# else:
#     vals = (ds[plot_var] / ds_nomap[plot_var]).mean('time').values

if plot_var == 'mass_balance':
    vals = ds[plot_var].sum('time').values / max(n_years, 1)
elif plot_var == 'albedo':
    vals = ds[plot_var].sel(time='2019-08-01', method='nearest').values
else:
    vals = ds[plot_var].mean('time').values

lats = ds['lat'].values
lons = ds['lon'].values
# rgi_ids = set(ds['rgiid'].values.tolist())
n_points = ds.sizes['point']

# ===================== CRS SETUP =====================
# build a local LAEA centered on the glacier
clon = float(np.mean(lons))
clat = float(np.mean(lats))
metric_crs = CRS(f'+proj=laea +lat_0={clat:.2f} +lon_0={clon:.2f} +datum=WGS84 +units=m')

to_metric = Transformer.from_crs('EPSG:4326', metric_crs, always_xy=True)
xs, ys = to_metric.transform(lons, lats)

# ===================== SHAPEFILE =====================
gdf = gpd.read_file(rgi_fp)
glacier = gdf[gdf['RGIId'].isin(['RGI60-' + i for i in rgi_ids])].to_crs(metric_crs)

# ===================== KRIGING =====================
# build a regular grid covering the glacier bounds
bounds = glacier.total_bounds  # (minx, miny, maxx, maxy)
res = 200  # grid resolution in meters
grid_x = np.arange(bounds[0], bounds[2], res)
grid_y = np.arange(bounds[1], bounds[3], res)

# mask to glacier outline
xx, yy = np.meshgrid(grid_x, grid_y)

# RBF interpolation (thin-plate spline)
rbf = RBFInterpolator(np.column_stack([xs, ys]), vals, kernel='thin_plate_spline')
z_grid = rbf(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

shapes = [(mapping(geom), 1) for geom in glacier.geometry]
transform = rasterio.transform.from_bounds(*bounds, len(grid_x), len(grid_y))
glacier_mask = np.flipud(rasterio.features.rasterize(shapes, out_shape=z_grid.shape, transform=transform))
z_masked = np.where(glacier_mask == 1, z_grid, np.nan)

# ===================== PLOT =====================
if plot_var == 'wind':
    vmin, vmax = 0.5, 1.5
elif plot_var == 'albedo':
    vmin, vmax = 0.1, 0.9
else:
    vmin = -4 # -0.5 # np.nanpercentile(np.abs(vals), 1)
    vmax = 4# 0.5 # np.nanpercentile(np.abs(vals), 95)

fig, ax = plt.subplots(figsize=(8, 7))

# kriged fill
im = ax.pcolormesh(xx, yy, z_masked, cmap=cm, vmin=vmin, vmax=vmax, shading='auto')

# glacier outline
if len(np.column_stack([xs, ys]).flatten()) <= 500:
    glacier.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=1.0)

# scatter points on top
ax.scatter(xs, ys, c=vals, cmap=cm, vmin=vmin, vmax=vmax,
           s=15, edgecolors='k', linewidths=0.3, zorder=5)

if plot_var == 'mass_balance':
    cbar = fig.colorbar(im, ax=ax, label='$\Delta$Annual mass balance [m w.e.]', shrink=0.8)
elif plot_var == 'wind':
    cbar = fig.colorbar(im, ax=ax, label='Wind speed mapped / unmapped', shrink=0.8)
else:
    cbar = fig.colorbar(im, ax=ax, label=plot_var, shrink=0.8)

ax.set_xlabel('Easting [m]')
ax.set_ylabel('Northing [m]')
# ax.set_title(f'Mass balance — {n_points} points ({", ".join(rgi_ids)})')
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(f'{output_dir}{rgi_ids[0]}_{plot_var}_map.png', dpi=150)
plt.show()
print(f'saved to {output_dir}{rgi_ids[0]}_{plot_var}_map.png')