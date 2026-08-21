"""
Find the day near USER_GUESS with ≥90% satellite albedo coverage, then plot
a two-panel map: satellite albedo (left) and kriged simulation albedo (right).

Edit USER_GUESS and output_dir before running.
"""
import sys
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from pyproj import CRS, Transformer
from shapely.geometry import mapping
import rasterio.features
import rasterio.transform

# ===================== USER CONFIG =====================
DATE = '2024-09-20'          # date to search near
output_dir = '../Output/gulkana_windfactor_precgrad_0/'   # simulation zarrs
albedo_dir = '../data/albedo/'
rgi_fp = '../RGI/rgi60/01_rgi60_Alaska/01_rgi60_Alaska.shp'
COVERAGE_THRESH = 0.90              # fraction of glacier pixels required

# Gulkana IDs (from translate_rgi)
RGI6_ID = '01.00570'
RGI7_NUM = '05299'   # translate_rgi['gulkana']['7'][3:]

# ===================== LOAD SATELLITE ALBEDO =====================
sensors = ['s2', 'l8', 'l9']
ds_list = []
epsg = None

for sensor in sensors:
    fp = f'{albedo_dir}{RGI7_NUM}/{RGI7_NUM}_{sensor}.nc'
    try:
        ds = xr.open_dataset(fp)
    except FileNotFoundError:
        print(f'  skipping {fp} (not found)')
        continue
    crs_wkt = ds['spatial_ref'].attrs['crs_wkt']
    if epsg is None:
        epsg = crs_wkt.split('AUTHORITY["EPSG","')[-1].split('"]')[0]
    ds = ds['albedo'].squeeze('band', drop=True).to_dataset()
    ds_list.append(ds)

assert ds_list, 'No albedo files loaded'
sat_ds = xr.concat(ds_list, dim='time').sortby('time')

# merge duplicate dates across sensors by averaging
sat_ds = sat_ds.groupby('time').mean()

# ===================== FIND BEST DATE =====================
# count non-NaN pixels per time step; normalise by max to get coverage fraction
valid_counts = sat_ds['albedo'].count(dim=('x', 'y')).values
max_valid    = valid_counts.max()
coverage     = valid_counts / max_valid

times = pd.to_datetime(sat_ds.time.values)
guess = pd.Timestamp(DATE)

# among dates with >= threshold coverage, find the one closest to guess
mask_cov = coverage >= COVERAGE_THRESH
assert mask_cov.any(), (
    f'No date reaches {COVERAGE_THRESH*100:.0f}% coverage. '
    f'Max is {coverage.max()*100:.1f}%'
)
candidate_times = times[mask_cov]
best_idx   = np.argmin(np.abs(candidate_times - guess))
best_date  = candidate_times[best_idx]

print(f'User input:  {DATE}')
print(f'Best date:   {best_date.date()}  '
      f'(coverage {coverage[mask_cov][best_idx]*100:.1f}%)')

# ===================== SATELLITE MAP FOR BEST DATE =====================
albedo_map = sat_ds['albedo'].sel(time=best_date)

sat_crs  = CRS.from_epsg(int(epsg))
sat_x    = albedo_map.x.values
sat_y    = albedo_map.y.values
xx_sat, yy_sat = np.meshgrid(sat_x, sat_y)

# ===================== LOAD SIMULATION POINTS =====================
ds_sim = xr.open_zarr(f'{output_dir}/output.zarr')
ds_sim['time'] = ds_sim.time - pd.Timedelta(hours=8)

idx_id = np.where(ds_sim['rgiid'].values == RGI6_ID)[0]
assert len(idx_id) > 0, f'RGI ID {RGI6_ID} not found in output.zarr'
points = ds_sim.point.values[idx_id]
model_glac = ds_sim.sel(point=points, layer=0)

# albedo is computed once daily at 14:00 local
target_mask = (
    (model_glac.time.dt.year  == best_date.year)  &
    (model_glac.time.dt.month == best_date.month) &
    (model_glac.time.dt.day   == best_date.day)   &
    (model_glac.time.dt.hour  == 14)
)
alb_day = model_glac['albedo'].where(target_mask, drop=True)
vals_alb = alb_day.isel(time=0).values if alb_day.sizes['time'] > 0 else np.full(len(points), np.nan)

lats = model_glac.lat.values
lons = model_glac.lon.values
rgi_ids = {RGI6_ID}

# ===================== CRS + GLACIER OUTLINE =====================
clon = float(np.mean(lons))
clat = float(np.mean(lats))
metric_crs = CRS(f'+proj=laea +lat_0={clat:.2f} +lon_0={clon:.2f} +datum=WGS84 +units=m')

to_metric = Transformer.from_crs('EPSG:4326', metric_crs, always_xy=True)
xs, ys = to_metric.transform(lons, lats)

gdf      = gpd.read_file(rgi_fp)
glacier  = gdf[gdf['RGIId'].isin(['RGI60-' + i for i in rgi_ids])].to_crs(metric_crs)
glacier_sat = glacier.to_crs(sat_crs)

# ===================== KRIGE SIMULATION ALBEDO =====================
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
    rasterio.features.rasterize(shapes, out_shape=(len(grid_y), len(grid_x)),
                                 transform=sw_transform)
)

rbf    = RBFInterpolator(pts, vals_alb, kernel='thin_plate_spline')
z_alb  = rbf(query).reshape(xx.shape)
z_alb  = np.where(glacier_mask == 1, z_alb, np.nan)

# ===================== PLOT =====================
alb_range = (0.1, 0.95)

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# ---- left: satellite albedo ----
ax = axes[0]
im_sat = ax.pcolormesh(xx_sat, yy_sat, albedo_map.values,
                        cmap='Greys_r', vmin=alb_range[0], vmax=alb_range[1],
                        shading='auto')
glacier_sat.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=1.0)
fig.colorbar(im_sat, ax=ax, label='Albedo [-]', fraction=0.03, pad=0.03)
ax.set_title(f'Satellite albedo — {best_date.date()}')
ax.set_xlabel('Easting [m]')
ax.set_ylabel('Northing [m]')
ax.set_aspect('equal')

# ---- right: simulated albedo (kriged) ----
ax = axes[1]
im_sim = ax.pcolormesh(xx, yy, z_alb,
                        cmap='Greys_r', vmin=alb_range[0], vmax=alb_range[1],
                        shading='auto')
glacier.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=1.0)
ax.scatter(xs, ys, c=vals_alb, cmap='Greys_r',
           vmin=alb_range[0], vmax=alb_range[1],
           s=15, edgecolors='k', linewidths=0.3, zorder=5)
fig.colorbar(im_sim, ax=ax, label='Albedo [-]', fraction=0.03, pad=0.03)
ax.set_title(f'Simulated albedo — {best_date.date()} (14:00 local)')
ax.set_xlabel('Easting [m]')
ax.set_ylabel('Northing [m]')
ax.set_aspect('equal')

rgi_label = ', '.join(sorted(rgi_ids))
fig.suptitle(f'Gulkana Glacier albedo — {best_date.date()} ({rgi_label})', fontsize=13)

plt.tight_layout()
out_path = f'{output_dir}/albedo_map_{best_date.date()}.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'saved to {out_path}')
