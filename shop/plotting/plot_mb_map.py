"""
Plot annual mass balance as Voronoi cells from a PEBSI output.zarr.
Usage: python plot_mb_map.py <output_directory>
"""
import sys
import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from pyproj import CRS, Transformer

output_dir = '/ocean/projects/ees260009p/cwilson4/Output/AD_result_0/' # 
rgi_fp = '/ocean/projects/ees260009p/cwilson4/RGI/rgi60/01_rgi60_Alaska/01_rgi60_Alaska.shp' # /ocean/projects/ees260009p/cwilson4/
plot_var = 'albedo'
cm = 'Greys_r'

# ===================== LOAD DATA =====================
# ds_map = xr.open_zarr(f'{output_dir.replace("now", "w")}/output.zarr', consolidated=False)
# ds_nomap = xr.open_zarr(f'{output_dir}/output.zarr', consolidated=False)

# ds = ds_map 

ds = xr.open_zarr(f'{output_dir}/output.zarr', consolidated=False)
ds = ds.sel(time=slice('2000-01-01', '2010-01-01'))

n_years = len(np.unique(ds.time.dt.year.values))

rgi_ids = ['01.15645'] # [np.unique(ds['rgiid'].values)[0]]
points = ds.point.values[ds['rgiid'].values == rgi_ids[0]]
ds = ds.sel(point=points)

# if plot_var == 'mass_balance':
#     vals = (ds[plot_var] - ds_nomap[plot_var]).sum('time').values / max(n_years, 1)
# else:
#     vals = (ds[plot_var] / ds_nomap[plot_var]).mean('time').values

if plot_var == 'mass_balance':
    if plot_var not in ds.variables:
        ds['mass_balance'] = ds['accumulation'] + ds['refreeze'] - ds['melt']

    # if 'total_mass' in ds.variables:
    #     condition = (ds['total_mass'] > 10).all(dim='time').compute()
    #     ds = ds.where(condition, drop=True)
    vals = ds[plot_var].sum('time').values / max(n_years, 1)
elif plot_var == 'albedo':
    vals = ds[plot_var].sel(time='2009-05-20', method='nearest').values
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

# ===================== VORONOI CELLS =====================
outline = glacier.geometry.union_all()
pts = gpd.GeoDataFrame({'val': vals}, geometry=gpd.points_from_xy(xs, ys), crs=metric_crs)
cells = gpd.GeoDataFrame(geometry=pts.geometry.voronoi_polygons(extend_to=outline), crs=metric_crs)
cells = gpd.sjoin(cells, pts, predicate='contains').drop(columns='index_right')
cells['geometry'] = cells.geometry.intersection(outline)

# ===================== PLOT =====================
if plot_var == 'wind':
    vmin, vmax = 1.5, 5
elif plot_var == 'albedo':
    vmin, vmax = 0.1, 0.9
elif plot_var == 'shortwave_in':
    vmin = np.nanpercentile(np.abs(vals), 1) 
    vmax = np.nanpercentile(np.abs(vals), 95)
else:
    vmin = -5 # -0.5 # np.nanpercentile(np.abs(vals), 1)
    vmax = 5  # 0.5 # np.nanpercentile(np.abs(vals), 95)

fig, ax = plt.subplots(figsize=(8, 7))

# voronoi cells
cells.plot(column='val', ax=ax, cmap=cm, vmin=vmin, vmax=vmax, edgecolor='k', linewidth=0.2)
im = ax.collections[0]

# glacier outline
# if len(np.column_stack([xs, ys]).flatten()) <= 1000:
glacier.plot(ax=ax, facecolor='none', edgecolor='k', linewidth=0.8)

print('N POINTS:', len(vals.flatten()))

if plot_var == 'mass_balance':
    cbar_lab = 'Mass balacne rate [m w.e. yr$^{-1}$]'
elif plot_var == 'wind':
    cbar_lab = 'Average wind speed [m s$^{-1}$]'
elif plot_var == 'shortwave_in':
    cbar_lab = 'Average incoming shortwave [W m$^{-2}$]'
elif plot_var == 'albedo':
    cbar_lab = 'Albedo'
else:
    cbar_lab = plot_var

cbar = fig.colorbar(im, ax=ax, label=cbar_lab, shrink=0.8)

ax.set_xlabel('Easting [m]')
ax.set_ylabel('Northing [m]')
# ax.set_title(f'Mass balance — {n_points} points ({", ".join(rgi_ids)})')
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(f'{output_dir}{rgi_ids[0]}_{plot_var}_map.png', dpi=150)
plt.show()
print(f'saved to {output_dir}{rgi_ids[0]}_{plot_var}_map.png')