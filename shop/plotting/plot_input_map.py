"""
Plot an input raster (ice albedo or ice thickness) resampled onto the
glacier's Voronoi cells the same way the model samples it.
"""
import numpy as np
import xarray as xr
import rioxarray as rxr
import geopandas as gpd
import matplotlib.pyplot as plt
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree

output_dir = '/ocean/projects/ees260009p/cwilson4/Output/AD_result_0/'
rgi_fp = '/ocean/projects/ees260009p/cwilson4/RGI/rgi60/01_rgi60_Alaska/01_rgi60_Alaska.shp'
ice_albedo_fn = '/ocean/projects/ees260009p/cwilson4/data/ice_albedo/{gid}_albedo.tif'
thickness_fn = '/ocean/projects/ees260009p/cwilson4/data/ice_thickness/RGI60-01/RGI60-{gid}_thickness.tif'
rgi_id = '01.15645'
plot_var = 'ice_albedo' # 'ice_albedo' or 'ice_thickness'

# ===================== LOAD POINTS =====================
ds = xr.open_zarr(f'{output_dir}/output.zarr', consolidated=False)
points = ds.point.values[ds['rgiid'].values == rgi_id]
ds = ds.sel(point=points)
lats = ds['lat'].values
lons = ds['lon'].values

# ===================== SAMPLE RASTER =====================
fn = (ice_albedo_fn if plot_var == 'ice_albedo' else thickness_fn).format(gid=rgi_id)
da = rxr.open_rasterio(fn).squeeze().drop_vars('band')
to_raster = Transformer.from_crs('EPSG:4326', da.rio.crs, always_xy=True)
x_pts, y_pts = to_raster.transform(lons, lats)

if plot_var == 'ice_albedo':
    # mean over each Voronoi cell (pixels owned by their nearest point, as in
    # Terrain.cell_mean); cells that catch no valid pixel stay NaN
    values = da.values.astype(float)
    if da.rio.nodata is not None:
        values = np.where(values == da.rio.nodata, np.nan, values)
    grid_x, grid_y = np.meshgrid(da.x.values, da.y.values)
    valid = np.isfinite(values)
    pts_xy = np.column_stack([x_pts, y_pts])
    pixels = np.column_stack([grid_x[valid], grid_y[valid]])
    owner = cKDTree(pts_xy).query(pixels, workers=-1)[1]
    count = np.bincount(owner, minlength=len(pts_xy))
    total = np.bincount(owner, weights=values[valid], minlength=len(pts_xy))
    vals = np.where(count > 0, total / np.maximum(count, 1), np.nan)
else:
    # nearest pixel to each point
    selected = da.sel(x=xr.DataArray(x_pts, dims='points'),
                      y=xr.DataArray(y_pts, dims='points'), method='nearest')
    vals = np.maximum(np.nan_to_num(selected.values, nan=0.0), 0.0)
da.close()

# ===================== CRS SETUP =====================
clon = float(np.mean(lons))
clat = float(np.mean(lats))
metric_crs = CRS(f'+proj=laea +lat_0={clat:.2f} +lon_0={clon:.2f} +datum=WGS84 +units=m')
to_metric = Transformer.from_crs('EPSG:4326', metric_crs, always_xy=True)
xs, ys = to_metric.transform(lons, lats)

# ===================== VORONOI CELLS =====================
gdf = gpd.read_file(rgi_fp)
glacier = gdf[gdf['RGIId'] == 'RGI60-' + rgi_id].to_crs(metric_crs)
outline = glacier.geometry.union_all()
pts = gpd.GeoDataFrame({'val': vals}, geometry=gpd.points_from_xy(xs, ys), crs=metric_crs)
cells = gpd.GeoDataFrame(geometry=pts.geometry.voronoi_polygons(extend_to=outline), crs=metric_crs)
cells = gpd.sjoin(cells, pts, predicate='contains').drop(columns='index_right')
cells['geometry'] = cells.geometry.intersection(outline)

# ===================== PLOT =====================
if plot_var == 'ice_albedo':
    vmin, vmax = 0.1, 0.5
    cm = 'Greys_r'
    cbar_lab = 'Ice albedo'
else:
    vmin, vmax = 0, np.nanpercentile(vals, 99)
    cm = 'viridis'
    cbar_lab = 'Ice thickness [m]'

fig, ax = plt.subplots(figsize=(8, 7))
# hatched glacier background shows through wherever a cell has no data
glacier.plot(ax=ax, facecolor='none', edgecolor='grey', hatch='////', linewidth=0)
cells.dropna(subset='val').plot(column='val', ax=ax, cmap=cm, vmin=vmin, vmax=vmax,
                                edgecolor='k', linewidth=0.2)
im = ax.collections[-1]
glacier.plot(ax=ax, facecolor='none', edgecolor='k', linewidth=0.8)
fig.colorbar(im, ax=ax, label=cbar_lab, shrink=0.8)

ax.set_xlabel('Easting [m]')
ax.set_ylabel('Northing [m]')
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(f'{output_dir}{rgi_id}_{plot_var}_map.png', dpi=150)
plt.show()
print(f'saved to {output_dir}{rgi_id}_{plot_var}_map.png')
