"""
Plot distributed model-measurement error on the model's point grid from a PEBSI output.zarr.
Usage: python plot_distributed_error.py <output_directory>
"""
import sys
import warnings
import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from pyproj import CRS, Transformer
from project.glacierwide_loss import *

output_dir = '/ocean/projects/ees260009p/cwilson4/Output/gulkana_0/' # 
rgi_fp = '/ocean/projects/ees260009p/cwilson4/RGI/rgi60/01_rgi60_Alaska/01_rgi60_Alaska.shp' # /ocean/projects/ees260009p/cwilson4/

# ===================== LOAD DATA =====================
ds = xr.open_zarr(f'{output_dir}/output.zarr', consolidated=False)

n_years = len(np.unique(ds.time.dt.year.values))
rgi_id = ds.rgiid.values[0]

# keep only the points on the glacier being evaluated
ds = ds.sel(point=ds.point.values[ds['rgiid'].values == rgi_id])

lats = ds['lat'].values
lons = ds['lon'].values
start = ds.time.values[0]
end = ds.time.values[-1]
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
glacier_outline = gdf[gdf['RGIId'] == f'RGI60-{rgi_id}'].to_crs(metric_crs)

# ===================== ERROR =====================
glacier = translate_names[rgi_id]
ab = Albedo(glacier)
mb = MassBalance(glacier, dates=('2016-04-01', '2020-01-01'))
sm = SnowlineMelt(glacier)

# get model vs measurements
ab.get_model_albedo(ds)
ab.get_deltas()
mb.get_model_mb(ds)
sm.get_model_snow(ds)

# per-point losses, averaged over time
def bce(mod, meas, eps=1e-3):
    p = np.clip(np.asarray(mod, dtype=float), eps, 1 - eps)
    y = np.asarray(meas, dtype=float)
    return -np.nanmean(y * np.log(p) + (1 - y) * np.log(1 - p), axis=0)

with warnings.catch_warnings():
    # points with no valid measurement in the record reduce to nan
    warnings.simplefilter('ignore', RuntimeWarning)
    albedo_residual = ab.mod - ab.meas
    albedo_pts = np.nanmean(0.5 * np.log(2 * np.pi * 0.03**2)
                            + albedo_residual**2 / (2 * 0.03**2), axis=0)
    snow_pts = bce(sm.mod_snow, sm.meas_snow)
    melt_pts = bce(sm.mod_melt, sm.meas_melt)

point_losses = {'snow': snow_pts, 'melt': melt_pts, 'albedo': albedo_pts}

# ===================== ACTUAL MODEL GRID =====================
# points are placed on a regular lattice in lon/lat degrees, not meters
# (see pebsi/io/terrain.py::_grid_polygon, built on the RGI shapefile's
# EPSG:4326 CRS) -- a degree of longitude is much shorter than a degree
# of latitude this far north, so recover the lattice in degree space
# (where spacing is isotropic) rather than in the reprojected xs/ys
lonlat = np.column_stack([lons, lats])
spacing_deg = np.median(cKDTree(lonlat).query(lonlat, k=2)[0][:, 1])

col = np.round((lons - lons.min()) / spacing_deg).astype(int)
row = np.round((lats - lats.min()) / spacing_deg).astype(int)
grid_lon = lons.min() + spacing_deg * np.arange(col.max() + 1)
grid_lat = lats.min() + spacing_deg * np.arange(row.max() + 1)
lon_nodes, lat_nodes = np.meshgrid(grid_lon, grid_lat)
grid_x, grid_y = to_metric.transform(lon_nodes, lat_nodes)

def grid_error(vals):
    grid = np.full(lon_nodes.shape, np.nan)
    grid[row, col] = vals
    return grid

error_maps = {var: grid_error(vals) for var, vals in point_losses.items()}

# ===================== PLOT =====================
error_labels = {'snow': 'Snowline', 'melt': 'Melt extent', 'albedo': 'Albedo'}
cm = 'magma_r'

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
axes = axes.flatten()

# glacier outline
for ax in axes[:-1]:
    glacier_outline.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=1.0)
    ax.set_xlabel('Easting [m]')
    ax.set_ylabel('Northing [m]')
    ax.set_aspect('equal')

# gridded loss for each dataset
for ax, var in zip(axes[:-1], ['snow', 'melt', 'albedo']):
    vals = point_losses[var]
    vmin, vmax = np.nanpercentile(vals, [2, 98])
    im = ax.pcolormesh(grid_x, grid_y, error_maps[var], cmap=cm,
                       vmin=vmin, vmax=vmax, shading='nearest', zorder=0)
    fig.colorbar(im, ax=ax, label='log loss [nats]', shrink=0.7)
    ax.set_title(error_labels[var])

# bar chart for last subpanel
ab_loss = ab.log_loss()
mb_loss = mb.log_loss()
snow_loss, melt_loss = sm.bernoulli_loss()
losses = [mb_loss, ab_loss, snow_loss, melt_loss]

axes[-1].bar(range(len(losses)), losses)
axes[-1].set_xticks(range(len(losses)))
axes[-1].set_xticklabels(['Mass balance', 'Albedo', 'Snowline', 'Melt extent'], rotation=20)
axes[-1].set_ylabel('log loss [nats]')

plt.tight_layout()

output_fn = f'{output_dir}{rgi_id}_loss_map.png'
plt.savefig(output_fn, dpi=150)
plt.show()
print(f'saved to {output_fn}')
