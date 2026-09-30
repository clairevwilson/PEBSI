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
from matplotlib.collections import PolyCollection
from scipy.spatial import cKDTree, Voronoi
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.prepared import prep
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
    albedo_pts = np.nanmean(0.5 * np.log(2 * np.pi * ALBEDO_SIGMA**2)
                            + albedo_residual**2 / (2 * ALBEDO_SIGMA**2), axis=0)
    snow_pts = bce(sm.mod_snow, sm.meas_snow)
    melt_pts = bce(sm.mod_melt, sm.meas_melt)

point_losses = {'snow': snow_pts, 'melt': melt_pts, 'albedo': albedo_pts}

# ===================== VORONOI CELLS =====================
# each model point owns the cell of ground closest to it (a Voronoi
# diagram in the metric CRS), so cell area follows the point density
# instead of assuming a regular lattice
def voronoi_finite_polygons_2d(vor, radius=None):
    """Reconstruct infinite Voronoi regions as finite polygons by
    extending their open ridges out to `radius`. Standard recipe
    (Pauli Virtanen, https://stackoverflow.com/a/20678647)."""
    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp(axis=0).max() * 2

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

pts = np.column_stack([xs, ys])
spacing_m = np.median(cKDTree(pts).query(pts, k=2)[0][:, 1])

vor = Voronoi(pts)
regions, vertices = voronoi_finite_polygons_2d(vor)

# crop the (otherwise unbounded) cells to the glacier outline itself, so
# no ice-free ground gets colored in -- most cells sit entirely inside
# the outline, so only run the expensive intersection on the ones that
# actually straddle the boundary (checked cheaply via a prepared geometry)
outline_geom = unary_union(glacier_outline.geometry.values)
prepared_outline = prep(outline_geom)
cell_polys = []
for region in regions:
    poly = Polygon(vertices[region])
    cell_polys.append(poly if prepared_outline.contains(poly)
                       else poly.intersection(outline_geom))

def poly_verts(geom):
    if geom.is_empty:
        return None
    if geom.geom_type == 'Polygon':
        return np.asarray(geom.exterior.coords)
    if geom.geom_type == 'MultiPolygon':
        largest = max(geom.geoms, key=lambda g: g.area)
        return np.asarray(largest.exterior.coords)
    return None

# plot extent: outline bounds with a little breathing room
bx0, by0, bx1, by1 = glacier_outline.total_bounds
bx0, by0 = bx0 - spacing_m, by0 - spacing_m
bx1, by1 = bx1 + spacing_m, by1 + spacing_m

# ===================== PLOT =====================
error_labels = {'snow': 'Snowline', 'melt': 'Melt extent', 'albedo': 'Albedo'}
cm = 'magma_r'

# size the figure to the glacier's own aspect ratio (4 map/bar cells in
# a single row tile to 4x the aspect ratio of one cell): derive both
# dimensions from that ratio, capped at 10in wide, with a legibility
# floor on height for glaciers wide enough that 10in would flatten it
aspect = (bx1 - bx0) / (by1 - by0)
max_w, min_h = 10, 3.0
fig_h = max(min_h, max_w / (4 * aspect))
fig_w = min(max_w, 4 * aspect * fig_h)

fig, axes = plt.subplots(1, 4, figsize=(fig_w, fig_h), constrained_layout=True)

map_axes = axes[:3]
for ax in map_axes[1:]:
    ax.sharex(map_axes[0])
    ax.sharey(map_axes[0])

# glacier outline
for ax in map_axes:
    glacier_outline.plot(ax=ax, facecolor='none', edgecolor='k', linewidth=1.0)
    ax.set_aspect('equal')
map_axes[0].set_ylabel('Northing [m]')
map_axes[1].set_xlabel('Easting [m]')
for ax in map_axes[1:]:
    ax.tick_params(labelleft=False)

# voronoi cells colored by loss for each dataset
for ax, var in zip(map_axes, ['snow', 'melt', 'albedo']):
    vals = point_losses[var]
    vmin, vmax = np.nanpercentile(vals, [2, 98])

    verts, cell_vals = [], []
    for geom, v in zip(cell_polys, vals):
        cv = poly_verts(geom)
        if cv is None:
            continue
        verts.append(cv)
        cell_vals.append(v)

    pc = PolyCollection(verts, array=np.asarray(cell_vals), cmap=cm,
                        edgecolors='none', zorder=0)
    pc.set_clim(vmin, vmax)
    ax.add_collection(pc)
    ax.set_xlim(bx0, bx1)
    ax.set_ylim(by0, by1)
    fig.colorbar(pc, ax=ax, label='log loss [nats]', shrink=0.7)
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

output_fn = f'{output_dir}{rgi_id}_loss_map.png'
plt.savefig(output_fn, dpi=150)
plt.show()
print(f'saved to {output_fn}')
