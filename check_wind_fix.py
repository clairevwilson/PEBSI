"""
Quantifies how much of the coarse-mesh wind bias an on-ice pixel mask
recovers, before changing get_wind_fields.

Compares ablation-zone speed-up at a coarse and a fine mesh, with the
grid unmasked (what the model does now) and masked to ice (what
load_dem_info already does for the DEM).
"""
import geopandas as gpd
import numpy as np
import shapely
import xarray as xr
from pyproj import Transformer
from scipy.spatial import cKDTree

from host_paths import host, HOST_PATHS
from pebsi.io import mesh
from pebsi.io.terrain import Terrain

GID = '01.00570'
COARSE, FINE = 1200.0, 80.0
ABLATION_MAX = 1500.0
DB_DWIND = -17.1465          # from the weighted fit in debug_attribution

p = HOST_PATHS[host]
g = gpd.read_file(p['rgi_fp'] + '../01_rgi60_Alaska/01_rgi60_Alaska.shp')
poly, crs = mesh.glacier_polygon(g, GID)
glacier = g.loc[g['RGIId'] == 'RGI60-' + GID]

# elevation, to define the ablation zone at fine resolution
import rioxarray as rxr
dem = rxr.open_rasterio(p['cop30_vrt_path']).squeeze().drop_vars('band')
b = glacier.to_crs(dem.rio.crs).total_bounds
dem = dem.rio.clip_box(b[0] - .02, b[1] - .02, b[2] + .02, b[3] + .02)
dem = dem.rio.reproject(crs)
if dem.rio.nodata is not None:
    dem = dem.where((dem != dem.rio.nodata) & np.isfinite(dem))
gx, gy = np.meshgrid(dem.x.values, dem.y.values)
dok = shapely.contains_xy(poly, gx, gy) & np.isfinite(dem.values)
dem_px = np.column_stack([gx[dok], gy[dok]])
dem_z = dem.values[dok]

w = xr.open_dataset(p['windmap_fn'].format(gid=GID))
lon, lat = np.meshgrid(w['lon'].values, w['lat'].values)
x, y = Transformer.from_crs('EPSG:4326', crs, always_xy=True).transform(lon, lat)
sp3 = w['spdup'].transpose('y', 'x', 'direction').values
fin = np.isfinite(sp3).all(axis=2)
sp = sp3.mean(axis=2)
ice = shapely.contains_xy(poly, x, y)
w.close()

grids = {
    'unmasked (current)': fin,
    'masked to ice':      fin & ice,
}


def cell_mean(pts, px, vals):
    m, _ = Terrain.cell_mean(pts, px, vals)
    empty = ~np.isfinite(m)
    if empty.any():
        m[empty] = vals[cKDTree(px).query(pts[empty], workers=-1)[1]]
    return m


meshes = {}
for h in (COARSE, FINE):
    xs, ys, ws, _ = mesh.mesh_polygon(poly, h)
    pts = np.column_stack([xs, ys])
    meshes[h] = (pts, ws, cell_mean(pts, dem_px, dem_z))

fpts, fw, fz = meshes[FINE]
cpts, cw, _ = meshes[COARSE]
abl = fz < ABLATION_MAX
owner = cKDTree(cpts).query(fpts, workers=-1)[1]
area = fw[abl].sum()

print(f'ablation zone: {abl.sum()} fine points, {area:.4f} of area')
print()
print(f'{"grid":>22} {"coarse":>9} {"fine":>9} {"shift":>9} {"MB effect":>11}')
for name, m in grids.items():
    px = np.column_stack([x[m], y[m]])
    v = sp[m]
    cvals = cell_mean(cpts, px, v)[owner][abl]
    fvals = cell_mean(fpts, px, v)[abl]
    cm = np.average(cvals, weights=fw[abl])
    fm = np.average(fvals, weights=fw[abl])
    print(f'{name:>22} {cm:>9.4f} {fm:>9.4f} {cm - fm:>+9.4f} '
          f'{DB_DWIND * (cm - fm) * area:>+11.4f}')
    print(f'{"":>22} pixels used: {m.sum()}')

print()
print('MB effect is that shift carried through db/dwind and the ablation')
print('zone area fraction. Currently about -0.41 of the -0.46 total drift.')
