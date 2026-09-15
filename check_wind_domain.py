"""
Checks whether the windmapper grid extends past the glacier margin, and
whether the off-glacier ground is windier. get_wind_fields averages the
speed-up over every finite pixel nearest each point, with no ice mask,
so any off-glacier pixel is pulled into a point's value.
"""
import geopandas as gpd
import numpy as np
import shapely
import xarray as xr
from pyproj import Transformer

from host_paths import host, HOST_PATHS
from pebsi.io import mesh

GID = '01.00570'
p = HOST_PATHS[host]

g = gpd.read_file(p['rgi_fp'] + '../01_rgi60_Alaska/01_rgi60_Alaska.shp')
poly, crs = mesh.glacier_polygon(g, GID)

w = xr.open_dataset(p['windmap_fn'].format(gid=GID))
lon, lat = np.meshgrid(w['lon'].values, w['lat'].values)
x, y = Transformer.from_crs('EPSG:4326', crs, always_xy=True).transform(lon, lat)

sp3 = w['spdup'].transpose('y', 'x', 'direction').values
fin = np.isfinite(sp3).all(axis=2)
sp = sp3.mean(axis=2)
ice = shapely.contains_xy(poly, x, y)

on, off = fin & ice, fin & ~ice
print(f'grid shape     {fin.shape}')
print(f'finite cells   {fin.sum()}')
print(f'  on-ice       {on.sum()}  ({100 * on.sum() / fin.sum():.1f}%)')
print(f'  off-ice      {off.sum()}  ({100 * off.sum() / fin.sum():.1f}%)')
print()
print(f'mean spdup  on-ice  {sp[on].mean():.4f}   std {sp[on].std():.4f}')
print(f'mean spdup  off-ice {sp[off].mean():.4f}   std {sp[off].std():.4f}')
print(f'off-ice is {100 * (sp[off].mean() / sp[on].mean() - 1):+.1f}% windier')
print()

res = np.abs(np.diff(w['lon'].values)).mean()
print(f'grid spacing   {res:.6f} deg  (~{res * 111000 * np.cos(np.deg2rad(63.28)):.0f} m in x)')
gb = poly.bounds
print(f'glacier bounds x {gb[0]:.0f} to {gb[2]:.0f}, y {gb[1]:.0f} to {gb[3]:.0f}')
print(f'wind bounds    x {x[fin].min():.0f} to {x[fin].max():.0f}, '
      f'y {y[fin].min():.0f} to {y[fin].max():.0f}')

# how far past the margin the grid reaches
d = shapely.distance(shapely.points(x[off], y[off]), poly)
print(f'off-ice distance from margin: median {np.median(d):.0f} m, max {d.max():.0f} m')
w.close()
