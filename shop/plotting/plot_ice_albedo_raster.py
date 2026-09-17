"""
Quick plot of the ice albedo raster for a given RGI id.

@author: clairevwilson
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import rasterio

gn = '09162'
fn = f'/ocean/projects/ees260009p/cwilson4/data/ice_albedo/01.{gn}_albedo.tif'

with rasterio.open(fn) as src:
    data = src.read(1)
    bounds = src.bounds
    transform = src.transform
    nodata = src.nodata

if nodata is not None:
    data = np.where(data == nodata, np.nan, data)

# transform.e is positive here (row 0 = south edge), the opposite of the
# usual north-up convention that imshow/plotting_extent assume, so flip
# the array and derive the extent directly instead of trusting src.bounds
if transform.e > 0:
    data = np.flipud(data)
south, north = sorted([bounds.top, bounds.bottom])
extent = (bounds.left, bounds.right, south, north)

fig, ax = plt.subplots(figsize=(6, 6))
ax.add_patch(Rectangle(
    (extent[0], extent[2]), extent[1] - extent[0], extent[3] - extent[2],
    facecolor='none', edgecolor='lightgray', hatch='///', zorder=0
))
im = ax.imshow(data, extent=extent, cmap='Greys_r', vmin=0.05, vmax=0.45, zorder=1)
fig.colorbar(im, ax=ax, label='Ice albedo', shrink=0.5)
ax.set_xlabel('Easting')
ax.set_ylabel('Northing')
fig.tight_layout()
fig.savefig(f'ice_albedo_01.{gn}.png', dpi=150)
print(f'Saved ice_albedo_01.{gn}.png')
