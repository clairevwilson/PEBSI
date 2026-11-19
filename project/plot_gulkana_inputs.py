"""
2x2 summary of the spatial inputs for Gulkana Glacier: ice thickness,
ice albedo, wind speed-up for one direction, and sky-view factor.

@author: clairevwilson
"""
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import rasterio
import xarray as xr
from pyproj import Transformer

gid = '01.00570'
wind_dir = 0  # direction the wind blows from [deg]; down-glacier for Gulkana
base_fp = '/ocean/projects/ees260009p/cwilson4/'
out_fn = base_fp + 'Figs/gulkana_inputs.png'
utm = 'EPSG:32606'

rgi = gpd.read_file(base_fp + 'RGI/rgi60/01_rgi60_Alaska/01_rgi60_Alaska.shp')
outline = rgi.loc[rgi['RGIId'] == 'RGI60-' + gid].to_crs(utm)

def read_tif(fn):
    with rasterio.open(fn) as src:
        data = src.read(1).astype(float)
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
        t = src.transform
        cols = np.arange(src.width + 1)
        rows = np.arange(src.height + 1)
        x = t.c + cols * t.a
        y = t.f + rows * t.e
    return x, y, data

# thickness and albedo are already in UTM 6N
x_h, y_h, thick = read_tif(base_fp + f'data/ice_thickness/RGI60-01/RGI60-{gid}_thickness.tif')
thick[thick <= 0] = np.nan
x_a, y_a, alb = read_tif(base_fp + f'data/ice_albedo/{gid}_albedo.tif')

# wind speed-up is on a lon/lat grid
wind = xr.open_dataset(base_fp + f'data/windmapper/{gid}.nc')
lon2d, lat2d = np.meshgrid(wind['lon'].values, wind['lat'].values)
x_w, y_w = Transformer.from_crs('EPSG:4326', utm, always_xy=True).transform(lon2d, lat2d)
spdup = wind['spdup'].sel(direction=wind_dir).values

# sky-view factor is on a local Lambert azimuthal grid
shade = xr.open_zarr(base_fp + f'data/shading/{gid}_shadows.zarr')
x2d, y2d = np.meshgrid(shade.x.values, shade.y.values)
to_utm = Transformer.from_crs(shade['spatial_ref'].attrs['crs_wkt'], utm, always_xy=True)
x_s, y_s = to_utm.transform(x2d, y2d)
svf = shade['sky_view_factor'].values

dev = np.nanmax(np.abs(spdup - 1))
panels = [
    (x_h, y_h, thick, 'Ice thickness', 'Thickness (m)', 'Blues', None, None),
    (x_a, y_a, alb, 'Ice albedo', 'Albedo (-)', 'Greys_r', None, None),
    (x_w, y_w, spdup, f'Wind speed-up, northerly wind', 'Speed-up factor (-)',
     'RdBu_r', 1 - dev, 1 + dev),
    (x_s, y_s, svf, 'Sky-view factor', 'Sky-view factor (-)', 'viridis', None, 1),
]

xmin, ymin, xmax, ymax = outline.total_bounds
pad = 600
fig, axes = plt.subplots(2, 2, figsize=(7, 4.5), sharex=True, sharey=True,
                         layout='constrained')
for ax, (x, y, z, title, units, cmap, vmin, vmax), label in zip(axes.flat, panels, 'abcd'):
    im = ax.pcolormesh(x, y, z, cmap=cmap, vmin=vmin, vmax=vmax, shading='auto')
    outline.boundary.plot(ax=ax, color='k', lw=1)
    fig.colorbar(im, ax=ax, label=units, shrink=0.9)
    ax.set_title(f'({label}) {title}', loc='left', fontsize=11)
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect('equal')
    ax.ticklabel_format(useOffset=False, style='plain')
    ax.tick_params(labelsize=8)
for ax in axes[1]:
    ax.set_xlabel('Easting (m)')
for ax in axes[:, 0]:
    ax.set_ylabel('Northing (m)')
# fig.suptitle('Gulkana Glacier (RGI60-01.00570) model inputs')
fig.savefig(out_fn, dpi=300)
print(f'Saved {out_fn}')
