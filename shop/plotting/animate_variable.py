"""
Single-panel animation of one modeled output variable over a glacier,
drawn on the voronoi cells of the model points and aggregated to daily
frames.

Set VARIABLE, GLACIER and the date window below and rerun.

@author: clairevwilson
"""
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image, ImageSequence
from matplotlib.collections import PathCollection
from matplotlib.path import Path
from scipy.spatial import Voronoi
from shapely.geometry import Polygon

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from project.glacierwide_loss import _load_rgi_outline
from project.parameters import translate_rgi

VARIABLE = 'surftemp'
GLACIER = 'gulkana'
START = '2010-04-01'
END = '2011-04-01'
RUN_DIR = '/ocean/projects/ees260009p/cwilson4/Output/controlled/base_2/'

# how each variable is reduced to daily frames and drawn. agg is 'mean'
# (state variables), 'sum' (daily total of a flux) or 'cumsum' (running
# total since START). center makes the color limits symmetric about that
# value for diverging maps; vmin/vmax left out are taken from the data.
STYLES = {
    # surface temperature is capped at 0 C, so the brightest color marks a
    # melting surface and cold nights fade to black
    'surftemp': dict(agg='mean', cmap='magma', vmin=-20, vmax=0,
                     label='Surface temperature (°C)'),
    'airtemp': dict(agg='mean', cmap='RdBu_r', center=0,
                    label='Air temperature (°C)'),
    # red is mass loss, blue is gain, white is no change since START
    'mass_balance': dict(agg='cumsum', cmap='RdBu', center=0,
                         label='Cumulative mass balance (m w.e.)'),
    'melt': dict(agg='sum', cmap='YlOrRd', vmin=0,
                 label='Daily melt (m w.e.)'),
    'runoff': dict(agg='sum', cmap='PuBu', vmin=0,
                   label='Daily runoff (m w.e.)'),
    'refreeze': dict(agg='sum', cmap='BuPu', vmin=0,
                     label='Daily refreeze (m w.e.)'),
    'accumulation': dict(agg='cumsum', cmap='Blues', vmin=0,
                         label='Cumulative accumulation (m w.e.)'),
    # grayscale so dark ice reads dark and fresh snow reads bright
    'albedo': dict(agg='mean', cmap='bone', vmin=0.2, vmax=0.9,
                   label='Albedo (-)'),
    'total_water': dict(agg='mean', cmap='Blues', vmin=0,
                        label='Liquid water content (kg m$^{-2}$)'),
    'melt_energy': dict(agg='mean', cmap='inferno', vmin=0,
                        label='Melt energy (W m$^{-2}$)'),
    'sensible_heat': dict(agg='mean', cmap='RdBu_r', center=0,
                          label='Sensible heat flux (W m$^{-2}$)'),
    'latent_heat': dict(agg='mean', cmap='RdBu_r', center=0,
                        label='Latent heat flux (W m$^{-2}$)'),
}

ds = xr.open_zarr(RUN_DIR + 'output.zarr', consolidated=False)
gmask = ds['rgiid'].values == translate_rgi[GLACIER]['6']
sub = ds[['lon', 'lat', VARIABLE]].isel(point=gmask).sel(time=slice(START, END)).load()
if sub.time.size == 0:
    raise ValueError(f'no model output between {START} and {END}; the run covers '
                     f'{ds.time.values[0]} to {ds.time.values[-1]}')

units = ds[VARIABLE].attrs.get('units', '')
style = STYLES.get(VARIABLE, dict(agg='mean', cmap='viridis',
                                  label=f'{VARIABLE} ({units})'))
daily = sub[VARIABLE].resample(time='1D')
if style['agg'] == 'mean':
    da = daily.mean()
else:
    da = daily.sum()
    if style['agg'] == 'cumsum':
        da = da.cumsum('time')
data = da.transpose('time', 'point').values
dates = da.time.values

# fill any limits the style leaves open from the 2nd-98th percentiles
vmin = style.get('vmin', np.nanpercentile(data, 2))
vmax = style.get('vmax', np.nanpercentile(data, 98))
if 'center' in style:
    half = max(abs(vmin - style['center']), abs(vmax - style['center']))
    vmin, vmax = style['center'] - half, style['center'] + half

outline = _load_rgi_outline(translate_rgi[GLACIER]['6'])
crs = outline.estimate_utm_crs()
outline = outline.to_crs(crs).geometry.iloc[0]
proj = Transformer.from_crs('EPSG:4326', crs, always_xy=True)
x, y = proj.transform(sub.lon.values, sub.lat.values)

# voronoi cell of each model point, clipped to the RGI outline; four far-off
# dummy points keep every real point's cell finite
far = 1e6
x0, y0 = x.mean(), y.mean()
vor = Voronoi(np.vstack([np.column_stack([x, y]),
                         [[x0 - far, y0 - far], [x0 - far, y0 + far],
                          [x0 + far, y0 - far], [x0 + far, y0 + far]]]))
paths = []
for p in range(len(x)):
    cell = Polygon(vor.vertices[vor.regions[vor.point_region[p]]]).intersection(outline)
    parts = getattr(cell, 'geoms', [cell])
    rings = [r for part in parts for r in [part.exterior, *part.interiors]]
    verts = np.concatenate([np.asarray(r.coords) for r in rings])
    codes = np.concatenate([[Path.MOVETO] + [Path.LINETO] * (len(r.coords) - 1) for r in rings])
    paths.append(Path(verts, codes))

cmap = plt.get_cmap(style['cmap']).with_extremes(bad='lightgray')
fig, ax = plt.subplots(figsize=(6, 4), dpi=100, layout='compressed')
cells = PathCollection(paths, cmap=cmap, edgecolors='none')
cells.set_clim(vmin, vmax)
cells.set_array(data[0])
ax.add_collection(cells)
ax.plot(*outline.exterior.xy, color='k', lw=0.8)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
fig.colorbar(cells, ax=ax, label=style['label'], shrink=0.8)
glacier_name = GLACIER.replace('_', ' ').title()
title = ax.set_title(f'{glacier_name}  {pd.Timestamp(dates[0]).date()}')


def update(t):
    cells.set_array(data[t])
    title.set_text(f'{glacier_name}  {pd.Timestamp(dates[t]).date()}')
    return [cells, title]


anim = FuncAnimation(fig, update, frames=len(dates), interval=100)
out_fn = RUN_DIR + f'animation_{GLACIER}_{VARIABLE}_{START}_{END}.gif'
anim.save(out_fn, writer='pillow', fps=10)

# matplotlib always writes the gif to loop forever; resave it without the
# loop flag so it plays once and holds on the last frame (e.g. in Google Slides)
gif = Image.open(out_fn)
frames = [f.copy() for f in ImageSequence.Iterator(gif)]
for f in frames:
    f.info.pop('loop', None)
frames[0].save(out_fn, save_all=True, append_images=frames[1:],
               duration=gif.info['duration'])
print(f'Wrote {out_fn}', flush=True)
