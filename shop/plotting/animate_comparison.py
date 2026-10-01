"""
Side-by-side animation of measured (left) vs modeled (right) fields at
gulkana's best (kp, wind_factor) from loss_surface_1, with the
measurement sampled at the model points the same way the glacier-wide
loss compares them.

Set VARIABLE below and rerun.

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

from project.glacierwide_loss import Albedo, Wet, _load_rgi_outline
from project.parameters import translate_rgi

VARIABLE = 'albedo'  # 'albedo' or 'wet'
GLACIER = 'gulkana'
YEAR = 2021
RUN_DIR = '/ocean/projects/ees260009p/cwilson4/Output/loss_surface_1/'

# pick the combination with the lowest summed loss for this glacier
grid = np.load(RUN_DIR + 'loss_surface_grid.npz', allow_pickle=True)
loss = np.load(RUN_DIR + 'loss_surface_losses.npz', allow_pickle=True)
total = sum(loss[k] for k in loss.files if k.startswith(f'{GLACIER}_'))
i, j = np.unravel_index(np.nanargmin(total), total.shape)
kp, wf = loss['kp'][i], loss['wind_factor'][j]
c = i * len(loss['wind_factor']) + j
n_base = int(grid['n_base'])
print(f'{GLACIER}: kp={kp}, wind_factor={wf}', flush=True)

# this combination's points for this glacier
ds = xr.open_zarr(RUN_DIR + 'output.zarr', consolidated=False)
block = ds.isel(point=slice(c * n_base, (c + 1) * n_base))
gmask = block['rgiid'].values == translate_rgi[GLACIER]['6']

if VARIABLE == 'albedo':
    obs = Albedo(GLACIER, use='s2')
    obs_time = obs.ds_meas.time.values
else:
    obs = Wet(GLACIER)
    obs_time = obs.ds_meas_wet.time.values

obs_years = np.unique(pd.DatetimeIndex(obs_time).year)
if YEAR not in obs_years:
    raise ValueError(f'no {VARIABLE} observations for {GLACIER} in {YEAR}; '
                     f'years with observations: {obs_years.tolist()}')

pad = np.timedelta64(5, 'D')
sub = (block[['lon', 'lat', 'albedo', 'total_water']]
       .isel(point=gmask)
       .sel(time=slice(obs_time.min() - pad, obs_time.max() + pad))
       .load())

# match measurement to model points and times exactly as the loss does
if VARIABLE == 'albedo':
    obs.get_model_albedo(sub)
    obs.get_deltas(method='march_mean')
    meas, mod = obs.meas, obs.mod
    dates = obs.ds_meas_matched.time.values
    cmap, vmin, vmax = 'Reds_r', -0.5, 0
    label = 'albedo change from March'
elif VARIABLE == 'wet':
    obs.get_model_wet(sub)
    meas, mod = obs.meas_wet, obs.mod_wet
    dates = obs.time
    cmap, vmin, vmax = 'Blues', 0, 1
    label = 'wet'

# the whole record is matched first so albedo's March reference is the same
# one the loss uses, then only this year's dates are kept
in_year = pd.DatetimeIndex(dates).year == YEAR
if not in_year.any():
    raise ValueError(f'no {VARIABLE} observations for {GLACIER} in {YEAR} '
                     'overlap the model period')
meas, mod, dates = meas[in_year], mod[in_year], dates[in_year]

proj = Transformer.from_crs('EPSG:4326', obs.crs, always_xy=True)
x, y = proj.transform(sub.lon.values, sub.lat.values)

# voronoi cell of each model point, clipped to the RGI outline; four far-off
# dummy points keep every real point's cell finite
outline = _load_rgi_outline(translate_rgi[GLACIER]['6']).to_crs(obs.crs).geometry.iloc[0]
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

cmap = plt.get_cmap(cmap).with_extremes(bad='lightgray')
# size the figure to the glacier's shape so the maps fill it; the height is
# snapped to a quarter inch so the frame is a whole number of pixels (the
# gif writer shears frames that aren't)
xmin, ymin, xmax, ymax = outline.bounds
height = np.ceil((4.5 * (ymax - ymin) / (xmax - xmin) + 0.6) * 4) / 4
fig, axes = plt.subplots(1, 2, figsize=(10, height), dpi=100,
                         sharex=True, sharey=True, layout='compressed')
cells = []
name = {'albedo': 'albedo', 'wet': 'wet extent'}[VARIABLE]
for ax, data, source in zip(axes, [meas, mod], ['Measured', 'Modeled']):
    cell = PathCollection(paths, cmap=cmap, edgecolors='none')
    cell.set_clim(vmin, vmax)
    cell.set_array(data[0].astype(float))
    ax.add_collection(cell)
    ax.plot(*outline.exterior.xy, color='k', lw=0.8)
    ax.set_title(f'{source} {name}')
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    cells.append(cell)
fig.colorbar(cells[1], ax=axes, label=label, shrink=0.8)
title = fig.suptitle(str(pd.Timestamp(dates[0]).date()))


def update(t):
    cells[0].set_array(meas[t].astype(float))
    cells[1].set_array(mod[t].astype(float))
    title.set_text(str(pd.Timestamp(dates[t]).date()))
    return cells + [title]


anim = FuncAnimation(fig, update, frames=len(dates), interval=250)
out_fn = RUN_DIR + f'comparison_{GLACIER}_{VARIABLE}_{YEAR}.gif'
anim.save(out_fn, writer='pillow', fps=3)

# matplotlib always writes the gif to loop forever; resave it without the
# loop flag so it plays once and holds on the last frame (e.g. in Google Slides)
gif = Image.open(out_fn)
frames = [f.copy() for f in ImageSequence.Iterator(gif)]
for f in frames:
    f.info.pop('loop', None)
frames[0].save(out_fn, save_all=True, append_images=frames[1:],
               duration=gif.info['duration'])
print(f'Wrote {out_fn}', flush=True)
