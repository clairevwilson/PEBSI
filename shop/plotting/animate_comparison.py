"""
Side-by-side animation of measured (left) vs modeled (right) fields at
a glacier's best (kp, wind_factor) from a loss surface run, with the
measurement sampled at the model points the same way the glacier-wide
loss compares them.

Settings come from the command line; see the parser below.

@author: clairevwilson
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
from PIL import Image, ImageSequence
from matplotlib.collections import PathCollection
from matplotlib.path import Path
from scipy.spatial import Voronoi
from shapely.geometry import Polygon
from shapely.plotting import plot_line, plot_polygon

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from project.glacierwide_loss import Albedo, Wet, _load_rgi_outline
from project.parameters import translate_rgi

parser = argparse.ArgumentParser()
parser.add_argument('--variable', default='wet', choices=['albedo', 'wet'])
parser.add_argument('--deltas', action=argparse.BooleanOptionalAction, default=False,
                    help='albedo only: change from March, or --no-deltas for raw albedo')
parser.add_argument('--glacier', default='gulkana')
parser.add_argument('--year', type=int, default=None, help='omit for the whole record')
parser.add_argument('--run_dir', default='/ocean/projects/ees260009p/cwilson4/Output/loss_surface_0/')
args = parser.parse_args()
VARIABLE = args.variable
DELTAS = args.deltas
GLACIER = args.glacier
YEAR = args.year
RUN_DIR = args.run_dir

if os.path.exists(RUN_DIR + 'best.zarr'):
    ds = xr.open_zarr(RUN_DIR + 'best.zarr', consolidated=False)
else:
    ds = xr.open_zarr(RUN_DIR + 'output.zarr', consolidated=False)
gmask = ds['rgiid'].values == translate_rgi[GLACIER]['6']
if 'kp' in ds:
    print(f"{GLACIER}: kp={ds['kp'].values[gmask][0]}, "
          f"wind_factor={ds['wind_factor'].values[gmask][0]}", flush=True)

if VARIABLE == 'albedo':
    obs = Albedo(GLACIER, use='s2')
    obs_time = obs.ds_meas.time.values
else:
    # older runs only saved total_water
    water_var = 'top_water' if 'top_water' in ds else 'total_water'
    obs = Wet(GLACIER, water_var=water_var)
    obs_time = obs.ds_meas_wet.time.values

obs_years = np.unique(pd.DatetimeIndex(obs_time).year)
if YEAR is not None and YEAR not in obs_years:
    raise ValueError(f'no {VARIABLE} observations for {GLACIER} in {YEAR}; '
                     f'years with observations: {obs_years.tolist()}')

pad = np.timedelta64(7, 'D')
sub = (ds[['lon', 'lat', 'albedo'] + ([water_var] if VARIABLE == 'wet' else [])]
       .isel(point=gmask)
       .sel(time=slice(obs_time.min() - pad, obs_time.max() + pad))
       .load())

# match measurement to model points and times exactly as the loss does
if VARIABLE == 'albedo':
    obs.get_model_albedo(sub)
    raw_mod = obs.mod
    if DELTAS:
        obs.get_deltas(method='march_mean')
        cmap, vmin, vmax = 'Reds_r', -0.5, 0
        label = 'albedo change from March'
    else:
        cmap, vmin, vmax = 'Greys_r', 0, 1
        label = 'albedo'
    meas, mod = obs.meas, obs.mod
    dates = obs.ds_meas_matched.time.values
elif VARIABLE == 'wet':
    obs.get_model_wet(sub)
    meas, mod = obs.meas_wet, obs.mod_wet
    dates = obs.time
    # wetness is binary, so the colorbar is just a dry block and a wet block
    cmap, vmin, vmax = ListedColormap(plt.get_cmap('Blues')([0., 1.])), 0, 1
    label = None

# the whole record is matched first so albedo's March reference is the same
# one the loss uses, then only this year's dates are kept
keep = pd.DatetimeIndex(dates).year == YEAR if YEAR is not None else np.ones(len(dates), bool)
if not keep.any():
    raise ValueError(f'no {VARIABLE} observations for {GLACIER} in {YEAR} '
                     'overlap the model period')
meas, mod, dates = meas[keep], mod[keep], dates[keep]

# the model is shown every day from the first to the last measurement, while
# each measurement is held until the next one arrives; winters are skipped
meas_days = pd.DatetimeIndex(dates).normalize()
years = np.unique(meas_days.year)
days = pd.date_range(meas_days[0], meas_days[-1], freq='D')
days = days[days.month.isin(range(3, 11)) & days.year.isin(years) & (days <= ds.time.values[-1])]
if VARIABLE == 'albedo':
    # sample the model at the overpass time of day, relative to the same
    # per-year March reference the loss subtracts
    offset = raw_mod[keep] - mod
    refs = np.array([np.nanmean(offset[meas_days.year == y], axis=0) for y in years])
    march_ref = refs[np.searchsorted(years, days.year)]
    time_of_day = (pd.DatetimeIndex(dates) - meas_days).mean()
    mod_daily = (sub['albedo'].sel(time=days + time_of_day, method='nearest')
                 .transpose('time', 'point').values - march_ref)
else:
    water_min = sub[water_var].resample(time='1d').min()
    mod_daily = ((water_min > obs.water_threshold * 1000).sel(time=days)
                 .transpose('time', 'point').values)
meas_idx = np.searchsorted(meas_days, days, side='right') - 1
# SAR revisits about weekly, so it is held longer than the optical scenes
max_age = pd.Timedelta(days=10 if VARIABLE == 'wet' else 5)
stale = (days - meas_days[meas_idx]) > max_age

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

cmap = plt.get_cmap(cmap).with_extremes(bad='none')
# size the figure to the glacier's shape so the maps fill it; the height is
# snapped to a quarter inch so the frame is a whole number of pixels (the
# gif writer shears frames that aren't)
xmin, ymin, xmax, ymax = outline.bounds
height = np.ceil((3.6 * (ymax - ymin) / (xmax - xmin) + 0.48) * 4) / 4
fig, axes = plt.subplots(1, 2, figsize=(8, height), dpi=100,
                         sharex=True, sharey=True, layout='compressed')
cells = []
name = {'albedo': 'Albedo', 'wet': 'Water extent'}[VARIABLE]
for ax, data, source in zip(axes, [meas, mod_daily], ['Measured', 'Modeled']):
    cell = PathCollection(paths, cmap=cmap, edgecolors='none')
    cell.set_clim(vmin, vmax)
    cell.set_array(data[0].astype(float))
    ax.add_collection(cell)
    # hatching under the cells shows through wherever the data is NaN
    plot_polygon(outline, ax=ax, add_points=False, facecolor='none', edgecolor='darkgray',
                 hatch='///', linewidth=0, zorder=0)
    plot_line(outline.boundary, ax=ax, add_points=False, color='k', linewidth=0.8)
    ax.set_title(f'{source} {name}')
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    cells.append(cell)
cbar = fig.colorbar(cells[1], ax=axes, label=label, shrink=0.8)
if VARIABLE == 'wet':
    cbar.set_ticks([0.25, 0.75], labels=['dry', 'wet'])
title = fig.suptitle(str(days[0].date()))


def update(t):
    # the measured panel goes blank when the latest measurement is too old
    cells[0].set_visible(not stale[t])
    cells[0].set_array(meas[meas_idx[t]].astype(float))
    cells[1].set_array(mod_daily[t].astype(float))
    title.set_text(str(days[t].date()))
    return cells + [title]


anim = FuncAnimation(fig, update, frames=len(days), interval=125)
suffix = '_raw' if VARIABLE == 'albedo' and not DELTAS else ''
out_fn = RUN_DIR + f'comparison_{GLACIER}_{VARIABLE}{suffix}_{YEAR or "all"}.gif'
anim.save(out_fn, writer='pillow', fps=16)

# matplotlib always writes the gif to loop forever; resave it without the
# loop flag so it plays once and holds on the last frame (e.g. in Google Slides)
gif = Image.open(out_fn)
frames = [f.copy() for f in ImageSequence.Iterator(gif)]
for f in frames:
    f.info.pop('loop', None)
frames[0].save(out_fn, save_all=True, append_images=frames[1:],
               duration=gif.info['duration'])
print(f'Wrote {out_fn}', flush=True)
