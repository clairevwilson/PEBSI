"""
Summary of the observations available on each calibration glacier, per year:
  - Winter / Summer: number of index sites with a winter (bw) or summer
    (ba and bw both present) mass balance that year
  - Albedo: number of scenes in the glacier's albedo datacube (S2 + L8 + L9)
  - SAR: number of scenes in the glacier's SAR wet-extent cube
  - AWS: number of variables measured at some point in JJA by any USGS
    benchmark station on the glacier
Panels: Kahiltna, Gulkana, Kennicott / Wolverine, Alaska map, Lemon Creek.
"""
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import box
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
from project.parameters import translate_rgi

base = '/home/claire/Research/'
mb_fn = base + 'data/mass_balance/point/{G}/Input_{G}_Glaciological_Data.csv'
albedo_fn = base + 'data/albedo/{n}/RGI2000-v7.0-G-01-{n}_{s}.nc'
sar_fn = base + 'data/sar/sar_cubes/{gid}_wet_extent.nc'
aws_fp = base + 'climate_data/AWS/Benchmark/{g}/'
rgi_fn = base + 'RGI/rgi60/00_rgi60_attribs/01_rgi60_Alaska.csv'
land_fn = base + 'data/outlines/ne_50m_land/ne_50m_land.shp'
border_fn = base + 'data/outlines/ne_50m_land/ne_50m_admin_0_boundary_lines_land.shp'
fig_fn = base + 'Figs/glacier_summary.png'

YEARS = np.arange(2000, 2027)
ROWS = ['Winter MB', 'Summer MB', 'Albedo', 'SAR', 'AWS']
# one color per glacier: map line, panel spines, and bars
COLORS = {'kahiltna': '#2a78d6', 'gulkana': '#eb6834', 'kennicott': '#1baf7a',
          'wolverine': '#eda100', 'lemon_creek': '#e87ba4'}
# AWS column name -> physical variable it measures
AWS_VARS = {'site_temp': 'temp', 'RelHum': 'rh', 'WindSpeed': 'wind',
            'Barom': 'sp', 'Precip': 'precip', 'ShortwaveIn': 'SWin',
            'ShortwaveOut': 'SWout', 'LongwaveIn': 'LWin',
            'LongwaveOut': 'LWout', 'RadiationNet': 'Rnet'}

# panel layout: glacier -> (row, col, panel corner the map line goes to)
LAYOUT = {'kahiltna': (0, 0, (1, 0)), 'gulkana': (0, 1, (1, 0)),
          'kennicott': (0, 2, (0, 0)), 'wolverine': (1, 0, (1, 0)),
          'lemon_creek': (1, 2, (0, 0))}
MB_NAMES = {'kahiltna': 'Kahiltna', 'gulkana': 'Gulkana', 'kennicott': 'Kennicott',
            'wolverine': 'Wolverine', 'lemon_creek': 'LemonCreek'}


def count_mb(glacier):
    df = pd.read_csv(mb_fn.format(G=MB_NAMES[glacier]))
    winter = df[df['bw'].notna()].groupby('Year')['site_name'].nunique()
    summer = df[df['bw'].notna() & df['ba'].notna()].groupby('Year')['site_name'].nunique()
    return winter, summer


def count_albedo(glacier):
    n = translate_rgi[glacier]['7'][3:]
    times = []
    for s in ['s2', 'l8', 'l9']:
        with xr.open_dataset(albedo_fn.format(n=n, s=s)) as ds:
            times.append(ds.time.to_index())
    return pd.Series(np.concatenate(times)).dt.year.value_counts()


def count_sar(glacier):
    with xr.open_dataset(sar_fn.format(gid=translate_rgi[glacier]['7'])) as ds:
        return ds.time.to_index().year.value_counts()


def count_aws(glacier):
    fp = aws_fp.format(g=glacier.replace('_', ''))
    if not os.path.exists(fp):
        return pd.Series(dtype=int)
    present = {}   # year -> set of variables
    for root, _, files in os.walk(fp):
        for fn in files:
            if 'hourly_LVL2' not in fn:
                continue
            df = pd.read_csv(os.path.join(root, fn))
            df.columns = df.columns.str.strip()
            time = pd.to_datetime(df.iloc[:, 0], format='mixed')
            jja = df[time.dt.month.isin([6, 7, 8]).values]
            year = int(fn.split('_')[-1][:4])
            for col in jja.columns:
                var = next((v for k, v in AWS_VARS.items() if col.startswith(k)), None)
                if var and pd.to_numeric(jja[col], errors='coerce').notna().any():
                    present.setdefault(year, set()).add(var)
    return pd.Series({y: len(v) for y, v in present.items()}, dtype=float)


counts = {}
for glacier in LAYOUT:
    winter, summer = count_mb(glacier)
    rows = [winter, summer, count_albedo(glacier), count_sar(glacier), count_aws(glacier)]
    counts[glacier] = [r.reindex(YEARS, fill_value=0) for r in rows]
    print(glacier, [int(r.sum()) for r in counts[glacier]])

plt.rcParams['font.size'] = 8
fig = plt.figure(figsize=(7, 4))
outer = fig.add_gridspec(2, 3, left=0.2, hspace=0.3, wspace=0.06)
# every axis shares x; each data row shares y across glaciers
ymax = [max(counts[g][i].max() for g in counts) for i in range(len(ROWS))]
row_axes = [None] * len(ROWS)
panel_axes = {}
for glacier, (r, c, _) in LAYOUT.items():
    inner = outer[r, c].subgridspec(len(ROWS), 1, hspace=0.1)
    axes = []
    for i in range(len(ROWS)):
        ax = fig.add_subplot(inner[i], sharex=row_axes[0], sharey=row_axes[i])
        row_axes[i] = row_axes[i] or ax
        axes.append(ax)
        # ax.grid(which='y')
        ax.bar(YEARS, counts[glacier][i], width=1, color=COLORS[glacier])
        # shared y values are labeled once, on the right edge of the figure
        ax.tick_params(length=1.5, pad=1, left=False, right=True, labelleft=False,
                       labelright=(c == 2), labelbottom=(i == len(ROWS) - 1))
        for spine in ax.spines.values():
            spine.set_color(COLORS[glacier])
        # row labels on the left column only; each row of panels shares them
        if c == 0:
            ax.set_ylabel(ROWS[i], rotation=0, ha='right', va='center')
        if glacier == 'kennicott' and ROWS[i] == 'AWS':
            ax.text(0.5, 0.35, 'no station', transform=ax.transAxes,
                    ha='center', va='center', color='0.5')
    axes[0].set_title(glacier.replace('_', ' ').title() + ' Glacier', fontsize=10, pad=2)
    panel_axes[glacier] = axes

# shared limits and ticks: set once on the shared axes
row_axes[0].set_xlim(YEARS[0] - 0.5, YEARS[-1] - 0.5)
row_axes[0].set_xticks([2000, 2010, 2020])
for i, ax in enumerate(row_axes):
    ax.set_ylim(0, ymax[i] * 1.1)
    ax.set_yticks([ymax[i]])
    ax.set_yticklabels([f'{ymax[i]/1e6:.0f}M' if ymax[i] >= 1e6 else
                        f'{ymax[i]/1e3:.0f}k' if ymax[i] >= 1e4 else f'{ymax[i]:.0f}'])

# Alaska map in Alaska Albers
ax_map = fig.add_subplot(outer[1, 1])
bbox = box(-175, 50, -115, 72)
land = gpd.read_file(land_fn).clip(bbox).to_crs(3338)
borders = gpd.read_file(border_fn).clip(bbox).to_crs(3338)
land.plot(ax=ax_map, color='0.9', edgecolor='0.4', lw=0.3)
borders.plot(ax=ax_map, color='0.4', lw=0.3)
ax_map.set_xlim(-0.75e6, 1.45e6)
ax_map.set_ylim(0.65e6, 2.45e6)
ax_map.set_aspect('equal')
ax_map.set_xticks([])
ax_map.set_yticks([])

rgi = pd.read_csv(rgi_fn, index_col='RGIId')
for glacier, (_, _, corner) in LAYOUT.items():
    row = rgi.loc[f'RGI60-{translate_rgi[glacier]["6"]}']
    pt = gpd.points_from_xy([row.CenLon], [row.CenLat], crs=4326).to_crs(3338)[0]
    ax_map.plot(pt.x, pt.y, 'o', ms=3, color=COLORS[glacier], mec='k', mew=0.3, zorder=5)
    # corner = (right?, top?) of the glacier's panel
    axes = panel_axes[glacier]
    ax_end = axes[0] if corner[1] else axes[-1]
    fig.add_artist(ConnectionPatch(xyA=(pt.x, pt.y), coordsA=ax_map.transData,
                                   xyB=corner, coordsB=ax_end.transAxes,
                                   color=COLORS[glacier], lw=0.6, zorder=10))

# fig.supylabel('Number of observations', fontsize=10)
fig.savefig(fig_fn, dpi=200, bbox_inches='tight')
print(f'Wrote {fig_fn}')
