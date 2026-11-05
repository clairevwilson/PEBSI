"""
Summer index-site mass balance, measured vs modeled, to show how much of
the interannual variability (and the extreme 2004 summer) the model
reproduces.

Left: summer balance time series at one site per glacier. Right: every
site-year from all glaciers, modeled vs measured summer balance.
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PEBSI_FP = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PEBSI_FP)

from project.data_handling import MassBalance, translate_rgi

ROOT = '/ocean/projects/ees260009p/cwilson4/Output/AD_result_1/'
FIG_FN = ROOT + 'summer_anomaly_2004.png'
YEAR = 2004
TIMESERIES_SITES = [('gulkana', 'B'), ('wolverine', 'B'), ('kahiltna', 'K17b')]

glacier_colors = {'wolverine': '#63c4c7', 'kennicott': '#fcc02e', 'kahiltna': '#4D559C',
                  'lemon_creek': '#60C252', 'gulkana': '#BF1F6A'}

with open(os.path.join(PEBSI_FP, 'project', 'sites.pkl'), 'rb') as f:
    sites_by_rgiid = pickle.load(f)
rgi_to_name = {v['6']: k for k, v in translate_rgi.items()}
df_meta = pd.read_csv(os.path.join(PEBSI_FP, 'data', 'glacier_metadata.csv'), dtype={'rgiid': str})

ds_all = xr.open_zarr(ROOT + 'output.zarr', consolidated=False)
rgiid_all = ds_all['rgiid'].values
lat_all = ds_all['lat'].values
lon_all = ds_all['lon'].values

rows = []
for rgiid, site_list in sites_by_rgiid.items():
    glac_points = np.where(rgiid_all == rgiid)[0]
    if len(glac_points) == 0:
        continue
    glacier = rgi_to_name[rgiid]
    for site in site_list:
        meta = df_meta.loc[(df_meta['rgiid'] == rgiid) & (df_meta['site'] == site)]
        if meta.empty:
            continue
        lat, lon = meta['lat'].iloc[0], meta['lon'].iloc[0]
        dist = np.hypot(lat_all[glac_points] - lat,
                        (lon_all[glac_points] - lon) * np.cos(np.deg2rad(lat)))
        point = glac_points[np.argmin(dist)]

        mb = MassBalance(glacier, site)
        if mb.dataset != 'seasonal':
            continue
        mb.get_model_mb(ds_all.isel(point=point)[['mass_balance']].load())
        for i in mb.idx_summer:
            rows.append({'glacier': glacier, 'site': site,
                         'year': pd.Timestamp(mb.period_starts[i]).year,
                         'meas': mb.meas[i], 'mod': mb.mod[i]})

df = pd.DataFrame(rows).dropna(subset=['meas'])

n_rows = len(TIMESERIES_SITES)
fig = plt.figure(figsize=(10, 1.8 * n_rows + 0.5))
gs = fig.add_gridspec(n_rows, 2, width_ratios=[1.6, 1], hspace=0.45, wspace=0.28)
ax_ts = [fig.add_subplot(gs[0, 0])]
ax_ts += [fig.add_subplot(gs[r, 0], sharex=ax_ts[0]) for r in range(1, n_rows)]
ax_sc = fig.add_subplot(gs[:, 1])

for ax, (glacier, site) in zip(ax_ts, TIMESERIES_SITES):
    df_site = df.loc[(df['glacier'] == glacier) & (df['site'] == site)].set_index('year')
    df_site = df_site.reindex(range(df_site.index.min(), df_site.index.max() + 1)).reset_index()
    ax.axvspan(YEAR - 0.5, YEAR + 0.5, color='0.9', zorder=0)
    ax.plot(df_site['year'], df_site['meas'], c='k', ls='--', lw=1.5, marker='.', zorder=3)
    ax.plot(df_site['year'], df_site['mod'], c=glacier_colors[glacier], lw=2, marker='.', zorder=4)
    ax.set_title(f'{glacier.replace("_", " ").title()} {site}', loc='left', fontsize=10)
    ax.tick_params(length=5)
for ax in ax_ts[:-1]:
    plt.setp(ax.get_xticklabels(), visible=False)
ax_ts[-1].set_xlabel('Year')
fig.supylabel('Summer balance (m w.e.)', x=0.06, fontsize=10)

others = df.loc[df['year'] != YEAR]
highlight = df.loc[df['year'] == YEAR]
for glacier, color in glacier_colors.items():
    o = others.loc[others['glacier'] == glacier]
    h = highlight.loc[highlight['glacier'] == glacier]
    ax_sc.scatter(o['meas'], o['mod'], s=12, c=color, alpha=0.35, lw=0)
    ax_sc.scatter(h['meas'], h['mod'], s=55, c=color, edgecolor='k', lw=0.8, zorder=3)

lims = [np.floor(df[['meas', 'mod']].values.min()) - 0.5, np.ceil(df[['meas', 'mod']].values.max()) + 0.5]
ax_sc.plot(lims, lims, c='k', ls=':', lw=0.8, zorder=0)
ax_sc.set_xlim(lims)
ax_sc.set_ylim(lims)
ax_sc.set_aspect('equal')
ax_sc.set_xlabel('Measured summer balance (m w.e.)')
ax_sc.set_ylabel('Modeled summer balance (m w.e.)')
ax_sc.set_title('All index sites', loc='left', fontsize=10)
ax_sc.tick_params(length=5)

handles = [Line2D([], [], c='k', ls='--', lw=1.5, label='Measured'),
           Line2D([], [], c='0.4', lw=2, label='Modeled')]
handles += [Line2D([], [], ls='none', marker='o', color=c, markeredgecolor='none',
                   label=g.replace('_', ' ').title()) for g, c in glacier_colors.items()]
handles.append(Line2D([], [], ls='none', marker='o', color='w', markeredgecolor='k', label=str(YEAR)))
handles.append(Line2D([], [], c='k', ls=':', lw=0.8, label='1:1'))
ax_sc.legend(handles=handles, loc='center left', bbox_to_anchor=(1.03, 0.5), fontsize=8, frameon=False)

fig.savefig(FIG_FN, dpi=200, bbox_inches='tight')
print(f'Saved {FIG_FN}')
