"""
1:1 plot of modeled vs measured seasonal (winter and summer) index-site
mass balance, using each glacier's own best (kp, wind_factor) combination
from best.zarr. Color is glacier, marker is site.

Sites without enough winter observations to split seasons are skipped.
Usage: python plot_index_seasonal_1to1.py [loss_surface_dir]
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
BEST_ZARR = ROOT + 'output.zarr'
FIG_FN = ROOT + 'index_site_seasonal_1to1.png'

colors = ['#63c4c7', '#fcc02e', '#4D559C', '#60C252', '#BF1F6A']
markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '<', '>', '*']

with open(os.path.join(PEBSI_FP, 'project', 'sites.pkl'), 'rb') as f:
    sites_by_rgiid = pickle.load(f)
rgi_to_name = {v['6']: k for k, v in translate_rgi.items()}

df_meta = pd.read_csv(os.path.join(PEBSI_FP, 'data', 'glacier_metadata.csv'), dtype={'rgiid': str})

ds_all = xr.open_zarr(BEST_ZARR, consolidated=False)
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
            print(f'{glacier} {site}: no metadata')
            continue
        lat, lon = meta['lat'].iloc[0], meta['lon'].iloc[0]
        dist = np.hypot(lat_all[glac_points] - lat,
                        (lon_all[glac_points] - lon) * np.cos(np.deg2rad(lat)))
        point = glac_points[np.argmin(dist)]

        mb = MassBalance(glacier, site)
        if mb.dataset != 'seasonal':
            print(f'{glacier} {site}: insufficient winter data, skipped')
            continue
        mb.get_model_mb(ds_all.isel(point=point)[['mass_balance']].load())
        if np.all(np.isnan(mb.meas)):
            print(f'{glacier} {site}: no overlapping seasonal data')
            continue

        for season, idx in (('winter', mb.idx_winter), ('summer', mb.idx_summer)):
            for start, mod, meas in zip(mb.period_starts[idx], mb.mod[idx], mb.meas[idx]):
                rows.append({'glacier': glacier, 'site': site, 'season': season,
                             'start': start, 'mod': mod, 'meas': meas})

df = pd.DataFrame(rows)
glaciers = list(df['glacier'].unique())

fig, axes = plt.subplots(1, 2, figsize=(6, 4))
handles = []
for g, glacier in enumerate(glaciers):
    df_glac = df.loc[df['glacier'] == glacier]
    for s, site in enumerate(df_glac['site'].unique()):
        df_site = df_glac.loc[df_glac['site'] == site]
        for ax, season in zip(axes, ('winter', 'summer')):
            df_season = df_site.loc[df_site['season'] == season]
            marker = markers[s % len(markers)]
            ax.scatter(df_season['meas'], df_season['mod'], c=colors[g % len(colors)],
                       marker=marker, s=25, alpha=0.8, edgecolor='k', linewidth=0.3)
        # handles.append(Line2D([], [], ls='none', color=colors[g % len(colors)], marker=marker,
        #                       markeredgecolor='k', markeredgewidth=0.3, label=f'{glacier}/{site}'))

    gg_lab = ''
    for gg in [f.capitalize() for f in glacier.split('_')]:
        gg_lab += gg + ' '
    handles.append(Line2D([], [], ls='none', color=colors[g % len(colors)], marker='o',
                            markeredgecolor='k', markeredgewidth=0.3, label=gg_lab))

for ax, season in zip(axes, ('winter', 'summer')):
    df_season = df.loc[df['season'] == season]
    lims = [min(df_season['meas'].min(), df_season['mod'].min()) - 0.2,
            max(df_season['meas'].max(), df_season['mod'].max()) + 0.2]
    ax.plot(lims, lims, c='k', lw=0.8, zorder=0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')
    err = df_season['mod'] - df_season['meas']
    ax.text(0.04, 0.96, f'n = {len(err)}\nbias = {err.mean():.2f}\nRMSE = {np.sqrt((err**2).mean()):.2f}',
            transform=ax.transAxes, va='top', fontsize=9)
    ax.set_title(season.capitalize())
    ax.set_xlabel('Measured (m w.e.)')
axes[0].set_ylabel('Modeled (m w.e.)')

fig.legend(handles=handles, loc='center left', 
            bbox_to_anchor=(0.92, 0.5), fontsize=10, 
            frameon=False, ncols=1)
fig.savefig(FIG_FN, dpi=200, bbox_inches='tight')
print(f'Saved {FIG_FN}')
