"""
Figures comparing the controlled wildfire/heatwave experiments (base,
no_fires, no_heatwaves, no_either): cumulative divergence, synergy,
elevation profiles, energy-balance attribution, seasonal timing,
the effects as a share of observed loss, and a single-summer case study.

Reads per_glacier.nc and per_point_annual.nc written by
shop/postprocessing/aggregate_per_glacier.py and saves PNGs into base_fp/figs/.
"""
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

sys.path.append('/jet/home/cwilson4/PEBSI/')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from glacierwide_loss import MassBalance, translate_rgi

base_fp = '/home/claire/Research/Output/controlled/'
fig_fp = base_fp + 'figs/'

# Inputs for the figures below. The effect of a forcing is Base minus the run with
# that forcing removed, so a negative effect means the forcing caused extra loss.
i = 0
runs = {'Base': 'base_merra2', 'No fires': 'no_fires_merra2', 'No heatwaves': 'no_heatwaves_merra2',
        'Neither': 'no_either_merra2'}
glac = {sim: xr.open_dataset(base_fp + f'{run}_{i}/per_glacier.nc').load() for sim, run in runs.items()}
pts = {sim: xr.open_dataset(base_fp + f'{run}_{i}/per_point_annual.nc').load() for sim, run in runs.items()}
glaciers = glac['Base'].glacier.values

effects = {'Wildfires': 'No fires', 'Heatwaves': 'No heatwaves', 'Both': 'Neither'}
effect_colors = {'Wildfires': 'r', 'Heatwaves': 'b', 'Both': 'gray', 'Interaction': 'k'}

# the run starts 2000-01-01, so WY2000 is missing Oct-Dec 1999
first_full_wy = 2001

def effect(dsets, name, var='mass_balance'):
    if name == 'Interaction':
        return effect(dsets, 'Both', var) - effect(dsets, 'Wildfires', var) - effect(dsets, 'Heatwaves', var)
    return dsets['Base'][var] - dsets[effects[name]][var]

def water_years(da):
    annual = da.resample(time='YS-OCT').sum()
    annual = annual.assign_coords(time=annual.time.dt.year + 1).rename(time='water_year')
    return annual.sel(water_year=slice(first_full_wy, None))

def fmt_name(glacier):
    return ' '.join(f.capitalize() for f in glacier.split('_'))

def daily_albedo(ds):
    daily = ds[['shortwave_net', 'shortwave_in']].resample(time='D').sum()
    return (1 - daily['shortwave_net'] / daily['shortwave_in']).to_dataset(name='albedo')

alb = {sim: daily_albedo(ds) for sim, ds in glac.items()}

# daily heatwave/wildfire flags and BC+OC deposition (May-Sep only) at each glacier's centroid
events = xr.open_dataset(base_fp + '../../climate_data/extreme_events_daily_ukesm.nc')
ev = events.sel(x=glac['Base'].lon, y=glac['Base'].lat, method='nearest').load()
# fill the winters so fill_between never spans the gap between Sep 30 and May 1
ev = ev.reindex(time=pd.date_range(ev.time.values[0], ev.time.values[-1], freq='D'),
                fill_value={'heatwave': False, 'wildfire': False})

# Cumulative divergence: running effect on glacier-wide mass balance, with event days shaded
fig, axes_2d = plt.subplots(2, 3, figsize=(8, 4.5), layout='constrained', sharex=True)
axes = axes_2d.flatten()[:-1]
secondary = axes_2d.flatten()[-1]

for ax, glacier in zip(axes, glaciers):
    ev_glacier = ev.sel(glacier=glacier)
    for flag, color in [('wildfire', 'r'), ('heatwave', 'b')]:
        ax.fill_between(ev_glacier.time, 0, 1, where=ev_glacier[flag].values, color=color,
                        alpha=0.15, linewidth=0, transform=ax.get_xaxis_transform())
    for name in ['Wildfires', 'Heatwaves', 'Both']:
        cumulative = effect(glac, name).sel(glacier=glacier).resample(time='D').sum().cumsum()
        ax.plot(cumulative.time, cumulative, color=effect_colors[name], linewidth=1)
    ax.axhline(0, linewidth=0.5, c='k')
    ax.set_title(fmt_name(glacier), fontsize=10)
    ax.tick_params(axis='x', labelrotation=45)

for name in ['Wildfires', 'Heatwaves', 'Both']:
    secondary.plot(np.nan, np.nan, color=effect_colors[name], label=f'{name} effect')
secondary.bar(np.nan, np.nan, color='r', alpha=0.15, label='Wildfire days')
secondary.bar(np.nan, np.nan, color='b', alpha=0.15, label='Heatwave days')
secondary.axis('off')
secondary.legend(loc='center')

fig.supylabel('Cumulative change in mass balance (m w.e.)')
plt.savefig(fig_fp + 'cumulative_divergence.png', dpi=300, bbox_inches='tight')
plt.close()

# Synergy: does removing both forcings do more than the sum of removing each one?
names = ['Wildfires', 'Heatwaves', 'Both', 'Interaction']
width = 0.2

# each bar figure is saved once per stage, adding effects in turn; hidden bars are still drawn
# (invisible) so every stage has the same bar positions and y limits
stages = {'fires': ['Wildfires'],
          'fires_heatwaves': ['Wildfires', 'Heatwaves'],
          'all': names}
bar_labels = {'Wildfires': 'No wildfires', 'Heatwaves': 'No heatwaves', 'Both': 'Neither',
              'Interaction': 'Interaction'}

# sigma: optional {glacier: value} drawn as a black line spanning that glacier's bars,
# only on the final stage with every effect shown
def effect_bars(values, ylabel, fn, sigma=None):
    for stage, shown in (stages if sigma is None else {'all': names}).items():
        fig, ax = plt.subplots(figsize=(7, 3.5), layout='constrained')
        for j, glacier in enumerate(glaciers):
            for k, name in enumerate(names):
                ax.bar(j + (k - 1.5) * width, values[glacier][name], width, facecolor=effect_colors[name],
                       edgecolor='k', linewidth=0.5, visible=name in shown,
                       label=bar_labels[name] if j == 0 and name in shown else None)
            if sigma is not None:
                ax.hlines(-sigma[glacier], j - 2 * width, j + 2 * width, colors='k', linewidth=1.5,
                          label='Hugonnet uncertainty' if j == 0 else None)
        ax.set_xticks(range(len(glaciers)))
        ax.set_xticklabels([fmt_name(g) for g in glaciers])
        ax.axhline(0, linewidth=0.5, c='k')
        ax.set_ylabel(ylabel)
        fig.legend(loc='outside upper center', ncol=len(names) + (sigma is not None), fontsize=11, frameon=False)
        plt.savefig(fig_fp + fn.replace('.png', f'_{stage}.png'), dpi=300, bbox_inches='tight')
        plt.close()

rates = {glacier: {name: float(water_years(effect(glac, name).sel(glacier=glacier)).mean()) * 100 for name in names}
         for glacier in glaciers}
for glacier, r in rates.items():
    print(f'{glacier:<12} fires {r["Wildfires"]:.2f}, heatwaves {r["Heatwaves"]:.2f}, '
          f'both {r["Both"]:.2f}, interaction {r["Interaction"]:.2f} cm w.e. a-1 '
          f'({r["Interaction"] / r["Both"] * 100:.0f}% of both)')
effect_bars(rates, r'Change in mass balance (cm w.e. a$^{-1}$)', 'synergy.png')

# same bars against the 2000-2020 Hugonnet uncertainty in the glacier-wide rate
hugonnet_sigma = {glacier: MassBalance(glacier).sigma * 100 for glacier in glaciers}
effect_bars(rates, r'Change in mass balance (cm w.e. a$^{-1}$)', 'synergy_hugonnet.png', sigma=hugonnet_sigma)

# Total modeled mass balance over the whole run, stacked into the part each extreme caused
# and the rest; each stage splits off the effects shown so the full bar is always Base
totals = {glacier: {sim: float(glac[sim]['mass_balance'].sel(glacier=glacier).sum()) for sim in runs}
          for glacier in glaciers}
parts = {glacier: {name: float(effect(glac, name).sel(glacier=glacier).sum()) for name in names}
         for glacier in glaciers}
stack_names = ['Wildfires', 'Heatwaves']
stack_colors = {'Other': '0.8', **effect_colors}

for stage in ['fires', 'fires_heatwaves']:
    shown = stages[stage]
    fig, ax = plt.subplots(figsize=(7, 3.5), layout='constrained')
    for j, glacier in enumerate(glaciers):
        other = totals[glacier]['Base'] - sum(parts[glacier][name] for name in shown)
        bottom = 0
        for name in ['Other'] + shown:
            height = other if name == 'Other' else parts[glacier][name]
            ax.bar(j, height, 0.6, bottom=bottom, facecolor=stack_colors[name], edgecolor='k',
                   linewidth=0.5, label=name if j == 0 else None)
            bottom += height
    ax.set_xticks(range(len(glaciers)))
    ax.set_xticklabels([fmt_name(g) for g in glaciers])
    ax.axhline(0, linewidth=0.5, c='k')
    ax.set_ylim(1.05 * min(t['Base'] for t in totals.values()), 0)
    ax.set_ylabel('Mass balance, 2000$-$2025 (m w.e.)')
    fig.legend(loc='outside upper center', ncol=len(stack_names) + 1, fontsize=11, frameon=False)
    plt.savefig(fig_fp + f'familiar_units_{stage}.png', dpi=300, bbox_inches='tight')
    plt.close()

shares = {name: {glacier: parts[glacier][name] / totals[glacier]['Base'] * 100 for glacier in glaciers}
          for name in names}
for glacier in glaciers:
    print(f'{glacier:<12} total {totals[glacier]["Base"]:.2f} m w.e., ' +
          ', '.join(f'{name.lower()} {shares[name][glacier]:.1f}%' for name in names))
for name in names:
    print(f'{name:<12} {min(shares[name].values()):.1f}-{max(shares[name].values()):.1f}% of modeled mass balance')

# Elevation profile of each effect, with hypsometry and the Base vs Neither ELAs
bin_width = 100

def binned_profile(values, glacier):
    point_mask = pts['Base'].rgiid.values == translate_rgi[glacier]['6']
    elev = pts['Base'].elev.values[point_mask]
    weight = pts['Base'].weight.values[point_mask]
    values = values[point_mask]
    edges = np.arange(elev.min() // bin_width, elev.max() // bin_width + 2) * bin_width
    idx = np.digitize(elev, edges) - 1
    centers, profile, area = [], [], []
    for b in np.unique(idx):
        in_bin = idx == b
        centers.append(edges[b] + bin_width / 2)
        profile.append(np.sum(values[in_bin] * weight[in_bin]) / np.sum(weight[in_bin]))
        area.append(np.sum(weight[in_bin]))
    return np.array(centers), np.array(profile), np.array(area)

def ela(centers, profile):
    crossing = np.where(np.diff(np.sign(profile)) != 0)[0]
    if len(crossing) == 0:
        return np.nan
    k = crossing[0]
    return centers[k] + (centers[k + 1] - centers[k]) * -profile[k] / (profile[k + 1] - profile[k])

def mean_annual(da):
    return da.sel(water_year=slice(first_full_wy, None)).mean('water_year').values

fig, axes_2d = plt.subplots(2, 3, figsize=(8, 6), layout='constrained')
axes = axes_2d.flatten()[:-1]
secondary = axes_2d.flatten()[-1]

for ax, glacier in zip(axes, glaciers):
    for name in ['Wildfires', 'Heatwaves', 'Both']:
        centers, profile, area = binned_profile(mean_annual(effect(pts, name)) * 100, glacier)
        ax.plot(profile, centers, color=effect_colors[name], linewidth=1)
    ax.set_ylim(min(centers), max(centers))
    ax.set_xlim(-16, 1)

    ax_area = ax.twiny()
    ax_area.barh(centers, area, height=bin_width, color='0.9')
    ax_area.set_xlim(0, area.max() * 3)
    ax_area.set_xticks([])
    ax_area.set_ylim(min(centers), max(centers))
    ax.set_zorder(ax_area.get_zorder() + 1)
    ax.patch.set_visible(False)

    ela_base = ela(*binned_profile(mean_annual(pts['Base']['mass_balance']), glacier)[:2])
    ela_neither = ela(*binned_profile(mean_annual(pts['Neither']['mass_balance']), glacier)[:2])
    ax.axhline(ela_base, c='k', linewidth=0.8)
    ax.axhline(ela_neither, c='k', linewidth=0.8, linestyle='--')
    ax.axvline(0, c='k', linewidth=0.5)
    ax.set_title(fmt_name(glacier), fontsize=10)
    print(f'{glacier:<12} ELA base {ela_base:.0f} m, neither {ela_neither:.0f} m, shift {ela_base - ela_neither:+.0f} m')

for name in ['Wildfires', 'Heatwaves', 'Both']:
    secondary.plot(np.nan, np.nan, color=effect_colors[name], label=f'{name} effect')
secondary.plot(np.nan, np.nan, c='k', linewidth=0.8, label='ELA, Base')
secondary.plot(np.nan, np.nan, c='k', linewidth=0.8, linestyle='--', label='ELA, Neither')
secondary.barh(np.nan, np.nan, color='0.9', label='Area')
secondary.axis('off')
secondary.legend(loc='center')

fig.supxlabel(r'Change in mass balance (cm w.e. a$^{-1}$)')
fig.supylabel('Elevation (m a.s.l.)')
plt.savefig(fig_fp + 'elevation_profile.png', dpi=300, bbox_inches='tight')
plt.close()

# Energy-balance attribution: mean change in each surface flux over May-Sep
fluxes = {'shortwave_net': 'SW$_{net}$', 'longwave_net': 'LW$_{net}$', 'sensible_heat': 'Sensible',
          'latent_heat': 'Latent', 'rain_heat': 'Rain', 'ground_heat': 'Ground', 'melt_energy': 'Melt energy'}
melt_season = glac['Base'].time.dt.month.isin([5, 6, 7, 8, 9])
names = ['Wildfires', 'Heatwaves', 'Both']

fig, axes_2d = plt.subplots(2, 3, figsize=(9, 5.5), layout='constrained', sharey=True)
axes = axes_2d.flatten()[:-1]
secondary = axes_2d.flatten()[-1]

for ax, glacier in zip(axes, glaciers):
    for k, name in enumerate(names):
        for f, flux in enumerate(fluxes):
            change = float(effect(glac, name, flux).sel(glacier=glacier, time=melt_season).mean())
            ax.bar(f + (k - 1) * 0.27, change, 0.27, color=effect_colors[name])
    ax.set_xticks(range(len(fluxes)))
    ax.set_xticklabels(fluxes.values(), rotation=60, fontsize=8)
    ax.axhline(0, linewidth=0.5, c='k')
    ax.set_title(fmt_name(glacier), fontsize=10)

for name in names:
    secondary.bar(np.nan, np.nan, color=effect_colors[name], label=f'{name} effect')
secondary.axis('off')
secondary.legend(loc='center')

fig.supylabel(r'May$-$Sep mean flux change (W m$^{-2}$)')
plt.savefig(fig_fp + 'energy_balance_attribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Seasonal timing: day-of-year climatology of the daily effect on mass balance and albedo
doy_range = (120, 280)
smooth_days = 7
names = ['Wildfires', 'Heatwaves', 'Both']

def climatology(da):
    da = da.sel(time=slice(f'{first_full_wy - 1}-10-01', None))
    clim = da.groupby('time.dayofyear').mean()
    return clim.rolling(dayofyear=smooth_days, center=True).mean().sel(dayofyear=slice(*doy_range))

fig, axes = plt.subplots(2, len(glaciers), figsize=(11, 4.5), layout='constrained', sharex=True, sharey='row')
for j, glacier in enumerate(glaciers):
    for name in names:
        daily_mb = effect(glac, name).sel(glacier=glacier).resample(time='D').sum() * 1000
        mb_clim = climatology(daily_mb)
        alb_clim = climatology(effect(alb, name, 'albedo').sel(glacier=glacier))
        axes[0, j].plot(mb_clim.dayofyear, mb_clim, color=effect_colors[name], linewidth=1, label=name)
        axes[1, j].plot(alb_clim.dayofyear, alb_clim, color=effect_colors[name], linewidth=1)
    axes[0, j].set_title(fmt_name(glacier), fontsize=10)
    for ax in axes[:, j]:
        ax.axhline(0, linewidth=0.5, c='k')

axes[0, 0].set_ylabel(r'$\Delta$ mass balance' + '\n' + r'(mm w.e. d$^{-1}$)')
axes[1, 0].set_ylabel(r'$\Delta$ albedo')
axes[0, 0].legend(fontsize=7)
fig.supxlabel('Day of year')
plt.savefig(fig_fp + 'seasonal_timing.png', dpi=300, bbox_inches='tight')
plt.close()

# Case study of one fire summer at one glacier
case_glacier = 'gulkana'
case_year = 2004

season = slice(f'{case_year}-05-01', f'{case_year}-09-30')
ev_case = ev.sel(glacier=case_glacier, time=season)

fig, axes = plt.subplots(4, 1, figsize=(6, 7), sharex=True, layout='constrained')
for ax in axes:
    for flag, color in [('wildfire', 'r'), ('heatwave', 'b')]:
        ax.fill_between(ev_case.time, 0, 1, where=ev_case[flag].values, color=color,
                        alpha=0.15, linewidth=0, transform=ax.get_xaxis_transform())

axes[0].plot(ev_case.time, ev_case['bc'] * 1e6, c='k', label='Base')
axes[0].plot(ev_case.time, ev_case['bc_nofire'] * 1e6, c='r', label='No fires')
for sim, color in [('Base', 'k'), ('No fires', 'r')]:
    albedo = alb[sim]['albedo'].sel(glacier=case_glacier, time=season)
    melt = glac[sim]['melt'].sel(glacier=case_glacier, time=season).resample(time='D').sum() * 1000
    axes[1].plot(albedo.time, albedo, c=color)
    axes[2].plot(melt.time, melt, c=color)
for name in ['Wildfires', 'Heatwaves']:
    cumulative = effect(glac, name).sel(glacier=case_glacier, time=season).resample(time='D').sum().cumsum() * 100
    axes[3].plot(cumulative.time, cumulative, c=effect_colors[name], label=f'{name} effect')

axes[0].set_ylabel('BC+OC deposition\n' + r'(mg m$^{-2}$ d$^{-1}$)')
axes[1].set_ylabel('Albedo')
axes[2].set_ylabel('Melt\n' + r'(mm w.e. d$^{-1}$)')
axes[3].set_ylabel('Cumulative change\n(cm w.e.)')
axes[3].axhline(0, linewidth=0.5, c='k')
axes[0].legend(fontsize=8)
axes[3].legend(fontsize=8)
axes[0].set_title(f'{fmt_name(case_glacier)} {case_year}', fontsize=10)
plt.savefig(fig_fp + f'case_study_{case_glacier}_{case_year}.png', dpi=300, bbox_inches='tight')
plt.close()

# Extra mass loss per water year, stacked by forcing (Wildfires, Heatwaves, Both). WY2004 and WY2019
# get their own colors. Other years are gray, unless they are most of a glacier's total, in which
# case the largest of them get colors too (up to MAX_EXTRA_YEARS).
from matplotlib.patches import Patch

SPECIAL_YEARS = [2004, 2019]
MAX_OTHER_SHARE = 0.5  # gray above this share of the Both total gets extra colors
MAX_EXTRA_YEARS = 6
# the shared palette (shop/plotting/plotting_fxns.py all_colors) without its gray, which 'Other years' uses
year_colors = ['#63c4c7', '#fcc02e']  # fixed to 2004 and 2019
extra_colors = ['#4D559C', '#60C252', '#BF1F6A', '#F77808', '#298282', '#FF89B0']
top_names = ['Wildfires', 'Heatwaves', 'Both']

fig, axes_2d = plt.subplots(2, 3, figsize=(7.5, 6), layout='constrained', sharey=True)
axes = axes_2d.flatten()[:-1]
legend_ax = axes_2d.flatten()[-1]

for ax, glacier in zip(axes, glaciers):
    extra = {name: -water_years(effect(glac, name).sel(glacier=glacier)) * 100 for name in top_names}
    both = extra['Both']
    total = float(both.sum())
    # other years, largest first, get their own colors until gray is no longer most of the total
    other = both.water_year.values[np.argsort(both.values)[::-1]]
    other = [int(y) for y in other if y not in SPECIAL_YEARS]
    extras = []
    while (total - float(both.sel(water_year=SPECIAL_YEARS + extras).sum())) / total > MAX_OTHER_SHARE \
            and other and len(extras) < MAX_EXTRA_YEARS:
        extras.append(other.pop(0))
    named = SPECIAL_YEARS + extras
    print(f'{glacier:<12} named years {named}, gray {100 * (1 - float(both.sel(water_year=named).sum()) / total):.0f}% of Both')

    for j, name in enumerate(top_names):
        v = extra[name]
        heights = [float(v.sel(water_year=y)) for y in named]
        heights.append(float(v.sum()) - sum(heights))
        colors = year_colors + extra_colors[:len(extras)] + ['0.85']
        bottom = 0
        for k, (h, c) in enumerate(zip(heights, colors)):
            ax.bar(j, h, 0.6, bottom=bottom, facecolor=c, edgecolor='k', linewidth=0.5)
            if name == 'Both' and len(SPECIAL_YEARS) <= k < len(named):
                ax.text(j, bottom + h / 2, str(named[k]), ha='center', va='center', fontsize=9,
                        color='w' if c == '#4D559C' else 'k')
            bottom += h
    ax.set_xticks(range(len(top_names)))
    ax.set_xticklabels(top_names)
    ax.axhline(0, linewidth=0.5, c='k')
    ax.set_title(fmt_name(glacier), fontsize=9)

handles = [Patch(facecolor=c, edgecolor='k', linewidth=0.5, label=str(y))
           for c, y in zip(year_colors, SPECIAL_YEARS)]
handles.append(Patch(facecolor='0.85', edgecolor='k', linewidth=0.5, label='Other years'))
legend_ax.legend(handles=handles, loc='center', fontsize=9, frameon=False)
legend_ax.axis('off')

fig.supylabel('Extra mass loss per water year (cm w.e.)')
plt.savefig(fig_fp + 'top_loss_years.png', dpi=300, bbox_inches='tight')
plt.close()

# Heatmap of extra loss by water year: rows are glaciers, panels are forcings stacked vertically.
# Color is each year's share of that glacier's total extra loss from that forcing.
heat_cmap = mpl.colormaps['magma']
years = water_years(effect(glac, 'Both')).water_year.values
shares, totals = {}, {}
for name in top_names:
    loss = np.array([-water_years(effect(glac, name).sel(glacier=g)).values * 100 for g in glaciers])
    if (loss < 0).any():
        print(f'{name}: some years are net gains; they show as zero share')
    totals[name] = loss.sum(axis=1)
    shares[name] = np.clip(loss / totals[name][:, None], 0, None)

fig, axes = plt.subplots(3, 1, figsize=(7.5, 4.2), layout='constrained', sharex=True)
fig.get_layout_engine().set(hspace=0.02)
vmax = max(s.max() for s in shares.values())
for ax, name in zip(axes, top_names):
    im = ax.imshow(shares[name], cmap=heat_cmap, vmin=0, vmax=vmax, aspect='auto', interpolation='nearest')
    ax.set_yticks(range(len(glaciers)))
    ax.set_yticklabels([fmt_name(g) for g in glaciers], fontsize=9)
    ax.set_title(name, fontsize=9, loc='left')
axes[-1].set_xticks(range(0, len(years), 4))
axes[-1].set_xticklabels(years[::4], fontsize=9)
fig.colorbar(im, ax=axes, label="Share of glacier's total extra loss (per forcing)", shrink=0.9)
plt.savefig(fig_fp + 'loss_year_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
for name in top_names:
    print(f'{name:<10} totals (cm w.e.): ' + ', '.join(f'{g} {t:.0f}' for g, t in zip(glaciers, totals[name])))

# 3D version of the heatmap for Both: each bar's height is that year's extra loss (cm w.e.) and its color
# is the same share as the heatmap. Glaciers run front to back by total extra loss, so the largest total
# sits at the back and the rest step down toward the front.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

order = np.argsort(totals['Both'])
n_years = len(years)

# year bars are full width so neighbouring years touch
fig = plt.figure(figsize=(7.5, 6))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.computed_zorder = False  # keep the draw order below instead of matplotlib's depth sort
zmax_both = (shares['Both'] * totals['Both'][:, None]).max()
# drawn year by year, and within each year from Kennicott (back) to Lemon Creek (front), so nearer
# bars draw over farther ones; years under 1 cm w.e. are skipped
for yr in range(n_years):
    for yi, gi in reversed(list(enumerate(order))):
        height = shares['Both'][gi][yr] * totals['Both'][gi]
        if height >= 1.0:
            # bar3d draws from its corner, so offset by half a bar to center it on the tick
            ax.bar3d(yr - 0.5, yi - 0.25, 0, 1.0, 0.5, height, color=heat_cmap(shares['Both'][gi][yr] / vmax), shade=True)
ax.set_zlim(0, zmax_both)
ax.set_yticks(range(len(order)))
ax.set_zticks([0, 25,50, 75,100,125, 150])
ax.tick_params(axis='z', pad=0)
ax.set_yticklabels([])
for yi, gi in enumerate(order):  # glacier names drawn at the end of each row
    # the year-axis direction is right and down on screen, so push along it and lift in z to cancel the drop
    ax.text(n_years + 3, yi, 1, fmt_name(glaciers[gi]), fontsize=9, ha='left', va='center')
ax.set_xticks(range(0, n_years, 4))
ax.set_xticklabels(years[::4], fontsize=9)
ax.set_zlabel('cm w.e.', fontsize=9, labelpad=4)
ax.tick_params(labelsize=9)
ax.view_init(elev=25, azim=-60)
ax.set_box_aspect((2.4, 1.6, 1.0))
sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, vmax), cmap=heat_cmap)
sm.set_array([])
cax = fig.add_axes([0.94, 0.3, 0.02, 0.4])
fig.colorbar(sm, cax=cax, label="Share of glacier's total extra loss")
plt.savefig(fig_fp + 'loss_year_3d_both.png', dpi=300, bbox_inches='tight')
plt.close()

# Both forcing as flat bars: one row per glacier (largest on top). Height is each year's share of
# that glacier's total extra loss; color is the year's extra loss (cm w.e.) on the same magma scale.
fig, axes = plt.subplots(len(glaciers), 1, figsize=(6, 6), layout='constrained', sharex=True, sharey=True)
for ax, gi in zip(axes, order[::-1]):
    share = shares['Both'][gi]
    magnitude = share * totals['Both'][gi]
    ax.bar(np.arange(n_years), share, width=1.0, color=mpl.colormaps['magma'](magnitude / zmax_both),
           edgecolor='none')
    ax.set_title(fmt_name(glaciers[gi]), loc='left', fontsize=9)
    ax.tick_params(labelsize=9)
axes[-1].set_xticks(range(0, n_years, 4))
axes[-1].set_xticklabels(years[::4], fontsize=9)
sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, zmax_both), cmap=mpl.colormaps['magma'])
sm.set_array([])
fig.colorbar(sm, ax=axes, label='Extra mass loss (cm w.e.)')
fig.supylabel("Share of glacier's total extra loss")
plt.savefig(fig_fp + 'loss_year_bars_both.png', dpi=300, bbox_inches='tight')
plt.close()
