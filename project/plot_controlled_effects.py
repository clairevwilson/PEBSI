"""
Figures comparing the controlled wildfire/heatwave experiments (base,
no_fires, no_heatwaves, no_either): cumulative divergence, synergy,
elevation profiles, energy-balance attribution, seasonal timing,
the effects as a share of observed loss, and a single-summer case study.

Reads per_glacier.nc and per_point_annual.nc written by
shop/postprocessing/aggregate_per_glacier.py and saves PNGs into base_fp.
"""
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

sys.path.append('/jet/home/cwilson4/PEBSI/')
from glacierwide_loss import translate_rgi

base_fp = '/ocean/projects/ees260009p/cwilson4/Output/controlled/'

# Inputs for the figures below. The effect of a forcing is Base minus the run with
# that forcing removed, so a negative effect means the forcing caused extra loss.
i = 0
runs = {'Base': 'base', 'No fires': 'no_fires', 'No heatwaves': 'no_heatwaves', 'Neither': 'no_either'}
glac = {sim: xr.open_dataset(base_fp + f'{run}_{i}/per_glacier.nc').load() for sim, run in runs.items()}
pts = {sim: xr.open_dataset(base_fp + f'{run}_{i}/per_point_annual.nc').load() for sim, run in runs.items()}
glaciers = glac['Base'].glacier.values

effects = {'Wildfires': 'No fires', 'Heatwaves': 'No heatwaves', 'Both': 'Neither'}
effect_colors = {'Wildfires': 'r', 'Heatwaves': 'b', 'Both': 'gray', 'Interaction': 'k'}

# the run starts 1997-04-01, so WY1997 is missing its winter
first_full_wy = 1998

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
plt.savefig(base_fp + 'cumulative_divergence.png', dpi=300, bbox_inches='tight')
plt.close()

# Synergy: does removing both forcings do more than the sum of removing each one?
names = ['Wildfires', 'Heatwaves', 'Both', 'Interaction']
width = 0.2

def effect_bars(values, ylabel, fn):
    fig, ax = plt.subplots(figsize=(7, 3.5), layout='constrained')
    for j, glacier in enumerate(glaciers):
        for k, name in enumerate(names):
            ax.bar(j + (k - 1.5) * width, values[glacier][name], width, facecolor=effect_colors[name],
                   edgecolor='k', linewidth=0.5, label=name if j == 0 else None)
    ax.set_xticks(range(len(glaciers)))
    ax.set_xticklabels([fmt_name(g) for g in glaciers])
    ax.axhline(0, linewidth=0.5, c='k')
    ax.set_ylabel(ylabel)
    fig.legend(loc='outside upper center', ncol=len(names), fontsize=8, frameon=False)
    plt.savefig(base_fp + fn, dpi=300, bbox_inches='tight')
    plt.close()

rates = {glacier: {name: float(water_years(effect(glac, name).sel(glacier=glacier)).mean()) * 100 for name in names}
         for glacier in glaciers}
for glacier, r in rates.items():
    print(f'{glacier:<12} fires {r["Wildfires"]:.2f}, heatwaves {r["Heatwaves"]:.2f}, '
          f'both {r["Both"]:.2f}, interaction {r["Interaction"]:.2f} cm w.e. a-1 '
          f'({r["Interaction"] / r["Both"] * 100:.0f}% of both)')
effect_bars(rates, r'Change in mass balance (cm w.e. a$^{-1}$)', 'synergy.png')

# Share of observed mass loss over 2000-2014, the calendar years both the run and Hugonnet
# et al. (2021) cover. Hugonnet's cumulative dh is converted with their 850 kg m-3 density.
obs_start, obs_end = pd.Timestamp('2000-01-01'), pd.Timestamp('2015-01-01')
n_years = obs_end.year - obs_start.year
cumul = pd.read_csv(base_fp + '../../data/mass_balance/dh_01_rgi60_pergla_cumul.csv',
                    usecols=['rgiid', 'time', 'dh'], parse_dates=['time']).set_index(['rgiid', 'time'])['dh']

shares = {}
for glacier in glaciers:
    dh = cumul['RGI60-' + translate_rgi[glacier]['6']]
    observed = (dh[obs_end] - dh[obs_start]) * 0.85 / n_years
    shares[glacier] = {}
    for name in names:
        modeled = float(effect(glac, name).sel(glacier=glacier, time=slice(obs_start, obs_end - pd.Timedelta('1s'))).sum()) / n_years
        shares[glacier][name] = modeled / observed * 100
    print(f'{glacier:<12} observed {observed:.2f} m w.e. a-1, both {shares[glacier]["Both"]:.1f}% of observed loss')
effect_bars(shares, 'Share of observed mass loss (%)', 'familiar_units.png')

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
plt.savefig(base_fp + 'elevation_profile.png', dpi=300, bbox_inches='tight')
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
plt.savefig(base_fp + 'energy_balance_attribution.png', dpi=300, bbox_inches='tight')
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
plt.savefig(base_fp + 'seasonal_timing.png', dpi=300, bbox_inches='tight')
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
plt.savefig(base_fp + f'case_study_{case_glacier}_{case_year}.png', dpi=300, bbox_inches='tight')
plt.close()
