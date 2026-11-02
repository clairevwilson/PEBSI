"""
Figures comparing the calibrated runs forced with raw hourly MERRA-2
(AD_result_1) against the same forcing downsampled to 3-hourly and daily
and re-upsampled to hourly (shop/preprocessing/downsample_climate.py):
annual glacier-wide mass balance against Hugonnet, the change in each
mass balance component, the summer energy balance, the mean summer diurnal
cycle, and the elevation profile of the mass balance change.

Reads per_glacier.nc and per_point_annual.nc written by
shop/postprocessing/aggregate_per_glacier.py and saves PNGs into out_fp.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

sys.path.append('/jet/home/cwilson4/PEBSI/')
from project.glacierwide_loss import MassBalance

base_fp = '/ocean/projects/ees260009p/cwilson4/Output/'
out_fp = base_fp + 'forcing_resolution/'
os.makedirs(out_fp, exist_ok=True)

runs = {'Hourly': 'AD_result_1', '3-hourly': 'AD_result_3hr_0', 'Daily': 'AD_result_daily_0'}
colors = {'Hourly': 'k', '3-hourly': 'tab:blue', 'Daily': 'tab:orange'}
glac = {sim: xr.open_dataset(base_fp + f'{run}/per_glacier.nc').load() for sim, run in runs.items()}
pts = {sim: xr.open_dataset(base_fp + f'{run}/per_point_annual.nc').load() for sim, run in runs.items()}
glaciers = glac['Hourly'].glacier.values

# the run starts 2000-01-01 and ends 2025-08-31, so WY2000 and WY2025 are partial
water_years_full = slice(2001, 2024)

mb_vars = ['mass_balance', 'accumulation', 'rainfall', 'melt', 'refreeze', 'runoff']
flux_vars = ['shortwave_net', 'longwave_net', 'sensible_heat', 'latent_heat', 'rain_heat', 'ground_heat']


def fmt_name(glacier):
    return ' '.join(f.capitalize() for f in glacier.split('_'))


def water_years(da):
    annual = da.resample(time='YS-OCT').sum()
    annual = annual.assign_coords(time=annual.time.dt.year + 1).rename(time='water_year')
    return annual.sel(water_year=water_years_full)


def summer(ds):
    return ds.sel(time=ds.time.dt.month.isin([6, 7, 8]))


annual = {sim: water_years(ds[mb_vars]) for sim, ds in glac.items()}

# Glacier-wide rate over the Hugonnet period
print('Glacier-wide mass balance over the Hugonnet period [m w.e. yr-1]')
rows = []
for glacier in glaciers:
    obs = MassBalance(glacier)
    row = {'glacier': glacier, 'Hugonnet': obs.meas, 'sigma': obs.sigma}
    for sim, ds in glac.items():
        mb = ds.mass_balance.sel(glacier=glacier, time=slice(obs.matched_start, obs.matched_end))
        row[sim] = float(mb.sum()) / obs.n_years
    rows.append(row)
rates = pd.DataFrame(rows).set_index('glacier')
for sim in ['3-hourly', 'Daily']:
    rates[f'{sim} - Hourly'] = rates[sim] - rates['Hourly']
print(rates.round(3).to_string())

# Annual glacier-wide mass balance
fig, axes_2d = plt.subplots(2, 3, figsize=(7, 4), layout='constrained', sharex=True)
axes = axes_2d.flatten()
for ax, glacier in zip(axes, glaciers):
    obs = rates.loc[glacier]
    ax.axhspan(obs.Hugonnet - obs.sigma, obs.Hugonnet + obs.sigma, color='gray', alpha=0.3, linewidth=0,
               label='Hugonnet $\\pm1\\sigma$')
    for sim, ds in annual.items():
        ax.plot(ds.water_year, ds.mass_balance.sel(glacier=glacier), color=colors[sim], linewidth=1, label=sim)
    ax.axhline(0, linewidth=0.5, c='k')
    ax.set_title(fmt_name(glacier), fontsize=12)
    ax.set_xticks([2005, 2015, 2025])
    ax.tick_params(labelsize=10)
fig.supylabel('Annual MB (m w.e.)', fontsize=12)
axes[-1].axis('off')
axes[-1].legend(*axes[0].get_legend_handles_labels(), loc='center', fontsize=10, frameon=False)
fig.savefig(out_fp + 'annual_mass_balance.png', dpi=200)

# Cumulative glacier-wide mass balance difference from the hourly run
daily_mb = {sim: ds.mass_balance.resample(time='D').sum() for sim, ds in glac.items()}
fig, axes_2d = plt.subplots(2, 3, figsize=(7, 4), layout='constrained', sharex=True, sharey=True)
axes = axes_2d.flatten()
for ax, glacier in zip(axes, glaciers):
    for sim in ['3-hourly', 'Daily']:
        cumulative = (daily_mb[sim] - daily_mb['Hourly']).sel(glacier=glacier).cumsum()
        years = cumulative.time.dt.year + (cumulative.time.dt.dayofyear - 1) / 365.25
        ax.plot(years, cumulative, color=colors[sim], linewidth=1, label=sim)
    ax.axhline(0, linewidth=0.5, c='k')
    ax.set_title(fmt_name(glacier), fontsize=12)
    ax.set_xticks([2005, 2015, 2025])
    ax.tick_params(labelsize=10)
fig.supylabel('Cumulative $\\Delta$ MB (m w.e.)', fontsize=12)
axes[-1].axis('off')
axes[-1].legend(*axes[0].get_legend_handles_labels(), loc='center', fontsize=10, frameon=False)
fig.savefig(out_fp + 'cumulative_difference.png', dpi=200)

# Same for Gulkana alone
fig, ax = plt.subplots(figsize=(3, 3), layout='constrained')
for sim in ['3-hourly', 'Daily']:
    cumulative = (daily_mb[sim] - daily_mb['Hourly']).sel(glacier='gulkana').cumsum()
    years = cumulative.time.dt.year + (cumulative.time.dt.dayofyear - 1) / 365.25
    ax.plot(years, cumulative, color=colors[sim], linewidth=1, label=sim)
ax.axhline(0, linewidth=0.5, c='k')
ax.set_xticks([2005, 2015, 2025])
ax.tick_params(labelsize=10)
ax.set_ylabel('Cumulative $\\Delta$ MB (m w.e.)', fontsize=10)
ax.legend(fontsize=10, frameon=False, loc='upper left')
fig.savefig(out_fp + 'cumulative_difference_gulkana.png', dpi=200)

# Mean annual change in each mass balance component, WY2001-2024
diff = {sim: (annual[sim] - annual['Hourly']).mean('water_year') for sim in ['3-hourly', 'Daily']}
print('\nMean annual change from hourly, WY2001-2024 [m w.e. yr-1]')
for sim, d in diff.items():
    print(f'  {sim}'); print(d.to_dataframe()[mb_vars].round(3).to_string())

fig, axes = plt.subplots(1, len(glaciers), figsize=(11, 3), layout='constrained', sharey=True)
x = np.arange(len(mb_vars))
for ax, glacier in zip(axes, glaciers):
    for k, (sim, d) in enumerate(diff.items()):
        ax.bar(x + (k - 0.5) * 0.4, [float(d[v].sel(glacier=glacier)) for v in mb_vars], width=0.4,
               color=colors[sim], label=sim)
    ax.axhline(0, linewidth=0.5, c='k')
    ax.set_xticks(x, [v.replace('_', ' ') for v in mb_vars], rotation=60, fontsize=8)
    ax.set_title(fmt_name(glacier), fontsize=10)
axes[0].set_ylabel('Change from hourly (m w.e. yr$^{-1}$)')
axes[0].legend(fontsize=8, frameon=False)
fig.savefig(out_fp + 'mass_balance_components.png', dpi=200)

# Summer (JJA) energy balance change
jja = {sim: summer(ds[flux_vars + ['airtemp', 'melt_energy']]).mean('time') for sim, ds in glac.items()}
print('\nJJA mean, hourly run, and change from hourly [W m-2, C]')
table = jja['Hourly'].to_dataframe()[flux_vars + ['melt_energy', 'airtemp']]
print(table.round(2).to_string())
for sim in ['3-hourly', 'Daily']:
    print(f'  {sim} - Hourly')
    print((jja[sim] - jja['Hourly']).to_dataframe()[flux_vars + ['melt_energy', 'airtemp']].round(2).to_string())

fig, axes = plt.subplots(1, len(glaciers), figsize=(11, 3), layout='constrained', sharey=True)
eb_vars = flux_vars + ['melt_energy']
x = np.arange(len(eb_vars))
for ax, glacier in zip(axes, glaciers):
    for k, sim in enumerate(['3-hourly', 'Daily']):
        d = jja[sim] - jja['Hourly']
        ax.bar(x + (k - 0.5) * 0.4, [float(d[v].sel(glacier=glacier)) for v in eb_vars], width=0.4,
               color=colors[sim], label=sim)
    ax.axhline(0, linewidth=0.5, c='k')
    ax.set_xticks(x, [v.replace('_', ' ') for v in eb_vars], rotation=60, fontsize=8)
    ax.set_title(fmt_name(glacier), fontsize=10)
axes[0].set_ylabel('JJA change from hourly (W m$^{-2}$)')
axes[0].legend(fontsize=8, frameon=False)
fig.savefig(out_fp + 'summer_energy_balance.png', dpi=200)

# Mean JJA diurnal cycle (local AKST hour) of air temperature, melt energy and net shortwave
diurnal_vars = ['airtemp', 'shortwave_net', 'melt_energy']
fig, axes = plt.subplots(len(diurnal_vars), len(glaciers), figsize=(11, 6), layout='constrained', sharex=True,
                         sharey='row')
for sim, ds in glac.items():
    s = summer(ds[diurnal_vars])
    cycle = s.groupby((s.time.dt.hour - 9) % 24).mean()
    for i, var in enumerate(diurnal_vars):
        for j, glacier in enumerate(glaciers):
            axes[i, j].plot(cycle.hour, cycle[var].sel(glacier=glacier), color=colors[sim], label=sim)
for j, glacier in enumerate(glaciers):
    axes[0, j].set_title(fmt_name(glacier), fontsize=10)
    axes[-1, j].set_xlabel('Local hour (AKST)')
for i, var in enumerate(diurnal_vars):
    axes[i, 0].set_ylabel(var.replace('_', ' '))
axes[0, 0].legend(fontsize=8, frameon=False)
fig.savefig(out_fp + 'summer_diurnal_cycle.png', dpi=200)

# Elevation profile of the mean annual point mass balance change
fig, axes = plt.subplots(1, len(glaciers), figsize=(11, 3.5), layout='constrained', sharex=True)
base = pts['Hourly'].mass_balance.sel(water_year=water_years_full).mean('water_year')
rgiid = pts['Hourly'].rgiid.values
for ax, glacier, gid in zip(axes, glaciers, glac['Hourly'].rgiid.values):
    mask = rgiid == gid
    elev = pts['Hourly'].elev.values[mask]
    order = np.argsort(elev)
    for sim in ['3-hourly', 'Daily']:
        d = (pts[sim].mass_balance.sel(water_year=water_years_full).mean('water_year') - base).values[mask]
        ax.scatter(d[order], elev[order], s=3, color=colors[sim], label=sim)
    ax.axvline(0, linewidth=0.5, c='k')
    ax.set_title(fmt_name(glacier), fontsize=10)
    ax.set_xlabel('$\\Delta$ MB (m w.e. yr$^{-1}$)')
axes[0].set_ylabel('Elevation (m)')
axes[0].legend(fontsize=8, frameon=False, markerscale=3)
fig.savefig(out_fp + 'elevation_profile.png', dpi=200)

print(f'\nSaved figures to {out_fp}')
