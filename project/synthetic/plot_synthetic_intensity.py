"""
How the effect of a synthetic event on Kennicott's May-Sep 2006 mass balance scales with its
intensity, from synthetic_heatwave_results.csv and synthetic_wildfire_results.csv.

Top row: the effect per event day against intensity, colored by the covariate that scatters it
most (precipitation removed for heatwaves, start date for wildfires). Bottom row: partial
residuals, i.e. the effect with timing, duration (and for heatwaves, removed precipitation)
regressed out, so only the intensity dependence is left. Heatwaves are fit linearly per event
day; wildfires as a power law, where a slope of 1 on log axes would be linear.
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

HW_DIR = '/ocean/projects/ees260009p/cwilson4/Output/synthetic_heatwaves/'
WF_DIR = '/ocean/projects/ees260009p/cwilson4/Output/synthetic_wildfires/'
PRECIP_FN = '/ocean/projects/ees260009p/cwilson4/climate_data/MERRA2/reg01/PRECTOTCORR_reg01.zarr'
FIG_FN = os.path.join(HW_DIR, 'synthetic_intensity_scaling.png')
LAT, LON = 61.5, -143.125
UTC_OFFSET = pd.Timedelta(hours=8)

INK = 'k'
GRID = '#e6e5e1'
TIMING = 'mid + I(mid**2) + I(mid**3)'


def load(fn):
    events = pd.read_csv(fn, index_col='run', parse_dates=['start']).drop(0)
    # event midpoint in months from late June
    events['mid'] = (events['start'].dt.dayofyear + events['duration'] / 2 - 180) / 30
    events['k'] = events['intensity']
    return events


def daily_precip():
    """Daily precipitation [mm] at the Kennicott cell, local days, in the original forcing."""
    p = xr.open_zarr(PRECIP_FN)['PRECTOTCORR'].sel(lat=LAT, lon=LON, method='nearest')
    p = p.sel(time=slice('2006-04-01', '2006-10-31')).to_series()
    p.index = p.index - UTC_OFFSET
    return p.resample('1D').sum() * 3600


hw = load(os.path.join(HW_DIR, 'synthetic_heatwave_results.csv'))
precip = daily_precip()
hw['precip_removed'] = [precip.loc[s:s + pd.Timedelta(days=d - 1)].sum()
                        for s, d in zip(hw['start'], hw['duration'])]
hw['pr'] = hw['precip_removed'] / hw['duration']
hw['per_day'] = hw['effect'] / hw['duration'] * 1000  # mm w.e. per event day
hw_fit = smf.ols(f'per_day ~ {TIMING} + duration + pr + k', hw).fit()
hw_quad = smf.ols(f'per_day ~ {TIMING} + duration + pr + k + I(k**2)', hw).fit()
hw['partial'] = hw_fit.resid + hw_fit.params['k'] * hw['k']

wf = load(os.path.join(WF_DIR, 'synthetic_wildfire_results.csv'))
wf['per_day'] = wf['effect'] / wf['duration'] * 1000
wf['log_loss'] = np.log(-wf['effect'])
wf_fit = smf.ols(f'log_loss ~ {TIMING} + np.log(duration) + np.log(k)', wf).fit()
wf_slope = wf_fit.params['np.log(k)']
wf['partial'] = np.exp(wf_fit.resid + wf_slope * np.log(wf['k']))
wf['partial'] /= np.exp(np.log(wf['partial']).mean())

fig, axes = plt.subplots(2, 2, figsize=(9, 7), layout='constrained')
dot = dict(s=28, edgecolor='white', linewidth=0.6)

ax = axes[0, 0]
sc = ax.scatter(hw['k'], hw['per_day'], c=hw['pr'], cmap='viridis', **dot)
fig.colorbar(sc, ax=ax, pad=0.02).set_label('Precip. removed (mm day$^{-1}$)', fontsize=8)
ax.set_ylabel('Effect per event day (mm w.e. day$^{-1}$)')
ax.set_title('Heatwaves', loc='left', fontsize=10)

ax = axes[1, 0]
ax.scatter(hw['k'], hw['partial'], color='#3b6ea5', **dot)
k = np.linspace(0.5, 2, 50)
ax.plot(k, hw_fit.params['k'] * k, color=INK, linewidth=1.5,
        label=f"linear fit: {hw_fit.params['k']:.1f} mm w.e. day$^{{-1}}$ per unit\n"
              f"(quadratic term p = {hw_quad.pvalues['I(k ** 2)']:.2f})")
ax.legend(frameon=False, fontsize=8)
ax.set_ylabel('Partial effect per event day (mm w.e. day$^{-1}$)\n'
              'timing, duration, precip. removed')
ax.set_xlabel('Intensity (× average heatwave)')

ax = axes[0, 1]
sc = ax.scatter(wf['k'], -wf['per_day'], c=wf['start'].dt.dayofyear, cmap='viridis', **dot)
cbar = fig.colorbar(sc, ax=ax, pad=0.02)
months = pd.date_range('2006-05-01', '2006-09-01', freq='MS')
cbar.set_ticks(months.dayofyear, labels=months.strftime('%b'))
cbar.set_label('Event start', fontsize=8)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel('Mass loss per event day (mm w.e. day$^{-1}$)')
ax.set_title('Wildfires', loc='left', fontsize=10)

ax = axes[1, 1]
ax.scatter(wf['k'], wf['partial'], color='#c0603a', **dot)
k = np.geomspace(0.5, 10, 50)
centre = np.exp(np.log(wf['k']).mean())
ax.plot(k, (k / centre) ** wf_slope, color=INK, linewidth=1.5,
        label=f"fit: slope {wf_slope:.2f} ± {wf_fit.bse['np.log(k)']:.2f}")
ax.plot(k, (k / centre) ** 1, color=INK, linewidth=1, linestyle='--', label='linear: slope 1')
ax.legend(frameon=False, fontsize=8)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel('Partial mass loss (relative)\ntiming, duration removed')
ax.set_xlabel('Intensity (× average wildfire)')

for ax in axes[:, 1]:
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(mpl.ticker.FormatStrFormatter('%g'))
        axis.set_minor_formatter(mpl.ticker.NullFormatter())
for ax in axes.flat:
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(colors=INK, labelsize=8)
    ax.yaxis.label.set_size(9)
    ax.xaxis.label.set_size(9)

plt.savefig(FIG_FN, dpi=200, bbox_inches='tight')
print(f"Saved {FIG_FN}")
