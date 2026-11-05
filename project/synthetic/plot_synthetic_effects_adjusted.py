"""
The synthetic heatwave and wildfire summaries (plot_synthetic_heatwaves.py, plot_synthetic_wildfires.py)
with each event colored by its adjusted rather than raw change in Kennicott's May-Sep 2006 mass balance,
the same adjusted values as the top row of project/plot_effect_scaling.py.

Heatwaves: loss is regressed on a cubic in the event's midpoint date, the K day added and the
precipitation removed; the color is the loss with the fitted date and precipitation contributions
subtracted, so every event is shown at the same date with no precipitation removed.
Wildfires: log loss is regressed on the same cubic and log deposition; the color is the loss scaled
to the mean midpoint date.
A third figure colors the heatwaves with only the precipitation contribution subtracted, keeping the
effect of timing, so every event is shown with no precipitation removed at its own date.
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects  # noqa: F401 -- registers mpl.patheffects
import statsmodels.formula.api as smf

OUT = '/ocean/projects/ees260009p/cwilson4/Output/'
HW_FN = OUT + 'synthetic_heatwaves/synthetic_heatwave_results.csv'
HW_ANOMALY_FN = OUT + 'synthetic_heatwaves/synthetic_heatwave_anomaly.csv'
WF_FN = OUT + 'synthetic_wildfires/synthetic_wildfire_results.csv'
PRECIP_FN = '/ocean/projects/ees260009p/cwilson4/climate_data/MERRA2/reg01/PRECTOTCORR_reg01.zarr'
HW_FIG_FN = OUT + 'synthetic_heatwaves/synthetic_heatwave_effects_adjusted.png'
WF_FIG_FN = OUT + 'synthetic_wildfires/synthetic_wildfire_effects_adjusted.png'
HW_PRECIP_FIG_FN = OUT + 'synthetic_heatwaves/synthetic_heatwave_effects_precip_adjusted.png'

KENNICOTT = (61.5, -143.125)
UTC_OFFSET = pd.Timedelta(hours=8)
TIMING = 'mid + I(mid**2) + I(mid**3)'

INK = 'k'
GRID = '#e6e5e1'


def load_synthetic(fn):
    events = pd.read_csv(fn, index_col='run', parse_dates=['start']).drop(0)
    events['end'] = events['start'] + pd.to_timedelta(events['duration'], unit='D')
    # event midpoint in months from late June
    events['mid'] = (events['start'].dt.dayofyear + events['duration'] / 2 - 180) / 30
    events['loss'] = -events['effect'] * 1000
    return events


def kennicott_daily_precip():
    """Daily precipitation [mm] at the Kennicott cell, local days, in the original forcing."""
    p = xr.open_zarr(PRECIP_FN)['PRECTOTCORR'].sel(lat=KENNICOTT[0], lon=KENNICOTT[1], method='nearest')
    p = p.sel(time=slice('2006-04-01', '2006-10-31')).to_series()
    p.index = p.index - UTC_OFFSET
    return p.resample('1D').sum() * 3600


def plot_events(events, y, loss_col, fig_fn, ylabel, cbar_label, log_y=False):
    """Each event a line over its dates at height y, colored by the change in mass balance from loss_col."""
    change = -events[loss_col] / 1000
    cmap = plt.get_cmap('viridis')
    norm = mpl.colors.Normalize(vmin=np.floor(change.min() * 50) / 50, vmax=0)
    # white outline keeps nearby events apart; larger effects drawn on top
    outline = [mpl.patheffects.Stroke(linewidth=3.4, foreground='white'), mpl.patheffects.Normal()]
    fig, ax = plt.subplots(figsize=(7, 4))
    for run in change.sort_values(ascending=False).index:
        ev = events.loc[run]
        ax.plot([ev['start'], ev['end']], [ev[y]] * 2, color=cmap(norm(change[run])),
                linewidth=2.2, solid_capstyle='butt', path_effects=outline)

    months = pd.date_range('2006-05-01', '2006-09-01', freq='MS')
    ax.set_xticks(months)
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%b'))
    ax.set_xlim(pd.Timestamp('2006-04-28'), pd.Timestamp('2006-09-24'))
    if log_y:
        ax.set_yscale('log')
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9)
    ax.set_ylabel(ylabel, color=INK, fontsize=9)
    ax.set_xlabel('Event dates, 2006', color=INK, fontsize=9)

    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(cbar_label, color=INK, fontsize=9)
    cbar.ax.tick_params(colors=INK, labelsize=8)
    plt.savefig(fig_fn, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {fig_fn}')


hw = load_synthetic(HW_FN)
hw['mag'] = hw['intensity'] * hw['duration'] * pd.read_csv(HW_ANOMALY_FN)['T2M'].mean()
precip = kennicott_daily_precip()
hw['pr'] = [precip.loc[s:s + pd.Timedelta(days=d - 1)].sum() for s, d in zip(hw['start'], hw['duration'])]
hw_fit = smf.ols(f'loss ~ {TIMING} + mag + pr', hw).fit()
hw['adjusted'] = hw_fit.resid + hw_fit.params['mag'] * hw['mag']
hw['precip_adjusted'] = hw['loss'] - hw_fit.params['pr'] * hw['pr']

wf = load_synthetic(WF_FN)
wf['mag'] = wf['total_deposition']
wf_fit = smf.ols(f'np.log(loss) ~ {TIMING} + np.log(mag)', wf).fit()
wf['adjusted'] = np.exp(wf_fit.predict(wf.assign(mid=wf['mid'].mean())) + wf_fit.resid)

plot_events(hw, 'intensity', 'adjusted', HW_FIG_FN, 'Intensity (× average heatwave)',
            'Change in mass balance (m w.e.)\nsame date, no precip. removed')
plot_events(wf, 'total_deposition', 'adjusted', WF_FIG_FN, 'Fire BC+OC deposition (mg m$^{-2}$)',
            'Change in mass balance (m w.e.)\nat the mean event date', log_y=True)
plot_events(hw, 'intensity', 'precip_adjusted', HW_PRECIP_FIG_FN, 'Intensity (× average heatwave)',
            'Change in mass balance (m w.e.)\nno precip. removed')
