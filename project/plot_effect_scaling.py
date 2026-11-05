"""
How the mass loss an event causes scales with its magnitude, for heatwaves and wildfires, in the
synthetic experiments (top) and the controlled experiment (bottom). Each panel has a linear fit
through the origin on linear axes, so points that fall above the line at small magnitudes and
below it at large ones are a response that tapers off.

Top: synthetic Kennicott 2006 events (synthetic_heatwave_results.csv, synthetic_wildfire_results.csv),
effect on May-Sep mass balance against event magnitude: the temperature anomaly added (K day,
intensity x duration x the average heatwave's daily mean T2M anomaly) and the fire-attributable
BC+OC deposition added (mg m-2). The scatter from event timing (and for heatwaves, precipitation
removed) is regressed out: heatwaves additively, wildfires multiplicatively in log loss against
log deposition, since a fire's effect is set by how much ice its albedo change can act on.
Wildfire points are the loss of the same event at the mean midpoint date.

Bottom: controlled runs 1997-2015 at the five calibration glaciers, one point per glacier-season.
Effects are Base minus No heatwaves or No fires, summed May-Sep (May-Aug in 2015, when the runs
end). Magnitude at the cell nearest each glacier centroid: the T2M removed by the no-heatwave
forcing (K day) and the GFED minus no-fires BC+OC deposition (mg m-2), both over the same season.

A second version colors each point by its timing: the midpoint date of a synthetic event, and for
a controlled season the mean day of year weighted by the K day removed or the deposition.
"""
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

OUT = '/ocean/projects/ees260009p/cwilson4/Output/'
CLIMATE = '/ocean/projects/ees260009p/cwilson4/climate_data/'
HW_FN = OUT + 'synthetic_heatwaves/synthetic_heatwave_results.csv'
HW_ANOMALY_FN = OUT + 'synthetic_heatwaves/synthetic_heatwave_anomaly.csv'
WF_FN = OUT + 'synthetic_wildfires/synthetic_wildfire_results.csv'
CONTROLLED_FP = OUT + 'controlled/{run}_0/per_glacier.nc'
EVENTS_FN = CLIMATE + 'extreme_events_daily_ukesm.nc'
T2M_FN = CLIMATE + 'MERRA2/reg01/T2M_reg01.zarr'
T2M_NOHW_FN = CLIMATE + 'MERRA2/reg01_no_heatwaves/T2M_reg01_no_heatwaves.zarr'
PRECIP_FN = CLIMATE + 'MERRA2/reg01/PRECTOTCORR_reg01.zarr'
FIG_FN = OUT + 'controlled/effect_scaling.png'
TIMING_FIG_FN = OUT + 'controlled/effect_scaling_timing.png'

KENNICOTT = (61.5, -143.125)
UTC_OFFSET = pd.Timedelta(hours=8)
YEARS = range(1997, 2016)
TIMING = 'mid + I(mid**2) + I(mid**3)'

HW_COLOR = 'b'
WF_COLOR = 'r'
INK = 'k'
GRID = '#e6e5e1'
TIMING_CMAP = 'viridis'


def load_synthetic(fn):
    events = pd.read_csv(fn, index_col='run', parse_dates=['start']).drop(0)
    # event midpoint in months from late June
    events['doy'] = events['start'].dt.dayofyear + events['duration'] / 2
    events['mid'] = (events['doy'] - 180) / 30
    events['loss'] = -events['effect'] * 1000
    return events


def kennicott_daily_precip():
    """Daily precipitation [mm] at the Kennicott cell, local days, in the original forcing."""
    p = xr.open_zarr(PRECIP_FN)['PRECTOTCORR'].sel(lat=KENNICOTT[0], lon=KENNICOTT[1], method='nearest')
    p = p.sel(time=slice('2006-04-01', '2006-10-31')).to_series()
    p.index = p.index - UTC_OFFSET
    return p.resample('1D').sum() * 3600


def season(year):
    return slice(f'{year}-05-01', f'{year}-08-31' if year == 2015 else f'{year}-09-30')


def weighted_doy(da):
    """Mean day of year weighted by the positive part of da; NaN if nothing is positive."""
    weight = da.clip(min=0)
    return float((weight * da.time.dt.dayofyear).sum() / weight.sum())


def controlled_seasons():
    """Heatwave and wildfire effect [mm w.e.], magnitude and timing for every glacier-season."""
    mb = {run: xr.open_dataset(CONTROLLED_FP.format(run=run))['mass_balance'].load()
          for run in ['base', 'no_heatwaves', 'no_fires']}
    lat, lon = mb['base'].lat, mb['base'].lon
    ev = xr.open_dataset(EVENTS_FN).sel(y=lat, x=lon, method='nearest').load()
    fire_dep = (ev['bc'] - ev['bc_nofire']) * 1e6
    t2m = xr.open_zarr(T2M_FN)['T2M'].sel(time=slice('1997', '2015'))
    t2m_nohw = xr.open_zarr(T2M_NOHW_FN)['T2M'].sel(time=slice('1997', '2015'))
    removed = (t2m.sel(lat=lat, lon=lon, method='nearest')
               - t2m_nohw.sel(lat=lat, lon=lon, method='nearest')).load() / 24

    rows = []
    for g in mb['base'].glacier.values:
        for year in YEARS:
            s = season(year)
            base = mb['base'].sel(glacier=g, time=s)
            rows.append({
                'glacier': g,
                'year': year,
                'hw_loss': -float((base - mb['no_heatwaves'].sel(glacier=g, time=s)).sum()) * 1000,
                'wf_loss': -float((base - mb['no_fires'].sel(glacier=g, time=s)).sum()) * 1000,
                'hw_mag': float(removed.sel(glacier=g, time=s).sum()),
                'wf_mag': float(fire_dep.sel(glacier=g, time=s).sum()),
                'hw_doy': weighted_doy(removed.sel(glacier=g, time=s)),
                'wf_doy': weighted_doy(fire_dep.sel(glacier=g, time=s)),
            })
    return pd.DataFrame(rows)


def slope_through_origin(x, y):
    return float((x * y).sum() / (x * x).sum())


def power_law(x, y):
    """Scale and exponent of y = a x^b, fit in log space over positive points."""
    keep = (x > 0) & (y > 0)
    b, log_a = np.polyfit(np.log(x[keep]), np.log(y[keep]), 1)
    return np.exp(log_a), b


def plot_panel(ax, title, x, y, doy, color, xlabel):
    """Scatter with linear and power-law fits; points colored by doy when given, else by color."""
    slope = slope_through_origin(x, y)
    scale, b = power_law(x, y)
    print(f'{title}: linear fit {slope:.3f} mm w.e. per unit, power law exponent {b:.2f}')
    if doy is None:
        ax.scatter(x, y, color=color, s=8, edgecolor='white', linewidth=0.3)
    else:
        ax.scatter(x, y, c=doy, cmap=TIMING_CMAP, norm=timing_norm, s=8, edgecolor='white', linewidth=0.3)
    grid = np.linspace(0, x.max() * 1.05, 200)
    ax.plot(grid, slope * grid, color=INK, linewidth=1, label='Linear')
    ax.plot(grid, scale * grid ** b, color=INK, linewidth=1, linestyle='--',
            label=f'Power law (exponent {b:.2f})')
    ax.legend(frameon=False, fontsize=6, loc='upper left')
    ax.set_xlim(0, x.max() * 1.05)
    ax.set_ylim(0, y.max() * 1.1)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=6)
    ax.xaxis.label.set_size(7)
    ax.yaxis.label.set_size(7)


# synthetic heatwaves: loss = timing + b * K day + c * precipitation removed
hw = load_synthetic(HW_FN)
hw_daily_anomaly = pd.read_csv(HW_ANOMALY_FN)['T2M'].mean()
hw['mag'] = hw['intensity'] * hw['duration'] * hw_daily_anomaly
precip = kennicott_daily_precip()
hw['pr'] = [precip.loc[s:s + pd.Timedelta(days=d - 1)].sum() for s, d in zip(hw['start'], hw['duration'])]
hw_fit = smf.ols(f'loss ~ {TIMING} + mag + pr', hw).fit()
hw['partial'] = hw_fit.resid + hw_fit.params['mag'] * hw['mag']

# synthetic wildfires: log loss = timing + b * log deposition, shifted to the mean midpoint date
wf = load_synthetic(WF_FN)
wf['mag'] = wf['total_deposition']
wf_fit = smf.ols(f'np.log(loss) ~ {TIMING} + np.log(mag)', wf).fit()
at_mean_timing = wf.assign(mid=wf['mid'].mean())
wf['partial'] = np.exp(wf_fit.predict(at_mean_timing) + wf_fit.resid)

seasons = controlled_seasons()
c_hw = seasons[seasons['hw_mag'] > 0]

# title, magnitude, loss, timing, color, x label
panels = [
    ('Synthetic heatwaves', hw['mag'], hw['partial'], hw['doy'], HW_COLOR, 'Temperature anomaly (K day)'),
    ('Synthetic wildfires', wf['mag'], wf['partial'], wf['doy'], WF_COLOR, 'BC+OC deposition (mg m$^{-2}$)'),
    ('Controlled heatwaves', c_hw['hw_mag'], c_hw['hw_loss'], c_hw['hw_doy'], HW_COLOR,
     'Temperature anomaly (K day)'),
    ('Controlled wildfires', seasons['wf_mag'], seasons['wf_loss'], seasons['wf_doy'], WF_COLOR,
     'BC+OC deposition (mg m$^{-2}$)'),
]
months = pd.date_range('2006-05-01', '2006-09-01', freq='MS')
timing_norm = plt.Normalize(months.dayofyear[0], pd.Timestamp('2006-09-30').dayofyear)

for fig_fn, by_timing in [(FIG_FN, False), (TIMING_FIG_FN, True)]:
    fig, axes = plt.subplots(2, 2, figsize=(5, 5), layout='constrained')
    for ax, (title, x, y, doy, color, xlabel) in zip(axes.flat, panels):
        plot_panel(ax, title, x, y, doy if by_timing else None, color, xlabel)
    for ax in axes[:, 0]:
        ax.set_ylabel('Additional mass loss (mm w.e.)')
    if by_timing:
        sm = plt.cm.ScalarMappable(norm=timing_norm, cmap=TIMING_CMAP)
        cbar = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.02)
        cbar.set_ticks(months.dayofyear, labels=months.strftime('%b'))
        cbar.set_label('Event timing', fontsize=7)
        cbar.ax.tick_params(labelsize=6)
    plt.savefig(fig_fn, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {fig_fn}')
