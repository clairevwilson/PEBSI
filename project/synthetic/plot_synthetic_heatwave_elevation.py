"""
Where on Kennicott the synthetic heatwaves act, from synthetic_heatwave_point_mb.csv
(analyze_synthetic_heatwaves.py). At each point the change in May-Sep 2006 mass balance per
event day is regressed on event timing, duration, the precipitation the event removed and its
intensity, as in plot_synthetic_intensity.py. The panels show the coefficients on removed
precipitation and on intensity against elevation: the first is near zero where summer
precipitation falls as rain and large where it falls as snow.
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

OUT_DIR = '/ocean/projects/ees260009p/cwilson4/Output/synthetic_heatwaves/'
RESULTS_FN = os.path.join(OUT_DIR, 'synthetic_heatwave_results.csv')
POINT_FN = os.path.join(OUT_DIR, 'synthetic_heatwave_point_mb.csv')
PRECIP_FN = '/ocean/projects/ees260009p/cwilson4/climate_data/MERRA2/reg01/PRECTOTCORR_reg01.zarr'
FIG_FN = os.path.join(OUT_DIR, 'synthetic_heatwave_elevation.png')
LAT, LON = 61.5, -143.125
UTC_OFFSET = pd.Timedelta(hours=8)
BAND = 250  # m

INK = 'k'
GRID = '#e6e5e1'


def daily_precip():
    """Daily precipitation [mm] at the Kennicott cell, local days, in the original forcing."""
    p = xr.open_zarr(PRECIP_FN)['PRECTOTCORR'].sel(lat=LAT, lon=LON, method='nearest')
    p = p.sel(time=slice('2006-04-01', '2006-10-31')).to_series()
    p.index = p.index - UTC_OFFSET
    return p.resample('1D').sum() * 3600


events = pd.read_csv(RESULTS_FN, index_col='run', parse_dates=['start']).drop(0)
points = pd.read_csv(POINT_FN, index_col='point')
precip = daily_precip()

duration = events['duration'].values
pr = np.array([precip.loc[s:s + pd.Timedelta(days=d - 1)].sum()
               for s, d in zip(events['start'], duration)]) / duration
mid = (events['start'].dt.dayofyear.values + duration / 2 - 180) / 30
X = np.column_stack([np.ones(len(events)), mid, mid**2, mid**3, duration, pr,
                     events['intensity'].values])

runs = [f'run_{run:03d}' for run in events.index]
# mm w.e. per event day, (run, point)
per_day = (points[runs].T.values - points['run_000'].values) / duration[:, None] * 1000
beta = np.linalg.lstsq(X, per_day, rcond=None)[0]
points = pd.DataFrame({'elev': points['elev'], 'weight': points['weight'],
                       'precip': beta[5], 'intensity': beta[6],
                       'band': (points['elev'] // BAND) * BAND + BAND / 2})
bands = points.groupby('band').apply(
    lambda d: pd.Series({c: np.average(d[c], weights=d['weight']) for c in ['precip', 'intensity']}))

fig, axes = plt.subplots(1, 2, figsize=(8, 4.5), sharey=True, layout='constrained')
panels = [
    ('precip', 'Per mm day$^{-1}$ of precipitation removed\n(mm w.e. day$^{-1}$)', '#3b6ea5'),
    ('intensity', 'Per unit of heatwave intensity\n(mm w.e. day$^{-1}$)', '#c0603a'),
]
for ax, (col, label, color) in zip(axes, panels):
    ax.scatter(points[col], points['elev'], s=6, color=color, alpha=0.3, linewidth=0)
    ax.plot(bands[col], bands.index, color=color, linewidth=2, marker='o', markersize=4,
            label=f'area-weighted mean, {BAND} m bands')
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_xlabel(label, fontsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(colors=INK, labelsize=8)
axes[0].set_ylabel('Elevation (m)', fontsize=9)
axes[0].legend(frameon=False, fontsize=8, loc='lower left')
axes[0].set_title('Effect of removed precipitation', loc='left', fontsize=10)
axes[1].set_title('Effect of added heat', loc='left', fontsize=10)

plt.savefig(FIG_FN, dpi=200, bbox_inches='tight')
print(f"Saved {FIG_FN}")
