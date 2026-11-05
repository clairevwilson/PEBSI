"""
Loss against event size for the fixed-timing synthetic events (analyze_fixed_events.py), with no
adjustment: at each start date every run has the same dates and duration and differs only in
intensity, so each date's points show directly how loss scales with size. Heatwave size is the
temperature anomaly added (K day, intensity x duration x the average heatwave's daily mean T2M
anomaly) and wildfire size the fire-attributable BC+OC deposition added (mg m-2). Heatwave loss is
the loss from the warming alone: each run minus the zero-intensity run at its start date, which
removes the same precipitation without warming. At each date the
points get a linear fit through the origin (dashed) and a power law (solid), colored by date on the
same scale as effect_scaling_timing.png.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = '/ocean/projects/ees260009p/cwilson4/Output/'
HW_FN = OUT + 'fixed_heatwaves/fixed_heatwave_results.csv'
WF_FN = OUT + 'fixed_wildfires/fixed_wildfire_results.csv'
HW_ANOMALY_FN = OUT + 'synthetic_heatwaves/synthetic_heatwave_anomaly.csv'
FIG_FN = OUT + 'fixed_heatwaves/fixed_event_scaling.png'

INK = 'k'
GRID = '#e6e5e1'
TIMING_CMAP = plt.get_cmap('viridis')
TIMING_NORM = plt.Normalize(pd.Timestamp('2006-05-01').dayofyear, pd.Timestamp('2006-09-30').dayofyear)


def load(fn):
    events = pd.read_csv(fn, index_col='run', parse_dates=['start']).drop(0)
    events['loss'] = -events['effect'] * 1000
    return events


def power_law(x, y):
    """Scale and exponent of y = a x^b, fit in log space over positive points."""
    keep = (x > 0) & (y > 0)
    b, log_a = np.polyfit(np.log(x[keep]), np.log(y[keep]), 1)
    return np.exp(log_a), b


hw = load(HW_FN)
precip_only = hw[hw['intensity'] == 0].set_index('start')['loss']
hw = hw[hw['intensity'] > 0].copy()
hw['loss'] = hw['loss'] - hw['start'].map(precip_only)
print('precipitation-removal loss by start date (mm w.e.):', precip_only.round(1).to_dict())
hw['mag'] = hw['intensity'] * hw['duration'] * pd.read_csv(HW_ANOMALY_FN)['T2M'].mean()
wf = load(WF_FN)
wf['mag'] = wf['total_deposition']

fig, axes = plt.subplots(1, 2, figsize=(6, 3), layout='constrained')
for ax, events, title, xlabel in [(axes[0], hw, 'Heatwaves (warming only)', 'Temperature anomaly (K day)'),
                                  (axes[1], wf, 'Wildfires', 'BC+OC deposition (mg m$^{-2}$)')]:
    handles = []
    for start, group in events.groupby('start'):
        x, y = group['mag'], group['loss']
        color = TIMING_CMAP(TIMING_NORM(start.dayofyear))
        slope = float((x * y).sum() / (x * x).sum())
        scale, b = power_law(x, y)
        print(f'{title} {start:%b %d}: linear {slope:.3f} mm w.e. per unit, power law exponent {b:.2f}')
        grid = np.linspace(0, x.max() * 1.05, 200)
        ax.scatter(x, y, color=color, s=10, edgecolor='white', linewidth=0.3, zorder=3)
        ax.plot(grid, slope * grid, color=color, linewidth=0.8, linestyle='--')
        ax.plot(grid, scale * grid ** b, color=color, linewidth=1.2)
        handles.append(Line2D([], [], color=color, marker='o', markersize=3, linewidth=1.2,
                              label=f'{start:%b %d} (exponent {b:.2f})'))
    handles += [Line2D([], [], color=INK, linewidth=1.2, label='Power law'),
                Line2D([], [], color=INK, linewidth=0.8, linestyle='--', label='Linear')]
    ax.legend(handles=handles, frameon=False, fontsize=6, loc='upper left')
    ax.set_xlim(0, events['mag'].max() * 1.05)
    ax.set_ylim(0, events['loss'].max() * 1.25)
    ax.set_title(title, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK, labelsize=6)
    ax.xaxis.label.set_size(7)
axes[0].set_ylabel('Additional mass loss (mm w.e.)', fontsize=7)

plt.savefig(FIG_FN, dpi=300, bbox_inches='tight')
print(f'Saved {FIG_FN}')
