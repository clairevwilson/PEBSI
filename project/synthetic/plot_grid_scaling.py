"""
How the loss from a synthetic event scales with its size in the MERRA-2 event grids (run_event_grid.py,
results from plot_event_grid.py), as a power law loss = a x size^b fit at each event center over every
duration x intensity (precipitation-kept heatwaves). Heatwave size is the temperature anomaly added (K day:
duration x intensity x the glacier cell's average heatwave daily mean T2M anomaly) and wildfire size the
fire BC + OC deposition added (mg m-2). The fit is least squares on the loss itself, so the smallest events,
whose losses of ~1 mm are close to noise, don't steer it as they would in log space.
Top row: Kennicott, power law (solid) and linear through the origin (dashed) at each center, as in
plot_fixed_events.py. Bottom row: the exponent of every glacier at each center.
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
os.environ['EVENT'] = 'wildfire'
from plot_event_grid import daily_fire_deposition, HEATWAVE_ANOMALY_NC, GLACIERS, YEAR

OUT = '/home/claire/Research/Output/'
FIG_FN = '/home/claire/Research/Figs/event_grid_scaling.png'
SAMPLE = 'kennicott'
CELLS = {'gulkana': (63.5, -145.625), 'lemon_creek': (58.5, -134.375), 'wolverine': (60.5, -148.75),
         'kennicott': (61.5, -143.125), 'kahiltna': (62.5, -151.25)}
# the shared glacier colors (shop/plotting/plot_summer_anomaly_2004.py) and one marker per glacier
GLACIER_COLORS = {'wolverine': '#63c4c7', 'kennicott': '#fcc02e', 'kahiltna': '#4D559C',
                  'lemon_creek': '#60C252', 'gulkana': '#BF1F6A'}
MARKERS = dict(zip(GLACIERS, ['o', 's', '^', 'D', 'v']))


def load(event):
    df = pd.read_csv(OUT + f'{event}_grid_merra2/{event}_grid_effects.csv')
    df = df[df['real_event_of'].isna() & ~df['remove_precip']].copy()
    df['loss'] = -df['effect'] * 1000  # mm w.e.
    if event == 'heatwave':
        t2m = xr.open_dataset(HEATWAVE_ANOMALY_NC)['T2M']
        per_day = {g: float(t2m.sel(lat=la, lon=lo).mean()) for g, (la, lo) in CELLS.items()}
    else:
        per_day = daily_fire_deposition()
    df['size'] = df['duration'] * df['intensity'] * df['glacier'].map(per_day)
    return df


def power_law(x, y):
    """Scale and exponent of y = a x^b, least squares on y."""
    (a, b), _ = curve_fit(lambda x, a, b: a * x ** b, x, y, p0=[y.max() / x.max(), 1.0], maxfev=20000)
    return a, b


def main():
    events = {'heatwave': load('heatwave'), 'wildfire': load('wildfire')}
    centers = sorted(events['heatwave']['center'].unique())
    center_colors = dict(zip(centers, plt.get_cmap('viridis')(np.linspace(0, 0.9, len(centers)))))
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 7))
    titles = {'heatwave': 'Heatwaves', 'wildfire': 'Wildfires'}
    xlabels = {'heatwave': 'Temperature anomaly (K day)', 'wildfire': 'BC + OC deposition (mg m$^{-2}$)'}
    for col, (event, df) in enumerate(events.items()):
        ax = axes[0, col]
        handles = []
        for c in centers:
            d = df[(df['glacier'] == SAMPLE) & (df['center'] == c)]
            x, y = d['size'].values, d['loss'].values
            a, b = power_law(x, y)
            slope = float((x * y).sum() / (x * x).sum())
            grid = np.linspace(0, x.max() * 1.05, 200)
            ax.scatter(x, y, s=10, color=center_colors[c], zorder=3)
            ax.plot(grid, a * grid ** b, color=center_colors[c], lw=1.2)
            ax.plot(grid, slope * grid, color=center_colors[c], lw=0.8, ls='--')
            handles.append(Line2D([], [], color=center_colors[c], marker='o', ms=3, lw=1.2,
                                  label=f'{pd.Timestamp(c):%b %d} (exponent {b:.2f})'))
        handles += [Line2D([], [], color='k', lw=1.2, label='Power law'),
                    Line2D([], [], color='k', lw=0.8, ls='--', label='Linear')]
        ax.legend(handles=handles, fontsize=9, frameon=False, loc='upper left')
        ax.set_xlim(0, None)
        ax.set_ylim(0, df.loc[df['glacier'] == SAMPLE, 'loss'].max() * 1.6)
        ax.set_title(f'{titles[event]}, {SAMPLE.title()}', fontsize=9)
        ax.set_xlabel(xlabels[event], fontsize=9)
        ax.tick_params(labelsize=9)

        ax = axes[1, col]
        for g in GLACIERS:
            b = [power_law(*df[(df['glacier'] == g) & (df['center'] == c)][['size', 'loss']].values.T)[1]
                 for c in centers]
            ax.plot(range(len(centers)), b, marker=MARKERS[g], color=GLACIER_COLORS[g], ms=5, lw=1,
                    label=g.replace('_', ' ').title())
            print(f'{event:<9} {g:<12} exponents by center', np.round(b, 2))
        ax.axhline(1, color='k', lw=0.5)
        ax.set_xticks(range(len(centers)))
        ax.set_xticklabels([f'{pd.Timestamp(c):%b %d}' for c in centers])
        ax.set_xlabel(f'Event center ({YEAR})', fontsize=9)
        ax.set_title(f'{titles[event]}, all glaciers', fontsize=9)
        ax.tick_params(labelsize=9)
    axes[0, 0].set_ylabel('Additional mass loss (mm w.e.)', fontsize=9)
    axes[1, 0].set_ylabel('Power law exponent', fontsize=9)
    axes[1, 1].sharey(axes[1, 0])
    axes[1, 0].set_ylim(0.5, 1.4)
    axes[1, 0].legend(fontsize=9, frameon=False, ncol=2, loc='lower center')
    plt.tight_layout()
    plt.savefig(FIG_FN, dpi=300)
    print(f'Saved {FIG_FN}')


if __name__ == '__main__':
    main()
