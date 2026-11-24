"""
How the loss from a synthetic event scales with its size in the MERRA-2 event grids (run_event_grid.py,
results from plot_event_grid.py), as a power law loss = a x size^b fit at each event center over every
duration x intensity (precipitation-kept heatwaves). Heatwave size is the temperature anomaly added (K day:
duration x intensity x the glacier cell's average heatwave daily mean T2M anomaly) and wildfire size the
fire BC + OC deposition added (mg m-2). The fit is least squares on the loss itself, so the smallest events,
whose losses of ~1 mm are close to noise, don't steer it as they would in log space.
Top row: Kennicott, power law (solid) and linear through the origin (dashed) at each center, as in
plot_fixed_events.py. Bottom row: the exponent of every glacier at each center.
SENS_FN: heatwave sensitivity of the May-Sep mass balance at each event center, one line per glacier, from
the split runs (run_event_grid.py RUN=split_*, the average heatwave over 10 days) against run 0: mass
loss per K day of warming for the full heatwave and for its temperature alone, and per W m-2 day of extra
shortwave for its shortwave alone.
"""
import os
import glob
import yaml
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
os.environ['EVENT'] = 'wildfire'
from plot_event_grid import (daily_fire_deposition, HEATWAVE_ANOMALY_NC, MERRA2_FP, GLACIERS, YEAR, translate_rgi,
                             FIG_DIR)

OUT = '/home/claire/Research/Output/'
FIG_FN = FIG_DIR + 'event_grid_scaling.png'
HW_DIR = OUT + 'heatwave_grid_current/'
SENS_FN = FIG_DIR + 'heatwave_grid_sensitivity.png'
SW_UNIT = 100  # shortwave sensitivity per SW_UNIT W m-2 day
SAMPLE = 'kennicott'
CELLS = {'gulkana': (63.5, -145.625), 'lemon_creek': (58.5, -134.375), 'wolverine': (60.5, -148.75),
         'kennicott': (61.5, -143.125), 'kahiltna': (62.5, -151.25)}
# the shared glacier colors (shop/plotting/plot_summer_anomaly_2004.py) and one marker per glacier
GLACIER_COLORS = {'wolverine': '#63c4c7', 'kennicott': '#fcc02e', 'kahiltna': '#4D559C',
                  'lemon_creek': '#60C252', 'gulkana': '#BF1F6A'}
MARKERS = dict(zip(GLACIERS, ['o', 's', '^', 'D', 'v']))


def load(event):
    df = pd.read_csv(OUT + f'{event}_grid_current/{event}_grid_effects.csv')
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
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
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
    # in the wildfire panel's empty upper half, above its exponents (all below 1)
    axes[1, 1].legend(fontsize=9, frameon=False, ncol=2, loc='upper center', markerscale=0.7, columnspacing=0.8,
                      handlelength=1.5)
    plt.tight_layout()
    plt.savefig(FIG_FN, dpi=300)
    print(f'Saved {FIG_FN}')
    sensitivity_figure(events['heatwave'])


def added_forcing(start, duration):
    """Warming [K day] and extra incoming shortwave [W m-2 day] the average heatwave adds to each glacier over
    duration local (UTC-8) days from start, as pebsi/io/climate.py adds it to each point's MERRA-2 cell: the
    shortwave clipped to SWGDN_cap on top of the no-heatwave MERRA-2 shortwave. Area-weighted over points."""
    anomaly = xr.open_dataset(HEATWAVE_ANOMALY_NC)
    sw_base = xr.open_zarr(MERRA2_FP.format(d='reg01_no_heatwaves', v='SWGDN'), consolidated=False)['SWGDN']
    utc_start = pd.Timestamp(start) + pd.Timedelta(hours=8)
    sw_base = sw_base.sel(time=slice(utc_start, utc_start + pd.Timedelta(days=duration) - pd.Timedelta(minutes=1)))
    local = pd.DatetimeIndex(sw_base.time.values).floor('h') - pd.Timedelta(hours=8)
    r0 = xr.open_zarr(HW_DIR + 'run_000/output.zarr', consolidated=False)
    cell_lat, cell_lon = np.round(r0['lat'].values / 0.5) * 0.5, np.round(r0['lon'].values / 0.625) * 0.625
    added = {}
    for la, lo in set(zip(cell_lat, cell_lon)):
        a = anomaly.sel(lat=la, lon=lo)
        base = sw_base.sel(lat=la, lon=lo).values
        sw = np.clip(base + a['SWGDN'].values[local.hour], 0, a['SWGDN_cap'].values[local.dayofyear - 1, local.hour])
        added[(la, lo)] = (a['T2M'].values[local.hour].sum() / 24, (sw - base).sum() / 24)
    per_point = np.array([added[c] for c in zip(cell_lat, cell_lon)])
    rgiid, w = r0['rgiid'].values, r0['weight'].values
    return {g: (per_point[rgiid == translate_rgi[g]['6']] * w[rgiid == translate_rgi[g]['6'], None]).sum(0)
            / w[rgiid == translate_rgi[g]['6']].sum() for g in GLACIERS}


def split_sensitivity():
    """May-Sep mass loss [mm w.e.] of each split run (the average heatwave with all its anomalies, only its
    temperature or only its shortwave) against run 0, per glacier and center, with the warming [K day] and
    extra incoming shortwave [W m-2 day] the heatwave adds (added_forcing)."""
    season = slice(f'{YEAR}-05-01', f'{YEAR}-09-30')

    def glacier_sum(run, var):
        ds = xr.open_zarr(HW_DIR + f'{run}/output.zarr', consolidated=False)
        x = ds[var].sel(time=season).sum('time').values  # daily values
        rgiid, w = ds['rgiid'].values, ds['weight'].values
        return pd.Series({g: (x * w)[rgiid == translate_rgi[g]['6']].sum() / w[rgiid == translate_rgi[g]['6']].sum()
                          for g in GLACIERS})

    runs, events = {}, {}
    for fn in glob.glob(HW_DIR + 'split_[0-9]*/config.yaml'):
        with open(fn) as f:
            cfg = yaml.safe_load(f)
        center = str((pd.Timestamp(cfg['heatwave_start']) + pd.Timedelta(days=cfg['heatwave_duration'] // 2)).date())
        anomaly = cfg['heatwave_anomaly_fn']
        part = 'temp' if '_temp_only' in anomaly else 'sw' if '_sw_only' in anomaly else 'all'
        runs[(center, part)] = os.path.basename(os.path.dirname(fn))
        events[center] = (cfg['heatwave_start'], cfg['heatwave_duration'])
    base = glacier_sum('run_000', 'mass_balance')
    rows = []
    for center in sorted(events):
        forcing = added_forcing(*events[center])
        loss = {p: -1000 * (glacier_sum(runs[(center, p)], 'mass_balance') - base) for p in ['all', 'temp', 'sw']}
        for g in GLACIERS:
            rows.append(dict(glacier=g, center=center, kday=forcing[g][0], swday=forcing[g][1],
                             **{f'loss_{p}': loss[p][g] for p in loss}))
    return pd.DataFrame(rows)


def sensitivity_figure(df):
    """Mass loss per unit of heatwave forcing at each center, one line per glacier, from the split runs (the
    average heatwave over 10 days): the full heatwave and its temperature alone per K day, and its shortwave
    alone per SW_UNIT W m-2 day."""
    centers = sorted(df['center'].unique())
    split = split_sensitivity()
    split['all'] = split['loss_all'] / split['kday']
    split['temp'] = split['loss_temp'] / split['kday']
    split['sw'] = SW_UNIT * split['loss_sw'] / split['swday']
    # what the full heatwave does beyond its temperature and shortwave alone: longwave, humidity, interactions
    split['rest'] = split['loss_all'] - split['loss_temp'] - split['loss_sw']
    pd.set_option('display.width', 200)
    print(split.round(2).to_string())
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 3.5))
    axes[1].sharey(axes[0])
    for g in GLACIERS:
        s = split[split['glacier'] == g].set_index('center').loc[centers]
        lines = [s['all'], s['temp'], s['sw']]
        for ax, y in zip(axes, lines):
            ax.plot(range(len(centers)), y, marker=MARKERS[g], color=GLACIER_COLORS[g], ms=5, lw=1,
                    label=g.replace('_', ' ').title())
    for ax, title in zip(axes, ['Full heatwave', 'Temperature only', 'Shortwave only']):
        ax.set_title(title, fontsize=9)
        ax.set_xticks(range(len(centers)))
        ax.set_xticklabels([f'{pd.Timestamp(c):%b %d}' for c in centers], rotation=45, ha='right')
        ax.set_ylim(0, None)
        ax.tick_params(labelsize=9)
    axes[0].set_ylabel('Mass loss per K day\nof warming (mm w.e.)', fontsize=9)
    axes[2].set_ylabel(f'Mass loss per {SW_UNIT} W m$^{{-2}}$ day\nof extra shortwave (mm w.e.)', fontsize=9)
    axes[1].set_xlabel(f'Heatwave center ({YEAR})', fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=9, frameon=False)
    plt.tight_layout(rect=(0, 0.07, 1, 1))
    plt.savefig(SENS_FN, dpi=300)
    print(f'Saved {SENS_FN}')


if __name__ == '__main__':
    main()
