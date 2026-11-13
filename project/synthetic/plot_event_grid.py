"""
Effect of each synthetic event in a grid experiment (run_event_grid.py; EVENT=heatwave, the
default, or wildfire) on each glacier's May-Sep 1999 mass balance, relative to run 0 (no event),
saved to <event>_grid_effects.csv.
Heatwaves: the precipitation-kept runs against event center, duration, intensity and
duration x intensity, the latter by center, and the effect by elevation band.
Wildfires: the effect against each glacier's total fire-attributable deposition, by center.
The by-center plots also show each glacier's largest real event (REAL_EVENTS) and, for wildfires,
its most damaging one (DAMAGING_EVENTS), run in 1999, in red. For wildfires, a second figure shows the
real seasons of the controlled runs, 1997-2015: Base minus No fires against the season's
fire-attributable deposition, colored by its deposition-weighted day of year.
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ.setdefault('RUN', '0')
EVENT = os.environ.setdefault('EVENT', 'heatwave')
from run_event_grid import OUT_DIR, design, REAL_EVENTS, DAMAGING_EVENTS
from prepare_synthetic_wildfires import ANOMALY_NC, COMPONENTS, ratio_fp
from project.parameters import GLACIERS, translate_rgi, host, HOST_PATHS

SEASON = slice('1999-05-01', '1999-09-30')
REAL_LABEL = 'Largest real event,\nrun in 1999'
DAMAGING_LABEL = 'Most damaging real\nevent, run in 1999'
RESULTS_FN = os.path.join(OUT_DIR, f'{EVENT}_grid_effects.csv')
FIG_FN = os.path.join(OUT_DIR, f'{EVENT}_grid_effects.png')


def season_mb(run):
    """Area-weighted May-Sep mass balance [m w.e.] of each glacier."""
    ds = xr.open_zarr(os.path.join(OUT_DIR, f'run_{run:03d}', 'output.zarr'), consolidated=False)
    mb = ds['mass_balance'].sel(time=SEASON).sum('time').values
    rgiid, weight = ds['rgiid'].values, ds['weight'].values
    return {g: (mb * weight)[rgiid == translate_rgi[g]['6']].sum() / weight[rgiid == translate_rgi[g]['6']].sum()
            for g in GLACIERS}


def daily_fire_deposition():
    """
    Area-weighted daily deposition [mg m-2] a 1x synthetic fire adds to each glacier as PEBSI
    sees it: each point's MERRA-2 cell's average fire times its UK-ESM to MERRA-2 ratio.
    """
    ds = xr.open_zarr(os.path.join(OUT_DIR, 'run_000', 'output.zarr'), consolidated=False)
    rgiid, weight = ds['rgiid'].values, ds['weight'].values
    cell_lat = xr.DataArray(np.round(ds['lat'].values / 0.5) * 0.5, dims='point')
    cell_lon = xr.DataArray(np.round(ds['lon'].values / 0.625) * 0.625, dims='point')
    anomaly = xr.open_dataset(ANOMALY_NC)
    daily = 0
    for sp, t in COMPONENTS:
        ratio = xr.open_dataarray(ratio_fp.format(sp=sp, t=t)).sel(lat=cell_lat, lon=cell_lon, method='nearest')
        daily = daily + anomaly[f'{sp}{t}'].sel(lat=cell_lat, lon=cell_lon).values * ratio.values * 86400 * 1e6
    return {g: (daily * weight)[rgiid == translate_rgi[g]['6']].sum() / weight[rgiid == translate_rgi[g]['6']].sum()
            for g in GLACIERS}


if __name__ == '__main__':
    base = season_mb(0)
    rows = []
    for run, (center, duration, intensity, remove_precip) in enumerate(design()):
        if run == 0:
            continue
        mb = season_mb(run)
        for g in GLACIERS:
            rows.append(dict(run=run, glacier=g, center=center, duration=duration, intensity=intensity,
                             remove_precip=remove_precip, season_mb=mb[g], effect=mb[g] - base[g]))
    df = pd.DataFrame(rows)
    # the last runs are each glacier's real events; only its own glacier's result is used
    real_runs = [(g, 'largest') for g in REAL_EVENTS] + [(g, 'damaging') for g in DAMAGING_EVENTS]
    first_real = len(design()) - len(real_runs)
    df['real_event_of'] = df['run'].map(lambda r: real_runs[r - first_real][0] if r >= first_real else '')
    df['real_kind'] = df['run'].map(lambda r: real_runs[r - first_real][1] if r >= first_real else '')
    df.to_csv(RESULTS_FN, index=False)
    print(f"Saved {RESULTS_FN}")
    own = df['real_event_of'] == df['glacier']
    real = df[own & (df['real_kind'] == 'largest')].set_index('glacier')
    damaging = df[own & (df['real_kind'] == 'damaging')].set_index('glacier')
    df = df[df['real_event_of'] == '']

    centers = sorted(df['center'].unique())
    colors = plt.get_cmap('viridis')(np.linspace(0, 0.9, len(centers)))
    if EVENT == 'wildfire':
        daily = daily_fire_deposition()
        df['total_deposition'] = df['duration'] * df['intensity'] * df['glacier'].map(daily)
        for r in (real, damaging):
            r['total_deposition'] = r['duration'] * r['intensity'] * r.index.map(daily)
            print(r[['center', 'duration', 'intensity', 'total_deposition', 'effect']])
        fig, axes = plt.subplots(2, 3, figsize=(6, 4), sharex=True, sharey=True)
        for ax, g in zip(axes.flat, GLACIERS):
            for c, col in zip(centers, colors):
                d = df[(df['glacier'] == g) & (df['center'] == c)]
                ax.scatter(d['total_deposition'], d['effect'], s=10, color=col,
                           label=pd.Timestamp(c).strftime('%b %d'))
            ax.scatter(real.loc[g, 'total_deposition'], real.loc[g, 'effect'], s=20, color='red', label=REAL_LABEL)
            if g in damaging.index:
                ax.scatter(damaging.loc[g, 'total_deposition'], damaging.loc[g, 'effect'], s=24, marker='^',
                           color='red', label=DAMAGING_LABEL)
            ax.set_title(g.replace('_', ' ').title(), fontsize=9)
            ax.tick_params(labelsize=9)
        axes[1, 2].axis('off')
        axes[0, 2].tick_params(labelbottom=True)
        fig.supxlabel('Total fire BC + OC deposition (mg m$^{-2}$)', fontsize=9)
        fig.supylabel('Change in May-Sep mass balance (m w.e.)', fontsize=9)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        axes[1, 2].legend(handles, labels, fontsize=9, title='Event center', title_fontsize=9, loc='center')
        plt.tight_layout()
        fn = os.path.join(OUT_DIR, 'wildfire_grid_effects_by_center.png')
        plt.savefig(fn, dpi=300)
        print(f"Saved {fn}")

        # real seasons of the controlled runs at the MERRA-2 cell nearest each glacier's centroid
        controlled = os.path.join(OUT_DIR, '..', 'controlled', '{}_0', 'per_glacier.nc')
        mb_base = xr.open_dataset(controlled.format('base'))['mass_balance']
        effect = mb_base - xr.open_dataset(controlled.format('no_fires'))['mass_balance']
        events = xr.open_dataset(HOST_PATHS[host]['climate_fp'] + 'extreme_events_daily_ukesm.nc')
        fig, axes = plt.subplots(2, 3, figsize=(6, 4), sharex=True, sharey=True)
        for ax, g in zip(axes.flat, GLACIERS):
            ev = events.sel(y=float(mb_base.lat.sel(glacier=g)), x=float(mb_base.lon.sel(glacier=g)), method='nearest')
            dep = ((ev['bc'] - ev['bc_nofire']) * 1e6).to_series()
            seasons = []
            for year in range(1997, 2016):
                s = slice(f'{year}-05-01', f'{year}-08-31' if year == 2015 else f'{year}-09-30')
                d = dep.loc[s].clip(lower=0)
                seasons.append((d.sum(), float(effect.sel(glacier=g, time=s).sum()),
                                (d * d.index.dayofyear).sum() / d.sum()))
            x, y, doy = np.array(seasons).T
            sc = ax.scatter(x, y, c=doy, s=10, cmap='viridis', vmin=135, vmax=227)
            ax.set_title(g.replace('_', ' ').title(), fontsize=9)
            ax.tick_params(labelsize=9)
        axes[1, 2].axis('off')
        axes[0, 2].tick_params(labelbottom=True)
        cax = axes[1, 2].inset_axes([0.1, 0.45, 0.8, 0.1])
        cbar = fig.colorbar(sc, cax=cax, orientation='horizontal')
        cbar.set_label('Deposition-weighted day of year', fontsize=9)
        cbar.ax.tick_params(labelsize=9)
        fig.supxlabel('Season fire BC + OC deposition (mg m$^{-2}$)', fontsize=9)
        fig.supylabel('Change in May-Sep mass balance (m w.e.)', fontsize=9)
        plt.tight_layout()
        fn = os.path.join(OUT_DIR, 'wildfire_real_seasons_by_doy.png')
        plt.savefig(fn, dpi=300)
        print(f"Saved {fn}")
        sys.exit()

    df = df[~df['remove_precip']].copy()
    df['center_doy'] = pd.to_datetime(df['center']).dt.dayofyear
    df['total'] = df['duration'] * df['intensity']
    xvars = [('center_doy', 'Event center (day of year)'), ('duration', 'Duration (days)'),
             ('intensity', 'Intensity (× average heatwave)'), ('total', 'Duration × intensity (days)')]
    markers = ['o', 's', '^', 'D', 'v']

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 7), sharey=True)
    for ax, (x, label) in zip(axes.flat, xvars):
        levels = np.sort(df[x].unique())
        # the first three have only four levels: offset each glacier sideways so they don't overlap
        dodge = 0.15 * np.diff(levels).min() if len(levels) == 4 else 0
        for k, (g, m) in enumerate(zip(GLACIERS, markers)):
            d = df[df['glacier'] == g]
            ax.scatter(d[x] + (k - 2) * dodge / 2, d['effect'], marker=m, s=18,
                       label=g.replace('_', ' ').title())
        ax.set_xlabel(label, fontsize=9)
        ax.tick_params(labelsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel('Change in May-Sep mass balance (m w.e.)', fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=9)
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    plt.savefig(FIG_FN, dpi=300)
    print(f"Saved {FIG_FN}")

    # duration x intensity alone, one glacier per panel, colored by event center; legend in the sixth slot
    real['total'] = real['duration'] * real['intensity']
    print(real[['center', 'duration', 'intensity', 'total', 'effect']])
    fig, axes = plt.subplots(2, 3, figsize=(6, 4), sharex=True, sharey=True)
    for ax, g in zip(axes.flat, GLACIERS):
        for c, col in zip(centers, colors):
            d = df[(df['glacier'] == g) & (df['center'] == c)]
            ax.scatter(d['total'], d['effect'], s=10, color=col,
                       label=pd.Timestamp(c).strftime('%b %d'))
        ax.scatter(real.loc[g, 'total'], real.loc[g, 'effect'], s=20, color='red', label=REAL_LABEL)
        ax.set_title(g.replace('_', ' ').title(), fontsize=9)
        ax.tick_params(labelsize=9)
    axes[1, 2].axis('off')
    axes[0, 2].tick_params(labelbottom=True)
    fig.supxlabel('Duration × intensity (days)', fontsize=9)
    fig.supylabel('Change in May-Sep mass balance (m w.e.)', fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[1, 2].legend(handles, labels, fontsize=9, title='Event center', title_fontsize=9, loc='center')
    plt.tight_layout()
    fn = os.path.join(OUT_DIR, 'heatwave_grid_effects_by_center.png')
    plt.savefig(fn, dpi=300)
    print(f"Saved {fn}")

    # effect by 100 m elevation band of the 10-day, 1.1x events at each center
    r0 = xr.open_zarr(os.path.join(OUT_DIR, 'run_000', 'output.zarr'), consolidated=False)
    rgiid, weight, elev = r0['rgiid'].values, r0['weight'].values, r0['elev'].values
    base_mb = r0['mass_balance'].sel(time=SEASON).sum('time').values
    fig, axes = plt.subplots(2, 3, figsize=(6, 4), sharex=True)
    for c, col in zip(centers, colors):
        run = design().index((c, 10, 1.1, False))
        ds = xr.open_zarr(os.path.join(OUT_DIR, f'run_{run:03d}', 'output.zarr'), consolidated=False)
        effect = ds['mass_balance'].sel(time=SEASON).sum('time').values - base_mb
        for ax, g in zip(axes.flat, GLACIERS):
            m = rgiid == translate_rgi[g]['6']
            band = (elev[m] // 100) * 100 + 50
            b = pd.DataFrame({'band': band, 'ew': effect[m] * weight[m], 'w': weight[m]}).groupby('band').sum()
            ax.plot(b['ew'] / b['w'], b.index, 'o-', ms=3, color=col, label=pd.Timestamp(c).strftime('%b %d'))
    for ax, g in zip(axes.flat, GLACIERS):
        ax.set_title(g.replace('_', ' ').title(), fontsize=9)
        ax.tick_params(labelsize=9)
    fig.delaxes(axes[1, 2])
    axes[0, 2].tick_params(labelbottom=True)
    fig.supxlabel('Change in May-Sep mass balance (m w.e.)', fontsize=9)
    fig.supylabel('Elevation (m)', fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=9, title='10-day, 1.1× event', title_fontsize=9,
               loc='center', bbox_to_anchor=(0.84, 0.3))
    plt.tight_layout()
    fn = os.path.join(OUT_DIR, 'heatwave_grid_effects_by_elevation.png')
    plt.savefig(fn, dpi=300)
    print(f"Saved {fn}")
