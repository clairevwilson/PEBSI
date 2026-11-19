"""
Effect of each synthetic event in a grid experiment (run_event_grid.py; EVENT=heatwave, the
default, or wildfire) on each glacier's May-Sep YEAR mass balance, relative to run 0 (no event),
saved to <event>_grid_effects.csv.
Heatwaves: the precipitation-kept runs against event center, duration, intensity and
duration x intensity, and the effect by elevation band.
Both events: a combined figure of the synthetic events (dots, colored by center) and the real seasons
of the MERRA-2 deposition controlled runs, 2000-2025 (plus signs, Base minus No fires / No heatwaves,
colored by event-weighted day of year), against duration x intensity in days of the cell's average
heatwave or fire-attributable deposition, with each glacier's largest real event (REAL_EVENTS) run in
YEAR in red.
Deposition is as PEBSI sees it: MERRA-2 representative bin times its ratio to total (dry only).
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
from run_event_grid import OUT_DIR, YEAR, design, REAL_EVENTS, DAMAGING_EVENTS
from prepare_synthetic_wildfires import MERRA2_ANOMALY_NC, MERRA2_FP, MERRA2_VARS
from project.parameters import GLACIERS, translate_rgi, host, HOST_PATHS

SEASON = slice(f'{YEAR}-05-01', f'{YEAR}-09-30')
REAL_LABEL = f'Largest real event,\nrun in {YEAR}'
DAMAGING_LABEL = f'Most damaging real\nevent, run in {YEAR}'
# PEBSI's MERRA-2 representative bin to total ratio, applied to dry deposition only (apply_merra_dep_ratio)
RATIO_FN = HOST_PATHS[host]['climate_fp'] + 'MERRA2/reg01_{sp}_regression_map.nc'
HEATWAVE_ANOMALY_NC = HOST_PATHS[host]['climate_fp'] + 'MERRA2/reg01_no_heatwaves/heatwave_anomaly.nc'
RESULTS_FN = os.path.join(OUT_DIR, f'{EVENT}_grid_effects.csv')
FIG_FN = os.path.join(OUT_DIR, f'{EVENT}_grid_effects.png')


def season_mb(run):
    """Area-weighted May-Sep mass balance [m w.e.] of each glacier."""
    ds = xr.open_zarr(os.path.join(OUT_DIR, f'run_{run:03d}', 'output.zarr'), consolidated=False)
    mb = ds['mass_balance'].sel(time=SEASON).sum('time').values
    rgiid, weight = ds['rgiid'].values, ds['weight'].values
    return {g: (mb * weight)[rgiid == translate_rgi[g]['6']].sum() / weight[rgiid == translate_rgi[g]['6']].sum()
            for g in GLACIERS}


def merra2_ratio(name, lat, lon):
    """PEBSI's multiplier on MERRA-2 component name ('bcdry', ...) at (lat, lon): the bin ratio if dry, else 1."""
    if 'wet' in name:
        return 1
    ratio = xr.open_dataset(RATIO_FN.format(sp=name[:2].upper()))['ratio']
    return ratio.sel(lat=lat, lon=lon, method='nearest').values


def daily_fire_deposition():
    """
    Area-weighted daily deposition [mg m-2] a 1x synthetic fire adds to each glacier as PEBSI
    sees it: each point's MERRA-2 cell's average fire times PEBSI's MERRA-2 ratio.
    """
    ds = xr.open_zarr(os.path.join(OUT_DIR, 'run_000', 'output.zarr'), consolidated=False)
    rgiid, weight = ds['rgiid'].values, ds['weight'].values
    cell_lat = xr.DataArray(np.round(ds['lat'].values / 0.5) * 0.5, dims='point')
    cell_lon = xr.DataArray(np.round(ds['lon'].values / 0.625) * 0.625, dims='point')
    anomaly = xr.open_dataset(MERRA2_ANOMALY_NC)
    daily = 0
    for name in MERRA2_VARS:
        daily = daily + (anomaly[name].sel(lat=cell_lat, lon=cell_lon).values
                         * merra2_ratio(name, cell_lat, cell_lon) * 86400 * 1e6)
    return {g: (daily * weight)[rgiid == translate_rgi[g]['6']].sum() / weight[rgiid == translate_rgi[g]['6']].sum()
            for g in GLACIERS}


def real_fire_deposition(lat, lon):
    """Daily (UTC) fire-attributable deposition [mg m-2] at the MERRA-2 cell nearest (lat, lon), as PEBSI
    sees it: reg01 minus reg01_no_wildfires, times PEBSI's MERRA-2 ratio."""
    lat, lon = round(lat / 0.5) * 0.5, round(lon / 0.625) * 0.625
    total = 0
    for name, vn in MERRA2_VARS.items():
        diff = 0
        for d in ['reg01', 'reg01_no_wildfires']:
            s = xr.open_zarr(MERRA2_FP.format(d=d, v=vn), consolidated=False)[vn].sel(lat=lat, lon=lon).to_series()
            diff = s if d == 'reg01' else diff - s
        total = total + diff * float(merra2_ratio(name, lat, lon)) * 3600 * 1e6
    return total.groupby(total.index.normalize()).sum()


def real_heatwave_days(lat, lon):
    """Daily (UTC) heatwave dose at the MERRA-2 cell nearest (lat, lon) in days of that cell's average
    heatwave: reg01 minus reg01_no_heatwaves T2M, summed over the day and divided by 24 x the average
    heatwave's mean T2M anomaly, so that a synthetic event's dose is its duration x intensity."""
    lat, lon = round(lat / 0.5) * 0.5, round(lon / 0.625) * 0.625
    t = [xr.open_zarr(MERRA2_FP.format(d=d, v='T2M'), consolidated=False)['T2M'].sel(lat=lat, lon=lon).to_series()
         for d in ['reg01', 'reg01_no_heatwaves']]
    mean_anomaly = float(xr.open_dataset(HEATWAVE_ANOMALY_NC)['T2M'].sel(lat=lat, lon=lon).mean())
    diff = (t[0] - t[1]) / (24 * mean_anomaly)
    return diff.groupby(diff.index.normalize()).sum()


def real_seasons(daily, effect, glacier):
    """(season total of daily, season sum of effect, daily-weighted day of year) for each May-Sep of
    2000-2025; the day of year is NaN for seasons without an event."""
    seasons = []
    for year in range(2000, 2026):
        s = slice(f'{year}-05-01', f'{year}-09-30')
        d = daily.loc[s].clip(lower=0)
        seasons.append((d.sum(), float(effect.sel(glacier=glacier, time=s).sum()),
                        (d * d.index.dayofyear).sum() / d.sum() if d.sum() > 0 else np.nan))
    return np.array(seasons).T


def combined_figure(df, x, real, glacier_daily, effect, xlabel, fn):
    """One panel per glacier: synthetic grid events (dots, colored by center) and real 2000-2025 seasons of
    the controlled runs with an event (plus signs, colored by event-weighted day of year) against x, with each glacier's
    largest real event run in YEAR in red; legend and colorbar in the sixth slot."""
    doy = dict(cmap='viridis', vmin=135, vmax=227)
    fig, axes = plt.subplots(2, 3, figsize=(6, 4), sharex=True, sharey=True)
    for ax, g in zip(axes.flat, GLACIERS):
        d = df[df['glacier'] == g]
        sc = ax.scatter(d[x], d['effect'], c=pd.to_datetime(d['center']).dt.dayofyear, s=10, **doy)
        rx, ry, rdoy = real_seasons(glacier_daily(g), effect, g)
        has_event = rx > 0  # seasons without an event are left out
        ax.scatter(rx[has_event], ry[has_event], c=rdoy[has_event], marker='+', s=30, linewidths=1, **doy)
        ax.scatter(real.loc[g, x], real.loc[g, 'effect'], s=20, color='red')
        ax.set_title(g.replace('_', ' ').title(), fontsize=9)
        ax.tick_params(labelsize=9)
    axes[1, 2].axis('off')
    axes[0, 2].tick_params(labelbottom=True)
    cax = axes[1, 2].inset_axes([0.05, 0.72, 0.9, 0.08])
    cbar = fig.colorbar(sc, cax=cax, orientation='horizontal')
    cbar.set_label('Day of year', fontsize=9)
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.tick_params(labelsize=9)
    handles = [plt.Line2D([], [], ls='', marker='o', ms=4, color='grey'),
               plt.Line2D([], [], ls='', marker='+', ms=7, color='grey'),
               plt.Line2D([], [], ls='', marker='o', ms=5, color='red')]
    labels = [f'Synthetic, {YEAR}', 'Real season', f'Largest real, {YEAR}']
    axes[1, 2].legend(handles, labels, fontsize=9, loc='lower center', bbox_to_anchor=(0.5, -0.1),
                      frameon=False, handletextpad=0.2)
    fig.supxlabel(xlabel, fontsize=9)
    fig.supylabel('Change in May-Sep mass balance (m w.e.)', fontsize=9)
    plt.tight_layout()
    plt.savefig(fn, dpi=300)
    print(f"Saved {fn}")


def controlled_effect(removed):
    """Base minus the MERRA-2 deposition controlled run with `removed` taken out, per glacier and hour,
    and the glaciers' centroid lat/lon."""
    controlled = os.path.join(OUT_DIR, '..', 'controlled', '{}_merra2_0', 'per_glacier.nc')
    base = xr.open_dataset(controlled.format('base'))['mass_balance']
    return base - xr.open_dataset(controlled.format(removed))['mass_balance'], base.lat, base.lon


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
            r['total_deposition'] = r['duration'].astype(float) * r['intensity'] * r.index.map(daily).astype(float)
            print(r[['center', 'duration', 'intensity', 'total_deposition', 'effect']])
        effect, lat, lon = controlled_effect('no_fires')
        combined_figure(df, 'total_deposition', real,
                        lambda g: real_fire_deposition(float(lat.sel(glacier=g)), float(lon.sel(glacier=g))),
                        effect, 'Fire BC + OC deposition (mg m$^{-2}$)',
                        os.path.join(OUT_DIR, 'wildfire_grid_effects_combined.png'))
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
    effect, lat, lon = controlled_effect('no_heatwaves')
    combined_figure(df, 'total', real,
                    lambda g: real_heatwave_days(float(lat.sel(glacier=g)), float(lon.sel(glacier=g))),
                    effect, 'Duration × intensity (days of average heatwave)',
                    os.path.join(OUT_DIR, 'heatwave_grid_effects_combined.png'))

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
