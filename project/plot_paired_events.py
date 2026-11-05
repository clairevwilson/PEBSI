"""
Paired heatwaves and wildfires at one glacier. Every heatwave and wildfire in the UK-ESM events
file (extreme_weather_analysis/detect_events.py) at the cell nearest the glacier centroid is
ranked by magnitude against every event of its kind at that cell. Heatwave magnitude is the
cumulative exceedance of the 90th percentile threshold (K day); wildfire magnitude is the total
fire-attributable BC+OC deposition (mg m-2). A heatwave and a wildfire are paired when they overlap
or start within MAX_GAP_DAYS of the other ending. Each pair is colored by its effect on glacier-wide
mass balance in the controlled runs (Base minus Neither), summed from the first day of the pair
to POST_DAYS after its last.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

base_fp = '/ocean/projects/ees260009p/cwilson4/Output/controlled/'
events_fn = '/ocean/projects/ees260009p/cwilson4/climate_data/extreme_events_daily_ukesm.nc'

GLACIER = 'kennicott'
MAX_GAP_DAYS = 7
POST_DAYS = 14
i = 0


def find_events(flag, magnitude):
    """Start, end and summed magnitude of each run of consecutive event days."""
    times = pd.DatetimeIndex(flag.time.values)
    flag = flag.values.astype(bool)
    # the file only holds May-Sep, so a gap in the dates also ends a run
    new_day = np.r_[True, np.diff(times.values) != np.timedelta64(1, 'D')]
    starts = flag & (new_day | ~np.r_[False, flag[:-1]])
    event_id = np.cumsum(starts)[flag]
    df = pd.DataFrame({'time': times[flag], 'magnitude': magnitude.values[flag], 'id': event_id})
    events = df.groupby('id').agg(start=('time', 'min'), end=('time', 'max'), magnitude=('magnitude', 'sum'))
    events['quantile'] = events['magnitude'].rank(pct=True)
    return events.reset_index(drop=True)


def pair_events(heatwaves, wildfires):
    """Every heatwave-wildfire pair that overlaps or is separated by at most MAX_GAP_DAYS."""
    pairs = []
    for _, hw in heatwaves.iterrows():
        for _, wf in wildfires.iterrows():
            gap = max((wf.start - hw.end).days, (hw.start - wf.end).days)
            if gap > MAX_GAP_DAYS:
                continue
            overlap = (min(hw.end, wf.end) - max(hw.start, wf.start)).days + 1
            pairs.append({'start': min(hw.start, wf.start), 'end': max(hw.end, wf.end),
                          'hw_start': hw.start, 'hw_days': (hw.end - hw.start).days + 1,
                          'hw_magnitude': hw.magnitude, 'hw_quantile': hw['quantile'],
                          'wf_start': wf.start, 'wf_days': (wf.end - wf.start).days + 1,
                          'wf_magnitude': wf.magnitude, 'wf_quantile': wf['quantile'],
                          'overlap_days': max(overlap, 0)})
    return pd.DataFrame(pairs)


base = xr.open_dataset(base_fp + f'base_{i}/per_glacier.nc').sel(glacier=GLACIER)
neither = xr.open_dataset(base_fp + f'no_either_{i}/per_glacier.nc').sel(glacier=GLACIER)
daily_effect = (base['mass_balance'] - neither['mass_balance']).resample(time='D').sum().load()

ev = xr.open_dataset(events_fn).sel(y=float(base.lat), x=float(base.lon), method='nearest').load()
heatwaves = find_events(ev['heatwave'], ev['temp'] - ev['temp_thresh'])
wildfires = find_events(ev['wildfire'], (ev['bc'] - ev['bc_nofire']) * 1e6)
pairs = pair_events(heatwaves, wildfires)
print(f'{GLACIER}: cell {float(ev.y)}, {float(ev.x)}; {len(heatwaves)} heatwaves, '
      f'{len(wildfires)} wildfires, {len(pairs)} pairs')

pairs['effect'] = [float(daily_effect.sel(time=slice(p.start, p.end + pd.Timedelta(days=POST_DAYS))).sum()) * 100
                   for p in pairs.itertuples()]
print(pairs.drop(columns=['start', 'end']).to_string(float_format='%.2f'))

fig, ax = plt.subplots(figsize=(5, 4), layout='constrained')
sc = ax.scatter(pairs['hw_quantile'], pairs['wf_quantile'], c=pairs['effect'], cmap='Reds_r',
                vmin=pairs['effect'].min(), vmax=0, s=60, edgecolors='k', linewidths=0.5, zorder=3)
for p in pairs.itertuples():
    ax.annotate(p.start.year,(p.hw_quantile, p.wf_quantile), xytext=(4, 4),
                textcoords='offset points', fontsize=7, color='0.3')
ax.set_xlim(0, 1.05)
ax.set_ylim(0, 1.05)
ax.set_xlabel('Heatwave magnitude quantile')
ax.set_ylabel('Wildfire magnitude quantile')
ax.set_title(f'{GLACIER.capitalize()}, 1997$-$2015', fontsize=10)
fig.colorbar(sc, ax=ax, label='Change in mass balance (cm w.e.)')
plt.savefig(base_fp + f'paired_events_{GLACIER}.png', dpi=300, bbox_inches='tight')
plt.close()
