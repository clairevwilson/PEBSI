"""
One-panel summary of the synthetic heatwave experiment from synthetic_heatwave_results.csv
(analyze_synthetic_heatwaves.py). Each event is a line spanning its dates, so its position
gives the timing and its length the duration; its height is the intensity and its color the
change in Kennicott's May-Sep 2006 mass balance relative to the run without an event, with the
snowfall the event removed (baseline_accumulation.py) added back, so the precipitation the event
sets to zero does not count toward its effect.
"""
import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects  # noqa: F401 -- registers mpl.patheffects

OUT_DIR = '/ocean/projects/ees260009p/cwilson4/Output/synthetic_heatwaves/'
RESULTS_FN = os.path.join(OUT_DIR, 'synthetic_heatwave_results.csv')
ANOMALY_FN = os.path.join(OUT_DIR, 'synthetic_heatwave_anomaly.csv')
ACCUM_FN = os.path.join(OUT_DIR, 'synthetic_heatwave_accumulation.csv')
FIG_FN = os.path.join(OUT_DIR, 'synthetic_heatwave_effects.png')

INK = 'k'
GRID = '#e6e5e1'

# real Kennicott heatwave (hours prepare_no_heatwaves.py replaced) with the greatest effect on mass
# balance in the controlled base minus no_heatwaves runs, 1997-2015: 17 Jun to 3 Jul 2004, shown
# at the same calendar dates in 2006; intensity is its mean T2M anomaly over the average heatwave's
REAL_START, REAL_END, REAL_INTENSITY = '2006-06-17', '2006-07-04', 1.29
# its effect: change in base minus no_heatwaves mass balance from its start until the next heatwave
# (no_heatwaves keeps the original precipitation, so nothing to add back)
REAL_EFFECT = -0.142

results = pd.read_csv(RESULTS_FN, index_col='run', parse_dates=['start'])
baseline_mb = results.loc[0, 'season_mb']
events = results.drop(0)
events['end'] = events['start'] + pd.to_timedelta(events['duration'], unit='D')
events['effect'] += pd.read_csv(ACCUM_FN, index_col='run')['accum_removed']
anomaly = pd.read_csv(ANOMALY_FN, index_col='local_hour').mean()

cmap = plt.get_cmap('viridis')
norm = mpl.colors.Normalize(vmin=np.floor(events['effect'].min() * 20) / 20, vmax=0)

# white outline keeps events at nearby intensities apart; larger effects drawn on top
outline = [mpl.patheffects.Stroke(linewidth=3.4, foreground='white'), mpl.patheffects.Normal()]
fig, ax = plt.subplots(figsize=(7, 4))

for _, ev in events.sort_values('effect', ascending=False).iterrows():
    ax.plot([ev['start'], ev['end']], [ev['intensity']] * 2, color=cmap(norm(ev['effect'])),
            linewidth=2.2, solid_capstyle='butt', path_effects=outline)
    
ax.plot([pd.Timestamp(REAL_START), pd.Timestamp(REAL_END)], [REAL_INTENSITY] * 2, color=cmap(norm(REAL_EFFECT)),
        linewidth=2.2, solid_capstyle='projecting',
        path_effects=[mpl.patheffects.Stroke(linewidth=3, foreground=INK), mpl.patheffects.Normal()])
ax.annotate('2004 heatwave', (pd.Timestamp(REAL_START), REAL_INTENSITY), xytext=(0, 4),
            textcoords='offset points', color=INK, fontsize=9)

months = pd.date_range('2006-05-01', '2006-09-01', freq='MS')
ax.set_xticks(months)
ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%b'))
ax.set_xlim(pd.Timestamp('2006-04-28'), pd.Timestamp('2006-09-24'))
ax.set_ylim(0.43, 2.07)
ax.grid(axis='x', color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
ax.spines[['top', 'right']].set_visible(False)
ax.spines[['left', 'bottom']].set_color(INK)
ax.tick_params(colors=INK, labelsize=9)
ax.set_ylabel('Intensity (× average heatwave)',
        #       f"1× = +{anomaly['T2M']:.1f} K, +{anomaly['SWGDN']:.0f} W m$^{{-2}}$ shortwave",
              color=INK, fontsize=9)
ax.set_xlabel('Event dates, 2006', color=INK, fontsize=9)

cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.85, pad=0.02)
cbar.set_label(f'Change in mass balance excl. lost snowfall (m w.e.)\n(No event: {baseline_mb:.2f} m w.e.)',
               color=INK, fontsize=9)
cbar.ax.tick_params(colors=INK, labelsize=8)

plt.savefig(FIG_FN, dpi=300, bbox_inches='tight')
print(f"Saved {FIG_FN}")
