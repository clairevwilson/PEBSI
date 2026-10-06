"""
Scores and plots the one-at-a-time sensitivity runs on Gulkana
(batch_jobs/sensitivity.sh): every run in Output/sensitivity is either
the baseline or one parameter moved to its low or high value.

Each run is summarized by what the model does rather than how well it
matches observations, all area-weighted by ds.weight:
    albedo  mean May-September albedo
    mb      glacier-wide mass balance rate over the run
    wet     mean May-September fraction of the glacier that is wet, using
            the rule project.glacierwide_loss.Wet applies (daily mean
            total_water above 2 kg m-2)

Values are cached to sensitivity_values.csv in the run directory, so the
figure can be iterated on without rereading every output.zarr. Only runs
missing from it are scored; drop a run's row (or the file) to rescore it.

@author: clairevwilson
"""
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GLACIER = 'gulkana'
RUN_FP = '/ocean/projects/ees260009p/cwilson4/Output/sensitivity/'
VALUES_FN = os.path.join(RUN_FP, 'sensitivity_values.csv')
FIG_FN = os.path.join(RUN_FP, 'sensitivity.png')

# plotting order, top to bottom
PARAMS = ['Boone_c1', 'Boone_c2', 'Boone_c3', 'Boone_c4', 'Boone_c5',
          'Sr', 'grainsize_rfz', 'ksp_BC', 'ksp_OC', 'ksp_dust', 'dust_factor',
          'snow_threshold', 'kp', 'precgrad', 'lapse_rate', 'wind_factor',
          'roughness_fresh_snow', 'roughness_aged_snow', 'roughness_firn',
          'roughness_ice', 'roughness_aging_rate']

COLORS = {'low': '#2a78d6', 'high': '#eb6834'}

MELT_MONTHS = [5, 6, 7, 8, 9]
WET_THRESHOLD = 2   # daily minimum total_water counted as wet [kg m-2]

PANELS = [('albedo', '$\\Delta$ May-Sep albedo'),
          ('mb', '$\\Delta$ Mass balance\n(m w.e. a$^{-1}$)'),
          ('wet', '$\\Delta$ May-Sep wet fraction')]


def score_runs():
    df = pd.read_csv(VALUES_FN, index_col='run') if os.path.exists(VALUES_FN) else pd.DataFrame()
    names = ['baseline'] + [f'{p}_{s}' for p in PARAMS for s in ('low', 'high')]
    missing = [n for n in names if n not in df.index]
    if not missing:
        return df

    rows = []
    for name in missing:
        ds = xr.open_zarr(os.path.join(RUN_FP, name, 'output.zarr'), consolidated=False)
        ds = ds[['time', 'albedo', 'total_water', 'mass_balance', 'weight']].load()
        weight = ds.weight / ds.weight.sum()

        n_years = (ds.time.values[-1] - ds.time.values[0]) / np.timedelta64(1, 'D') / 365.25
        mb = float((ds.mass_balance.sum(dim='time') * weight).sum()) / n_years

        is_wet = ds.total_water.resample(time='1d').mean() > WET_THRESHOLD
        melt_albedo = ds.albedo.sel(time=ds.time.dt.month.isin(MELT_MONTHS))
        melt_wet = is_wet.sel(time=is_wet.time.dt.month.isin(MELT_MONTHS))

        rows.append(dict(run=name,
                         albedo=float((melt_albedo * weight).sum('point').mean()),
                         mb=mb,
                         wet=float((melt_wet * weight).sum('point').mean())))
        print(f'{name:<28} albedo={rows[-1]["albedo"]:.4f} '
              f'mb={mb:+.3f} wet={rows[-1]["wet"]:.4f}', flush=True)

    df = pd.concat([df, pd.DataFrame(rows).set_index('run')])
    df.index.name = 'run'
    df.to_csv(VALUES_FN)
    return df


def plot(df):
    y = np.arange(len(PARAMS))
    height = 0.38

    # every bar is the change from the baseline run, on an axis centered on 0
    fig, axes = plt.subplots(1, 3, figsize=(7, 5), sharey=True)
    for ax, (metric, label) in zip(axes, PANELS):
        base = df.loc['baseline', metric]
        diffs = {side: np.array([df.loc[f'{p}_{side}', metric] for p in PARAMS]) - base
                 for side in ('low', 'high')}
        for offset, side in [(-height / 2, 'low'), (height / 2, 'high')]:
            ax.barh(y + offset, diffs[side], height=height * 0.9,
                    color=COLORS[side], label=side)
        ax.axvline(0, color='0.2', ls='--', lw=1, label='baseline')
        xmax = 1.05 * max(np.abs(diffs['low']).max(), np.abs(diffs['high']).max())
        ax.set_xlim(-xmax, xmax)
        ax.set_xlabel(label)
        ax.grid(axis='x', color='0.9', lw=0.8)
        ax.set_axisbelow(True)
        ax.patch.set_visible(False)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
    axes[0].set_xticks([-0.04, 0, 0.04])

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(PARAMS)
    axes[0].set_ylim(len(PARAMS) - 0.5, -0.5)
    axes[-1].legend(loc='lower right', facecolor='white', framealpha=1)
    # fig.suptitle('Gulkana one-at-a-time sensitivity, Apr 2020 to Apr 2023')
    fig.tight_layout()

    # thin row separators running continuously across all three panels, drawn
    # behind the axes (whose backgrounds are hidden) so bars and the legend
    # sit on top; placed after tight_layout so the panel edges are final
    trans = blended_transform_factory(fig.transFigure, axes[0].transData)
    x0, x1 = axes[0].get_position().x0, axes[-1].get_position().x1
    for b in y[:-1] + 0.5:
        fig.add_artist(Line2D([x0, x1], [b, b], transform=trans,
                              color='0.75', lw=0.5, zorder=-1))

    fig.savefig(FIG_FN, dpi=200)
    print(f'Wrote {FIG_FN}', flush=True)


if __name__ == '__main__':
    plot(score_runs())
