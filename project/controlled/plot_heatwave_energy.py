"""
Where the heatwave melt energy comes from, in the MERRA-2 deposition controlled runs (Base minus No
heatwaves): the May-Sep mean change in each surface flux [W m-2, glacier mean], with net shortwave split
into
  more sunlight    the change in incoming shortwave, absorbed at the No-heatwaves albedo
  snow lost early  lower albedo on point-days snow-covered without heatwaves but bare with them
  darker snow      lower albedo on point-days snow-covered in both runs
  other SW         the rest (point-days snow-covered only with heatwaves, and bare in both)
Snow cover is from noon albedo as in plot_fire_albedo.py. The per-glacier fluxes are per_glacier.nc;
the shortwave split uses each run's per_point_daily.nc. Every year goes to CSV_FN; the figure is the
2004 and 2019 seasons side by side as May-Sep totals [MJ m-2], without ground heat and other SW (both about
zero).
"""
import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__))
from plot_fire_albedo import snow_mask  # also puts project/synthetic on the path
from plot_event_grid import GLACIERS, translate_rgi

OUT = '/home/claire/Research/Output/controlled/'
RUNS = {'b': 'base_merra2', 'n': 'no_heatwaves_merra2'}
CSV_FN = OUT + 'heatwave_energy_partition.csv'
FIG_FN = OUT + 'figs/heatwave_energy_2004_2019.png'
CASE_YEARS = [2004, 2019]
FLUXES = {'sensible_heat': 'Sensible', 'latent_heat': 'Latent', 'longwave_net': 'Net longwave',
          'rain_heat': 'Rain', 'ground_heat': 'Ground'}
SW_PARTS = {'sw_sun': 'Net shortwave:\nmore sunlight', 'sw_snowloss': 'Net shortwave:\nsnow lost early',
            'sw_darker': 'Net shortwave:\ndarker snow', 'sw_other': 'Net shortwave: other'}
# the shared palette (shop/plotting/plotting_fxns.py all_colors)
COLORS = {'sensible_heat': '#4D559C', 'latent_heat': '#63c4c7', 'longwave_net': '#60C252', 'rain_heat': '#FF89B0',
          'sw_sun': '#fcc02e', 'sw_snowloss': '#F77808', 'sw_darker': '#BF1F6A'}


def partition():
    """May-Sep mean flux changes [W m-2] per glacier and year."""
    glac = {k: xr.open_dataset(OUT + f'{r}_0/per_glacier.nc') for k, r in RUNS.items()}
    pts = {k: xr.open_dataset(OUT + f'{r}_0/per_point_daily.nc').load() for k, r in RUNS.items()}
    sw_in = {k: p['shortwave_in'].values for k, p in pts.items()}
    alb = {k: 1 - p['shortwave_net'].values / np.where(sw_in[k] > 0, sw_in[k], np.nan) for k, p in pts.items()}
    alb = {k: np.nan_to_num(a) for k, a in alb.items()}
    snow_b, snow_n = snow_mask(pts['b']['albedo'].values, pts['n']['albedo'].values)
    # absorbed shortwave change = more sunlight at the No-heatwaves albedo - Base sunlight x albedo change
    sun = (sw_in['b'] - sw_in['n']) * (1 - alb['n'])
    darken = -sw_in['b'] * (alb['b'] - alb['n'])
    total_sw = pts['b']['shortwave_net'].values - pts['n']['shortwave_net'].values
    years = pts['b'].time.dt.year.values
    rgiid, weight = pts['b']['rgiid'].values.astype(str), pts['b']['weight'].values

    rows = []
    for g in GLACIERS:
        m = rgiid == translate_rgi[g]['6']
        for y in range(2000, 2026):
            t = years == y
            n_hours = t.sum() * 24  # daily values are sums of hourly W m-2
            def mean(x, mask=None):
                x = x[t][:, m] if mask is None else np.where(mask[t][:, m], x[t][:, m], 0)
                return float((x * weight[m]).sum()) / n_hours
            row = dict(glacier=g, year=y, sw_sun=mean(sun),
                       sw_snowloss=mean(darken, snow_n & ~snow_b), sw_darker=mean(darken, snow_n & snow_b))
            row['sw_other'] = mean(total_sw) - row['sw_sun'] - row['sw_snowloss'] - row['sw_darker']
            s = slice(f'{y}-05-01', f'{y}-09-30')
            for f in FLUXES:
                row[f] = float((glac['b'][f] - glac['n'][f]).sel(glacier=g, time=s).mean())
            row['melt_energy'] = float((glac['b']['melt_energy'] - glac['n']['melt_energy']).sel(glacier=g, time=s).mean())
            row['shortwave_net'] = float((glac['b']['shortwave_net'] - glac['n']['shortwave_net']).sel(glacier=g, time=s).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    df = partition()
    df.to_csv(CSV_FN, index=False)
    print(f'Saved {CSV_FN}')
    # the point-day split should add up to the glacier-level net shortwave change
    split = df[list(SW_PARTS)].sum(axis=1)
    print('max |SW split - per_glacier SW| [W m-2]:', round(float((split - df['shortwave_net']).abs().max()), 3))
    cols = list(FLUXES) + list(SW_PARTS) + ['melt_energy']
    pd.set_option('display.width', 220)
    print(df[df['year'].isin(CASE_YEARS)].set_index(['year', 'glacier'])[cols].round(1).to_string())
    print('\nall-years mean\n', df.groupby('glacier')[cols].mean().round(1).to_string())

    # stacked from the bottom; ground heat and the leftover shortwave are about zero, so only the CSV has them
    parts = ['sensible_heat', 'latent_heat', 'sw_sun', 'sw_snowloss', 'sw_darker', 'rain_heat', 'longwave_net']
    labels = {**FLUXES, **SW_PARTS}
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4), sharey=True)
    for ax, y in zip(axes, CASE_YEARS):
        # May-Sep totals [MJ m-2]: the seasonal mean flux times the season's length
        n_seconds = (pd.Timestamp(f'{y}-10-01') - pd.Timestamp(f'{y}-05-01')).total_seconds()
        d = df[df['year'] == y].set_index('glacier').loc[GLACIERS][parts] * n_seconds / 1e6
        x = np.arange(len(GLACIERS))
        up, down = np.zeros(len(x)), np.zeros(len(x))
        for p in parts:
            v = d[p].values
            ax.bar(x, np.where(v > 0, v, 0), 0.6, bottom=up, color=COLORS[p], label=labels[p])
            ax.bar(x, np.where(v < 0, v, 0), 0.6, bottom=down, color=COLORS[p])
            up += np.where(v > 0, v, 0)
            down += np.where(v < 0, v, 0)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([g.replace('_', ' ').title() for g in GLACIERS], rotation=45, ha='right', fontsize=9)
        ax.set_title(str(y), fontsize=9)
        ax.tick_params(labelsize=9)
    axes[0].set_ylim(top=1.1 * axes[0].get_ylim()[1])
    axes[0].set_ylabel('Change in energy fluxes from heatwaves (MJ m$^{-2}$)', fontsize=9)
    handles, labs = axes[0].get_legend_handles_labels()
    # top to bottom in the same order as the bars
    fig.legend(handles[::-1], labs[::-1], loc='center left', bbox_to_anchor=(0.79, 0.5), fontsize=9, frameon=False)
    plt.tight_layout(rect=(0, 0, 0.8, 1))
    plt.savefig(FIG_FN, dpi=300)
    print(f'Saved {FIG_FN}')


if __name__ == '__main__':
    main()
