"""
Reads a wet_layers.sh run (one glacier at its best kp/wind_factor, gridded
over lapse rate, storing layer water) and answers three questions:

1. Where is the liquid water in the accumulation zone through fall and
   winter -- near the surface, or buried under dry snow?
2. Does the wet extent score improve if only near-surface water counts
   as wet, i.e. is the column total_water the wrong thing to compare to
   SAR?
3. On the bare-ice terminus in winter, is the water coming from rain or
   melt under too-warm air, and does a shallower lapse rate remove it?

@author: clairevwilson
"""
import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from project.glacierwide_loss import Wet

OUTPUT_FP = '/ocean/projects/ees260009p/cwilson4/Output/'
SURFACE_DEPTHS = [0.1, 0.5, 1.0]
WET_LAYER = 0.1
TERMINUS_ELEV = 1000


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('glaciers', nargs='+')
    p.add_argument('--start', default='2015-10-01')
    p.add_argument('--end', default='2024-09-30')
    return p.parse_args()


def find_run(glacier):
    runs = sorted(glob.glob(os.path.join(OUTPUT_FP, f'wet_layers_{glacier}*', 'loss_surface_grid.npz')),
                  key=os.path.getmtime)
    assert runs, f'no finished wet_layers run for {glacier}'
    return os.path.dirname(runs[-1])


def water_profile(block):
    """
    Per (time, point): water-weighted mean depth of liquid water, depth of
    the shallowest wet layer, and how much water sits above each of
    SURFACE_DEPTHS.
    """
    h = block.layerheight.values
    w = np.nan_to_num(block.layerwater.values)
    bottom = np.cumsum(h, axis=-1)
    mid = bottom - h / 2
    total = w.sum(-1)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean_depth = np.where(total > 0, (w * mid).sum(-1) / total, np.nan)
    wet = w > WET_LAYER
    top_wet = np.where(wet.any(-1), np.take_along_axis(mid, wet.argmax(-1)[..., None], -1)[..., 0], np.nan)
    # a layer straddling the cutoff contributes the fraction of it above
    above = {d: (w * np.clip((d - (bottom - h)) / np.where(h > 0, h, 1), 0, 1)).sum(-1)
             for d in SURFACE_DEPTHS}
    return total, mean_depth, top_wet, above


def wet_scores(wet, block, water):
    sub = block[['lon', 'lat', 'total_water']].copy()
    sub['total_water'] = block.total_water.copy(data=water)
    wet.get_model_wet(sub)
    ok = ~np.isnan(wet.meas_wet)
    obs = wet.meas_wet.astype(bool) & ok
    mod = wet.mod_wet
    month = pd.DatetimeIndex(wet.time).month
    fall = np.isin(month, [10, 11, 12])[:, None] & ok
    return dict(brier=wet.brier_loss(), mod_wet=mod[ok].mean(),
                false_wet=(mod & ~obs)[ok].sum() / (~obs & ok).sum(),
                missed_wet=(~mod & obs)[ok].sum() / obs.sum(),
                false_wet_OND=(mod & ~obs)[fall].sum() / (~obs & fall).sum())


def main():
    args = parse_args()
    pd.set_option('display.float_format', '{:.2f}'.format)
    pd.set_option('display.width', 220)

    for g in args.glaciers:
        run = find_run(g)
        grid = np.load(os.path.join(run, 'loss_surface_grid.npz'), allow_pickle=True)
        n_base = int(grid['n_base'])
        lapse_rates = grid['lapse_rate']
        ds = xr.open_zarr(os.path.join(run, 'output.zarr'), consolidated=False)
        wet = Wet(g)
        print(f'\n########## {g}: kp={float(grid["kp"][0])} wf={float(grid["wind_factor"][0])} '
              f'({n_base} points) {run} ##########')

        scores = {}
        terminus = {}
        for c, lr in enumerate(lapse_rates):
            block = (ds.isel(point=slice(c * n_base, (c + 1) * n_base))
                     .sel(time=slice(args.start, args.end)).load())
            elev = block.elev.values
            month = block.time.dt.month.values
            sep15 = (month == 9) & (block.time.dt.day.values == 15)
            ice_end_summer = (block.surftype.values[sep15] == 2).mean(0)
            accum = ice_end_summer < 0.5
            term = (elev < TERMINUS_ELEV) & (ice_end_summer > 0.95)

            total, mean_depth, top_wet, above = water_profile(block)
            near = above[0.5]

            print(f'\n--- lapse rate {lr} K/km ---')
            print(f'storage error check: max |sum(layerwater) - total_water| '
                  f'{np.nanmax(np.abs(total - block.total_water.values)):.1f} kg m-2 '
                  f'(daily mean vs daily min, so not expected to be 0)')

            # 1. accumulation zone, by month
            rows = []
            for m in [9, 10, 11, 12, 1, 2, 3, 4]:
                t = month == m
                sel = np.zeros_like(total, dtype=bool)
                sel[t] = True
                sel &= accum[None, :]
                wet_cols = sel & (total > 2)
                rows.append(dict(
                    month=m,
                    col_wet=(total[sel] > 2).mean(),
                    top50_wet=(near[sel] > 2).mean(),
                    water_mm=np.median(total[wet_cols]) if wet_cols.any() else np.nan,
                    mean_depth_m=np.nanmedian(mean_depth[wet_cols]) if wet_cols.any() else np.nan,
                    top_of_wet_m=np.nanmedian(top_wet[wet_cols]) if wet_cols.any() else np.nan,
                    **{f'frac_top{d}m': (above[d][wet_cols].sum() / total[wet_cols].sum())
                       if wet_cols.any() else np.nan for d in SURFACE_DEPTHS},
                    T_top50cm=np.nanmean(np.where(
                        np.cumsum(block.layerheight.values, -1) <= 0.5,
                        block.layertemp.values, np.nan)[sel])))
            print(f'Accumulation zone ({accum.sum()} points, ice exposed at end of summer '
                  f'in <50% of years), wet columns only for depth/water:')
            print(pd.DataFrame(rows).set_index('month').to_string())

            # 2. wet extent scored on column vs near-surface water
            scores[lr] = {'column': wet_scores(wet, block, block.total_water.values)}
            for d in SURFACE_DEPTHS:
                scores[lr][f'top {d} m'] = wet_scores(wet, block, above[d])

            # 3. terminus winter
            if term.any():
                jfm = np.isin(month, [1, 2, 3])
                ond = np.isin(month, [10, 11, 12])
                n_winters = len(np.unique(block.time.dt.year.values[jfm]))
                bt = block.isel(point=term)
                terminus[lr] = dict(
                    n=int(term.sum()),
                    elev=elev[term].mean(),
                    T_air_JFM=bt.airtemp.values[jfm].mean(),
                    days_above0_JFM=(bt.airtemp.values[jfm] > 0).mean(),
                    T_surf_JFM=bt.surftemp.values[jfm].mean(),
                    rain_JFM_mm=1000 * bt.rainfall.values[jfm].sum(0).mean() / n_winters,
                    melt_JFM_mm=1000 * bt.melt.values[jfm].sum(0).mean() / n_winters,
                    refreeze_JFM_mm=1000 * bt.refreeze.values[jfm].sum(0).mean() / n_winters,
                    runoff_JFM_mm=1000 * bt.runoff.values[jfm].sum(0).mean() / n_winters,
                    rain_OND_mm=1000 * bt.rainfall.values[ond].sum(0).mean() / n_winters,
                    water_JFM_mm=np.median(total[:, term][jfm]),
                    wet_JFM=(total[:, term][jfm] > 2).mean(),
                    water_depth_JFM_m=np.nanmedian(mean_depth[:, term][jfm]),
                    T_wet_layers_JFM=np.nanmean(np.where(
                        np.nan_to_num(bt.layerwater.values[jfm]) > WET_LAYER,
                        bt.layertemp.values[jfm], np.nan)))

        print(f'\n===== {g}: wet extent score, column vs near-surface water =====')
        print(pd.DataFrame({(lr, k): v for lr, s in scores.items() for k, v in s.items()}).T.to_string())
        if terminus:
            print(f'\n===== {g}: bare-ice terminus (<{TERMINUS_ELEV} m), per-winter means =====')
            print(pd.DataFrame(terminus).T.to_string())


if __name__ == '__main__':
    main()
