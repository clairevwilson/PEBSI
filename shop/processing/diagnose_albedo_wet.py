"""
Breaks down where each glacier's best loss-surface run misses the
remotely sensed albedo and SAR wet extent.

Albedo (Sentinel-2, Mar-Sep, as scored): by month, the observed and
modeled change from March and each month's share of the CRPS loss;
by modeled surface type, the raw observed and modeled albedo; and, on
modeled summer snow, how much is observed as ice-dark (a surface-type
mismatch) and how biased the rest is.

Wet extent (SAR): by month and by elevation third, the observed and
modeled wet fractions, the missed-wet and false-wet rates, and the
Brier score; and the same split by modeled surface type.

@author: clairevwilson
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from project.glacierwide_loss import Albedo, Wet, ALBEDO_SIGMA, crps_gaussian
from project.parameters import translate_names

BEST_FN = '/ocean/projects/ees260009p/cwilson4/Output/loss_surface_0/best.zarr'
SURFACE_NAMES = {0: 'snow', 1: 'firn', 2: 'ice'}
ICE_DARK = 0.4


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--best-fn', default=BEST_FN,
                   help='best.zarr, or any run output.zarr (default %(default)s)')
    p.add_argument('--glaciers', nargs='+', default=None)
    return p.parse_args()


def albedo_tables(g, sub):
    alb = Albedo(g, use='s2')
    alb.get_model_albedo(sub)
    times = pd.DatetimeIndex(alb.ds_meas_matched.time.values)
    surf = sub['surftype'].sel(time=alb.ds_meas_matched.time, method='nearest').values
    raw_mod, raw_meas = alb.mod.copy(), alb.meas.copy()
    alb.get_deltas(method='march_mean')
    ok = np.isfinite(alb.meas) & np.isfinite(alb.mod)
    crps = np.where(ok, crps_gaussian(alb.mod, alb.meas, ALBEDO_SIGMA) / ALBEDO_SIGMA, np.nan)
    month = np.broadcast_to(times.month.values[:, None], ok.shape)

    by_month = pd.DataFrame({
        m: dict(n=int(ok[month == m].sum()),
                share_of_loss=np.nansum(crps[month == m]) / np.nansum(crps),
                loss=np.nanmean(crps[month == m]),
                dalb_meas=np.nanmean(alb.meas[(month == m) & ok]),
                dalb_mod=np.nanmean(alb.mod[(month == m) & ok]))
        for m in range(3, 10)}).T

    by_surface = pd.DataFrame({
        f'{SURFACE_NAMES[k]} {mlab}': dict(
            n=int((ok & (surf == k) & mmask).sum()),
            alb_meas=np.nanmean(raw_meas[ok & (surf == k) & mmask]),
            alb_mod=np.nanmean(raw_mod[ok & (surf == k) & mmask]),
            loss=np.nanmean(crps[ok & (surf == k) & mmask]))
        for k in (0, 1, 2)
        for mlab, mmask in (('Mar-Jun', month <= 6), ('Jul-Sep', month >= 7))
        if (ok & (surf == k) & mmask).any()}).T

    summer_snow = ok & (surf == 0) & (month >= 7)
    dark = summer_snow & (raw_meas < ICE_DARK)
    bright = summer_snow & (raw_meas >= ICE_DARK)
    snow_check = pd.Series(dict(
        n=int(summer_snow.sum()),
        share_obs_dark=dark.sum() / summer_snow.sum(),
        share_of_snow_loss_from_dark=np.nansum(crps[dark]) / np.nansum(crps[summer_snow]),
        bias_all=np.mean(raw_mod[summer_snow] - raw_meas[summer_snow]),
        bias_obs_bright_only=np.mean(raw_mod[bright] - raw_meas[bright]),
        alb_meas_bright_only=np.mean(raw_meas[bright]),
        alb_mod_bright_only=np.mean(raw_mod[bright])))

    # surface-type mismatch in both directions, Jul-Sep, by elevation third
    elev = sub['elev'].values
    third = np.broadcast_to(np.digitize(elev, np.quantile(elev, [1 / 3, 2 / 3]))[None, :], ok.shape)
    edges = np.quantile(elev, [0, 1 / 3, 2 / 3, 1])
    summer = ok & (month >= 7)
    mismatch = pd.DataFrame({
        f'{edges[b]:.0f}-{edges[b + 1]:.0f} m': dict(
            n=int((summer & (third == b)).sum()),
            mod_snow_obs_dark=np.mean(((surf == 0) & (raw_meas < ICE_DARK))[summer & (third == b)]),
            mod_ice_obs_bright=np.mean(((surf == 2) & (raw_meas >= 0.5))[summer & (third == b)]),
            mod_snow=np.mean((surf == 0)[summer & (third == b)]),
            obs_dark=np.mean((raw_meas < ICE_DARK)[summer & (third == b)]))
        for b in range(3)}).T
    return by_month, by_surface, snow_check, mismatch


def wet_tables(g, sub, elev):
    wet = Wet(g)
    wet.get_model_wet(sub)
    meas = wet.meas_wet
    mod = wet.mod_wet.astype(float)
    ok = np.isfinite(meas)
    brier = np.where(ok, (wet.p_wet - meas) ** 2, np.nan)
    times = pd.DatetimeIndex(wet.time)
    month = np.broadcast_to(times.month.values[:, None], ok.shape)
    surf = sub['surftype'].reindex(time=wet.time, method='nearest').values
    third = np.broadcast_to(np.digitize(elev, np.quantile(elev, [1 / 3, 2 / 3]))[None, :], ok.shape)
    edges = np.quantile(elev, [0, 1 / 3, 2 / 3, 1])

    def summarize(mask):
        m = ok & mask
        return dict(n=int(m.sum()),
                    wet_meas=np.mean(meas[m] == 1), wet_mod=np.mean(mod[m] == 1),
                    missed_wet=np.mean((meas[m] == 1) & (mod[m] == 0)),
                    false_wet=np.mean((meas[m] == 0) & (mod[m] == 1)),
                    brier=np.nanmean(brier[m]),
                    share_of_loss=np.nansum(brier[m]) / np.nansum(brier))

    by_month = pd.DataFrame({m: summarize(month == m) for m in range(1, 13)
                             if (ok & (month == m)).any()}).T
    by_elev = pd.DataFrame({f'{edges[b]:.0f}-{edges[b + 1]:.0f} m': summarize(third == b)
                            for b in range(3)}).T
    by_surface = pd.DataFrame({SURFACE_NAMES[k]: summarize(surf == k)
                               for k in (0, 1, 2) if (ok & (surf == k)).any()}).T
    return by_month, by_elev, by_surface


def main():
    args = parse_args()
    ds = xr.open_zarr(args.best_fn, consolidated=False)
    rgiid = ds['rgiid'].values.astype(str)
    glaciers = args.glaciers or [translate_names[r] for r in pd.unique(rgiid)]

    fmt = {'display.width': 200, 'display.max_columns': None,
           'display.float_format': '{:.3f}'.format}
    for g in glaciers:
        pts = np.where(rgiid == [r for r in pd.unique(rgiid) if translate_names[r] == g][0])[0]
        sub = ds.isel(point=pts)
        params = (f' (kp={float(sub.kp[0]):.1f}, wind_factor={float(sub.wind_factor[0]):.1f})'
                  if 'kp' in sub else '')
        print(f'\n===== {g}{params} =====', flush=True)
        a_month, a_surf, a_snow, a_mismatch = albedo_tables(g, sub)
        w_month, w_elev, w_surf = wet_tables(g, sub, ds['elev'].values[pts])
        with pd.option_context(*[x for kv in fmt.items() for x in kv]):
            print('\nAlbedo by month (dalb = change since March)')
            print(a_month)
            print('\nAlbedo by modeled surface type (raw albedo)')
            print(a_surf)
            print(f'\nModeled Jul-Sep snow: observed albedo < {ICE_DARK} counted as a surface mismatch')
            print(a_snow.to_string())
            print(f'\nJul-Sep surface mismatch by elevation (obs dark = albedo < {ICE_DARK}, bright = >= 0.5)')
            print(a_mismatch)
            print('\nWet extent by month (fractions of point-dates)')
            print(w_month)
            print('\nWet extent by elevation third')
            print(w_elev)
            print('\nWet extent by modeled surface type')
            print(w_surf)


if __name__ == '__main__':
    main()
