"""
For one glacier of a loss surface run, shows how much of the albedo
comparison (every matched point x satellite date, Mar-Sep) falls on
modeled snow vs exposed firn/ice in each (kp, wind_factor) combination,
alongside the albedo loss and the mean modeled/measured albedo change
from March.

Only the matched point-dates where the model surface changes type can
respond to kp/wind_factor; where the model stays snow, more snow leaves
the albedo trajectory unchanged.

@author: clairevwilson
"""
import argparse
import os
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from project.glacierwide_loss import Albedo, ALBEDO_SIGMA
from project.parameters import translate_rgi

RUN_DIR = '/ocean/projects/ees260009p/cwilson4/Output/loss_surface_0/'


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('glacier')
    p.add_argument('--run-dir', default=RUN_DIR)
    p.add_argument('--wind-factor', type=float, nargs='+', default=[2.5],
                   help='wind_factor columns to show (default %(default)s)')
    p.add_argument('--months', type=int, nargs='+', default=list(range(3, 10)),
                   help='months compared (default %(default)s)')
    p.add_argument('--ice-albedo', type=float, default=0.35,
                   help='measured albedo below this counts as bare ice (default %(default)s)')
    return p.parse_args()


def main():
    args = parse_args()
    grid = np.load(os.path.join(args.run_dir, 'loss_surface_grid.npz'), allow_pickle=True)
    kp_values, wf_values = grid['kp'], grid['wind_factor']
    n_base = int(grid['n_base'])

    ds = xr.open_zarr(os.path.join(args.run_dir, 'output.zarr'), consolidated=False)
    gid = translate_rgi[args.glacier]['6']
    gmask = ds['rgiid'].values[:n_base].astype(str) == gid

    alb = Albedo(args.glacier, use='s2')
    print(f'{args.glacier}, months {args.months}: kp, wind_factor | albedo loss | matched point-dates on modeled '
          f'snow / firn / ice | measured ice (albedo < {args.ice_albedo}) | mean albedo mod / meas', flush=True)
    for wf in args.wind_factor:
        j = int(np.argmin(np.abs(wf_values - wf)))
        for i, kp in enumerate(kp_values):
            c = i * len(wf_values) + j
            pts = np.arange(c * n_base, (c + 1) * n_base)[gmask]
            sub = ds[['albedo', 'surftype', 'lon', 'lat', 'rgiid']].isel(point=pts)

            alb.get_model_albedo(sub, months_range=args.months)
            surf = sub['surftype'].sel(time=alb.ds_meas_matched.time, method='nearest').values
            has_obs = np.isfinite(alb.meas)
            meas_ice = np.mean(alb.meas[has_obs] < args.ice_albedo)
            mean_mod, mean_meas = np.nanmean(alb.mod[has_obs]), np.nanmean(alb.meas[has_obs])
            alb.get_deltas(method='march_mean')
            loss = alb.crps_loss(sigma=ALBEDO_SIGMA)

            s = surf[has_obs]
            frac = [np.mean(s == k) for k in (0, 1, 2)]
            print(f'  kp={kp:.1f} wf={wf_values[j]:.1f} | {loss:.3f} | '
                  f'{frac[0]:.0%} / {frac[1]:.0%} / {frac[2]:.0%} | {meas_ice:.0%} | '
                  f'{mean_mod:.3f} / {mean_meas:.3f}', flush=True)


if __name__ == '__main__':
    main()
