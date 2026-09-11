"""
Runs PEBSI at a sweep of point counts (method_distribute='grid') for one
glacier and reports how glacier-wide mass balance changes with n_points --
i.e. how many points are actually needed before adding more stops changing
the answer. Run once for a small glacier (gulkana) and once for a big one
(kennicott) to see how the point count needed to converge scales with
glacier area.

Reports both the plain point mean and the area-weighted mean (each point
weighted by its own grid cell, clipped to the glacier polygon -- see
Terrain.weight_n in pebsi/io/terrain.py), so the two can be compared
directly instead of assuming a uniform point mean is representative.

Each run's output.zarr is kept under OUTDIR (not deleted after the MB
is computed), so a later re-analysis with a different metric doesn't
require re-simulating everything.

Usage:
    python point_density_convergence.py gulkana
    python point_density_convergence.py kennicott
    python point_density_convergence.py gulkana --n-points 120 140 160 180

@author: clairevwilson
"""
import argparse
import glob
import os
import shutil

import numpy as np
import pandas as pd
import xarray as xr
import yaml

import simulation as sim
from AD_optimize import BASE_CONFIG, HOST_PATHS, host, baseline
from project.bayes_calibrate import align_end_date_for_daily_output
from project.glacierwide_loss import translate_rgi

OUTDIR = os.path.normpath(os.path.join(HOST_PATHS[host]['output_fp'], '..', 'point_density_test')) + '/'
RESULTS_DIR = 'project/point_density_results/'

DEFAULT_N_POINTS = [5, 10, 20, 35, 50, 75, 100, 150, 200, 300, 400, 450, 500,
                     550, 600, 670, 700, 750, 800, 850, 900, 950, 1000]

START_DATE = '2015-04-01 00:00'
END_DATE_RAW = '2018-03-29 23:00'


def run_one(glacier, rgi_id, n_points, wind_factor):
    """Runs one grid-point sim, returns (actual_n_points, area_km2, mb_unweighted, mb_weighted)."""
    end_date = align_end_date_for_daily_output(START_DATE, END_DATE_RAW)

    run_output_fp = os.path.join(OUTDIR, f'{glacier}_n{n_points}_wf{wind_factor}')
    for old in glob.glob(run_output_fp + '_*'):
        shutil.rmtree(old)

    configs = dict(BASE_CONFIG)
    configs.update(HOST_PATHS[host])
    configs['temporal_chunk_years'] = 1
    configs['start_date'] = START_DATE
    configs['end_date'] = end_date
    configs['rgi_ids'] = [rgi_id]
    configs['method_distribute'] = 'grid'
    configs['n_points'] = n_points
    configs['kp'] = baseline['kp']
    configs['wind_factor'] = wind_factor
    configs['store_data'] = True
    configs['store_vars'] = ['mass_balance']
    configs['output_fp'] = run_output_fp

    tmp_config_fn = f'_point_density_{glacier}_{n_points}.yaml'
    with open(tmp_config_fn, 'w') as f:
        yaml.dump(configs, f, sort_keys=False)

    args = sim.get_args(parse=False).parse_args([])
    args.config_fn = tmp_config_fn
    model = sim.PEBSI(args)
    model.run()
    os.remove(tmp_config_fn)

    rgi_row = model.terrain.rgi_df.loc[model.terrain.rgi_df['RGIId'] == 'RGI60-' + rgi_id]
    area_km2 = rgi_row['Area'].item()
    weight_n = model.terrain.weight_n

    out_dirs = sorted(glob.glob(run_output_fp + '_*'))
    assert out_dirs, f'No output directory found for {run_output_fp}'
    ds = xr.open_zarr(os.path.join(out_dirs[-1], 'output.zarr'))

    mb_per_point = ds.mass_balance.sum(dim='time').compute()
    actual_n_points = ds.sizes['point']
    mb_unweighted = float(mb_per_point.mean())
    mb_weighted = float((mb_per_point.values * weight_n).sum())

    ds.close()
    return actual_n_points, area_km2, mb_unweighted, mb_weighted


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('glacier', choices=sorted(translate_rgi.keys()))
    parser.add_argument('--n-points', type=int, nargs='+', default=None)
    parser.add_argument('--wind-factor', type=float, default=baseline['wind_factor'])
    args = parser.parse_args()

    glacier = args.glacier
    rgi_id = translate_rgi[glacier]['6']
    wind_factor = args.wind_factor

    n_points_list = args.n_points if args.n_points is not None else DEFAULT_N_POINTS

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_csv = os.path.join(RESULTS_DIR, f'{glacier}_point_density.csv')

    rows = []
    for n in sorted(set(n_points_list)):
        actual_n, area_km2, mb_unweighted, mb_weighted = run_one(glacier, rgi_id, n, wind_factor)
        print(f'requested n_points={n:>5}  actual={actual_n:>5}  '
              f'unweighted MB={mb_unweighted:+.4f}  weighted MB={mb_weighted:+.4f} m w.e.')
        rows.append({
            'glacier': glacier,
            'area_km2': area_km2,
            'wind_factor': wind_factor,
            'requested_n_points': n,
            'actual_n_points': actual_n,
            'mass_balance': mb_weighted,
            'mass_balance_unweighted': mb_unweighted,
        })

    df = pd.DataFrame(rows).sort_values('requested_n_points').reset_index(drop=True)

    reference_mb = df.iloc[-1]['mass_balance']
    df['mb_diff_from_densest'] = df['mass_balance'] - reference_mb

    df.to_csv(out_csv, index=False)

    print()
    print(df[['requested_n_points', 'actual_n_points', 'mass_balance',
              'mass_balance_unweighted', 'mb_diff_from_densest']].to_string(index=False))
    print(f'\nGlacier area: {df["area_km2"].iloc[0]:.1f} km2')
    print(f'Saved results to {out_csv}')


if __name__ == '__main__':
    main()
