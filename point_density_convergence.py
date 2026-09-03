"""
Runs PEBSI at a sweep of point counts (method_distribute='scatter') for one
glacier and reports how glacier-wide mass balance changes with n_points --
i.e. how many scatter points are actually needed before adding more stops
changing the answer. Run once for a small glacier (gulkana) and once for a
big one (kennicott) to see how the point count needed to converge scales
with glacier area.

Rerunning with a different --n-points appends to the existing results CSV
instead of overwriting it -- already-run point counts are skipped, so you
can add new counts (e.g. to zoom in near a suspected level-off point)
without re-simulating everything.

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
import sys

import pandas as pd
import xarray as xr
import yaml

import simulation as sim
from project.bayes_calibrate import BASE_CONFIG, baseline, align_end_date_for_daily_output
from project.glacierwide_loss import translate_rgi

OUTDIR = '/ocean/projects/ees260009p/cwilson4/Output/point_density_test/'
RESULTS_DIR = 'project/point_density_results/'

DEFAULT_N_POINTS = [5, 10, 20, 35, 50, 75, 100, 150, 200, 300, 500, 750, 1000]

START_DATE = '2015-04-01 00:00'
END_DATE_RAW = '2018-03-29 23:00'


def run_one(glacier, rgi_id, n_points):
    """Runs one scatter-point sim and returns (actual_n_points, glacier_area_km2, glacier_wide_mb)."""
    end_date = align_end_date_for_daily_output(START_DATE, END_DATE_RAW)

    run_output_fp = os.path.join(OUTDIR, f'{glacier}_n{n_points}')
    for old in glob.glob(run_output_fp + '_*'):
        shutil.rmtree(old)

    configs = dict(BASE_CONFIG)
    configs['temporal_chunk_years'] = 1
    configs['start_date'] = START_DATE
    configs['end_date'] = end_date
    configs['rgi_ids'] = [rgi_id]
    configs['n_points'] = n_points
    configs['kp'] = baseline['kp']
    configs['wind_factor'] = baseline['wind_factor']
    configs['output_fp'] = run_output_fp

    tmp_config_fn = f'_point_density_{glacier}_{n_points}.yaml'
    with open(tmp_config_fn, 'w') as f:
        yaml.dump(configs, f, sort_keys=False)

    # sim.get_args() parses sys.argv itself -- clear it first so it doesn't
    # trip over this script's own CLI args (e.g. the glacier name, --n-points)
    real_argv, sys.argv = sys.argv, sys.argv[:1]
    args = sim.get_args()
    sys.argv = real_argv
    args.config_fn = tmp_config_fn
    model = sim.PEBSI(args)
    model.run()
    os.remove(tmp_config_fn)

    rgi_row = model.terrain.rgi_df.loc[model.terrain.rgi_df['RGIId'] == 'RGI60-' + rgi_id]
    area_km2 = rgi_row['Area'].item()

    out_dirs = sorted(glob.glob(run_output_fp + '_*'))
    assert out_dirs, f'No output directory found for {run_output_fp}'
    ds = xr.open_zarr(os.path.join(out_dirs[-1], 'output.zarr'))

    actual_n_points = ds.sizes['point']
    mb = ds.mass_balance.sum(dim='time').mean(dim='point').compute().item()

    ds.close()
    shutil.rmtree(out_dirs[-1])
    return actual_n_points, area_km2, mb


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('glacier', choices=sorted(translate_rgi.keys()))
    parser.add_argument('--n-points', type=int, nargs='+', default=None)
    args = parser.parse_args()

    glacier = args.glacier
    rgi_id = translate_rgi[glacier]['6']

    if args.n_points is not None:
        n_points_list = args.n_points
    elif glacier == 'kennicott':
        n_points_list = [800, 850, 900, 950]
    elif glacier == 'gulkana':
        n_points_list = [150, 175, 200, 225, 250, 275, 300]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_csv = os.path.join(RESULTS_DIR, f'{glacier}_point_density.csv')

    # reuse whatever's already been run for this glacier 
    if os.path.exists(out_csv):
        df_old = pd.read_csv(out_csv)
        already_run = set(df_old['requested_n_points'])
    else:
        df_old = pd.DataFrame()
        already_run = set()

    rows = []
    for n in sorted(set(n_points_list) - already_run):
        actual_n, area_km2, mb = run_one(glacier, rgi_id, n)
        print(f'requested n_points={n:>5}  actual={actual_n:>5}  '
              f'glacier-wide MB={mb:+.4f} m w.e.')
        rows.append({
            'glacier': glacier,
            'area_km2': area_km2,
            'requested_n_points': n,
            'actual_n_points': actual_n,
            'mass_balance': mb,
        })

    df = pd.concat([df_old[['glacier', 'area_km2', 'requested_n_points',
                             'actual_n_points', 'mass_balance']], pd.DataFrame(rows)],
                    ignore_index=True) if rows or not df_old.empty else pd.DataFrame(rows)
    df = df.sort_values('requested_n_points').reset_index(drop=True)

    reference_mb = df.iloc[-1]['mass_balance']
    df['mb_diff_from_densest'] = df['mass_balance'] - reference_mb

    df.to_csv(out_csv, index=False)

    print()
    print(df[['requested_n_points', 'actual_n_points', 'mass_balance', 'mb_diff_from_densest']].to_string(index=False))
    print(f'\nGlacier area: {df["area_km2"].iloc[0]:.1f} km2')
    print(f'Saved results to {out_csv}')


if __name__ == '__main__':
    main()
