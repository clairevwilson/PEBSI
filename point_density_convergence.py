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

# element edge lengths [m] for the patch-conforming mesh. Unlike a point
# count these mean the same spatial resolution on every glacier, so the
# same sweep is comparable between a small glacier and a big one.
DEFAULT_SPACINGS = [1200, 1000, 800, 700, 600, 500, 450, 400, 350, 300, 250, 200]

START_DATE = '2015-04-01 00:00'
END_DATE_RAW = '2018-03-29 23:00'


def run_one(glacier, rgi_id, wind_factor, n_points=None, spacing=None,
           precomputed_costheta=False):
    """
    Runs one sim, returns (actual_n_points, area_km2, mb_unweighted, mb_weighted).

    Pass n_points to place points on the clipped lattice ('grid'), or
    spacing to mesh the glacier into triangular elements of that edge
    length ('mesh') and use one point per element.

    precomputed_costheta sets option_precomputed_costheta: cos(solar
    incidence) is cell-averaged from per-pixel slope/aspect (cached in
    the shading .zarr) rather than computed once from the point's own
    cell-mean slope/aspect.
    """
    assert (n_points is None) != (spacing is None), \
        'run_one takes exactly one of n_points and spacing'

    end_date = align_end_date_for_daily_output(START_DATE, END_DATE_RAW)

    tag = f'n{n_points}' if spacing is None else f'h{spacing}'
    if precomputed_costheta:
        tag += 'ct'
    run_output_fp = os.path.join(OUTDIR, f'{glacier}_{tag}_wf{wind_factor}')
    for old in glob.glob(run_output_fp + '_*'):
        shutil.rmtree(old)

    configs = dict(BASE_CONFIG)
    configs.update(HOST_PATHS[host])
    configs['temporal_chunk_years'] = 1
    configs['start_date'] = START_DATE
    configs['end_date'] = end_date
    configs['rgi_ids'] = [rgi_id]
    if spacing is None:
        configs['method_distribute'] = 'grid'
        configs['n_points'] = n_points
    else:
        configs['method_distribute'] = 'mesh'
        configs['point_spacing'] = spacing
    configs['option_precomputed_costheta'] = precomputed_costheta
    configs['kp'] = baseline['kp']
    configs['wind_factor'] = wind_factor
    configs['store_data'] = True
    configs['store_vars'] = ['mass_balance']
    configs['output_fp'] = run_output_fp

    tmp_config_fn = f'_point_density_{glacier}_{tag}.yaml'
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
    parser.add_argument('--spacing', type=float, nargs='*', default=None,
                        help='sweep mesh element edge length [m] instead of point count; '
                             'pass with no values to use DEFAULT_SPACINGS')
    parser.add_argument('--wind-factor', type=float, default=baseline['wind_factor'])
    parser.add_argument('--precomputed-costheta', action='store_true',
                        help='enable option_precomputed_costheta for this sweep')
    parser.add_argument('--tag', default='',
                        help='suffix for the results CSV, so two jobs sweeping '
                             'different resolutions of the same glacier do not '
                             'overwrite each other')
    args = parser.parse_args()

    glacier = args.glacier
    rgi_id = translate_rgi[glacier]['6']
    wind_factor = args.wind_factor
    use_mesh = args.spacing is not None

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if use_mesh:
        sweep = sorted(set(args.spacing if args.spacing else DEFAULT_SPACINGS), reverse=True)
        out_csv = os.path.join(RESULTS_DIR, f'{glacier}_mesh_convergence{args.tag}.csv')
        sweep_col = 'point_spacing'
    else:
        sweep = sorted(set(args.n_points if args.n_points is not None else DEFAULT_N_POINTS))
        out_csv = os.path.join(RESULTS_DIR, f'{glacier}_point_density.csv')
        sweep_col = 'requested_n_points'

    rows = []
    for value in sweep:
        kwargs = {'spacing': value} if use_mesh else {'n_points': value}
        actual_n, area_km2, mb_unweighted, mb_weighted = run_one(
            glacier, rgi_id, wind_factor, precomputed_costheta=args.precomputed_costheta,
            **kwargs)
        label = f'h={value:>6.0f} m' if use_mesh else f'n_points={value:>5}'
        print(f'{label}  actual N={actual_n:>6}  '
              f'unweighted MB={mb_unweighted:+.4f}  weighted MB={mb_weighted:+.4f} m w.e.')
        rows.append({
            'glacier': glacier,
            'area_km2': area_km2,
            'wind_factor': wind_factor,
            sweep_col: value,
            'actual_n_points': actual_n,
            'mass_balance': mb_weighted,
            'mass_balance_unweighted': mb_unweighted,
        })

    df = pd.DataFrame(rows)

    # fold in any earlier sweep of the same glacier, so extending a sweep
    # with a few more resolutions does not discard the ones already run.
    # A resolution present in both is taken from this run.
    if os.path.exists(out_csv):
        previous = pd.read_csv(out_csv)
        if sweep_col in previous.columns:
            df = pd.concat([previous, df], ignore_index=True)
            df = df.drop_duplicates(subset=sweep_col, keep='last')

    # sort so the densest mesh is last either way: point count rises with
    # n_points but falls as the element edge length grows
    df = df.sort_values('actual_n_points').reset_index(drop=True)

    # the densest run is not a truth to measure against -- it is just
    # one more sample, and can sit at the edge of the spread itself.
    # The median is the robust centre of the sweep.
    df = df.drop(columns=['mb_diff_from_median'], errors='ignore')
    df['mb_diff_from_median'] = df['mass_balance'] - df['mass_balance'].median()

    df.to_csv(out_csv, index=False)

    print()
    print(df[[sweep_col, 'actual_n_points', 'mass_balance',
              'mass_balance_unweighted', 'mb_diff_from_median']].to_string(index=False))
    print(f'\nGlacier area: {df["area_km2"].iloc[0]:.1f} km2')
    print(f'Saved results to {out_csv}')


if __name__ == '__main__':
    main()
