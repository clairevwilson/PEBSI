"""
Works out how many points each glacier should be run with, from one
survey simulation per glacier.

The error on a glacier-wide mass balance comes from sampling a spatially
varying field at finitely many points, so it is set by how much mass
balance varies across that glacier, rather than by mesh resolution alone. 
Two glaciers of the same area can need very different point counts, and 
the larger of two glaciers is not reliably the one that needs more.

Number of points is thus decided from the variability in mass balance:

    N = (confidence * coefficient * sigma / tolerance) ** 2

with a default 95% confidence interval and the coefficient determined
from empirical data.

This survey runs at a fine spacing such that sigma is converged. 

Sigma does move with the calibration parameters. The survey runs at one
parameter set, so --box-margin inflates sigma to cover a range around
it. On the five glaciers this was mapped on, the largest sigma anywhere
in wind_factor and kp over {1, 2.5, 5} was 1.3 to 2.0 times the sigma at
the middle of that box. Therefore, box-marging of 2.0 would be the most
rigorous mesh the glacier would possibly need within bounds of these parameters.
The method is untested on other parameter combinations.

See convergence_summary.md for where the coefficient comes from.

Usage:
    python point_density_rule.py gulkana
    python point_density_rule.py gulkana kennicott kahiltna --append
    python point_density_rule.py gulkana --survey-spacing 500
    python point_density_rule.py gulkana --box-margin 2.0 --append
    python point_density_rule.py gulkana --tolerance 2 --append

@author: clairevwilson
"""
import argparse
import glob
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
import yaml

import simulation as sim
from project.parameters import BASE_CONFIG, HOST_PATHS, host, translate_rgi
from pebsi import defaults
from pebsi.io import mesh

OUTDIR = os.path.normpath(os.path.join(HOST_PATHS[host]['output_fp'], '..', 'sigma_survey')) + '/'

# daily output requires the inclusive hourly count to be a multiple of
# 24, which a 00:00 start and 23:00 end always satisfies
START_DATE = '2015-04-01 00:00'
END_DATE = '2018-03-29 23:00'
YEARS = 3.0

# sigma is within about 5% of its converged value by this spacing on every
# glacier it has been checked on. Coarser than this and the cell averaging
# smooths the field the survey is trying to measure: at the coarse end of a
# sweep sigma reads up to 24% low, which would recommend too few points.
SURVEY_SPACING = 700.0


def load_outline(rgi_id, rgi_fp):
    """Reads one glacier's outline out of the regional RGI shapefile."""
    region = rgi_id.split('.')[0]
    names = [f.split('.')[0] for f in os.listdir(rgi_fp) if f.startswith(region)]
    assert len(names) == 1, f'Did not find RGI region {region} data'
    gdf = gpd.read_file(os.path.join(rgi_fp, f'../{names[0]}/{names[0]}.shp'))
    return mesh.glacier_polygon(gdf, rgi_id)


def survey_output(glacier, spacing):
    """Path prefix the survey run for this glacier and spacing writes to."""
    return os.path.join(OUTDIR, f'{glacier}_h{spacing}')


def read_sigma(run_fp):
    """
    Reads sigma [cm w.e. a-1] and the point count out of a survey output.

    Sigma is the unweighted standard deviation across points, which is
    what the coefficient in the rule was fit against.
    """
    out_dirs = sorted(glob.glob(run_fp + '_*'))
    if not out_dirs:
        return None, None

    ds = xr.open_zarr(os.path.join(out_dirs[-1], 'output.zarr'))
    mb = ds.mass_balance.sum(dim='time').compute().values / YEARS * 100
    n_points = ds.sizes['point']
    ds.close()

    return float(np.std(mb, ddof=1)), n_points


def run_survey(glacier, rgi_id, spacing):
    """Runs the one simulation this glacier's point count is chosen from."""
    run_fp = survey_output(glacier, spacing)

    configs = dict(BASE_CONFIG)
    configs.update(HOST_PATHS[host])
    configs['temporal_chunk_years'] = 1
    configs['start_date'] = START_DATE
    configs['end_date'] = END_DATE
    configs['rgi_ids'] = [rgi_id]
    configs['method_distribute'] = 'mesh'
    configs['point_spacing'] = spacing
    configs['store_data'] = True
    configs['store_vars'] = ['mass_balance']
    configs['output_fp'] = run_fp

    tmp_config_fn = f'_sigma_survey_{glacier}.yaml'
    with open(tmp_config_fn, 'w') as f:
        yaml.dump(configs, f, sort_keys=False)

    args = sim.get_args(parse=False).parse_args([])
    args.config_fn = tmp_config_fn
    model = sim.PEBSI(args)
    model.run()
    os.remove(tmp_config_fn)

    return read_sigma(run_fp)


def update_table(fn, rows):
    """Adds or replaces these glaciers in the sigma table."""
    fresh = pd.DataFrame(rows)
    if os.path.exists(fn):
        table = pd.read_csv(fn, dtype={'rgiid': str})
        table = table.loc[~table['rgiid'].isin(fresh['rgiid'])]
        table = pd.concat([table, fresh], ignore_index=True)
    else:
        table = fresh
    os.makedirs(os.path.dirname(fn) or '.', exist_ok=True)
    table.sort_values('sigma_max', ascending=False).to_csv(fn, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('glaciers', nargs='+', choices=sorted(translate_rgi.keys()))
    parser.add_argument('--survey-spacing', type=float, default=SURVEY_SPACING,
                        help=f'element edge length for the survey run [m] '
                             f'(default {SURVEY_SPACING:.0f})')
    parser.add_argument('--tolerance', type=float, default=defaults.point_error_tolerance,
                        help='allowed error on glacier-wide mass balance [cm w.e. a-1]')
    parser.add_argument('--coefficient', type=float, default=defaults.point_error_coefficient,
                        help='fitted constant relating sigma to the error of the mean')
    parser.add_argument('--box-margin', type=float, default=1.0,
                        help='multiply sigma by this to cover parameter drift; '
                             '1.3 to 2.0 covered wind_factor and kp over {1, 2.5, 5} '
                             'on the five glaciers it was mapped on (default 1.0)')
    parser.add_argument('--force', action='store_true',
                        help='rerun the survey even if its output already exists')
    parser.add_argument('--append', action='store_true',
                        help=f'add or update these glaciers in {defaults.sigma_table_fn}')
    args = parser.parse_args()

    h = args.survey_spacing
    print(f'Survey at h={h:.0f} m, tolerance {args.tolerance:g} cm w.e. a-1 '
          f'at {defaults.point_error_confidence:g} sigma confidence')
    if args.box_margin != 1.0:
        print(f'sigma inflated by {args.box_margin:g} to cover parameter drift')
    print()
    print(f'{"glacier":<14}{"survey N":>10}{"sigma":>9}{"N needed":>10}'
          f'{"spacing":>10}{"":>4}')

    rows = []
    for glacier in args.glaciers:
        rgi_id = translate_rgi[glacier]['6']
        run_fp = survey_output(glacier, h)

        sigma, survey_n = (None, None) if args.force else read_sigma(run_fp)
        reused = sigma is not None
        if not reused:
            sigma, survey_n = run_survey(glacier, rgi_id, h)
        assert sigma is not None, f'survey produced no output for {glacier}'

        sigma_used = sigma * args.box_margin
        target_n = mesh.point_count_for_sigma(
            sigma_used, args.tolerance, args.coefficient,
            defaults.point_error_confidence)
        polygon, _ = load_outline(rgi_id, HOST_PATHS[host]['rgi_fp'])
        spacing, actual_n = mesh.spacing_for_target_n(polygon, target_n)

        # the survey is only trustworthy if it resolved the field at least
        # as well as the mesh it is recommending
        adequate = survey_n >= target_n
        note = 'reused' if reused else ''
        if not adequate:
            note = (note + ' SURVEY TOO COARSE').strip()

        print(f'{glacier:<14}{survey_n:>10}{sigma:>9.1f}{target_n:>10}'
              f'{spacing:>9.0f}m{"  " + note if note else ""}')

        rows.append(dict(rgiid=rgi_id, name=glacier, sigma_max=round(sigma_used, 1),
                         tolerance_cm=args.tolerance,
                         n_points=target_n, point_spacing=round(spacing, 1),
                         actual_n_points=actual_n,
                         area_km2=round(polygon.area / 1e6, 3),
                         survey_spacing=h, survey_n_points=survey_n,
                         sigma_survey=round(sigma, 1), box_margin=args.box_margin))

    stale = [r['name'] for r in rows if r['survey_n_points'] < r['n_points']]
    if stale:
        print()
        print(f'! {", ".join(stale)} need more points than the survey itself used, '
              f'so their sigma is likely low. Resurvey them finer than '
              f'{h:.0f} m before trusting those counts.')

    print()
    print(f'Total: {sum(r["n_points"] for r in rows)} points across '
          f'{len(rows)} glaciers')

    if args.append:
        update_table(defaults.sigma_table_fn, rows)
        print(f"Updated {defaults.sigma_table_fn}; method_distribute='adaptive' "
              'will use these.')


if __name__ == '__main__':
    main()
