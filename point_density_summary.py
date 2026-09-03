"""
Compares the point_density_convergence.py results for two glaciers (a
small endmember and a big one) and fits a simple log-log line between
"area" and "points needed to converge", so that a target point count
for any other glacier can be interpolated from its area instead of
using one fixed n_points everywhere (PEBSI's current 'scatter' default).

"Converged" is defined as the smallest requested_n_points at which
mb_diff_from_densest first falls within --tol m w.e. of the densest run.
Later n_points can still exceed tol -- the scatter/adaptive grid is
recomputed from scratch at each n rather than refining the previous
one, so a bigger n is a different lattice draw over the glacier, not a
strict superset of points, and one draw can land on enough extreme
terrain to swing a small glacier's area-mean back out of tolerance.
Those are reported as a warning, not folded into the convergence point.

Usage:
    python point_density_summary.py gulkana kennicott
    python point_density_summary.py gulkana kennicott --tol 0.05

@author: clairevwilson
"""
import argparse
import os

import numpy as np
import pandas as pd

RESULTS_DIR = 'project/point_density_results/'


def converged_n_points(df, tol):
    """Returns (n_points at first dip within tol, later n_points that exceed tol anyway)."""
    df = df.sort_values('requested_n_points').reset_index(drop=True)
    diffs = df['mb_diff_from_densest'].abs().values
    within = np.flatnonzero(diffs <= tol)
    i = within[0] if len(within) else len(df) - 1
    n_conv = int(df.loc[i, 'requested_n_points'])

    later = df.iloc[i + 1:]
    outliers = later.loc[later['mb_diff_from_densest'].abs() > tol, 'requested_n_points'].tolist()
    return n_conv, outliers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('glaciers', nargs='+', help='glacier names with saved results')
    parser.add_argument('--tol', type=float, default=0.1,
                         help='convergence tolerance in m w.e. of the densest run''s mass balance')
    args = parser.parse_args()

    rows = []
    for glacier in args.glaciers:
        fn = os.path.join(RESULTS_DIR, f'{glacier}_point_density.csv')
        df = pd.read_csv(fn)
        n_conv, outliers = converged_n_points(df, args.tol)
        area = df['area_km2'].iloc[0]
        rows.append({'glacier': glacier, 'area_km2': area, 'converged_n_points': n_conv})
        print(f'{glacier:>12}: area={area:8.1f} km2  converged at n_points={n_conv}')
        if outliers:
            print(f'{"":>12}  warning: denser runs {outliers} still exceed tol -- likely just a noisy lattice draw')

    summary = pd.DataFrame(rows)

    if len(summary) >= 2:
        # fit log(n_points) = a*log(area) + b using the endmembers directly
        summary = summary.sort_values('area_km2').reset_index(drop=True)
        small, big = summary.iloc[0], summary.iloc[-1]
        log_a1, log_n1 = np.log(small['area_km2']), np.log(small['converged_n_points'])
        log_a2, log_n2 = np.log(big['area_km2']), np.log(big['converged_n_points'])
        slope = (log_n2 - log_n1) / (log_a2 - log_a1)
        intercept = log_n1 - slope * log_a1

        print(f'\nFit between endmembers ({small["glacier"]}, {big["glacier"]}):')
        print(f'  n_points(area) = exp({intercept:.4f}) * area_km2 ** {slope:.4f}')
        print('  i.e. required point density (n_points / area) scales as '
              f'area ** {slope - 1:.4f}')

        out_fn = os.path.join(RESULTS_DIR, 'area_vs_points_fit.csv')
        summary.to_csv(out_fn, index=False)
        print(f'\nSaved summary to {out_fn}')


if __name__ == '__main__':
    main()
