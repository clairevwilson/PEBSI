"""
Compares the point_density_convergence.py results across glaciers and fits
a log-log line between "area" and "points needed to converge", so that a
target point count for any other glacier can be interpolated from its area
instead of using one fixed n_points everywhere.

"Converged" is found relative to a reference plateau: the --seed densest
runs, whose mean and standard deviation estimate the settled value and its
natural noise level. Walking from densest to sparsest, a point is "in" the
plateau if it falls within mean +/- --k-std * std of that fixed reference;
converged_n_points is the n_points of the sparsest point in the longest run
of consecutive in-plateau points extending back from the densest point,
tolerating up to --max-gap consecutive out-of-band points along the way (a
single isolated blip -- one noisy lattice draw -- shouldn't by itself erase
an otherwise-converged stretch below it; two in a row is treated as the
real end of the plateau). The reference is fixed (not recomputed as points
are added) so a lucky small sample doesn't produce an unstable, overly
tight band. The reference mean is also the value plotted as a horizontal
line by plot_point_density.py.

Usage:
    python point_density_summary.py gulkana kennicott
    python point_density_summary.py gulkana kahiltna kennicott lemon_creek wolverine taku --k-std 1.0

@author: clairevwilson
"""
import argparse
import os

import numpy as np
import pandas as pd

RESULTS_DIR = 'project/point_density_results/'


def converged_n_points(df, k_std, seed=6, max_gap=1):
    """Returns (n_points at the start of the stable plateau, reference mean)."""
    df = df.sort_values('requested_n_points').reset_index(drop=True)
    values = df['mass_balance'].values
    n_points = df['requested_n_points'].values

    reference_vals = values[-seed:]
    mean_ref, std_ref = reference_vals.mean(), reference_vals.std()
    in_band = np.abs(values - mean_ref) <= k_std * std_ref

    # walk backward from just before the seed, tolerating up to max_gap
    # consecutive out-of-band points before calling it the real end
    last_good = len(df) - seed
    i = len(df) - seed - 1
    gap = 0
    while i >= 0:
        if in_band[i]:
            gap = 0
            last_good = i
        else:
            gap += 1
            if gap > max_gap:
                break
        i -= 1

    n_conv = int(n_points[last_good])
    return n_conv, float(mean_ref)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('glaciers', nargs='+', help='glacier names with saved results')
    parser.add_argument('--k-std', type=float, default=1.0,
                         help='how many standard deviations of the current plateau a point may '
                              'deviate by and still join it (default 1.0)')
    parser.add_argument('--seed', type=int, default=6,
                         help='number of densest points used to build the reference plateau (default 6)')
    parser.add_argument('--max-gap', type=int, default=1,
                         help='consecutive out-of-band points tolerated as noise before calling it '
                              'the real end of the plateau (default 1)')
    args = parser.parse_args()

    rows = []
    for glacier in args.glaciers:
        fn = os.path.join(RESULTS_DIR, f'{glacier}_point_density.csv')
        df = pd.read_csv(fn)
        n_conv, reference = converged_n_points(df, args.k_std, args.seed, args.max_gap)
        area = df['area_km2'].iloc[0]
        rows.append({'glacier': glacier, 'area_km2': area, 'converged_n_points': n_conv})
        print(f'{glacier:>12}: area={area:8.1f} km2  plateau mean={reference:+.3f}  converged at n_points={n_conv}')

    summary = pd.DataFrame(rows).sort_values('area_km2').reset_index(drop=True)

    if len(summary) >= 2:
        # fit log(n_points) = a*log(area) + b by least squares across all glaciers
        # (reduces to the exact two-point line when there are only two)
        log_area = np.log(summary['area_km2'])
        log_n = np.log(summary['converged_n_points'])
        slope, intercept = np.polyfit(log_area, log_n, 1)

        print(f'\nFit across {len(summary)} glaciers ({", ".join(summary["glacier"])}):')
        print(f'  n_points(area) = exp({intercept:.4f}) * area_km2 ** {slope:.4f}')
        print('  i.e. required point density (n_points / area) scales as '
              f'area ** {slope - 1:.4f}')

        out_fn = os.path.join(RESULTS_DIR, 'area_vs_points_fit.csv')
        summary.to_csv(out_fn, index=False)
        print(f'\nSaved summary to {out_fn}')


if __name__ == '__main__':
    main()
