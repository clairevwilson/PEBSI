"""
Picks the point count/spacing to actually run per glacier: fits an
exponential to the mesh sweep, takes its asymptote as the converged
mass balance, and reports the coarsest (smallest N) resolution whose
fitted curve is within a tolerance of that asymptote.

Merges any split CSVs from point_density_convergence.py --tag first
(e.g. _A/_B, or _c/_f for a cheap/expensive split) if parts exist and
a combined file doesn't.

Usage:
    python recommend_n.py gulkana kennicott
    python recommend_n.py gulkana --tolerance 0.02
    python recommend_n.py gulkana --suffix _costheta

@author: clairevwilson
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

RESULTS = 'project/point_density_results/'
YEARS = 3.0


def merge_if_needed(glacier, suffix):
    out = os.path.join(RESULTS, f'{glacier}_mesh_convergence{suffix}.csv')
    parts = sorted(glob.glob(os.path.join(
        RESULTS, f'{glacier}_mesh_convergence{suffix}_*.csv')))
    if parts and not os.path.exists(out):
        df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        df = df.drop_duplicates(subset='point_spacing', keep='last')
        df = df.sort_values('actual_n_points').reset_index(drop=True)
        df.to_csv(out, index=False)
        print(f'merged {len(parts)} parts ({", ".join(parts)}) -> {out}')
    return out


def expo(n, mb_inf, a, tau):
    return mb_inf - a * np.exp(-n / tau)


def recommend(glacier, suffix, tolerance):
    fn = merge_if_needed(glacier, suffix)
    assert os.path.exists(fn), f'no sweep found at {fn}'
    df = pd.read_csv(fn).sort_values('actual_n_points').reset_index(drop=True)

    n = df['actual_n_points'].values.astype(float)
    mb = df['mass_balance'].values / YEARS
    h = df['point_spacing'].values.astype(float)

    p0 = [mb[-1], mb[-1] - mb[0], n.mean()]
    popt, _ = curve_fit(expo, n, mb, p0=p0, maxfev=60000)
    mb_inf, a, tau = popt
    pred = expo(n, *popt)
    r2 = 1 - ((mb - pred) ** 2).sum() / ((mb - mb.mean()) ** 2).sum()

    # smallest N in a fine grid where the FIT is within tolerance of mb_inf,
    # then report the nearest h actually in the sweep at or below that N
    n_grid = np.geomspace(n.min(), n.max() * 3, 20000)
    within = np.abs(expo(n_grid, *popt) - mb_inf) <= tolerance
    n_needed = n_grid[within][0] if within.any() else np.nan

    print('=' * 66)
    print(f'{glacier.upper()}   ({fn})')
    print('=' * 66)
    print(f'fit: MB_inf={mb_inf:+.4f} m w.e./yr, R2={r2:.4f}, '
          f'tau={tau:.0f} points')
    print(f'sweep spans N={n.min():.0f} to {n.max():.0f}, '
          f'finest run={mb[-1]:+.4f}')
    print()

    if np.isnan(n_needed):
        print(f'sweep never reaches within {tolerance} m w.e./yr of the '
              f'fitted asymptote -- extend the sweep finer before trusting this')
        return None

    print(f'N needed for |MB - MB_inf| <= {tolerance} m w.e./yr: '
          f'{n_needed:.0f}')

    # nearest actually-run resolution at or above the needed N
    ok = n >= n_needed
    if ok.any():
        i = np.argmin(n[ok])
        rec_n, rec_h = n[ok][i], h[ok][i]
        print(f'nearest resolution actually in the sweep: '
              f'h={rec_h:.0f} m, N={rec_n:.0f}, '
              f'MB={mb[n == rec_n][0]:+.4f} '
              f'(fit says {expo(rec_n, *popt):+.4f}, '
              f'{abs(expo(rec_n, *popt) - mb_inf):.4f} from asymptote)')
    else:
        print('every run in the sweep is coarser than the recommended N -- '
              'extrapolating past the sweep, extend it to confirm')
    return dict(glacier=glacier, mb_inf=mb_inf, r2=r2, n_needed=n_needed)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('glaciers', nargs='+')
    p.add_argument('--tolerance', type=float, default=0.03,
                   help='m w.e./yr from the fitted asymptote (default 0.03 = 3 cm)')
    p.add_argument('--suffix', default='',
                   help='e.g. _costheta to read {glacier}_mesh_convergence_costheta.csv')
    args = p.parse_args()

    for g in args.glaciers:
        recommend(g, args.suffix, args.tolerance)
        print()


if __name__ == '__main__':
    main()
