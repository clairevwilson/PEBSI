"""
Plots how the glacier-wide mass balance rate settles as the
patch-conforming mesh is refined.

Plotted as deviation from the sweep median rather than from the densest
run, since the densest run is just one more sample and can sit at the
edge of the spread itself.

Rates are annual, matching plot_mb_cells.py, rather than the raw total
over the simulated period that the convergence CSV stores.

Usage:
    python plot_mesh_convergence.py gulkana kennicott

@author: clairevwilson
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from plot_mb_cells import HOURS_PER_YEAR, run_dirs

RESULTS_DIR = 'project/point_density_results/'


def simulated_years(glacier):
    """
    Length of the simulated period, read off one of the runs so the
    plot cannot drift from whatever dates the sweep actually used.
    """
    runs = run_dirs(glacier)
    assert runs, f'No simulated output found for {glacier}'
    ds = xr.open_zarr(runs[max(runs)])
    years = ds.sizes['time'] / HOURS_PER_YEAR
    ds.close()
    return years


def load(glacier):
    """Reads the mesh sweep and converts it to an annual rate."""
    fn = os.path.join(RESULTS_DIR, f'{glacier}_mesh_convergence.csv')
    if not os.path.exists(fn):
        return None
    df = pd.read_csv(fn).sort_values('actual_n_points').reset_index(drop=True)
    df['rate'] = df['mass_balance'] / simulated_years(glacier)
    df['dev'] = df['rate'] - df['rate'].median()
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('glaciers', nargs='+')
    parser.add_argument('--out', default='mesh_convergence.png')
    args = parser.parse_args()

    n = len(args.glaciers)
    fig, axes = plt.subplots(1, n, figsize=(8, 6), squeeze=False)

    for ax, glacier in zip(axes[0], args.glaciers):
        df = load(glacier)
        if df is None:
            continue

        ax.plot(df['actual_n_points'], df['dev'], 'o-', color='C0', ms=5, lw=1.5)
        ax.axhline(0, color='k', lw=0.8)

        ax.set_xscale('log')
        ax.set_xlabel('points simulated')
        ax.set_ylabel('mass balance rate $-$ sweep median [m w.e. a$^{-1}$]')
        ax.set_title(glacier)
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(args.out, dpi=300, bbox_inches='tight')
    print(f'Saved {args.out}')

    for glacier in args.glaciers:
        df = load(glacier)
        if df is None:
            continue
        fine = df.iloc[len(df) // 2:]
        print(f'{glacier}: median rate {df["rate"].median():+.4f} m w.e. a-1, '
              f'densest-half spread {fine["dev"].max() - fine["dev"].min():.4f} '
              f'(N={int(fine["actual_n_points"].iloc[0])}-{int(fine["actual_n_points"].iloc[-1])})')


if __name__ == '__main__':
    main()
