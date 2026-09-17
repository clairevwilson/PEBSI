"""
Plots how the glacier-wide mass balance rate settles as the
patch-conforming mesh is refined.

Shows mass balance rate against point count with no baseline subtracted,
since any reference either assumes a converged value or drags the zero
line into the coarse part of the sweep.

Rates are annual, matching plot_mb_cells.py, rather than the raw total
over the simulated period that the convergence CSV stores.

Usage:
    python plot_mesh_convergence.py
    python plot_mesh_convergence.py gulkana kennicott kahiltna --suffix _costheta

@author: clairevwilson
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from plot_mb_cells import HOURS_PER_YEAR, run_dirs

RESULTS_DIR = 'project/point_density_results/'
DEFAULT_GLACIERS = ['gulkana', 'kennicott', 'kahiltna', 'wolverine', 'lemon_creek']
# tried in order per glacier when --suffix isn't given, newest first
SUFFIX_PRIORITY = ['_ct', '_final', '_albedofix', '_costheta', '']


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


def load(glacier, suffix=None):
    """
    Reads the mesh sweep and converts it to an annual rate.

    suffix=None tries SUFFIX_PRIORITY in order and uses the first file
    that exists, so each glacier picks up its own newest sweep without
    needing one suffix that's valid for all of them.
    """
    suffixes = SUFFIX_PRIORITY if suffix is None else [suffix]
    for s in suffixes:
        fn = os.path.join(RESULTS_DIR, f'{glacier}_mesh_convergence{s}.csv')
        if os.path.exists(fn):
            df = pd.read_csv(fn).sort_values('actual_n_points').reset_index(drop=True)
            df['rate'] = df['mass_balance'] / simulated_years(glacier)
            df.attrs['suffix'] = s
            return df
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('glaciers', nargs='*', default=DEFAULT_GLACIERS)
    parser.add_argument('--suffix', default=None,
                        help='e.g. _costheta to force {glacier}_mesh_convergence_costheta.csv '
                             'for every glacier; omit to use each glacier\'s newest sweep '
                             '(see SUFFIX_PRIORITY)')
    args = parser.parse_args()

    n = len(args.glaciers)
    ncols = 3
    nrows = -(-n // ncols)  # ceil
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.0 * nrows),
                             squeeze=False)
    flat_axes = axes.flatten()

    for i, ax in enumerate(flat_axes):
        if i >= n:
            ax.axis('off')
            continue

        glacier = args.glaciers[i]
        df = load(glacier, args.suffix)
        if df is None:
            print(f'no data for {glacier}, skipping')
            ax.axis('off')
            continue

        ax.plot(df['actual_n_points'], df['rate'], 'o-', color='C0', ms=5, lw=1.5)
        ax.set_xscale('log')
        ax.set_xlabel('points simulated')
        ax.set_ylabel('mass balance rate [m w.e. a$^{-1}$]')
        ax.set_title(f"{glacier} ({df.attrs['suffix'] or 'no suffix'})")
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig('mesh_convergence.png', dpi=300, bbox_inches='tight')
    print(f'Saved mesh_convergence.png')


if __name__ == '__main__':
    main()
