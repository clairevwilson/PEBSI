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

@author: clairevwilson
"""
import argparse
import os

import matplotlib.pyplot as plt
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


def load(glacier, suffix=''):
    """Reads the mesh sweep and converts it to an annual rate."""
    fn = os.path.join(RESULTS_DIR, f'{glacier}_mesh_convergence{suffix}.csv')
    if not os.path.exists(fn):
        return None
    df = pd.read_csv(fn).sort_values('actual_n_points').reset_index(drop=True)
    df['rate'] = df['mass_balance'] / simulated_years(glacier)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--suffix', default='',
                        help='e.g. _costheta to read {glacier}_mesh_convergence_costheta.csv')
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))

    for col, glacier in enumerate(['gulkana', 'kennicott']):
        df = load(glacier, args.suffix)
        if df is None:
            continue

        ax = axes[col]

        ax.plot(df['actual_n_points'], df['rate'], 'o-', color='C0', ms=5, lw=1.5)
        ax.set_xscale('log')
        ax.set_xlabel('points simulated')
        ax.set_ylabel('mass balance rate [m w.e. a$^{-1}$]')
        ax.set_title(glacier)
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig('mesh_convergence.png', dpi=300, bbox_inches='tight')
    print(f'Saved mesh_convergence.png')


if __name__ == '__main__':
    main()
