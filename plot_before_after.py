"""
Overlays the original (pre-fix) and current (wind ice-mask + cell-averaged
shading + precomputed cos_theta) mesh convergence sweeps, so the change
from the code updates is visible directly.

    python plot_before_after.py
"""
import os

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

from plot_mb_cells import HOURS_PER_YEAR, run_dirs

RESULTS_DIR = 'project/point_density_results/'


def simulated_years(glacier):
    runs = run_dirs(glacier)
    assert runs, f'No simulated output found for {glacier}'
    ds = xr.open_zarr(runs[max(runs)])
    years = ds.sizes['time'] / HOURS_PER_YEAR
    ds.close()
    return years


def load(glacier, suffix):
    fn = os.path.join(RESULTS_DIR, f'{glacier}_mesh_convergence{suffix}.csv')
    if not os.path.exists(fn):
        return None
    df = pd.read_csv(fn).sort_values('actual_n_points').reset_index(drop=True)
    df['rate'] = df['mass_balance'] / simulated_years(glacier)
    return df


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    for col, glacier in enumerate(['gulkana', 'kennicott']):
        before = load(glacier, '_prewindfix')
        after = load(glacier, '_costheta')
        ax = axes[col]

        if before is not None:
            ax.plot(before['actual_n_points'], before['rate'], 'o-',
                    color='C3', ms=5, lw=1.5, label='before')
        if after is not None:
            ax.plot(after['actual_n_points'], after['rate'], 'o-',
                    color='C0', ms=5, lw=1.5, label='after')

        ax.set_xscale('log')
        ax.set_xlabel('points simulated')
        ax.set_ylabel('mass balance rate [m w.e. a$^{-1}$]')
        ax.set_title(glacier)
        ax.grid(alpha=0.25)
        ax.legend()

    fig.tight_layout()
    fig.savefig('mesh_convergence_before_after.png', dpi=300, bbox_inches='tight')
    print('Saved mesh_convergence_before_after.png')


if __name__ == '__main__':
    main()
