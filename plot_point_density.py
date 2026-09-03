"""
Plots n_points vs. glacier-wide mass balance for each glacier with saved
point_density_convergence.py results, one subplot per glacier.

Usage:
    python plot_point_density.py gulkana kennicott
    python plot_point_density.py gulkana kennicott -o point_density.png

@author: clairevwilson
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = 'project/point_density_results/'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('glaciers', nargs='+', help='glacier names with saved results')
    parser.add_argument('-o', '--out', default='point_density.png')
    args = parser.parse_args()

    fig, axes = plt.subplots(1, len(args.glaciers), figsize=(6 * len(args.glaciers), 4.5))
    if len(args.glaciers) == 1:
        axes = [axes]

    for ax, glacier in zip(axes, args.glaciers):
        fn = os.path.join(RESULTS_DIR, f'{glacier}_point_density.csv')
        df = pd.read_csv(fn).sort_values('requested_n_points')

        ax.plot(df['actual_n_points'], df['mass_balance'], 'o-')
        ax.set_xlabel('N points')
        ax.set_ylabel('Glacier-wide mass balance [m w.e.]')
        ax.set_title(f'{glacier} (area={df["area_km2"].iloc[0]:.1f} km2)')
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f'Saved plot to {args.out}')


if __name__ == '__main__':
    main()
