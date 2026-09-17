"""
Recalculates each glacier's converged mass balance from its actual
sweep data (not a hardcoded guess), then plots each point's deviation
from that value, all five glaciers on one shared y-axis so the
magnitude of drift is directly comparable across glaciers.

Converged value = mean over the longest run of consecutive-by-N points
whose spread (max-min) is within tolerance -- the same rule that caught
Kennicott's earlier false convergence, extended to not assume the run
has to end at the finest point, since Kennicott's fine end just proved
that's not always the trustworthy stretch.

    python plot_convergence_diff.py
    python plot_convergence_diff.py --tolerance 0.02
"""
import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

RESULTS = 'project/point_density_results/'
YEARS = 3.0
GLACIERS = ['gulkana', 'kennicott', 'kahiltna', 'wolverine', 'lemon_creek']
SUFFIX_PRIORITY = ['_ct', '_final', '_albedofix', '_costheta', '']


def load(glacier):
    for s in SUFFIX_PRIORITY:
        fn = os.path.join(RESULTS, f'{glacier}_mesh_convergence{s}.csv')
        if os.path.exists(fn):
            df = pd.read_csv(fn).sort_values('actual_n_points').reset_index(drop=True)
            return df, s
    return None, None


def find_converged(n, mb, tolerance):
    """
    Longest run ENDING AT THE FINEST POINT with max-min <= tolerance.
    A tight cluster elsewhere in the sweep is not evidence of
    convergence -- finer points are trusted more, not less, so the
    run has to include them, not stop short of them.
    """
    last = len(mb) - 1
    best_i = last
    for i in range(last - 2, -1, -1):
        if np.ptp(mb[i:last + 1]) <= tolerance:
            best_i = i
        else:
            break
    if best_i == last:
        return None, None, None
    return mb[best_i:last + 1].mean(), n[best_i], n[last]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tolerance', type=float, default=0.02)
    args = p.parse_args()

    results = {}
    print(f'{"glacier":>12} {"converged":>10} {"run N range":>18} {"suffix":>10}')
    for glacier in GLACIERS:
        df, suffix = load(glacier)
        n = df['actual_n_points'].values
        mb = df['mass_balance'].values / YEARS
        converged, n_lo, n_hi = find_converged(n, mb, args.tolerance)
        results[glacier] = (df, suffix, converged, n_lo, n_hi)
        if converged is None:
            print(f'{glacier:>12} {"NO RUN FOUND":>10} -- extend the sweep or loosen tolerance')
        else:
            ends_at_finest = n_hi == n.max()
            flag = '' if ends_at_finest else '  <- NOT the finest point!'
            print(f'{glacier:>12} {converged:>10.4f} {f"{n_lo:.0f}-{n_hi:.0f}":>18} '
                  f'{suffix:>10}{flag}')

    fig, axes = plt.subplots(2, 3, figsize=(8, 5), squeeze=False)
    flat_axes = axes.flatten()

    tolerance_cm = args.tolerance * 100

    all_diffs = []
    for glacier in GLACIERS:
        df, suffix, converged, n_lo, n_hi = results[glacier]
        if converged is None:
            continue
        mb = df['mass_balance'].values / YEARS
        all_diffs.append((mb - converged) * 100)
    ylim = max(abs(np.concatenate(all_diffs).min()), abs(np.concatenate(all_diffs).max()))
    ylim = max(ylim * 1.1, tolerance_cm * 1.2)

    all_n = np.concatenate([results[g][0]['actual_n_points'].values
                            for g in GLACIERS if results[g][2] is not None])
    n_lo_shared, n_hi_shared = all_n.min(), all_n.max()

    for i, ax in enumerate(flat_axes):
        if i >= len(GLACIERS):
            ax.axis('off')
            continue
        glacier = GLACIERS[i]
        df, suffix, converged, n_lo, n_hi = results[glacier]
        if converged is None:
            ax.axis('off')
            continue

        n = df['actual_n_points'].values
        h = df['point_spacing'].values
        mb = df['mass_balance'].values / YEARS
        diff = (mb - converged) * 100

        ax.axhline(0, color='gray', lw=1, alpha=0.5)
        ax.axhspan(-tolerance_cm, tolerance_cm, color='C0', alpha=0.08)
        in_run = (n >= n_lo) & (n <= n_hi)
        ax.plot(n[~in_run], diff[~in_run], 'o', color='C3', ms=6)
        ax.plot(n[in_run], diff[in_run], 'o-', color='C0', ms=6, lw=1.5)

        ax.set_xscale('log')
        ax.set_xlim(n_lo_shared / 1.15, n_hi_shared * 1.15)
        ax.set_ylim(-ylim, ylim)

        ax.text(0.5, 0.95, f'{glacier}', transform=ax.transAxes,
               ha='center', va='top', fontsize=12)

        max_labels = 5
        step = max(1, -(-len(n) // max_labels))  # ceil
        show = sorted(set(list(range(0, len(n), step)) + [len(n) - 1]))

        # identity functions: secax shares the parent's N coordinates
        # exactly, so ticks go at real N positions with no separate
        # h<->N transform for matplotlib to get backwards
        secax = ax.secondary_xaxis('top', functions=(lambda x: x, lambda x: x))
        secax.set_xscale('log')
        secax.set_xticks(n[show])
        secax.set_xticklabels([f'{v:g}' for v in h[show]], fontsize=8, rotation=45)
        secax.xaxis.set_minor_formatter(mticker.NullFormatter())

    fig.suptitle('Element edge length [m]')
    fig.supxlabel('Points simulated')
    fig.supylabel('MB - converged [cm w.e. a$^{-1}$]')
    fig.tight_layout()
    fig.savefig('mesh_convergence_diff.png', dpi=300, bbox_inches='tight')
    print()
    print('Saved mesh_convergence_diff.png')
    print('blue = the run used to define "converged"; red = outside that run; '
          'shaded band = tolerance')


if __name__ == '__main__':
    main()
