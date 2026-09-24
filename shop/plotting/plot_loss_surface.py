"""
Plots the (kp, wind_factor) log-loss surfaces as a 2x2 heatmap figure
per glacier.

Reads only the .npz that compute_loss_surface.py writes, so this reruns
in a second while iterating on the figure. compute_loss_surface.py
scores every glacier separately and sums them into its 'total'/metric
keys for the calibration objective, but it also keeps each glacier's
own '{glacier}_{metric}' grid, which is what gets plotted here --
summing across glaciers first would hide a glacier whose surface looks
totally different from the rest.

@author: clairevwilson
"""
import argparse
import os

import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

ROOT = '/ocean/projects/ees260009p/cwilson4/Output/loss_surface_0/'
LOSS_FN = ROOT + 'loss_surface_losses.npz'

PANELS = [('albedo', 'Albedo'), ('mb', 'Mass balance'),
          ('snow', 'Snow extent'), ('melt', 'Melt extent')]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--loss-fn', default=LOSS_FN,
                   help='loss .npz from compute_loss_surface.py '
                        '(default %(default)s)')
    p.add_argument('--fig-dir', default=None,
                   help='where the figures go (default: beside --loss-fn)')
    return p.parse_args()


def plot_glacier(glacier, grids, kp_values, wf_values, fig_fn_template):
    fig, axes = plt.subplots(2, 2, figsize=(8, 7), gridspec_kw={'hspace': 0.3, 'wspace': 0.3})
    extent = [kp_values[0] - 0.25, kp_values[-1] + 0.25,
              wf_values[0] - 0.25, wf_values[-1] + 0.25]

    total_err = np.zeros((len(wf_values), len(kp_values)))
    for ax, (key, label) in zip(axes.flat, PANELS):
        # transposed so kp runs along x and wind_factor along y
        field = grids[f'{glacier}_{key}'].T
        im = ax.imshow(field, origin='lower', extent=extent, aspect='auto',
                       cmap='viridis_r', interpolation='nearest')
        cb = fig.colorbar(im, ax=ax, pad=0.02)
        cb.set_label('log loss [nats]', fontsize=9)
        cb.ax.tick_params(labelsize=10)

        j, i = np.unravel_index(np.nanargmin(field), field.shape)
        total_err += field 

        ax.plot(kp_values[i], wf_values[j], marker='o', ms=9, mfc='none',
                mec='r', mew=1.2, zorder=4)

        ax.set_title(f'{label}', fontsize=12)

        fig.supxlabel('Precipitation factor', fontsize=11)
        fig.supylabel('Wind factor', fontsize=11)
        ax.set_xticks(kp_values[::2])
        ax.set_yticks(wf_values[::2])
        ax.tick_params(labelsize=10)
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.5))

    j, i = np.unravel_index(np.nanargmin(total_err), total_err.shape)
    best_kp = str(kp_values[i])
    best_kw = str(wf_values[j])
    best_err = total_err[j, i]
    for ax in axes.flat:
        ax.plot(kp_values[i], wf_values[j], marker='x', ms=9, mfc='none',
                mec='r', mew=2.0, zorder=5)

    title = (f'{glacier}: log loss by parameter combination'
              f'\nBest $k_p=${best_kp} $k_w=${best_kw} log loss: {best_err:.1f}')
    fig.suptitle(title, fontsize=12)
    fig_fn = fig_fn_template.format(glacier=glacier)
    fig.savefig(fig_fn, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {fig_fn}', flush=True)


def main():
    args = parse_args()
    grids = np.load(args.loss_fn, allow_pickle=True)
    kp_values = grids['kp']
    wf_values = grids['wind_factor']

    fig_dir = args.fig_dir or os.path.dirname(args.loss_fn)
    os.makedirs(fig_dir, exist_ok=True)
    fig_fn_template = os.path.join(fig_dir, 'loss_surface_{glacier}.png')

    for glacier in grids['glaciers']:
        plot_glacier(str(glacier), grids, kp_values, wf_values, fig_fn_template)


if __name__ == '__main__':
    main()
