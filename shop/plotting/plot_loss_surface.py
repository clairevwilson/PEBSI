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
import os

import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

ROOT = '/ocean/projects/ees260009p/cwilson4/Output/AD_optimize_2/'
LOSS_FN = ROOT + 'loss_surface_losses.npz'
FIG_FN_TEMPLATE = os.path.join(ROOT, 'loss_surface_{glacier}.png')

PANELS = [('albedo', 'Albedo'), ('mb', 'Mass balance'),
          ('snow', 'Snow extent'), ('melt', 'Melt extent')]


def plot_glacier(glacier, grids, kp_values, wf_values):
    fig, axes = plt.subplots(2, 2, figsize=(8, 7), gridspec_kw={'hspace': 0.3, 'wspace': 0.3})
    extent = [kp_values[0] - 0.25, kp_values[-1] + 0.25,
              wf_values[0] - 0.25, wf_values[-1] + 0.25]

    total_err = np.zeros((10, 10))
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
    fig_fn = FIG_FN_TEMPLATE.format(glacier=glacier)
    fig.savefig(fig_fn, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {fig_fn}', flush=True)


def main():
    grids = np.load(LOSS_FN, allow_pickle=True)
    kp_values = grids['kp']
    wf_values = grids['wind_factor']

    for glacier in grids['glaciers']:
        plot_glacier(str(glacier), grids, kp_values, wf_values)


if __name__ == '__main__':
    main()
