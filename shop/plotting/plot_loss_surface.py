"""
Plots the (kp, wind_factor) log-loss surfaces as 2x2 heatmaps.

Reads only the .npz that compute_loss_surface.py writes, so this reruns
in a second while iterating on the figure.

@author: clairevwilson
"""
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

LOSS_FN = '/ocean/projects/ees260009p/cwilson4/Output/AD_optimize/loss_surface_losses.npz'
FIG_FN = '/ocean/projects/ees260009p/cwilson4/Figs/loss_surface.png'

PANELS = [('albedo', 'Albedo'), ('mb', 'Mass balance'),
          ('snow', 'Snow extent'), ('melt', 'Melt extent')]


def main():
    grids = np.load(LOSS_FN, allow_pickle=True)
    kp_values = grids['kp']
    wf_values = grids['wind_factor']

    fig, axes = plt.subplots(2, 2, figsize=(8, 7), gridspec_kw={'hspace': 0.3, 'wspace': 0.3})
    extent = [kp_values[0] - 0.25, kp_values[-1] + 0.25,
              wf_values[0] - 0.25, wf_values[-1] + 0.25]

    for ax, (key, label) in zip(axes.flat, PANELS):
        # transposed so kp runs along x and wind_factor along y
        field = grids[key].T
        im = ax.imshow(field, origin='lower', extent=extent, aspect='auto',
                       cmap='viridis_r', interpolation='nearest')
        cb = fig.colorbar(im, ax=ax, pad=0.02)
        cb.set_label('log loss [nats]', fontsize=9)
        cb.ax.tick_params(labelsize=10)

        j, i = np.unravel_index(np.nanargmin(field), field.shape)
        # ax.plot(kp_values[i], wf_values[j], marker='o', ms=9, mfc='none',
        #         mec='white', mew=2.0, zorder=3)
        # ax.plot(kp_values[i], wf_values[j], marker='o', ms=9, mfc='none',
        #         mec='#c1121f', mew=1.2, zorder=4)

        ax.set_title(f'{label}', fontsize=12)

        fig.supxlabel('Precipitation factor', fontsize=11)
        fig.supylabel('Wind factor', fontsize=11)
        ax.set_xticks(kp_values[::2])
        ax.set_yticks(wf_values[::2])
        ax.tick_params(labelsize=10)
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.5))
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.5))

    # fig.suptitle(
    #     f'{glacier} glacier-wide log loss by parameter combination',
    #     fontsize=12)
    fig.savefig(FIG_FN, dpi=160, bbox_inches='tight')
    print(f'Wrote {FIG_FN}', flush=True)


if __name__ == '__main__':
    main()
