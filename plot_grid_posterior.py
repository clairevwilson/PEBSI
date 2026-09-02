"""
Combines the per-job (cells x glaciers) log-likelihoods saved by
grid_calibrate_glaciers.py into a discretized posterior over
(kp, wind_factor) for EACH glacier independently, via Bayes' rule with
a uniform prior over the grid (every cell is within bounds by
construction, so the prior is just a constant that drops out of the
normalization). Glaciers were pooled into shared simulation runs for
compute efficiency, but their log-likelihoods were kept separate the
whole way through, so each glacier's posterior only uses its own
column of the saved results.

@author: clairevwilson
"""
import os
import glob
import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt

from project.bayes_calibrate import GLACIERS
from project.grid_calibrate import GRID_RES, RESULTS_DIR

job_files = sorted(glob.glob(os.path.join(RESULTS_DIR, 'job*.npz')))
if not job_files:
    raise SystemExit('No grid results found yet in ' + RESULTS_DIR)

theta_all, log_like_all = [], []
for fn in job_files:
    data = np.load(fn, allow_pickle=True)
    theta_all.append(data['theta'])
    log_like_all.append(data['log_like'])
theta_all = np.concatenate(theta_all)             # (n_cells_done, 2)
log_like_all = np.concatenate(log_like_all)        # (n_cells_done, n_glaciers)

if len(theta_all) != GRID_RES * GRID_RES:
    print(f'! only {len(theta_all)}/{GRID_RES**2} cells finished so far')

fig, axes = plt.subplots(len(GLACIERS), 1, figsize=(6, 5 * len(GLACIERS)))
if len(GLACIERS) == 1:
    axes = [axes]

kp_vals = np.unique(theta_all[:, 0])
wf_vals = np.unique(theta_all[:, 1])

for ax, glacier_name, gi in zip(axes, GLACIERS, range(len(GLACIERS))):
    log_like = log_like_all[:, gi]

    # discrete posterior via Bayes' rule (uniform prior) + log-sum-exp
    log_post = log_like - logsumexp(log_like)
    posterior = np.exp(log_post)

    post_grid = np.full((len(kp_vals), len(wf_vals)), np.nan)
    for (kp, wf), p in zip(theta_all, posterior):
        i = np.argmin(np.abs(kp_vals - kp))
        j = np.argmin(np.abs(wf_vals - wf))
        post_grid[i, j] = p

    im = ax.imshow(post_grid.T, origin='lower', aspect='auto', cmap='magma',
                    extent=[kp_vals[0], kp_vals[-1], wf_vals[0], wf_vals[-1]])
    ax.set_xlabel('kp')
    ax.set_ylabel('wind_factor')
    ax.set_title(f'{glacier_name}: posterior over (kp, wind_factor)')
    plt.colorbar(im, ax=ax, label='posterior probability')

    best_idx = np.nanargmax(post_grid)
    bi, bj = np.unravel_index(best_idx, post_grid.shape)
    print(f'{glacier_name}: MAP estimate kp={kp_vals[bi]:.3f}, wind_factor={wf_vals[bj]:.3f}')

plt.tight_layout()
plt.savefig('grid_posterior_all_glaciers.png', dpi=330, bbox_inches='tight')
plt.show()
