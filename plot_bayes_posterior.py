import os
import numpy as np
import matplotlib.pyplot as plt

from project.bayes_calibrate import GLACIERS, baseline, CHAIN_DIR

fig, axes = plt.subplots(len(GLACIERS), 3, figsize=(12, 3 * len(GLACIERS)),
                          gridspec_kw={'hspace': 0.4, 'wspace': 0.35})

for row, glacier_name in enumerate(GLACIERS):
    data = np.load(os.path.join(CHAIN_DIR, f'{glacier_name}_posterior.npz'))
    chain = data['chain']
    kp, wf = chain[:, 0], chain[:, 1]

    ax_kp, ax_wf, ax_joint = axes[row]

    ax_kp.hist(kp, bins=40, color='steelblue')
    ax_kp.axvline(baseline['kp'], c='k', linestyle='--', linewidth=1)
    ax_kp.set_title(f'{glacier_name}: kp posterior')

    ax_wf.hist(wf, bins=40, color='indianred')
    ax_wf.axvline(baseline['wind_factor'], c='k', linestyle='--', linewidth=1)
    ax_wf.set_title(f'{glacier_name}: wind_factor posterior')

    ax_joint.hist2d(kp, wf, bins=40, cmap='magma')
    ax_joint.set_xlabel('kp')
    ax_joint.set_ylabel('wind_factor')
    ax_joint.set_title('joint posterior')

plt.savefig('bayes_posterior_all_glaciers.png', dpi=330, bbox_inches='tight')
plt.show()
