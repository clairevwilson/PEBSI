import os
import yaml
import numpy as np
import matplotlib.pyplot as plt

progress_fn = '/ocean/projects/ees260009p/cwilson4/Output/AD_optimize/AD_optimize_progress.yaml'
out_fn = '/ocean/projects/ees260009p/cwilson4/Output/AD_optimize/AD_iteration_errors.png'

with open(progress_fn) as f:
    progress = yaml.safe_load(f)
glaciers = progress['glaciers']
history = progress['history']

step = np.array([h['step'] for h in history])
mb = np.array([h['mb'] for h in history])
albedo = np.array([h['albedo'] for h in history])
wet = np.array([h['wet'] for h in history])
rows = [('Mass balance error', mb), ('Albedo error', albedo),
        ('Wet error', wet), ('Combined', mb + albedo + wet)]

fig, axs = plt.subplots(4, 1, figsize=(4.2, 6), sharex=True)
for ax, (label, vals) in zip(axs, rows):
    for g, name in enumerate(glaciers):
        ax.plot(step, vals[:, g], marker='o', color=f'C{g}', label=name)
    ax.set_ylabel(label)
axs[0].legend(ncol=3, fontsize=8)
axs[-1].set_xlabel('Evaluation')
fig.tight_layout()
fig.savefig(out_fn, dpi=300)
