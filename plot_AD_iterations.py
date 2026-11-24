import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt

# usage: python plot_AD_iterations.py [PROGRESS_YAML] [OUT_PNG]
progress_fn = sys.argv[1] if len(sys.argv) > 1 else '/ocean/projects/ees260009p/cwilson4/Output/AD_optimize/AD_optimize_progress.yaml'
out_fn = sys.argv[2] if len(sys.argv) > 2 else '/ocean/projects/ees260009p/cwilson4/Output/AD_optimize/AD_iteration_errors.png'

with open(progress_fn) as f:
    progress = yaml.safe_load(f)
glaciers = progress['glaciers']
history = progress['history']

step = np.array([h['step'] for h in history])
mb = np.array([h['mb'] for h in history])
albedo = np.array([h['albedo'] for h in history])
wet = np.array([h['wet'] for h in history])
rows = [('Mass balance error', mb), ('Albedo error', albedo), ('Wet error', wet)]
# runs with --index_mb also fit winter index balance (summer is tracked but weighted 0)
if 'index_winter' in history[0]:
    rows.append(('Winter index error', np.array([h['index_winter'] for h in history])))
rows.append(('Combined', sum(vals for _, vals in rows)))

fig, axs = plt.subplots(len(rows), 1, figsize=(4.2, 1.5 * len(rows)), sharex=True)
for ax, (label, vals) in zip(axs, rows):
    for g, name in enumerate(glaciers):
        ax.plot(step, vals[:, g], marker='o', color=f'C{g}', label=name)
    ax.set_ylabel(label)
axs[0].legend(ncol=3, fontsize=9)
axs[-1].set_xlabel('Evaluation')
fig.tight_layout()
fig.savefig(out_fn, dpi=300)
