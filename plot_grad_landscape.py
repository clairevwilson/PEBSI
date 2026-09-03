"""
Plots the AD gradient landscape from collect_grad_landscape.csv.

Usage:
    python plot_grad_landscape.py [input.csv] [output.png]
Defaults to grad_landscape.csv and grad_landscape.png.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CSV_IN  = sys.argv[1] if len(sys.argv) > 1 else 'grad_landscape.csv'
PNG_OUT = sys.argv[2] if len(sys.argv) > 2 else 'grad_landscape.png'

data  = np.genfromtxt(CSV_IN, delimiter=',', names=True, dtype=None, encoding='utf-8')
wf    = data['wf'].astype(float)
loss  = data['loss_mwe'].astype(float)
grad  = data['grad_mwe'].astype(float)
sweep = data['sweep']

coarse = np.array([s in ('coarse', 'coarse+fine') for s in sweep])
wf_c, loss_c = wf[coarse], loss[coarse]
idx_c = np.argsort(wf_c)

fig, (ax_loss, ax_grad) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)

ax_loss.plot(wf_c[idx_c], loss_c[idx_c], '-', color='#2166ac', lw=1.5)
ax_loss.set_ylabel('Ice mass (m w.e.)')
ax_loss.grid(True, lw=0.4, alpha=0.5)
ax_loss.set_title('K53 / Kahiltna — AD gradient landscape\n2015 - 2020 period')

fin = np.isfinite(grad)
ax_grad.scatter(wf[fin], grad[fin], color='#333333', s=10, zorder=3)
ax_grad.axhline(0, color='k', lw=0.8)
ax_grad.set_yscale('symlog', linthresh=1.0)
ax_grad.set_ylabel('d(ice) / d(wf)')
ax_grad.set_xlabel('wind_factor (wf)')
ax_grad.grid(True, lw=0.4, alpha=0.5)
ax_grad.set_xlim(wf_c.min(), wf_c.max())
ax_grad.set_ylim(-1e20, 1e20)
ax_grad.set_yticks([-1e20, -1e10, 0, 1e10, 1e20])

fig.tight_layout()
fig.savefig(PNG_OUT, dpi=150)
print(f'wrote {PNG_OUT}')
