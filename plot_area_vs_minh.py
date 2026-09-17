"""
Plots recommended minimum mesh spacing against glacier area.

min_h per glacier comes from the convergence analysis in this session:
the coarsest h whose run, plus >=2 independently finer runs, all agree
within 0.02 m w.e./yr of each other (Kennicott judged converged just
above N=1000 given how small the plateau's noise band is relative to
the swings this whole investigation started from).

    python plot_area_vs_minh.py
"""
import matplotlib.pyplot as plt

# glacier: (area_km2, min_h_m)
DATA = {
    'gulkana':     (17.567, 1200),
    'wolverine':   (16.749, 500),
    'lemon_creek': (9.528, 1200),
    'kahiltna':    (479.521, 1500),
    'kennicott':   (292.5, 950),
}

fig, ax = plt.subplots(figsize=(5.5, 4.5))

for glacier, (area, h) in DATA.items():
    ax.scatter(area, h, s=60, color='C0', zorder=3)
    ax.annotate(glacier, (area, h), textcoords='offset points',
               xytext=(6, 6), fontsize=9)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('glacier area [km$^2$]')
ax.set_ylabel('minimum usable spacing h [m]')
ax.grid(alpha=0.25, which='both')
fig.tight_layout()
fig.savefig('area_vs_minh.png', dpi=300, bbox_inches='tight')
print('Saved area_vs_minh.png')
