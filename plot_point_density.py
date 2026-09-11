import sys

import pandas as pd
import matplotlib.pyplot as plt

from point_density_summary import converged_n_points

DEFAULT_GLACIERS = ['gulkana', 'kahiltna', 'kennicott', 'lemon_creek', 'wolverine']

args = sys.argv[1:]
suffix = ''
if args and args[0].startswith('_'):
    suffix = args[0]
    args = args[1:]
glaciers = args if args else DEFAULT_GLACIERS

fig, axes = plt.subplots(1, len(glaciers), figsize=(4 * len(glaciers), 3.5))

wf = None
for ax, glacier in zip(axes, glaciers):
    df = pd.read_csv(f'project/point_density_results/{glacier}_point_density{suffix}.csv').sort_values('requested_n_points')
    wf = df['wind_factor'].iloc[0]
    _, reference = converged_n_points(df, k_std=1.0)
    ax.plot(df['actual_n_points'], df['mass_balance'], 'o-')
    ax.axhline(reference, color='gray', linestyle='--', linewidth=1)
    ax.set_title(glacier)
    ax.set_xlabel('n points')
    ax.set_ylabel('MB')

fig.suptitle(f'wind factor = {wf}')
fig.tight_layout()
fig.savefig(f'point_density{suffix}.png', dpi=150)
