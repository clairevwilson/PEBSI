import xarray as xr 
import os
import yaml
import matplotlib.pyplot as plt 
import matplotlib as mpl
from project.glacierwide_loss import * 

output_fp = '/ocean/projects/ees260009p/cwilson4/Output/sensitivity_gulkana_1/'
ds_all = xr.open_zarr(output_fp + 'output.zarr')
ds_all['time'] = ds_all.time - pd.Timedelta(hours=8)

with open(output_fp + 'config.yaml') as f:
    configs = yaml.safe_load(f) 

sites = np.unique(configs['sites'])
rgiids = np.unique(configs['rgi_ids'])

# assert ds_all.error.sum().values <= 1e-3

ab = Albedo('gulkana')
sm = SnowlineMelt('gulkana')

# only keep the necessary variables in memory
ds_all['lwc'] = ds_all['layerwater'].sum(dim='layer')
ds_all['surftype'] = ds_all['layertype'].isel(layer=0)
needed_vars = ['albedo', 'lwc', 'surftype', 'melt', 'lon', 'lat']
ds_loss = ds_all[needed_vars].load()

ab.get_model_albedo(ds_loss)
sm.get_model_snow(ds_loss)

site_arr = np.array(configs['sites'])
n_points = len(site_arr)

# every per-point parameter array in the config, excluding point
# identifiers and the redundant half of the snow_threshold pair
# (snow_threshold_low/high always move together, so one represents both)
exclude_keys = {'rgi_ids', 'sites', 'n_points', 'snow_threshold_high'}
param_keys = [
    k for k in configs
    if k not in exclude_keys
    and isinstance(configs[k], list)
    and len(configs[k]) == n_points
    and all(isinstance(v, (int, float)) for v in configs[k])
]
param_label = lambda k: 'snow_threshold' if k == 'snow_threshold_low' else k
param_vals = {k: np.array(configs[k], dtype=float) for k in param_keys}

# each site cycles through the same baseline + low/high perturbations,
# so find each site's baseline point: the one where every parameter
# sits at its per-site mode (unperturbed) simultaneously
site_mode = {}
baseline_points = []
for s in sites:
    s_idx = np.where(site_arr == s)[0]
    modes = {k: pd.Series(param_vals[k][s_idx]).mode()[0] for k in param_keys}
    site_mode[s] = modes
    at_mode = np.all([np.isclose(param_vals[k][s_idx], modes[k]) for k in param_keys], axis=0)
    baseline_points.append(s_idx[np.where(at_mode)[0][0]])

# for each parameter, find the low/high perturbation point per site
param_points = {}
for k in param_keys:
    low_pts, high_pts = [], []
    for s in sites:
        s_idx = np.where(site_arr == s)[0]
        diffs = ~np.isclose(param_vals[k][s_idx], site_mode[s][k])
        changed = s_idx[diffs]
        if len(changed) != 2:
            continue
        order = np.argsort(param_vals[k][changed])
        low_pts.append(changed[order[0]])
        high_pts.append(changed[order[1]])
    if len(low_pts) == len(sites) and len(high_pts) == len(sites):
        param_points[param_label(k)] = {'low': low_pts, 'high': high_pts}

def compute_losses(points):
    albedo_loss = ab.log_loss(ab.mod[:, points], ab.meas[:, points])
    snow_loss, melt_loss = sm.bernoulli_loss(
        sm.mod_snow[:, points], sm.meas_snow[:, points],
        sm.mod_melt[:, points], sm.meas_melt[:, points])
    return np.array([albedo_loss, snow_loss, melt_loss])

baseline_result = compute_losses(baseline_points)
results = {
    label: {level: compute_losses(pts[level]) for level in ('low', 'high')}
    for label, pts in param_points.items()
}

labels = list(results.keys())
x = np.arange(len(labels))

fig, axes = plt.subplots(3, figsize=(max(8, len(labels) * 0.6), 9),
                          gridspec_kw={'hspace': 0.3, 'wspace': 0.3}, sharex=True)

for i, name in enumerate(['Albedo Gaussian log-loss', 'Snowline Bernoulli log-loss', 'Snowmelt Bernoulli log-loss']):
    ax = axes[i]
    low_deltas = [baseline_result[i] - results[l]['low'][i] for l in labels]
    high_deltas = [baseline_result[i] - results[l]['high'][i] for l in labels]

    ax.bar(x - 0.1, low_deltas, width=0.2, label='low')
    ax.bar(x + 0.1, high_deltas, width=0.2, label='high')
    ax.axhline(0, c='k', linewidth=0.5)
    ax.set_title(name)
    ax.tick_params(length=5)

axes[-1].set_xticks(x)
axes[-1].set_xticklabels(labels, rotation=45, ha='right')
axes[0].legend()

plt.savefig(output_fp + 'sensitivity_loss.png', dpi=330, bbox_inches='tight')
plt.show()