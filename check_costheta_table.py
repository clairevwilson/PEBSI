"""
Sanity-checks Terrain.load_cos_theta's output directly, without running
the full model. The controlled experiment moved mass balance by ~0.20
m w.e./yr when switching cos_theta ordering -- about 100x the ~0.002
predicted by the earlier offline scalar-summary estimate. Before
trusting that, check whether the table itself is reasonable or whether
there's an implementation bug (sign, units, shape) inflating it.
"""
import jax
jax.config.update('jax_enable_x64', True)

import numpy as np
import yaml

import simulation as sim
from AD_optimize import BASE_CONFIG, HOST_PATHS, host, baseline
from point_density_convergence import START_DATE, END_DATE_RAW
from project.bayes_calibrate import align_end_date_for_daily_output
from project.glacierwide_loss import translate_rgi

GLACIER = 'gulkana'
SPACING = float(__import__("os").environ.get("CT_SPACING", 1200))

rgi_id = translate_rgi[GLACIER]['6']
end_date = align_end_date_for_daily_output(START_DATE, END_DATE_RAW)
configs = dict(BASE_CONFIG)
configs.update(HOST_PATHS[host])
configs['start_date'] = START_DATE
configs['end_date'] = end_date
configs['rgi_ids'] = [rgi_id]
configs['method_distribute'] = 'mesh'
configs['point_spacing'] = SPACING
configs['option_precomputed_costheta'] = True
configs['wind_factor'] = baseline['wind_factor']
configs['kp'] = baseline['kp']

fn = '_check_costheta_table.yaml'
with open(fn, 'w') as f:
    yaml.dump(configs, f, sort_keys=False)
args = sim.get_args(parse=False).parse_args([])
args.config_fn = fn
model = sim.PEBSI(args)
model.prepare_spatial_inputs()
import os; os.remove(fn)

t = model.terrain
ct = t.cos_theta_table
zen = t.solar_zenith
w = t.weight_n
slope_deg, aspect_deg = t.slope_n, t.aspect_n

print(f'N points: {t.N_POINTS}')
print(f'cos_theta_table shape: {ct.shape}')
print(f'slope_n:  mean {np.average(slope_deg, weights=w):.3f} deg, '
      f'min {slope_deg.min():.2f}, max {slope_deg.max():.2f}')
print(f'aspect_n: mean {aspect_deg.mean():.1f} deg (unweighted)')
print()

daylight = zen[0] < (np.pi / 2)
print(f'daylight steps: {daylight.sum()} of {zen.shape[1]}')

ct_day = ct[:, daylight]
zen_day = zen[:, daylight]
w_t = np.clip(np.cos(zen_day), 0, None)

print()
print('per-point time stats (daylight only):')
print(f'  cos_theta min over all points/times: {ct_day.min():.4f}')
print(f'  cos_theta max over all points/times: {ct_day.max():.4f}')
print(f'  fraction of (point,time) entries < 0: {(ct_day < 0).mean():.4f}')
print(f'  fraction < 0 but NOT clipped by cos_zen path: '
      f'{(ct_day < 0).mean():.4f}')

# insolation-weighted time-mean per point, then area-weighted
w_row_sum = w_t.sum(axis=1, keepdims=True)
w_row_sum = np.where(w_row_sum > 0, w_row_sum, 1)
pt_mean = (ct_day * w_t).sum(axis=1) / w_row_sum[:, 0]
print()
print(f'insolation-weighted time-mean cos_theta per point:')
print(f'  min {pt_mean.min():.4f}, max {pt_mean.max():.4f}')
print(f'  area-weighted mean: {np.average(pt_mean, weights=w):.4f}')
print()
print('compare to check_costheta_order.py offline estimate for h=1200:')
print('  current order area-weighted mean (ablation zone): 0.5116')
print('  correct  order area-weighted mean (ablation zone): 0.5012')
print('  (that script used a different weighting and ablation-only subset,')
print('   so exact match isn\'t expected, but the same order of magnitude is)')

# also compute the "current" (on-the-fly, biased) value the same way for comparison
az = t.solar_azimuth
slope_r = np.deg2rad(slope_deg)[:, None]
aspect_r = np.deg2rad(aspect_deg)[:, None]
ct_current = (np.cos(zen) * np.cos(slope_r)
             + np.sin(zen) * np.sin(slope_r) * np.cos(az - aspect_r))
ct_current_day = ct_current[:, daylight]
pt_mean_current = (ct_current_day * w_t).sum(axis=1) / w_row_sum[:, 0]
print()
print(f'current-order (biased) insolation-weighted time-mean cos_theta:')
print(f'  area-weighted mean: {np.average(pt_mean_current, weights=w):.4f}')
print(f'  vs precomputed:     {np.average(pt_mean, weights=w):.4f}')
print(f'  difference: {np.average(pt_mean, weights=w) - np.average(pt_mean_current, weights=w):+.4f}')

# fraction of hours where current stays positive but precomputed clips negative (or vice versa)
clip_current = (ct_current_day <= 0)
clip_precomp = (ct_day <= 0)
print()
print(f'fraction of (point,time) self-shaded (cos_theta<=0):')
print(f'  current order:     {clip_current.mean():.4f}')
print(f'  precomputed order: {clip_precomp.mean():.4f}')
