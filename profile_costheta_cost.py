"""
Times Terrain.load_cos_theta directly, per glacier, to get real cost
numbers for whether it's worth precomputing and caching to disk the
way shading already is.

    python profile_costheta_cost.py
"""
import time

import jax
jax.config.update('jax_enable_x64', True)

import yaml

import simulation as sim
from AD_optimize import BASE_CONFIG, HOST_PATHS, host, baseline
from point_density_convergence import START_DATE, END_DATE_RAW
from project.bayes_calibrate import align_end_date_for_daily_output
from project.glacierwide_loss import translate_rgi

SPACING = 300.0

for glacier in ('gulkana', 'kennicott'):
    rgi_id = translate_rgi[glacier]['6']
    end_date = align_end_date_for_daily_output(START_DATE, END_DATE_RAW)
    configs = dict(BASE_CONFIG)
    configs.update(HOST_PATHS[host])
    configs['start_date'] = START_DATE
    configs['end_date'] = end_date
    configs['rgi_ids'] = [rgi_id]
    configs['method_distribute'] = 'mesh'
    configs['point_spacing'] = SPACING
    configs['option_precomputed_costheta'] = False   # load it manually below, timed
    configs['wind_factor'] = baseline['wind_factor']
    configs['kp'] = baseline['kp']

    fn = f'_profile_{glacier}.yaml'
    with open(fn, 'w') as f:
        yaml.dump(configs, f, sort_keys=False)
    args = sim.get_args(parse=False).parse_args([])
    args.config_fn = fn
    model = sim.PEBSI(args)

    t0 = time.time()
    model.prepare_spatial_inputs()
    t1 = time.time()
    print(f'{glacier}: N={model.terrain.N_POINTS} points, '
          f'terrain+shading setup total: {t1 - t0:.1f}s', flush=True)

    t2 = time.time()
    model.terrain.load_cos_theta()
    t3 = time.time()
    n_pix = getattr(model.terrain, '_last_on_ice_count', None)
    print(f'{glacier}: load_cos_theta alone: {t3 - t2:.1f}s', flush=True)

    import os
    os.remove(fn)
