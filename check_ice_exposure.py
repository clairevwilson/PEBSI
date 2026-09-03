"""
For each site, runs the forward model in monthly chunks and checks whether
the surface layer (ltype[0]) is ever bare ice (ltype==2).

Reports: fraction of months with bare ice at surface, and first occurrence.
"""
import os
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
os.environ.setdefault('PEBSI_NORMAL_RUN_2015_2020', '1')
import jax
jax.config.update('jax_debug_nans', False)
jax.config.update('jax_enable_x64', True)

import time
import yaml
import numpy as np
import jax.numpy as jnp

import jax_optimize as jo
from pebsi.main import main as pebsi_main

ICE_TYPE = 2
CHUNK_HOURS = 720  # check surface type once per month


def build_site_model_no_density(glacier, site):
    single_site_dict = {glacier: [site]}
    config_fp = jo.build_generated_config(
        single_site_dict, jo.host,
        start_date=jo.DEBUG_START_DATE, end_date=jo.DEBUG_END_DATE,
        temporal_chunk_years=1,
    )
    with open(config_fp) as f:
        cfg = yaml.safe_load(f)
    cfg['constant_snowfall_density'] = False
    with open(config_fp, 'w') as f:
        yaml.dump(cfg, f, sort_keys=False)
    return jo.init_pebsi(config_fp)


if __name__ == '__main__':
    site_dict = jo.load_reduced_site_dict(jo.REDUCED_SITES_CONFIG)
    site_order = jo.flatten_site_order(site_dict)

    print(f"{'idx':>4}  {'site':>30}  {'bare_ice_months':>15}  {'frac':>8}  {'first_hr':>10}  {'time':>6}", flush=True)
    print('-' * 82, flush=True)

    for i, (glacier, site) in enumerate(site_order):
        t0 = time.time()
        try:
            model = build_site_model_no_density(glacier, site)
            params = model.config.params
            static_args = model.config.static_args
            dynamic_args = model.config.dynamic_args

            dargs = dynamic_args._replace(
                wind_factor=jnp.array([1.0], dtype=jnp.float32),
                kp=jnp.array([1.0], dtype=jnp.float32),
            )

            n_hours = len(model.dates)
            chunk_fn = jax.jit(lambda state, forcings: pebsi_main(
                state, forcings, model.point_attrs, static_args, dargs
            ))

            state = model.initial_state
            bare_ice_months = 0
            total_months = 0
            first_hr = -1
            h = 0
            while h < n_hours:
                end = min(h + CHUNK_HOURS, n_hours)
                forcings = model.pack_forcings(params, model.dates[h:end], h)
                state, _ = chunk_fn(state, forcings)
                ltype0 = int(np.asarray(state.ltype).flatten()[0])
                total_months += 1
                if ltype0 == ICE_TYPE:
                    bare_ice_months += 1
                    if first_hr == -1:
                        first_hr = end
                h = end

            frac = bare_ice_months / total_months if total_months > 0 else 0.0
        except Exception as e:
            bare_ice_months, frac, first_hr = -1, float('nan'), -1
        elapsed = time.time() - t0
        label = f"{glacier}/{site}"
        print(f"{i:>4}  {label:>30}  {bare_ice_months:>15}  {frac:>8.4f}  {first_hr:>10}  ({elapsed:.0f}s)", flush=True)
