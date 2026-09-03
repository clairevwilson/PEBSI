"""
Computes d(sum(lice)) / d(wf) for the full 5-year window at every site,
using the normal model (no constant_snowfall_density).

Reports gradient magnitude per site so we can see which ones blow up.
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

WF = float(os.environ.get('PEBSI_WF', '1.0'))
KP = float(os.environ.get('PEBSI_KP', '1.0'))
MELT_SMOOTH_ALPHA = int(os.environ.get('PEBSI_MELT_SMOOTH_ALPHA', '0'))


def build_site_model_no_density(glacier, site):
    single_site_dict = {glacier: [site]}
    config_fp = jo.build_generated_config(
        single_site_dict, jo.host,
        start_date=jo.DEBUG_START_DATE, end_date=jo.DEBUG_END_DATE,
        temporal_chunk_years=1,
    )
    # patch out constant_snowfall_density
    with open(config_fp) as f:
        cfg = yaml.safe_load(f)
    cfg['constant_snowfall_density'] = False
    if MELT_SMOOTH_ALPHA > 0:
        cfg['melt_smooth_alpha'] = MELT_SMOOTH_ALPHA
    with open(config_fp, 'w') as f:
        yaml.dump(cfg, f, sort_keys=False)
    return jo.init_pebsi(config_fp)


if __name__ == '__main__':
    site_dict = jo.load_reduced_site_dict(jo.REDUCED_SITES_CONFIG)
    site_order = jo.flatten_site_order(site_dict)
    n_sites = len(site_order)
    print(f"Sites: {n_sites}  WF={WF}  KP={KP}\n", flush=True)
    print(f"{'idx':>4}  {'site':>30}  {'grad':>16}  {'|grad|':>14}  {'time':>6}", flush=True)
    print('-' * 80, flush=True)

    for i, (glacier, site) in enumerate(site_order):
        t0 = time.time()
        try:
            model = build_site_model_no_density(glacier, site)
            params = model.config.params
            static_args = model.config.static_args
            dynamic_args = model.config.dynamic_args

            n_hours = len(model.dates)
            forcings_all = model.pack_forcings(params, model.dates, 0)

            def loss(wf_val):
                dargs = dynamic_args._replace(
                    wind_factor=jnp.array([wf_val], dtype=jnp.float64),
                    kp=jnp.array([KP], dtype=jnp.float64),
                )
                init_f64 = jax.tree_util.tree_map(
                    lambda x: x.astype(jnp.float64) if jnp.issubdtype(x.dtype, jnp.floating) else x,
                    model.initial_state
                )
                final, _ = pebsi_main(init_f64, forcings_all, model.point_attrs, static_args, dargs)
                return jnp.sum(final.lice.astype(jnp.float64))

            grad_fn = jax.jit(jax.grad(loss))
            g = float(grad_fn(jnp.float64(WF)))
        except Exception as e:
            print(f"  EXCEPTION: {type(e).__name__}: {e}", flush=True)
            g = float('nan')
        elapsed = time.time() - t0
        label = f"{glacier}/{site}"
        print(f"{i:>4}  {label:>30}  {g:>+16.4e}  {abs(g):>14.4e}  ({elapsed:.0f}s)", flush=True)
