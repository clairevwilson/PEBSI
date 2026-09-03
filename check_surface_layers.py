"""
Fast-forwards each site to SNAP_HOUR and prints the top N_LAYERS_SHOW
layers (ltype, lice, lheight) to see what the surface looks like.
"""
import os
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
os.environ.setdefault('PEBSI_NORMAL_RUN_2015_2020', '1')
import jax
jax.config.update('jax_debug_nans', False)
jax.config.update('jax_enable_x64', True)

import yaml
import numpy as np
import jax.numpy as jnp

import jax_optimize as jo
from pebsi.main import main as pebsi_main

SNAP_HOUR = int(os.environ.get('PEBSI_SNAP_HOUR', '21350'))
N_LAYERS_SHOW = int(os.environ.get('PEBSI_N_LAYERS', '6'))
TYPE_NAMES = {0: 'snow', 1: 'firn', 2: 'ice'}


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

    print(f"Surface layers at hour {SNAP_HOUR}\n", flush=True)

    for i, (glacier, site) in enumerate(site_order):
        model = build_site_model_no_density(glacier, site)
        params = model.config.params
        static_args = model.config.static_args
        dynamic_args = model.config.dynamic_args

        dargs = dynamic_args._replace(
            wind_factor=jnp.array([1.0], dtype=jnp.float32),
            kp=jnp.array([1.0], dtype=jnp.float32),
        )

        forcings = model.pack_forcings(params, model.dates[:SNAP_HOUR], 0)
        state, _ = pebsi_main(model.initial_state, forcings, model.point_attrs, static_args, dargs)

        ltype = np.asarray(state.ltype).flatten()
        lice = np.asarray(state.lice).flatten()
        lheight = np.asarray(state.lheight).flatten()

        label = f"{glacier}/{site}"
        print(f"  [{i:>2}] {label}", flush=True)
        for j in range(N_LAYERS_SHOW):
            tname = TYPE_NAMES.get(int(ltype[j]), '?')
            print(f"        layer {j}: {tname:>4}  lice={lice[j]:.4e} kg/m²  h={lheight[j]:.4e} m", flush=True)
        print(flush=True)
