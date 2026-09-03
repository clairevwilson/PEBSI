"""
Two-wf forward probe tracking surface layer (layer 0) mass and type for
both lo and hi runs. Tests whether the discrete diff(lice) jump coincides
with one run's surface layer crossing min_layer_mass = 0.001 kg/m².
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
from decompose_gradient_by_field import build_single_site_model, SITE_INDEX, KP_VAL

WF0 = float(os.environ.get('PEBSI_WF0', '0.8125'))
DELTA = float(os.environ.get('PEBSI_DELTA', '1e-7'))
SNAP_HOUR = int(os.environ.get('PEBSI_SNAP_HOUR', '21340'))
END_HOUR = int(os.environ.get('PEBSI_END_HOUR', '21420'))
MIN_LAYER_MASS = 0.001  # kg/m²
TYPE_NAMES = {0: 'snow', 1: 'firn', 2: 'ice'}


def build_no_density(site_index):
    site_dict = jo.load_reduced_site_dict(jo.REDUCED_SITES_CONFIG)
    site_order = jo.flatten_site_order(site_dict)
    glacier, site = site_order[site_index]
    single_site_dict = {glacier: [site]}
    print(f"Site: {glacier}/{site}", flush=True)
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
    model = build_no_density(SITE_INDEX)
    params = model.config.params
    static_args = model.config.static_args
    dynamic_args = model.config.dynamic_args

    dargs_f32 = dynamic_args._replace(
        wind_factor=jnp.array([WF0], dtype=jnp.float32),
        kp=jnp.array([KP_VAL], dtype=jnp.float32),
    )

    print(f"Fast-forwarding to hour {SNAP_HOUR}...", flush=True)
    forcings_pre = model.pack_forcings(params, model.dates[:SNAP_HOUR], 0)
    snap, _ = pebsi_main(model.initial_state, forcings_pre, model.point_attrs, static_args, dargs_f32)
    snap = jax.lax.stop_gradient(snap)

    snap_f64 = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64) if jnp.issubdtype(x.dtype, jnp.floating) else x,
        snap
    )

    dargs_lo = dynamic_args._replace(
        wind_factor=jnp.array([WF0], dtype=jnp.float64),
        kp=jnp.array([KP_VAL], dtype=jnp.float64),
    )
    dargs_hi = dynamic_args._replace(
        wind_factor=jnp.array([WF0 + DELTA], dtype=jnp.float64),
        kp=jnp.array([KP_VAL], dtype=jnp.float64),
    )

    step_fn = jax.jit(lambda state, forcings, dargs: pebsi_main(
        state, forcings, model.point_attrs, static_args, dargs
    ))

    state_lo = snap_f64
    state_hi = snap_f64

    print(f"wf0={WF0}  delta={DELTA}  snap={SNAP_HOUR}  end={END_HOUR}", flush=True)
    print(f"min_layer_mass = {MIN_LAYER_MASS}\n", flush=True)
    hdr = (f"{'hour':>6}  {'diff(lice)':>14}  "
           f"{'lice0_lo':>12}  {'type0_lo':>8}  "
           f"{'lice0_hi':>12}  {'type0_hi':>8}  {'elim?':>6}")
    print(hdr, flush=True)
    print('-' * len(hdr), flush=True)

    for h in range(SNAP_HOUR, END_HOUR):
        forcings = model.pack_forcings(params, model.dates[h:h+1], h)
        state_lo, _ = step_fn(state_lo, forcings, dargs_lo)
        state_hi, _ = step_fn(state_hi, forcings, dargs_hi)

        lice_total_lo = float(jnp.sum(state_lo.lice))
        lice_total_hi = float(jnp.sum(state_hi.lice))
        diff_lice = lice_total_hi - lice_total_lo

        lice0_lo = float(np.asarray(state_lo.lice).flatten()[0])
        lice0_hi = float(np.asarray(state_hi.lice).flatten()[0])
        type0_lo = TYPE_NAMES.get(int(np.asarray(state_lo.ltype).flatten()[0]), '?')
        type0_hi = TYPE_NAMES.get(int(np.asarray(state_hi.ltype).flatten()[0]), '?')

        # flag if one run is near or below min_layer_mass while the other isn't
        lo_near = lice0_lo < 10 * MIN_LAYER_MASS
        hi_near = lice0_hi < 10 * MIN_LAYER_MASS
        elim_flag = '***' if (lo_near or hi_near) else ''

        print(f"{h+1:>6}  {diff_lice:>+14.6e}  "
              f"{lice0_lo:>12.6e}  {type0_lo:>8}  "
              f"{lice0_hi:>12.6e}  {type0_hi:>8}  {elim_flag:>6}", flush=True)
