"""
Prints per-hour state around the gradient jump at step 13465->13466 (wf=0.8125).
Runs forward hour-by-hour from LOOK_START to LOOK_END and prints
albedo, snow/ice layer structure, melt, and surface conditions.
"""
import os
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
os.environ.setdefault('PEBSI_NORMAL_RUN_2015_2020', '1')
import jax
jax.config.update('jax_debug_nans', False)

import numpy as np
import jax.numpy as jnp

import jax_optimize as jo
from pebsi.main import main as pebsi_main
from decompose_gradient_by_field import build_single_site_model, SITE_INDEX, KP_VAL

WF = float(os.environ.get('PEBSI_NAN_WF', '0.8125'))
SNAP_HOUR = int(os.environ.get('PEBSI_SNAP_HOUR', '10962'))
LOOK_START = int(os.environ.get('PEBSI_LOOK_START', '13450'))
LOOK_END = int(os.environ.get('PEBSI_LOOK_END', '13480'))

if __name__ == '__main__':
    model = build_single_site_model(SITE_INDEX)
    params = model.config.params
    static_args = model.config.static_args
    dynamic_args = model.config.dynamic_args

    dargs = dynamic_args._replace(
        wind_factor=jnp.array([WF], dtype=jnp.float32),
        kp=jnp.array([KP_VAL], dtype=jnp.float32),
    )

    print(f"Fast-forwarding to hour {SNAP_HOUR}...", flush=True)
    forcings_pre = model.pack_forcings(params, model.dates[:SNAP_HOUR], 0)
    state, _ = pebsi_main(model.initial_state, forcings_pre, model.point_attrs, static_args, dargs)
    state = jax.lax.stop_gradient(state)
    print(f"Done. Running hour-by-hour from {LOOK_START} to {LOOK_END}.\n", flush=True)

    # fast-forward to LOOK_START
    if LOOK_START > SNAP_HOUR:
        forcings_mid = model.pack_forcings(params, model.dates[SNAP_HOUR:LOOK_START], SNAP_HOUR)
        state, _ = pebsi_main(state, forcings_mid, model.point_attrs, static_args, dargs)
        state = jax.lax.stop_gradient(state)

    # header
    print(f"{'hour':>6}  {'albedo':>7}  {'n_ice_layers':>12}  {'sum_lice_kg':>14}  "
          f"{'top_lice_kg':>12}  {'top_ltemp':>10}  {'top_ldensity':>12}", flush=True)
    print('-' * 90, flush=True)

    for hour in range(LOOK_START, LOOK_END + 1):
        forcings_step = model.pack_forcings(params, model.dates[hour:hour+1], hour)
        next_state, _ = pebsi_main(state, forcings_step, model.point_attrs, static_args, dargs)

        lice = np.array(state.lice).flatten()
        ltemp = np.array(state.ltemp).flatten()
        ldensity = np.array(state.ldensity).flatten()
        albedo = float(np.array(state.albedo).flatten()[0])

        active = lice > 0
        n_ice = int(active.sum())
        sum_ice = float(lice[active].sum()) if n_ice > 0 else 0.0
        top_idx = int(np.argmax(active)) if n_ice > 0 else 0
        top_ice = float(lice[top_idx]) if n_ice > 0 else 0.0
        top_temp = float(ltemp[top_idx]) if n_ice > 0 else float('nan')
        top_dens = float(ldensity[top_idx]) if n_ice > 0 else float('nan')

        marker = ' <<<<' if hour == 13465 else ''
        print(f"{hour:>6}  {albedo:>7.4f}  {n_ice:>12d}  {sum_ice:>14.1f}  "
              f"{top_ice:>12.1f}  {top_temp:>10.4f}  {top_dens:>12.1f}{marker}", flush=True)

        state = jax.lax.stop_gradient(next_state)

    print('\nNote: "top" = lowest-index active layer (surface-most).', flush=True)
    print('ldensity ~917 kg/m³ = pure ice; ~300-600 = snow/firn.', flush=True)
