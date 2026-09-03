"""
Two-wf forward probe focused on ice layer heights near the stability threshold.

Runs wf_lo and wf_hi from SNAP_HOUR to END_HOUR, tracking:
  - diff(lice): total ice mass difference
  - min ice layer height in lo run
  - number of ice layers below ice_stability_min (~0.12 m) in lo run

This is designed to catch the moment an ice layer drops below the stability
threshold (ice_stability_min) and triggers a merge event, which is the
hypothesised mechanism for gradient explosions with constant snow density.

Uses constant_snowfall_density=150 (already in build_generated_config via jax_optimize.py).
"""
import os
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
os.environ.setdefault('PEBSI_NORMAL_RUN_2015_2020', '1')
import jax
jax.config.update('jax_debug_nans', False)
jax.config.update('jax_enable_x64', True)

import math
import numpy as np
import jax.numpy as jnp

import jax_optimize as jo
from pebsi.main import main as pebsi_main
from decompose_gradient_by_field import build_single_site_model, SITE_INDEX, KP_VAL

WF0 = float(os.environ.get('PEBSI_WF0', '1.0'))
DELTA = float(os.environ.get('PEBSI_DELTA', '1e-7'))
SNAP_HOUR = int(os.environ.get('PEBSI_SNAP_HOUR', '21340'))
END_HOUR = int(os.environ.get('PEBSI_END_HOUR', '21420'))

# ice_stability_min = 2*sqrt(4*k_ice*dt_heat / (Cp_ice*density_ice))
# dt_heat = 3600 / 5 = 720 s  (params.dt / params.n_heat_steps)
K_ICE = 2.25
CP_ICE = 2050.0
DENSITY_ICE = 900.0
DT_HEAT = 3600.0 / 5.0
ICE_STAB_MIN = 2.0 * math.sqrt(4.0 * K_ICE * DT_HEAT / (CP_ICE * DENSITY_ICE))
ICE_TYPE = 2

if __name__ == '__main__':
    print(f"ice_stability_min = {ICE_STAB_MIN:.5f} m", flush=True)

    model = build_single_site_model(SITE_INDEX)
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

    print(f"wf0={WF0}, delta={DELTA}, snap={SNAP_HOUR}, end={END_HOUR}\n", flush=True)
    hdr = (f"{'hour':>6}  {'diff(lice)':>14}  {'n_ice':>8}  "
           f"{'min_ice_h':>10}  {'n_bl':>6}  "
           f"{'min_nonice_mass':>14}  {'type':>6}  {'diff(lt0)':>12}")
    print(hdr, flush=True)
    print('-' * len(hdr), flush=True)

    for h in range(SNAP_HOUR, END_HOUR):
        forcings = model.pack_forcings(params, model.dates[h:h+1], h)
        state_lo, _ = step_fn(state_lo, forcings, dargs_lo)
        state_hi, _ = step_fn(state_hi, forcings, dargs_hi)

        lice_lo = float(jnp.sum(state_lo.lice))
        lice_hi = float(jnp.sum(state_hi.lice))
        diff_lice = lice_hi - lice_lo

        ltype_lo = np.asarray(state_lo.ltype).flatten()
        lheight_lo = np.asarray(state_lo.lheight).flatten()
        lice_arr_lo = np.asarray(state_lo.lice).flatten()

        ice_mask = ltype_lo == ICE_TYPE
        n_ice = int(ice_mask.sum())
        if n_ice > 0:
            ice_heights = lheight_lo[ice_mask]
            min_ice_h = float(ice_heights.min())
            n_below = int((ice_heights < ICE_STAB_MIN).sum())
        else:
            min_ice_h = float('nan')
            n_below = 0

        # find the minimum-mass non-ice layer (snow/firn near elimination threshold)
        non_ice_mask = (ltype_lo != ICE_TYPE) & (lice_arr_lo > 0)
        if non_ice_mask.any():
            non_ice_masses = lice_arr_lo[non_ice_mask]
            min_nonice_mass = float(non_ice_masses.min())
            min_nonice_type = int(ltype_lo[non_ice_mask][np.argmin(non_ice_masses)])
        else:
            min_nonice_mass = float('nan')
            min_nonice_type = -1

        lt0_lo = float(state_lo.ltemp.flatten()[0])
        lt0_hi = float(state_hi.ltemp.flatten()[0])
        diff_lt0 = lt0_hi - lt0_lo

        print(f"{h+1:>6}  {diff_lice:>+14.6e}  {n_ice:>8}  "
              f"{min_ice_h:>10.5f}  {n_below:>6}  "
              f"{min_nonice_mass:>14.6e}  {min_nonice_type:>6}  {diff_lt0:>+12.3e}", flush=True)
