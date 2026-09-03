"""
Runs two forward simulations with wf and wf+delta starting from snapshot(SNAP_HOUR),
tracking sum(lice) at each hour up to END_HOUR.

If sum(lice) is ever higher with higher wf, the model physics genuinely
produces more ice with more wind at that point — explaining positive gradients.
"""
import os
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
os.environ.setdefault('PEBSI_NORMAL_RUN_2015_2020', '1')
import jax
jax.config.update('jax_debug_nans', False)
jax.config.update('jax_enable_x64', True)

import numpy as np
import jax.numpy as jnp

import jax_optimize as jo
from pebsi.main import main as pebsi_main
from decompose_gradient_by_field import build_single_site_model, SITE_INDEX, KP_VAL

WF0 = float(os.environ.get('PEBSI_WF0', '0.8125'))
DELTA = float(os.environ.get('PEBSI_DELTA', '1e-7'))
SNAP_HOUR = int(os.environ.get('PEBSI_SNAP_HOUR', '13340'))
END_HOUR = int(os.environ.get('PEBSI_END_HOUR', '13466'))

if __name__ == '__main__':
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
    print(f"{'hour':>6}  {'lice_lo':>14}  {'diff(lice)':>14}  {'albedo_lo':>10}  {'diff(alb)':>12}  {'ltemp0_lo':>10}  {'diff(lt0)':>12}", flush=True)
    print('-' * 90, flush=True)

    for h in range(SNAP_HOUR, END_HOUR):
        forcings = model.pack_forcings(params, model.dates[h:h+1], h)
        state_lo, _ = step_fn(state_lo, forcings, dargs_lo)
        state_hi, _ = step_fn(state_hi, forcings, dargs_hi)

        lice_lo = float(jnp.sum(state_lo.lice))
        lice_hi = float(jnp.sum(state_hi.lice))
        diff_lice = lice_hi - lice_lo

        alb_lo = float(state_lo.albedo.flatten()[0])
        alb_hi = float(state_hi.albedo.flatten()[0])
        diff_alb = alb_hi - alb_lo

        lt0_lo = float(state_lo.ltemp.flatten()[0])
        lt0_hi = float(state_hi.ltemp.flatten()[0])
        diff_lt0 = lt0_hi - lt0_lo

        print(f"{h+1:>6}  {lice_lo:>14.4f}  {diff_lice:>+14.6e}  {alb_lo:>10.6f}  {diff_alb:>+12.3e}  {lt0_lo:>10.4f}  {diff_lt0:>+12.3e}", flush=True)
