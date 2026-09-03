"""
Finds WHERE in the backward pass the gradient explosion accumulates.
Fixes end_hour=13466, scans start_hour from 13466 back to SNAP_HOUR.
Prints d(loss)/d(wf) for window [start, 13466] at each start hour.

If gradient jumps at a specific start hour, that's the step where the
backward pass accumulates the explosion.
"""
import os
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
os.environ.setdefault('PEBSI_NORMAL_RUN_2015_2020', '1')
import jax
jax.config.update('jax_debug_nans', False)
jax.config.update('jax_enable_x64', True)

import time
import numpy as np
import jax.numpy as jnp

import jax_optimize as jo
from pebsi.main import main as pebsi_main
from decompose_gradient_by_field import build_single_site_model, SITE_INDEX, KP_VAL

WF = float(os.environ.get('PEBSI_NAN_WF', '0.8125'))
SNAP_HOUR = int(os.environ.get('PEBSI_SNAP_HOUR', '10962'))
END_HOUR = int(os.environ.get('PEBSI_END_HOUR', '13466'))
STEP = int(os.environ.get('PEBSI_STEP', '50'))  # hours between probes

if __name__ == '__main__':
    model = build_single_site_model(SITE_INDEX)
    params = model.config.params
    static_args = model.config.static_args
    dynamic_args = model.config.dynamic_args

    dargs_f32 = dynamic_args._replace(
        wind_factor=jnp.array([WF], dtype=jnp.float32),
        kp=jnp.array([KP_VAL], dtype=jnp.float32),
    )

    # pre-compute snapshots at each probe start hour
    probe_hours = list(range(SNAP_HOUR, END_HOUR, STEP)) + [END_HOUR - 1]
    probe_hours = sorted(set(probe_hours))

    print(f"Pre-computing snapshots at {len(probe_hours)} start hours...", flush=True)
    snapshots = {}
    state = model.initial_state
    prev_hour = 0
    t0 = time.time()
    for h in probe_hours:
        if h > prev_hour:
            forcings = model.pack_forcings(params, model.dates[prev_hour:h], prev_hour)
            state, _ = pebsi_main(state, forcings, model.point_attrs, static_args, dargs_f32)
            state = jax.lax.stop_gradient(state)
        snapshots[h] = state
        prev_hour = h
    print(f"  done in {time.time()-t0:.1f}s\n", flush=True)

    # for each start hour, compute grad over [start, END_HOUR]
    print(f"{'start_hour':>12} {'window_len':>10} {'grad (f64)':>16} {'|grad|':>14}", flush=True)
    print('-' * 60, flush=True)

    def window_grad(wf_val, snap, forcings):
        snap_f64 = jax.tree_util.tree_map(
            lambda x: x.astype(jnp.float64) if jnp.issubdtype(x.dtype, jnp.floating) else x,
            snap
        )
        dargs = dynamic_args._replace(
            wind_factor=jnp.array([wf_val], dtype=jnp.float64),
            kp=jnp.array([KP_VAL], dtype=jnp.float64),
        )
        final, _ = pebsi_main(snap_f64, forcings, model.point_attrs, static_args, dargs)
        return jnp.sum(final.lice.astype(jnp.float64))

    grad_fn = jax.jit(jax.grad(window_grad))

    for h in reversed(probe_hours):
        snap = snapshots[h]
        forcings = model.pack_forcings(params, model.dates[h:END_HOUR], h)
        t0 = time.time()
        try:
            g = float(grad_fn(jnp.float64(WF), snap, forcings))
        except Exception as e:
            g = float('nan')
        elapsed = time.time() - t0
        window = END_HOUR - h
        print(f"{h:>12} {window:>10} {g:>+16.4e} {abs(g):>14.4e}  ({elapsed:.0f}s)", flush=True)
