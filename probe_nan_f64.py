"""
Tests if the NaN at wf=0.8125, window 10962..13466 is float32 overflow by
running the same gradient in float64. If float64 gives a finite (large) value,
the NaN is purely a float32 precision issue. If float64 also gives NaN/inf,
there's a genuine mathematical singularity (jnp.where footgun or division).
"""
import os
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
os.environ.setdefault('PEBSI_NORMAL_RUN_2015_2020', '1')

import jax
jax.config.update('jax_debug_nans', False)
jax.config.update('jax_enable_x64', True)  # enable float64

import numpy as np
import jax.numpy as jnp

import jax_optimize as jo
from pebsi.main import main as pebsi_main
from decompose_gradient_by_field import build_single_site_model, SITE_INDEX, KP_VAL

WF_NAN = float(os.environ.get('PEBSI_NAN_WF', '0.8125'))
LO_HOUR = int(os.environ.get('PEBSI_BISECT_LO', '10962'))
END_HOUR = int(os.environ.get('PEBSI_END_HOUR', '13466'))

if __name__ == '__main__':
    model = build_single_site_model(SITE_INDEX)
    params = model.config.params
    static_args = model.config.static_args
    dynamic_args = model.config.dynamic_args

    fixed_dargs = dynamic_args._replace(
        wind_factor=jnp.array([WF_NAN], dtype=jnp.float32),
        kp=jnp.array([KP_VAL], dtype=jnp.float32),
    )

    print(f"Fast-forwarding to hour {LO_HOUR} (float32 forward, detached)...", flush=True)
    forcings_pre = model.pack_forcings(params, model.dates[:LO_HOUR], 0)
    snap_lo, _ = pebsi_main(model.initial_state, forcings_pre, model.point_attrs, static_args, fixed_dargs)
    snap_lo = jax.lax.stop_gradient(snap_lo)

    # Cast snapshot to float64 for the gradient computation
    def cast_to_f64(arr):
        if jnp.issubdtype(arr.dtype, jnp.floating):
            return arr.astype(jnp.float64)
        return arr
    snap_lo_f64 = jax.tree_util.tree_map(cast_to_f64, snap_lo)

    log_wf0_f64 = jnp.float64(jnp.log(jnp.float32(WF_NAN)))
    win_forcings = model.pack_forcings(params, model.dates[LO_HOUR:END_HOUR], LO_HOUR)

    def loss_f64(log_wf):
        dargs = dynamic_args._replace(
            wind_factor=jnp.exp(log_wf)[None].astype(jnp.float64),
            kp=jnp.array([KP_VAL], dtype=jnp.float64),
        )
        final, _ = pebsi_main(snap_lo_f64, win_forcings, model.point_attrs, static_args, dargs)
        return jnp.sum(jnp.square(final.lice.astype(jnp.float64)))

    print(f"Computing float64 gradient over {LO_HOUR}..{END_HOUR} (window {END_HOUR-LO_HOUR}h)...", flush=True)
    try:
        g = float(jax.grad(loss_f64)(log_wf0_f64))
        print(f"  float64 grad = {g:.6e}", flush=True)
        if np.isfinite(g):
            print("  -> FINITE: NaN in float32 is pure float32 overflow (not a mathematical singularity)", flush=True)
        else:
            print("  -> NaN/inf in float64 too: genuine mathematical singularity exists", flush=True)
    except Exception as e:
        print(f"  Exception: {e}", flush=True)

    # Also check one step earlier (13465) for comparison
    END_HOUR_PREV = END_HOUR - 1
    win_forcings_prev = model.pack_forcings(params, model.dates[LO_HOUR:END_HOUR_PREV], LO_HOUR)

    def loss_f64_prev(log_wf):
        dargs = dynamic_args._replace(
            wind_factor=jnp.exp(log_wf)[None].astype(jnp.float64),
            kp=jnp.array([KP_VAL], dtype=jnp.float64),
        )
        final, _ = pebsi_main(snap_lo_f64, win_forcings_prev, model.point_attrs, static_args, dargs)
        return jnp.sum(jnp.square(final.lice.astype(jnp.float64)))

    print(f"\nFor comparison, float64 gradient over {LO_HOUR}..{END_HOUR_PREV}:", flush=True)
    try:
        g_prev = float(jax.grad(loss_f64_prev)(log_wf0_f64))
        print(f"  float64 grad = {g_prev:.6e}", flush=True)
    except Exception as e:
        print(f"  Exception: {e}", flush=True)
