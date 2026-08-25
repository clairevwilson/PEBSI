"""
Runs a single gradient evaluation at a wf value that produced NaN in
probe_wf_sweep.py, with jax_debug_nans enabled so the FloatingPointError
traceback points to the exact backward-pass primitive.
"""
import os
os.environ.setdefault('PEBSI_NORMAL_RUN_2015_2020', '1')

import jax
import jax.numpy as jnp

import jax_optimize as jo
from pebsi.main import main as pebsi_main
from decompose_gradient_by_field import build_single_site_model, SITE_INDEX, KP_VAL, END_HOUR

WF_NAN = float(os.environ.get('PEBSI_NAN_WF', '0.8125'))

if __name__ == '__main__':
    print(f"JAX debug_nans: {jax.config.jax_debug_nans}", flush=True)
    model = build_single_site_model(SITE_INDEX)
    params = model.config.params
    static_args = model.config.static_args
    dynamic_args = model.config.dynamic_args

    forcings = model.pack_forcings(params, model.dates[:END_HOUR], 0)

    def loss(wf_val):
        dargs = dynamic_args._replace(
            wind_factor=jnp.array([wf_val], dtype=jnp.float32),
            kp=jnp.array([KP_VAL], dtype=jnp.float32),
        )
        final_state, _ = pebsi_main(model.initial_state, forcings, model.point_attrs, static_args, dargs)
        return jnp.sum(jnp.square(final_state.lice))

    print(f"Computing gradient at wf={WF_NAN}...", flush=True)
    L, g = jax.value_and_grad(loss)(WF_NAN)
    print(f"L={float(L):.8e}  dL/dwf={float(g):.6e}", flush=True)
