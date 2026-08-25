"""
Forward-only probe: fast-forwards to PEBSI_DECOMPOSE_FASTFWD_HOUR and prints
per-layer lheight/lice/ldensity for the site, to check whether near-degenerate
layers (lheight near update_layer_props' 1e-6 floor) exist where the backward
factor lice/lheight^2 would detonate. Reuses decompose_gradient_by_field's
model builder and env vars.
"""
import os
os.environ.setdefault('PEBSI_NORMAL_RUN_2015_2020', '1')

import numpy as np
import jax
import jax.numpy as jnp

import jax_optimize as jo
from pebsi.main import main as pebsi_main
from decompose_gradient_by_field import build_single_site_model, SITE_INDEX, WF_VAL, KP_VAL, FASTFWD_HOUR, END_HOUR

if __name__ == '__main__':
    model = build_single_site_model(SITE_INDEX)
    params = model.config.params
    static_args = model.config.static_args
    dynamic_args = model.config.dynamic_args

    fixed_dargs = dynamic_args._replace(
        wind_factor=jnp.array([WF_VAL], dtype=jnp.float32),
        kp=jnp.array([KP_VAL], dtype=jnp.float32),
    )

    for hour in (FASTFWD_HOUR, END_HOUR):
        forcings = model.pack_forcings(params, model.dates[:hour], 0)
        state, _ = pebsi_main(model.initial_state, forcings, model.point_attrs, static_args, fixed_dargs)
        lh = np.asarray(state.lheight[0])
        li = np.asarray(state.lice[0])
        ld = np.asarray(state.ldensity[0])
        lt = np.asarray(state.ltype[0])
        print(f"\nhour {hour}: layer (ltype, lheight, lice, ldensity), backward factor lice/lheight^2:")
        for j in range(lh.shape[0]):
            bf = li[j] / max(lh[j], 1e-6) ** 2
            print(f"  {j:2d} type={lt[j]} lh={lh[j]:.3e} lice={li[j]:.3e} rho={ld[j]:.3e} lice/lh^2={bf:.3e}")
