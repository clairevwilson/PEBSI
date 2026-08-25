"""
Finite-difference check of the full-run wf gradient of ||lice(END_HOUR)||^2.
AD (decompose_gradient_by_field.py, FASTFWD=1, END=27405) gives -1.1232e+10;
this evaluates the same loss with forward-only runs at wf * (1 +/- rel_step)
for several step sizes. Agreement -> genuine (chaotic/threshold) sensitivity;
disagreement by orders -> a wrong VJP somewhere. Same env vars as the
decompose script.
"""
import os
os.environ.setdefault('PEBSI_NORMAL_RUN_2015_2020', '1')

import numpy as np
import jax
import jax.numpy as jnp

import jax_optimize as jo
from pebsi.main import main as pebsi_main
from decompose_gradient_by_field import build_single_site_model, SITE_INDEX, WF_VAL, KP_VAL, END_HOUR

if __name__ == '__main__':
    model = build_single_site_model(SITE_INDEX)
    params = model.config.params
    static_args = model.config.static_args
    dynamic_args = model.config.dynamic_args

    forcings = model.pack_forcings(params, model.dates[:END_HOUR], 0)

    # PEBSI_FD_PARAM=wf (default) or kp
    param = os.environ.get('PEBSI_FD_PARAM', 'wf')
    base = WF_VAL if param == 'wf' else KP_VAL

    def loss(val):
        wf, kp = (val, KP_VAL) if param == 'wf' else (WF_VAL, val)
        dargs = dynamic_args._replace(
            wind_factor=jnp.array([wf], dtype=jnp.float32),
            kp=jnp.array([kp], dtype=jnp.float32),
        )
        final_state, _ = pebsi_main(model.initial_state, forcings, model.point_attrs, static_args, dargs)
        return float(jnp.sum(jnp.square(final_state.lice)))

    L0 = loss(base)
    print(f"loss({param}={base}) = {L0:.10e}", flush=True)

    # AD reference is in log-space, so also report fd * base = dL/dlog(param)
    for rel in (1e-2, 1e-3, 1e-4):
        h = base * rel
        Lp = loss(base + h)
        Lm = loss(base - h)
        fd = (Lp - Lm) / (2 * h)
        fd_log = fd * base
        print(f"rel_step={rel:.0e}: L+={Lp:.10e} L-={Lm:.10e} "
              f"dL/d{param}={fd:.4e} dL/dlog{param}={fd_log:.4e}", flush=True)
