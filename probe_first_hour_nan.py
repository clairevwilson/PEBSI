"""
Decomposes which output field carries the NaN when differentiating through
the very first timestep (initial_state -> hour 1) at wf=0.8125.
"""
import os
os.environ.setdefault('PEBSI_NORMAL_RUN_2015_2020', '1')
import jax
jax.config.update('jax_debug_nans', False)

import numpy as np
import jax.numpy as jnp

import jax_optimize as jo
from pebsi.main import main as pebsi_main
from decompose_gradient_by_field import build_single_site_model, SITE_INDEX, KP_VAL

WF = float(os.environ.get('PEBSI_NAN_WF', '0.8125'))

if __name__ == '__main__':
    model = build_single_site_model(SITE_INDEX)
    params = model.config.params
    static_args = model.config.static_args
    dynamic_args = model.config.dynamic_args

    forcings_1h = model.pack_forcings(params, model.dates[:1], 0)
    log_wf0 = jnp.log(jnp.array(WF, dtype=jnp.float32))

    print(f"Decomposing NaN over first timestep (initial_state -> hour 1) at wf={WF}:\n")
    print(f"{'field':>22} {'wf_grad':>14} {'NaN?':>6}", flush=True)

    float_fields = [f for f, v in model.initial_state._asdict().items()
                    if jnp.issubdtype(jnp.asarray(v).dtype, jnp.floating)]

    for field in float_fields:
        def field_loss(log_wf, field=field):
            dargs = dynamic_args._replace(
                wind_factor=jnp.exp(log_wf)[None],
                kp=jnp.array([KP_VAL], dtype=jnp.float32),
            )
            final, _ = pebsi_main(model.initial_state, forcings_1h, model.point_attrs, static_args, dargs)
            return jnp.sum(jnp.square(getattr(final, field)))

        try:
            g = float(jax.grad(field_loss)(log_wf0))
        except FloatingPointError:
            g = float('nan')
        nan = 'YES' if not np.isfinite(g) else ''
        print(f"{field:>22} {g:>14.4e} {nan:>6}", flush=True)
