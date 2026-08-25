"""
Differentiates ||lice(END_HOUR)||^2 w.r.t. each float field of the SNAPSHOT
state at FASTFWD_HOUR (wf/kp held fixed), i.e. the window's state-to-state
Jacobian row that the full-run gradient actually multiplies through. The
per-field wf-gradient decompose detaches the snapshot, so it is blind to
pathological d(out)/d(state_in) entries -- this probe sees them directly,
and prints the argmax layer alongside its forward values (lheight, lice)
to expose any tiny-denominator amplification. Same env vars as
decompose_gradient_by_field.py.
"""
import os
os.environ.setdefault('PEBSI_NORMAL_RUN_2015_2020', '1')

import numpy as np
import jax
import jax.numpy as jnp

import jax_optimize as jo
from pebsi.main import main as pebsi_main
from decompose_gradient_by_field import build_single_site_model, SITE_INDEX, WF_VAL, KP_VAL, FASTFWD_HOUR, END_HOUR

TARGET_FIELD = os.environ.get('PEBSI_JAC_TARGET_FIELD', 'lice')
# comma-separated snapshot hours to sweep (END_HOUR fixed): tests whether the
# adjoint compounds exponentially with window span vs. one bad op in a window
START_HOURS = [int(h) for h in os.environ.get('PEBSI_JAC_START_HOURS', str(FASTFWD_HOUR)).split(',')]

if __name__ == '__main__':
    model = build_single_site_model(SITE_INDEX)
    params = model.config.params
    static_args = model.config.static_args
    dynamic_args = model.config.dynamic_args

    fixed_dargs = dynamic_args._replace(
        wind_factor=jnp.array([WF_VAL], dtype=jnp.float32),
        kp=jnp.array([KP_VAL], dtype=jnp.float32),
    )

    for start_hour in START_HOURS:
        t0 = __import__('time').time()
        forcings_prefix = model.pack_forcings(params, model.dates[:start_hour], 0)
        snapshot_state, _ = pebsi_main(model.initial_state, forcings_prefix, model.point_attrs, static_args, fixed_dargs)
        snapshot_state = jax.lax.stop_gradient(snapshot_state)

        window_forcings = model.pack_forcings(params, model.dates[start_hour:END_HOUR], start_hour)

        float_fields = {
            f: v for f, v in snapshot_state._asdict().items()
            if jnp.issubdtype(jnp.asarray(v).dtype, jnp.floating)
        }

        def loss(ff):
            st = snapshot_state._replace(**ff)
            final_state, _ = pebsi_main(st, window_forcings, model.point_attrs, static_args, fixed_dargs)
            return jnp.sum(jnp.square(getattr(final_state, TARGET_FIELD)))

        grads = jax.grad(loss)(float_fields)

        rows = []
        for f, g in grads.items():
            g = np.asarray(g)
            rows.append((f, np.abs(g).max(), g))
        rows.sort(key=lambda r: -r[1])

        lh = np.asarray(snapshot_state.lheight)
        li = np.asarray(snapshot_state.lice)
        span = END_HOUR - start_hour
        print(f"\n=== start {start_hour} -> {END_HOUR} (span {span}h, {__import__('time').time()-t0:.1f}s) "
              f"d ||{TARGET_FIELD}||^2 / d state, top 6:", flush=True)
        for f, mx, g in rows[:6]:
            idx = np.unravel_index(np.abs(g).argmax(), g.shape)
            layer = idx[-1] if g.ndim > 1 else None
            ctx = f"lh={lh[0, layer]:.3e} lice={li[0, layer]:.3e}" if layer is not None and lh.ndim > 1 and layer < lh.shape[1] else ""
            print(f"{f:>22} {mx:12.4e}  {str(idx):>7} {ctx}", flush=True)
