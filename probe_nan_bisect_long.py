"""
Bisects within hours 10962..21924 at wf=0.8125 to find the first hour where
the backward pass produces NaN. Uses a detached snapshot at hour 10962 so
each bisect evaluation only covers the remaining span.

Once the NaN hour is found, runs field decomposition to identify which
GlacierState field carries the NaN through the backward pass.
"""
import os
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
os.environ.setdefault('PEBSI_NORMAL_RUN_2015_2020', '1')
import jax
jax.config.update('jax_debug_nans', False)

import time
import numpy as np
import jax.numpy as jnp

import jax_optimize as jo
from pebsi.main import main as pebsi_main
from decompose_gradient_by_field import build_single_site_model, SITE_INDEX, KP_VAL

WF_NAN = float(os.environ.get('PEBSI_NAN_WF', '0.8125'))
LO_HOUR = int(os.environ.get('PEBSI_BISECT_LO', '10962'))
HI_HOUR = int(os.environ.get('PEBSI_BISECT_HI', '21924'))

if __name__ == '__main__':
    model = build_single_site_model(SITE_INDEX)
    params = model.config.params
    static_args = model.config.static_args
    dynamic_args = model.config.dynamic_args

    fixed_dargs = dynamic_args._replace(
        wind_factor=jnp.array([WF_NAN], dtype=jnp.float32),
        kp=jnp.array([KP_VAL], dtype=jnp.float32),
    )

    print(f"Fast-forwarding to hour {LO_HOUR} (detached)...", flush=True)
    t0 = time.time()
    forcings_pre = model.pack_forcings(params, model.dates[:LO_HOUR], 0)
    snap_lo, _ = pebsi_main(model.initial_state, forcings_pre, model.point_attrs, static_args, fixed_dargs)
    snap_lo = jax.lax.stop_gradient(snap_lo)
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    log_wf0 = jnp.log(jnp.array(WF_NAN, dtype=jnp.float32))

    def grad_window_end(end_hour):
        """Gradient of ||lice||^2 from snap_lo to end_hour w.r.t. log_wf."""
        win_forcings = model.pack_forcings(params, model.dates[LO_HOUR:end_hour], LO_HOUR)
        def loss(log_wf):
            dargs = dynamic_args._replace(
                wind_factor=jnp.exp(log_wf)[None],
                kp=jnp.array([KP_VAL], dtype=jnp.float32),
            )
            final, _ = pebsi_main(snap_lo, win_forcings, model.point_attrs, static_args, dargs)
            return jnp.sum(jnp.square(final.lice))
        try:
            g = float(jax.grad(loss)(log_wf0))
        except FloatingPointError:
            g = float('nan')
        return g

    print(f"\nBisecting hours {LO_HOUR}..{HI_HOUR} at wf={WF_NAN} for first NaN...", flush=True)
    lo, hi = LO_HOUR, HI_HOUR
    while hi - lo > 1:
        mid = (lo + hi) // 2
        t0 = time.time()
        g = grad_window_end(mid)
        finite = np.isfinite(g)
        status = 'finite' if finite else 'NaN/inf'
        print(f"  window {LO_HOUR}..{mid}: grad={g:.4e} -> {status} ({time.time()-t0:.1f}s)", flush=True)
        if finite:
            lo = mid
        else:
            hi = mid

    first_nan_hour = hi
    print(f"\nFirst NaN appears at end_hour={first_nan_hour} "
          f"(window {LO_HOUR}..{first_nan_hour})", flush=True)

    # Field decomposition over the minimal NaN window
    print(f"\nDecomposing which state field carries the NaN over {LO_HOUR}..{first_nan_hour}:", flush=True)
    win_forcings = model.pack_forcings(params, model.dates[LO_HOUR:first_nan_hour], LO_HOUR)
    float_fields = [f for f, v in snap_lo._asdict().items()
                    if jnp.issubdtype(jnp.asarray(v).dtype, jnp.floating)]

    print(f"{'field':>22} {'wf_grad':>14} {'nonfinite':>10}", flush=True)
    for field in float_fields:
        def field_loss(log_wf, field=field):
            dargs = dynamic_args._replace(
                wind_factor=jnp.exp(log_wf)[None],
                kp=jnp.array([KP_VAL], dtype=jnp.float32),
            )
            final, _ = pebsi_main(snap_lo, win_forcings, model.point_attrs, static_args, dargs)
            return jnp.sum(jnp.square(getattr(final, field)))
        t0 = time.time()
        try:
            g = float(jax.grad(field_loss)(log_wf0))
        except FloatingPointError:
            g = float('nan')
        bad = 'YES' if not np.isfinite(g) else ''
        print(f"{field:>22} {g:>14.4e} {bad:>10}  ({time.time()-t0:.1f}s)", flush=True)
