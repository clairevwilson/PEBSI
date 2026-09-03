"""
Binary-searches hours 1..21 to find the first hour where the backward pass
goes NaN at wf=0.8125. Then runs decompose_gradient_by_field over that
single-hour window to identify which state field carries the NaN.
"""
import os
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

if __name__ == '__main__':
    model = build_single_site_model(SITE_INDEX)
    params = model.config.params
    static_args = model.config.static_args
    dynamic_args = model.config.dynamic_args

    fixed_dargs = dynamic_args._replace(
        wind_factor=jnp.array([WF_NAN], dtype=jnp.float32),
        kp=jnp.array([KP_VAL], dtype=jnp.float32),
    )

    def grad_over_window(start, end):
        forcings_pre = model.pack_forcings(params, model.dates[:start], 0)
        snap, _ = pebsi_main(model.initial_state, forcings_pre, model.point_attrs, static_args, fixed_dargs)
        snap = jax.lax.stop_gradient(snap)
        win_forcings = model.pack_forcings(params, model.dates[start:end], start)

        def loss(log_wf):
            wf = jnp.exp(log_wf)
            dargs = dynamic_args._replace(
                wind_factor=jnp.array([wf], dtype=jnp.float32),
                kp=jnp.array([KP_VAL], dtype=jnp.float32),
            )
            final, _ = pebsi_main(snap, win_forcings, model.point_attrs, static_args, dargs)
            return jnp.sum(jnp.square(final.lice))

        log_wf = jnp.log(jnp.array(WF_NAN, dtype=jnp.float32))
        try:
            g = float(jax.grad(loss)(log_wf))
        except FloatingPointError:
            g = float('nan')
        return g

    # binary search over hours 1..21
    print(f"Binary searching hours 1..21 at wf={WF_NAN} for first NaN backward pass...", flush=True)
    lo, hi = 1, 21
    while lo < hi:
        mid = (lo + hi) // 2
        t0 = time.time()
        g = grad_over_window(1, mid + 1)
        finite = np.isfinite(g)
        print(f"  window 1..{mid+1}: grad={g:.4e} -> {'finite' if finite else 'NaN/inf'} ({time.time()-t0:.1f}s)", flush=True)
        if finite:
            lo = mid + 1
        else:
            hi = mid

    first_bad_hour = lo + 1
    print(f"\nFirst NaN appears ending at hour {first_bad_hour} (window 1..{first_bad_hour})", flush=True)

    # decompose which field carries the NaN over that minimal window
    print(f"\nDecomposing which state field carries the NaN over window 1..{first_bad_hour}:", flush=True)

    forcings_pre = model.pack_forcings(params, model.dates[:1], 0)
    snap, _ = pebsi_main(model.initial_state, forcings_pre, model.point_attrs, static_args, fixed_dargs)
    snap = jax.lax.stop_gradient(snap)
    win_forcings = model.pack_forcings(params, model.dates[1:first_bad_hour], 1)

    log_wf0 = jnp.log(jnp.array(WF_NAN, dtype=jnp.float32))

    float_fields = [f for f, v in snap._asdict().items()
                    if jnp.issubdtype(jnp.asarray(v).dtype, jnp.floating)]

    print(f"{'field':>22} {'wf_grad':>14} {'nonfinite':>10}", flush=True)
    for field in float_fields:
        def field_loss(log_wf, field=field):
            wf = jnp.exp(log_wf)
            dargs = dynamic_args._replace(
                wind_factor=jnp.array([wf], dtype=jnp.float32),
                kp=jnp.array([KP_VAL], dtype=jnp.float32),
            )
            final, _ = pebsi_main(snap, win_forcings, model.point_attrs, static_args, dargs)
            return jnp.sum(jnp.square(getattr(final, field)))
        t0 = time.time()
        try:
            g = float(jax.grad(field_loss)(log_wf0))
        except FloatingPointError:
            g = float('nan')
        print(f"{field:>22} {g:>14.4e} {'YES' if not np.isfinite(g) else '':>10} ({time.time()-t0:.1f}s)", flush=True)
