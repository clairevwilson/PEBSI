"""
Decomposes the wind_factor gradient by STATE FIELD over the exact window
where it detonates (31942h -> 31985h, from bisect_window_length.py's
recalibrated bisection) -- three specific hypotheses (layers.py epsilon
floors, energybalance.py secant floor in two failure modes, massbalance.py
grain-size power law) have each been checked by hand and individually
ruled out for this window, despite each being confirmed to be a real,
non-hypothetical numerical hazard elsewhere in the code.

Instead of continuing to guess candidate lines, this asks the computation
directly: fast-forward ONCE (forward-only, cheap) to hour 31941, detach
that state with stop_gradient, then for EACH floating-point field of
GlacierState separately, differentiate ONLY that field's sum-of-squares
after running the real 44-hour window forward from there. Whichever
field's gradient is wildly larger than the others points directly at the
specific physics function that writes it -- narrower and more reliable
than guessing.

Reuses jax_optimize.py's model-building helpers. Lives in its own file
alongside bisect_window_length.py / inspect_secant_solver.py.
"""
import os
os.environ.setdefault('PEBSI_NORMAL_RUN_2015_2020', '1')

import time
import numpy as np
import jax
import jax.numpy as jnp

import jax_optimize as jo
from pebsi.main import main as pebsi_main

SITE_INDEX = int(os.environ.get('PEBSI_BISECT_SITE_INDEX', '5'))  # 5 = kahiltna/K53
WF_VAL = float(os.environ.get('PEBSI_BISECT_WF', '0.7879'))
KP_VAL = float(os.environ.get('PEBSI_BISECT_KP', '1.1442'))
FASTFWD_HOUR = int(os.environ.get('PEBSI_DECOMPOSE_FASTFWD_HOUR', '31941'))
END_HOUR = int(os.environ.get('PEBSI_DECOMPOSE_END_HOUR', '31985'))


def build_single_site_model(site_index):
    site_dict = jo.load_reduced_site_dict(jo.REDUCED_SITES_CONFIG)
    site_order = jo.flatten_site_order(site_dict)
    glacier, site = site_order[site_index]
    single_site_dict = {glacier: [site]}
    print(f"Building single-site model: site index {site_index} -> {glacier}/{site}", flush=True)

    config_fp = jo.build_generated_config(
        single_site_dict, jo.host, start_date=jo.DEBUG_START_DATE, end_date=jo.DEBUG_END_DATE,
        temporal_chunk_years=1,
    )
    return jo.init_pebsi(config_fp)


if __name__ == '__main__':
    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}", flush=True)
    model = build_single_site_model(SITE_INDEX)

    params = model.config.params
    static_args = model.config.static_args
    dynamic_args = model.config.dynamic_args

    print(f"Fast-forwarding to hour {FASTFWD_HOUR} (forward-only)...", flush=True)
    forcings_prefix = model.pack_forcings(params, model.dates[:FASTFWD_HOUR], 0)
    fixed_dargs = dynamic_args._replace(
        wind_factor=jnp.array([WF_VAL], dtype=jnp.float32),
        kp=jnp.array([KP_VAL], dtype=jnp.float32),
    )
    snapshot_state, _ = pebsi_main(model.initial_state, forcings_prefix, model.point_attrs, static_args, fixed_dargs)
    snapshot_state = jax.lax.stop_gradient(snapshot_state)

    n_window = END_HOUR - FASTFWD_HOUR
    window_forcings = model.pack_forcings(params, model.dates[FASTFWD_HOUR:END_HOUR], FASTFWD_HOUR)
    log_wf = jnp.log(jnp.array([WF_VAL], dtype=jnp.float32))
    log_kp = jnp.log(jnp.array([KP_VAL], dtype=jnp.float32))

    # PEBSI_DECOMPOSE_FIELDS=lgrainsize (or a comma list) restricts to just
    # those fields -- ~50s/field, so checking one instead of all 22 is a
    # ~20min -> ~1min difference when re-verifying a specific fix
    only_fields_env = os.environ.get('PEBSI_DECOMPOSE_FIELDS')
    only_fields = only_fields_env.split(',') if only_fields_env else None

    field_names = [
        f for f, v in snapshot_state._asdict().items()
        if jnp.issubdtype(jnp.asarray(v).dtype, jnp.floating)
        and (only_fields is None or f in only_fields)
    ]
    print(f"\nDecomposing gradient over hours {FASTFWD_HOUR}..{END_HOUR} "
          f"({n_window}h) across {len(field_names)} floating fields:\n", flush=True)
    print(f"{'field':>20} {'wf_grad':>14} {'kp_grad':>14} {'nonfinite':>10}", flush=True)

    results = []
    for field in field_names:
        def field_loss(log_wf, log_kp, field=field):
            wf = jnp.exp(log_wf)
            kp = jnp.exp(log_kp)
            dargs = dynamic_args._replace(wind_factor=wf, kp=kp)
            final_state, _ = pebsi_main(snapshot_state, window_forcings, model.point_attrs, static_args, dargs)
            value = getattr(final_state, field)
            return jnp.sum(jnp.square(value))

        t0 = time.time()
        try:
            grads = jax.grad(field_loss, argnums=(0, 1))(log_wf, log_kp)
            wf_grad = float(grads[0][0])
            kp_grad = float(grads[1][0])
            nonfinite = not (np.isfinite(wf_grad) and np.isfinite(kp_grad))
        except FloatingPointError:
            wf_grad = kp_grad = float('inf')
            nonfinite = True
        jax.clear_caches()
        elapsed = time.time() - t0

        print(f"{field:>20} {wf_grad:>14.4e} {kp_grad:>14.4e} {str(nonfinite):>10}  ({elapsed:.1f}s)",
              flush=True)
        results.append((field, wf_grad, kp_grad, nonfinite))

    print("\nRanked by |wf_grad| (largest first):", flush=True)
    for field, wf_grad, kp_grad, nonfinite in sorted(results, key=lambda r: -abs(r[1])):
        print(f"  {field:>20}: wf_grad={wf_grad:.4e} kp_grad={kp_grad:.4e}", flush=True)
