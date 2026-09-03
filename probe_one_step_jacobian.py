"""
Computes the one-step state Jacobian at hour 13465->13466 at wf=0.8125:
for each float state field at hour 13465, what is the sensitivity of
||lice_{13466}||^2 to a uniform scaling of that field?

If a single field has a large Jacobian entry, that names the physics
responsible for the enormous backward-pass signal at that timestep.

Also computes the direct one-step wf-gradient for comparison.
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
STEP_HOUR = int(os.environ.get('PEBSI_STEP_HOUR', '13465'))  # state we perturb
END_HOUR = STEP_HOUR + 1  # 13466

if __name__ == '__main__':
    model = build_single_site_model(SITE_INDEX)
    params = model.config.params
    static_args = model.config.static_args
    dynamic_args = model.config.dynamic_args

    fixed_dargs = dynamic_args._replace(
        wind_factor=jnp.array([WF], dtype=jnp.float32),
        kp=jnp.array([KP_VAL], dtype=jnp.float32),
    )

    print(f"Fast-forwarding to hour {STEP_HOUR} (forward-only, detached)...", flush=True)
    t0 = time.time()
    forcings_pre = model.pack_forcings(params, model.dates[:STEP_HOUR], 0)
    snap_step, _ = pebsi_main(model.initial_state, forcings_pre, model.point_attrs, static_args, fixed_dargs)
    snap_step = jax.lax.stop_gradient(snap_step)
    print(f"  done in {time.time()-t0:.1f}s\n", flush=True)

    one_step_forcings = model.pack_forcings(params, model.dates[STEP_HOUR:END_HOUR], STEP_HOUR)

    # ------------------------------------------------------------------ #
    # 1. Direct one-step wf-gradient (for comparison)
    # ------------------------------------------------------------------ #
    def loss_wf(log_wf):
        dargs = dynamic_args._replace(
            wind_factor=jnp.exp(log_wf)[None].astype(jnp.float64),
            kp=jnp.array([KP_VAL], dtype=jnp.float64),
        )
        snap_f64 = jax.tree_util.tree_map(
            lambda x: x.astype(jnp.float64) if jnp.issubdtype(x.dtype, jnp.floating) else x,
            snap_step
        )
        final, _ = pebsi_main(snap_f64, one_step_forcings, model.point_attrs, static_args, dargs)
        return jnp.sum(jnp.square(final.lice.astype(jnp.float64)))

    log_wf0 = jnp.float64(jnp.log(jnp.float32(WF)))
    t0 = time.time()
    g_wf = float(jax.grad(loss_wf)(log_wf0))
    print(f"Direct one-step wf-gradient (float64): {g_wf:.4e}  ({time.time()-t0:.1f}s)\n", flush=True)

    # ------------------------------------------------------------------ #
    # 2. One-step state Jacobian: d(||lice_end||^2) / d(alpha_field)
    #    where snap_perturbed[field] = snap[field] * (1 + alpha)
    # ------------------------------------------------------------------ #
    fixed_dargs_f64 = dynamic_args._replace(
        wind_factor=jnp.array([WF], dtype=jnp.float64),
        kp=jnp.array([KP_VAL], dtype=jnp.float64),
    )

    float_fields = [f for f, v in snap_step._asdict().items()
                    if jnp.issubdtype(jnp.asarray(v).dtype, jnp.floating)]

    print(f"One-step state Jacobian at hour {STEP_HOUR}->{END_HOUR} (float64):", flush=True)
    print(f"  = d(||lice_{END_HOUR}||^2) / d(alpha_f) where snap[f] *= (1+alpha)\n", flush=True)
    print(f"{'field':>22} {'jacobian':>18} {'|ratio to wf_grad|':>20}", flush=True)

    results = []
    for field in float_fields:
        field_vals = getattr(snap_step, field)
        field_vals_f64 = field_vals.astype(jnp.float64)

        def field_jac(alpha, field=field, field_vals_f64=field_vals_f64):
            perturbed = snap_step._replace(**{field: field_vals_f64 * (1.0 + alpha)})
            perturbed_f64 = jax.tree_util.tree_map(
                lambda x: x.astype(jnp.float64) if jnp.issubdtype(x.dtype, jnp.floating) else x,
                perturbed
            )
            final, _ = pebsi_main(perturbed_f64, one_step_forcings, model.point_attrs, static_args, fixed_dargs_f64)
            return jnp.sum(jnp.square(final.lice.astype(jnp.float64)))

        t0 = time.time()
        try:
            g = float(jax.grad(field_jac)(jnp.float64(0.0)))
        except Exception as e:
            g = float('nan')
        elapsed = time.time() - t0

        ratio = abs(g / g_wf) if g_wf != 0 else float('inf')
        print(f"{field:>22} {g:>18.4e} {ratio:>20.4e}  ({elapsed:.1f}s)", flush=True)
        results.append((field, g, ratio))

    print("\nRanked by |jacobian| (largest first):", flush=True)
    for field, g, ratio in sorted(results, key=lambda r: -abs(r[1])):
        print(f"  {field:>22}: {g:.4e}  (ratio {ratio:.2e})", flush=True)
