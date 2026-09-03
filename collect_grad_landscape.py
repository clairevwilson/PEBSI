"""
Collects AD gradient data at many closely-spaced wf values near wf=1.0.
Writes results to CSV for separate plotting (plot_grad_landscape.py).

Loss: total ice mass in m w.e. (physical units, not squared).
Gradient: d(total_ice_mwe)/d(wf), float64 to avoid overflow artifacts.

Runtime: roughly (N_COARSE + N_FINE) * END_HOUR/45 * 7s + compilation.
Default settings (~90 points, 2000h window): ~8 hours. Reduce END_HOUR
or N_* env vars for a quick test; increase for production.

Env vars:
  PEBSI_GRAD_END_HOUR   simulation hours to run (default 2000)
  PEBSI_GRAD_OUTPUT     output CSV path (default grad_landscape.csv)
  PEBSI_GRAD_KP         kp value to hold fixed (default 1.0)
  PEBSI_COARSE_LO/HI   wf range for coarse sweep (default 0.98 / 1.02)
  PEBSI_N_COARSE        points in coarse sweep (default 30)
  PEBSI_FINE_LO/HI      wf range for fine sweep (default 0.998 / 1.002)
  PEBSI_N_FINE          points in fine sweep (default 60)
"""
import os
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
os.environ.setdefault('PEBSI_NORMAL_RUN_2015_2020', '1')
import jax
jax.config.update('jax_debug_nans', False)
jax.config.update('jax_enable_x64', True)

import csv
import time
import numpy as np
import jax.numpy as jnp

import jax_optimize as jo
from pebsi.main import main as pebsi_main
from decompose_gradient_by_field import build_single_site_model, SITE_INDEX

END_HOUR = int(os.environ.get('PEBSI_GRAD_END_HOUR', '2000'))
OUTPUT_CSV = os.environ.get('PEBSI_GRAD_OUTPUT', 'grad_landscape.csv')
KP_VAL = float(os.environ.get('PEBSI_GRAD_KP', '1.0'))

WF_COARSE = np.linspace(
    float(os.environ.get('PEBSI_COARSE_LO', '0.98')),
    float(os.environ.get('PEBSI_COARSE_HI', '1.02')),
    int(os.environ.get('PEBSI_N_COARSE', '30')),
)
WF_FINE = np.linspace(
    float(os.environ.get('PEBSI_FINE_LO', '0.998')),
    float(os.environ.get('PEBSI_FINE_HI', '1.002')),
    int(os.environ.get('PEBSI_N_FINE', '60')),
)

DENSITY_WATER = 1000.0

if __name__ == '__main__':
    model = build_single_site_model(SITE_INDEX)
    params = model.config.params
    static_args = model.config.static_args
    dynamic_args = model.config.dynamic_args

    forcings = model.pack_forcings(params, model.dates[:END_HOUR], 0)

    init_f64 = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64) if jnp.issubdtype(x.dtype, jnp.floating) else x,
        model.initial_state,
    )

    def total_ice_mwe(wf_val):
        dargs = dynamic_args._replace(
            wind_factor=jnp.array([wf_val], dtype=jnp.float64),
            kp=jnp.array([KP_VAL], dtype=jnp.float64),
        )
        final, _ = pebsi_main(init_f64, forcings, model.point_attrs, static_args, dargs)
        return jnp.sum(final.lice.astype(jnp.float64)) / DENSITY_WATER

    val_and_grad_jit = jax.jit(jax.value_and_grad(total_ice_mwe))

    coarse_set = set(WF_COARSE.round(10).tolist())
    fine_set = set(WF_FINE.round(10).tolist())
    all_wf = np.unique(np.concatenate([WF_COARSE, WF_FINE]))

    total = len(all_wf)
    site_label = f"{model.glacier}/{model.site}" if hasattr(model, 'glacier') else f"index {SITE_INDEX}"
    print(f"site: {site_label}  kp={KP_VAL}  END_HOUR={END_HOUR}", flush=True)
    print(f"coarse: {len(WF_COARSE)} pts over [{WF_COARSE[0]:.4f}, {WF_COARSE[-1]:.4f}]", flush=True)
    if len(WF_FINE):
        print(f"fine:   {len(WF_FINE)} pts over [{WF_FINE[0]:.5f}, {WF_FINE[-1]:.5f}]", flush=True)
    print(f"total unique wf values: {total}", flush=True)
    print(f"output: {OUTPUT_CSV}\n", flush=True)

    rows = []
    for i, wf in enumerate(all_wf):
        t0 = time.time()
        try:
            v, g = val_and_grad_jit(jnp.float64(wf))
            loss = float(v)
            grad = float(g)
        except Exception as e:
            loss = float('nan')
            grad = float('nan')
        elapsed = time.time() - t0

        tags = []
        if round(wf, 10) in coarse_set:
            tags.append('coarse')
        if round(wf, 10) in fine_set:
            tags.append('fine')
        sweep = '+'.join(tags)

        rows.append({
            'wf': wf,
            'loss_mwe': loss,
            'grad_mwe': grad,
            'sweep': sweep,
        })
        print(f"  {i+1}/{total}  wf={wf:.6f}  loss={loss:.4f}  grad={grad:+.4e}  "
              f"[{sweep}]  ({elapsed:.0f}s)", flush=True)

    with open(OUTPUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['wf', 'loss_mwe', 'grad_mwe', 'sweep'])
        w.writeheader()
        w.writerows(rows)

    n_finite = sum(np.isfinite(r['grad_mwe']) for r in rows)
    n_sign_flips = sum(
        1 for a, b in zip(rows, rows[1:])
        if np.isfinite(a['grad_mwe']) and np.isfinite(b['grad_mwe'])
        and np.sign(a['grad_mwe']) != np.sign(b['grad_mwe'])
    )
    print(f"\nDone. {n_finite}/{total} finite gradients, {n_sign_flips} sign flips.", flush=True)
    print(f"Wrote {OUTPUT_CSV}", flush=True)
