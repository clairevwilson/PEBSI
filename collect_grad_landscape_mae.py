"""
Like collect_grad_landscape.py but uses the real MAE loss (summer + winter
mass balance MAE against observations) instead of total ice mass.

Writes grad_landscape_mae.csv with columns: wf, loss_mae, grad_mae, sweep.

Env vars:
  PEBSI_COARSE_LO/HI   wf range for coarse sweep (default 0.98 / 1.02)
  PEBSI_N_COARSE        points in coarse sweep (default 30)
  PEBSI_GRAD_OUTPUT     output CSV path (default grad_landscape_mae.csv)
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
from decompose_gradient_by_field import build_single_site_model, SITE_INDEX, KP_VAL

OUTPUT_CSV = os.environ.get('PEBSI_GRAD_OUTPUT', 'grad_landscape_mae.csv')

WF_COARSE = np.linspace(
    float(os.environ.get('PEBSI_COARSE_LO', '0.98')),
    float(os.environ.get('PEBSI_COARSE_HI', '1.02')),
    int(os.environ.get('PEBSI_N_COARSE', '30')),
)
WF_FINE = np.linspace(
    float(os.environ.get('PEBSI_FINE_LO', '0.998')),
    float(os.environ.get('PEBSI_FINE_HI', '1.002')),
    int(os.environ.get('PEBSI_N_FINE', '0')),
)

if __name__ == '__main__':
    model = build_single_site_model(SITE_INDEX)

    site_dict = jo.load_reduced_site_dict(jo.REDUCED_SITES_CONFIG)
    site_order = jo.flatten_site_order(site_dict)
    glacier, site = site_order[SITE_INDEX]
    single_site_dict = {glacier: [site]}

    obs_by_season = jo.load_all_observations(single_site_dict)
    summer_labels, summer_meas, summer_mask, summer_starts, summer_ends = obs_by_season['summer']
    winter_labels, winter_meas, winter_mask, winter_starts, winter_ends = obs_by_season['winter']
    summer_period_idx = jo.build_period_indices(model.dates, summer_starts, summer_ends)
    winter_period_idx = jo.build_period_indices(model.dates, winter_starts, winter_ends)

    loss_fn = jo.make_loss_fn(
        model, [(glacier, site)],
        summer=(summer_labels, summer_period_idx, summer_meas, summer_mask),
        winter=(winter_labels, winter_period_idx, winter_meas, winter_mask),
    )

    def mae_vs_wf(wf_val):
        log_wf = jnp.log(jnp.array([wf_val], dtype=jnp.float64))
        log_kp = jnp.log(jnp.array([KP_VAL], dtype=jnp.float64))
        total_loss, _ = loss_fn(log_wf, log_kp)
        return total_loss.astype(jnp.float64)

    val_and_grad_jit = jax.jit(jax.value_and_grad(mae_vs_wf))

    coarse_set = set(WF_COARSE.round(10).tolist())
    fine_set = set(WF_FINE.round(10).tolist())
    all_wf = np.unique(np.concatenate([WF_COARSE, WF_FINE]))
    total = len(all_wf)

    site_label = f"{glacier}/{site}"
    print(f"site: {site_label}  kp={KP_VAL}", flush=True)
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

        rows.append({'wf': wf, 'loss_mae': loss, 'grad_mae': grad, 'sweep': sweep})
        print(f"  {i+1}/{total}  wf={wf:.6f}  loss={loss:.4f}  grad={grad:+.4e}  "
              f"[{sweep}]  ({elapsed:.0f}s)", flush=True)

    with open(OUTPUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['wf', 'loss_mae', 'grad_mae', 'sweep'])
        w.writeheader()
        w.writerows(rows)

    n_finite = sum(np.isfinite(r['grad_mae']) for r in rows)
    n_sign_flips = sum(
        1 for a, b in zip(rows, rows[1:])
        if np.isfinite(a['grad_mae']) and np.isfinite(b['grad_mae'])
        and np.sign(a['grad_mae']) != np.sign(b['grad_mae'])
    )
    print(f"\nDone. {n_finite}/{total} finite gradients, {n_sign_flips} sign flips.", flush=True)
    print(f"Wrote {OUTPUT_CSV}", flush=True)
