"""
Probes the REAL single-site MAE loss (the one investigate_site_blowup /
run_optimization differentiate) around the step-1 blowup point found on the
UNSMOOTHED code: wf=kp=0.9512, where AD gives dL/dlog_wf = -1.5e16 while at
wf=kp=1.0 it gives +13.1.

Three measurements:
  1. dense forward-only L(wf) over [PEBSI_MAE_LO, PEBSI_MAE_HI], kp fixed
  2. AD gradient at a handful of wf values including exactly 0.9512
  3. central FD slopes at wf=0.9512 at several relative steps

If the AD 1e16 is the true local slope, FD at the smallest steps should grow
toward it; if the landscape is a cliff (loss jump), the dense grid shows it;
if the landscape is smooth O(1..100) slope everywhere, the AD number is an
adjoint artifact of the recurrence, not a property of the landscape.

Writes mae_landscape.csv. Run: conda activate gpu-env && python probe_mae_landscape.py
"""
import os
os.environ.setdefault('PEBSI_INVESTIGATE_SITE_BLOWUP', '1')

import time
import csv
import numpy as np
import jax
import jax.numpy as jnp

import jax_optimize as jo

jax.config.update('jax_debug_nans', False)
jax.config.update('jax_debug_infs', False)

SITE_INDEX = int(os.environ.get('PEBSI_MAE_SITE_INDEX', '5'))
WF_CENTER = float(os.environ.get('PEBSI_MAE_WF_CENTER', '0.951229'))  # exp(-0.05)
KP_VAL = float(os.environ.get('PEBSI_MAE_KP', '0.951229'))
WF_LO = float(os.environ.get('PEBSI_MAE_LO', '0.93'))
WF_HI = float(os.environ.get('PEBSI_MAE_HI', '1.01'))
N_DENSE = int(os.environ.get('PEBSI_MAE_N_DENSE', '33'))

if os.environ.get('PEBSI_MAE_QUICK', '0') == '1':
    # 3 AD points (incl. the blowup point and the previously-inf point) and
    # one FD scale, no dense scan -- for fast A/B tests of forward-model edits
    GRAD_WF = [WF_CENTER, 0.99, 1.0]
    FD_REL_STEPS = [1e-3]
    N_DENSE = 0
else:
    GRAD_WF = [1.0, 0.99, 0.98, 0.97, 0.96, WF_CENTER, 0.945, 0.94]
    FD_REL_STEPS = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]

if __name__ == '__main__':
    site_dict = jo.load_reduced_site_dict(jo.REDUCED_SITES_CONFIG)
    site_order = jo.flatten_site_order(site_dict)
    glacier, site = site_order[SITE_INDEX]
    single_site_dict = {glacier: [site]}
    print(f"probing site {SITE_INDEX} -> {glacier}/{site}, kp={KP_VAL}", flush=True)

    config_fp = jo.build_generated_config(
        single_site_dict, jo.host,
        start_date=jo.DEBUG_START_DATE, end_date=jo.DEBUG_END_DATE,
        temporal_chunk_years=1,
    )
    model = jo.init_pebsi(config_fp)

    obs = jo.load_all_observations(single_site_dict)
    s_labels, s_meas, s_mask, s_starts, s_ends = obs['summer']
    w_labels, w_meas, w_mask, w_starts, w_ends = obs['winter']
    s_idx = jo.build_period_indices(model.dates, s_starts, s_ends)
    w_idx = jo.build_period_indices(model.dates, w_starts, w_ends)

    loss_fn = jo.make_loss_fn(
        model, [(glacier, site)],
        summer=(s_labels, s_idx, s_meas, s_mask),
        winter=(w_labels, w_idx, w_meas, w_mask),
    )

    log_kp = jnp.array([np.log(KP_VAL)], dtype=jnp.float32)

    def total_loss(log_wf_scalar):
        total, _aux = loss_fn(jnp.array([log_wf_scalar], dtype=jnp.float32), log_kp)
        return total

    loss_jit = jax.jit(total_loss)
    grad_jit = jax.jit(jax.value_and_grad(total_loss))

    rows = []

    # 2. AD gradients (do first: includes the exact blowup point sanity check)
    for i, wf in enumerate(GRAD_WF):
        t0 = time.time()
        try:
            L, g = grad_jit(jnp.log(jnp.float32(wf)))
            L, g = float(L), float(g)
        except FloatingPointError:
            L, g = float('nan'), float('nan')
        rows.append(['grad', wf, L, g])
        print(f"grad {i+1}/{len(GRAD_WF)} wf={wf:.6f} L={L:.8e} "
              f"dL/dlogwf={g:.6e} ({time.time()-t0:.1f}s)", flush=True)

    # 3. FD at the blowup point, multiple scales (central, in log_wf space to
    # match the AD gradient's units)
    log_wc = np.log(WF_CENTER)
    L_center = float(loss_jit(jnp.float32(log_wc)))
    print(f"center wf={WF_CENTER:.6f} L={L_center:.8e}", flush=True)
    for i, h in enumerate(FD_REL_STEPS):
        t0 = time.time()
        Lp = float(loss_jit(jnp.float32(log_wc + h)))
        Lm = float(loss_jit(jnp.float32(log_wc - h)))
        slope = (Lp - Lm) / (2 * h)
        rows.append(['fd', WF_CENTER, L_center, slope, h, Lp, Lm])
        print(f"fd {i+1}/{len(FD_REL_STEPS)} h={h:.1e} L+={Lp:.8e} L-={Lm:.8e} "
              f"slope={slope:.6e} ({time.time()-t0:.1f}s)", flush=True)

    # 1. dense landscape
    dense_wf = np.linspace(WF_LO, WF_HI, N_DENSE) if N_DENSE else np.array([])
    for i, wf in enumerate(dense_wf):
        t0 = time.time()
        L = float(loss_jit(jnp.float32(np.log(wf))))
        rows.append(['dense', wf, L, ''])
        print(f"dense {i+1}/{N_DENSE} wf={wf:.6f} L={L:.8e} ({time.time()-t0:.1f}s)", flush=True)

    with open('mae_landscape.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['kind', 'wf', 'L', 'dLdlogwf_or_fd_slope', 'fd_h', 'L_plus', 'L_minus'])
        w.writerows(rows)
    print("wrote mae_landscape.csv", flush=True)
