"""
Binary-search over K53's 5-year time window to find which sub-window
causes the gradient explosion. Matches main optimizer settings exactly:
constant_snowfall_density=150, no x64, debug_nans off.
"""
import os
os.environ['PEBSI_NORMAL_RUN_2015_2020'] = '1'
os.environ['PEBSI_DEBUG_NANS'] = '0'  # must be set before jax_optimize import

import jax
jax.config.update('jax_debug_nans', False)
jax.config.update('jax_debug_infs', False)

import yaml
import time
import numpy as np
import jax.numpy as jnp

import jax_optimize as jo

# Ensure debug_nans stays off even after jax_optimize import
jax.config.update('jax_debug_nans', False)
jax.config.update('jax_debug_infs', False)

MELT_SMOOTH_ALPHA = int(os.environ.get('PEBSI_MELT_SMOOTH_ALPHA', '100'))
WF_A = float(os.environ.get('PROBE_WF_A', '1.0'))
WF_B = float(os.environ.get('PROBE_WF_B', '0.99'))


def build_k53_model(end_date):
    single_site_dict = {'kahiltna': ['K53']}
    config_fp = jo.build_generated_config(
        single_site_dict, jo.host,
        start_date=jo.DEBUG_START_DATE,
        end_date=end_date,
        temporal_chunk_years=1,
    )
    # patch melt_smooth_alpha only — leave constant_snowfall_density as-is
    # (build_generated_config already sets it to 150, matching main optimizer)
    if MELT_SMOOTH_ALPHA > 0:
        with open(config_fp) as f:
            cfg = yaml.safe_load(f)
        cfg['melt_smooth_alpha'] = MELT_SMOOTH_ALPHA
        with open(config_fp, 'w') as f:
            yaml.dump(cfg, f, sort_keys=False)
    return jo.init_pebsi(config_fp)


def grad_at_wf(model, wf):
    """Returns d(MAE_loss)/d(log_wf) for K53 at given wf. Returns nan on failure."""
    obs = jo.load_all_observations({'kahiltna': ['K53']})
    summer_labels, summer_meas, summer_mask, summer_starts, summer_ends = obs['summer']
    winter_labels, winter_meas, winter_mask, winter_starts, winter_ends = obs['winter']
    summer_idx = jo.build_period_indices(model.dates, summer_starts, summer_ends)
    winter_idx = jo.build_period_indices(model.dates, winter_starts, winter_ends)
    loss_fn = jo.make_loss_fn(
        model, [('kahiltna', 'K53')],
        summer=(summer_labels, summer_idx, summer_meas, summer_mask),
        winter=(winter_labels, winter_idx, winter_meas, winter_mask),
    )
    log_wf = jnp.array([np.log(wf)], dtype=jnp.float32)
    log_kp = jnp.zeros(1, dtype=jnp.float32)
    def scalar_loss(lw):
        return loss_fn(lw, log_kp)[0]
    grad_fn = jax.jit(jax.grad(scalar_loss))
    try:
        g = grad_fn(log_wf)
        jax.block_until_ready(g)
        return float(g[0])
    except Exception as e:
        print(f"    [exception: {type(e).__name__}: {e}]", flush=True)
        return float('nan')


def probe_window(label, end_date, prev_a, prev_b):
    t0 = time.time()
    model = build_k53_model(end_date)
    g_a = grad_at_wf(model, WF_A)
    g_b = grad_at_wf(model, WF_B)
    delta = g_b - g_a if (np.isfinite(g_a) and np.isfinite(g_b)) else float('nan')
    elapsed = time.time() - t0
    flag = ''
    if prev_a is not None and np.isfinite(g_a) and np.isfinite(prev_a):
        added_a = g_a - prev_a
        added_b = g_b - prev_b if np.isfinite(g_b) and np.isfinite(prev_b) else float('nan')
        if abs(added_a) > 1e4 or (np.isfinite(added_b) and abs(added_b) > 1e4):
            flag = '  <-- LARGE JUMP'
        elif np.isfinite(added_b) and np.sign(added_a) != np.sign(added_b) and abs(added_a) > 1:
            flag = '  <-- SIGN FLIP'
    print(f"{label:>14}  {g_a:>+14.4e}  {g_b:>+14.4e}  {delta:>+14.4e}  {elapsed:.1f}s{flag}", flush=True)
    return g_a, g_b


if __name__ == '__main__':
    print(f"K53 gradient probe: alpha={MELT_SMOOTH_ALPHA}  WF_A={WF_A}  WF_B={WF_B}")
    print(f"Settings: constant_snowfall_density=150 (main-optimizer match), debug_nans=off")
    print(f"{'window':>14}  {'grad@WF_A':>14}  {'grad@WF_B':>14}  {'delta':>14}  time")
    print('-' * 80)

    # Year-level pass: DEBUG_START_DATE=2015-04-01, step by full years
    year_ends = [
        ('2016-04-01', '2016-04-01'),
        ('2017-04-01', '2017-04-01'),
        ('2018-04-01', '2018-04-01'),
        ('2019-04-01', '2019-04-01'),
        ('2020-04-01', '2020-04-01'),
    ]
    prev_a = prev_b = None
    kink_year_end = None
    for label, end in year_ends:
        g_a, g_b = probe_window(label, end, prev_a, prev_b)
        if prev_a is not None and np.isfinite(g_a) and np.isfinite(prev_a):
            added_a = g_a - prev_a
            added_b = (g_b - prev_b) if np.isfinite(g_b) and np.isfinite(prev_b) else float('nan')
            if abs(added_a) > 1e4 or (np.isfinite(added_b) and abs(added_b) > 1e4):
                kink_year_end = end
        if not np.isfinite(g_a) or not np.isfinite(g_b):
            kink_year_end = end
            break
        prev_a, prev_b = g_a, g_b

    if kink_year_end is None:
        print("\nNo large jump found in annual scan. Kink is too small to detect at this resolution.")
    else:
        print(f"\nKink year ends at {kink_year_end}. Scanning that year month by month...")
        # figure out the start of the kink year
        kink_year_idx = [e for _, e in year_ends].index(kink_year_end)
        year_start = year_ends[kink_year_idx - 1][1] if kink_year_idx > 0 else jo.DEBUG_START_DATE

        # month-level: 12 months within the kink year
        import datetime
        start_dt = datetime.datetime.strptime(year_start, '%Y-%m-%d')
        month_ends = []
        for m in range(1, 13):
            dt = start_dt + datetime.timedelta(days=30 * m)
            month_ends.append(dt.strftime('%Y-%m-%d'))

        print(f"{'month_end':>14}  {'grad@WF_A':>14}  {'grad@WF_B':>14}  {'delta':>14}  time")
        print('-' * 80)
        prev_a = prev_b = None
        for label in month_ends:
            # skip if end would give < 1 full chunk
            try:
                g_a, g_b = probe_window(label, label, prev_a, prev_b)
            except Exception as e:
                print(f"{label:>14}  [skipped: {e}]", flush=True)
                continue
            prev_a, prev_b = g_a, g_b

    print("\nDone.")
