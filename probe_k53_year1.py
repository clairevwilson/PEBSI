"""
Year-1 gradient probe for K53. Writes directly to a log file so output
is visible even when conda buffers stdout.
"""
import os, sys
os.environ['PEBSI_NORMAL_RUN_2015_2020'] = '1'
os.environ['PEBSI_DEBUG_NANS'] = '0'

LOG = '/tmp/probe_k53_year1.log'
_lf = open(LOG, 'w', buffering=1)  # line-buffered

def log(msg):
    _lf.write(msg + '\n')
    _lf.flush()
    print(msg, flush=True)

log("starting imports...")

import jax
jax.config.update('jax_debug_nans', False)
jax.config.update('jax_debug_infs', False)

import yaml, time, numpy as np, jax.numpy as jnp
log("jax imported")

import jax_optimize as jo
jax.config.update('jax_debug_nans', False)
jax.config.update('jax_debug_infs', False)
log("jax_optimize imported")

MELT_SMOOTH_ALPHA = int(os.environ.get('PEBSI_MELT_SMOOTH_ALPHA', '100'))
WF_A = float(os.environ.get('PROBE_WF_A', '1.0'))
WF_B = float(os.environ.get('PROBE_WF_B', '0.99'))

log(f"alpha={MELT_SMOOTH_ALPHA}  WF_A={WF_A}  WF_B={WF_B}")

def build_model(end_date):
    config_fp = jo.build_generated_config(
        {'kahiltna': ['K53']}, jo.host,
        start_date=jo.DEBUG_START_DATE,
        end_date=end_date,
        temporal_chunk_years=1,
    )
    if MELT_SMOOTH_ALPHA > 0:
        with open(config_fp) as f:
            cfg = yaml.safe_load(f)
        cfg['melt_smooth_alpha'] = MELT_SMOOTH_ALPHA
        with open(config_fp, 'w') as f:
            yaml.dump(cfg, f, sort_keys=False)
    return jo.init_pebsi(config_fp)

def grad_at_wf(model, wf, label):
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
    log(f"  [{label}] compiling grad_fn at wf={wf}...")
    t0 = time.time()
    grad_fn = jax.jit(jax.grad(scalar_loss))
    g = grad_fn(log_wf)
    jax.block_until_ready(g)
    elapsed = time.time() - t0
    val = float(g[0])
    log(f"  [{label}] grad={val:+.4e}  ({elapsed:.1f}s)")
    return val

log("building year-1 model (2016-04-01)...")
t0 = time.time()
model = build_model('2016-04-01')
log(f"model built in {time.time()-t0:.1f}s")

g_a = grad_at_wf(model, WF_A, 'WF_A')
g_b = grad_at_wf(model, WF_B, 'WF_B')

sign_a = '+' if g_a > 0 else '-'
sign_b = '+' if g_b > 0 else '-'
same_sign = (g_a * g_b > 0)
status = "OK (same sign)" if same_sign else "SIGN FLIP"
log(f"\nYear-1 result: grad@WF_A={g_a:+.4e}  grad@WF_B={g_b:+.4e}  -> {status}")
_lf.close()
