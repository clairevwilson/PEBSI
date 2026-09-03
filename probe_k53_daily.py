"""
Daily-resolution forward comparison of K53 for just summer 2015,
to find the first hour where WF=1.0 and WF=0.99 diverge significantly.
Uses hourly chunks (temporal_chunk_years=1/8760).
"""
import os
os.environ['PEBSI_NORMAL_RUN_2015_2020'] = '1'
os.environ['PEBSI_DEBUG_NANS'] = '0'

import jax
jax.config.update('jax_debug_nans', False)
jax.config.update('jax_debug_infs', False)

import yaml
import time
import numpy as np
import jax.numpy as jnp

import jax_optimize as jo

jax.config.update('jax_debug_nans', False)
jax.config.update('jax_debug_infs', False)

MELT_SMOOTH_ALPHA = int(os.environ.get('PEBSI_MELT_SMOOTH_ALPHA', '100'))
WF_A = float(os.environ.get('PROBE_WF_A', '1.0'))
WF_B = float(os.environ.get('PROBE_WF_B', '0.99'))
# Only simulate through end of 2015 to keep this fast
END_DATE = os.environ.get('PROBE_END_DATE', '2015-12-31')
CHUNK_DAYS = int(os.environ.get('PROBE_CHUNK_DAYS', '1'))


def build_k53_model():
    chunk_years = CHUNK_DAYS / 365.0
    config_fp = jo.build_generated_config(
        {'kahiltna': ['K53']}, jo.host,
        start_date=jo.DEBUG_START_DATE,
        end_date=END_DATE,
        temporal_chunk_years=chunk_years,
    )
    if MELT_SMOOTH_ALPHA > 0:
        with open(config_fp) as f:
            cfg = yaml.safe_load(f)
        cfg['melt_smooth_alpha'] = MELT_SMOOTH_ALPHA
        with open(config_fp, 'w') as f:
            yaml.dump(cfg, f, sort_keys=False)
    return jo.init_pebsi(config_fp)


def run_forward_all_snapshots(model, wf):
    """Run forward, saving state every chunk (daily)."""
    model.config.dynamic_args = model.config.dynamic_args._replace(
        wind_factor=jnp.array([wf], dtype=jnp.float32),
        kp=jnp.ones(1, dtype=jnp.float32),
    )
    params = model.config.params
    static_args = model.config.static_args
    dynamic_args = model.config.dynamic_args
    chunk_size = params.temporal_chunk_hours
    total_steps = (len(model.dates) // chunk_size) * chunk_size
    n_chunks = total_steps // chunk_size

    from pebsi.main import main as pebsi_main
    state = model.initial_state
    snapshots = []
    for c in range(n_chunks):
        start = c * chunk_size
        chunk_forcings = model.pack_forcings(params, model.dates[start:start + chunk_size], start)
        state, _ = pebsi_main(state, chunk_forcings, model.point_attrs, static_args, dynamic_args)
        snapshots.append((model.dates[start + chunk_size - 1], state))
    return snapshots


def total_ice(state):
    lice = np.asarray(state.lice)[0]
    return float(lice.sum())

def n_active(state):
    lice = np.asarray(state.lice)[0]
    return int((lice > 0.001).sum())


if __name__ == '__main__':
    print(f"K53 daily probe: WF_A={WF_A} vs WF_B={WF_B}, alpha={MELT_SMOOTH_ALPHA}")
    print(f"End={END_DATE}, chunk={CHUNK_DAYS}d")

    t0 = time.time()
    model_a = build_k53_model()
    snaps_a = run_forward_all_snapshots(model_a, WF_A)
    print(f"  WF={WF_A}: {len(snaps_a)} snapshots ({time.time()-t0:.1f}s)", flush=True)

    t0 = time.time()
    model_b = build_k53_model()
    snaps_b = run_forward_all_snapshots(model_b, WF_B)
    print(f"  WF={WF_B}: {len(snaps_b)} snapshots ({time.time()-t0:.1f}s)", flush=True)

    print(f"\nShowing snapshots where |ice_diff| > 10 kg/m² or n_active differs:")
    print(f"{'date':>22}  {'nA':>3}  {'nB':>3}  {'ice_A':>12}  {'ice_B':>12}  {'diff':>12}")
    print('-' * 72)

    prev_diff = 0.0
    for (date_a, state_a), (date_b, state_b) in zip(snaps_a, snaps_b):
        ice_a = total_ice(state_a)
        ice_b = total_ice(state_b)
        na = n_active(state_a)
        nb = n_active(state_b)
        diff = ice_b - ice_a
        if abs(diff) > 10 or na != nb or abs(diff - prev_diff) > 5:
            print(f"{str(date_a):>22}  {na:>3}  {nb:>3}  {ice_a:>12.2f}  {ice_b:>12.2f}  {diff:>+12.2f}")
        prev_diff = diff
