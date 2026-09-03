"""
Diagnose what triggers the Feb 13, 2019 add_top_layer event for WF=0.99 but not WF=1.0.
Runs the model from 2018-12-01 to 2019-02-15 in daily chunks, logging top-layer conditions.
"""
import os
os.environ['PEBSI_NORMAL_RUN_2015_2020'] = '1'
os.environ['PEBSI_DEBUG_NANS'] = '0'

import sys
LOG = '/tmp/probe_k53_feb13.log'
log_file = open(LOG, 'w', buffering=1)
def pr(*args, **kwargs):
    print(*args, **kwargs, file=log_file, flush=True)
    print(*args, **kwargs, flush=True)

pr("starting imports...")
import jax
jax.config.update('jax_debug_nans', False)
jax.config.update('jax_debug_infs', False)
import yaml
import time
import numpy as np
import jax.numpy as jnp
import jax_optimize as jo
pr("imports done")

MELT_SMOOTH_ALPHA = 100
WF_A = 1.0
WF_B = 0.99
START_DATE = '2018-12-01'
END_DATE = '2019-02-15'
CHUNK_DAYS = 1


def build_model():
    chunk_years = CHUNK_DAYS / 365.0
    config_fp = jo.build_generated_config(
        {'kahiltna': ['K53']}, jo.host,
        start_date=jo.DEBUG_START_DATE,
        end_date=END_DATE,
        temporal_chunk_years=chunk_years,
    )
    with open(config_fp) as f:
        cfg = yaml.safe_load(f)
    cfg['melt_smooth_alpha'] = MELT_SMOOTH_ALPHA
    with open(config_fp, 'w') as f:
        yaml.dump(cfg, f, sort_keys=False)
    return jo.init_pebsi(config_fp)


def run_forward_all(model, wf):
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


def extract_top_layer_info(state, params, new_density=150.0):
    """Extract the conditions that determine new_layer_cond for the top layer."""
    ltype = int(np.asarray(state.ltype)[0, 0])
    ldensity = float(np.asarray(state.ldensity)[0, 0])
    lheight = float(np.asarray(state.lheight)[0, 0])
    lice = float(np.asarray(state.lice)[0, 0])
    lice_sum = float(np.asarray(state.lice)[0].sum())

    surf_not_snow = (ltype > 0)
    density_threshold = (ldensity > new_density * 3)
    large_top_layer = (lheight > params.dz_toplayer * 2)
    new_layer_cond = surf_not_snow or density_threshold or large_top_layer

    return {
        'ltype': ltype,
        'ldensity': ldensity,
        'lheight': lheight,
        'lice': lice,
        'lice_sum': lice_sum,
        'surf_not_snow': surf_not_snow,
        'density_threshold': density_threshold,
        'large_top_layer': large_top_layer,
        'new_layer_cond': new_layer_cond,
        'dz_toplayer': params.dz_toplayer,
        'density_thresh_val': new_density * 3,
    }


if __name__ == '__main__':
    t0 = time.time()
    pr("Building model A (WF=1.0)...")
    model_a = build_model()
    pr(f"  Spinup running...")
    snaps_a = run_forward_all(model_a, WF_A)
    pr(f"  WF={WF_A}: {len(snaps_a)} snapshots ({time.time()-t0:.1f}s)")

    t0 = time.time()
    pr("Building model B (WF=0.99)...")
    model_b = build_model()
    snaps_b = run_forward_all(model_b, WF_B)
    pr(f"  WF={WF_B}: {len(snaps_b)} snapshots ({time.time()-t0:.1f}s)")

    params = model_a.config.params
    pr(f"\ndz_toplayer = {params.dz_toplayer}")
    pr(f"large_top_layer threshold = {params.dz_toplayer * 2:.4f} m")
    pr(f"density_threshold value = {150.0 * 3} kg/m³")
    pr()

    hdr = f"{'date':>22}  {'type_A':>6}  {'dens_A':>8}  {'ht_A':>7}  {'sns_A':>5}  {'dt_A':>5}  {'lt_A':>5}  {'nlc_A':>5}  ||  {'type_B':>6}  {'dens_B':>8}  {'ht_B':>7}  {'sns_B':>5}  {'dt_B':>5}  {'lt_B':>5}  {'nlc_B':>5}  {'liceB-A':>10}"
    pr(hdr)
    pr('-' * len(hdr))

    prev_nlc_a = False
    prev_nlc_b = False

    for (date_a, sa), (date_b, sb) in zip(snaps_a, snaps_b):
        if date_a < np.datetime64('2018-12-28') or date_a > np.datetime64('2019-02-14'):
            continue
        ia = extract_top_layer_info(sa, params)
        ib = extract_top_layer_info(sb, params)
        diff = ib['lice_sum'] - ia['lice_sum']

        def yn(v): return 'T' if v else 'F'

        pr(
            f"{str(date_a):>22}  "
            f"{ia['ltype']:>6}  {ia['ldensity']:>8.1f}  {ia['lheight']:>7.4f}  "
            f"{yn(ia['surf_not_snow']):>5}  {yn(ia['density_threshold']):>5}  {yn(ia['large_top_layer']):>5}  {yn(ia['new_layer_cond']):>5}  ||  "
            f"{ib['ltype']:>6}  {ib['ldensity']:>8.1f}  {ib['lheight']:>7.4f}  "
            f"{yn(ib['surf_not_snow']):>5}  {yn(ib['density_threshold']):>5}  {yn(ib['large_top_layer']):>5}  {yn(ib['new_layer_cond']):>5}  "
            f"{diff:>+10.2f}"
        )
        prev_nlc_a = ia['new_layer_cond']
        prev_nlc_b = ib['new_layer_cond']

    log_file.close()
