"""
Forward-only comparison of K53 layer states at WF=1.0 vs WF=0.99.
Runs the full 5-year forward pass at both WF values and snapshots
layer structure at each melt-season month, looking for differences.
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


def build_k53_model():
    config_fp = jo.build_generated_config(
        {'kahiltna': ['K53']}, jo.host,
        start_date=jo.DEBUG_START_DATE,
        end_date=jo.DEBUG_END_DATE,
        temporal_chunk_years=1/12,  # monthly chunks → monthly snapshot resolution
    )
    if MELT_SMOOTH_ALPHA > 0:
        with open(config_fp) as f:
            cfg = yaml.safe_load(f)
        cfg['melt_smooth_alpha'] = MELT_SMOOTH_ALPHA
        with open(config_fp, 'w') as f:
            yaml.dump(cfg, f, sort_keys=False)
    return jo.init_pebsi(config_fp)


def run_forward_snapshots(model, wf):
    """Run full forward pass at given WF, return monthly snapshots."""
    model.config.dynamic_args = model.config.dynamic_args._replace(
        wind_factor=jnp.array([wf], dtype=jnp.float32),
        kp=jnp.ones(1, dtype=jnp.float32),
    )
    snapshots = jo.generate_snapshots(model, n_snapshots=60)  # ~monthly for 5 years
    return snapshots


def summarize_layers(state):
    """Return (n_active_layers, lice_per_layer, lheight_per_layer) for site 0."""
    lice = np.asarray(state.lice)[0]       # (N_LAYERS,)
    lheight = np.asarray(state.lheight)[0]
    ldensity = np.asarray(state.ldensity)[0]
    ltype = np.asarray(state.ltype)[0]
    min_mass = 0.001  # min_layer_mass default
    active = lice > min_mass
    return {
        'n_active': int(active.sum()),
        'total_ice': float(lice[active].sum()),
        'lice': lice[active].tolist(),
        'lheight': lheight[active].tolist(),
        'ltype': ltype[active].tolist(),
    }


if __name__ == '__main__':
    print(f"K53 layer comparison: WF_A={WF_A} vs WF_B={WF_B}, alpha={MELT_SMOOTH_ALPHA}")
    print("Building model and running forward at both WF values...")

    model = build_k53_model()

    t0 = time.time()
    snaps_a = run_forward_snapshots(model, WF_A)
    print(f"  WF={WF_A} done ({time.time()-t0:.1f}s), {len(snaps_a)} snapshots", flush=True)

    t0 = time.time()
    # Re-init state so second run starts from the same initial condition
    model2 = build_k53_model()
    snaps_b = run_forward_snapshots(model2, WF_B)
    print(f"  WF={WF_B} done ({time.time()-t0:.1f}s), {len(snaps_b)} snapshots", flush=True)

    print(f"\nComparing layer structure (only showing snapshots where n_active differs):")
    print(f"{'date':>12}  {'n(A)':>5}  {'n(B)':>5}  {'ice_A':>12}  {'ice_B':>12}  {'diff':>12}")
    print('-' * 70)

    any_diff = False
    for (date_a, state_a, _), (date_b, state_b, _) in zip(snaps_a, snaps_b):
        assert date_a == date_b, f"Snapshot dates mismatched: {date_a} vs {date_b}"
        s_a = summarize_layers(state_a)
        s_b = summarize_layers(state_b)
        if s_a['n_active'] != s_b['n_active'] or abs(s_a['total_ice'] - s_b['total_ice']) > 1e-3:
            any_diff = True
            diff = s_b['total_ice'] - s_a['total_ice']
            print(f"{str(date_a):>12}  {s_a['n_active']:>5}  {s_b['n_active']:>5}  "
                  f"{s_a['total_ice']:>12.4f}  {s_b['total_ice']:>12.4f}  {diff:>+12.4f}")
            if s_a['n_active'] != s_b['n_active']:
                print(f"    WF={WF_A} layers: ice={[f'{x:.4e}' for x in s_a['lice']]} "
                      f"type={s_a['ltype']}")
                print(f"    WF={WF_B} layers: ice={[f'{x:.4e}' for x in s_b['lice']]} "
                      f"type={s_b['ltype']}")

    if not any_diff:
        print("No differences in layer count or total ice found at snapshot resolution.")
        print("(Kink may be sub-monthly — need finer snapshots or a different probe.)")

    # Also print all snapshots for both to show the full trajectory
    print(f"\nFull trajectory (n_active layers and total ice):")
    print(f"{'date':>12}  {'n_A':>4}  {'n_B':>4}  {'ice_A':>12}  {'ice_B':>12}")
    print('-' * 55)
    for (date_a, state_a, _), (date_b, state_b, _) in zip(snaps_a, snaps_b):
        s_a = summarize_layers(state_a)
        s_b = summarize_layers(state_b)
        marker = ' *' if (s_a['n_active'] != s_b['n_active']) else ''
        print(f"{str(date_a):>12}  {s_a['n_active']:>4}  {s_b['n_active']:>4}  "
              f"{s_a['total_ice']:>12.4f}  {s_b['total_ice']:>12.4f}{marker}")
