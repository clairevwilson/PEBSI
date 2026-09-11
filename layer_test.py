"""
How few vertical layers can the model run with before the mass balance
it simulates changes?

max_nlayers caps the layer array; make_layers fills it from the same
exponential growth curve regardless, so lowering the cap truncates the
deep end of the column and leaves the near-surface distribution alone.
dz_toplayer, layer_growth, dz_snowlayer and min_dz are untouched.

Forward model only -- no autodiff, no observations, no loss. Each layer
count runs the same period and is compared against the tallest column on
the one output that matters: simulated mass balance.
"""
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import simulation as sim
from pebsi.main import main as pebsi_main

import jax
import jax.numpy as jnp
import numpy as np
import yaml

import AD_optimize as ad

START = '2019-01-01 00:00'
END = '2022-01-01 23:00'
LAYER_COUNTS = [50, 40, 35, 30, 25, 20, 15, 12, 10]
SCRATCH = os.environ.get('LAYER_SWEEP_DIR', '/tmp')


def run_one(nlayers):
    config = dict(ad.BASE_CONFIG)
    config.update(
        rgi_ids=[ad.translate_rgi[g]['6'] for g in ad.GLACIERS],
        method_distribute='adaptive',
        start_date=START,
        end_date=END,
        output_freq='daily',
        store_vars=['mass_balance'],
        store_data=False,
        debug=False,
        progress_bar=False,
        kp=ad.baseline['kp'],
        wind_factor=ad.baseline['wind_factor'],
        max_nlayers=nlayers,
        **ad.HOST_PATHS[ad.host],
    )
    config_fn = os.path.join(SCRATCH, f'config_layers_{nlayers}.yaml')
    with open(config_fn, 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    args = sim.get_args(parse=False).parse_args([])
    args.config_fn = config_fn
    model = sim.PEBSI(args)
    model.config.static_args = model.config.static_args._replace(
        store_vars=('mass_balance',), differentiable=False)
    model.config.params.store_vars = ('mass_balance',)
    model.initialize()

    state = model.spinup(model.initial_state)
    forcings = model.pack_forcings(model.params, model.dates, 0)

    t0 = time.time()
    _, records = pebsi_main(state, forcings, model.point_attrs,
                            model.config.static_args, model.config.dynamic_args)
    mb = np.asarray(jax.block_until_ready(records.mass_balance))
    elapsed = time.time() - t0

    del model, forcings, records, state
    jax.clear_caches()
    return dict(nlayers=nlayers, mb=mb, seconds=elapsed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--layers', type=int, nargs='*', default=LAYER_COUNTS)
    opts = parser.parse_args()

    ad.GLACIERS = ['gulkana']

    print(f'JAX backend: {jax.default_backend()}  devices: {jax.devices()}', flush=True)
    print(f"{ad.GLACIERS}, {START} to {END}, layers {opts.layers}", flush=True)

    results = []
    for nlayers in opts.layers:
        print(f'\n===== max_nlayers = {nlayers} =====', flush=True)
        try:
            res = run_one(nlayers)
        except Exception as err:
            print(f'  FAILED: {type(err).__name__}: {err}', flush=True)
            continue
        cum = res['mb'].sum(axis=0)
        print(f"  glacier-wide MB {cum.mean():+.5f} m w.e.  "
              f"({res['seconds']:.1f}s, {res['mb'].shape[1]} points)", flush=True)
        results.append(res)

    if not results:
        print('\nno layer count completed', flush=True)
        return

    ref = max(results, key=lambda r: r['nlayers'])
    ref_cum = ref['mb'].sum(axis=0)
    print(f"\n\nReference: {ref['nlayers']} layers, glacier-wide "
          f"{ref_cum.mean():+.5f} m w.e. over {ref['mb'].shape[0]} days, "
          f"{ref['seconds']:.1f}s\n", flush=True)

    head = (f"{'layers':>7} {'MB [m w.e.]':>13} {'diff':>11} {'diff %':>9} "
            f"{'max pt diff':>12} {'sec':>8} {'speedup':>8}")
    print(head, flush=True)
    print('-' * len(head), flush=True)
    for r in sorted(results, key=lambda x: -x['nlayers']):
        cum = r['mb'].sum(axis=0)
        diff = cum.mean() - ref_cum.mean()
        pct = 100.0 * diff / abs(ref_cum.mean()) if ref_cum.mean() != 0 else np.nan
        print(f"{r['nlayers']:>7} {cum.mean():>13.5f} {diff:>11.5f} {pct:>8.3f}% "
              f"{np.abs(cum - ref_cum).max():>12.5f} {r['seconds']:>8.1f} "
              f"{ref['seconds'] / r['seconds']:>7.2f}x", flush=True)

    print('\ndiff is the glacier-wide cumulative difference from the tallest '
          'column; max pt diff is the largest single-point difference, which '
          'catches a few points diverging while the mean still looks fine.',
          flush=True)


if __name__ == '__main__':
    main()
