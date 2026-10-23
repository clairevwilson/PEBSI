"""
Per-point gradient health check for PEBSI's differentiable core.

Runs one glacier for a few years with the shared physics settings
(BASE_CONFIG) and differentiates the glacier-wide mass balance with respect
to a PER-POINT parameter. Because the parameter is an (N_POINTS,) array,
reverse mode returns one gradient per point in a single backward pass, so a
few hundred points give a few hundred independent samples of the adjoint:
a blow-up that a small run might miss shows up as a handful of outliers.

What to look for:
  - no non-finite gradients
  - max |sensitivity| within an order of magnitude or two of the median; a
    few points orders of magnitude above the rest is the signature of a
    gradient blow-up, and their elevations show where on the glacier it is

Run it after any change to the physics or the albedo emulator, and before
trusting an AD calibration. Set PEBSI_FLOAT32=1 to check float32.

@author: clairevwilson
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# netCDF4 must load before JAX to avoid a numpy ABI conflict
import simulation as sim

import jax
jax.config.update("jax_traceback_filtering", "off")
import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml

from pebsi.main import main as pebsi_main
from project.parameters import host, HOST_PATHS, BASE_CONFIG, translate_rgi

GLACIER = 'gulkana'
N_POINTS = 300
START_YEAR = 2015
YEARS = 5
CHUNK_DAYS = 365

# parameter whose per-point gradient is checked ('wind_factor' or 'kp'),
# and the values both parameters are held at
PARAM = 'wind_factor'
KP = 2.5
WIND_FACTOR = 3.0


def build_config(start_date, end_date):
    config = dict(BASE_CONFIG)
    config.update(
        rgi_ids=[translate_rgi[GLACIER]['6']],
        method_distribute='grid',
        n_points=N_POINTS,
        start_date=start_date,
        end_date=end_date,
        output_freq='daily',
        store_vars=['mass_balance'],
        store_data=False,
        debug=False,
        progress_bar=False,
        kp=KP,
        wind_factor=WIND_FACTOR,
        **HOST_PATHS[host],
    )
    config['output_fp'] = config['output_fp'].replace('AD_optimize', 'gradient_check')
    config_fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '_gradient_check.yaml')
    with open(config_fn, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    return config_fn


def init_pebsi(config_fn):
    args = sim.get_args(parse=False).parse_args([])
    args.config_fn = config_fn
    model = sim.PEBSI(args)
    model.config.static_args = model.config.static_args._replace(
        store_vars=('mass_balance',), differentiable=True)
    model.config.params.store_vars = ('mass_balance',)
    model.initialize()
    print('Running spinup...', flush=True)
    model.initial_state = model.spinup(model.initial_state)
    print('Spinup complete.\n', flush=True)
    return model


def make_value_and_grad(model):
    """
    Returns value_and_grad taking log(PARAM) as an (N_POINTS,) array. The
    loss is the glacier-wide mean mass balance over the whole run [m w.e.];
    the aux output is the per-point total.

    Chunks are unrolled in a python loop rather than scanned, so they need
    not be equal length, and each is wrapped in jax.checkpoint so reverse
    mode holds only one chunk's interior at a time. Forcings are passed as
    an argument, never captured, or jit would fold them into the lowered
    HLO as multi-GB constants.
    """
    static_args = model.config.static_args
    dynamic_args = model.config.dynamic_args
    point_attrs = model.point_attrs

    def run_chunk(state, forcings, new_dynamic_args):
        return pebsi_main(state, forcings, point_attrs, static_args,
                          new_dynamic_args)

    def loss_fn(log_param, initial_state, chunk_forcings):
        new_dynamic_args = dynamic_args._replace(**{PARAM: jnp.exp(log_param)})
        state = initial_state
        per_point = jnp.zeros(model.terrain.N_POINTS)
        for forcings in chunk_forcings:
            state, records = jax.checkpoint(run_chunk)(
                state, forcings, new_dynamic_args)
            per_point = per_point + jnp.sum(records.mass_balance, axis=0)
        return jnp.mean(per_point), per_point

    return jax.jit(jax.value_and_grad(loss_fn, argnums=0, has_aux=True))


def report(grad, model, n_points):
    """
    Prints the per-point gradient distribution. Points are near-independent
    columns, so d(mean MB)/d(param_i) carries a 1/N_POINTS factor; multiplying
    it back out gives each point's own sensitivity [m w.e. per unit
    log(PARAM)], which is comparable across runs with different N_POINTS.
    """
    grad = np.asarray(grad)
    sensitivity = grad * n_points

    finite = np.isfinite(sensitivity)
    print(f'{"non-finite gradients":<28} {(~finite).sum()} of {n_points}')
    if not finite.any():
        print('  every gradient is non-finite -- nothing further to report')
        return
    if (~finite).any():
        bad = np.where(~finite)[0]
        print(f'  at point index {bad[:20].tolist()}'
              + (' ...' if len(bad) > 20 else ''))

    mag = np.abs(sensitivity[finite])
    zero = (mag == 0).sum()
    pct = np.percentile(mag, [50, 90, 99, 100])
    print(f'{"exactly zero":<28} {zero}')
    print(f'{"|sensitivity| median":<28} {pct[0]:.4e}')
    print(f'{"|sensitivity| p90":<28} {pct[1]:.4e}')
    print(f'{"|sensitivity| p99":<28} {pct[2]:.4e}')
    print(f'{"|sensitivity| max":<28} {pct[3]:.4e}')

    spread = pct[3] / pct[0] if pct[0] > 0 else np.inf
    print(f'{"max / median":<28} {spread:.1f}x')
    verdict = ('looks healthy' if spread < 100 else
               'SUSPICIOUS -- a few points dominate, check the tail below')
    print(f'{"verdict":<28} {verdict}\n')

    elev = np.asarray(model.terrain.elev_n)
    order = np.argsort(-np.abs(np.nan_to_num(sensitivity, nan=np.inf)))[:10]
    print('Largest |sensitivity| points:')
    print(f'  {"idx":>6} {"sensitivity":>14} {"elev [m]":>10} '
          f'{"lat":>9} {"lon":>10}')
    for i in order:
        print(f'  {i:>6} {sensitivity[i]:>14.4e} {elev[i]:>10.1f} '
              f'{model.terrain.lat_n[i]:>9.4f} {model.terrain.lon_n[i]:>10.4f}')


def main():
    start = pd.Timestamp(f'{START_YEAR}-01-01')
    end = pd.Timestamp(f'{START_YEAR + YEARS}-01-01') - pd.Timedelta(hours=1)
    print(f'JAX backend: {jax.default_backend()}  devices: {jax.devices()}')
    print(f'{GLACIER}, {N_POINTS} points, {start.date()} to {end.date()}, '
          f'per-point d/dlog({PARAM}) at kp={KP}, wind_factor={WIND_FACTOR}, '
          f'dust_factor={BASE_CONFIG.get("dust_factor")}\n', flush=True)

    model = init_pebsi(build_config(str(start), str(end)))
    n_points = model.terrain.N_POINTS
    n_days = len(model.dates) // 24
    print(f'{n_points} points placed, {n_days} days', flush=True)

    bounds = [(d, min(d + CHUNK_DAYS, n_days))
              for d in range(0, n_days, CHUNK_DAYS)]
    chunk_forcings = []
    for i, (d0, d1) in enumerate(bounds):
        chunk_forcings.append(model.pack_forcings(
            model.params, model.dates[d0 * 24:d1 * 24], d0 * 24))
        print(f'\033[2K\r~ Packing forcings [{i + 1}/{len(bounds)}] ~',
              end='', flush=True)
    nbytes = sum(x.nbytes for x in jax.tree.leaves(chunk_forcings))
    print(f'\n{len(bounds)} chunks, forcings {nbytes / 1e9:.2f} GB\n', flush=True)

    value_and_grad = make_value_and_grad(model)
    value = KP if PARAM == 'kp' else WIND_FACTOR
    log_param = jnp.log(jnp.full(n_points, value))

    print('Running forward + backward...', flush=True)
    t0 = time.time()
    (loss, per_point), grad = value_and_grad(log_param, model.initial_state,
                                             chunk_forcings)
    jax.block_until_ready((loss, grad))
    print(f'done in {time.time() - t0:.1f} s\n', flush=True)

    mb = np.asarray(per_point)
    print(f'{"glacier-wide MB":<28} {float(loss):+.4f} m w.e. over {YEARS} yr')
    print(f'{"per-point MB range":<28} {mb.min():+.3f} to {mb.max():+.3f} m w.e.\n')

    report(grad, model, n_points)


if __name__ == '__main__':
    main()
