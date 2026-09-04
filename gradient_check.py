"""
Per-point gradient health check for PEBSI's differentiable core.

Runs one glacier for a few years and differentiates the glacier-wide mass
balance with respect to a PER-POINT wind_factor. Because wind_factor is an
(N_POINTS,) array, reverse mode returns one gradient per point in a single
backward pass -- so a few hundred points give a few hundred independent
samples of the adjoint, which is what makes this a useful test of whether
the gradient blow-ups are actually gone rather than just absent at the one
or two points a small run happens to cover.

The heat equation solver is selectable (-heateq implicit|explicit) so the
same run can be repeated against the old scheme for comparison.

What to look for:
  - no non-finite gradients
  - max |grad| within an order of magnitude or two of the median; a single
    point orders of magnitude above the rest is the signature of the old
    blow-up
  - with -fd, the analytic gradient matching finite differences

@author: clairevwilson
"""
import os
import sys
import time
import socket
import argparse

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

GLACIER_RGI = '01.00570'   # Gulkana

BASE_FP = ('/trace/group/rounce/cvwilson/' if 'trace' in socket.gethostname()
           else '/ocean/projects/ees260009p/cwilson4/')

PATHS = dict(
    climate_fp=BASE_FP + 'climate_data/',
    rgi_fp=BASE_FP + 'RGI/rgi60/00_rgi60_attribs/',
    output_fp=BASE_FP + 'Output/gradient_check/',
    cop30_vrt_path=BASE_FP + 'data/dems/COP30/COP30_reg01.vrt',
    shading_fp=BASE_FP + 'data/shading/',
    ice_albedo_fn=BASE_FP + 'data/ice_albedo/{gid}_albedo.tif',
    windmap_fn=BASE_FP + 'data/windmapper/{gid}.nc',
)


def build_config(n_points, start_date, end_date, heateq):
    config = dict(
        rgi_ids=[GLACIER_RGI],
        method_distribute='grid',
        n_points=n_points,
        start_date=start_date,
        end_date=end_date,
        output_freq='daily',
        store_vars=['mass_balance'],
        store_data=False,
        debug=False,
        progress_bar=False,
        method_heateq=heateq,
        option_ice_albedo_tif=True,
        option_windmaps=True,
        option_accel_grains=True,
        option_flat_plates=True,
        option_dynamics=False,
        constant_freshgrainsize=54.5,
        bias_vars=['temp'],
        kp=2.5,
        wind_factor=3.0,
        **PATHS,
    )
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


def make_fns(model):
    """
    Returns (value_and_grad, forward), both taking log(wind_factor) as an
    (N_POINTS,) array. The loss is the glacier-wide mean mass balance over
    the whole run [m w.e.]; the aux output is the per-point total.

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

    def loss_fn(log_wind_factor, initial_state, chunk_forcings):
        new_dynamic_args = dynamic_args._replace(
            wind_factor=jnp.exp(log_wind_factor))
        state = initial_state
        per_point = jnp.zeros(model.terrain.N_POINTS)
        for forcings in chunk_forcings:
            state, records = jax.checkpoint(run_chunk)(
                state, forcings, new_dynamic_args)
            per_point = per_point + jnp.sum(records.mass_balance, axis=0)
        return jnp.mean(per_point), per_point

    vg = jax.jit(jax.value_and_grad(loss_fn, argnums=0, has_aux=True))
    fwd = jax.jit(loss_fn)
    return vg, fwd


def report(grad, model, n_points):
    """
    Prints the per-point gradient distribution. Points are near-independent
    columns, so d(mean MB)/d(wf_i) carries a 1/N_POINTS factor; multiplying
    it back out gives each point's own sensitivity [m w.e. per unit
    wind_factor], which is comparable across runs with different N_POINTS.
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


def finite_difference(fwd, log_wf, initial_state, chunk_forcings,
                      grad, n_check, eps=1e-3):
    """
    Central-difference check on a few points, chosen as the largest-gradient
    points plus a random sample. Each probe is two more full forward runs,
    so this is opt-in.
    """
    grad = np.asarray(grad)
    order = np.argsort(-np.abs(np.nan_to_num(grad, nan=np.inf)))
    rng = np.random.default_rng(0)
    picks = list(dict.fromkeys(
        order[:max(1, n_check // 2)].tolist()
        + rng.choice(len(grad), size=n_check, replace=False).tolist()))[:n_check]

    print(f'\nFinite-difference check on {len(picks)} point(s), eps={eps}:')
    print(f'  {"idx":>6} {"analytic":>14} {"finite diff":>14} {"rel err":>10}')
    for i in picks:
        plus = fwd(log_wf.at[i].add(eps), initial_state, chunk_forcings)[0]
        minus = fwd(log_wf.at[i].add(-eps), initial_state, chunk_forcings)[0]
        fd = float((plus - minus) / (2 * eps))
        rel = abs(fd - grad[i]) / max(abs(fd), 1e-30)
        flag = '' if rel < 1e-3 else '   <-- mismatch'
        print(f'  {i:>6} {grad[i]:>14.6e} {fd:>14.6e} {rel:>10.2e}{flag}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-n', '--n_points', type=int, default=300)
    parser.add_argument('-y', '--years', type=int, default=5)
    parser.add_argument('-start', '--start_year', type=int, default=2015)
    parser.add_argument('-heateq', '--method_heateq', default='implicit',
                        choices=['implicit', 'explicit'])
    parser.add_argument('-cd', '--chunk_days', type=int, default=365)
    parser.add_argument('-wf', '--wind_factor', type=float, default=3.0)
    parser.add_argument('-fd', '--finite_difference', type=int, default=0,
                        help='number of points to verify by finite differences '
                             '(each costs two extra forward runs)')
    opts = parser.parse_args()

    start = pd.Timestamp(f'{opts.start_year}-01-01')
    end = pd.Timestamp(f'{opts.start_year + opts.years}-01-01') - pd.Timedelta(hours=1)
    print(f'JAX backend: {jax.default_backend()}  devices: {jax.devices()}')
    print(f'Gulkana, {opts.n_points} points, {start.date()} to {end.date()}, '
          f'heat equation: {opts.method_heateq}\n', flush=True)

    model = init_pebsi(build_config(opts.n_points, str(start), str(end),
                                    opts.method_heateq))
    n_points = model.terrain.N_POINTS
    n_days = len(model.dates) // 24
    print(f'{n_points} points placed, {n_days} days', flush=True)

    bounds = [(d, min(d + opts.chunk_days, n_days))
              for d in range(0, n_days, opts.chunk_days)]
    chunk_forcings = []
    for i, (d0, d1) in enumerate(bounds):
        chunk_forcings.append(model.pack_forcings(
            model.params, model.dates[d0 * 24:d1 * 24], d0 * 24))
        print(f'\033[2K\r~ Packing forcings [{i + 1}/{len(bounds)}] ~',
              end='', flush=True)
    nbytes = sum(x.nbytes for x in jax.tree.leaves(chunk_forcings))
    print(f'\n{len(bounds)} chunks, forcings {nbytes / 1e9:.2f} GB\n', flush=True)

    value_and_grad, fwd = make_fns(model)
    log_wf = jnp.log(jnp.full(n_points, opts.wind_factor, dtype=jnp.float64))

    print('Running forward + backward...', flush=True)
    t0 = time.time()
    (loss, per_point), grad = value_and_grad(log_wf, model.initial_state,
                                             chunk_forcings)
    jax.block_until_ready((loss, grad))
    print(f'done in {time.time() - t0:.1f} s\n', flush=True)

    mb = np.asarray(per_point)
    print(f'{"glacier-wide MB":<28} {float(loss):+.4f} m w.e. '
          f'over {opts.years} yr')
    print(f'{"per-point MB range":<28} {mb.min():+.3f} to {mb.max():+.3f} m w.e.\n')

    report(grad, model, n_points)

    if opts.finite_difference:
        finite_difference(fwd, log_wf, model.initial_state, chunk_forcings,
                          np.asarray(grad), opts.finite_difference)


if __name__ == '__main__':
    main()
