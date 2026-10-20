"""
Times one value_and_grad call of AD_optimize's 2000-2025 objective at a
chosen point count, to find how many points one GPU can differentiate.

Points are spread by area (method_distribute='grid') over the largest
glaciers in RGI region 1 plus the five calibration glaciers the objective
scores, so the run touches many MERRA-2 cells and the climate and shading
tables are sized like a regional run's. The per-glacier ice albedo, ice
thickness and wind map files only exist for a few glaciers, so those
options are off.

Compiles ahead of time so compilation and execution are timed separately,
and prints XLA's memory analysis before running, so a size that cannot
fit is reported without waiting on the gradient. Nothing is written to
disk besides the per-job config and any missing shading files.
"""
import os
import time
import argparse

import AD_optimize as ad
import simulation as sim
from pebsi.io.terrain import Terrain

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml

N_LARGEST = 100

# eastern limit [deg E] for glacier centroids, with a margin inside the climate data
CLIMATE_EAST_EDGE = -134.0


def glacier_ids():
    """
    The N_LARGEST glaciers in region 1 by area that lie inside the
    climate data, plus any calibration glacier not among them, as RGI 6.0
    IDs without the 'RGI60-' prefix.
    """
    rgi_fn = os.path.join(ad.HOST_PATHS[ad.host]['rgi_fp'], '01_rgi60_Alaska.csv')
    rgi = pd.read_csv(rgi_fn, encoding='latin1')
    # most of the regional MERRA-2 files end at -133.75 E, so glaciers
    # east of it would read a climate cell too far away
    rgi = rgi[rgi['CenLon'] <= CLIMATE_EAST_EDGE]
    largest = rgi.nlargest(N_LARGEST, 'Area')['RGIId'].str.replace('RGI60-', '')
    ids = list(largest)
    for name in ad.GLACIERS:
        gid = ad.translate_rgi[name]['6']
        if gid not in ids:
            ids.append(gid)
    return ids


def build_config(n_points, checkpoint_hourly, tolerance=None):
    if n_points is None:
        overrides = dict(checkpoint_hourly=checkpoint_hourly)
        tag = 'AD_timing_adaptive'
        if tolerance is not None:
            overrides['point_error_tolerance'] = tolerance
            tag += f'_{tolerance}'
        return ad.build_config(overrides, tag=tag)
    overrides = dict(
        checkpoint_hourly=checkpoint_hourly,
        rgi_ids=glacier_ids(),
        method_distribute='grid',
        n_points=n_points,
        option_ice_albedo_tif=False,
        option_ice_thickness_tif=False,
        option_windmaps=False,
    )
    return ad.build_config(overrides, tag=f'AD_timing_{n_points}')


def directional(value_and_grad, log_kp, log_wf, g_param):
    """
    Forward-mode derivative of the (term, glacier) loss components with
    respect to one glacier's log(kp), carrying a single tangent. An
    independent check of that glacier's reverse-mode d/dlog_kp, split by
    term; every other glacier's entries should be exactly zero.
    """
    terms = ('albedo', 'wet', 'mb')
    n_glaciers = len(ad.GLACIERS)
    args = value_and_grad.args

    def components(lkp):
        _, metrics = value_and_grad.loss_fn(lkp, jnp.asarray(log_wf), *args)
        return jnp.stack([metrics[t] for t in terms])

    tangent = np.zeros(n_glaciers)
    tangent[g_param] = 1.0
    fn = jax.jit(lambda lkp, t: jax.jvp(components, (lkp,), (t,)))
    lkp = jnp.asarray(log_kp)

    t0 = time.time()
    compiled = fn.lower(lkp, jnp.asarray(tangent)).compile()
    print(f'jvp compile: {(time.time() - t0) / 60:.1f} min', flush=True)

    t0 = time.time()
    values, derivs = compiled(lkp, jnp.asarray(tangent))
    values, derivs = np.asarray(values), np.asarray(derivs)
    print(f'jvp run: {(time.time() - t0) / 3600:.2f} hr', flush=True)

    print(f'\nd(term)/d(log_kp of {ad.GLACIERS[g_param]}):')
    print(f'  {"glacier":<12}' + ''.join(f'{t:>14}' for t in terms) + f'{"sum":>14}')
    for g, name in enumerate(ad.GLACIERS):
        d = derivs[:, g]
        print(f'  {name:<12}' + ''.join(f'{v:>14.4e}' for v in d) + f'{d.sum():>14.4e}')

    print('\nterm values:')
    print(f'  {"glacier":<12}' + ''.join(f'{t:>14}' for t in terms))
    for g, name in enumerate(ad.GLACIERS):
        print(f'  {name:<12}' + ''.join(f'{v:>14.4f}' for v in values[:, g]))
    print(f'non-finite entries: {(~np.isfinite(derivs)).sum()} of {derivs.size}', flush=True)


def jacobian(value_and_grad, log_kp, log_wf):
    """
    Forward-mode Jacobian of the (term, glacier) loss components with
    respect to all 2 * n_glaciers parameters, in one pass carrying every
    tangent. It never touches the backward pass or its checkpoints, so it
    is an independent check of the reverse-mode gradient, and it shows
    which term a large gradient comes from. Cross-glacier entries should
    be exactly zero.
    """
    terms = ('albedo', 'wet', 'mb')
    n_glaciers = len(ad.GLACIERS)
    args = value_and_grad.args

    def components(params):
        _, metrics = value_and_grad.loss_fn(params[:n_glaciers], params[n_glaciers:], *args)
        out = jnp.stack([metrics[t] for t in terms])
        return out, out

    params = jnp.concatenate([jnp.asarray(log_kp), jnp.asarray(log_wf)])
    t0 = time.time()
    jac, values = jax.jit(jax.jacfwd(components, has_aux=True))(params)
    jac, values = np.asarray(jac), np.asarray(values)   # jac is (term, glacier, param)
    print(f'jacfwd: {(time.time() - t0) / 3600:.2f} hr', flush=True)

    for label, offset in (('log_kp', 0), ('log_wind_factor', n_glaciers)):
        print(f'\nd(term)/d({label}) of the same glacier:')
        print(f'  {"glacier":<12}' + ''.join(f'{t:>14}' for t in terms) + f'{"sum":>14}')
        for g, name in enumerate(ad.GLACIERS):
            own = jac[:, g, offset + g]
            print(f'  {name:<12}' + ''.join(f'{v:>14.4e}' for v in own) + f'{own.sum():>14.4e}')

    print('\nterm values:')
    print(f'  {"glacier":<12}' + ''.join(f'{t:>14}' for t in terms))
    for g, name in enumerate(ad.GLACIERS):
        print(f'  {name:<12}' + ''.join(f'{v:>14.4f}' for v in values[:, g]))

    own = np.zeros_like(jac, dtype=bool)
    for g in range(n_glaciers):
        own[:, g, g] = own[:, g, n_glaciers + g] = True
    cross = np.abs(np.where(own, 0.0, jac))
    print(f'\nlargest cross-glacier entry: {cross.max():.3e}')
    if cross.max() > 0:
        t, g, p = np.unravel_index(cross.argmax(), cross.shape)
        print(f'  d({terms[t]} of {ad.GLACIERS[g]})/d(param {p})')
    print(f'non-finite entries: {(~np.isfinite(jac)).sum()} of {jac.size}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n_points', type=int, default=None,
                        help='grid points over the largest glaciers; omit for '
                             "AD_optimize's own adaptive mesh on the calibration glaciers")
    parser.add_argument('--chunk_days', type=int, nargs='+', default=[96],
                        help='more than one only with --compile_only')
    parser.add_argument('--shading_only', action='store_true',
                        help='only compute the missing shading files, then exit')
    parser.add_argument('--tolerance', type=float, default=None,
                        help='point_error_tolerance for the adaptive mesh (no --n_points); '
                             'points scale as 1/tolerance^2, ~1900 at the default 3.0')
    parser.add_argument('--checkpoint_hourly', action='store_true',
                        help='also checkpoint each hourly step (see pebsi/defaults.py)')
    parser.add_argument('--jacfwd', action='store_true',
                        help='forward-mode Jacobian of every per-glacier loss term '
                             'with respect to every glacier\'s log(kp) and log(wind_factor)')
    parser.add_argument('--jvp', choices=ad.GLACIERS,
                        help='forward-mode derivative of every per-glacier loss term '
                             'with respect to this glacier\'s log(kp)')
    parser.add_argument('--at_result', action='store_true',
                        help='evaluate at the parameters in AD_optimize_result.yaml instead of the baseline')
    parser.add_argument('--compile_only', action='store_true',
                        help='compile at each chunk_days and print the memory analysis, without running')
    opts = parser.parse_args()
    assert opts.compile_only or len(opts.chunk_days) == 1, \
        'several chunk_days only make sense with --compile_only'

    device = jax.devices()[0]
    print(f'JAX backend: {jax.default_backend()}  device: {device.device_kind}', flush=True)

    t0 = time.time()
    config_fn = build_config(opts.n_points, opts.checkpoint_hourly, opts.tolerance)

    if opts.shading_only:
        args = sim.get_args(parse=False).parse_args([])
        args.config_fn = config_fn
        model = sim.PEBSI(args)
        print(f'{len(model.params.rgi_ids)} glaciers', flush=True)
        Terrain(model.params).run_dem_functions()
        print(f'shading done: {(time.time() - t0) / 60:.1f} min', flush=True)
        return

    model = ad.init_pebsi(config_fn)
    model_days = model.dates[::24]
    n_days = len(model_days)
    n_points = model.terrain.N_POINTS
    print(f'{n_points} points on {len(model.params.rgi_ids)} glaciers, '
          f'{n_days} days', flush=True)
    print(f'init + spinup: {time.time() - t0:.0f} s', flush=True)

    t0 = time.time()
    obs = ad.build_observations(model, model_days)
    print(f'observations: {time.time() - t0:.0f} s', flush=True)

    n_glaciers = len(ad.GLACIERS)
    log_kp = np.log(np.full(n_glaciers, ad.baseline['kp']))
    log_wf = np.log(np.full(n_glaciers, ad.baseline['wind_factor']))
    if opts.at_result:
        with open(ad.HOST_PATHS[ad.host]['output_fp'] + 'AD_optimize_result.yaml') as f:
            result = yaml.safe_load(f)
        log_kp, log_wf = np.log(result['kp']), np.log(result['wind_factor'])
        print(f'evaluating at AD_optimize result: kp {result["kp"]}, '
              f'wind_factor {result["wind_factor"]}', flush=True)

    for chunk_days in opts.chunk_days:
        print(f'\n========== chunk_days={chunk_days} ==========', flush=True)
        chunk_bounds = [(s, min(s + chunk_days, n_days))
                        for s in range(0, n_days, chunk_days)]
        stacked, tail, counts, march = ad.build_chunk_data(model, obs, model_days, chunk_bounds)
        value_and_grad, _ = ad.make_loss_fn(model, obs, counts, march, stacked, tail)

        t0 = time.time()
        compiled = value_and_grad.lower(log_kp, log_wf).compile()
        t_compile = time.time() - t0
        print(f'compile: {t_compile / 60:.1f} min', flush=True)

        mem = compiled.memory_analysis()
        temp_gb = mem.temp_size_in_bytes / 1e9
        print(f'XLA memory analysis: temp {temp_gb:.1f} GB '
              f'({mem.temp_size_in_bytes / n_points / 1e6:.2f} MB per point), '
              f'arguments {mem.argument_size_in_bytes / 1e9:.1f} GB, '
              f'output {mem.output_size_in_bytes / 1e9:.1f} GB', flush=True)
        stats = device.memory_stats() or {}
        print(f'device limit: {stats.get("bytes_limit", 0) / 1e9:.1f} GB  '
              f'in use: {stats.get("bytes_in_use", 0) / 1e9:.1f} GB', flush=True)

        if opts.jvp:
            directional(value_and_grad, log_kp, log_wf, ad.GLACIERS.index(opts.jvp))
            continue

        if opts.jacfwd:
            jacobian(value_and_grad, log_kp, log_wf)
            continue

        if opts.compile_only:
            print(f'RESULT points={n_points} chunk_days={chunk_days} '
                  f'compile_min={t_compile / 60:.1f} temp_GB={temp_gb:.1f}', flush=True)
            del compiled, stacked, tail, value_and_grad
            continue

        t0 = time.time()
        (total, metrics), grads = compiled(log_kp, log_wf, *value_and_grad.args)
        jax.block_until_ready(grads)
        t_run = time.time() - t0

        stats = device.memory_stats() or {}
        peak = stats.get('peak_bytes_in_use', float('nan')) / 1e9
        print(f'loss {float(total):.2f}  d/dlog_kp {np.asarray(grads[0])}  '
              f'd/dlog_wind_factor {np.asarray(grads[1])}', flush=True)
        print(f'\nRESULT points={n_points} chunk_days={chunk_days} '
              f'compile_min={t_compile / 60:.1f} gradient_hr={t_run / 3600:.2f} '
              f'peak_GB={peak:.1f}', flush=True)

if __name__ == '__main__':
    main()
