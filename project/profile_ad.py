"""
Cost attribution for PEBSI's differentiable core.

Answers, with measurements rather than guesses, where the wall clock of one
forward+backward pass actually goes, and what a 20-year 1600-point optimizer
step would really cost.

Modes
-----
  ratio    forward-only vs forward+backward, at several run lengths. This is
           the licence for every other mode: if the ratio is stable, the cheap
           forward-only sweeps below can be scaled up to the real AD cost.
  points   wall time vs N_POINTS at fixed length. Flat => launch-latency-bound
           (points are nearly free, attack step count and op count); linear =>
           bandwidth-bound (attack precision, layers, array traffic).
  time     wall time vs run length at fixed N. The slope is the marginal cost
           per simulated hour; the intercept is fixed setup.
  ablate   leave-one-out timing. Each row prices exactly one suspect.
  census   static graph size: jaxpr equations, HLO while/fusion counts, and
           cost_analysis FLOPs vs device peak. No timing, no GPU time.

Every timing separates compile from run by calling twice: the first call pays
compilation, the minimum of the repeats afterwards is the warm cost.

@author: profiling harness for claire's AD calibration
"""
import os
import sys
import csv
import time
import socket
import argparse
import functools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# netCDF4 must load before JAX (numpy ABI conflict); gradient_check does this
import gradient_check as gc

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from pebsi.main import main as pebsi_main
from pebsi.physics.massbalance import MassBalanceDriver
from pebsi.physics.energybalance import EnergyBalanceDriver

# V100-32 fp64 peak and HBM2 bandwidth, for the launch-bound verdict
DEVICE_PEAK = {'v100': (7.8e12, 900e9), 'a100': (9.7e12, 1555e9), 'h100': (34e12, 3350e9)}


# ---------------------------------------------------------------------------
# Platform guard -- three of the last seven runs silently fell back to CPU
# ---------------------------------------------------------------------------

def guard(require_gpu=True):
    backend = jax.default_backend()
    node = os.environ.get('SLURMD_NODENAME', socket.gethostname())
    print(f'node {node}   backend {backend}   devices {jax.devices()}', flush=True)
    if require_gpu and backend != 'gpu':
        raise SystemExit(
            f'refusing to profile on {backend}: jaxlib fell back off the GPU '
            f'(node {node} likely has a CUDA driver mismatch). Re-submit with '
            f'--exclude={node}, or pass -cpu to profile the CPU path on purpose.')
    return node


def enable_compile_cache(path):
    """Persistent XLA cache: each config here costs ~2 min to compile, and the
    sweeps recompile constantly. Purely infrastructural, changes no numerics."""
    os.makedirs(path, exist_ok=True)
    jax.config.update('jax_compilation_cache_dir', path)
    jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 1.0)


# ---------------------------------------------------------------------------
# Ablation patches -- applied to the driver classes, reverted after each run,
# so the model source is never modified
# ---------------------------------------------------------------------------

def tridiagonal_implicit_conduction(self, state, params, lcond, is_temperate):
    """
    Identical backward-Euler discretization to
    pebsi/physics/massbalance.py:implicit_conduction, but the tridiagonal
    system is solved as a tridiagonal system instead of being padded into a
    dense (N_POINTS, 50, 50) matrix for a general LU.

    Same equations, same stability, same implicit scheme -- only the linear
    algebra changes, O(n) instead of O(n^3). jax.lax.linalg.tridiagonal_solve
    has JVP, transpose and batching rules and lowers to cusparse gtsv2 on GPU.
    The diagonal is strictly dominant (cap_dt > 0 plus both off-diagonal
    magnitudes), so gtsv2's pivot-free cyclic reduction is safe here.
    """
    CP_ICE = params.Cp_ice
    TEMP_TEMP = params.temp_temp

    surftemp = state.surftemp
    lheight = state.lheight
    ldensity = state.ldensity
    ltemp = state.ltemp

    safe_lheight = jnp.where(lheight > 0, lheight, 1.0)

    dz = 0.5 * (lheight[:, :-1] + lheight[:, 1:])
    safe_dz = jnp.where(dz > 0, dz, 1.0)
    k_inter = 0.5 * (lcond[:, :-1] + lcond[:, 1:])
    k_iface = k_inter / safe_dz
    k_top = lcond[:, 0] / (0.5 * safe_lheight[:, 0])

    therm_mass = CP_ICE * ldensity * safe_lheight
    valid = (lheight > 0) & (therm_mass > 0)
    cap_dt = jnp.where(valid, therm_mass, 1.0) / params.dt

    lower = jnp.zeros_like(ltemp).at[:, 1:].set(-k_iface)
    upper = jnp.zeros_like(ltemp).at[:, :-1].set(-k_iface)
    diag = cap_dt - lower - upper
    diag = diag.at[:, 0].add(k_top)
    b_vec = cap_dt * ltemp
    b_vec = b_vec.at[:, 0].add(k_top * surftemp)

    frozen = (~valid) | is_temperate
    frozen = frozen.at[:, -1].set(True)
    lower = jnp.where(frozen, 0.0, lower)
    upper = jnp.where(frozen, 0.0, upper)
    diag = jnp.where(frozen, 1.0, diag)
    b_vec = jnp.where(frozen, ltemp, b_vec)

    # lower[:, 0] and upper[:, -1] are already zero, as the primitive requires
    final_temperatures = jax.lax.linalg.tridiagonal_solve(
        lower, diag, upper, b_vec[..., None])[..., 0]
    final_temperatures = jnp.where(is_temperate, TEMP_TEMP, final_temperatures)

    return state._replace(ltemp=final_temperatures)


def make_secant_solver(n_iter):
    """
    Mirrors pebsi/physics/energybalance.py:solve_energy_balance with the
    fixed-length secant loop made configurable, so the ablation can price the
    root find's 3 + n_iter + 1 flux evaluations per hourly step. Only used for
    timing -- a shortened loop is not a converged surface temperature.
    """
    def solve_energy_balance(self, state, forcings, point_attrs):
        t_melt = jnp.zeros_like(state.surftemp)
        y_melt, _ = self.compute_fluxes(t_melt, state, forcings, point_attrs)

        t0 = state.surftemp
        t1 = forcings.temp
        y0, _ = self.compute_fluxes(t0, state, forcings, point_attrs)
        y1, _ = self.compute_fluxes(t1, state, forcings, point_attrs)

        def secant_step(carry, i):
            t_prev, t_curr, y_prev, y_curr = carry
            converged = jnp.abs(y_curr) < 1e-6
            denom = jnp.where(jnp.abs(y_curr - y_prev) < 1e-4, 1e-4, y_curr - y_prev)
            t_next_secant = t_curr - y_curr * (t_curr - t_prev) / denom
            t_next_secant = jnp.clip(t_next_secant, -60.0, 0.0)
            t_next = jnp.where(converged, t_curr, t_next_secant)
            y_next_secant, _ = self.compute_fluxes(t_next, state, forcings, point_attrs)
            y_next = jnp.where(converged, y_curr, y_next_secant)
            return (t_curr, t_next, y_curr, y_next), None

        carry = (t0, t1, y0, y1)
        if n_iter > 0:
            carry, _ = jax.lax.scan(secant_step, carry,
                                    xs=jnp.arange(n_iter), length=n_iter)
        surftemp_cooling = jnp.clip(carry[1], -60.0, 0.0)

        is_melting = y_melt > 0.0
        surftemp_final = jnp.where(is_melting, t_melt, surftemp_cooling)
        _, fluxes = self.compute_fluxes(surftemp_final, state, forcings, point_attrs)
        return state._replace(surftemp=surftemp_final), fluxes

    return solve_energy_balance


class patched:
    """Context manager applying a set of (class, attr, fn) monkeypatches."""

    def __init__(self, *patches):
        self.patches = [p for p in patches if p is not None]

    def __enter__(self):
        self.saved = [(c, a, getattr(c, a)) for c, a, _ in self.patches]
        for c, a, fn in self.patches:
            setattr(c, a, fn)
        return self

    def __exit__(self, *exc):
        for c, a, old in self.saved:
            setattr(c, a, old)
        return False


PATCHES = {
    'tridiag': lambda: (MassBalanceDriver, 'implicit_conduction',
                        tridiagonal_implicit_conduction),
    'secant4': lambda: (EnergyBalanceDriver, 'solve_energy_balance',
                        make_secant_solver(4)),
    'secant1': lambda: (EnergyBalanceDriver, 'solve_energy_balance',
                        make_secant_solver(1)),
    'secant0': lambda: (EnergyBalanceDriver, 'solve_energy_balance',
                        make_secant_solver(0)),
}


# ---------------------------------------------------------------------------
# Model setup and timing
# ---------------------------------------------------------------------------

def build(n_points, start_year, days, heateq='implicit', spinup=False, **overrides):
    """
    Builds a Gulkana model and packs its forcings. Spinup is off by default:
    it costs ~3 min and the timing does not depend on the state values, only
    on their shapes.
    """
    start = pd.Timestamp(f'{start_year}-01-01')
    end = start + pd.Timedelta(days=days) - pd.Timedelta(hours=1)
    config_fn = gc.build_config(n_points, str(start), str(end), heateq)

    # gc.build_config writes a fixed path, so concurrently-running profile jobs
    # would read each other's config. Give each process its own copy.
    import shutil
    unique = config_fn.replace('.yaml', f'_{os.getpid()}.yaml')
    shutil.copy(config_fn, unique)
    config_fn = unique

    if overrides:
        import yaml
        with open(config_fn) as f:
            cfg = yaml.safe_load(f)
        cfg.update(overrides)
        with open(config_fn, 'w') as f:
            yaml.dump(cfg, f, sort_keys=False)

    import simulation as sim
    args = sim.get_args(parse=False).parse_args([])
    args.config_fn = config_fn
    model = sim.PEBSI(args)
    model.config.static_args = model.config.static_args._replace(
        store_vars=('mass_balance',), differentiable=True)
    model.config.params.store_vars = ('mass_balance',)
    model.initialize()
    if spinup:
        model.initial_state = model.spinup(model.initial_state)

    n_days = len(model.dates) // 24
    chunk = min(365, n_days)
    bounds = [(d, min(d + chunk, n_days)) for d in range(0, n_days, chunk)]
    forcings = [model.pack_forcings(model.params, model.dates[d0 * 24:d1 * 24], d0 * 24)
                for d0, d1 in bounds]
    return model, forcings, n_days


def timed(fn, *args, repeat=1):
    """Returns (compile_s, warm_s). The first call pays compilation; the
    minimum of the repeats is the steady-state cost."""
    t0 = time.time()
    out = fn(*args)
    jax.block_until_ready(out)
    first = time.time() - t0

    warm = np.inf
    for _ in range(repeat):
        t0 = time.time()
        out = fn(*args)
        jax.block_until_ready(out)
        warm = min(warm, time.time() - t0)
    return max(first - warm, 0.0), warm


def bench(model, forcings, n_days, both=True, repeat=1):
    """Times forward-only and (optionally) forward+backward for one config."""
    vg, fwd = gc.make_fns(model)
    log_wf = jnp.log(jnp.full(model.terrain.N_POINTS, 3.0, dtype=jnp.float64))
    row = {'n_points': model.terrain.N_POINTS, 'days': n_days,
           'steps': n_days * 24}

    c, w = timed(fwd, log_wf, model.initial_state, forcings, repeat=repeat)
    row['fwd_compile_s'], row['fwd_s'] = c, w
    if both:
        c, w = timed(vg, log_wf, model.initial_state, forcings, repeat=repeat)
        row['vg_compile_s'], row['vg_s'] = c, w
        row['ratio'] = row['vg_s'] / row['fwd_s']
    row['ms_per_step'] = 1e3 * (row.get('vg_s') or row['fwd_s']) / row['steps']
    return row


def show(rows, cols, title):
    print(f'\n=== {title} ===', flush=True)
    print('  '.join(f'{c:>14}' for c in cols))
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            cells.append(f'{v:>14.3f}' if isinstance(v, float) else f'{str(v):>14}')
        print('  '.join(cells))


def save(rows, path):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f'\nwrote {path}', flush=True)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def mode_ratio(opts):
    """Is the forward pass a valid proxy for the AD cost?

    Times both at several run lengths. A ratio that is flat across lengths
    means forward-only sweeps can be scaled by it; a drifting ratio means the
    backward pass has a length-dependent cost (checkpoint bookkeeping, stored
    carries) and forward-only extrapolation would understate long runs.
    """
    rows = []
    for days in opts.days_list:
        model, forcings, n = build(opts.n_points, opts.start_year, days)
        r = bench(model, forcings, n, both=True, repeat=opts.repeat)
        rows.append(r)
        show([r], ['days', 'steps', 'fwd_s', 'vg_s', 'ratio', 'ms_per_step'],
             f'{days} d')
    show(rows, ['days', 'steps', 'fwd_s', 'fwd_compile_s', 'vg_s',
                'vg_compile_s', 'ratio', 'ms_per_step'], 'fwd vs fwd+bwd')

    ratios = [r['ratio'] for r in rows]
    spread = max(ratios) / min(ratios)
    print(f'\nratio range {min(ratios):.2f}-{max(ratios):.2f}  (spread {spread:.2f}x)')
    print('verdict: ' + ('forward-only is a sound proxy; scale sweeps by the mean ratio'
                         if spread < 1.15 else
                         'ratio drifts with length -- do not extrapolate forward-only '
                         'timings without re-measuring the ratio at that length'))
    save(rows, opts.out or 'project/profile_ratio.csv')
    return rows


def mode_points(opts):
    """Flat => launch-bound => 1600 points are nearly free."""
    rows = []
    for n in opts.points_list:
        model, forcings, nd = build(n, opts.start_year, opts.days)
        r = bench(model, forcings, nd, both=not opts.fwd_only, repeat=opts.repeat)
        rows.append(r)
        show([r], ['n_points', 'fwd_s', 'vg_s', 'ms_per_step'], f'N={n}')
    show(rows, ['n_points', 'days', 'fwd_s', 'vg_s', 'ms_per_step'],
         'point scaling')

    key = 'fwd_s' if opts.fwd_only else 'vg_s'
    lo, hi = rows[0], rows[-1]
    growth = (hi[key] / lo[key]) / (hi['n_points'] / lo['n_points'])
    print(f'\ntime grew {hi[key] / lo[key]:.2f}x for a '
          f'{hi["n_points"] / lo["n_points"]:.2f}x point increase '
          f'(efficiency {growth:.2f})')
    print('verdict: ' + ('LAUNCH-BOUND -- points are nearly free; attack step count '
                         'and per-step op count' if growth < 0.35 else
                         'BANDWIDTH/FLOP-BOUND -- point count and precision are real '
                         'levers' if growth > 0.7 else
                         'mixed regime -- partially saturated'))
    save(rows, opts.out or 'project/profile_points.csv')
    return rows


def mode_time(opts):
    """Slope gives the marginal cost per simulated hour; intercept the fixed cost."""
    rows = []
    for days in opts.days_list:
        model, forcings, n = build(opts.n_points, opts.start_year, days)
        rows.append(bench(model, forcings, n, both=not opts.fwd_only,
                          repeat=opts.repeat))
        show([rows[-1]], ['days', 'fwd_s', 'vg_s', 'ms_per_step'], f'{days} d')
    show(rows, ['days', 'steps', 'fwd_s', 'vg_s', 'ms_per_step'], 'time scaling')

    key = 'fwd_s' if opts.fwd_only else 'vg_s'
    x = np.array([r['steps'] for r in rows], float)
    y = np.array([r[key] for r in rows], float)
    slope, intercept = np.polyfit(x, y, 1)
    print(f'\nmarginal cost {slope * 1e3:.3f} ms per simulated hour, '
          f'fixed cost {intercept:.1f} s')
    target_steps = 7396 * 24
    print(f'projection at this N: {(slope * target_steps + intercept) / 3600:.2f} h '
          f'for the full 2000-2020 run ({target_steps} steps)')
    save(rows, opts.out or 'project/profile_time.csv')
    return rows


def mode_ablate(opts):
    """
    Leave-one-out. Rows above the divider are config changes (rebuild the
    model); rows below are monkeypatches on a shared model. Every row except
    'tridiag' changes the physics -- they are timing probes, not proposals.
    """
    configs = [
        ('baseline',    dict()),
        ('layers30',    dict(max_nlayers=30)),
        ('layers20',    dict(max_nlayers=20)),
        ('no_SWpen',    dict(option_SWpen=False)),
        ('no_windmaps', dict(option_windmaps=False)),
        ('explicit_heat', dict(method_heateq='explicit')),
    ]
    patch_names = ['tridiag', 'secant4', 'secant1', 'secant0']
    if opts.only:
        keep = set(opts.only)
        configs = [c for c in configs if c[0] in keep]
        patch_names = [p for p in patch_names if p in keep]

    rows = []
    base = None
    for name, over in configs:
        heateq = over.pop('method_heateq', 'implicit')
        model, forcings, nd = build(opts.n_points, opts.start_year, opts.days,
                                    heateq=heateq, **over)
        r = bench(model, forcings, nd, both=not opts.fwd_only, repeat=opts.repeat)
        r['ablation'] = name
        rows.append(r)
        if name == 'baseline':
            base = r
        show([r], ['ablation', 'fwd_s', 'vg_s'], name)

    if patch_names:
        model, forcings, nd = build(opts.n_points, opts.start_year, opts.days)
        for name in patch_names:
            with patched(PATCHES[name]()):
                r = bench(model, forcings, nd, both=not opts.fwd_only,
                          repeat=opts.repeat)
            r['ablation'] = name
            rows.append(r)
            show([r], ['ablation', 'fwd_s', 'vg_s'], name)

    key = 'fwd_s' if opts.fwd_only else 'vg_s'
    if base:
        for r in rows:
            r['saved_s'] = base[key] - r[key]
            r['saved_pct'] = 100.0 * r['saved_s'] / base[key]
    show(rows, ['ablation', 'fwd_s', 'vg_s', 'saved_s', 'saved_pct'],
         f'ablations ({opts.n_points} pts, {opts.days} d)')
    print('\nnote: only "tridiag" leaves the physics answer unchanged; every '
          'other row is a probe for attribution, not a proposal.')
    save(rows, opts.out or 'project/profile_ablate.csv')
    return rows


def mode_census(opts):
    """Static graph size and arithmetic intensity. No GPU time needed."""
    model, forcings, nd = build(opts.n_points, opts.start_year, opts.days)
    vg, fwd = gc.make_fns(model)
    log_wf = jnp.log(jnp.full(model.terrain.N_POINTS, 3.0, dtype=jnp.float64))
    args = (log_wf, model.initial_state, forcings)

    for label, fn in (('forward', fwd), ('value_and_grad', vg)):
        lowered = fn.lower(*args)
        compiled = lowered.compile()
        hlo = compiled.as_text()
        eqns = len(jax.make_jaxpr(fn)(*args).jaxpr.eqns)
        counts = {op: hlo.count(f' {op}(') for op in
                  ('fusion', 'while', 'custom-call', 'dot', 'reduce', 'dynamic-slice')}
        print(f'\n=== {label} ===')
        print(f'  jaxpr equations   {eqns}')
        print(f'  HLO lines         {hlo.count(chr(10))}')
        for k, v in counts.items():
            print(f'  {k:<17} {v}')
        try:
            cost = compiled.cost_analysis()
            cost = cost[0] if isinstance(cost, (list, tuple)) else cost
            flops = float(cost.get('flops', 0))
            byts = float(cost.get('bytes accessed', 0))
            print(f'  flops             {flops:.3e}')
            print(f'  bytes accessed    {byts:.3e}')
            _, warm = timed(fn, *args, repeat=1)
            peak_f, peak_b = DEVICE_PEAK[opts.device]
            print(f'  warm time         {warm:.2f} s')
            print(f'  achieved          {flops / warm:.3e} FLOP/s '
                  f'({100 * flops / warm / peak_f:.3f}% of {opts.device} fp64 peak)')
            print(f'  achieved          {byts / warm:.3e} B/s '
                  f'({100 * byts / warm / peak_b:.3f}% of HBM)')
            if flops / warm / peak_f < 0.01 and byts / warm / peak_b < 0.01:
                print('  VERDICT: launch/latency-bound -- neither FLOPs nor '
                      'bandwidth is the limit, op count is')
        except Exception as e:
            print(f'  cost_analysis unavailable: {e}')
    return []


MODES = dict(ratio=mode_ratio, points=mode_points, time=mode_time,
             ablate=mode_ablate, census=mode_census)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('mode', choices=sorted(MODES))
    p.add_argument('-n', '--n_points', type=int, default=300)
    p.add_argument('-d', '--days', type=int, default=90)
    p.add_argument('-start', '--start_year', type=int, default=2015)
    p.add_argument('-r', '--repeat', type=int, default=1)
    p.add_argument('--days_list', type=int, nargs='+', default=[30, 90, 180, 365])
    p.add_argument('--points_list', type=int, nargs='+', default=[100, 300, 900, 1600])
    p.add_argument('--only', nargs='+', help='restrict ablate to these rows')
    p.add_argument('--fwd_only', action='store_true',
                   help='skip the backward pass (cheap sweeps; scale by the '
                        'ratio from `ratio` mode)')
    p.add_argument('--device', default='v100', choices=sorted(DEVICE_PEAK))
    p.add_argument('--smoke', action='store_true',
                   help='tiny sweep (~10 min) to shake the harness out before '
                        'spending an hour on the real one')
    p.add_argument('-cpu', '--allow_cpu', action='store_true')
    p.add_argument('-o', '--out')
    p.add_argument('--cache', default='/ocean/projects/ees260009p/cwilson4/.jax_cache')
    opts = p.parse_args()

    if opts.smoke:
        opts.n_points, opts.days = 100, 7
        opts.days_list = [7, 14]
        opts.points_list = [50, 200]
        opts.repeat = 1
        print('smoke mode: shrunken sweep, timings are not meaningful\n')

    guard(require_gpu=not opts.allow_cpu)
    enable_compile_cache(opts.cache)
    t0 = time.time()
    MODES[opts.mode](opts)
    print(f'\ntotal {time.time() - t0:.1f} s', flush=True)


if __name__ == '__main__':
    main()
