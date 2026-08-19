"""
Inspects PEBSI's secant-method surface-temperature root-finder
(pebsi/physics/energybalance.py: EnergyBalanceDriver.solve_energy_balance)
at kahiltna/K53's exact blown-up parameter values (wf=0.7879, kp=1.1442).

bisect_window_length.py's hourly degenerate-layer scan ruled out a bad
LAYER state at every hour 1-21 -- the trigger must be somewhere else in
the per-timestep physics. wind_factor's first point of contact each hour
is turbulent heat flux (get_turbulent), which feeds directly into the
secant solver's residual -- and that solver has the same "safe forward,
unsafe backward" pattern found earlier in layers.py's epsilon floors:

    denom = jnp.where(jnp.abs(y_curr - y_prev) < 1e-4, 1e-4, y_curr - y_prev)
    t_next = t_curr - y_curr * (t_curr - t_prev) / denom

The floor keeps t_next finite even if y_curr == y_prev exactly, but does
NOT bound the derivative of t_next w.r.t. anything upstream (like
wind_factor, via y_curr/y_prev) as denom approaches that floor -- and this
runs for 8 iterations via jax.lax.scan (an unrolled, directly-differentiated
root-finder, not implicit differentiation at the converged root), so if
that happens at more than one step, the amplification compounds through
the chain rule.

Step 1: per-hour differentiated check (linear scan, cheap at this scale)
to find the EXACT hour -- not just "somewhere in 1-21" -- where the
gradient first blows up.
Step 2: manually unrolls (plain Python, not jax.lax.scan) that one hour's
8-step secant iteration and prints y_curr, y_prev, denom, t_next at each
step, to see directly whether denom is landing near the floor.

Reuses jax_optimize.py's model-building helpers rather than duplicating
them, but lives in its own file alongside bisect_window_length.py.
"""
import os
os.environ.setdefault('PEBSI_NORMAL_RUN_2015_2020', '1')

from types import SimpleNamespace
import numpy as np
import jax
import jax.numpy as jnp

import jax_optimize as jo
from pebsi.main import main as pebsi_main
from pebsi.physics.massbalance import MassBalanceDriver
from pebsi.physics.energybalance import EnergyBalanceDriver
from pebsi.forcing import domain_expansion

SITE_INDEX = int(os.environ.get('PEBSI_BISECT_SITE_INDEX', '5'))  # 5 = kahiltna/K53
WF_VAL = float(os.environ.get('PEBSI_BISECT_WF', '0.7879'))
KP_VAL = float(os.environ.get('PEBSI_BISECT_KP', '1.1442'))
MAX_HOUR = int(os.environ.get('PEBSI_INSPECT_MAX_HOUR', '21'))
GRAD_BLOWUP_THRESHOLD = 1e3

# Set to scan a range far from hour 1 (e.g. the ~31942-31963h transition
# bisect_window_length.py's second, recalibrated bisection found) instead
# of the default linear hour-1 scan -- see scan_secant_denoms.
SCAN_START_HOUR = os.environ.get('PEBSI_INSPECT_START_HOUR')
SCAN_END_HOUR = os.environ.get('PEBSI_INSPECT_END_HOUR')


def _state_reduction(state):
    """Generic scalar reduction over every floating field in a state pytree."""
    total = jnp.float32(0.0)
    for value in state._asdict().values():
        value = jnp.asarray(value)
        if jnp.issubdtype(value.dtype, jnp.floating):
            total = total + jnp.sum(jnp.square(value))
    return total


def build_single_site_model(site_index):
    site_dict = jo.load_reduced_site_dict(jo.REDUCED_SITES_CONFIG)
    site_order = jo.flatten_site_order(site_dict)
    glacier, site = site_order[site_index]
    single_site_dict = {glacier: [site]}
    print(f"Building single-site model: site index {site_index} -> {glacier}/{site}", flush=True)

    config_fp = jo.build_generated_config(
        single_site_dict, jo.host, start_date=jo.DEBUG_START_DATE, end_date=jo.DEBUG_END_DATE,
        temporal_chunk_years=1,
    )
    return jo.init_pebsi(config_fp)


def grad_at_window(model, n_hours, wf_val, kp_val):
    params = model.config.params
    static_args = model.config.static_args
    dynamic_args = model.config.dynamic_args

    forcings = model.pack_forcings(params, model.dates[:n_hours], 0)
    log_wf = jnp.log(jnp.array([wf_val], dtype=jnp.float32))
    log_kp = jnp.log(jnp.array([kp_val], dtype=jnp.float32))

    def loss(log_wf, log_kp):
        wf = jnp.exp(log_wf)
        kp = jnp.exp(log_kp)
        dargs = dynamic_args._replace(wind_factor=wf, kp=kp)
        final_state, _ = pebsi_main(model.initial_state, forcings, model.point_attrs, static_args, dargs)
        return _state_reduction(final_state)

    try:
        grads = jax.grad(loss, argnums=(0, 1))(log_wf, log_kp)
        wf_grad = float(grads[0][0])
        kp_grad = float(grads[1][0])
        nonfinite = not (np.isfinite(wf_grad) and np.isfinite(kp_grad))
    except FloatingPointError:
        wf_grad = kp_grad = float('inf')
        nonfinite = True
    # each distinct n_hours is a different array shape -> a fresh XLA
    # compile; without this, compiled executables accumulate across the
    # scan and the process can abort with "LLVM ERROR: Unable to allocate
    # section memory!" (see bisect_window_length.py's crash/fix)
    jax.clear_caches()
    return wf_grad, kp_grad, nonfinite


def find_first_bad_hour(model, wf_val, kp_val, max_hour):
    print(f"\nScanning hours 1..{max_hour} for the first one whose cumulative "
          f"gradient exceeds {GRAD_BLOWUP_THRESHOLD:.0e}:", flush=True)
    for n in range(1, max_hour + 1):
        wf_grad, kp_grad, nonfinite = grad_at_window(model, n, wf_val, kp_val)
        large = nonfinite or abs(wf_grad) > GRAD_BLOWUP_THRESHOLD or abs(kp_grad) > GRAD_BLOWUP_THRESHOLD
        print(f"  hour {n}: wf_grad={wf_grad:.3e} kp_grad={kp_grad:.3e} "
              f"-> {'BLOWN UP' if large else 'sane'}", flush=True)
        if large:
            return n
    print("  nothing crossed the threshold within max_hour.", flush=True)
    return None


def inspect_secant_at_hour(model, wf_val, kp_val, hour):
    """
    Runs real production physics (forward-only) from model.initial_state
    through hour-1, then manually replicates stages 1-2 of main.step() plus
    solve_energy_balance's secant loop for exactly `hour`, as a plain
    Python loop (not jax.lax.scan) so every iteration's y_curr, y_prev,
    denom, t_next can be printed directly.
    """
    params_ns = model.config.params
    static_args = model.config.static_args
    dynamic_args = model.config.dynamic_args._replace(
        wind_factor=jnp.array([wf_val], dtype=jnp.float32),
        kp=jnp.array([kp_val], dtype=jnp.float32),
    )

    state = model.initial_state
    if hour > 1:
        forcings_prefix = model.pack_forcings(params_ns, model.dates[:hour - 1], 0)
        state, _ = pebsi_main(state, forcings_prefix, model.point_attrs, static_args, dynamic_args)

    # replicate main.step()'s stages 1-2 for exactly this one hour
    full_params = SimpleNamespace(**{**dynamic_args._asdict(), **static_args._asdict()})
    mb = MassBalanceDriver(full_params)
    eb = EnergyBalanceDriver(full_params)

    forcings_hour = model.pack_forcings(params_ns, model.dates[hour - 1:hour], hour - 1)
    # pack_forcings builds the (T, N_POINTS, ...) array jax.lax.scan expects
    # to iterate over -- since main()'s step() only ever runs inside that
    # scan, it always receives an already-sliced, leading-axis-free
    # per-timestep forcings; calling run_new_mass/etc. directly here (no
    # scan) means squeezing that T=1 axis ourselves first.
    forcings_hour = jax.tree.map(lambda x: x[0], forcings_hour)
    forcings_hour = domain_expansion(forcings_hour, model.point_attrs, full_params)

    rainfall, snowfall, state = mb.run_new_mass(state, forcings_hour)
    state = mb.run_daily_routines(state, forcings_hour)

    # ----- manual (non-scan) secant loop, mirroring solve_energy_balance -----
    t_melt = jnp.zeros_like(state.surftemp)
    y_melt, _ = eb.compute_fluxes(t_melt, state, forcings_hour, model.point_attrs)

    t0 = state.surftemp
    t1 = forcings_hour.temp
    y0, _ = eb.compute_fluxes(t0, state, forcings_hour, model.point_attrs)
    y1, _ = eb.compute_fluxes(t1, state, forcings_hour, model.point_attrs)

    print(f"\n  Manually unrolling the secant solve for hour {hour} "
          f"({model.dates[hour - 1]}):", flush=True)
    print(f"    y_melt={float(y_melt[0]):.6g}  t0={float(t0[0]):.6g}  y0={float(y0[0]):.6g}  "
          f"t1={float(t1[0]):.6g}  y1={float(y1[0]):.6g}", flush=True)

    t_prev, t_curr, y_prev, y_curr = t0, t1, y0, y1
    for i in range(8):
        raw_denom = y_curr - y_prev
        denom = jnp.where(jnp.abs(raw_denom) < 1e-4, 1e-4, raw_denom)
        t_next = t_curr - y_curr * (t_curr - t_prev) / denom
        t_next = jnp.clip(t_next, -60.0, 0.0)
        y_next, _ = eb.compute_fluxes(t_next, state, forcings_hour, model.point_attrs)

        floored = bool(jnp.abs(raw_denom[0]) < 1e-4)
        print(f"    step {i}: y_curr-y_prev={float(raw_denom[0]):.6g}  "
              f"denom={float(denom[0]):.6g} (floored={floored})  "
              f"t_next={float(t_next[0]):.6g}  y_next={float(y_next[0]):.6g}", flush=True)

        t_prev, t_curr, y_prev, y_curr = t_curr, t_next, y_curr, y_next

    is_melting = y_melt > 0.0
    surftemp_final = jnp.where(is_melting, t_melt, jnp.clip(t_curr, -60.0, 0.0))
    print(f"    is_melting={bool(is_melting[0])}  surftemp_final={float(surftemp_final[0]):.6g}", flush=True)


def scan_secant_denoms(model, wf_val, kp_val, start_hour, max_hour):
    """
    Fast-forwards ONCE (forward-only) to start_hour, then for each hour in
    [start_hour, max_hour]: manually replicates stages 1-2 plus the 8-step
    secant loop (concise summary only, not the full per-step trace from
    inspect_secant_at_hour) to flag hours where denom gets floored WHILE
    y_curr is NOT small, AND scans the resulting real state for the same
    degenerate-layer signatures bisect_window_length.py's scan_hourly_states
    checks (merged in here so one pass covers both, instead of two separate
    expensive fast-forwards). The energybalance.py convergence-freeze fix
    only masks the update once |y_curr| < 1e-6 (over-iteration past
    convergence) -- a genuinely stalled/oscillating secant step (y_curr not
    small, but y_curr ~= y_prev) hits the exact same floor, completely
    unprotected by that fix, and isn't just a gradient issue there since
    the FORWARD value is affected too. Advances state for real via
    pebsi_main each hour so the trajectory stays correct -- this also
    naturally exercises massbalance.py's evolve_grain_size (currently
    instrumented with a conditional debug print for the safe_base/kap power
    -law floor, a third candidate for the same failure pattern).
    """
    params_ns = model.config.params
    static_args = model.config.static_args
    dynamic_args = model.config.dynamic_args._replace(
        wind_factor=jnp.array([wf_val], dtype=jnp.float32),
        kp=jnp.array([kp_val], dtype=jnp.float32),
    )
    full_params = SimpleNamespace(**{**dynamic_args._asdict(), **static_args._asdict()})
    mb = MassBalanceDriver(full_params)
    eb = EnergyBalanceDriver(full_params)

    state = model.initial_state
    if start_hour > 1:
        print(f"  Fast-forwarding to hour {start_hour} (forward-only)...", flush=True)
        forcings_prefix = model.pack_forcings(params_ns, model.dates[:start_hour - 1], 0)
        state, _ = pebsi_main(state, forcings_prefix, model.point_attrs, static_args, dynamic_args)

    print(f"\n  Scanning secant denom behavior for hours {start_hour}..{max_hour}:", flush=True)
    for hour in range(start_hour, max_hour + 1):
        forcings_hour_raw = model.pack_forcings(params_ns, model.dates[hour - 1:hour], hour - 1)
        forcings_hour = jax.tree.map(lambda x: x[0], forcings_hour_raw)
        forcings_hour = domain_expansion(forcings_hour, model.point_attrs, full_params)

        diag_state = state  # snapshot for the manual diagnostic only
        _, _, diag_state = mb.run_new_mass(diag_state, forcings_hour)
        diag_state = mb.run_daily_routines(diag_state, forcings_hour)

        t0 = diag_state.surftemp
        t1 = forcings_hour.temp
        y0, _ = eb.compute_fluxes(t0, diag_state, forcings_hour, model.point_attrs)
        y1, _ = eb.compute_fluxes(t1, diag_state, forcings_hour, model.point_attrs)

        t_prev, t_curr, y_prev, y_curr = t0, t1, y0, y1
        flagged_steps = []
        for i in range(8):
            raw_denom = y_curr - y_prev
            floored = bool(jnp.abs(raw_denom[0]) < 1e-4)
            not_converged = bool(jnp.abs(y_curr[0]) >= 1e-6)
            if floored and not_converged:
                flagged_steps.append((i, float(y_curr[0]), float(raw_denom[0])))
            denom = jnp.where(jnp.abs(raw_denom) < 1e-4, 1e-4, raw_denom)
            t_next = jnp.clip(t_curr - y_curr * (t_curr - t_prev) / denom, -60.0, 0.0)
            y_next, _ = eb.compute_fluxes(t_next, diag_state, forcings_hour, model.point_attrs)
            t_prev, t_curr, y_prev, y_curr = t_curr, t_next, y_curr, y_next

        # advance the REAL state for real continuity into the next hour
        # (also naturally exercises evolve_grain_size's debug print)
        state, _ = pebsi_main(state, forcings_hour_raw, model.point_attrs, static_args, dynamic_args)

        lice = np.asarray(state.lice)[0]
        ldensity = np.asarray(state.ldensity)[0]
        lheight = np.asarray(state.lheight)[0]
        ltype = np.asarray(state.ltype)[0]
        layer_flags = []
        for layer in range(lice.shape[0]):
            m, d, h, t = lice[layer], ldensity[layer], lheight[layer], ltype[layer]
            flags = []
            if 0 < m < 1e-2:
                flags.append('near-zero-but-positive mass')
            if (m == 0) != (h == 0):
                flags.append('mass/height inconsistency')
            if abs(d - 1e-3) < 1e-6:
                flags.append('density at epsilon floor')
            if abs(h - 1e-6) < 1e-9:
                flags.append('height at epsilon floor')
            if flags:
                layer_flags.append(f"layer {layer}: lice={m:.4e} ldensity={d:.4e} "
                                    f"lheight={h:.4e} ltype={t} <- {', '.join(flags)}")

        if flagged_steps or layer_flags:
            details = "; ".join(
                f"step{i}(y_curr={yc:.4g}, y_curr-y_prev={dn:.4g})" for i, yc, dn in flagged_steps
            )
            print(f"    hour {hour} ({model.dates[hour - 1]}): "
                  f"secant=[{details or 'clean'}] layers=[{'; '.join(layer_flags) or 'clean'}]",
                  flush=True)
        else:
            print(f"    hour {hour}: clean (secant + layers)", flush=True)


if __name__ == '__main__':
    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}", flush=True)
    model = build_single_site_model(SITE_INDEX)

    if SCAN_START_HOUR is not None and SCAN_END_HOUR is not None:
        scan_secant_denoms(model, WF_VAL, KP_VAL, int(SCAN_START_HOUR), int(SCAN_END_HOUR))
    else:
        bad_hour = find_first_bad_hour(model, WF_VAL, KP_VAL, MAX_HOUR)
        if bad_hour is not None:
            inspect_secant_at_hour(model, WF_VAL, KP_VAL, bad_hour)
