"""
Window-length bisection at kahiltna/K53's exact blown-up parameter values
(wf=0.7879, kp=1.1442, from jax_optimize.py's investigate_site_blowup,
step 5) -- distinguishes whether that 1e9-scale gradient blowup is a
specific, localized trigger (would already show up in a short window) or a
cumulative/compounding effect of differentiating through the full ~5-year
(~43800-step) scan (would only appear as window length approaches the full
window). investigate_site_blowup's forward-only degenerate-layer scan came
back clean at all 5 yearly snapshots -- ruling out a persistent bad layer,
but not a transient one or pure long-horizon compounding.

Reuses jax_optimize.py's config/model-building helpers (site set, host
paths, spinup) rather than duplicating them, but lives in its own file so
jax_optimize.py doesn't keep growing.

Usage:
    python3 bisect_window_length.py
    PEBSI_BISECT_SITE_INDEX=5 PEBSI_BISECT_WF=0.7879 PEBSI_BISECT_KP=1.1442 python3 bisect_window_length.py
"""
import os
# must be set before importing jax_optimize, since it reads these at
# import time to pick the 2015-2020/16-site config this blowup came from
os.environ.setdefault('PEBSI_NORMAL_RUN_2015_2020', '1')

import time
import numpy as np
import jax
import jax.numpy as jnp

import jax_optimize as jo
from pebsi.main import main as pebsi_main

SITE_INDEX = int(os.environ.get('PEBSI_BISECT_SITE_INDEX', '5'))  # 5 = kahiltna/K53
WF_VAL = float(os.environ.get('PEBSI_BISECT_WF', '0.7879'))
KP_VAL = float(os.environ.get('PEBSI_BISECT_KP', '1.1442'))
GRAD_BLOWUP_THRESHOLD = float(os.environ.get('PEBSI_BISECT_THRESHOLD', '1e10'))
# 1e3 turned out to catch ordinary large-but-legitimate gradients (e.g. 1.1e3
# at hour 5, unaffected by the energybalance.py convergence-freeze fix) --
# the real pathology only showed up as a 10-order-of-magnitude jump between
# 2.5yr (~1e7-1e8) and 5yr (~1e17-1e18), unaffected by that fix. 1e10 sits
# cleanly between the two.


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
    """
    Real, untouched pebsi_main forward+backward over the first n_hours of
    the model's date range, at the fixed (wf_val, kp_val) that caused the
    blowup, reduced to a scalar via _state_reduction (same reduction used
    throughout jax_optimize.py's probes). Returns (wf_grad, kp_grad,
    elapsed_s, nonfinite).
    """
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

    t0 = time.time()
    try:
        grads = jax.grad(loss, argnums=(0, 1))(log_wf, log_kp)
        wf_grad = float(grads[0][0])
        kp_grad = float(grads[1][0])
        nonfinite = not (np.isfinite(wf_grad) and np.isfinite(kp_grad))
    except FloatingPointError:
        # jax_debug_nans raises instead of returning a NaN/Inf value
        wf_grad = kp_grad = float('inf')
        nonfinite = True
    elapsed = time.time() - t0
    # each distinct n_hours is a different array shape -> a fresh XLA
    # compile; without this, compiled executables accumulate across the
    # bisection's ~log2(max_hours) distinct window lengths until the
    # process aborts with "LLVM ERROR: Unable to allocate section memory!"
    # (a compiled-code-memory crash, not an array-size OOM) -- same failure
    # mode jax_optimize.py's own probes guard against the same way.
    jax.clear_caches()
    return wf_grad, kp_grad, elapsed, nonfinite


def bisect_window_length(model, wf_val, kp_val, max_hours, lo=0, hi=None, skip_hi_check=False):
    """
    Binary search for the smallest n_hours where either gradient exceeds
    GRAD_BLOWUP_THRESHOLD -- a small answer means a specific/localized
    trigger (worth hunting for with finer-grained scanning); an answer
    close to max_hours means cumulative compounding over the long scan
    (a structural issue, not a one-line bug).

    lo/hi/skip_hi_check let a prior (e.g. crashed) run's already-known
    bounds be passed back in via PEBSI_BISECT_LO_HOURS/PEBSI_BISECT_HI_HOURS
    instead of re-testing every already-confirmed-blown-up window length.
    """
    def is_large(n_hours):
        wf_grad, kp_grad, elapsed, nonfinite = grad_at_window(model, n_hours, wf_val, kp_val)
        large = nonfinite or abs(wf_grad) > GRAD_BLOWUP_THRESHOLD or abs(kp_grad) > GRAD_BLOWUP_THRESHOLD
        print(f"  {n_hours:>6}h ({n_hours / 8760:.2f}yr): wf_grad={wf_grad:.3e} kp_grad={kp_grad:.3e} "
              f"-> {'BLOWN UP' if large else 'sane'} ({elapsed:.1f}s)", flush=True)
        return large

    hi = max_hours if hi is None else hi
    if not skip_hi_check and not is_large(hi):
        print("  full window is sane at these parameters -- double-check wf_val/kp_val actually "
              "match the real blowup, or this site/regime doesn't reproduce it after all.", flush=True)
        return None

    while hi - lo > 24:  # stop at day resolution -- finer isn't physically meaningful here
        mid = (lo + hi) // 2
        if is_large(mid):
            hi = mid
        else:
            lo = mid

    print(f"\n  bisect result: blows up somewhere between {lo}h ({lo / 8760:.2f}yr) "
          f"and {hi}h ({hi / 8760:.2f}yr)", flush=True)
    return lo, hi


def scan_hourly_states(model, wf_val, kp_val, max_hour, start_hour=1):
    """
    Forward-only (no grad), hour-by-hour, at the fixed (wf_val, kp_val)
    that bisect_window_length found blows up somewhere in [start_hour,
    max_hour]. Runs one real hour of production physics at a time (chaining
    state forward, not a single big scan) so the FULL state -- not just
    whatever fields store_vars keeps in records -- is available to inspect
    every single hour, not just at yearly snapshots the way
    investigate_site_blowup's version did.

    start_hour > 1 fast-forwards there first with ONE real (still cheap,
    forward-only) production run over [0, start_hour) instead of walking
    every hour from 1 -- needed once the bisected transition is thousands
    of hours in, not near the very start.

    Scans the same degenerate-layer signatures used throughout this
    investigation: near-zero-but-positive mass, mass/height inconsistency,
    density or height sitting exactly at an epsilon floor.
    """
    params = model.config.params
    static_args = model.config.static_args
    dynamic_args = model.config.dynamic_args._replace(
        wind_factor=jnp.array([wf_val], dtype=jnp.float32),
        kp=jnp.array([kp_val], dtype=jnp.float32),
    )

    state = model.initial_state
    if start_hour > 1:
        print(f"  Fast-forwarding to hour {start_hour} (forward-only)...", flush=True)
        forcings_prefix = model.pack_forcings(params, model.dates[:start_hour - 1], 0)
        state, _ = pebsi_main(state, forcings_prefix, model.point_attrs, static_args, dynamic_args)

    print(f"  Scanning hourly states {start_hour}..{max_hour}h at wf={wf_val}, kp={kp_val}:", flush=True)
    any_found = False
    for hour in range(start_hour, max_hour + 1):
        forcings = model.pack_forcings(params, model.dates[hour - 1:hour], hour - 1)
        state, _ = pebsi_main(state, forcings, model.point_attrs, static_args, dynamic_args)

        lice = np.asarray(state.lice)[0]
        ldensity = np.asarray(state.ldensity)[0]
        lheight = np.asarray(state.lheight)[0]
        ltype = np.asarray(state.ltype)[0]

        flags_this_hour = []
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
                flags_this_hour.append(f"layer {layer}: lice={m:.4e} ldensity={d:.4e} "
                                        f"lheight={h:.4e} ltype={t} <- {', '.join(flags)}")
        if flags_this_hour:
            any_found = True
            print(f"    hour {hour} ({model.dates[hour - 1]}): " + "; ".join(flags_this_hour), flush=True)
        else:
            print(f"    hour {hour}: nothing flagged", flush=True)

    if not any_found:
        print(f"\n  nothing flagged in any of hours {start_hour}..{max_hour} -- the trigger isn't "
              "a layer sitting at one of the known epsilon floors. Likely something else entirely "
              "(e.g. an extreme value in the energy balance / surface temperature solver, given "
              "wind_factor's first point of influence is turbulent heat flux, not a layer-merge "
              "artifact). Worth inspecting ltemp/flux fields next, same way.",
              flush=True)


if __name__ == '__main__':
    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}", flush=True)
    model = build_single_site_model(SITE_INDEX)
    max_hours = len(model.dates)

    resume_lo = int(os.environ.get('PEBSI_BISECT_LO_HOURS', '0'))
    resume_hi_env = os.environ.get('PEBSI_BISECT_HI_HOURS')
    resume_hi = int(resume_hi_env) if resume_hi_env is not None else None

    print(f"\nBisecting window length at wf={WF_VAL}, kp={KP_VAL}, up to {max_hours}h "
          f"({max_hours / 8760:.2f}yr)"
          + (f", resuming from lo={resume_lo}h hi={resume_hi}h" if resume_hi is not None else "")
          + "...\n", flush=True)
    result = bisect_window_length(
        model, WF_VAL, KP_VAL, max_hours,
        lo=resume_lo, hi=resume_hi, skip_hi_check=(resume_hi is not None),
    )

    if result is not None:
        lo, hi = result
        scan_start = max(1, lo - 20)  # a little buffer before the confirmed-sane point
        print(f"\nScanning hourly states {scan_start}..{hi}h for degenerate layers...\n", flush=True)
        scan_hourly_states(model, WF_VAL, KP_VAL, max_hour=hi, start_hour=scan_start)
