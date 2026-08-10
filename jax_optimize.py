"""
JAX optimization of per-site wind_factor and precipitation factor (kp)
using PEBSI's differentiable core.

The full PEBSI forward model (pebsi/main.py) is @jax.jit compiled and runs
via jax.lax.scan, so wind_factor and kp flow through as traced JAX arrays.
Both parameters are fit from a single forward simulation per step, but each
is scored against a different mass balance signal:
  - wind_factor is fit against summer (ablation season) MB, since wind speed
    primarily controls melt through turbulent heat fluxes.
  - kp is fit against winter (accumulation season) MB, since it directly
    scales snowfall.

Because both losses come from the same forward pass, any cross-season
coupling (e.g. kp affecting summer melt through albedo) still contributes
gradient signal to both parameters.

Steps:
  1. Load MassBalance observations for all sites, split into summer and
     winter periods.
  2. Initialize PEBSI (sets up state, forcings, etc.) — this is the numpy side.
  3. Define a loss function that swaps wind_factor and kp into dynamic_args,
     calls pebsi.main.main(), sums MB over observation periods for each
     season, and returns the combined RMSE loss.
  4. jax.grad + optax to optimize wind_factor and kp per site simultaneously.
"""
import os 

# Toggle back to True to return to the full sites.pkl set (34 sites) once
# the long-run NaN gradient issue is resolved.
USE_REDUCED_SITE_SET = True
REDUCED_SITES_CONFIG = os.path.join(os.path.dirname(__file__), 'configs_opt_bridges.yaml')

# Short windows haven't reproduced the NaN; climate forcing covers 1980-2025,
# so stretch to a 15-year window to see if it's a long-horizon accumulation issue.
DEBUG_START_DATE = '2023-04-01'
DEBUG_END_DATE = '2025-04-01'

# NaN-localization mode (PEBSI_NAN_BISECT=1): instead of optimizing,
# binary-search for the smallest number of chunks whose gradient goes
# non-finite. The forward pass is clean, so this pins the bad op to a
# specific chunk's date window. Uses ~monthly chunks (730 h) for finer
# localization than the default 8760.
NAN_BISECT = os.environ.get('PEBSI_NAN_BISECT', '0') == '1'
BISECT_CHUNK_SIZE = 730

# Stage-localization mode (PEBSI_STAGE_BISECT=1): all 6 per-timestep physics
# stages run every timestep regardless of date, so knowing *when* the NaN
# appears doesn't tell you *which* stage is responsible. This instead takes
# a handful of forward-only (no grad -> cheap) state snapshots spread across
# the debug window, then from each one runs a short differentiated probe
# bisecting over which stage's inclusion breaks the gradient.
STAGE_BISECT = os.environ.get('PEBSI_STAGE_BISECT', '0') == '1'
N_SNAPSHOTS = 8
PROBE_HOURS = 72

# Sub-stage localization (PEBSI_VERTICAL_SUBSTAGE_BISECT=1): PEBSI_STAGE_BISECT
# already pinned it to stage 4 (run_vertical_processes), which itself bundles
# 6 sub-calls (heating_melting, percolation/route_particles, refreezing,
# phase_changes, check_layer_sizes, resolve_temperature_profile). This
# bisects one level deeper, over those, using the same confirmed-bad
# chunk-0/initial_state window.
VERTICAL_SUBSTAGE_BISECT = os.environ.get('PEBSI_VERTICAL_SUBSTAGE_BISECT', '0') == '1'

# One level deeper still (PEBSI_LAYER_PHASE_BISECT=1): PEBSI_VERTICAL_SUBSTAGE_BISECT
# already pinned it to check_layer_sizes, which bundles dead-layer zeroing,
# the merge scan (merge_existing_layers), and the split scan (split_layer).
# Bisects over those 3 phases, same confirmed-bad window.
LAYER_PHASE_BISECT = os.environ.get('PEBSI_LAYER_PHASE_BISECT', '0') == '1'

# One level deeper still (PEBSI_MERGE_PHASE_BISECT=1): two rounds of fixes
# targeting the ">0 doesn't catch tiny-but-positive" pattern in
# merge_existing_layers/add_bottom_layer/update_layer_props/split_layer did
# NOT resolve phase 2 (merge_scan) -- so bisect merge_existing_layers' own
# internal structure directly instead of guessing at more candidates.
MERGE_PHASE_BISECT = os.environ.get('PEBSI_MERGE_PHASE_BISECT', '0') == '1'

# One level deeper still (PEBSI_MERGE_VAR_BISECT=1): PEBSI_MERGE_PHASE_BISECT
# confirmed it's phase 2 (shift float-typed vars), not phase 3 (ltype/lage
# int-cast, ruled out) -- bisect over which of the 12 shifted variables
# (MERGE_VAR_NAMES) is actually responsible.
MERGE_VAR_BISECT = os.environ.get('PEBSI_MERGE_VAR_BISECT', '0') == '1'

# PEBSI_MERGE_VAR_BISECT's result (culprit: 'lice') showed a DIFFERENT
# failure mode ("in the OUTPUT", i.e. forward-pass NaN) than every other
# level of this bisection chain ("in the BACKWARD pass") -- a sign that its
# prefix-truncation probe compounds its own torn-state artifact rather than
# finding the real bug. This mode (PEBSI_MERGE_SKIPVAR_TEST=1) re-tests each
# variable individually while keeping everything else fully faithful to
# production, via find_nan_merge_skipvar.
MERGE_SKIPVAR_TEST = os.environ.get('PEBSI_MERGE_SKIPVAR_TEST', '0') == '1'

# PEBSI_MERGE_SKIPVAR_TEST confirmed: omitting any single one of the 12
# shifted variables is NOT enough to fix it. Bisect over how many
# *simultaneously* shifted variables (first N, in order) are needed to
# trigger the non-finite gradient, with full downstream fidelity.
MERGE_NVARS_BISECT = os.environ.get('PEBSI_MERGE_NVARS_BISECT', '0') == '1'


import socket
if 'trace' in socket.gethostname():
    host = 'trace'
elif 'bridges' in socket.gethostname():
    host = 'bridges'

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# netCDF4 must load before JAX to avoid numpy ABI conflict
import simulation as sim

import time
import pickle
import yaml
import jax
jax.config.update("jax_traceback_filtering", "off")
# Raises immediately at the exact primitive (forward or backward) that first
# produces a NaN/Inf, instead of letting it silently propagate through the
# scan. Set PEBSI_DEBUG_NANS=0 to disable once the source is found -- it adds
# real overhead (blocks XLA fusion, forces per-op synchronization).
if os.environ.get('PEBSI_DEBUG_NANS', '1') != '0':
    jax.config.update("jax_debug_nans", True)
    jax.config.update("jax_debug_infs", True)
import jax.numpy as jnp
import optax
import numpy as np
import pandas as pd

from project.data_handling import MassBalance, translate_rgi
from pebsi.main import (
    main as pebsi_main, main_stage_probe, STAGE_NAMES,
    main_vertical_substage_probe, VERTICAL_SUBSTAGE_NAMES,
    main_layer_phase_probe, LAYER_PHASE_NAMES,
    main_merge_phase_probe, MERGE_PHASE_NAMES,
    main_merge_var_probe, MERGE_VAR_NAMES,
    main_merge_skipvar_probe,
    main_merge_nvars_probe,
)

# per-host filepaths, carried over from configs_opt_{host}.yaml
HOST_PATHS = {
    'trace': dict(
        climate_fp='/trace/group/rounce/cvwilson/climate_data/',
        rgi_fp='/trace/group/rounce/shared/RGI/rgi60/00_rgi60_attribs/',
        output_fp='/trace/group/rounce/cvwilson/Output/recalibrate/',
        cop30_vrt_path='/trace/group/rounce/cvwilson/dems/RGI1_DEM/rgi_dem.vrt',
        shading_fp='/trace/group/rounce/cvwilson/shading/',
    ),
    'bridges': dict(
        climate_fp='/ocean/projects/ees260009p/cwilson4/climate_data/',
        rgi_fp='/ocean/projects/ees260009p/cwilson4/RGI/rgi60/00_rgi60_attribs/',
        output_fp='/ocean/projects/ees260009p/cwilson4/Output/recalibrate/',
        cop30_vrt_path='/ocean/projects/ees260009p/cwilson4/data/dems/COP30/COP30_reg01.vrt',
        shading_fp='/ocean/projects/ees260009p/cwilson4/data/shading/',
        ice_albedo_fn='/ocean/projects/ees260009p/cwilson4/data/ice_albedo/{gid}_albedo.tif'
    ),
}


def load_site_dict():
    """
    Loads every (glacier, site) pair from project/sites.pkl — keyed by RGI6
    glacier id, mapping to a list of site names — and converts it into the
    {glacier_name: [site, ...]} form used throughout this script.
    """
    sites_fp = os.path.join(os.path.dirname(__file__), 'project', 'sites.pkl')
    with open(sites_fp, 'rb') as f:
        sites_by_rgi6 = pickle.load(f)

    rgi6_to_name = {ids['6']: name for name, ids in translate_rgi.items()}

    site_dict = {}
    for rgi6, sites in sites_by_rgi6.items():
        if not sites:
            continue
        site_dict[rgi6_to_name[rgi6]] = list(sites)
    return site_dict


def load_reduced_site_dict(reference_config_fn):
    """
    Restricts to the (glacier, site) pairs listed in an existing config yaml
    (e.g. configs_opt_bridges.yaml) instead of the full sites.pkl set, for
    debugging on a smaller problem. load_site_dict()/sites.pkl is untouched
    so switching back to the full set is just flipping USE_REDUCED_SITE_SET.
    """
    with open(reference_config_fn) as f:
        ref = yaml.safe_load(f)
    rgi6_to_name = {ids['6']: name for name, ids in translate_rgi.items()}

    reduced = {}
    for rgi6, site in zip(ref['rgi_ids'], ref['sites']):
        glacier = rgi6_to_name[rgi6]
        reduced.setdefault(glacier, []).append(site)
    return reduced


def flatten_site_order(site_dict):
    """
    Flat (glacier, site) list in the same order as the model's per-point
    dynamic_args arrays (wind_factor, kp, ...) — this matches the order of
    the `sites` list in the model config.
    """
    return [(glacier, site) for glacier in site_dict for site in site_dict[glacier]]


def build_generated_config(site_dict, host, start_date='2024-04-01', end_date='2025-04-01',
                           temporal_chunk_years=1):
    """
    Builds a PEBSI config yaml from site_dict (see load_site_dict) so the
    model's `sites`/`rgi_ids` always match the sites used for observations.
    Ice albedo now comes from the preprocessed per-glacier .tif rather than
    a per-site scalar, so albedo_ice is left out of the config entirely.
    """
    site_order = flatten_site_order(site_dict)
    rgi6_by_name = {name: ids['6'] for name, ids in translate_rgi.items()}

    config = dict(
        rgi_ids=[rgi6_by_name[glacier] for glacier, site in site_order],
        sites=[site for glacier, site in site_order],
        n_points=len(site_order),
        temporal_chunk_years=temporal_chunk_years,
        bias_vars=['temp'],
        start_date=start_date,
        end_date=end_date,
        dust_factor=20,
        ksp_BC=1,
        ksp_OC=1,

        kp=[1.0] * len(site_order),
        wind_factor=[1.0] * len(site_order),
        option_ice_albedo_tif=True,
        option_accel_grains=False,
        option_flat_plates=True,
        constant_freshgrainsize=54.5,
        debug=False,
        progress_bar=False,
        store_data=False,
        store_vars=['minimal'],
        **HOST_PATHS[host],
    )

    config_fn = os.path.join(os.path.dirname(__file__), f'configs_{host}_calibration.yaml')
    with open(config_fn, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    return config_fn


# ---------------------------------------------------------------------------
# 1. Load observations
# ---------------------------------------------------------------------------

def load_all_observations(site_dict):
    """
    Loads MassBalance observations for all sites and splits each site's
    periods into summer (ablation season) and winter (accumulation season)
    using the sign of (end_doy - start_doy): summer periods fall within a
    single calendar year, winter periods span the new year.

    Returns a dict with keys 'summer' and 'winter', each mapping to:
        site_labels    : list of (glacier, site) tuples, length S
        meas_padded    : np.ndarray (S, N_max) measured MB [m w.e.], NaN-padded
        mask           : np.ndarray (S, N_max) bool
        period_starts  : list of np.ndarray, one per site
        period_ends    : list of np.ndarray, one per site
    """
    raw = {
        'summer': {'labels': [], 'meas': [], 'starts': [], 'ends': []},
        'winter': {'labels': [], 'meas': [], 'starts': [], 'ends': []},
    }

    failed = []
    for glacier in site_dict:
        for site in site_dict[glacier]:
            try:
                obs = MassBalance(glacier, site, use='benchmark', min_n_winter=1)
                start_doy = pd.to_datetime(obs.period_starts).day_of_year
                end_doy = pd.to_datetime(obs.period_ends).day_of_year
                season_idx = {
                    'summer': end_doy - start_doy > 0,
                    'winter': end_doy - start_doy <= 0,
                }
                for season, idx in season_idx.items():
                    data = obs.data[idx]
                    if len(data) == 0:
                        continue
                    raw[season]['labels'].append((glacier, site))
                    raw[season]['meas'].append(data.astype(np.float32))
                    raw[season]['starts'].append(obs.period_starts[idx])
                    raw[season]['ends'].append(obs.period_ends[idx])
            except Exception as e:
                failed.append((glacier, site, str(e)))

    if failed:
        print("Warning: failed to load observations for:")
        for glacier, site, err in failed:
            print(f"  {glacier}/{site}: {err}")

    result = {}
    for season, d in raw.items():
        site_labels = d['labels']
        meas_list = d['meas']
        N_max = max(len(m) for m in meas_list)
        N_sites = len(site_labels)

        meas_padded = np.full((N_sites, N_max), np.nan, dtype=np.float32)
        mask = np.zeros((N_sites, N_max), dtype=bool)
        for i, m in enumerate(meas_list):
            n = len(m)
            meas_padded[i, :n] = m
            mask[i, :n] = True

        result[season] = (site_labels, meas_padded, mask, d['starts'], d['ends'])

    return result


# ---------------------------------------------------------------------------
# 2. Build per-period timestep index arrays from the model time axis
# ---------------------------------------------------------------------------

def build_period_indices(model_dates, period_starts_list, period_ends_list):
    """
    For each site and each observation period, find the slice of timestep
    indices in model_dates that falls within [period_start, period_end].

    Returns period_slices: list (S,) of list (N_periods,) of (start_idx, end_idx)
    so that records.MB[start_idx:end_idx, site_idx].sum() gives modeled MB
    for that period.

    We store as padded integer arrays (S, N_max, 2) for use inside JAX.
    """
    model_dates = pd.to_datetime(model_dates)
    S = len(period_starts_list)
    N_max = max(len(s) for s in period_starts_list)

    # (S, N_max, 2) — columns are [start_idx, end_idx]; -1 means padding
    period_idx = np.full((S, N_max, 2), -1, dtype=np.int32)

    for i, (starts, ends) in enumerate(zip(period_starts_list, period_ends_list)):
        for j, (start, end) in enumerate(zip(starts, ends)):
            start = pd.to_datetime(start)
            end = pd.to_datetime(end)
            idx = np.where((model_dates >= start) & (model_dates <= end))[0]
            if len(idx) > 0:
                period_idx[i, j, 0] = idx[0]
                period_idx[i, j, 1] = idx[-1] + 1  # exclusive end

    return period_idx


# ---------------------------------------------------------------------------
# 3. Initialize PEBSI (numpy side — done once before optimization)
# ---------------------------------------------------------------------------

def init_pebsi(config_fn):
    """
    Initializes PEBSI and returns everything needed to call pebsi_main()
    inside the loss function. Forcings are loaded per-chunk inside the
    loss function via model.pack_forcings().
    """
    args = sim.get_args()
    args.config_fn = config_fn
    model = sim.PEBSI(args)

    # only compute MB fields during optimization to keep memory bounded
    model.config.static_args = model.config.static_args._replace(store_vars=('minimal',), differentiable=True)
    model.config.params.store_vars = ('minimal',)
    model.initialize()

    # run spinup once here so every optimization step starts from a
    # spun-up state without re-running it each forward pass
    print("Running spinup...", flush=True)
    spun_up_state = model.spinup(model.initial_state)
    print("Spinup complete.", flush=True)
    model.initial_state = spun_up_state

    return model


# ---------------------------------------------------------------------------
# 4. Loss function
# ---------------------------------------------------------------------------

def make_loss_fn(model, site_order, summer, winter):
    """
    Returns a loss function that takes (log_wind_factor, log_kp) — each
    (N_POINTS,), in model site order — and returns
    (total_loss, (summer_rmse, winter_rmse)).

    summer / winter : (site_labels, period_idx, meas, mask) tuples, where
        period_idx : (S, N_max, 2) int32 — timestep [start, end) per period
        meas       : (S, N_max) float32 — observed MB
        mask       : (S, N_max) bool
    site_order : flat (glacier, site) list matching the model's per-point
        arrays (see flatten_site_order) — used to map each season's site
        rows onto the right column of the model's per-timestep MB output.
    """
    static_args = model.config.static_args
    dynamic_args = model.config.dynamic_args
    params = model.config.params
    chunk_size = params.temporal_chunk_hours
    # truncate dates to a multiple of chunk_size so chunks are always full
    total_steps = (len(model.dates) // chunk_size) * chunk_size

    def _prep(season_data):
        site_labels, period_idx, meas, mask = season_data
        meas_jax = jnp.array(np.nan_to_num(meas, nan=0.0))
        period_idx_jax = jnp.array(period_idx)
        # a period only contributes to the loss if it actually falls inside
        # the simulated date range — otherwise period_sums is left at its
        # zero init and would be wrongly scored against a real measurement
        mask_jax = jnp.array(mask) & (period_idx_jax[:, :, 0] >= 0)
        site_col_idx = jnp.array([site_order.index(label) for label in site_labels])
        return period_idx_jax, meas_jax, mask_jax, site_col_idx

    summer_period_idx, summer_meas, summer_mask, summer_col_idx = _prep(summer)
    winter_period_idx, winter_meas, winter_mask, winter_col_idx = _prep(winter)
    S_summer, N_max_summer = summer_meas.shape
    S_winter, N_max_winter = winter_meas.shape

    # Stack all chunk forcings into a single JAX array so lax.scan can
    # iterate over them — this is the key to bounded memory during autodiff.
    # Each leaf of the forcing namedtuple gets an extra leading chunk dimension.
    chunk_list = [
        model.pack_forcings(params, model.dates[start:start + chunk_size], start)
        for start in range(0, total_steps, chunk_size)
    ]
    n_chunks = len(chunk_list)
    stacked_forcings = jax.tree.map(lambda *xs: jnp.stack(xs), *chunk_list)

    # static_args and point_attrs captured in closure — jax.checkpoint only
    # sees JAX-traceable arguments
    def _run_chunk(state, chunk_forcings, new_dynamic_args):
        return pebsi_main(state, chunk_forcings, model.point_attrs, static_args, new_dynamic_args)

    # chunk_size is static, so period index offsets are known at trace time
    chunk_offsets = jnp.arange(n_chunks) * chunk_size

    def _accumulate(period_sums, chunk_mb, t_offset, period_idx_jax, site_col_idx, N_max):
        """Adds this chunk's contribution to every (site, period) sum."""
        def accumulate_site(period_sums, row_idx):
            mb_site = chunk_mb[:, site_col_idx[row_idx]]  # (T_chunk,)

            def add_period(carry, j):
                p_start = period_idx_jax[row_idx, j, 0]
                p_end = period_idx_jax[row_idx, j, 1]
                t = jnp.arange(chunk_size) + t_offset
                contrib = jnp.where((t >= p_start) & (t < p_end), mb_site, 0.0).sum()
                return carry.at[j].add(contrib), None

            row, _ = jax.lax.scan(add_period, period_sums[row_idx], jnp.arange(N_max))
            return period_sums.at[row_idx].set(row), None

        period_sums, _ = jax.lax.scan(accumulate_site, period_sums, jnp.arange(period_sums.shape[0]))
        return period_sums

    def _rmse(period_sums, meas_jax, mask_jax):
        safe_meas = jnp.where(mask_jax, meas_jax, 0.0)
        residuals = period_sums - safe_meas
        sq = jnp.where(mask_jax, jnp.square(residuals), 0.0)
        counts = jnp.sum(mask_jax, axis=1).clip(min=1)
        per_site_rmse = jnp.sqrt(jnp.sum(sq, axis=1) / counts + 1e-8)
        return jnp.mean(per_site_rmse)

    def _make_loss(n_chunks_used):
        # truncate to the first n_chunks_used chunks (static slice -> one
        # recompile per distinct n_chunks_used; used by the NaN bisect)
        used_forcings = jax.tree.map(lambda x: x[:n_chunks_used], stacked_forcings)
        used_offsets = chunk_offsets[:n_chunks_used]

        def loss_fn(log_wind_factor, log_kp):
            wind_factor = jnp.exp(log_wind_factor)
            kp = jnp.exp(log_kp)
            new_dynamic_args = dynamic_args._replace(wind_factor=wind_factor, kp=kp)

            def scan_chunk(carry, xs):
                state, summer_sums, winter_sums = carry
                chunk_forcings, t_offset = xs

                state, records = jax.checkpoint(_run_chunk)(state, chunk_forcings, new_dynamic_args)
                chunk_mb = records.accumulation + records.refreeze - records.melt  # (T_chunk, N_POINTS)

                summer_sums = _accumulate(summer_sums, chunk_mb, t_offset,
                                           summer_period_idx, summer_col_idx, N_max_summer)
                winter_sums = _accumulate(winter_sums, chunk_mb, t_offset,
                                           winter_period_idx, winter_col_idx, N_max_winter)
                return (state, summer_sums, winter_sums), None

            init_carry = (
                model.initial_state,
                jnp.zeros((S_summer, N_max_summer)),
                jnp.zeros((S_winter, N_max_winter)),
            )
            (_, summer_sums, winter_sums), _ = jax.lax.scan(
                scan_chunk, init_carry, (used_forcings, used_offsets)
            )

            summer_rmse = _rmse(summer_sums, summer_meas, summer_mask)
            winter_rmse = _rmse(winter_sums, winter_meas, winter_mask)
            total_loss = summer_rmse + winter_rmse
            return total_loss, (summer_rmse, winter_rmse)

        return loss_fn

    loss_fn = _make_loss(n_chunks)
    loss_fn.make_truncated = _make_loss
    loss_fn.n_chunks = n_chunks
    return loss_fn


def bisect_nan_chunk(loss_fn, wf_init, kp_init, model_dates, chunk_size):
    """
    Finds the smallest number of chunks k such that grad(loss over the first
    k chunks) is non-finite, via binary search. The forward pass is clean
    (verified), so the chunk where the gradient first goes bad is where the
    offending op lives. Prints the date window of that chunk.

    Each probe recompiles (different static scan length), so total cost is
    ~log2(n_chunks) compiles + runs of at most the full window.
    """
    log_wf = jnp.log(jnp.array(wf_init, dtype=jnp.float32))
    log_kp = jnp.log(jnp.array(kp_init, dtype=jnp.float32))
    n_chunks = loss_fn.n_chunks

    def grad_is_finite(k):
        f = loss_fn.make_truncated(k)
        t0 = time.time()
        try:
            (_, _), grads = jax.value_and_grad(f, argnums=(0, 1), has_aux=True)(log_wf, log_kp)
            ok = all(np.isfinite(np.asarray(g)).all() for g in grads)
        except FloatingPointError:
            # jax_debug_nans raises instead of returning NaN
            ok = False
        print(f"  bisect: first {k}/{n_chunks} chunks -> "
              f"{'finite' if ok else 'NON-FINITE'} ({time.time()-t0:.1f}s)", flush=True)
        return ok

    if grad_is_finite(n_chunks):
        print("  bisect: full-window gradient is finite; nothing to localize.", flush=True)
        return None

    lo, hi = 0, n_chunks  # grad over first `lo` chunks finite; first `hi` non-finite
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if grad_is_finite(mid):
            lo = mid
        else:
            hi = mid

    bad = hi - 1  # 0-based index of the chunk that introduces the NaN
    c_start, c_end = bad * chunk_size, min((bad + 1) * chunk_size, len(model_dates)) - 1
    print(f"  bisect result: NaN gradient first appears in chunk {bad} "
          f"({model_dates[c_start]} to {model_dates[c_end]})", flush=True)
    return bad


def generate_snapshots(model, n_snapshots=8):
    """
    Forward-only (no grad) pass through the full debug window at the
    model's current dynamic_args, saving state every ~1/n_snapshots of the
    way through. No backprop means no checkpoint recomputation cost, so
    this is cheap relative to any differentiated run over the same span --
    it's just how we get realistic, physically-evolved starting states to
    probe from, without paying to differentiate through years of history.

    Returns a list of (date, state, next_chunk_start_idx) tuples.
    """
    params = model.config.params
    static_args = model.config.static_args
    dynamic_args = model.config.dynamic_args
    chunk_size = params.temporal_chunk_hours
    total_steps = (len(model.dates) // chunk_size) * chunk_size
    n_chunks = total_steps // chunk_size
    save_every = max(1, n_chunks // n_snapshots)

    state = model.initial_state
    snapshots = []
    for c in range(n_chunks):
        start = c * chunk_size
        chunk_forcings = model.pack_forcings(params, model.dates[start:start + chunk_size], start)
        state, _ = pebsi_main(state, chunk_forcings, model.point_attrs, static_args, dynamic_args)
        if c % save_every == 0:
            snapshots.append((model.dates[start + chunk_size - 1], state, start + chunk_size))
    return snapshots


def _state_reduction(state):
    """Generic scalar reduction over every floating field in a state pytree."""
    total = jnp.float32(0.0)
    for value in state._asdict().values():
        value = jnp.asarray(value)
        if jnp.issubdtype(value.dtype, jnp.floating):
            total = total + jnp.sum(jnp.square(value))
    return total


def bisect_nan_stage(model, snapshot_state, probe_start_idx, wf_init, kp_init, probe_hours=72):
    """
    From a fixed (non-differentiated) state snapshot, runs a short forward
    window through main_stage_probe, binary-searching over which of the 6
    per-timestep physics stages (STAGE_NAMES) first introduces a
    non-finite gradient w.r.t. wind_factor/kp -- as opposed to bisect_nan_chunk,
    which finds *when* but can't distinguish *which stage* since all of them
    run every timestep regardless of date.

    Returns the smallest bad stage number (1-6), or None if all stages are
    finite over this particular probe window.
    """
    params = model.config.params
    static_args = model.config.static_args
    dynamic_args = model.config.dynamic_args

    probe_hours = min(probe_hours, len(model.dates) - probe_start_idx)
    probe_forcings = model.pack_forcings(
        params, model.dates[probe_start_idx:probe_start_idx + probe_hours], probe_start_idx
    )
    # detach from any prior graph -- only wind_factor/kp's effect on THIS
    # short window should be differentiated, not the history that produced
    # the snapshot (that history is already known-finite forward, and we
    # don't want to pay to backprop through it again here)
    frozen_state = jax.lax.stop_gradient(snapshot_state)

    log_wf = jnp.log(jnp.array(wf_init, dtype=jnp.float32))
    log_kp = jnp.log(jnp.array(kp_init, dtype=jnp.float32))

    def stage_finite(stage):
        def probe_loss(log_wf, log_kp):
            wf = jnp.exp(log_wf)
            kp = jnp.exp(log_kp)
            dargs = dynamic_args._replace(wind_factor=wf, kp=kp)
            final_state = main_stage_probe(
                frozen_state, probe_forcings, model.point_attrs, static_args, dargs, stage
            )
            return _state_reduction(final_state)

        t0 = time.time()
        try:
            grads = jax.grad(probe_loss, argnums=(0, 1))(log_wf, log_kp)
            ok = all(np.isfinite(np.asarray(g)).all() for g in grads)
        except FloatingPointError:
            ok = False
        print(f"    stage {stage} ({STAGE_NAMES[stage]}): "
              f"{'finite' if ok else 'NON-FINITE'} ({time.time()-t0:.1f}s)", flush=True)
        return ok

    if stage_finite(6):
        return None

    lo, hi = 0, 6  # stages 1..lo finite; stages 1..hi non-finite
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if stage_finite(mid):
            lo = mid
        else:
            hi = mid
    return hi


def bisect_nan_vertical_substage(model, snapshot_state, probe_start_idx, wf_init, kp_init, probe_hours=72):
    """
    Same idea as bisect_nan_stage, one level deeper: stage 4
    (run_vertical_processes) bundles 6 sub-calls (VERTICAL_SUBSTAGE_NAMES) --
    this bisects over which of those, specifically, first introduces a
    non-finite gradient, using main_vertical_substage_probe.
    """
    params = model.config.params
    static_args = model.config.static_args
    dynamic_args = model.config.dynamic_args

    probe_hours = min(probe_hours, len(model.dates) - probe_start_idx)
    probe_forcings = model.pack_forcings(
        params, model.dates[probe_start_idx:probe_start_idx + probe_hours], probe_start_idx
    )
    frozen_state = jax.lax.stop_gradient(snapshot_state)

    log_wf = jnp.log(jnp.array(wf_init, dtype=jnp.float32))
    log_kp = jnp.log(jnp.array(kp_init, dtype=jnp.float32))

    def substage_finite(substage):
        def probe_loss(log_wf, log_kp):
            wf = jnp.exp(log_wf)
            kp = jnp.exp(log_kp)
            dargs = dynamic_args._replace(wind_factor=wf, kp=kp)
            final_state = main_vertical_substage_probe(
                frozen_state, probe_forcings, model.point_attrs, static_args, dargs, substage
            )
            return _state_reduction(final_state)

        t0 = time.time()
        try:
            grads = jax.grad(probe_loss, argnums=(0, 1))(log_wf, log_kp)
            ok = all(np.isfinite(np.asarray(g)).all() for g in grads)
        except FloatingPointError:
            ok = False
        print(f"    substage {substage} ({VERTICAL_SUBSTAGE_NAMES[substage]}): "
              f"{'finite' if ok else 'NON-FINITE'} ({time.time()-t0:.1f}s)", flush=True)
        return ok

    if substage_finite(6):
        return None

    lo, hi = 0, 6  # substages 1..lo finite; 1..hi non-finite
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if substage_finite(mid):
            lo = mid
        else:
            hi = mid
    return hi


def bisect_nan_layer_phase(model, snapshot_state, probe_start_idx, wf_init, kp_init, probe_hours=72):
    """
    One level deeper still: substage 5 (check_layer_sizes) bundles 3 phases
    (LAYER_PHASE_NAMES) -- dead-layer zeroing, the merge scan
    (merge_existing_layers), and the split scan (split_layer). Bisects over
    which of those first introduces a non-finite gradient, using
    main_layer_phase_probe.
    """
    params = model.config.params
    static_args = model.config.static_args
    dynamic_args = model.config.dynamic_args

    probe_hours = min(probe_hours, len(model.dates) - probe_start_idx)
    probe_forcings = model.pack_forcings(
        params, model.dates[probe_start_idx:probe_start_idx + probe_hours], probe_start_idx
    )
    frozen_state = jax.lax.stop_gradient(snapshot_state)

    log_wf = jnp.log(jnp.array(wf_init, dtype=jnp.float32))
    log_kp = jnp.log(jnp.array(kp_init, dtype=jnp.float32))

    def phase_finite(phase):
        def probe_loss(log_wf, log_kp):
            wf = jnp.exp(log_wf)
            kp = jnp.exp(log_kp)
            dargs = dynamic_args._replace(wind_factor=wf, kp=kp)
            final_state = main_layer_phase_probe(
                frozen_state, probe_forcings, model.point_attrs, static_args, dargs, phase
            )
            return _state_reduction(final_state)

        t0 = time.time()
        try:
            grads = jax.grad(probe_loss, argnums=(0, 1))(log_wf, log_kp)
            ok = all(np.isfinite(np.asarray(g)).all() for g in grads)
        except FloatingPointError:
            ok = False
        print(f"    phase {phase} ({LAYER_PHASE_NAMES[phase]}): "
              f"{'finite' if ok else 'NON-FINITE'} ({time.time()-t0:.1f}s)", flush=True)
        return ok

    if phase_finite(3):
        return None

    lo, hi = 0, 3  # phases 1..lo finite; 1..hi non-finite
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if phase_finite(mid):
            lo = mid
        else:
            hi = mid
    return hi


def bisect_nan_merge_phase(model, snapshot_state, probe_start_idx, wf_init, kp_init, probe_hours=72):
    """
    One level deeper still: merge_existing_layers (the merge scan inside
    check_layer_sizes) bundles 6 internal phases (MERGE_PHASE_NAMES) --
    weighted-average/extensive-sum, shift, two lheight recomputes,
    add_bottom_layer, and update_layer_props. Bisects over which of those
    first introduces a non-finite gradient, using main_merge_phase_probe.
    """
    params = model.config.params
    static_args = model.config.static_args
    dynamic_args = model.config.dynamic_args

    probe_hours = min(probe_hours, len(model.dates) - probe_start_idx)
    probe_forcings = model.pack_forcings(
        params, model.dates[probe_start_idx:probe_start_idx + probe_hours], probe_start_idx
    )
    frozen_state = jax.lax.stop_gradient(snapshot_state)

    log_wf = jnp.log(jnp.array(wf_init, dtype=jnp.float32))
    log_kp = jnp.log(jnp.array(kp_init, dtype=jnp.float32))

    def merge_phase_finite(phase):
        def probe_loss(log_wf, log_kp):
            wf = jnp.exp(log_wf)
            kp = jnp.exp(log_kp)
            dargs = dynamic_args._replace(wind_factor=wf, kp=kp)
            final_state = main_merge_phase_probe(
                frozen_state, probe_forcings, model.point_attrs, static_args, dargs, phase
            )
            return _state_reduction(final_state)

        t0 = time.time()
        try:
            grads = jax.grad(probe_loss, argnums=(0, 1))(log_wf, log_kp)
            ok = all(np.isfinite(np.asarray(g)).all() for g in grads)
        except FloatingPointError:
            ok = False
        print(f"      merge phase {phase} ({MERGE_PHASE_NAMES[phase]}): "
              f"{'finite' if ok else 'NON-FINITE'} ({time.time()-t0:.1f}s)", flush=True)
        return ok

    if merge_phase_finite(7):
        return None

    lo, hi = 0, 7  # phases 1..lo finite; 1..hi non-finite
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if merge_phase_finite(mid):
            lo = mid
        else:
            hi = mid
    return hi


def bisect_nan_merge_var(model, snapshot_state, probe_start_idx, wf_init, kp_init, probe_hours=72):
    """
    One level deeper still: merge phase 2 (shift float-typed vars) bundles
    12 variables (MERGE_VAR_NAMES). Bisects over which variable's shift
    first introduces a non-finite gradient, using main_merge_var_probe.

    CAUTION: main_merge_var_probe truncates a PREFIX of variables and
    returns early -- called repeatedly across the whole layer scan, this
    compounds torn-state inconsistency (every not-yet-shifted field stays
    stale at every merge site), which can itself produce a forward-pass
    artifact NaN distinct from the real (backward-only) bug. If a result
    here shows "in the OUTPUT of a jax.jit function" rather than "in the
    BACKWARD pass", treat it as suspect and prefer test_skip_var/
    bisect_nan_merge_skipvar instead, which ablates one variable at a time
    while keeping everything else fully faithful to production.
    """
    params = model.config.params
    static_args = model.config.static_args
    dynamic_args = model.config.dynamic_args
    n_vars = len(MERGE_VAR_NAMES)

    probe_hours = min(probe_hours, len(model.dates) - probe_start_idx)
    probe_forcings = model.pack_forcings(
        params, model.dates[probe_start_idx:probe_start_idx + probe_hours], probe_start_idx
    )
    frozen_state = jax.lax.stop_gradient(snapshot_state)

    log_wf = jnp.log(jnp.array(wf_init, dtype=jnp.float32))
    log_kp = jnp.log(jnp.array(kp_init, dtype=jnp.float32))

    def n_vars_finite(n):
        def probe_loss(log_wf, log_kp):
            wf = jnp.exp(log_wf)
            kp = jnp.exp(log_kp)
            dargs = dynamic_args._replace(wind_factor=wf, kp=kp)
            final_state = main_merge_var_probe(
                frozen_state, probe_forcings, model.point_attrs, static_args, dargs, n
            )
            return _state_reduction(final_state)

        t0 = time.time()
        try:
            grads = jax.grad(probe_loss, argnums=(0, 1))(log_wf, log_kp)
            ok = all(np.isfinite(np.asarray(g)).all() for g in grads)
        except FloatingPointError:
            ok = False
        var_name = MERGE_VAR_NAMES[n - 1] if n > 0 else '(none)'
        print(f"        first {n}/{n_vars} vars shifted (up to '{var_name}'): "
              f"{'finite' if ok else 'NON-FINITE'} ({time.time()-t0:.1f}s)", flush=True)
        return ok

    if n_vars_finite(n_vars):
        return None

    lo, hi = 0, n_vars  # first lo vars finite; first hi vars non-finite
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if n_vars_finite(mid):
            lo = mid
        else:
            hi = mid
    return hi  # 1-based count; MERGE_VAR_NAMES[hi-1] is the culprit


def find_nan_merge_skipvar(model, snapshot_state, probe_start_idx, wf_init, kp_init, probe_hours=72):
    """
    Tests each of the 12 merge-shift variables individually: is the gradient
    finite when *only that one* variable's shift is skipped (a no-op),
    with every other field handled exactly as in production
    (merge_existing_layers_skip_var_probe)? Unlike bisect_nan_merge_var's
    prefix-truncation approach, this doesn't compound torn-state
    inconsistency across the scan, so it should reproduce the same
    backward-pass-only failure mode as every earlier (fully faithful)
    bisection level, not a forward-pass artifact.

    Not a bisection (skip-conditions aren't a monotonic/prefix relationship
    like the other probes) -- just tests all 12, plus a None baseline that
    should reproduce the confirmed-non-finite full-shift result as a sanity
    check. Returns the list of variable names whose omission made the
    gradient finite (the culprit(s)), or [] if none did.
    """
    params = model.config.params
    static_args = model.config.static_args
    dynamic_args = model.config.dynamic_args

    probe_hours = min(probe_hours, len(model.dates) - probe_start_idx)
    probe_forcings = model.pack_forcings(
        params, model.dates[probe_start_idx:probe_start_idx + probe_hours], probe_start_idx
    )
    frozen_state = jax.lax.stop_gradient(snapshot_state)

    log_wf = jnp.log(jnp.array(wf_init, dtype=jnp.float32))
    log_kp = jnp.log(jnp.array(kp_init, dtype=jnp.float32))

    def skip_finite(skip_var):
        def probe_loss(log_wf, log_kp):
            wf = jnp.exp(log_wf)
            kp = jnp.exp(log_kp)
            dargs = dynamic_args._replace(wind_factor=wf, kp=kp)
            final_state = main_merge_skipvar_probe(
                frozen_state, probe_forcings, model.point_attrs, static_args, dargs, skip_var
            )
            return _state_reduction(final_state)

        t0 = time.time()
        try:
            grads = jax.grad(probe_loss, argnums=(0, 1))(log_wf, log_kp)
            ok = all(np.isfinite(np.asarray(g)).all() for g in grads)
        except FloatingPointError:
            ok = False
        label = f"skip '{skip_var}'" if skip_var is not None else "skip none (baseline)"
        print(f"        {label}: {'finite' if ok else 'NON-FINITE'} ({time.time()-t0:.1f}s)", flush=True)
        # each skip_var is a distinct static arg -> a fresh XLA compilation;
        # without this, 13 full checkpointed-scan executables accumulate in
        # memory over the loop and can exhaust RAM (seen as an LLVM
        # "unable to allocate section memory" abort, not an array-size OOM)
        jax.clear_caches()
        return ok

    print("      baseline (should match confirmed non-finite full shift):", flush=True)
    baseline_ok = skip_finite(None)
    if baseline_ok:
        print("      unexpected: baseline is finite here but not in the merge-phase probe -- "
              "the two probe implementations may have diverged.", flush=True)

    print("      testing each variable individually:", flush=True)
    culprits = []
    for var in MERGE_VAR_NAMES:
        if skip_finite(var):
            culprits.append(var)
    return culprits


def bisect_nan_merge_nvars(model, snapshot_state, probe_start_idx, wf_init, kp_init, probe_hours=72):
    """
    find_nan_merge_skipvar showed omitting any single one of 12 variables
    isn't enough to fix it. This finds the minimum number of *simultaneously*
    shifted variables (first N of MERGE_VAR_NAMES, in order) needed to
    trigger the non-finite gradient, using main_merge_nvars_probe -- which,
    unlike main_merge_var_probe, keeps every downstream step (recompute,
    add_bottom_layer, update_layer_props) fully faithful at every iteration
    rather than returning early, avoiding that probe's forward-pass artifact.
    """
    params = model.config.params
    static_args = model.config.static_args
    dynamic_args = model.config.dynamic_args
    n_vars = len(MERGE_VAR_NAMES)

    probe_hours = min(probe_hours, len(model.dates) - probe_start_idx)
    probe_forcings = model.pack_forcings(
        params, model.dates[probe_start_idx:probe_start_idx + probe_hours], probe_start_idx
    )
    frozen_state = jax.lax.stop_gradient(snapshot_state)

    log_wf = jnp.log(jnp.array(wf_init, dtype=jnp.float32))
    log_kp = jnp.log(jnp.array(kp_init, dtype=jnp.float32))

    def n_vars_finite(n):
        def probe_loss(log_wf, log_kp):
            wf = jnp.exp(log_wf)
            kp = jnp.exp(log_kp)
            dargs = dynamic_args._replace(wind_factor=wf, kp=kp)
            final_state = main_merge_nvars_probe(
                frozen_state, probe_forcings, model.point_attrs, static_args, dargs, n
            )
            return _state_reduction(final_state)

        t0 = time.time()
        try:
            grads = jax.grad(probe_loss, argnums=(0, 1))(log_wf, log_kp)
            ok = all(np.isfinite(np.asarray(g)).all() for g in grads)
        except FloatingPointError:
            ok = False
        var_name = MERGE_VAR_NAMES[n - 1] if n > 0 else '(none)'
        print(f"        first {n}/{n_vars} vars shifted (up to '{var_name}'): "
              f"{'finite' if ok else 'NON-FINITE'} ({time.time()-t0:.1f}s)", flush=True)
        jax.clear_caches()  # each n is a distinct compile; see find_nan_merge_skipvar
        return ok

    if n_vars_finite(n_vars):
        return None

    lo, hi = 0, n_vars  # first lo vars finite; first hi vars non-finite
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if n_vars_finite(mid):
            lo = mid
        else:
            hi = mid
    return hi  # 1-based count; MERGE_VAR_NAMES[hi-1] is the last var added


# ---------------------------------------------------------------------------
# 5. Optimization
# ---------------------------------------------------------------------------

def run_optimization(loss_fn, init_wind_factors, init_kp, site_order, n_steps=100, lr=1e-2):
    params = {
        'log_wind_factor': jnp.log(jnp.array(init_wind_factors, dtype=jnp.float32)),
        'log_kp': jnp.log(jnp.array(init_kp, dtype=jnp.float32)),
    }

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    def wrapped_loss(params):
        return loss_fn(params['log_wind_factor'], params['log_kp'])

    grad_fn = jax.jit(jax.value_and_grad(wrapped_loss, has_aux=True))

    # Fallback diagnostic for whenever PEBSI_DEBUG_NANS=0 (or a NaN somehow
    # slips past the debug_nans tripwire): identify which of wind_factor/kp
    # is bad and at which sites, then stop instead of burning further steps.
    def _report_nan_grads(step, grads):
        found = False
        for name, g in (('wind_factor', grads['log_wind_factor']), ('kp', grads['log_kp'])):
            bad = np.where(~np.isfinite(np.asarray(g)))[0]
            if len(bad):
                found = True
                bad_sites = [site_order[j] for j in bad]
                print(f"  step {step}: non-finite grad for {name} at {len(bad)} site(s): {bad_sites}", flush=True)
        return found

    print(f"{'Step':>6}  {'Summer RMSE':>12}  {'Winter RMSE':>12}  {'wf |grad|':>10}  {'kp |grad|':>10}", flush=True)
    for i in range(n_steps):
        t0 = time.time()
        (total_loss, (summer_rmse, winter_rmse)), grads = grad_fn(params)
        jax.block_until_ready((total_loss, grads))

        if _report_nan_grads(i, grads):
            print("  Stopping: non-finite gradient detected (see above).", flush=True)
            break

        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        wf_grad_norm = float(jnp.linalg.norm(grads['log_wind_factor']))
        kp_grad_norm = float(jnp.linalg.norm(grads['log_kp']))
        print(f"{i:>6}  {float(summer_rmse):>12.4f}  {float(winter_rmse):>12.4f}  "
              f"{wf_grad_norm:>10.3e}  {kp_grad_norm:>10.3e}  ({time.time()-t0:.1f}s)", flush=True)

    return jnp.exp(params['log_wind_factor']), jnp.exp(params['log_kp'])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}", flush=True)

    site_dict = load_site_dict()
    if USE_REDUCED_SITE_SET:
        full_n = sum(len(v) for v in site_dict.values())
        site_dict = load_reduced_site_dict(REDUCED_SITES_CONFIG)
        reduced_n = sum(len(v) for v in site_dict.values())
        print(f"Using reduced site set from {REDUCED_SITES_CONFIG}: "
              f"{reduced_n} sites (full sites.pkl set has {full_n})", flush=True)
    site_order = flatten_site_order(site_dict)

    config_fp = build_generated_config(
        site_dict, host, start_date=DEBUG_START_DATE, end_date=DEBUG_END_DATE,
        temporal_chunk_years=(BISECT_CHUNK_SIZE / 8760) if (NAN_BISECT or STAGE_BISECT) else 1,
    )
    print(f"Generated config for {len(site_order)} sites -> {config_fp}")

    print("Loading observations...")
    obs_by_season = load_all_observations(site_dict)
    summer_labels, summer_meas, summer_mask, summer_starts, summer_ends = obs_by_season['summer']
    winter_labels, winter_meas, winter_mask, winter_starts, winter_ends = obs_by_season['winter']
    print(f"Loaded {len(summer_labels)} sites with summer obs, "
          f"{len(winter_labels)} sites with winter obs\n")

    print("Initializing PEBSI...")
    model = init_pebsi(config_fp)

    print("Building period index arrays...")
    summer_period_idx = build_period_indices(model.dates, summer_starts, summer_ends)
    winter_period_idx = build_period_indices(model.dates, winter_starts, winter_ends)
    print(f"  Valid summer periods: {(summer_period_idx[:, :, 0] >= 0).sum()} / {summer_period_idx[:, :, 0].size}")
    print(f"  Valid winter periods: {(winter_period_idx[:, :, 0] >= 0).sum()} / {winter_period_idx[:, :, 0].size}")

    loss_fn = make_loss_fn(
        model, site_order,
        summer=(summer_labels, summer_period_idx, summer_meas, summer_mask),
        winter=(winter_labels, winter_period_idx, winter_meas, winter_mask),
    )

    wf_init = model.config.dynamic_args.wind_factor
    kp_init = model.config.dynamic_args.kp
    print(f"wind_factor shape={jnp.asarray(wf_init).shape}  value={wf_init}", flush=True)
    print(f"kp shape={jnp.asarray(kp_init).shape}  value={kp_init}", flush=True)

    print("Checking post-spinup state for NaNs...", flush=True)
    st = model.initial_state
    any_nan = False
    for field, value in st._asdict().items():
        arr = np.asarray(value)
        if not np.issubdtype(arr.dtype, np.floating):
            continue
        nan_sites = np.isnan(arr).reshape(arr.shape[0], -1).any(axis=-1)
        if nan_sites.any():
            any_nan = True
            print(f"  NaN in '{field}': site(s) {np.where(nan_sites)[0].tolist()}", flush=True)
    if not any_nan:
        print("  No NaNs found in any post-spinup state field.", flush=True)

    if NAN_BISECT:
        print("PEBSI_NAN_BISECT=1: localizing NaN gradient by chunk bisection...\n", flush=True)
        bisect_nan_chunk(
            loss_fn, list(wf_init), list(kp_init),
            model_dates=model.dates,
            chunk_size=model.config.params.temporal_chunk_hours,
        )
        sys.exit(0)

    if STAGE_BISECT:
        # PEBSI_NAN_BISECT already established the NaN reproduces within
        # chunk 0 (the very first ~month post-spinup) -- no accumulation
        # over years needed. So bisect stages directly from the model's
        # actual initial_state over that same window, rather than scattering
        # across forward-simulated snapshots (generate_snapshots/PROBE_HOURS
        # are still here if a *different* window ever needs checking).
        print(f"PEBSI_STAGE_BISECT=1: bisecting physics stages over chunk 0 "
              f"(first {BISECT_CHUNK_SIZE}h from initial_state)...\n", flush=True)
        bad_stage = bisect_nan_stage(
            model, model.initial_state, 0, list(wf_init), list(kp_init),
            probe_hours=BISECT_CHUNK_SIZE,
        )
        if bad_stage is not None:
            print(f"\n-> first non-finite stage: {bad_stage} ({STAGE_NAMES[bad_stage]})", flush=True)
        else:
            print("\nAll stages finite over chunk 0 -- unexpected given PEBSI_NAN_BISECT's "
                  "result; the discrepancy itself is worth understanding (e.g. probe's scalar "
                  "reduction may not be sensitive to whatever's blowing up).", flush=True)
        sys.exit(0)

    if VERTICAL_SUBSTAGE_BISECT:
        print(f"PEBSI_VERTICAL_SUBSTAGE_BISECT=1: bisecting run_vertical_processes' "
              f"sub-calls over chunk 0 (first {BISECT_CHUNK_SIZE}h from initial_state)...\n", flush=True)
        bad_substage = bisect_nan_vertical_substage(
            model, model.initial_state, 0, list(wf_init), list(kp_init),
            probe_hours=BISECT_CHUNK_SIZE,
        )
        if bad_substage is not None:
            print(f"\n-> first non-finite sub-call: {bad_substage} "
                  f"({VERTICAL_SUBSTAGE_NAMES[bad_substage]})", flush=True)
        else:
            print("\nAll sub-calls finite over chunk 0 -- unexpected given the stage-4 result.", flush=True)
        sys.exit(0)

    if LAYER_PHASE_BISECT:
        print(f"PEBSI_LAYER_PHASE_BISECT=1: bisecting check_layer_sizes' phases "
              f"over chunk 0 (first {BISECT_CHUNK_SIZE}h from initial_state)...\n", flush=True)
        bad_phase = bisect_nan_layer_phase(
            model, model.initial_state, 0, list(wf_init), list(kp_init),
            probe_hours=BISECT_CHUNK_SIZE,
        )
        if bad_phase is not None:
            print(f"\n-> first non-finite phase: {bad_phase} ({LAYER_PHASE_NAMES[bad_phase]})", flush=True)
        else:
            print("\nAll phases finite over chunk 0 -- unexpected given the substage-5 result.", flush=True)
        sys.exit(0)

    if MERGE_PHASE_BISECT:
        print(f"PEBSI_MERGE_PHASE_BISECT=1: bisecting merge_existing_layers' internal "
              f"phases over chunk 0 (first {BISECT_CHUNK_SIZE}h from initial_state)...\n", flush=True)
        bad_merge_phase = bisect_nan_merge_phase(
            model, model.initial_state, 0, list(wf_init), list(kp_init),
            probe_hours=BISECT_CHUNK_SIZE,
        )
        if bad_merge_phase is not None:
            print(f"\n-> first non-finite merge phase: {bad_merge_phase} "
                  f"({MERGE_PHASE_NAMES[bad_merge_phase]})", flush=True)
        else:
            print("\nAll merge phases finite over chunk 0 -- unexpected given the "
                  "layer-phase result.", flush=True)
        sys.exit(0)

    if MERGE_VAR_BISECT:
        print(f"PEBSI_MERGE_VAR_BISECT=1: bisecting which shifted variable is "
              f"responsible, over chunk 0 (first {BISECT_CHUNK_SIZE}h from initial_state)...\n",
              flush=True)
        bad_n = bisect_nan_merge_var(
            model, model.initial_state, 0, list(wf_init), list(kp_init),
            probe_hours=BISECT_CHUNK_SIZE,
        )
        if bad_n is not None:
            print(f"\n-> first non-finite variable: {MERGE_VAR_NAMES[bad_n - 1]}", flush=True)
        else:
            print("\nAll variables finite over chunk 0 -- unexpected given the merge-phase result.",
                  flush=True)
        sys.exit(0)

    if MERGE_SKIPVAR_TEST:
        print(f"PEBSI_MERGE_SKIPVAR_TEST=1: testing each merge-shift variable "
              f"individually over chunk 0 (first {BISECT_CHUNK_SIZE}h from initial_state)...\n",
              flush=True)
        culprits = find_nan_merge_skipvar(
            model, model.initial_state, 0, list(wf_init), list(kp_init),
            probe_hours=BISECT_CHUNK_SIZE,
        )
        if culprits:
            print(f"\n-> skipping these variable(s) made the gradient finite: {culprits}", flush=True)
        else:
            print("\nNo single variable's omission fixed it -- likely needs two or more "
                  "variables together, or the bug isn't in the shift step's variable loop "
                  "itself but something structural common to all of them.", flush=True)
        sys.exit(0)

    if MERGE_NVARS_BISECT:
        print(f"PEBSI_MERGE_NVARS_BISECT=1: bisecting minimum simultaneous-shift count "
              f"over chunk 0 (first {BISECT_CHUNK_SIZE}h from initial_state)...\n", flush=True)
        bad_n = bisect_nan_merge_nvars(
            model, model.initial_state, 0, list(wf_init), list(kp_init),
            probe_hours=BISECT_CHUNK_SIZE,
        )
        if bad_n is not None:
            print(f"\n-> minimum {bad_n} simultaneously shifted variables needed "
                  f"(up to '{MERGE_VAR_NAMES[bad_n - 1]}' in order)", flush=True)
        else:
            print("\nEven shifting all 12 is finite here -- unexpected given the "
                  "skip-var result.", flush=True)
        sys.exit(0)

    print("Optimizing wind_factor (summer MB) and kp (winter MB) for all sites...\n", flush=True)
    wind_factors, kps = run_optimization(
        loss_fn,
        init_wind_factors=list(wf_init),
        init_kp=list(kp_init),
        site_order=site_order,
        n_steps=10, lr=1e-2,
    )

    print("\nOptimized parameters:")
    print(f"{'Glacier':<14} {'Site':<6} {'wind_factor':>12} {'kp':>10}")
    for (glacier, site), wf, kp in zip(site_order, wind_factors, kps):
        print(f"{glacier:<14} {site:<6} {float(wf):>12.4f} {float(kp):>10.4f}")
