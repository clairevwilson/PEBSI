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
     season, and returns the combined MAE loss.
  4. jax.grad + optax to optimize wind_factor and kp per site simultaneously.
"""
import os

# Toggle back to True to return to the full sites.pkl set once a wider
# calibration run is wanted; currently restricted to the 16 sites in
# configs_opt_2015_2020.yaml, whose benchmark observations all span the
# 2015-04-01 to 2020-04-01 window used below.
USE_REDUCED_SITE_SET = True
REDUCED_SITES_CONFIG = os.path.join(os.path.dirname(__file__), 'configs_opt_2015_2020.yaml')
START_DATE = '2015-04-01'
END_DATE = '2020-04-01'

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
from pebsi.main import main as pebsi_main

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
    (e.g. configs_opt_2015_2020.yaml) instead of the full sites.pkl set.
    load_site_dict()/sites.pkl is untouched so switching back to the full
    set is just flipping USE_REDUCED_SITE_SET.
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
    (total_loss, (summer_mae, winter_mae)).

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

    def _mae(period_sums, meas_jax, mask_jax):
        safe_meas = jnp.where(mask_jax, meas_jax, 0.0)
        residuals = period_sums - safe_meas
        abs_res = jnp.where(mask_jax, jnp.abs(residuals), 0.0)
        counts = jnp.sum(mask_jax, axis=1).clip(min=1)
        per_site_mae = jnp.sum(abs_res, axis=1) / counts
        return jnp.mean(per_site_mae)

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
            scan_chunk, init_carry, (stacked_forcings, chunk_offsets)
        )

        summer_mae = _mae(summer_sums, summer_meas, summer_mask)
        winter_mae = _mae(winter_sums, winter_meas, winter_mask)
        total_loss = summer_mae + winter_mae
        return total_loss, (summer_mae, winter_mae)

    return loss_fn


# ---------------------------------------------------------------------------
# 5. Optimization
# ---------------------------------------------------------------------------

def run_optimization(loss_fn, init_wind_factors, init_kp, site_order, n_steps=100, lr=1e-2, clip_norm=1.0):
    params = {
        'log_wind_factor': jnp.log(jnp.array(init_wind_factors, dtype=jnp.float32)),
        'log_kp': jnp.log(jnp.array(init_kp, dtype=jnp.float32)),
    }

    # clip_by_global_norm caps the raw gradient BEFORE adam sees it, so a
    # single site hitting a steep-but-finite region of the loss surface
    # (e.g. a layer landing near one of the epsilon floors in layers.py --
    # see update_layer_props/merge_existing_layers) can't blow up the whole
    # trajectory the way it did at lr=5e-2/step 4 (wf/kp grad norms jumping
    # from O(1) to O(1e7) in one step). clip_norm=1.0 is comfortably above
    # the O(0.2-1.0) per-parameter norms seen in a normal step, so it's a
    # no-op most of the time and only engages on an actual spike.
    optimizer = optax.chain(
        optax.clip_by_global_norm(clip_norm),
        optax.adam(lr),
    )
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

    # Per-site breakdown so a spike like the lr=5e-2 run's step 3/4 (kp grad
    # norm 0.17 -> 8.5 -> 5.8e7) points straight at the responsible site
    # instead of just the aggregate norm.
    def _top_grad_sites(g, n=3):
        g = np.asarray(g)
        idx = np.argsort(-np.abs(g))[:n]
        return ', '.join(f"{site_order[j]}={g[j]:.3e}" for j in idx)

    print(f"{'Step':>6}  {'Summer MAE':>12}  {'Winter MAE':>12}  {'wf |grad|':>10}  {'kp |grad|':>10}", flush=True)
    for i in range(n_steps):
        t0 = time.time()
        (total_loss, (summer_mae, winter_mae)), grads = grad_fn(params)
        jax.block_until_ready((total_loss, grads))

        if _report_nan_grads(i, grads):
            print("  Stopping: non-finite gradient detected (see above).", flush=True)
            break

        raw_global_norm = float(optax.global_norm(grads))
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        wf_grad_norm = float(jnp.linalg.norm(grads['log_wind_factor']))
        kp_grad_norm = float(jnp.linalg.norm(grads['log_kp']))
        print(f"{i:>6}  {float(summer_mae):>12.4f}  {float(winter_mae):>12.4f}  "
              f"{wf_grad_norm:>10.3e}  {kp_grad_norm:>10.3e}  ({time.time()-t0:.1f}s)", flush=True)
        print(f"    top wf grad sites: {_top_grad_sites(grads['log_wind_factor'])}", flush=True)
        print(f"    top kp grad sites: {_top_grad_sites(grads['log_kp'])}", flush=True)
        if raw_global_norm > clip_norm:
            print(f"    clipped: raw global grad norm {raw_global_norm:.3e} -> {clip_norm}", flush=True)

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
        site_dict, host, start_date=START_DATE, end_date=END_DATE,
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

    print("Optimizing wind_factor (summer MB) and kp (winter MB) for all sites...\n", flush=True)
    wind_factors, kps = run_optimization(
        loss_fn,
        init_wind_factors=list(wf_init),
        init_kp=list(kp_init),
        site_order=site_order,
        n_steps=10, lr=5e-2,
    )

    print("\nOptimized parameters:")
    print(f"{'Glacier':<14} {'Site':<6} {'wind_factor':>12} {'kp':>10}")
    for (glacier, site), wf, kp in zip(site_order, wind_factors, kps):
        print(f"{glacier:<14} {site:<6} {float(wf):>12.4f} {float(kp):>10.4f}")
