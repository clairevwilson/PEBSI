"""
JAX optimization of per-site wind_factor using PEBSI's differentiable core.

The full PEBSI forward model (pebsi/main.py) is @jax.jit compiled and runs
via jax.lax.scan, so wind_factor flows through as a traced JAX array.
We differentiate the MB RMSE loss w.r.t. wind_factor directly.

Steps:
  1. Load MassBalance observations for all sites.
  2. Initialize PEBSI (sets up state, forcings, etc.) — this is the numpy side.
  3. Define a loss function that swaps wind_factor into dynamic_args, calls
     pebsi.main.main(), sums MB over observation periods, and returns RMSE.
  4. jax.grad + optax to optimize wind_factor per site simultaneously.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# netCDF4 must load before JAX to avoid numpy ABI conflict
import simulation as sim

import time
import jax
jax.config.update("jax_traceback_filtering", "off")
import jax.numpy as jnp
import optax
import numpy as np
import pandas as pd

from project.data_handling import MassBalance
from pebsi.main import main as pebsi_main

site_dict = {
    'wolverine':['N','B','EC'],
    'kahiltna':['K53','K17b'], 
    'kennicott':['GTL','GTH'],
    'lemon_creek':['B','C','D'],
    'taku':['NWB1','TKG3'],
    'gulkana':['AU','B','D']
}

# ---------------------------------------------------------------------------
# 1. Load observations
# ---------------------------------------------------------------------------

def load_all_observations(site_dict):
    """
    Returns:
        site_labels    : list of (glacier, site) tuples, length S
        meas_padded    : np.ndarray (S, N_max) measured MB [m w.e.], NaN-padded
        period_starts  : list of np.ndarray, one per site
        period_ends    : list of np.ndarray, one per site
    """
    site_labels = []
    meas_list = []
    period_starts_list = []
    period_ends_list = []

    failed = []
    for glacier in site_dict:
        for site in site_dict[glacier]:
            try:
                obs = MassBalance(glacier, site, use='benchmark', min_n_winter=1)
                site_labels.append((glacier, site))
                meas_list.append(obs.data.astype(np.float32))
                period_starts_list.append(obs.period_starts)
                period_ends_list.append(obs.period_ends)
            except Exception as e:
                failed.append((glacier, site, str(e)))

    if failed:
        print("Warning: failed to load observations for:")
        for glacier, site, err in failed:
            print(f"  {glacier}/{site}: {err}")

    N_max = max(len(m) for m in meas_list)
    N_sites = len(site_labels)

    meas_padded = np.full((N_sites, N_max), np.nan, dtype=np.float32)
    mask = np.zeros((N_sites, N_max), dtype=bool)
    for i, m in enumerate(meas_list):
        n = len(m)
        meas_padded[i, :n] = m
        mask[i, :n] = True

    return site_labels, meas_padded, mask, period_starts_list, period_ends_list


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
    print("Running spinup...")
    spun_up_state = model.spinup(model.initial_state)
    model.initial_state = spun_up_state

    return model


# ---------------------------------------------------------------------------
# 4. Loss function
# ---------------------------------------------------------------------------

def make_loss_fn(model, period_idx, meas_batch, mask_batch):
    """
    Returns a loss function that takes log_wind_factor (S,) and returns
    scalar mean RMSE across sites.

    period_idx : (S, N_max, 2) int32 — timestep [start, end) per period
    meas_batch : (S, N_max) float32 — observed MB
    mask_batch : (S, N_max) bool
    """
    meas_jax = jnp.array(meas_batch)
    mask_jax = jnp.array(mask_batch)
    period_idx_jax = jnp.array(period_idx)

    static_args = model.config.static_args
    dynamic_args = model.config.dynamic_args
    params = model.config.params
    chunk_size = params.temporal_chunks
    # truncate dates to a multiple of chunk_size so chunks are always full
    total_steps = (len(model.dates) // chunk_size) * chunk_size
    S, N_max = meas_jax.shape

    # Stack all chunk forcings into a single JAX array so lax.scan can
    # iterate over them — this is the key to bounded memory during autodiff.
    # Each leaf of the forcing namedtuple gets an extra leading chunk dimension.
    chunk_list = [
        jax.tree.map(lambda x: jnp.nan_to_num(x, nan=0.0),
                     model.pack_forcings(params, model.dates[start:start + chunk_size], start))
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

    def loss_fn(log_wind_factor):
        wind_factor = jnp.exp(log_wind_factor)
        new_dynamic_args = dynamic_args._replace(wind_factor=wind_factor)

        def scan_chunk(carry, xs):
            state, period_sums = carry
            chunk_forcings, t_offset = xs

            state, records = jax.checkpoint(_run_chunk)(state, chunk_forcings, new_dynamic_args)
            chunk_mb = records.accumulation + records.refreeze - records.melt  # (T_chunk, N_POINTS)

            # accumulate MB into per-period sums for each site
            def accumulate_site(period_sums, site_idx):
                mb_site = chunk_mb[:, site_idx]  # (T_chunk,)

                def add_period(carry, j):
                    p_start = period_idx_jax[site_idx, j, 0]
                    p_end = period_idx_jax[site_idx, j, 1]
                    t = jnp.arange(chunk_size) + t_offset
                    contrib = jnp.where((t >= p_start) & (t < p_end), mb_site, 0.0).sum()
                    return carry.at[j].add(contrib), None

                row, _ = jax.lax.scan(add_period, period_sums[site_idx], jnp.arange(N_max))
                return period_sums.at[site_idx].set(row), None

            period_sums, _ = jax.lax.scan(accumulate_site, period_sums, jnp.arange(S))
            return (state, period_sums), None

        init_carry = (model.initial_state, jnp.zeros((S, N_max)))
        (_, period_sums), _ = jax.lax.scan(
            scan_chunk, init_carry, (stacked_forcings, chunk_offsets)
        )

        safe_meas = jnp.where(mask_jax, meas_jax, 0.0)
        residuals = period_sums - safe_meas
        sq = jnp.where(mask_jax, jnp.square(residuals), 0.0)
        counts = jnp.sum(mask_jax, axis=1).clip(min=1)
        per_site_rmse = jnp.sqrt(jnp.sum(sq, axis=1) / counts + 1e-8)
        return jnp.mean(per_site_rmse)

    return loss_fn


# ---------------------------------------------------------------------------
# 5. Optimization
# ---------------------------------------------------------------------------

def run_optimization(loss_fn, n_sites, n_steps=100, lr=1e-2):
    # Start from wind_factor = 1 for all sites (log(1) = 0)
    log_wf = jnp.zeros((n_sites,))

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(log_wf)

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))

    print(f"{'Step':>6}  {'Mean RMSE':>10}")
    for i in range(n_steps):
        t0 = time.time()
        loss, grads = grad_fn(log_wf)
        jax.block_until_ready((loss, grads))
        updates, opt_state = optimizer.update(grads, opt_state)
        log_wf = optax.apply_updates(log_wf, updates)
        print(f"{i:>6}  {float(loss):>10.4f}  ({time.time()-t0:.1f}s)")

    return jnp.exp(log_wf)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Loading observations...")
    site_labels, meas_padded, mask, period_starts_list, period_ends_list = \
        load_all_observations(site_dict)
    S = len(site_labels)
    print(f"Loaded {S} sites\n")

    print("Initializing PEBSI...")
    model = init_pebsi('configs_opt.yaml')

    print("Building period index arrays...")
    period_idx = build_period_indices(model.dates, period_starts_list, period_ends_list)
    print(f"  Valid periods found: {(period_idx[:, :, 0] >= 0).sum()} / {period_idx[:, :, 0].size}")

    loss_fn = make_loss_fn(model, period_idx, meas_padded, mask)

    print("Optimizing wind_factor for all sites...\n")
    wind_factors = run_optimization(loss_fn, n_sites=S, n_steps=2, lr=1e-2)

    print("\nOptimized wind factors:")
    print(f"{'Glacier':<14} {'Site':<6} {'wind_factor':>12}")
    for (glacier, site), wf in zip(site_labels, wind_factors):
        print(f"{glacier:<14} {site:<6} {float(wf):>12.4f}")
