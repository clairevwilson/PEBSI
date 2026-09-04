"""
Gradient-based (automatic differentiation) calibration of per-glacier
wind_factor and precipitation factor (kp) against glacier-wide
observations, using PEBSI's differentiable core.

This is the distributed successor to the point-scale jax_optimize.py.
The domain is the adaptive point mesh (method_distribute='adaptive'),
so every glacier is represented by an area-scaled cloud of points
rather than a handful of named benchmark sites, and the objective is
built from the four glacier-wide error metrics in
project/glacierwide_loss.py:

  - remotely sensed albedo         (Gaussian NLL on the residuals)
  - SAR snow extent                (Bernoulli log-loss)
  - SAR melt extent                (Bernoulli log-loss)
  - Hugonnet glacier-wide balance  (Gaussian NLL on the 20-year total)

Every glacier keeps its own (kp, wind_factor) pair and its own losses;
the objective is their sum, so a glacier's gradient never depends on
another glacier's parameters. Pooling them into one forward pass is
purely a compute convenience, exactly as in the Bayesian grid runs.

Two details make a 20-year distributed run tractable that the
point-scale version did not need:

  1. Day-level rematerialization. jax_optimize checkpointed the hourly
     step, but reverse mode over a scan still retains one carry per
     scanned element, so every hour of every output day stayed resident.
     PEBSI now checkpoints the whole output period when
     differentiable=True, cutting the live state by steps_per_output.
     Combined with checkpointing each temporal chunk here, peak memory
     is set by the chunk length rather than the run length -- at ~1600
     points a 172-day chunk holds roughly 1.3 GB of state.

  2. Softened snow/melt indicators. The observed quantities are binary
     masks, and the model's own snow/melt tests are step functions with
     no useful gradient. Both are replaced by a logistic on the
     underlying continuous field (snow + firn depth; column liquid
     water), which reduces to the hard threshold as the width goes to
     zero.
"""
import os
import sys
import time
import socket
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# netCDF4 must load before JAX to avoid a numpy ABI conflict
import simulation as sim
from pebsi.main import main as pebsi_main

import jax
jax.config.update("jax_traceback_filtering", "off")
if os.environ.get('PEBSI_DEBUG_NANS', '0') != '0':
    # raises at the exact primitive that first produces a NaN/Inf instead of
    # letting it propagate; real overhead, so off unless asked for
    jax.config.update("jax_debug_nans", True)
    jax.config.update("jax_debug_infs", True)
import jax.numpy as jnp
import optax
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from pyproj import Transformer

from project.glacierwide_loss import Albedo, SnowlineMelt, MassBalance, translate_rgi

# ---------------------------------------------------------------------------
# Run settings
# ---------------------------------------------------------------------------

GLACIERS = ['gulkana', 'kahiltna', 'kennicott', 'wolverine', 'lemon_creek']

# whole days, as output_freq='daily' requires: the inclusive hourly range
# 2000-01-01 00:00 -> 2020-03-31 23:00 is exactly 7396 days
START_DATE = '2000-01-01 00:00'
END_DATE = '2020-03-31 23:00'

# the Hugonnet dh/dt product's own period, which is shorter than the
# simulation; only days inside it enter the mass balance residual
MB_START = '2000-01-01'
MB_END = '2020-01-01'

# Days per temporal chunk. This sets peak device memory: the backward pass
# holds one chunk's rematerialized interior, roughly 10 MB per simulated day
# at ~1600 points (13 float64 (N_POINTS, 50) layer fields, ~9 MB, plus that
# day's forcing residuals). On top of that sit the run's stacked forcings
# (~5 GB) and observations (~1.2 GB), which do not vary with chunk length.
# So on a 32 GB card: 1 year ~ 10 GB total, 3 years ~ 17 GB, 5 years ~ 24 GB
# and getting tight. The run's day count need not divide evenly -- any
# remainder is run as a shorter trailing chunk outside the scan, at the cost
# of one extra XLA compile.
CHUNK_DAYS = 365

# only these are needed by the four metrics; every extra field is carried
# through the whole differentiated graph, so the list is kept minimal
STORE_VARS = ('mass_balance', 'albedo', 'total_water', 'snowdepth')

# observation uncertainties, matching glacierwide_loss's defaults
ALBEDO_SIGMA = 0.03
BERNOULLI_EPS = 1e-3

# thresholds and logistic widths for the softened snow/melt indicators.
# The thresholds are glacierwide_loss's own (0.05 m of snow+firn depth,
# 0.05 kg m-2 of column water); the widths set how far either side of the
# threshold carries gradient.
SNOW_DEPTH_THRESHOLD = 0.05
SNOW_DEPTH_WIDTH = 0.02
MELT_WATER_THRESHOLD = 0.05
MELT_WATER_WIDTH = 0.02

# months of the year the albedo product is scored over, as in
# Albedo.get_model_albedo
ALBEDO_MONTHS = list(range(3, 10))

baseline = {'kp': 2.5, 'wind_factor': 3.0}

if 'trace' in socket.gethostname():
    host = 'trace'
else:
    host = 'bridges'

HOST_PATHS = {
    'trace': dict(
        climate_fp='/trace/group/rounce/cvwilson/climate_data/',
        rgi_fp='/trace/group/rounce/shared/RGI/rgi60/00_rgi60_attribs/',
        output_fp='/trace/group/rounce/cvwilson/Output/AD_optimize/',
        cop30_vrt_path='/trace/group/rounce/cvwilson/dems/RGI1_DEM/rgi_dem.vrt',
        shading_fp='/trace/group/rounce/cvwilson/shading/',
    ),
    'bridges': dict(
        climate_fp='/ocean/projects/ees260009p/cwilson4/climate_data/',
        rgi_fp='/ocean/projects/ees260009p/cwilson4/RGI/rgi60/00_rgi60_attribs/',
        output_fp='/ocean/projects/ees260009p/cwilson4/Output/AD_optimize/',
        cop30_vrt_path='/ocean/projects/ees260009p/cwilson4/data/dems/COP30/COP30_reg01.vrt',
        shading_fp='/ocean/projects/ees260009p/cwilson4/data/shading/',
        ice_albedo_fn='/ocean/projects/ees260009p/cwilson4/data/ice_albedo/{gid}_albedo.tif',
        thickness_fn='/ocean/projects/ees260009p/cwilson4/data/ice_thickness/RGI60-01/RGI60-{gid}_thickness.tif',
        windmap_fn='/ocean/projects/ees260009p/cwilson4/data/windmapper/{gid}.nc',
    ),
}

# physics settings shared with the distributed reference run (config.yaml)
BASE_CONFIG = dict(
    option_ice_albedo_tif=True,
    option_windmaps=True,
    option_accel_grains=True,
    option_flat_plates=True,
    option_dynamics=False,
    constant_freshgrainsize=54.5,
    constant_irrwater=True,
    precgrad=0.000100,
    bias_vars=['temp'],
)


def build_config():
    """
    Writes the PEBSI config for the optimization run: the five calibration
    glaciers on the adaptive point mesh, daily output, and nothing written
    to disk (every field the objective needs is read straight off the JAX
    records).
    """
    config = dict(BASE_CONFIG)
    config.update(
        rgi_ids=[translate_rgi[g]['6'] for g in GLACIERS],
        method_distribute='adaptive',
        start_date=START_DATE,
        end_date=END_DATE,
        output_freq='daily',
        store_vars=list(STORE_VARS),
        store_data=False,
        debug=False,
        progress_bar=False,
        kp=baseline['kp'],
        wind_factor=baseline['wind_factor'],
        **HOST_PATHS[host],
    )
    config_fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             f'configs_{host}_AD_optimize.yaml')
    with open(config_fn, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    return config_fn


def init_pebsi(config_fn):
    """
    Initializes PEBSI and runs the spin-up once, so every optimization step
    starts from the same spun-up state instead of repeating it. The spin-up
    uses the baseline (kp, wind_factor) and is not differentiated through.
    """
    # get_args parses sys.argv, which holds this script's flags, not PEBSI's
    args = sim.get_args(parse=False).parse_args([])
    args.config_fn = config_fn
    model = sim.PEBSI(args)

    model.config.static_args = model.config.static_args._replace(
        store_vars=STORE_VARS, differentiable=True)
    model.config.params.store_vars = STORE_VARS
    model.initialize()

    print('Running spinup...', flush=True)
    model.initial_state = model.spinup(model.initial_state)
    print('Spinup complete.', flush=True)
    return model


# ---------------------------------------------------------------------------
# 1. Observations, matched onto the model's daily grid and point cloud
# ---------------------------------------------------------------------------

def _to_grid_xy(crs, lon, lat):
    proj = Transformer.from_crs('EPSG:4326', crs, always_xy=True)
    return proj.transform(lon, lat)


def _match_times(obs_times, model_days):
    """
    Maps each observation time onto the model's daily output index,
    dropping observations outside the simulated period. Returns
    (day_idx, keep) where keep selects the surviving observations.
    """
    obs_times = pd.to_datetime(obs_times)
    keep = (obs_times >= model_days[0]) & (obs_times <= model_days[-1] + pd.Timedelta('1d'))
    day_idx = model_days.get_indexer(obs_times[keep].normalize(), method='nearest')
    return day_idx, np.asarray(keep)


def _scatter_to_points(values, point_idx, n_points):
    """
    Places a (T, n_glacier_points) observation block into the full
    (T, N_POINTS) point axis, leaving every other glacier's columns NaN.
    """
    full = np.full((values.shape[0], n_points), np.nan, dtype=np.float32)
    full[:, point_idx] = values
    return full


def load_albedo_obs(name, lon, lat, point_idx, model_days, n_points, use='s2'):
    """
    Remotely sensed albedo, sampled at the nearest raster cell to every
    point of this glacier and matched to the nearest model day. Mirrors
    Albedo.get_model_albedo's selection, but returns index arrays instead
    of a matched model array, since the model side only exists inside the
    differentiated graph.
    """
    ab = Albedo(name, use=use)
    x, y = _to_grid_xy(ab.crs, lon[point_idx], lat[point_idx])
    measured = (ab.ds_meas
                .sel(x=xr.DataArray(x, dims='point'),
                     y=xr.DataArray(y, dims='point'),
                     method='nearest')
                .drop_duplicates('time'))
    in_season = np.isin(pd.to_datetime(measured.time.values).month, ALBEDO_MONTHS)
    measured = measured.isel(time=np.where(in_season)[0])

    day_idx, keep = _match_times(measured.time.values, model_days)
    values = measured['albedo'].transpose('time', 'point').values[keep]
    return day_idx, _scatter_to_points(values.astype(np.float32), point_idx, n_points)


def load_snow_melt_obs(name, lon, lat, point_idx, model_days, n_points, direction=None):
    """
    SAR snow and melt extent masks, sampled at each point and matched to
    the nearest model day, following SnowlineMelt.get_model_snow. Returns
    (None, None) if this glacier has no SAR cube, so a missing product
    simply drops those two terms from that glacier's objective.
    """
    try:
        sm = SnowlineMelt(name, direction=direction)
    except (FileNotFoundError, OSError) as err:
        print(f'  ! no SAR cube for {name} ({err}); '
              'snow and melt terms dropped for this glacier', flush=True)
        return None, None

    x, y = _to_grid_xy(sm.crs, lon[point_idx], lat[point_idx])
    sel = dict(x=xr.DataArray(x, dims='point'),
               y=xr.DataArray(y, dims='point'), method='nearest')
    meas_snow = sm.ds_meas_snow.sel(**sel)
    meas_melt = sm.ds_meas_melt.sel(**sel)

    out = []
    for meas in (meas_snow, meas_melt):
        day_idx, keep = _match_times(meas.time.values, model_days)
        values = meas.transpose('time', 'point').values[keep].astype(np.float32)
        out.append((day_idx, _scatter_to_points(values, point_idx, n_points)))
    return out[0], out[1]


def build_observations(model, model_days):
    """
    Loads every glacier's observations and returns them keyed on the
    model's own point axis, ready to be sliced per adjoint chunk.
    """
    lon = np.asarray(model.terrain.lon_n)
    lat = np.asarray(model.terrain.lat_n)
    rgiid = np.asarray(model.terrain.rgiid_n)
    n_points = len(lon)

    obs = {'albedo': [], 'snow': [], 'melt': [],
           'mb_meas': [], 'mb_sigma': [], 'point_idx': []}

    for name in GLACIERS:
        point_idx = np.where(rgiid == translate_rgi[name]['6'])[0]
        assert len(point_idx) > 0, f'no simulation points landed on {name}'
        obs['point_idx'].append(point_idx)

        obs['albedo'].append(
            load_albedo_obs(name, lon, lat, point_idx, model_days, n_points))
        snow, melt = load_snow_melt_obs(name, lon, lat, point_idx, model_days, n_points)
        obs['snow'].append(snow)
        obs['melt'].append(melt)

        mb = MassBalance(name, dates=(MB_START, MB_END))
        obs['mb_meas'].append(float(mb.meas))
        obs['mb_sigma'].append(float(mb.sigma))

        n_alb = 0 if obs['albedo'][-1] is None else len(obs['albedo'][-1][0])
        n_sar = 0 if snow is None else len(snow[0])
        print(f'  {name:<12} {len(point_idx):>5} points  '
              f'{n_alb:>5} albedo scenes  {n_sar:>5} SAR scenes  '
              f'MB {mb.meas:+.2f} +/- {mb.sigma:.2f} m w.e.', flush=True)

    return obs


# ---------------------------------------------------------------------------
# 2. Per-chunk observation blocks
# ---------------------------------------------------------------------------

def _pack_metric(entries, chunk_bounds, n_points):
    """
    Splits one metric's (day_idx, values) pairs -- one per glacier, some
    possibly None -- into per-chunk arrays padded to a common width so
    every chunk compiles to the same shapes.

    Returns (idx, meas, mask, counts): idx/meas/mask are lists (one per
    chunk) of (n_glaciers, K, N_POINTS) arrays holding the day index
    within that chunk, the observed value, and whether the entry is real;
    counts is the (n_glaciers,) total number of valid observed values,
    which turns the accumulated sums into means.
    """
    n_glaciers = len(entries)
    n_chunks = len(chunk_bounds)

    # per (glacier, chunk) selection of observations falling in that chunk
    selections = [[None] * n_chunks for _ in range(n_glaciers)]
    for g, entry in enumerate(entries):
        if entry is None:
            continue
        day_idx, values = entry
        for c, (d0, d1) in enumerate(chunk_bounds):
            sel = np.where((day_idx >= d0) & (day_idx < d1))[0]
            selections[g][c] = (day_idx[sel] - d0, values[sel])

    width = max((len(s[0]) for row in selections for s in row if s is not None),
                default=0)
    width = max(width, 1)  # keep a well-formed (all-masked) block if empty

    counts = np.zeros(n_glaciers, dtype=np.int64)
    idx_chunks, meas_chunks, mask_chunks = [], [], []
    for c in range(n_chunks):
        idx = np.zeros((n_glaciers, width), dtype=np.int32)
        meas = np.zeros((n_glaciers, width, n_points), dtype=np.float32)
        mask = np.zeros((n_glaciers, width, n_points), dtype=bool)
        for g in range(n_glaciers):
            s = selections[g][c]
            if s is None or len(s[0]) == 0:
                continue
            n = len(s[0])
            idx[g, :n] = s[0]
            valid = np.isfinite(s[1])
            meas[g, :n] = np.where(valid, s[1], 0.0)
            mask[g, :n] = valid
            counts[g] += int(valid.sum())
        idx_chunks.append(idx)
        meas_chunks.append(meas)
        mask_chunks.append(mask)

    return idx_chunks, meas_chunks, mask_chunks, counts


def build_chunk_data(model, obs, model_days, chunk_bounds):
    """
    Packs, once per run, everything the objective needs: each chunk's
    climate forcings and its slice of every observation.

    The full-length chunks are stacked along a leading chunk axis for
    lax.scan. A final short chunk, if the run does not divide evenly, is
    returned separately and run outside the scan, so chunk length can be
    chosen for the device's memory rather than for the run's factorization.
    """
    n_points = model.terrain.N_POINTS

    alb = _pack_metric(obs['albedo'], chunk_bounds, n_points)
    snow = _pack_metric(obs['snow'], chunk_bounds, n_points)
    melt = _pack_metric(obs['melt'], chunk_bounds, n_points)

    # days inside the Hugonnet period contribute to the mass balance total
    in_mb_period = ((model_days >= pd.to_datetime(MB_START))
                    & (model_days < pd.to_datetime(MB_END))).astype(np.float32)

    forcings, obs_chunks = [], []
    for c, (d0, d1) in enumerate(chunk_bounds):
        chunk_dates = model.dates[d0 * 24:d1 * 24]
        forcings.append(model.pack_forcings(model.params, chunk_dates, d0 * 24))
        obs_chunks.append(dict(
            alb=(alb[0][c], alb[1][c], alb[2][c]),
            snow=(snow[0][c], snow[1][c], snow[2][c]),
            melt=(melt[0][c], melt[1][c], melt[2][c]),
            mb_weight=in_mb_period[d0:d1],
        ))
        print(f'\033[2K\r~ Packing forcings [{c + 1}/{len(chunk_bounds)}] ~',
              end='', flush=True)
    print(flush=True)

    full = max(d1 - d0 for d0, d1 in chunk_bounds)
    n_full = sum(1 for d0, d1 in chunk_bounds if d1 - d0 == full)
    assert all(d1 - d0 == full for d0, d1 in chunk_bounds[:n_full]), \
        'only the final chunk may be short'

    stack = lambda parts: jax.tree.map(lambda *xs: jnp.stack(xs), *parts)
    stacked = (stack(forcings[:n_full]), stack(obs_chunks[:n_full]))
    tail = None if n_full == len(chunk_bounds) else (forcings[-1], obs_chunks[-1])

    # release the per-chunk arrays; only the stacked copies and the tail
    # are needed from here on
    del forcings[:n_full], obs_chunks[:n_full]

    for label, tree in (('forcings', stacked[0]), ('observations', stacked[1])):
        nbytes = sum(x.nbytes for x in jax.tree.leaves(tree))
        print(f'  stacked {label}: {nbytes / 1e9:.2f} GB', flush=True)

    counts = dict(albedo=alb[3], snow=snow[3], melt=melt[3])
    return stacked, tail, counts


# ---------------------------------------------------------------------------
# 3. The differentiated objective
# ---------------------------------------------------------------------------

def make_loss_fn(model, obs, counts, stacked, tail):
    """
    Builds the value-and-gradient of the whole run with respect to the
    per-glacier log(kp) and log(wind_factor).

    The full-length temporal chunks are walked by one lax.scan, carrying
    the glacier state and a set of running sums the four metrics are built
    from; a short trailing chunk, if there is one, runs after it. Each
    chunk's forward pass is wrapped in jax.checkpoint, so reverse mode
    keeps only the state at the chunk boundaries and rematerializes a
    chunk's interior when it reaches it -- without that, the scan would
    retain every chunk's full inner graph at once.

    The forcings, observations and initial state are passed as ARGUMENTS
    to the jitted function rather than captured in its closure. Captured
    arrays are constants: jax would fold all ~5 GB of them into the
    lowered HLO, serializing a second host-side copy at lowering time on
    top of the numpy originals and the device arrays -- enough to get the
    job killed by the cgroup OOM handler before a single step ran.
    """
    static_args = model.config.static_args
    dynamic_args = model.config.dynamic_args
    point_attrs = model.point_attrs
    n_points = model.terrain.N_POINTS
    n_glaciers = len(GLACIERS)

    # (N_POINTS,) glacier index, mapping each glacier's scalar parameters
    # onto the per-point arrays PEBSI expects
    glacier_of_point = np.zeros(n_points, dtype=np.int32)
    for g, point_idx in enumerate(obs['point_idx']):
        glacier_of_point[point_idx] = g
    glacier_of_point = jnp.asarray(glacier_of_point)

    # (n_glaciers, N_POINTS) membership, for averaging point mass balance
    # into a glacier-wide value
    member = np.zeros((n_glaciers, n_points), dtype=np.float64)
    for g, point_idx in enumerate(obs['point_idx']):
        member[g, point_idx] = 1.0 / len(point_idx)
    member = jnp.asarray(member)

    mb_meas = jnp.asarray(obs['mb_meas'])
    mb_sigma = jnp.asarray(obs['mb_sigma'])
    count = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in counts.items()}

    def init_acc():
        return dict(
            albedo_ssr=jnp.zeros(n_glaciers),   # sum of squared albedo residuals
            snow_nll=jnp.zeros(n_glaciers),     # summed Bernoulli log-loss
            melt_nll=jnp.zeros(n_glaciers),
            mb=jnp.zeros(n_points),             # per-point mass balance total
        )

    def _bernoulli(prob, meas, mask):
        p = jnp.clip(prob, BERNOULLI_EPS, 1 - BERNOULLI_EPS)
        loss = -(meas * jnp.log(p) + (1 - meas) * jnp.log(1 - p))
        return jnp.sum(jnp.where(mask, loss, 0.0), axis=(1, 2))

    def accumulate(acc, records, obs_chunk):
        alb_idx, alb_meas, alb_mask = obs_chunk['alb']
        snow_idx, snow_meas, snow_mask = obs_chunk['snow']
        melt_idx, melt_meas, melt_mask = obs_chunk['melt']

        # (n_glaciers, K, N_POINTS) gathers of the model's daily fields at
        # each glacier's observation days
        resid = records.albedo[alb_idx] - alb_meas
        albedo_ssr = jnp.sum(jnp.where(alb_mask, resid ** 2, 0.0), axis=(1, 2))

        # the hard indicators (snow+firn depth > 0.05 m, column water >
        # 0.05 kg m-2) are step functions with no gradient, so both are
        # relaxed to a logistic about the same threshold
        p_snow = jax.nn.sigmoid(
            (records.snowdepth[snow_idx] - SNOW_DEPTH_THRESHOLD) / SNOW_DEPTH_WIDTH)
        p_melt = jax.nn.sigmoid(
            (records.total_water[melt_idx] - MELT_WATER_THRESHOLD) / MELT_WATER_WIDTH)

        return dict(
            albedo_ssr=acc['albedo_ssr'] + albedo_ssr,
            snow_nll=acc['snow_nll'] + _bernoulli(p_snow, snow_meas, snow_mask),
            melt_nll=acc['melt_nll'] + _bernoulli(p_melt, melt_meas, melt_mask),
            mb=acc['mb'] + jnp.sum(
                records.mass_balance * obs_chunk['mb_weight'][:, None], axis=0),
        )

    def finalize(acc):
        """
        Turns the accumulated sums into the four glacier-wide metrics, each
        a negative log-likelihood in nats so they add without weighting --
        the same construction the glacierwide_loss metrics use. A metric
        with no observations for a glacier (a missing SAR cube, say) has
        count 0 and drops out of that glacier's objective.
        """
        def per_obs(total, n):
            return jnp.where(n > 0, total / jnp.where(n > 0, n, 1.0), 0.0)

        albedo = jnp.where(
            count['albedo'] > 0,
            0.5 * jnp.log(2 * jnp.pi * ALBEDO_SIGMA ** 2)
            + per_obs(acc['albedo_ssr'], count['albedo']) / (2 * ALBEDO_SIGMA ** 2),
            0.0)
        snow = per_obs(acc['snow_nll'], count['snow'])
        melt = per_obs(acc['melt_nll'], count['melt'])

        mb_mod = member @ acc['mb']
        mb = (0.5 * jnp.log(2 * jnp.pi * mb_sigma ** 2)
              + (mb_mod - mb_meas) ** 2 / (2 * mb_sigma ** 2))

        total = jnp.sum(albedo + snow + melt + mb)
        return total, dict(albedo=albedo, snow=snow, melt=melt, mb=mb, mb_mod=mb_mod)

    # static_args and point_attrs stay in the closure: jax.checkpoint only
    # sees JAX-traceable arguments
    def _run_chunk(state, forcings, new_dynamic_args):
        return pebsi_main(state, forcings, point_attrs, static_args,
                          new_dynamic_args)

    def loss_fn(log_kp, log_wind_factor, initial_state, stacked, tail):
        stacked_forcings, stacked_obs = stacked
        kp = jnp.exp(log_kp)[glacier_of_point]
        wind_factor = jnp.exp(log_wind_factor)[glacier_of_point]
        new_dynamic_args = dynamic_args._replace(kp=kp, wind_factor=wind_factor)

        def scan_chunk(carry, xs):
            state, acc = carry
            forcings, obs_chunk = xs
            state, records = jax.checkpoint(_run_chunk)(
                state, forcings, new_dynamic_args)
            return (state, accumulate(acc, records, obs_chunk)), None

        carry, _ = jax.lax.scan(
            scan_chunk, (initial_state, init_acc()),
            (stacked_forcings, stacked_obs))
        if tail is not None:
            carry, _ = scan_chunk(carry, tail)
        return finalize(carry[1])

    initial_state = model.initial_state
    jitted = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True))

    # plain python wrapper, so the big arrays reach the jitted function as
    # arguments without the optimizer loop having to carry them around
    def value_and_grad(log_kp, log_wind_factor):
        return jitted(log_kp, log_wind_factor, initial_state, stacked, tail)

    return value_and_grad


# ---------------------------------------------------------------------------
# 4. Optimization
# ---------------------------------------------------------------------------

def run_optimization(value_and_grad, init_kp, init_wind_factor,
                     n_steps=30, lr=5e-2, clip_norm=1.0):
    """
    Adam on log(kp) and log(wind_factor), one pair per glacier, so both stay
    positive without a constrained optimizer.

    clip_by_global_norm caps the raw gradient before adam sees it. A single
    glacier landing in a steep region of the loss surface can otherwise move
    the whole trajectory; clip_norm=1.0 sits well above a normal step's norm,
    so it only engages on a genuine spike.
    """
    params = {
        'log_kp': jnp.log(jnp.asarray(init_kp, dtype=jnp.float64)),
        'log_wind_factor': jnp.log(jnp.asarray(init_wind_factor, dtype=jnp.float64)),
    }
    optimizer = optax.chain(optax.clip_by_global_norm(clip_norm), optax.adam(lr))
    opt_state = optimizer.init(params)

    header = (f"{'Step':>4}  {'Total':>12}  {'Albedo':>10}  {'Snow':>10}  "
              f"{'Melt':>10}  {'MB':>10}  {'|grad|':>10}")
    print(header, flush=True)

    history = []
    for i in range(n_steps):
        t0 = time.time()
        (total, metrics), (d_kp, d_wf) = value_and_grad(
            params['log_kp'], params['log_wind_factor'])
        grads = {'log_kp': d_kp, 'log_wind_factor': d_wf}
        jax.block_until_ready((total, grads))

        bad = {k: np.where(~np.isfinite(np.asarray(v)))[0] for k, v in grads.items()}
        if any(len(v) for v in bad.values()):
            for key, idx in bad.items():
                if len(idx):
                    print(f'  step {i}: non-finite {key} for '
                          f'{[GLACIERS[j] for j in idx]}', flush=True)
            print('  Stopping: non-finite gradient detected.', flush=True)
            break

        raw_norm = float(optax.global_norm(grads))
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        print(f'{i:>4}  {float(total):>12.2f}  '
              f"{float(jnp.sum(metrics['albedo'])):>10.3f}  "
              f"{float(jnp.sum(metrics['snow'])):>10.3f}  "
              f"{float(jnp.sum(metrics['melt'])):>10.3f}  "
              f"{float(jnp.sum(metrics['mb'])):>10.3f}  "
              f'{raw_norm:>10.3e}  ({time.time() - t0:.1f}s)', flush=True)
        if raw_norm > clip_norm:
            print(f'    clipped: raw global grad norm {raw_norm:.3e} -> {clip_norm}',
                  flush=True)

        history.append(dict(
            step=i, total=float(total),
            kp=np.exp(np.asarray(params['log_kp'])).tolist(),
            wind_factor=np.exp(np.asarray(params['log_wind_factor'])).tolist(),
        ))

    return (np.exp(np.asarray(params['log_kp'])),
            np.exp(np.asarray(params['log_wind_factor'])), history)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-n', '--n_steps', type=int, default=30)
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-2)
    parser.add_argument('--chunk_days', type=int, default=CHUNK_DAYS)
    opts = parser.parse_args()

    print(f'JAX backend: {jax.default_backend()}  devices: {jax.devices()}',
          flush=True)

    config_fn = build_config()
    print(f'Config -> {config_fn}', flush=True)

    model = init_pebsi(config_fn)
    model_days = model.dates[::24]
    n_days = len(model_days)
    print(f'{model.terrain.N_POINTS} points, {n_days} days '
          f'({model.dates[0]} to {model.dates[-1]})', flush=True)

    chunk_starts = list(range(0, n_days, opts.chunk_days))
    chunk_bounds = [(s, min(s + opts.chunk_days, n_days)) for s in chunk_starts]
    remainder = n_days % opts.chunk_days
    print(f'{len(chunk_bounds)} chunks of {opts.chunk_days} days'
          + (f' plus a {remainder}-day tail (one extra compile)' if remainder else '')
          + f'; ~{opts.chunk_days * 10 / 1000:.1f} GB of reverse-pass state',
          flush=True)

    print('Loading observations...', flush=True)
    obs = build_observations(model, model_days)

    print(f'Packing {len(chunk_bounds)} chunks...', flush=True)
    stacked, tail, counts = build_chunk_data(model, obs, model_days, chunk_bounds)
    for metric, c in counts.items():
        print(f'  {metric:<7} observations per glacier: {c.tolist()}', flush=True)

    value_and_grad = make_loss_fn(model, obs, counts, stacked, tail)

    n_glaciers = len(GLACIERS)
    kp, wind_factor, history = run_optimization(
        value_and_grad,
        init_kp=np.full(n_glaciers, baseline['kp']),
        init_wind_factor=np.full(n_glaciers, baseline['wind_factor']),
        n_steps=opts.n_steps, lr=opts.learning_rate)

    print('\nOptimized parameters:')
    print(f"{'Glacier':<14} {'kp':>10} {'wind_factor':>13}")
    for name, k, w in zip(GLACIERS, kp, wind_factor):
        print(f'{name:<14} {k:>10.4f} {w:>13.4f}')

    out_fn = os.path.join(HOST_PATHS[host]['output_fp'], 'AD_optimize_result.yaml')
    os.makedirs(os.path.dirname(out_fn), exist_ok=True)
    with open(out_fn, 'w') as f:
        yaml.dump(dict(glaciers=GLACIERS,
                       kp=[float(v) for v in kp],
                       wind_factor=[float(v) for v in wind_factor],
                       history=history), f, sort_keys=False)
    print(f'\nWrote {out_fn}')


if __name__ == '__main__':
    main()
