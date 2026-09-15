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
  - Hugonnet glacier-wide balance  (Gaussian NLL on the run's cumulative total)

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
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# netCDF4 must load before JAX to avoid a numpy ABI conflict
import simulation as sim
from pebsi.main import main as pebsi_main

import jax
jax.config.update("jax_traceback_filtering", "off")
if os.environ.get('PEBSI_DEBUG_NANS', '0') != '0':
    jax.config.update("jax_debug_nans", True)
    jax.config.update("jax_debug_infs", True)
import jax.numpy as jnp
import optax
import numpy as np
import pandas as pd
from scipy.optimize import minimize, BFGS, SR1
import xarray as xr
import yaml
from pyproj import Transformer

from project.glacierwide_loss import Albedo, SnowlineMelt, MassBalance, translate_rgi

# ---------------------------------------------------------------------------
# Run settings
# ---------------------------------------------------------------------------

GLACIERS = ['gulkana', 'kahiltna', 'kennicott', 'wolverine', 'lemon_creek'] 

START_DATE = '2000-01-01 00:00'
END_DATE = '2025-01-01 23:00'

# days per temporal chunk to set peak device memory
# on a 32 GB card: 1 year ~ 10 GB total, 3 years ~ 17 GB, 5 years ~ 24 GB
CHUNK_DAYS = 365

# only these are needed by the four metrics; every extra field is carried
# through the whole differentiated graph, so the list is kept minimal
STORE_VARS = ('mass_balance', 'albedo', 'total_water', 'snowdepth')

# observation uncertainty
ALBEDO_SIGMA = 0.07770
BERNOULLI_EPS = 1e-3

# threshold for model output to consider snow=True and melt=True
SNOW_DEPTH_THRESHOLD = 0.05
MELT_WATER_THRESHOLD = 0.05

# width of logistic function used to smooth snow/melt thresholds
SNOW_DEPTH_WIDTH = 0.02
MELT_WATER_WIDTH = 0.02

# months of the year the albedo product is scored over
ALBEDO_MONTHS = list(range(3, 10))

ALBEDO_USE_DELTAS = True
ALBEDO_BASELINE_MONTH = 3

# starting point
baseline = {'kp': 2.5, 'wind_factor': 2.5}

from host_paths import host, HOST_PATHS

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
    max_nlayers=25,
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

def _tick(label, t0=None):
    """Timestamped progress print for locating where wall time goes;
    pass the previous _tick's return value as t0 to also print elapsed."""
    now = time.time()
    suffix = f' ({now - t0:.1f}s)' if t0 is not None else ''
    print(f'[{time.strftime("%H:%M:%S")}] {label}{suffix}', flush=True)
    return now


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
    keep = (obs_times >= model_days[0]) & (obs_times <= model_days[-1] + pd.Timedelta('1D'))
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
    times = pd.to_datetime(measured.time.values)[keep]
    return (day_idx,
            _scatter_to_points(values.astype(np.float32), point_idx, n_points),
            np.asarray(times.year), np.asarray(times.month))


def _albedo_entry(name, lon, lat, point_idx, model_days, n_points,
                  year_pos, meas_march):
    """
    Loads one glacier's albedo observations and, when ALBEDO_USE_DELTAS is
    on, works out the MEASURED per-year, per-point March baseline they are
    scored relative to -- the observation half of
    Albedo.get_deltas(method='march_mean'). Writes that baseline into the
    shared (n_years, N_POINTS) `meas_march` array in place.

    A year with no March scene at a point inherits that point's mean over
    the years that do have one, exactly as get_deltas fills its NaNs. If a
    point has no March scene in ANY year the baseline stays NaN, and every
    observation at that point is marked invalid here rather than later:
    get_deltas would hand those on as NaN deltas for a downstream nanmean
    to skip, and dropping them at the mask is the same thing, but it also
    keeps NaN out of the differentiated graph entirely.

    Returns (day_idx, values, year_idx) -- the third element is each
    surviving observation's position on the run's year axis.
    """
    day_idx, values, obs_years, obs_months = load_albedo_obs(
        name, lon, lat, point_idx, model_days, n_points)

    year_idx = np.array([year_pos[int(y)] for y in obs_years], dtype=np.int32)
    if not ALBEDO_USE_DELTAS:
        return day_idx, values, year_idx

    block = values[:, point_idx]                        # (T, n_glacier_points)
    baseline = np.full((len(year_pos), len(point_idx)), np.nan, dtype=np.float64)
    is_march = obs_months == ALBEDO_BASELINE_MONTH
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN slices
        for y, i in year_pos.items():
            sel = np.where(is_march & (obs_years == y))[0]
            if len(sel):
                baseline[i] = np.nanmean(block[sel], axis=0)

        across = np.nanmean(baseline, axis=0)            # (n_glacier_points,)
    missing = np.isnan(baseline)
    baseline[missing] = np.broadcast_to(across, baseline.shape)[missing]
    meas_march[:, point_idx] = baseline

    # an observation whose baseline is undefined cannot form a delta
    usable = np.isfinite(baseline)[year_idx]             # (T, n_glacier_points)
    orphan = ~np.isfinite(across)
    if orphan.any():
        lost = int(np.sum(np.isfinite(block) & ~usable))
        print(f'  ! {name}: {orphan.sum()} of {len(point_idx)} points have no '
              f'March scene in any year, so no delta baseline; '
              f'{lost} of {int(np.isfinite(block).sum())} albedo observations '
              'dropped', flush=True)
    values[:, point_idx] = np.where(usable, block, np.nan)
    return day_idx, values, year_idx


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
           'mb_meas': [], 'mb_sigma': [], 'mb_end_day': [], 'point_idx': []}

    # year axis the March baselines live on, one entry per calendar year the
    # run touches. Observations are matched onto model days first, so their
    # years are a subset of these.
    years = np.unique(model_days.year)
    year_pos = {int(y): i for i, y in enumerate(years)}
    obs['years'] = years
    obs['meas_march'] = np.full((len(years), n_points), np.nan, dtype=np.float64)

    for name in GLACIERS:
        point_idx = np.where(rgiid == translate_rgi[name]['6'])[0]
        assert len(point_idx) > 0, f'no simulation points landed on {name}'
        obs['point_idx'].append(point_idx)

        obs['albedo'].append(
            _albedo_entry(name, lon, lat, point_idx, model_days, n_points,
                          year_pos, obs['meas_march']))
        snow, melt = load_snow_melt_obs(name, lon, lat, point_idx, model_days, n_points)
        obs['snow'].append(snow)
        obs['melt'].append(melt)

        mb = MassBalance(name, dates=(START_DATE, END_DATE))
        obs['mb_meas'].append(float(mb.meas))
        obs['mb_sigma'].append(float(mb.sigma))

        mb_end_day = int(model_days.get_indexer([mb.matched_end], method='nearest')[0])
        obs['mb_end_day'].append(mb_end_day)
        truncated = mb.matched_end != pd.to_datetime(END_DATE.split()[0])
        mb_note = (f'  ! MB truncated to {mb.matched_start.date()} -> '
                   f'{mb.matched_end.date()} (data ends there)' if truncated else '')

        n_alb = 0 if obs['albedo'][-1] is None else len(obs['albedo'][-1][0])
        n_sar = 0 if snow is None else len(snow[0])
        print(f'  {name:<12} {len(point_idx):>5} points  '
              f'{n_alb:>5} albedo scenes  {n_sar:>5} SAR scenes  '
              f'MB {mb.meas:+.2f} +/- {mb.sigma:.2f} m w.e.{mb_note}', flush=True)

    return obs


# ---------------------------------------------------------------------------
# 2. Per-chunk observation blocks
# ---------------------------------------------------------------------------

def _pack_metric(entries, chunk_bounds, n_points):
    """
    Splits one metric's (day_idx, values) pairs -- one per glacier, some
    possibly None -- into per-chunk arrays padded to a common width so
    every chunk compiles to the same shapes.

    An entry may carry a third element, a per-observation integer label
    (albedo's year index); it is packed alongside as a (n_glaciers, K)
    array so the graph can group observations by year.

    Returns (idx, meas, mask, tag, counts): idx/meas/mask are lists (one
    per chunk) of (n_glaciers, K, N_POINTS) arrays holding the day index
    within that chunk, the observed value, and whether the entry is real;
    tag is the matching list of (n_glaciers, K) label arrays, or None if
    no entry carried labels; counts is the (n_glaciers,) total number of
    valid observed values, which turns the accumulated sums into means.
    """
    n_glaciers = len(entries)
    n_chunks = len(chunk_bounds)
    tagged = any(e is not None and len(e) > 2 for e in entries)

    # per (glacier, chunk) selection of observations falling in that chunk
    selections = [[None] * n_chunks for _ in range(n_glaciers)]
    for g, entry in enumerate(entries):
        if entry is None:
            continue
        day_idx, values = entry[0], entry[1]
        labels = entry[2] if tagged else None
        for c, (d0, d1) in enumerate(chunk_bounds):
            sel = np.where((day_idx >= d0) & (day_idx < d1))[0]
            selections[g][c] = (day_idx[sel] - d0, values[sel],
                                None if labels is None else labels[sel])

    width = max((len(s[0]) for row in selections for s in row if s is not None),
                default=0)
    width = max(width, 1)  # keep a well-formed (all-masked) block if empty

    counts = np.zeros(n_glaciers, dtype=np.int64)
    idx_chunks, meas_chunks, mask_chunks, tag_chunks = [], [], [], []
    for c in range(n_chunks):
        idx = np.zeros((n_glaciers, width), dtype=np.int32)
        meas = np.zeros((n_glaciers, width, n_points), dtype=np.float32)
        mask = np.zeros((n_glaciers, width, n_points), dtype=bool)
        tag = np.zeros((n_glaciers, width), dtype=np.int32)
        for g in range(n_glaciers):
            s = selections[g][c]
            if s is None or len(s[0]) == 0:
                continue
            n = len(s[0])
            idx[g, :n] = s[0]
            valid = np.isfinite(s[1])
            meas[g, :n] = np.where(valid, s[1], 0.0)
            mask[g, :n] = valid
            if s[2] is not None:
                tag[g, :n] = s[2]
            counts[g] += int(valid.sum())
        idx_chunks.append(idx)
        meas_chunks.append(meas)
        mask_chunks.append(mask)
        tag_chunks.append(tag)

    return (idx_chunks, meas_chunks, mask_chunks,
            tag_chunks if tagged else None, counts)


def _march_baseline(obs, model_days, chunk_bounds):
    """
    Everything the March-referenced albedo delta needs that is a constant
    of the run, plus the per-chunk selector that lets the MODEL half of the
    baseline be accumulated as the forward pass goes.

    The model's baseline is parameter-dependent, so it cannot be computed
    up front the way get_deltas does from a finished run -- and it must not
    cost a second pass, since a pass is the expensive thing here. Instead
    the residual is expanded so a single chunked pass suffices. Writing
    e_t = mod_t - meas_t and E = mod_march - meas_march, the delta residual
    is just e_t - E[year, point], so

        SSR = sum_t mask (e_t - E)^2
            = sum_t mask e_t^2  -  2 sum_yp S1[y,p] E[y,p]
              + sum_yp N[y,p] E[y,p]^2

    with S1[y,p] the sum of masked residuals in that year at that point and
    N[y,p] their count. S1 and mod_march accumulate additively over chunks
    exactly like the raw sum of squares already does, and the correction is
    applied once in finalize.

    Returns:
      weights  per-chunk (n_years, chunk_days) selector, 1 on the model
               days inside that year's March window (get_deltas slices
               MAR-01 to APR-01 inclusive, so Apr 1 is included). A year
               with no model day in March PROPER is zeroed out rather than
               left to average its lone Apr 1 -- a run starting Apr 1
               would otherwise get a one-day "March" mean instead of
               falling through to the across-year fill.
      count    (n_years,) model days per March window, over the whole run
      have     (n_years,) whether that year has a usable March window
      meas     (n_years, N_POINTS) measured baseline, NaN replaced by 0 --
               every entry it is missing at has N == 0 and S1 == 0, so the
               placeholder is multiplied by exact zeros. The value is
               neutralized BEFORE it enters the graph rather than the term
               masked after, since 0 * NaN is NaN in a gradient.
      n        (n_glaciers, n_years, N_POINTS) count of valid observations
    """
    years = obs['years']
    n_years = len(years)
    n_points = obs['meas_march'].shape[1]

    in_month = model_days.month == ALBEDO_BASELINE_MONTH
    in_march = in_month | ((model_days.month == 4) & (model_days.day == 1))
    year_of_day = np.array([np.searchsorted(years, y) for y in model_days.year])

    has_march = np.zeros(n_years, dtype=bool)
    np.logical_or.at(has_march, year_of_day, np.asarray(in_month))

    weights = []
    count = np.zeros(n_years)
    for d0, d1 in chunk_bounds:
        w = np.zeros((n_years, d1 - d0), dtype=np.float64)
        sel = np.where(np.asarray(in_march)[d0:d1])[0]
        w[year_of_day[d0:d1][sel], sel] = 1.0
        w[~has_march] = 0.0
        count += w.sum(axis=1)
        weights.append(w)

    if count.sum() == 0:
        raise ValueError(
            f'ALBEDO_USE_DELTAS is on but {START_DATE} to {END_DATE} contains '
            f'no model days in month {ALBEDO_BASELINE_MONTH}, so the March '
            'baseline the deltas are referenced to is undefined. Widen the '
            'window to include a March, or set ALBEDO_USE_DELTAS = False to '
            'score raw albedo residuals.')

    n = np.zeros((len(GLACIERS), n_years, n_points))
    for g, entry in enumerate(obs['albedo']):
        if entry is None:
            continue
        _, values, year_idx = entry
        np.add.at(n[g], year_idx, np.isfinite(values).astype(np.float64))

    return dict(weights=weights, count=count, have=count > 0,
                meas=np.nan_to_num(obs['meas_march']), n=n)


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
    n_days = len(model_days)

    alb = _pack_metric(obs['albedo'], chunk_bounds, n_points)
    snow = _pack_metric(obs['snow'], chunk_bounds, n_points)
    melt = _pack_metric(obs['melt'], chunk_bounds, n_points)

    march = _march_baseline(obs, model_days, chunk_bounds) \
        if ALBEDO_USE_DELTAS else None

    mb_end_day = np.asarray(obs['mb_end_day'])
    mb_valid_full = np.arange(n_days)[None, :] <= mb_end_day[:, None]

    forcings, obs_chunks = [], []
    for c, (d0, d1) in enumerate(chunk_bounds):
        chunk_dates = model.dates[d0 * 24:d1 * 24]
        forcings.append(model.pack_forcings(model.params, chunk_dates, d0 * 24))
        chunk_obs = dict(
            alb=(alb[0][c], alb[1][c], alb[2][c], alb[3][c]),
            snow=(snow[0][c], snow[1][c], snow[2][c]),
            melt=(melt[0][c], melt[1][c], melt[2][c]),
            mb_valid=mb_valid_full[:, d0:d1],
        )
        if march is not None:
            chunk_obs['march_w'] = march['weights'][c]
        obs_chunks.append(chunk_obs)
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

    counts = dict(albedo=alb[4], snow=snow[4], melt=melt[4])
    return stacked, tail, counts, march


# ---------------------------------------------------------------------------
# 3. The differentiated objective
# ---------------------------------------------------------------------------

def make_loss_fn(model, obs, counts, march, stacked, tail):
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
    # into a glacier-wide value.
    #
    # The weights are each point's Voronoi cell area clipped to the glacier
    # outline, normalized to sum to 1 within the glacier
    # (Terrain.voronoi_weights, reached via adaptive_points -> _grid_polygon).
    # The adaptive mesh is an even lattice clipped to the outline, so interior
    # cells are all the same size and the weights only depart from uniform at
    # the margin, where cells are cut by the boundary. Those margin points are
    # a large fraction of a small glacier's mesh and carry its most extreme
    # mass balance, and the Hugonnet target they are compared against is a
    # genuine area-weighted mean over uniform-area pixels (see
    # MassBalance._load_from_tif), so a flat 1/N average here would be
    # measuring a different quantity than the observation.
    weight_n = model.terrain.weight_n
    assert weight_n is not None, (
        f'method_distribute={model.config.params.method_distribute!r} produced '
        'no Voronoi weights; the glacier-wide mass balance needs area weights')
    weight_n = np.asarray(weight_n, dtype=np.float64)

    member = np.zeros((n_glaciers, n_points), dtype=np.float64)
    for g, point_idx in enumerate(obs['point_idx']):
        w = weight_n[point_idx]
        total = w.sum()
        assert np.isclose(total, 1.0, atol=1e-6), (
            f"{GLACIERS[g]}'s Voronoi weights sum to {total:.6f}, not 1")
        member[g, point_idx] = w
    member = jnp.asarray(member)

    mb_meas = jnp.asarray(obs['mb_meas'])
    mb_sigma = jnp.asarray(obs['mb_sigma'])
    count = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in counts.items()}

    # fields to be checked directly for non-finite values anywhere
    RAW_FIELDS = ('albedo', 'snowdepth', 'total_water', 'mass_balance')

    # March-referenced albedo deltas: constants of the run, and the two
    # extra running sums the correction is assembled from
    n_years = 0 if march is None else len(march['count'])
    if march is not None:
        march_count = jnp.asarray(march['count'])
        march_have = jnp.asarray(march['have'])
        meas_march = jnp.asarray(march['meas'])
        march_n = jnp.asarray(march['n'])

    def init_acc():
        acc = dict(
            albedo_ssr=jnp.zeros(n_glaciers),   # sum of squared albedo residuals
            snow_nll=jnp.zeros(n_glaciers),     # summed Bernoulli log-loss
            melt_nll=jnp.zeros(n_glaciers),
            mb=jnp.zeros(n_points),             # per-point mass balance total
            raw_nonfinite={f: jnp.array(False) for f in RAW_FIELDS},
        )
        if march is not None:
            acc['alb_resid'] = jnp.zeros((n_glaciers, n_years, n_points))
            acc['march_sum'] = jnp.zeros((n_years, n_points))
        return acc

    def _bernoulli(prob, meas, mask):
        p = jnp.clip(prob, BERNOULLI_EPS, 1 - BERNOULLI_EPS)
        loss = -(meas * jnp.log(p) + (1 - meas) * jnp.log(1 - p))
        return jnp.sum(jnp.where(mask, loss, 0.0), axis=(1, 2))

    def accumulate(acc, records, obs_chunk):
        alb_idx, alb_meas, alb_mask, alb_year = obs_chunk['alb']
        mb_valid = obs_chunk['mb_valid']
        snow_idx, snow_meas, snow_mask = obs_chunk['snow']
        melt_idx, melt_meas, melt_mask = obs_chunk['melt']

        # (n_glaciers, K, N_POINTS) gathers of the model's daily fields at
        # each glacier's observation days
        #
        # padding entries (this glacier's own width padded to match whichever
        # glacier has the most albedo scenes) gather day-index 0's REAL
        # model value; if that's ever NaN, masking only the final loss term
        # protects the forward sum but not the gradient -- 2*resid is NaN at
        # a NaN resid, and 0 (mask) * NaN = NaN through the masked branch.
        # So albedo is masked to a neutral placeholder (resid == 0) BEFORE
        # squaring, same pattern as the snow/melt sigmoid inputs below.
        safe_albedo = jnp.where(alb_mask, records.albedo[alb_idx], alb_meas)
        resid = safe_albedo - alb_meas
        albedo_ssr = jnp.sum(jnp.where(alb_mask, resid ** 2, 0.0), axis=(1, 2))

        # relax hard cut-off functions to logistic thresholds for derivation
        safe_snowdepth = jnp.where(snow_mask, records.snowdepth[snow_idx],
                                   SNOW_DEPTH_THRESHOLD)
        safe_water = jnp.where(melt_mask, records.total_water[melt_idx],
                               MELT_WATER_THRESHOLD)
        p_snow = jax.nn.sigmoid((safe_snowdepth - SNOW_DEPTH_THRESHOLD) / SNOW_DEPTH_WIDTH)
        p_melt = jax.nn.sigmoid((safe_water - MELT_WATER_THRESHOLD) / MELT_WATER_WIDTH)

        raw_nonfinite = {
            f: acc['raw_nonfinite'][f] | jnp.any(~jnp.isfinite(getattr(records, f)))
            for f in RAW_FIELDS
        }

        out = dict(
            albedo_ssr=acc['albedo_ssr'] + albedo_ssr,
            snow_nll=acc['snow_nll'] + _bernoulli(p_snow, snow_meas, snow_mask),
            melt_nll=acc['melt_nll'] + _bernoulli(p_melt, melt_meas, melt_mask),
            mb=acc['mb'] + jnp.sum(
                jnp.where(mb_valid[glacier_of_point].T, records.mass_balance, 0.0),
                axis=0),
            raw_nonfinite=raw_nonfinite,
        )

        if march is not None:
            # residuals grouped by the year they fall in, and the model's
            # own March mean, both summed across chunks; the delta
            # correction is applied once, in finalize
            masked_resid = jnp.where(alb_mask, resid, 0.0)
            by_year = jax.nn.one_hot(alb_year, n_years, dtype=masked_resid.dtype)
            out['alb_resid'] = acc['alb_resid'] + jnp.einsum(
                'gky,gkp->gyp', by_year, masked_resid)
            out['march_sum'] = (acc['march_sum']
                                + obs_chunk['march_w'] @ records.albedo)

        return out

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

        albedo_ssr = acc['albedo_ssr']
        if march is not None:
            # model March mean per year; fill years with no March values with avg from other years
            safe_count = jnp.where(march_have, march_count, 1.0)[:, None]
            per_year = acc['march_sum'] / safe_count
            across = (jnp.sum(jnp.where(march_have[:, None], per_year, 0.0), axis=0)
                      / jnp.sum(march_have))
            mod_march = jnp.where(march_have[:, None], per_year, across[None, :])

            # apply the "delta"
            offset = mod_march - meas_march
            albedo_ssr = (albedo_ssr
                          - 2.0 * jnp.sum(acc['alb_resid'] * offset[None], axis=(1, 2))
                          + jnp.sum(march_n * offset[None] ** 2, axis=(1, 2)))

        albedo = jnp.where(
            count['albedo'] > 0,
            0.5 * jnp.log(2 * jnp.pi * ALBEDO_SIGMA ** 2)
            + per_obs(albedo_ssr, count['albedo']) / (2 * ALBEDO_SIGMA ** 2),
            0.0)
        snow = per_obs(acc['snow_nll'], count['snow'])
        melt = per_obs(acc['melt_nll'], count['melt'])

        mb_mod = member @ acc['mb']
        mb = (0.5 * jnp.log(2 * jnp.pi * mb_sigma ** 2)
              + (mb_mod - mb_meas) ** 2 / (2 * mb_sigma ** 2))

        total = jnp.sum(albedo + snow + melt + mb)
        return total, dict(albedo=albedo, snow=snow, melt=melt, mb=mb,
                           mb_mod=mb_mod, acc=acc)

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

    # bare, no-aux, single-pytree-arg wrapper around the SAME loss_fn, for
    # optax.lbfgs's linesearch: it differentiates this function itself
    # (jax.value_and_grad(value_fn)), so it must be a plain scalar function,
    # not the pre-differentiated value_and_grad above. Sharing loss_fn
    # guarantees the two paths compute identical math.
    params_jitted = jax.jit(
        lambda params: loss_fn(params['log_kp'], params['log_wind_factor'],
                               initial_state, stacked, tail)[0])

    return value_and_grad, params_jitted


# ---------------------------------------------------------------------------
# 4. Optimization
# ---------------------------------------------------------------------------

def report_forward_health(step, metrics, model, obs):
    """
    Checks the FORWARD outputs (always computed, whether or not the
    gradient came out finite) for non-finite values, and reports exactly
    where. This is the actual localization step -- a non-finite gradient
    only says a NaN exists somewhere upstream, not where it entered.

    acc['mb'] gets its own per-point check because member @ acc['mb'] is a
    dense matmul over every point: one NaN point anywhere poisons mb_mod,
    and therefore the mb term and its gradient, for EVERY glacier at once
    (0 * nan = nan even at a zero-weighted entry) -- which is exactly the
    all-five-glaciers-at-once failure mode seen in practice. albedo/snow/
    melt are already per-glacier sums, so the finalized values are enough.

    Returns True if nothing came back non-finite.
    """
    healthy = True

    for key in ('albedo', 'snow', 'melt', 'mb'):
        vals = np.asarray(metrics[key])
        bad = np.where(~np.isfinite(vals))[0]
        if len(bad):
            healthy = False
            print(f'  step {step}: non-finite {key} for '
                  f'{[GLACIERS[j] for j in bad]}', flush=True)

    mb_point = np.asarray(metrics['acc']['mb'])
    bad_points = np.where(~np.isfinite(mb_point))[0]
    if len(bad_points):
        healthy = False
        glacier_of_point = np.empty(len(mb_point), dtype=object)
        for name, idx in zip(GLACIERS, obs['point_idx']):
            glacier_of_point[idx] = name
        elev = np.asarray(model.terrain.elev_n)
        lat = np.asarray(model.terrain.lat_n)
        lon = np.asarray(model.terrain.lon_n)
        print(f'  step {step}: non-finite per-point mass_balance at '
              f'{len(bad_points)} of {len(mb_point)} point(s) -- poisons '
              'the mb term for every glacier via the membership matmul',
              flush=True)
        print(f'    {"idx":>6} {"glacier":<12} {"elev [m]":>10} '
              f'{"lat":>9} {"lon":>10}', flush=True)
        for i in bad_points[:10]:
            print(f'    {i:>6} {glacier_of_point[i]:<12} {elev[i]:>10.1f} '
                  f'{lat[i]:>9.4f} {lon[i]:>10.4f}', flush=True)
        if len(bad_points) > 10:
            print(f'    ... and {len(bad_points) - 10} more', flush=True)

    bad_raw = [f for f, v in metrics['acc']['raw_nonfinite'].items() if bool(v)]
    if bad_raw:
        healthy = False
        print(f'  step {step}: non-finite value(s) somewhere in raw '
              f'{bad_raw} -- hidden by masking from the per-glacier sums '
              'above, but can still poison the gradient through a masked '
              'branch', flush=True)

    if healthy:
        print(f'  step {step}: forward metrics AND raw fields all finite -- '
              'the break is in the backward pass only (e.g. a masked '
              'jnp.where branch with an infinite local derivative), not a '
              'bad forward value', flush=True)

    return healthy


def run_optimization(value_and_grad, init_kp, init_wind_factor, model, obs,
                     n_steps=30, lr=5e-2, lr_final_frac=0.1):
    """
    Adam on log(kp) and log(wind_factor), one pair per glacier, so both stay
    positive without a constrained optimizer.

    Adam's own per-parameter normalization keeps step SIZE roughly constant
    regardless of gradient magnitude -- that's the adaptive part of Adam,
    and it's exactly why it doesn't self-correct near an optimum. A 20-step
    trial at these settings (job 45437036) bottomed out around step 14-15
    then climbed back up for the rest of the run: classic overshoot, not a
    plateau. lr is cosine-decayed from lr down to lr*lr_final_frac over
    n_steps so later steps naturally shrink; a nonzero floor (rather than
    decaying to 0) leaves room to keep improving if convergence lands later
    than expected. As a second line of defense, the best total loss seen is
    tracked independently of where the run ends up.
    """
    params = {
        'log_kp': jnp.log(jnp.asarray(init_kp, dtype=jnp.float64)),
        'log_wind_factor': jnp.log(jnp.asarray(init_wind_factor, dtype=jnp.float64)),
    }
    schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=n_steps,
                                           alpha=lr_final_frac)
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(params)

    header = (f"{'Step':>4}  {'Total':>12}  {'Albedo':>10}  {'Snow':>10}  "
              f"{'Melt':>10}  {'MB':>10}  {'|grad|':>10}  {'lr':>10}")
    print(header, flush=True)

    history = []
    best = dict(step=-1, total=np.inf, kp=init_kp, wind_factor=init_wind_factor)
    for i in range(n_steps):
        t0 = time.time()
        (total, metrics), (d_kp, d_wf) = value_and_grad(
            params['log_kp'], params['log_wind_factor'])
        grads = {'log_kp': d_kp, 'log_wind_factor': d_wf}
        jax.block_until_ready((total, grads))

        bad = {k: np.where(~np.isfinite(np.asarray(v)))[0] for k, v in grads.items()}
        if any(len(v) for v in bad.values()):
            print(f'  step {i}: non-finite gradient after {time.time() - t0:.1f}s',
                  flush=True)
            for key, idx in bad.items():
                if len(idx):
                    print(f'  step {i}: non-finite {key} for '
                          f'{[GLACIERS[j] for j in idx]}', flush=True)
            report_forward_health(i, metrics, model, obs)
            print('  Stopping: non-finite gradient detected.', flush=True)
            break

        # params/total here are a matched pair -- the loss just evaluated at
        # the params BEFORE this step's update -- so this is the right place
        # to both log history and check for a new best, ahead of updating
        cur_kp = np.exp(np.asarray(params['log_kp'])).tolist()
        cur_wf = np.exp(np.asarray(params['log_wind_factor'])).tolist()
        history.append(dict(step=i, total=float(total), kp=cur_kp, wind_factor=cur_wf))
        if float(total) < best['total']:
            best = dict(step=i, total=float(total), kp=cur_kp, wind_factor=cur_wf)

        raw_norm = float(optax.global_norm(grads))
        cur_lr = float(schedule(opt_state[-1].count if hasattr(opt_state[-1], 'count')
                                else opt_state.count))
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        print(f'{i:>4}  {float(total):>12.2f}  '
              f"{float(jnp.sum(metrics['albedo'])):>10.3f}  "
              f"{float(jnp.sum(metrics['snow'])):>10.3f}  "
              f"{float(jnp.sum(metrics['melt'])):>10.3f}  "
              f"{float(jnp.sum(metrics['mb'])):>10.3f}  "
              f'{raw_norm:>10.3e}  {cur_lr:>10.3e}  ({time.time() - t0:.1f}s)', flush=True)

    print(f'\nBest total loss {best["total"]:.2f} at step {best["step"]} '
          f'(final step was {history[-1]["step"] if history else -1})', flush=True)

    return np.asarray(best['kp']), np.asarray(best['wind_factor']), history


class _NonFiniteGradient(Exception):
    """Raised out of the trust-region objective when a pass comes back bad."""


def run_optimization_trust_region(value_and_grad, init_kp, init_wind_factor,
                                  model, obs, n_steps=30, initial_radius=0.2,
                                  gtol=1e-4, xtol=1e-10, hessian='bfgs'):
    """
    Trust-region optimization of log(kp) and log(wind_factor), via scipy's
    trust-constr with a quasi-Newton Hessian built from the gradients we
    already pay for.

    Why scipy rather than an optax/JAX solver: the parameter vector is
    2 * n_glaciers long (2 numbers here), while a single value_and_grad call
    is a full multi-year forward+backward pass costing minutes. Nothing about
    the optimizer itself is worth putting on the device; what matters is
    extracting as much as possible from each evaluation, and scipy's
    trust-region implementations are the mature ones for exactly this regime
    (tiny dimension, very expensive oracle, gradients but no Hessian).

    Why trust region rather than Adam: Adam picks a step SIZE from a schedule
    and only takes the direction from the gradient, so near an optimum it
    keeps stepping at whatever the schedule says and overshoots -- the
    step-14 bottom-then-climb seen in the Adam runs. A trust region instead
    fits a local quadratic model, takes the step that minimizes it inside a
    radius, then compares the reduction the model PREDICTED against the one
    actually observed. A step that disagrees badly is rejected outright and
    the radius shrinks; a step that agrees well expands it. The step length
    is therefore set by measured agreement with the objective rather than by
    the iteration index, which is the self-correction Adam structurally
    cannot do.

    The Hessian is accumulated by BFGS (damped, so a noisy curvature pair
    cannot destroy positive definiteness) or SR1 from successive gradients,
    so it costs no extra passes. SR1 may go indefinite, which is harmless
    here -- the trust region bounds the step regardless -- and it often
    tracks true curvature better; BFGS is the safer default.

    Cost accounting: every objective evaluation is one full backward pass,
    accepted or rejected. Iterations and evaluations therefore differ, and
    both are reported. As with the Adam loop, the best total loss seen is
    tracked independently of where the solver stops.
    """
    n_glaciers = len(init_kp)
    x0 = np.concatenate([np.log(np.asarray(init_kp, dtype=np.float64)),
                         np.log(np.asarray(init_wind_factor, dtype=np.float64))])

    progress = dict(n_eval=0, n_iter=0, radius=float(initial_radius))
    history = []
    best = dict(step=-1, total=np.inf,
                kp=list(np.asarray(init_kp, dtype=np.float64)),
                wind_factor=list(np.asarray(init_wind_factor, dtype=np.float64)))

    header = (f"{'Eval':>4}  {'Iter':>4}  {'Total':>12}  {'Albedo':>10}  "
              f"{'Snow':>10}  {'Melt':>10}  {'MB':>10}  {'|grad|':>10}  "
              f"{'radius':>10}")
    print(header, flush=True)

    def objective(x):
        t0 = time.time()
        step = progress['n_eval']
        progress['n_eval'] += 1

        (total, metrics), (d_kp, d_wf) = value_and_grad(
            jnp.asarray(x[:n_glaciers]), jnp.asarray(x[n_glaciers:]))
        jax.block_until_ready((total, d_kp, d_wf))
        grad = np.concatenate([np.asarray(d_kp, dtype=np.float64),
                               np.asarray(d_wf, dtype=np.float64)])

        if not (np.isfinite(float(total)) and np.isfinite(grad).all()):
            print(f'  step {step}: non-finite value/gradient after '
                  f'{time.time() - t0:.1f}s', flush=True)
            bad = np.where(~np.isfinite(grad))[0]
            for j in bad:
                key = 'log_kp' if j < n_glaciers else 'log_wind_factor'
                print(f'  step {step}: non-finite {key} for '
                      f'{GLACIERS[j % n_glaciers]}', flush=True)
            report_forward_health(step, metrics, model, obs)
            raise _NonFiniteGradient

        cur_kp = np.exp(x[:n_glaciers]).tolist()
        cur_wf = np.exp(x[n_glaciers:]).tolist()
        history.append(dict(step=step, iteration=progress['n_iter'],
                            total=float(total), kp=cur_kp, wind_factor=cur_wf,
                            radius=progress['radius']))
        if float(total) < best['total']:
            best.update(step=step, total=float(total), kp=cur_kp, wind_factor=cur_wf)

        print(f"{step:>4}  {progress['n_iter']:>4}  {float(total):>12.2f}  "
              f"{float(jnp.sum(metrics['albedo'])):>10.3f}  "
              f"{float(jnp.sum(metrics['snow'])):>10.3f}  "
              f"{float(jnp.sum(metrics['melt'])):>10.3f}  "
              f"{float(jnp.sum(metrics['mb'])):>10.3f}  "
              f"{np.linalg.norm(grad):>10.3e}  {progress['radius']:>10.3e}  "
              f'({time.time() - t0:.1f}s)', flush=True)

        return float(total), grad

    # trust-constr hands the callback its whole internal state, which is where
    # the current radius lives; it is only read for reporting, so the next
    # evaluations are labelled with the radius that produced them
    def callback(xk, state):
        progress['n_iter'] = int(state.niter)
        progress['radius'] = float(state.tr_radius)
        return False

    if hessian == 'sr1':
        hess = SR1()
    else:
        hess = BFGS(exception_strategy='damp_update')

    try:
        result = minimize(objective, x0, jac=True, hess=hess,
                          method='trust-constr', callback=callback,
                          options=dict(initial_tr_radius=float(initial_radius),
                                       maxiter=n_steps, gtol=gtol, xtol=xtol,
                                       verbose=0))
        print(f'\n{result.message.strip()} -- {result.niter} iteration(s), '
              f"{progress['n_eval']} objective evaluation(s), "
              f'final |grad| {result.optimality:.3e}', flush=True)
    except _NonFiniteGradient:
        print('  Stopping: non-finite gradient detected.', flush=True)

    if not history:
        raise RuntimeError('trust-region solver produced no usable evaluation')

    print(f'\nBest total loss {best["total"]:.2f} at evaluation {best["step"]} '
          f'(last evaluation was {history[-1]["step"]})', flush=True)

    return np.asarray(best['kp']), np.asarray(best['wind_factor']), history



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-n', '--n_steps', type=int, default=30)
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-2)
    parser.add_argument('--chunk_days', type=int, default=CHUNK_DAYS)
    parser.add_argument('--optimizer', choices=['trust', 'adam', 'lbfgs'],
                        default='trust')
    parser.add_argument('--tr_radius', type=float, default=0.2,
                        help='initial trust-region radius, in log-parameter units: '
                             '0.2 is about a 20%% change in kp/wind_factor')
    parser.add_argument('--tr_hessian', choices=['bfgs', 'sr1'], default='bfgs')
    opts = parser.parse_args()

    print(f'JAX backend: {jax.default_backend()}  devices: {jax.devices()}',
          flush=True)

    t0 = _tick('starting')
    config_fn = build_config()
    print(f'Config -> {config_fn}', flush=True)

    model = init_pebsi(config_fn)
    model_days = model.dates[::24]
    n_days = len(model_days)
    print(f'{model.terrain.N_POINTS} points, {n_days} days '
          f'({model.dates[0]} to {model.dates[-1]})', flush=True)
    t0 = _tick('init + spinup done', t0)

    chunk_starts = list(range(0, n_days, opts.chunk_days))
    chunk_bounds = [(s, min(s + opts.chunk_days, n_days)) for s in chunk_starts]
    last_len = chunk_bounds[-1][1] - chunk_bounds[-1][0]

    # a "tail" only exists when the last chunk is shorter than a full chunk
    n_full = len(chunk_bounds) - (1 if last_len != opts.chunk_days else 0)
    if n_full == len(chunk_bounds):
        print(f'{n_full} chunk(s) of {opts.chunk_days} days; '
              f'~{opts.chunk_days * 10 / 1000:.1f} GB of reverse-pass state', flush=True)
    elif n_full == 0:
        print(f'1 chunk of {last_len} days (shorter than chunk_days={opts.chunk_days}, '
              f'no scan/tail); ~{last_len * 10 / 1000:.1f} GB of reverse-pass state', flush=True)
    else:
        print(f'{n_full} chunk(s) of {opts.chunk_days} days plus a {last_len}-day tail '
              f'(one extra compile); ~{opts.chunk_days * 10 / 1000:.1f} GB of reverse-pass state',
              flush=True)

    print('Loading observations...', flush=True)
    obs = build_observations(model, model_days)
    t0 = _tick('observations loaded', t0)

    print(f'Packing {len(chunk_bounds)} chunks...', flush=True)
    stacked, tail, counts, march = build_chunk_data(model, obs, model_days, chunk_bounds)
    for metric, c in counts.items():
        print(f'  {metric:<7} observations per glacier: {c.tolist()}', flush=True)
    t0 = _tick('chunks packed', t0)

    value_and_grad, params_loss_fn = make_loss_fn(model, obs, counts, march, stacked, tail)

    n_glaciers = len(GLACIERS)
    init_kp = np.full(n_glaciers, baseline['kp'])
    init_wf = np.full(n_glaciers, baseline['wind_factor'])
    print(f'Optimizer: {opts.optimizer}', flush=True)
    if opts.optimizer == 'trust':
        kp, wind_factor, history = run_optimization_trust_region(
            value_and_grad, init_kp=init_kp, init_wind_factor=init_wf,
            model=model, obs=obs, n_steps=opts.n_steps,
            initial_radius=opts.tr_radius, hessian=opts.tr_hessian)
    elif opts.optimizer == 'lbfgs':
        kp, wind_factor, history = run_optimization_lbfgs(
            value_and_grad, params_loss_fn, init_kp=init_kp, init_wind_factor=init_wf,
            model=model, obs=obs, n_steps=opts.n_steps)
    else:
        kp, wind_factor, history = run_optimization(
            value_and_grad, init_kp=init_kp, init_wind_factor=init_wf,
            model=model, obs=obs, n_steps=opts.n_steps, lr=opts.learning_rate)
    _tick('optimization loop done', t0)

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
