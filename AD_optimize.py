"""
Gradient-based (automatic differentiation) calibration of per-glacier
wind_factor and precipitation factor (kp) against glacier-wide
observations, using PEBSI's differentiable core.

This is the distributed successor to the point-scale jax_optimize.py.
The domain is the adaptive point mesh (method_distribute='adaptive'),
so every glacier is represented by an area-scaled cloud of points
rather than a handful of named benchmark sites, and the objective is
built from the three glacier-wide error metrics in
project/glacierwide_loss.py:

  - remotely sensed albedo         (Gaussian CRPS / sigma on the raw albedo)
  - SAR wet extent                 (Brier score)
  - Hugonnet glacier-wide balance  (Gaussian CRPS / sigma on the run's cumulative total)

and optionally (--index_mb) two more, from the USGS index sites, each
read at the nearest mesh point:

  - winter point balance           (Gaussian CRPS / sigma, absolute; --sigma_winter)
  - summer point balance anomaly   (Gaussian CRPS / sigma after removing each
                                    site's mean model-minus-measured summer
                                    offset; --sigma_summer)

Summer is scored as an anomaly because the index sites under-represent
glacier-wide loss (USGS subtracts a geodetic calibration of 0.3-0.8 m w.e.
yr-1 from their own index-based glacier-wide balance), so their mean level
is left to the Hugonnet term and they constrain the year-to-year variability.

Extra per-glacier parameters can be switched on alongside kp and
wind_factor: --fit_kappa switches the turbulent scheme to 'OG_hybrid' and
optimizes the katabatic constant og_kappa; --fit_precgrad optimizes the
precipitation gradient precgrad. Both are kept linear and bounded at 0.

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

  2. Softened wet indicator. The observed quantity is a binary mask,
     and the model's own wet test is a step function with no useful
     gradient. It is replaced by a logistic on the underlying continuous
     field (column liquid water), which reduces to the hard threshold as
     the width goes to zero.
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
from jax.scipy.special import ndtr
import optax
import numpy as np
import pandas as pd
from scipy.optimize import minimize, BFGS, SR1, Bounds
import xarray as xr
import yaml
from pyproj import Transformer

from project.glacierwide_loss import Albedo, Wet, MassBalance, translate_rgi, ALBEDO_SIGMA
from project.parameters import host, HOST_PATHS, GLACIERS, baseline, BASE_CONFIG

# ---------------------------------------------------------------------------
# Run settings
# ---------------------------------------------------------------------------

START_DATE = '2000-01-01 00:00'
END_DATE = '2025-08-31 23:00'

# days per temporal chunk to set peak device memory
# on a 32 GB card: 1 year ~ 10 GB total, 3 years ~ 17 GB, 5 years ~ 24 GB
CHUNK_DAYS = 365

# only these are needed by the three metrics; every extra field is carried
# through the whole differentiated graph, so the list is kept minimal
STORE_VARS = ('mass_balance', 'albedo', 'top_water')

# near-surface water [m w.e.] above which the model surface is considered wet
WET_WATER_THRESHOLD = 0.002

# width [m w.e.] of logistic function used to smooth the wet threshold
WET_WATER_WIDTH = 0.001

# months of the year the albedo product is scored over
ALBEDO_MONTHS = list(range(3, 10))

# optional extra per-glacier parameters (PEBSI dynamic fields, all >= 0),
# optimized linearly as value / scale so a unit step is the same order as
# the ~20% steps taken in log(kp)/log(wind_factor):
# og_kappa ~ 0-1.2e-3 [-]; precgrad ~ 1e-4 [fraction per m] (1% per 100 m)
EXTRA_SCALE = {'og_kappa': 1e-3, 'precgrad': 1e-4}


def build_config(overrides=None, tag='AD_optimize'):
    """
    Writes the PEBSI config for the optimization run: the five calibration
    glaciers on the adaptive point mesh, daily output, and nothing written
    to disk (every field the objective needs is read straight off the JAX
    records). overrides replaces any of those settings; tag names the
    config file, so concurrent jobs do not share one.
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
    config.update(overrides or {})
    config_fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             f'configs_{host}_{tag}.yaml')
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
    # drop implausible retrievals (cloud, shadow, artefacts)
    values = np.where((values >= 0.1) & (values <= 0.9), values, np.nan)
    return day_idx, _scatter_to_points(values.astype(np.float32), point_idx, n_points)


def load_wet_obs(name, lon, lat, point_idx, model_days, n_points, direction=None):
    """
    SAR wet extent mask, sampled at each point and matched to the nearest
    model day, following Wet.get_model_wet. Returns None if this glacier
    has no SAR cube, so a missing product simply drops the wet term from
    that glacier's objective.
    """
    try:
        wet = Wet(name, direction=direction)
    except (FileNotFoundError, OSError) as err:
        print(f'  ! no SAR cube for {name} ({err}); '
              'wet term dropped for this glacier', flush=True)
        return None

    x, y = _to_grid_xy(wet.crs, lon[point_idx], lat[point_idx])
    sel = dict(x=xr.DataArray(x, dims='point'),
               y=xr.DataArray(y, dims='point'), method='nearest')
    meas = wet.ds_meas_wet.sel(**sel)

    day_idx, keep = _match_times(meas.time.values, model_days)
    values = meas.transpose('time', 'point').values[keep].astype(np.float32)
    return day_idx, _scatter_to_points(values, point_idx, n_points)


def load_index_obs(model, model_days):
    """
    Index-site seasonal point mass balance from every site in the USGS
    benchmark glaciological files with data in the simulated period and
    enough winter records for project.data_handling.MassBalance to split
    seasons (annual-only sites are left out, as are annual balances, which
    repeat the seasonal information), each site matched to the nearest mesh
    point of its glacier. Site coordinates come from
    data/glacier_metadata.csv, falling back to the benchmark site table;
    sites with neither, and periods with implausible dates, are dropped.
    Only periods lying wholly inside the simulated days are kept, and a
    site's summers only if it has at least two (they are scored as
    anomalies). Returns flat per-period arrays: glacier index, site index,
    summer flag, point index, first and last model day (inclusive, matching
    get_model_mb's time slice), measurement.
    """
    from project.data_handling import MassBalance as IndexMassBalance, base_fp

    here = os.path.dirname(os.path.abspath(__file__))
    meta = pd.read_csv(os.path.join(here, 'data', 'glacier_metadata.csv'),
                       dtype={'rgiid': str})
    site_table = pd.read_csv(os.path.join(base_fp, 'MB_data',
                                          'Glacier_Mass_Balance_Data_Sites.csv'))
    lon = np.asarray(model.terrain.lon_n)
    lat = np.asarray(model.terrain.lat_n)
    rgiid = np.asarray(model.terrain.rgiid_n)

    out = {k: [] for k in ('glacier', 'site', 'summer', 'point', 'start_day',
                           'end_day', 'meas')}
    n_sites = 0
    for g, name in enumerate(GLACIERS):
        rgi = translate_rgi[name]['6']
        gp = np.where(rgiid == rgi)[0]
        folder = ''.join(part.capitalize() for part in name.split('_'))
        data = pd.read_csv(os.path.join(base_fp, 'data', 'mass_balance', 'point', folder,
                                        f'Input_{folder}_Glaciological_Data.csv'))
        in_period = data.Year.between(model_days[0].year, model_days[-1].year)
        sites = sorted(data.loc[in_period & data.ba.notna(), 'site_name'].unique())
        n_winter = n_summer = 0
        for site in sites:
            m = meta[(meta.rgiid == rgi) & (meta.site == site)]
            if m.empty:
                m = site_table[(site_table.Glacier.str.replace(' ', '') == folder)
                               & (site_table.site_name == site)].rename(
                    columns={'latitude': 'lat', 'longitude': 'lon'})
            if m.empty:
                continue
            slat, slon = m.lat.iloc[0], m.lon.iloc[0]
            p = gp[np.argmin(np.hypot(lat[gp] - slat,
                                      (lon[gp] - slon) * np.cos(np.deg2rad(slat))))]
            try:
                mb = IndexMassBalance(name, site)
            except Exception:  # e.g. a site with a single record
                continue
            # seasonal sites only: annual balances repeat the seasonal information
            if mb.dataset != 'seasonal':
                continue
            starts = pd.to_datetime(mb.period_starts).normalize()
            ends = pd.to_datetime(mb.period_ends).normalize()
            # MassBalance lists the summer periods first, then the winter ones
            summer = np.arange(len(starts)) < mb.n_summer
            # a few records have dates out of order or a period spanning a
            # missed year; keep only periods of a plausible length
            length = (ends - starts).days
            in_run = (np.minimum(starts, ends) >= model_days[0]) & \
                     (np.maximum(starts, ends) <= model_days[-1]) & np.isfinite(mb.data)
            inside = in_run & (length > 30) & (length <= 400)
            # a summer anomaly needs at least two summers at the site
            if (inside & summer).sum() < 2:
                inside &= ~summer
            for s, e, v, is_summer in zip(starts[inside], ends[inside],
                                          mb.data[inside], summer[inside]):
                out['glacier'].append(g)
                out['site'].append(n_sites)
                out['summer'].append(bool(is_summer))
                out['point'].append(int(p))
                out['start_day'].append(int(model_days.get_loc(s)))
                out['end_day'].append(int(model_days.get_loc(e)))
                out['meas'].append(float(v))
            n_summer += int((inside & summer).sum())
            n_winter += int((inside & ~summer).sum())
            n_sites += 1
        print(f'  {name:<12} {n_winter:>5} winter and {n_summer:>5} summer '
              'index-site periods', flush=True)
    return {k: np.asarray(v) for k, v in out.items()}


def build_observations(model, model_days, index_mb=False):
    """
    Loads every glacier's observations and returns them keyed on the
    model's own point axis, ready to be sliced per adjoint chunk.
    """
    lon = np.asarray(model.terrain.lon_n)
    lat = np.asarray(model.terrain.lat_n)
    rgiid = np.asarray(model.terrain.rgiid_n)
    n_points = len(lon)

    obs = {'albedo': [], 'wet': [],
           'mb_meas': [], 'mb_sigma': [], 'mb_years': [], 'mb_end_day': [], 'point_idx': []}

    for name in GLACIERS:
        point_idx = np.where(rgiid == translate_rgi[name]['6'])[0]
        assert len(point_idx) > 0, f'no simulation points landed on {name}'
        obs['point_idx'].append(point_idx)

        obs['albedo'].append(
            load_albedo_obs(name, lon, lat, point_idx, model_days, n_points))
        wet = load_wet_obs(name, lon, lat, point_idx, model_days, n_points)
        obs['wet'].append(wet)

        mb = MassBalance(name, dates=(START_DATE, END_DATE))
        obs['mb_meas'].append(float(mb.meas))
        obs['mb_sigma'].append(float(mb.sigma))
        obs['mb_years'].append(mb.n_years)

        mb_end_day = int(model_days.get_indexer([mb.matched_end], method='nearest')[0])
        obs['mb_end_day'].append(mb_end_day)
        truncated = mb.matched_end != pd.to_datetime(END_DATE.split()[0])
        mb_note = (f'  ! MB truncated to {mb.matched_start.date()} -> '
                   f'{mb.matched_end.date()} (data ends there)' if truncated else '')

        n_alb = 0 if obs['albedo'][-1] is None else len(obs['albedo'][-1][0])
        n_sar = 0 if wet is None else len(wet[0])
        print(f'  {name:<12} {len(point_idx):>5} points  '
              f'{n_alb:>5} albedo scenes  {n_sar:>5} SAR scenes  '
              f'MB {mb.meas:+.2f} +/- {mb.sigma:.2f} m w.e. yr-1{mb_note}', flush=True)

    obs['index'] = load_index_obs(model, model_days) if index_mb else None
    return obs


# ---------------------------------------------------------------------------
# 2. Per-chunk observation blocks
# ---------------------------------------------------------------------------

def _pack_metric(entries, chunk_bounds, n_points):
    """
    Splits one metric's (day_idx, values) pairs -- one per glacier, some
    possibly None -- into per-chunk arrays padded to a common width so
    every chunk compiles to the same shapes.

    Returns (idx, meas, mask, counts): idx/meas/mask are lists (one
    per chunk) of (n_glaciers, K, N_POINTS) arrays holding the day index
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
    n_days = len(model_days)

    alb = _pack_metric(obs['albedo'], chunk_bounds, n_points)
    wet = _pack_metric(obs['wet'], chunk_bounds, n_points)

    mb_end_day = np.asarray(obs['mb_end_day'])
    mb_valid_full = np.arange(n_days)[None, :] <= mb_end_day[:, None]

    # (n_index_periods, n_days) mask of the days each index-site period covers
    # (inclusive at both ends); zero periods when the index term is off
    index = obs['index']
    if index is None:
        index_days_full = np.zeros((0, n_days), dtype=np.float32)
    else:
        days = np.arange(n_days)[None, :]
        index_days_full = ((days >= index['start_day'][:, None])
                           & (days <= index['end_day'][:, None])).astype(np.float32)

    forcings, obs_chunks = [], []
    for c, (d0, d1) in enumerate(chunk_bounds):
        chunk_dates = model.dates[d0 * 24:d1 * 24]
        forcings.append(model.pack_forcings(model.params, chunk_dates, d0 * 24))
        chunk_obs = dict(
            alb=(alb[0][c], alb[1][c], alb[2][c]),
            wet=(wet[0][c], wet[1][c], wet[2][c]),
            mb_valid=mb_valid_full[:, d0:d1],
            index_days=index_days_full[:, d0:d1],
        )
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

    n_glaciers = len(obs['point_idx'])

    def per_glacier(sel):
        if index is None:
            return np.zeros(n_glaciers, dtype=np.int64)
        return np.bincount(index['glacier'][sel], minlength=n_glaciers)

    counts = dict(albedo=alb[3], wet=wet[3],
                  index_winter=per_glacier(None if index is None else ~index['summer']),
                  index_summer=per_glacier(None if index is None else index['summer']))
    return stacked, tail, counts


# ---------------------------------------------------------------------------
# 3. The differentiated objective
# ---------------------------------------------------------------------------

def _crps_gaussian(mod, meas, sigma):
    """
    JAX form of glacierwide_loss.crps_gaussian: CRPS of a Gaussian forecast
    N(mod, sigma^2) against meas, in the units of meas.
    """
    z = (meas - mod) / sigma
    pdf = jnp.exp(-0.5 * z ** 2) / jnp.sqrt(2 * jnp.pi)
    return sigma * (z * (2 * ndtr(z) - 1) + 2 * pdf - 1 / jnp.sqrt(jnp.pi))


def make_loss_fn(model, obs, counts, stacked, tail, extras=(),
                 sigma_winter=0.3, sigma_summer=0.1, index_weight=1.0):
    """
    Builds the value-and-gradient of the whole run with respect to the
    per-glacier log(kp), log(wind_factor) and an (len(extras), n_glaciers)
    array of extra parameters, each stored as value / EXTRA_SCALE[name].
    Fields not in extras keep their config values.

    The full-length temporal chunks are walked by one lax.scan, carrying
    the glacier state and a set of running sums the three metrics are built
    from; a short trailing chunk, if there is one, runs after it. Each
    chunk's forward pass is wrapped in jax.checkpoint, so reverse mode
    keeps only the state at the chunk boundaries and rematerializes a
    chunk's interior when it reaches it -- without that, the scan would
    retain every chunk's full inner graph at once.

    The forcings, observations, initial state and point attributes (whose
    shading and solar-angle tables are (N_POINTS, hours of a year) each)
    are passed as ARGUMENTS to the jitted function rather than captured in
    its closure. Captured arrays are constants: jax would fold them into the
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
    # (mesh.voronoi_weights, reached via distribute_rule -> mesh_polygon).
    # The mesh puts one point per triangular element and reads its weight off
    # the true outline, so weights only depart from uniform where an element
    # is cut by the boundary. Those margin points are
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
    mb_years = jnp.asarray(obs['mb_years'])
    count = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in counts.items()}

    # index-site periods: the mesh point each is read at, its measurement,
    # whether it is a summer, one-hots for summing per glacier (winter and
    # summer separately) and per site (for each site's mean summer offset)
    index = obs['index']
    if index is None:
        index = dict(point=np.zeros(0, dtype=np.int32), meas=np.zeros(0),
                     summer=np.zeros(0, dtype=bool), glacier=np.zeros(0, dtype=np.int32),
                     site=np.zeros(0, dtype=np.int32))
    index_point = jnp.asarray(index['point'], dtype=jnp.int32)
    index_meas = jnp.asarray(index['meas'])
    is_summer = np.asarray(index['summer'], dtype=bool)
    glacier_hot = index['glacier'][None, :] == np.arange(n_glaciers)[:, None]
    winter_member = jnp.asarray((glacier_hot & ~is_summer).astype(np.float64))
    summer_member = jnp.asarray((glacier_hot & is_summer).astype(np.float64))
    n_sites = int(index['site'].max()) + 1 if len(index['site']) else 0
    site_summer = (index['site'][None, :] == np.arange(n_sites)[:, None]) & is_summer
    site_summer_mean = jnp.asarray(  # (n_sites, n_periods): mean over that site's summers
        site_summer / np.maximum(site_summer.sum(axis=1, keepdims=True), 1))
    index_site = jnp.asarray(index['site'], dtype=jnp.int32)
    is_summer = jnp.asarray(is_summer)

    # fields to be checked directly for non-finite values anywhere
    RAW_FIELDS = ('albedo', 'top_water', 'mass_balance')

    def init_acc():
        return dict(
            albedo_crps=jnp.zeros(n_glaciers),
            wet_brier=jnp.zeros(n_glaciers),
            mb=jnp.zeros(n_points),
            index_mb=jnp.zeros(index_point.shape[0]),
            raw_nonfinite={f: jnp.array(False) for f in RAW_FIELDS},
        )

    def accumulate(acc, records, obs_chunk):
        alb_idx, alb_meas, alb_mask = obs_chunk['alb']
        mb_valid = obs_chunk['mb_valid']
        wet_idx, wet_meas, wet_mask = obs_chunk['wet']

        # (n_glaciers, K, N_POINTS) gathers of the model's daily fields at
        # each glacier's observation days
        #
        # padding entries (this glacier's own width padded to match whichever
        # glacier has the most albedo scenes) gather day-index 0's REAL
        # model value; if that's ever NaN, masking only the final loss term
        # protects the forward sum but not the gradient -- the CRPS gradient
        # is NaN at a NaN model value, and 0 (mask) * NaN = NaN through the
        # masked branch. So albedo is masked to a neutral placeholder
        # (resid == 0) BEFORE scoring, same pattern as the wet sigmoid input
        # below.
        safe_albedo = jnp.where(alb_mask, records.albedo[alb_idx], alb_meas)
        albedo_crps = jnp.sum(
            jnp.where(alb_mask,
                      _crps_gaussian(safe_albedo, alb_meas, ALBEDO_SIGMA) / ALBEDO_SIGMA,
                      0.0),
            axis=(1, 2))

        # relax hard cut-off function to a logistic threshold for derivation;
        # top_water is in kg m-2, so convert to m w.e.
        safe_water = jnp.where(wet_mask, records.top_water[wet_idx] / 1000,
                               WET_WATER_THRESHOLD)
        p_wet = jax.nn.sigmoid((safe_water - WET_WATER_THRESHOLD) / WET_WATER_WIDTH)

        raw_nonfinite = {
            f: acc['raw_nonfinite'][f] | jnp.any(~jnp.isfinite(getattr(records, f)))
            for f in RAW_FIELDS
        }

        wet_brier = jnp.sum(jnp.where(wet_mask, (p_wet - wet_meas) ** 2, 0.0),
                            axis=(1, 2))

        return dict(
            albedo_crps=acc['albedo_crps'] + albedo_crps,
            wet_brier=acc['wet_brier'] + wet_brier,
            mb=acc['mb'] + jnp.sum(
                jnp.where(mb_valid[glacier_of_point].T, records.mass_balance, 0.0),
                axis=0),
            # each index-site period's running sum of its point's daily balance
            index_mb=acc['index_mb'] + jnp.sum(
                obs_chunk['index_days'].T * records.mass_balance[:, index_point], axis=0),
            raw_nonfinite=raw_nonfinite,
        )

    def finalize(acc):
        """
        Turns the accumulated sums into the three glacier-wide metrics, each
        unitless so they add without weighting: Gaussian CRPS / sigma for
        albedo and mass balance and the Brier score for wet extent, the same
        construction as Albedo.crps_loss, Wet.brier_loss and
        MassBalance.crps_loss in glacierwide_loss. A metric with no
        observations for a glacier (a missing SAR cube, say) has count 0 and
        drops out of that glacier's objective.
        """
        def per_obs(total, n):
            return jnp.where(n > 0, total / jnp.where(n > 0, n, 1.0), 0.0)

        albedo = per_obs(acc['albedo_crps'], count['albedo'])
        wet = per_obs(acc['wet_brier'], count['wet'])

        mb_mod = member @ acc['mb'] / mb_years
        mb = _crps_gaussian(mb_mod, mb_meas, mb_sigma) / mb_sigma

        # index sites: winter scored absolutely; summer as an anomaly, by
        # removing each site's mean model-minus-measured summer offset first
        err = acc['index_mb'] - index_meas
        site_offset = (site_summer_mean @ err)[index_site] if n_sites else jnp.zeros_like(err)
        index_mod = acc['index_mb'] - jnp.where(is_summer, site_offset, 0.0)
        sigma = jnp.where(is_summer, sigma_summer, sigma_winter)
        index_crps = _crps_gaussian(index_mod, index_meas, sigma) / sigma
        index_winter = per_obs(winter_member @ index_crps, count['index_winter'])
        index_summer = per_obs(summer_member @ index_crps, count['index_summer'])

        # index_weight multiplies both index terms (1 = one vote each, like
        # the other per-glacier terms)
        index_winter = index_weight * index_winter
        index_summer = index_weight * index_summer
        total = jnp.sum(albedo + wet + mb + index_winter + index_summer)
        return total, dict(albedo=albedo, wet=wet, mb=mb, index_winter=index_winter,
                           index_summer=index_summer, mb_mod=mb_mod, acc=acc)

    # static_args stays in the closure: jax.checkpoint only sees
    # JAX-traceable arguments
    def _run_chunk(state, forcings, new_dynamic_args, point_attrs):
        return pebsi_main(state, forcings, point_attrs, static_args,
                          new_dynamic_args)

    def loss_fn(log_kp, log_wind_factor, extra, initial_state, stacked, tail, point_attrs):
        stacked_forcings, stacked_obs = stacked
        kp = jnp.exp(log_kp)[glacier_of_point]
        wind_factor = jnp.exp(log_wind_factor)[glacier_of_point]
        new_dynamic_args = dynamic_args._replace(kp=kp, wind_factor=wind_factor)
        for i, name in enumerate(extras):
            new_dynamic_args = new_dynamic_args._replace(
                **{name: (extra[i] * EXTRA_SCALE[name])[glacier_of_point]})

        def scan_chunk(carry, xs):
            state, acc = carry
            forcings, obs_chunk = xs
            state, records = jax.checkpoint(_run_chunk)(
                state, forcings, new_dynamic_args, point_attrs)
            return (state, accumulate(acc, records, obs_chunk)), None

        carry, _ = jax.lax.scan(
            scan_chunk, (initial_state, init_acc()),
            (stacked_forcings, stacked_obs))
        if tail is not None:
            carry, _ = scan_chunk(carry, tail)
        return finalize(carry[1])

    initial_state = model.initial_state
    jitted = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1, 2), has_aux=True))

    # plain python wrapper, so the big arrays reach the jitted function as
    # arguments without the optimizer loop having to carry them around
    def value_and_grad(log_kp, log_wind_factor, extra):
        return jitted(log_kp, log_wind_factor, extra, initial_state, stacked, tail,
                      point_attrs)

    # lets a caller compile ahead of time, to time compilation separately
    # and read XLA's memory analysis before anything runs
    value_and_grad.lower = lambda log_kp, log_wind_factor, extra: jitted.lower(
        log_kp, log_wind_factor, extra, initial_state, stacked, tail, point_attrs)
    value_and_grad.args = (initial_state, stacked, tail, point_attrs)
    value_and_grad.loss_fn = loss_fn

    # bare, no-aux, single-pytree-arg wrapper around the SAME loss_fn, for
    # optax.lbfgs's linesearch: it differentiates this function itself
    # (jax.value_and_grad(value_fn)), so it must be a plain scalar function,
    # not the pre-differentiated value_and_grad above. Sharing loss_fn
    # guarantees the two paths compute identical math.
    params_jitted = jax.jit(
        lambda params: loss_fn(params['log_kp'], params['log_wind_factor'],
                               params['extra'], initial_state, stacked, tail,
                               point_attrs)[0])

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
    all-five-glaciers-at-once failure mode seen in practice. albedo/wet
    are already per-glacier sums, so the finalized values are enough.

    Returns True if nothing came back non-finite.
    """
    healthy = True

    for key in ('albedo', 'wet', 'mb', 'index_winter', 'index_summer'):
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


def _extras_physical(extra, extras):
    """{name: per-glacier list in physical units} from the scaled extra array."""
    return {name: (np.asarray(extra[i]) * EXTRA_SCALE[name]).tolist()
            for i, name in enumerate(extras)}


def run_optimization(value_and_grad, init_kp, init_wind_factor, model, obs,
                     n_steps=10, lr=5e-2, lr_final_frac=0.1,
                     init_extra=None, extras=()):
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

    Any extra parameters (og_kappa, precgrad; see EXTRA_SCALE) are optimized
    linearly, so they can reach 0, and clipped at 0 after each update.
    """
    params = {
        'log_kp': jnp.log(jnp.asarray(init_kp, dtype=jnp.float64)),
        'log_wind_factor': jnp.log(jnp.asarray(init_wind_factor, dtype=jnp.float64)),
        'extra': jnp.asarray(np.zeros((0, len(init_kp))) if init_extra is None
                             else init_extra, dtype=jnp.float64),
    }
    schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=n_steps,
                                           alpha=lr_final_frac)
    optimizer = optax.adam(schedule)
    opt_state = optimizer.init(params)

    header = (f"{'Step':>4}  {'Total':>12}  {'Albedo':>10}  {'Wet':>10}  "
              f"{'MB':>10}  {'IdxWin':>10}  {'IdxSum':>10}  {'|grad|':>10}  {'lr':>10}")
    print(header, flush=True)

    history = []
    best = dict(step=-1, total=np.inf, kp=init_kp, wind_factor=init_wind_factor,
                **_extras_physical(params['extra'], extras))
    for i in range(n_steps):
        t0 = time.time()
        (total, metrics), (d_kp, d_wf, d_extra) = value_and_grad(
            params['log_kp'], params['log_wind_factor'], params['extra'])
        grads = {'log_kp': d_kp, 'log_wind_factor': d_wf, 'extra': d_extra}
        jax.block_until_ready((total, grads))

        bad = {k: np.where(~np.isfinite(np.asarray(v)))[0] for k, v in grads.items()}
        if any(len(v) for v in bad.values()):
            print(f'  step {i}: non-finite gradient after {time.time() - t0:.1f}s',
                  flush=True)
            for key, idx in bad.items():
                if len(idx):
                    print(f'  step {i}: non-finite {key} for '
                          f'{[GLACIERS[j % len(GLACIERS)] for j in idx]}', flush=True)
            report_forward_health(i, metrics, model, obs)
            print('  Stopping: non-finite gradient detected.', flush=True)
            break

        # params/total here are a matched pair -- the loss just evaluated at
        # the params BEFORE this step's update -- so this is the right place
        # to both log history and check for a new best, ahead of updating
        cur_kp = np.exp(np.asarray(params['log_kp'])).tolist()
        cur_wf = np.exp(np.asarray(params['log_wind_factor'])).tolist()
        cur_extra = _extras_physical(params['extra'], extras)
        history.append(dict(step=i, total=float(total), kp=cur_kp, wind_factor=cur_wf,
                            **cur_extra))
        if float(total) < best['total']:
            best = dict(step=i, total=float(total), kp=cur_kp, wind_factor=cur_wf,
                        **cur_extra)

        raw_norm = float(optax.global_norm(grads))
        cur_lr = float(schedule(opt_state[-1].count if hasattr(opt_state[-1], 'count')
                                else opt_state.count))
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        params['extra'] = jnp.maximum(params['extra'], 0.0)

        print(f'{i:>4}  {float(total):>12.2f}  '
              f"{float(jnp.sum(metrics['albedo'])):>10.3f}  "
              f"{float(jnp.sum(metrics['wet'])):>10.3f}  "
              f"{float(jnp.sum(metrics['mb'])):>10.3f}  "
              f"{float(jnp.sum(metrics['index_winter'])):>10.3f}  "
              f"{float(jnp.sum(metrics['index_summer'])):>10.3f}  "
              f'{raw_norm:>10.3e}  {cur_lr:>10.3e}  ({time.time() - t0:.1f}s)', flush=True)

    print(f'\nBest total loss {best["total"]:.2f} at step {best["step"]} '
          f'(final step was {history[-1]["step"] if history else -1})', flush=True)

    return (np.asarray(best['kp']), np.asarray(best['wind_factor']),
            {name: np.asarray(best[name]) for name in extras}, history)


class _NonFiniteGradient(Exception):
    """Raised out of the trust-region objective when a pass comes back bad."""


def write_progress(best, history, tag=''):
    """
    Rewrites the best parameters so far and the full evaluation history
    after every objective evaluation, so a run that hits its time limit
    still leaves its results behind.
    """
    fn = os.path.join(HOST_PATHS[host]['output_fp'], f'AD_optimize_progress{tag}.yaml')
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    with open(fn, 'w') as f:
        yaml.dump(dict(glaciers=GLACIERS, best=best, history=history), f, sort_keys=False)


def run_optimization_trust_region(value_and_grad, init_kp, init_wind_factor,
                                  model, obs, n_steps=10, initial_radius=0.2,
                                  gtol=1e-4, xtol=1e-10, hessian='bfgs',
                                  init_extra=None, extras=(), tag=''):
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

    Each extra parameter (og_kappa, precgrad; see EXTRA_SCALE) adds one more
    block of n_glaciers to x, kept linear (so it can reach 0) and bounded
    below by 0 through trust-constr's bounds.
    """
    n_glaciers = len(init_kp)
    n_extra = len(extras)
    extra0 = np.asarray(np.zeros((0, n_glaciers)) if init_extra is None else init_extra,
                        dtype=np.float64).reshape(n_extra, n_glaciers)
    x0 = np.concatenate([np.log(np.asarray(init_kp, dtype=np.float64)),
                         np.log(np.asarray(init_wind_factor, dtype=np.float64)),
                         extra0.ravel()])
    keys = ['log_kp', 'log_wind_factor'] + list(extras)

    def unpack(x):
        return (x[:n_glaciers], x[n_glaciers:2 * n_glaciers],
                x[2 * n_glaciers:].reshape(n_extra, n_glaciers))

    progress = dict(n_eval=0, n_iter=0, radius=float(initial_radius))
    history = []
    best = dict(step=-1, total=np.inf,
                kp=list(np.asarray(init_kp, dtype=np.float64)),
                wind_factor=list(np.asarray(init_wind_factor, dtype=np.float64)),
                **_extras_physical(extra0, extras))

    header = (f"{'Eval':>4}  {'Iter':>4}  {'Total':>12}  {'Albedo':>10}  "
              f"{'Wet':>10}  {'MB':>10}  {'IdxWin':>10}  {'IdxSum':>10}  {'|grad|':>10}  "
              f"{'radius':>10}")
    print(header, flush=True)

    def objective(x):
        t0 = time.time()
        step = progress['n_eval']
        progress['n_eval'] += 1

        log_kp, log_wf, extra = unpack(x)
        (total, metrics), (d_kp, d_wf, d_extra) = value_and_grad(
            jnp.asarray(log_kp), jnp.asarray(log_wf), jnp.asarray(extra))
        jax.block_until_ready((total, d_kp, d_wf, d_extra))
        grad = np.concatenate([np.asarray(d_kp, dtype=np.float64),
                               np.asarray(d_wf, dtype=np.float64),
                               np.asarray(d_extra, dtype=np.float64).ravel()])

        if not (np.isfinite(float(total)) and np.isfinite(grad).all()):
            print(f'  step {step}: non-finite value/gradient after '
                  f'{time.time() - t0:.1f}s', flush=True)
            bad = np.where(~np.isfinite(grad))[0]
            for j in bad:
                print(f'  step {step}: non-finite {keys[j // n_glaciers]} for '
                      f'{GLACIERS[j % n_glaciers]}', flush=True)
            report_forward_health(step, metrics, model, obs)
            raise _NonFiniteGradient

        cur_kp = np.exp(log_kp).tolist()
        cur_wf = np.exp(log_wf).tolist()
        cur_extra = _extras_physical(extra, extras)
        history.append(dict(step=step, iteration=progress['n_iter'],
                            total=float(total), kp=cur_kp, wind_factor=cur_wf,
                            **cur_extra, radius=progress['radius'],
                            albedo=np.asarray(metrics['albedo']).tolist(),
                            wet=np.asarray(metrics['wet']).tolist(),
                            mb=np.asarray(metrics['mb']).tolist(),
                            index_winter=np.asarray(metrics['index_winter']).tolist(),
                            index_summer=np.asarray(metrics['index_summer']).tolist()))
        if float(total) < best['total']:
            best.update(step=step, total=float(total), kp=cur_kp, wind_factor=cur_wf,
                        **cur_extra)
        write_progress(best, history, tag)

        print(f"{step:>4}  {progress['n_iter']:>4}  {float(total):>12.2f}  "
              f"{float(jnp.sum(metrics['albedo'])):>10.3f}  "
              f"{float(jnp.sum(metrics['wet'])):>10.3f}  "
              f"{float(jnp.sum(metrics['mb'])):>10.3f}  "
              f"{float(jnp.sum(metrics['index_winter'])):>10.3f}  "
              f"{float(jnp.sum(metrics['index_summer'])):>10.3f}  "
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

    # extras >= 0; kp and wind_factor are already positive through the log
    bounds = None
    if n_extra:
        lower = np.full(len(x0), -np.inf)
        lower[2 * n_glaciers:] = 0.0
        bounds = Bounds(lower, np.full(len(x0), np.inf))

    try:
        result = minimize(objective, x0, jac=True, hess=hess, bounds=bounds,
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

    return (np.asarray(best['kp']), np.asarray(best['wind_factor']),
            {name: np.asarray(best[name]) for name in extras}, history)



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-n', '--n_steps', type=int, default=10)
    parser.add_argument('-lr', '--learning_rate', type=float, default=5e-2)
    parser.add_argument('--chunk_days', type=int, default=CHUNK_DAYS)
    parser.add_argument('--checkpoint_hourly', action='store_true',
                        help='also checkpoint each hourly step: ~4x less VRAM per point, '
                             'one extra forward pass; needed above a few thousand points')
    parser.add_argument('--optimizer', choices=['trust', 'adam', 'lbfgs'],
                        default='trust')
    parser.add_argument('--tr_radius', type=float, default=0.2,
                        help='initial trust-region radius, in log-parameter units: '
                             '0.2 is about a 20%% change in kp/wind_factor')
    parser.add_argument('--tr_hessian', choices=['bfgs', 'sr1'], default='bfgs')
    parser.add_argument('--fit_kappa', action='store_true',
                        help="switch to method_turbulent='OG_hybrid' and optimize og_kappa "
                             'per glacier alongside kp and wind_factor')
    parser.add_argument('--fit_precgrad', action='store_true',
                        help='optimize the precipitation gradient precgrad per glacier')
    parser.add_argument('--index_mb', action='store_true',
                        help='add index-site point mass balance (CRPS / sigma) to the objective')
    parser.add_argument('--sigma_winter', type=float, default=0.3,
                        help='uncertainty [m w.e.] of an index-site winter balance '
                             '(USGS README: snow accumulation, 0.3 m w.e.)')
    parser.add_argument('--sigma_summer', type=float, default=0.1,
                        help='uncertainty [m w.e.] of an index-site summer balance anomaly '
                             '(USGS README: snow and ice melt, 0.10 m w.e.)')
    parser.add_argument('--index_weight', type=float, default=1.0,
                        help='multiplier on the two per-glacier index-site terms')
    parser.add_argument('--init_from', default=None,
                        help='yaml with glaciers and per-glacier kp, wind_factor and '
                             '(optionally) og_kappa / precgrad to start from, instead of '
                             'the baseline / config values')
    parser.add_argument('--tag', default='',
                        help='suffix for the config, progress and result files, so a '
                             'run does not overwrite AD_optimize_result.yaml')
    opts = parser.parse_args()
    tag = f'_{opts.tag}' if opts.tag else ''
    extras = tuple(name for name, on in (('og_kappa', opts.fit_kappa),
                                         ('precgrad', opts.fit_precgrad)) if on)

    print(f'JAX backend: {jax.default_backend()}  devices: {jax.devices()}',
          flush=True)

    t0 = _tick('starting')
    overrides = dict(checkpoint_hourly=opts.checkpoint_hourly)
    if opts.fit_kappa:
        overrides['method_turbulent'] = 'OG_hybrid'
    config_fn = build_config(overrides, tag=f'AD_optimize{tag}')
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
    obs = build_observations(model, model_days, index_mb=opts.index_mb)
    t0 = _tick('observations loaded', t0)

    print(f'Packing {len(chunk_bounds)} chunks...', flush=True)
    stacked, tail, counts = build_chunk_data(model, obs, model_days, chunk_bounds)
    for metric, c in counts.items():
        print(f'  {metric:<7} observations per glacier: {c.tolist()}', flush=True)
    t0 = _tick('chunks packed', t0)

    value_and_grad, params_loss_fn = make_loss_fn(
        model, obs, counts, stacked, tail,
        extras=extras, sigma_winter=opts.sigma_winter,
        sigma_summer=opts.sigma_summer, index_weight=opts.index_weight)

    n_glaciers = len(GLACIERS)
    init_kp = np.full(n_glaciers, baseline['kp'])
    init_wf = np.full(n_glaciers, baseline['wind_factor'])
    # extras start from the config's value unless the init file gives one
    init_phys = {name: np.full(n_glaciers, float(np.mean(getattr(model.config.params, name))))
                 for name in extras}
    if opts.init_from:
        with open(opts.init_from) as f:
            start = yaml.safe_load(f)
        order = [start['glaciers'].index(g) for g in GLACIERS]
        init_kp = np.asarray(start['kp'], dtype=float)[order]
        init_wf = np.asarray(start['wind_factor'], dtype=float)[order]
        for name in extras:
            if name in start:
                init_phys[name] = np.asarray(start[name], dtype=float)[order]
    init_extra = np.array([init_phys[name] / EXTRA_SCALE[name] for name in extras]
                          ).reshape(len(extras), n_glaciers)
    print(f'Optimizer: {opts.optimizer}', flush=True)
    print(f"{'Start':<14} {'kp':>10} {'wind_factor':>13}"
          + ''.join(f' {name:>10}' for name in extras), flush=True)
    for g, name in enumerate(GLACIERS):
        print(f'{name:<14} {init_kp[g]:>10.4f} {init_wf[g]:>13.4f}'
              + ''.join(f' {init_phys[e][g]:>10.2e}' for e in extras), flush=True)
    if opts.optimizer == 'trust':
        kp, wind_factor, fitted, history = run_optimization_trust_region(
            value_and_grad, init_kp=init_kp, init_wind_factor=init_wf,
            model=model, obs=obs, n_steps=opts.n_steps,
            initial_radius=opts.tr_radius, hessian=opts.tr_hessian,
            init_extra=init_extra, extras=extras, tag=tag)
    elif opts.optimizer == 'lbfgs':
        kp, wind_factor, history = run_optimization_lbfgs(
            value_and_grad, params_loss_fn, init_kp=init_kp, init_wind_factor=init_wf,
            model=model, obs=obs, n_steps=opts.n_steps)
    else:
        kp, wind_factor, fitted, history = run_optimization(
            value_and_grad, init_kp=init_kp, init_wind_factor=init_wf,
            model=model, obs=obs, n_steps=opts.n_steps, lr=opts.learning_rate,
            init_extra=init_extra, extras=extras)
    _tick('optimization loop done', t0)

    print('\nOptimized parameters:')
    print(f"{'Glacier':<14} {'kp':>10} {'wind_factor':>13}"
          + ''.join(f' {name:>10}' for name in extras))
    for g, name in enumerate(GLACIERS):
        print(f'{name:<14} {kp[g]:>10.4f} {wind_factor[g]:>13.4f}'
              + ''.join(f' {fitted[e][g]:>10.2e}' for e in extras))

    out_fn = os.path.join(HOST_PATHS[host]['output_fp'], f'AD_optimize_result{tag}.yaml')
    os.makedirs(os.path.dirname(out_fn), exist_ok=True)
    result = dict(glaciers=GLACIERS,
                  kp=[float(v) for v in kp],
                  wind_factor=[float(v) for v in wind_factor])
    for name in extras:
        result[name] = [float(v) for v in fitted[name]]
    result['history'] = history
    with open(out_fn, 'w') as f:
        yaml.dump(result, f, sort_keys=False)
    print(f'\nWrote {out_fn}')


if __name__ == '__main__':
    main()
