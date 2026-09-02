"""
Bayesian calibration of (kp, wind_factor) for one glacier via emcee,
batching each MCMC step's full walker ensemble into a single PEBSI
multi-point config run (one 'site' block per walker) so the model is
only invoked once per step rather than once per walker per step.

Likelihood combines Albedo.log_loss (Gaussian NLL) and
SnowlineMelt.bernoulli_loss (BCE, snow + melt) from
project/glacierwide_loss.py, matching the loss terms already used in
sensitivity_gulkana.py / plot_sensitivity.py.

@author: clairevwilson
"""
import os
import glob
import shutil
import yaml
import numpy as np
import pandas as pd
import xarray as xr
import emcee

import simulation as sim
from project.glacierwide_loss import Albedo, SnowlineMelt, MassBalance, translate_rgi
from project.scatter_sites import get_scattered_sites

OUTPUT_ROOT = '/ocean/projects/ees260009p/cwilson4/Output/bayes_calibrate/'
CHAIN_DIR = 'project/bayes_chains/'

GLACIERS = ['gulkana', 'kahiltna', 'kennicott', 'wolverine', 'lemon_creek']

PARAM_NAMES = ['kp', 'wind_factor']

# baseline/bounds mirror sensitivity_gulkana.py's kp/wind_factor entries
baseline = {'kp': 2.5, 'wind_factor': 3}
bounds = {'kp': (0.5, 4), 'wind_factor': (0.5, 5)}

BASE_CONFIG = {
    # PHYSICS
    'option_ice_albedo_tif': True,
    'option_windmaps': True,
    'option_accel_grains': True,
    'option_flat_plates': True,
    'option_dynamics': False,
    'constant_freshgrainsize': 54.5,
    'constant_irrwater': True,
    'precgrad': 0.000100,

    # CONFIGURATION
    'debug': False,
    'store_data': True,
    'progress_bar': False,
    'output_freq': 'daily',
    'temporal_chunk_years': 1,
    'store_vars': ['mass_balance', 'albedo', 'surftype', 'total_water'],
    'bias_vars': ['temp'],

    # FILEPATHS
    'climate_fp': '/ocean/projects/ees260009p/cwilson4/climate_data/',
    'rgi_fp': '/ocean/projects/ees260009p/cwilson4/RGI/rgi60/00_rgi60_attribs/',
    'cop30_vrt_path': '/ocean/projects/ees260009p/cwilson4/data/dems/COP30/COP30_reg01.vrt',
    'shading_fp': '/ocean/projects/ees260009p/cwilson4/data/shading/',
    'ice_albedo_fn': '/ocean/projects/ees260009p/cwilson4/data/ice_albedo/{gid}_albedo.tif',
    'thickness_fn': '/ocean/projects/ees260009p/cwilson4/data/ice_thickness/RGI60-01/RGI60-{gid}_thickness.tif',
    'windmap_fn': '/ocean/projects/ees260009p/cwilson4/data/windmapper/{gid}.nc',
}


def align_end_date_for_daily_output(start_date, end_date):
    """
    output_freq='daily' requires len(date_range(start_date, end_date,
    freq='h')) (inclusive) to be a multiple of 24 (pebsi/config.py's
    validate_config). Nudges end_date backward by the minimum number of
    hours needed to satisfy that, rather than requiring every caller to
    hand-align it (e.g. same start/end time-of-day never satisfies this,
    since the inclusive count is always 24*D + 1).
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    hours_between = int((end - start) / pd.Timedelta(hours=1))
    remainder = (hours_between + 1) % 24
    if remainder:
        end -= pd.Timedelta(hours=remainder)
    return str(end)


def log_prior(theta):
    kp, wf = theta
    if not (bounds['kp'][0] <= kp <= bounds['kp'][1]):
        return -np.inf
    if not (bounds['wind_factor'][0] <= wf <= bounds['wind_factor'][1]):
        return -np.inf
    return 0.0


def build_batched_config(glacier_name, rgi_id, theta_batch, site_names, metadata_fn,
                          start_date, end_date, run_tag=None):
    n_pts = len(site_names)

    configs = dict(BASE_CONFIG)
    configs['start_date'] = start_date
    configs['end_date'] = end_date
    configs['metadata_fn'] = metadata_fn
    # run_tag defaults to glacier_name (MCMC: one job per glacier at a time,
    # so no collision); grid jobs pass a per-job tag since several jobs for
    # the same glacier run concurrently and would otherwise write to the
    # same output_fp
    configs['output_fp'] = os.path.join(OUTPUT_ROOT, run_tag or glacier_name) + '/'

    configs['sites'] = site_names * len(theta_batch)
    configs['rgi_ids'] = [rgi_id] * (n_pts * len(theta_batch))
    configs['n_points'] = n_pts * len(theta_batch)
    configs['kp'] = [float(v) for v in np.repeat(theta_batch[:, 0], n_pts)]
    configs['wind_factor'] = [float(v) for v in np.repeat(theta_batch[:, 1], n_pts)]
    return configs


def run_batch_and_get_losses(glacier_name, rgi_id, theta_batch, site_names,
                              metadata_fn, ab, sm, mb, tmp_config_fn, start_date, end_date,
                              run_tag=None):
    n_pts = len(site_names)

    configs = build_batched_config(glacier_name, rgi_id, theta_batch, site_names, metadata_fn,
                                    start_date, end_date, run_tag=run_tag)

    # clear stale directories to guarantee this simulation ends in _0
    output_root = configs['output_fp'].rstrip('/')
    for stale_dir in glob.glob(output_root + '_*'):
        shutil.rmtree(stale_dir)

    with open(tmp_config_fn, 'w') as f:
        yaml.dump(configs, f, sort_keys=False)

    args = sim.get_args()
    args.config_fn = tmp_config_fn
    sim.PEBSI(args).run()

    output_fp = output_root + '_0/'
    ds = xr.open_zarr(output_fp + 'output.zarr')
    ds = ds[['albedo', 'surftype', 'total_water', 'mass_balance', 'lon', 'lat']].load()

    ab.get_model_albedo(ds)
    sm.get_model_snow(ds)

    log_likes = np.full(len(theta_batch), -np.inf)
    for w in range(len(theta_batch)):
        pts = slice(w * n_pts, (w + 1) * n_pts)
        albedo_nll = ab.log_loss(ab.mod[:, pts], ab.meas[:, pts])
        snow_nll, melt_nll = sm.bernoulli_loss(
            sm.mod_snow[:, pts], sm.meas_snow[:, pts],
            sm.mod_melt[:, pts], sm.meas_melt[:, pts])
        # MassBalance.get_model_mb averages over the 'point' dim internally,
        # so it has to run on just this walker's points, not the full batch.
        mb.get_model_mb(ds.isel(point=pts))
        mb_nll = mb.log_loss()
        log_likes[w] = -(albedo_nll + snow_nll + melt_nll + mb_nll)

    shutil.rmtree(output_fp)
    return log_likes


def make_log_posterior(glacier_name, rgi_id, site_names, metadata_fn, start_date, end_date):
    # Albedo/SnowlineMelt/MassBalance load their observation data once
    ab = Albedo(glacier_name)
    sm = SnowlineMelt(glacier_name)

    # MassBalance is loaded dynamically based on simulation dates
    mb_start = max(pd.to_datetime(start_date), pd.to_datetime('2000-01-01'))
    mb_end = min(pd.to_datetime(end_date), pd.to_datetime('2020-01-01'))
    mb = MassBalance(glacier_name, dates=(str(mb_start), str(mb_end)))
    tmp_config_fn = f'_bayes_{glacier_name}.yaml'

    def log_posterior_batch(theta_batch):
        theta_batch = np.atleast_2d(theta_batch)
        lp = np.array([log_prior(t) for t in theta_batch])
        finite = np.isfinite(lp)

        ll = np.full(len(theta_batch), -np.inf)
        if finite.any():
            ll[finite] = run_batch_and_get_losses(
                glacier_name, rgi_id, theta_batch[finite], site_names,
                metadata_fn, ab, sm, mb, tmp_config_fn, start_date, end_date)

        return lp + ll

    return log_posterior_batch


def calibrate_glacier(glacier_name, n_points=3000, n_walkers=32,
                       n_steps=500, n_burn=100, seed=0,
                       start_date='2000-01-01 00:00', end_date='2025-09-01 00:00'):
    rgi_id = translate_rgi[glacier_name]['6']
    end_date = align_end_date_for_daily_output(start_date, end_date)
    site_names, metadata_fn = get_scattered_sites(glacier_name, rgi_id, n_points)
    log_posterior_batch = make_log_posterior(
        glacier_name, rgi_id, site_names, metadata_fn, start_date, end_date)

    rng = np.random.default_rng(seed)
    theta0 = np.column_stack([
        baseline['kp'] + rng.normal(0, 0.05, n_walkers),
        baseline['wind_factor'] + rng.normal(0, 0.05, n_walkers),
    ])

    sampler = emcee.EnsembleSampler(
        n_walkers, len(PARAM_NAMES), log_posterior_batch, vectorize=True)
    sampler.run_mcmc(theta0, n_steps, progress=True)

    tau = sampler.get_autocorr_time(quiet=True)
    print(f'{glacier_name}: autocorrelation time (steps) = {tau}')

    chain = sampler.get_chain(discard=n_burn, flat=True)

    os.makedirs(CHAIN_DIR, exist_ok=True)
    np.savez(os.path.join(CHAIN_DIR, f'{glacier_name}_posterior.npz'),
              chain=chain, param_names=PARAM_NAMES, tau=tau)
    return chain
