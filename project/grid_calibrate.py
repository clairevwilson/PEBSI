"""
Grid-based Bayesian calibration of (kp, wind_factor), pooling all 5
glaciers' spatial points into ONE batched PEBSI call per job instead of
running each glacier as its own set of jobs. Every grid-cell replica in
a job packs n_points area-weighted across all 5 glaciers (PEBSI's own
multi-glacier 'scatter' distribution already area-weights by RGI
polygon area when handed several distinct rgi_ids -- see
scatter_sites.get_multiglacier_scattered_sites), so simulating 5
glaciers costs the same combined-point compute as simulating 1 glacier
at 5x the points, not 5x the jobs.

Each glacier still gets its OWN independent posterior -- per-glacier
log-likelihoods are kept separate (never summed across glaciers), only
the underlying simulation is pooled for compute efficiency.

Run one job per job_idx via batch_jobs/grid_calibrate.sh's array, then
combine every job's saved (cells x glaciers) log-likelihoods into the
per-glacier posteriors with plot_grid_posterior.py.

@author: clairevwilson
"""
import os
import glob
import shutil
import yaml
import numpy as np
import pandas as pd
import xarray as xr

import simulation as sim
from project.bayes_calibrate import (
    OUTPUT_ROOT, GLACIERS, bounds, align_end_date_for_daily_output, BASE_CONFIG,
)
from project.glacierwide_loss import Albedo, SnowlineMelt, MassBalance, translate_rgi
from project.scatter_sites import get_multiglacier_scattered_sites

GRID_RES = 4                   # 10 x 10 = 100 (kp, wind_factor) cells
CELLS_PER_JOB = 10               # 10 jobs total (not per glacier)
N_JOBS = (GRID_RES * GRID_RES) // CELLS_PER_JOB
RESULTS_DIR = 'project/grid_results/'


def build_grid():
    kp_vals = np.linspace(bounds['kp'][0], bounds['kp'][1], GRID_RES)
    wf_vals = np.linspace(bounds['wind_factor'][0], bounds['wind_factor'][1], GRID_RES)
    kp_grid, wf_grid = np.meshgrid(kp_vals, wf_vals, indexing='ij')
    return np.column_stack([kp_grid.ravel(), wf_grid.ravel()])  # (GRID_RES**2, 2)


def run_grid_job(job_idx, n_points=3000,
                  start_date='2000-01-01 00:00', end_date='2025-09-01 00:00'):
    end_date = align_end_date_for_daily_output(start_date, end_date)

    grid = build_grid()
    cell_idx = np.arange(job_idx * CELLS_PER_JOB, (job_idx + 1) * CELLS_PER_JOB)
    theta_batch = grid[cell_idx]
    n_cells = len(theta_batch)

    rgi_ids = [translate_rgi[g]['6'] for g in GLACIERS]
    site_names, site_rgi_ids, site_glacier_names, metadata_fn = \
        get_multiglacier_scattered_sites(GLACIERS, rgi_ids, n_points)
    n_pts = len(site_names)
    site_glacier_names = np.array(site_glacier_names)

    # build the batched multi-point config: n_cells replicas of the same
    # (5-glacier-mixed) point set, each replica tagged with its own
    # (kp, wind_factor)
    configs = dict(BASE_CONFIG)
    configs['start_date'] = start_date
    configs['end_date'] = end_date
    configs['metadata_fn'] = metadata_fn
    run_tag = f'grid_job{job_idx:03d}'
    configs['output_fp'] = os.path.join(OUTPUT_ROOT, run_tag) + '/'
    configs['sites'] = list(site_names) * n_cells
    configs['rgi_ids'] = list(site_rgi_ids) * n_cells
    configs['n_points'] = n_pts * n_cells
    configs['kp'] = [float(v) for v in np.repeat(theta_batch[:, 0], n_pts)]
    configs['wind_factor'] = [float(v) for v in np.repeat(theta_batch[:, 1], n_pts)]

    output_root = configs['output_fp'].rstrip('/')
    for stale_dir in glob.glob(output_root + '_*'):
        shutil.rmtree(stale_dir)

    tmp_config_fn = f'_{run_tag}.yaml'
    with open(tmp_config_fn, 'w') as f:
        yaml.dump(configs, f, sort_keys=False)

    args = sim.get_args()
    args.config_fn = tmp_config_fn
    sim.PEBSI(args).run()
    os.remove(tmp_config_fn)

    output_fp = output_root + '_0/'
    ds = xr.open_zarr(output_fp + 'output.zarr')
    ds = ds[['albedo', 'surftype', 'total_water', 'mass_balance', 'lon', 'lat']].load()

    log_likes = np.full((n_cells, len(GLACIERS)), -np.inf)
    for gi, glacier_name in enumerate(GLACIERS):
        # local_idx: this glacier's positions within one cell's n_pts block
        # (the same for every cell, since sites/site_glacier_names repeat
        # identically per cell); global_idx: those positions replicated
        # across all n_cells blocks, cell-major, matching ds's point order
        local_idx = np.where(site_glacier_names == glacier_name)[0]
        k_g = len(local_idx)
        global_idx = np.concatenate([local_idx + w * n_pts for w in range(n_cells)])
        ds_g = ds.isel(point=global_idx)

        ab = Albedo(glacier_name)
        sm = SnowlineMelt(glacier_name)
        mb_start = max(pd.to_datetime(start_date), pd.to_datetime('2000-01-01'))
        mb_end = min(pd.to_datetime(end_date), pd.to_datetime('2020-01-01'))
        mb = MassBalance(glacier_name, dates=(str(mb_start), str(mb_end)))

        ab.get_model_albedo(ds_g)
        sm.get_model_snow(ds_g)

        for w in range(n_cells):
            pts = slice(w * k_g, (w + 1) * k_g)
            albedo_nll = ab.log_loss(ab.mod[:, pts], ab.meas[:, pts])
            snow_nll, melt_nll = sm.bernoulli_loss(
                sm.mod_snow[:, pts], sm.meas_snow[:, pts],
                sm.mod_melt[:, pts], sm.meas_melt[:, pts])
            mb.get_model_mb(ds_g.isel(point=pts))
            mb_nll = mb.log_loss()
            log_likes[w, gi] = -(albedo_nll + snow_nll + melt_nll + mb_nll)

    shutil.rmtree(output_fp)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    np.savez(os.path.join(RESULTS_DIR, f'job{job_idx:03d}.npz'),
              theta=theta_batch, log_like=log_likes, glaciers=np.array(GLACIERS))
    return theta_batch, log_likes
