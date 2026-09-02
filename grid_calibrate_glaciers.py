"""
Driver for one grid-calibration SLURM array task. Each task runs a
block of grid cells across ALL 5 glaciers at once (see
project/grid_calibrate.py) -- task id == job_idx directly, no
per-glacier split.

@author: clairevwilson
"""
import os

from project.bayes_calibrate import GLACIERS
from project.grid_calibrate import CELLS_PER_JOB, run_grid_job

if __name__ == '__main__':
    job_idx = int(os.environ['SLURM_ARRAY_TASK_ID'])

    print(f'Grid job {job_idx} (cells {job_idx * CELLS_PER_JOB} - '
          f'{job_idx * CELLS_PER_JOB + CELLS_PER_JOB - 1}), all 5 glaciers')
    theta_batch, log_likes = run_grid_job(job_idx)
    print(f'Job {job_idx} done:')
    for (kp, wf), ll_row in zip(theta_batch, log_likes):
        per_glacier = ', '.join(f'{g}={ll:.3f}' for g, ll in zip(GLACIERS, ll_row))
        print(f'  kp={kp:.3f} wind_factor={wf:.3f}  {per_glacier}')
