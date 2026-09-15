"""
Holds the mesh fixed and varies only the length the DEM is smoothed over
before slope and aspect are taken from it.

The sweep leaves slope variance tied to the point spacing, so terrain
roughness and resolution move together and neither can be blamed. Fixing
the mesh and varying the smoothing separates them: if roughness is what
drives the drift, mass balance moves across these runs at constant N. If
it does not move, terrain is not the cause.

    python experiment_dem_smoothing.py

@author: clairevwilson
"""
import glob
import os
import shutil

import numpy as np
import xarray as xr
import yaml

import simulation as sim
from AD_optimize import BASE_CONFIG, HOST_PATHS, host, baseline
from point_density_convergence import OUTDIR, START_DATE, END_DATE_RAW
from project.bayes_calibrate import align_end_date_for_daily_output
from project.glacierwide_loss import translate_rgi

GLACIER = 'gulkana'
SPACING = 300.0
SMOOTHS = [0, 100, 250, 500]
YEARS = 3.0


def run(smooth):
    rgi_id = translate_rgi[GLACIER]['6']
    wf = baseline['wind_factor']
    end_date = align_end_date_for_daily_output(START_DATE, END_DATE_RAW)

    tag = f'h{SPACING:.0f}_sm{smooth}'
    fp = os.path.join(OUTDIR, f'{GLACIER}_{tag}_wf{wf}')
    for old in glob.glob(fp + '_*'):
        shutil.rmtree(old)

    configs = dict(BASE_CONFIG)
    configs.update(HOST_PATHS[host])
    configs['temporal_chunk_years'] = 1
    configs['start_date'] = START_DATE
    configs['end_date'] = end_date
    configs['rgi_ids'] = [rgi_id]
    configs['method_distribute'] = 'mesh'
    configs['point_spacing'] = SPACING
    configs['dem_smooth_m'] = smooth
    configs['kp'] = baseline['kp']
    configs['wind_factor'] = wf
    configs['store_data'] = True
    configs['store_vars'] = ['mass_balance']
    configs['output_fp'] = fp

    fn = f'_dem_smooth_{smooth}.yaml'
    with open(fn, 'w') as f:
        yaml.dump(configs, f, sort_keys=False)
    args = sim.get_args(parse=False).parse_args([])
    args.config_fn = fn
    model = sim.PEBSI(args)
    model.run()
    os.remove(fn)

    w = model.terrain.weight_n
    slope = model.terrain.slope_n

    dirs = sorted(glob.glob(fp + '_*'))
    assert dirs, f'no output for {fp}'
    ds = xr.open_zarr(os.path.join(dirs[-1], 'output.zarr'))
    mb = ds.mass_balance.sum(dim='time').compute().values
    n = ds.sizes['point']
    ds.close()

    mean = np.average(slope, weights=w)
    std = np.sqrt(np.average((slope - mean) ** 2, weights=w))
    return n, float((mb * w).sum()) / YEARS, mean, std


rows = []
for s in SMOOTHS:
    print(f'\n=== dem_smooth_m = {s} ===', flush=True)
    rows.append((s,) + run(s))

print()
print('=' * 72)
print(f'{GLACIER}, mesh fixed at h={SPACING:.0f} m')
print('=' * 72)
print(f'{"smooth_m":>10} {"N":>7} {"slope mean":>12} {"slope std":>11} '
      f'{"MB/yr":>10} {"vs raw":>9}')
base = rows[0][2]
for s, n, mb, mean, std in rows:
    print(f'{s:>10} {n:>7} {mean:>12.3f} {std:>11.3f} {mb:>10.4f} '
          f'{mb - base:>+9.4f}')
print()
print('N is identical by construction, so any movement in MB is terrain')
print('roughness alone. Flat means terrain is not driving the drift.')
