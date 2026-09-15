"""
Control test: forces the precomputed cos_theta table to be built from
CELL-MEAN slope/aspect -- i.e. exactly what the on-the-fly (current)
path computes -- then runs it through the real table -> forcings ->
get_SW pipeline with option_precomputed_costheta=True.

If the plumbing (table gathering, shade_idx indexing, the get_SW
branch) is clean, this must reproduce the on-the-fly run's mass balance
almost exactly, since the two paths would be computing the identical
cos_theta by construction. Any mismatch here is a wiring bug, separate
from whether pixel-level averaging is itself correct.

    python control_costheta_plumbing.py
"""
import glob
import os
import shutil

import numpy as np
import xarray as xr
import yaml

import simulation as sim
from AD_optimize import BASE_CONFIG, HOST_PATHS, host, baseline
from pebsi.io.terrain import Terrain
from point_density_convergence import OUTDIR, START_DATE, END_DATE_RAW
from project.bayes_calibrate import align_end_date_for_daily_output
from project.glacierwide_loss import translate_rgi

GLACIER = 'gulkana'
SPACING = 300.0
YEARS = 3.0

# monkeypatch: build the table from the point's own cell-mean slope/aspect,
# exactly matching the on-the-fly formula in energybalance.py's else-branch
def load_cos_theta_control(self):
    slope_r = np.deg2rad(self.slope_n)[:, None]
    aspect_r = np.deg2rad(self.aspect_n)[:, None]
    zen, az = self.solar_zenith, self.solar_azimuth
    self.cos_theta_table = (np.cos(zen) * np.cos(slope_r)
                            + np.sin(zen) * np.sin(slope_r)
                            * np.cos(az - aspect_r))
    return

Terrain.load_cos_theta = load_cos_theta_control


def run(precomputed):
    rgi_id = translate_rgi[GLACIER]['6']
    wf = baseline['wind_factor']
    end_date = align_end_date_for_daily_output(START_DATE, END_DATE_RAW)

    tag = f'h{SPACING:.0f}_control{int(precomputed)}'
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
    configs['option_precomputed_costheta'] = precomputed
    configs['kp'] = baseline['kp']
    configs['wind_factor'] = wf
    configs['store_data'] = True
    configs['store_vars'] = ['mass_balance']
    configs['output_fp'] = fp

    fn = f'_control_{tag}.yaml'
    with open(fn, 'w') as f:
        yaml.dump(configs, f, sort_keys=False)
    args = sim.get_args(parse=False).parse_args([])
    args.config_fn = fn
    model = sim.PEBSI(args)
    model.run()
    os.remove(fn)

    w = model.terrain.weight_n
    dirs = sorted(glob.glob(fp + '_*'))
    assert dirs, f'no output for {fp}'
    ds = xr.open_zarr(os.path.join(dirs[-1], 'output.zarr'))
    mb = ds.mass_balance.sum(dim='time').compute().values
    ds.close()
    return float((mb * w).sum()) / YEARS


print(f'=== on-the-fly (option_precomputed_costheta=False) ===', flush=True)
mb_onfly = run(False)
print(f'\n=== precomputed table, but built from cell-mean slope/aspect '
      f'(should match) ===', flush=True)
mb_control = run(True)

print()
print('=' * 70)
print(f'on-the-fly:                     {mb_onfly:+.4f} m w.e./yr')
print(f'precomputed (cell-mean control): {mb_control:+.4f} m w.e./yr')
print(f'difference:                     {mb_control - mb_onfly:+.6f} m w.e./yr')
print()
print('This should be ~0 (float-precision only). If it is not, the bug is')
print('in the table/forcings/get_SW plumbing, not in pixel-level averaging.')
print()
print('For reference, known values from experiment_costheta.py at h=300:')
print('  on-the-fly:            -1.1962')
print('  precomputed (real):    -1.0184')
