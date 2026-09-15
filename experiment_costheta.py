"""
Controlled test of whether computing cos(solar incidence) per pixel and
cell-averaging (option_precomputed_costheta=True) moves mass balance
relative to the model's normal order -- cell-mean slope/aspect first,
cos_theta from the means second.

Runs each mesh (h=1200 and h=300, unsmoothed DEM) twice, with the option
on and off, holding everything else fixed. Since both meshes carry the
option through, this also gives the coarse-vs-fine gap under each
ordering directly, comparable to the offline estimate in
check_costheta_order.py and to the DEM-smoothing result (0.0133 of the
0.0244 h=300->h=1200 gap from terrain roughness broadly).

    python experiment_costheta.py

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
SPACINGS = [1200.0, 300.0]
YEARS = 3.0


def run(spacing, precomputed):
    rgi_id = translate_rgi[GLACIER]['6']
    wf = baseline['wind_factor']
    end_date = align_end_date_for_daily_output(START_DATE, END_DATE_RAW)

    tag = f'h{spacing:.0f}_ct{int(precomputed)}'
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
    configs['point_spacing'] = spacing
    configs['option_precomputed_costheta'] = precomputed
    configs['kp'] = baseline['kp']
    configs['wind_factor'] = wf
    configs['store_data'] = True
    configs['store_vars'] = ['mass_balance']
    configs['output_fp'] = fp

    fn = f'_costheta_{tag}.yaml'
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
    n = ds.sizes['point']
    ds.close()
    return n, float((mb * w).sum()) / YEARS


rows = {}
for h in SPACINGS:
    for pc in (False, True):
        print(f'\n=== h={h:.0f}  option_precomputed_costheta={pc} ===', flush=True)
        rows[(h, pc)] = run(h, pc)

print()
print('=' * 72)
print(f'{GLACIER}, cos_theta ordering test')
print('=' * 72)
print(f'{"h":>7} {"N":>7} {"on-the-fly (current)":>22} {"precomputed":>13} {"moved":>9}')
for h in SPACINGS:
    n0, mb0 = rows[(h, False)]
    n1, mb1 = rows[(h, True)]
    print(f'{h:>7.0f} {n0:>7} {mb0:>22.4f} {mb1:>13.4f} {mb1 - mb0:>+9.4f}')

gap_current = rows[(1200.0, False)][1] - rows[(300.0, False)][1]
gap_precomp = rows[(1200.0, True)][1] - rows[(300.0, True)][1]
print()
print(f'h=1200 - h=300 gap, current order:     {gap_current:+.4f} m w.e./yr')
print(f'h=1200 - h=300 gap, precomputed order: {gap_precomp:+.4f} m w.e./yr')
print(f'gap closed: {100 * (1 - abs(gap_precomp) / abs(gap_current)):.1f}%')
print(f'implied share of the 0.0244 m w.e./yr residual: '
      f'{100 * abs(gap_current - gap_precomp) / 0.0244:.1f}%')
