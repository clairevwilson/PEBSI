"""
Tests whether the melt-albedo feedback is what amplifies the cos_theta
averaging-order effect from a modest direct SW change into a large mass
balance shift.

Monkeypatches albedo.get_albedo to a no-op: every point's albedo stays
frozen at its initial value (params.albedo_fresh_snow) for the whole
run, so accumulated melt can no longer expose bare ice / darker firn and
darken the surface. The cos_theta -> SWin -> melt pathway is otherwise
untouched.

Runs current vs precomputed cos_theta order at h=300, both with albedo
frozen, and compares the resulting gap to the real (unfrozen) gap of
0.1778 m w.e./yr already established in experiment_costheta.py. A gap
that shrinks a lot under frozen albedo confirms the feedback is doing
most of the amplification; a gap that stays close to 0.1778 says it
isn't.

    python control_albedo_feedback.py
"""
import glob
import os
import shutil

import xarray as xr
import yaml

import simulation as sim
import pebsi.physics.albedo as albedo_mod
from AD_optimize import BASE_CONFIG, HOST_PATHS, host, baseline
from point_density_convergence import OUTDIR, START_DATE, END_DATE_RAW
from project.bayes_calibrate import align_end_date_for_daily_output
from project.glacierwide_loss import translate_rgi

GLACIER = 'gulkana'
SPACING = 300.0
YEARS = 3.0


def frozen_albedo(state, params, forcings):
    """No-op: every point keeps its initial albedo for the whole run."""
    return state.albedo, state.annual_min_albedo


albedo_mod.get_albedo = frozen_albedo


def run(precomputed):
    rgi_id = translate_rgi[GLACIER]['6']
    wf = baseline['wind_factor']
    end_date = align_end_date_for_daily_output(START_DATE, END_DATE_RAW)

    tag = f'h{SPACING:.0f}_frozenalb{int(precomputed)}'
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

    fn = f'_frozenalb_{tag}.yaml'
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


print('=== frozen albedo, current (on-the-fly) cos_theta ===', flush=True)
mb_current = run(False)
print('\n=== frozen albedo, precomputed cos_theta ===', flush=True)
mb_precomp = run(True)

print()
print('=' * 70)
print('ALBEDO-FROZEN CONTROL')
print('=' * 70)
print(f'current (frozen albedo):     {mb_current:+.4f} m w.e./yr')
print(f'precomputed (frozen albedo): {mb_precomp:+.4f} m w.e./yr')
gap_frozen = mb_precomp - mb_current
print(f'gap, albedo frozen:  {gap_frozen:+.4f} m w.e./yr')
print()
print(f'for reference, gap with albedo feedback ON (h=300, from '
      f'verify_costheta_gap.py): {-1.018432 - (-1.196241):+.4f} m w.e./yr')
print()
print(f'fraction of the feedback-on gap that survives with albedo frozen: '
      f'{100 * abs(gap_frozen) / abs(-1.018432 - (-1.196241)):.1f}%')
print('A small surviving fraction confirms the albedo feedback -- not the')
print('direct SW change -- does most of the amplifying.')
