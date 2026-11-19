"""
Runs the extremes experiment (fires / heatwaves on and off) on the
five calibration glaciers, at the parameters AD_optimize.py calibrated
for each glacier.

SIM_TYPE in the environment selects which of the four runs this is,
so one array job covers the set (see batch_jobs/extremes.sh).

The parameters come from AD_optimize_result.yaml, one (kp, wind_factor)
per glacier. PEBSI wants per-point arrays, and the adaptive mesh decides how many points each
glacier gets -- so the mesh is built once up front to count them
before the parameter arrays are sized to match.
"""
import os
import yaml
import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import simulation as sim
from pebsi.io.terrain import Terrain
from project.parameters import (host, HOST_PATHS, BASE_CONFIG, GLACIERS,
                                translate_rgi, baseline)

# one of 'base', 'no_fires', 'no_heatwaves', 'no_either', 'ssp245_mpi', 'ssp245_gfdl'
sim_type = os.environ['SIM_TYPE']
# turbulent flux scheme: 'bulk' (default) or 'OG' (katabatic, see pebsi/defaults.py);
# non-default schemes get their own output folder so they don't mix with the bulk runs
turb_method = os.environ.get('TURB_METHOD', 'OG_hybrid')
run_name = sim_type if turb_method == 'OG_hybrid' else f'{sim_type}_{turb_method}'
# optional override of the OG katabatic constant (default in pebsi/defaults.py)
og_kappa = os.environ.get('OG_KAPPA')
if og_kappa is not None:
    run_name += f'_k{og_kappa}'
# optional multiplier on the calibrated per-glacier wind factors (e.g. to offset
# the katabatic add-on in 'OG_hybrid' without recalibrating)
wind_scale = os.environ.get('WIND_SCALE')
if wind_scale is not None:
    run_name += f'_ws{wind_scale}'
# 'UKESM' (default) or 'MERRA2': MERRA-2 BC+OC deposition, with the no-fire runs reading the
# wildfire-removed copy in MERRA2/reg01_no_wildfires (extreme_weather_analysis/prepare_no_wildfires.py)
deposition = os.environ.get('DEPOSITION', 'UKESM')
MERRA2_DEP_VARS = ['bcdry', 'bcwet', 'ocdry', 'ocwet']
if deposition == 'MERRA2':
    run_name += '_merra2'
    # the SSP runs only have UKESM future deposition
    assert sim_type not in ('ssp245_mpi', 'ssp245_gfdl'), \
        f'DEPOSITION=MERRA2 is not set up for {sim_type}'

START_DATE = '2000-01-01'
END_DATE = '2025-09-30'
STORE_VARS = ['MB', 'EB', 'climate', 'surftype']  # surftype: 0 snow, 1 firn, 2 ice
if host == 'forge':
    RESULT_FN = '/home/claire/Research/Output/AD_optimize/AD_optimize_result_kappa_kp2.yaml'
    output_fp = f'/home/claire/Research/Output/controlled/{run_name}/'
else:
    RESULT_FN = '/ocean/projects/ees260009p/cwilson4/Output/AD_optimize/AD_optimize_result_kappa_kp2.yaml'
    output_fp = f'/ocean/projects/ees260009p/cwilson4/Output/controlled/{run_name}/'
out_config_fn = f'controlled_{run_name}.yaml'

rgi_ids = [translate_rgi[g]['6'] for g in GLACIERS]


# og_kappa is held fixed during the fit, so it is not in the result YAML (see CALIBRATION_NOTES.md)
OG_KAPPA_FN = os.path.join(os.path.dirname(RESULT_FN), 'og_kappa_kp2.yaml')


def get_best_params():
    """
    Reads the per-glacier (kp, wind_factor, precgrad) from RESULT_FN and og_kappa from
    OG_KAPPA_FN, as a dict keyed by RGI ID.
    """
    with open(RESULT_FN) as f:
        result = yaml.safe_load(f)
    with open(OG_KAPPA_FN) as f:
        og_result = yaml.safe_load(f)
    og_by_glacier = dict(zip(og_result['glaciers'], og_result['og_kappa']))
    return {translate_rgi[g]['6']: (kp, wf, pg, og_by_glacier[g]) for g, kp, wf, pg
            in zip(result['glaciers'], result['kp'], result['wind_factor'], result['precgrad'])}


def build_config(kp=None, wind_factor=None, precgrad=None, og=None):
    """
    Writes the run's config: AD_optimize's configuration without the
    calibration, plus the UKESM deposition and this run's climate.
    Called twice: once with the baseline parameters, to get a mesh to
    count, and again with the per-point arrays.
    """
    configs = dict(BASE_CONFIG)
    configs.update(
        rgi_ids=rgi_ids,
        method_distribute='adaptive',
        start_date=START_DATE,
        end_date=END_DATE,
        output_freq='hourly',
        store_vars=STORE_VARS,
        store_data=True,
        debug=True,
        progress_bar=False,
        kp=baseline['kp'],
        wind_factor=baseline['wind_factor'],
        deposition_data=None if deposition == 'MERRA2' else 'UKESM',
        method_turbulent=turb_method,
        **({'og_kappa': float(og_kappa)} if og_kappa is not None else {}),
        output_fp=output_fp,
        **{k: v for k, v in HOST_PATHS[host].items() if k != 'output_fp'},
    )

    if sim_type == 'base':
        configs['ukesm_fp'] = '../UKESM/dr401_GFED/'

    elif sim_type == 'no_fires':
        configs['ukesm_fp'] = '../UKESM/dw068_nofires/'

    elif sim_type == 'no_heatwaves':
        configs['ukesm_fp'] = '../UKESM/dr401_GFED/'
        configs['secondary_climate_string'] = '_no_heatwaves'
        configs['secondary_vars'] = ['temp', 'SWin', 'LWin', 'rh']

    elif sim_type == 'no_either':
        configs['ukesm_fp'] = '../UKESM/dw068_nofires/'
        configs['secondary_climate_string'] = '_no_heatwaves'
        configs['secondary_vars'] = ['temp', 'SWin', 'LWin', 'rh']

    # future fire deposition; these files end 2014-11-30
    elif sim_type == 'ssp245_mpi':
        configs['ukesm_fp'] = '../UKESM/dt397_SSP245_MPI/'
        configs['end_date'] = '2014-09-01'

    elif sim_type == 'ssp245_gfdl':
        configs['ukesm_fp'] = '../UKESM/dt398_SSP245_GFDL/'
        configs['end_date'] = '2014-09-01'

    else:
        raise ValueError(f'unknown sim_type {sim_type!r}')

    # MERRA-2 deposition: no_fires reads BC/OC from the wildfire-removed folder; no_either reads
    # its heatwave and deposition variables from reg01_no_either (extreme_weather_analysis/prepare_no_either.py)
    if deposition == 'MERRA2' and sim_type == 'no_fires':
        configs['secondary_climate_string'] = '_no_wildfires'
        configs['secondary_vars'] = MERRA2_DEP_VARS
    if deposition == 'MERRA2' and sim_type == 'no_either':
        configs['secondary_climate_string'] = '_no_either'
        configs['secondary_vars'] = ['temp', 'SWin', 'LWin', 'rh'] + MERRA2_DEP_VARS

    if kp is not None:
        configs['kp'] = [float(v) for v in kp]
        configs['wind_factor'] = [float(v) for v in wind_factor]
        configs['precgrad'] = [float(v) for v in precgrad]
        if og_kappa is None:  # OG_KAPPA, if set, overrides the calibrated values
            configs['og_kappa'] = [float(v) for v in og]

    with open(out_config_fn, 'w') as f:
        yaml.dump(configs, f, sort_keys=False)
    return configs


def main():
    best = get_best_params()

    # the adaptive mesh sizes each glacier, so build it before the
    # parameter arrays that have to line up with it point for point
    build_config()
    args = sim.get_args(parse=False).parse_args([])
    args.config_fn = out_config_fn
    terrain = Terrain(sim.PEBSI(args).params)

    rgiid_n = np.asarray(terrain.rgiid_n, dtype=str)
    kp = np.array([best[gid][0] for gid in rgiid_n])
    wind_factor = np.array([best[gid][1] for gid in rgiid_n])
    precgrad = np.array([best[gid][2] for gid in rgiid_n])
    og = np.array([best[gid][3] for gid in rgiid_n])
    if wind_scale is not None:
        wind_factor = wind_factor * float(wind_scale)

    print(f'sim_type  {sim_type}', flush=True)
    print(f'turbulent {turb_method}', flush=True)
    print(f'wind_scale {wind_scale or 1}', flush=True)
    print(f'{terrain.N_POINTS} adaptive points across '
          f'{len(GLACIERS)} glaciers', flush=True)
    for g in GLACIERS:
        gid = translate_rgi[g]['6']
        n = int((rgiid_n == gid).sum())
        kp_g, wf_g = best[gid][:2]
        print(f'  {g:<12} {n:>5} points  kp={kp_g:.2f} '
              f'wind_factor={wf_g:.2f}', flush=True)
    print(f'output_fp {output_fp}', flush=True)

    build_config(kp=kp, wind_factor=wind_factor, precgrad=precgrad, og=og)
    args = sim.get_args(parse=False).parse_args([])
    args.config_fn = out_config_fn

    model = sim.PEBSI(args)
    model.run()


if __name__ == '__main__':
    try:
        main()
    finally:
        # remove config file since it was copied to output directory
        if os.path.exists(out_config_fn):
            os.remove(out_config_fn)
