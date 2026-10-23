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

import simulation as sim
from pebsi.io.terrain import Terrain
from project.parameters import (host, HOST_PATHS, BASE_CONFIG, GLACIERS,
                                translate_rgi, baseline)

# one of 'base', 'no_fires', 'no_heatwaves', 'no_either'
sim_type = os.environ['SIM_TYPE']

START_DATE = '1997-04-01'
END_DATE = '2015-09-01'
STORE_VARS = ['MB', 'EB', 'climate']
RESULT_FN = '/ocean/projects/ees260009p/cwilson4/Output/AD_optimize/AD_optimize_result.yaml'
output_fp = f'/ocean/projects/ees260009p/cwilson4/Output/controlled/{sim_type}/'
out_config_fn = f'controlled_{sim_type}.yaml'

rgi_ids = [translate_rgi[g]['6'] for g in GLACIERS]


def get_best_params():
    """
    Reads the per-glacier (kp, wind_factor) from RESULT_FN,
    as a dict keyed by RGI ID.
    """
    with open(RESULT_FN) as f:
        result = yaml.safe_load(f)
    return {translate_rgi[g]['6']: (kp, wf) for g, kp, wf
            in zip(result['glaciers'], result['kp'], result['wind_factor'])}


def build_config(kp=None, wind_factor=None):
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
        deposition_data='UKESM',
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

    else:
        raise ValueError(f'unknown sim_type {sim_type!r}')

    if kp is not None:
        configs['kp'] = [float(v) for v in kp]
        configs['wind_factor'] = [float(v) for v in wind_factor]

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

    print(f'sim_type  {sim_type}', flush=True)
    print(f'{terrain.N_POINTS} adaptive points across '
          f'{len(GLACIERS)} glaciers', flush=True)
    for g in GLACIERS:
        gid = translate_rgi[g]['6']
        n = int((rgiid_n == gid).sum())
        kp_g, wf_g = best[gid]
        print(f'  {g:<12} {n:>5} points  kp={kp_g:.2f} '
              f'wind_factor={wf_g:.2f}', flush=True)
    print(f'output_fp {output_fp}', flush=True)

    build_config(kp=kp, wind_factor=wind_factor)
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
