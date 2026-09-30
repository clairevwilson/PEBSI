"""
Runs the extremes experiment (fires / heatwaves on and off) on the
five calibration glaciers, at the parameters the loss surface grid
search picked out for each glacier.

SIM_TYPE in the environment selects which of the four runs this is,
so one array job covers the set (see batch_jobs/extremes.sh).

The parameters come from best.zarr, which compute_best_zarr.py builds:
every point in it carries the (kp, wind_factor) of its own glacier's
best combination. Those are per-glacier values, but PEBSI wants
per-point arrays, and the adaptive mesh decides how many points each
glacier gets -- so the mesh is built once up front to count them
before the parameter arrays are sized to match.

@author: clairevwilson
"""
import os
import yaml
import numpy as np
import xarray as xr

import simulation as sim
from pebsi.io.terrain import Terrain
from project.parameters import (host, HOST_PATHS, BASE_CONFIG,
                                GLACIERS, translate_rgi)

RUN = os.environ.get('RUN', 'controlled')
END_DATE = os.environ.get('END_DATE', '2015-09-01')
DEPOSITION = os.environ.get('DEPOSITION', 'UKESM')
BEST_FN = '/ocean/projects/ees260009p/cwilson4/Output/AD_optimize_2/best.zarr'

sim_type = os.environ.get('SIM_TYPE', 'base')
output_fp = f'/ocean/projects/ees260009p/cwilson4/Output/{RUN}/{sim_type}/'
out_config_fn = f'{sim_type}.yaml'

rgi_ids = [translate_rgi[g]['6'] for g in GLACIERS]


def get_best_params():
    """
    Reads the per-glacier (kp, wind_factor) out of best.zarr.

    Returns a dict keyed by RGI ID. Every point of a glacier in
    best.zarr shares that glacier's pair, so each one collapses to
    a single value.
    """
    ds = xr.open_zarr(BEST_FN, consolidated=False)
    rgiid = np.asarray(ds['rgiid'].values, dtype=str)
    kp = np.asarray(ds['kp'].values, dtype=float)
    wf = np.asarray(ds['wind_factor'].values, dtype=float)

    params = {}
    for gid in rgi_ids:
        mask = rgiid == gid
        assert mask.any(), f'{gid} has no points in {BEST_FN}'
        kp_g = np.unique(kp[mask])
        wf_g = np.unique(wf[mask])
        assert kp_g.size == 1 and wf_g.size == 1, \
            f'{gid} carries more than one parameter pair in {BEST_FN}'
        params[gid] = (float(kp_g[0]), float(wf_g[0]))
    return params


def build_config(kp=None, wind_factor=None):
    """
    Writes the run's config. Called twice: once without parameters,
    to get a mesh to count, and again with the per-point arrays.
    """
    configs = dict(BASE_CONFIG)
    configs.update(HOST_PATHS[host])

    # POINTS
    configs['rgi_ids'] = rgi_ids
    configs['method_distribute'] = 'adaptive'

    # PHYSICS
    configs['option_ice_thickness_tif'] = False 
    configs['initial_ice_depth'] = 200

    # CONFIGURATION
    configs['start_date'] = '1997-04-01'
    configs['end_date'] = END_DATE
    configs['debug'] = True
    configs['store_data'] = True
    configs['progress_bar'] = False
    configs['store_vars'] = ['MB','EB','climate']

    # None falls back to MERRA-2 deposition
    configs['deposition_data'] = None if DEPOSITION == 'MERRA2' else DEPOSITION

    # SPECIAL SIMS
    if sim_type == 'base':
        configs['ukesm_fp'] = '../UKESM/dr401_GFED/'

    elif sim_type == 'no_fires':
        configs['ukesm_fp'] = '../UKESM/dw068_nofires/'

    elif sim_type == 'no_heatwaves':
        configs['ukesm_fp'] = '../UKESM/dr401_GFED/'
        configs['secondary_climate_string'] = '_no_heatwaves'
        configs['secondary_vars'] = ['temp', 'SWin', 'LWin']

    elif sim_type == 'no_either':
        configs['ukesm_fp'] = '../UKESM/dw068_nofires/'
        configs['secondary_climate_string'] = '_no_heatwaves'
        configs['secondary_vars'] = ['temp', 'SWin', 'LWin']

    else:
        raise ValueError(f'unknown SIM_TYPE {sim_type!r}')

    # after the splat, so this run's directory wins over the host default
    configs['output_fp'] = output_fp

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

    print(f'SIM_TYPE  {sim_type}', flush=True)
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
