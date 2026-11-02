"""
Runs Kennicott through summer 2006 with one synthetic heatwave per run,
from the forcing written by project/synthetic/prepare_synthetic_heatwaves.py
(run 0 has no event). SLURM_ARRAY_TASK_ID selects the run, so one array job
covers the design (see batch_jobs/synthetic_heatwaves.sh).

The run starts 2005-04-01 so the spin-up year, which repeats the first
year of forcing, ends before any event. Kennicott uses the (kp, wind_factor)
AD_optimize.py calibrated for it.
"""
import os
import yaml

import simulation as sim
from project.parameters import host, HOST_PATHS, BASE_CONFIG, translate_rgi

run = int(os.environ['SLURM_ARRAY_TASK_ID'])

GLACIER = 'kennicott'
START_DATE = '2005-04-01'
END_DATE = '2006-10-01'
STORE_VARS = ['mass_balance', 'albedo']
RESULT_FN = '/ocean/projects/ees260009p/cwilson4/Output/AD_optimize/AD_optimize_result.yaml'
output_fp = f'/ocean/projects/ees260009p/cwilson4/Output/synthetic_heatwaves/run_{run:03d}'
out_config_fn = f'synthetic_heatwaves_{run:03d}.yaml'


def get_params():
    """Kennicott's calibrated (kp, wind_factor) from RESULT_FN."""
    with open(RESULT_FN) as f:
        result = yaml.safe_load(f)
    i = result['glaciers'].index(GLACIER)
    return result['kp'][i], result['wind_factor'][i]


def main():
    kp, wind_factor = get_params()
    configs = dict(BASE_CONFIG)
    configs.update(
        rgi_ids=[translate_rgi[GLACIER]['6']],
        method_distribute='adaptive',
        start_date=START_DATE,
        end_date=END_DATE,
        output_freq='hourly',
        store_vars=STORE_VARS,
        store_data=True,
        debug=True,
        progress_bar=False,
        kp=float(kp),
        wind_factor=float(wind_factor),
        deposition_data='UKESM',
        ukesm_fp='../UKESM/dr401_GFED/',
        secondary_climate_string=f'_kennicott_hw{run:03d}',
        secondary_vars=['temp', 'SWin', 'LWin', 'rh', 'tp'],
        output_fp=output_fp,
        **{k: v for k, v in HOST_PATHS[host].items() if k != 'output_fp'},
    )
    with open(out_config_fn, 'w') as f:
        yaml.dump(configs, f, sort_keys=False)

    print(f'run {run:03d}  kp={kp:.2f} wind_factor={wind_factor:.2f}', flush=True)
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
