"""
Synthetic heatwave grid on the five calibration glaciers in summer 1999, at the
parameters AD_optimize.py calibrated for each glacier, on the no-heatwave forcing.

RUN in the environment selects the run:
  RUN=spinup  1998-04-01 to 1999-04-30, saving the checkpoint every event run starts from
  RUN=0       no event
  RUN=1-64    every event center x duration x intensity, precipitation kept
  RUN=65-80   every event center x duration at intensity 1, precipitation removed
Each event run copies the spin-up's folder into its own and resumes from it, so its
output.zarr holds the spin-up followed by May-Sep 1999.

The heatwave is added by PEBSI on the fly (heatwave_* in pebsi/defaults.py), from the
average heatwave of each MERRA-2 cell (extreme_weather_analysis/prepare_no_heatwaves.py).
"""
import os
import shutil
import itertools
import yaml
import numpy as np
import pandas as pd

import simulation as sim
from pebsi.io.terrain import Terrain
from project.parameters import host, HOST_PATHS, BASE_CONFIG, GLACIERS, translate_rgi, baseline

RUN = os.environ['RUN']

START_DATE = '1998-04-01'
SPINUP_END = '1999-04-30 23:00'
END_DATE = '1999-09-30 23:00'
STORE_VARS = ['mass_balance', 'accumulation', 'melt', 'albedo', 'snowdepth']

CENTERS = ['1999-05-15', '1999-06-15', '1999-07-15', '1999-08-15']
DURATIONS = [5, 10, 15, 20]  # days
INTENSITIES = [0.5, 1, 1.5, 2]  # multiple of each cell's average heatwave

if host == 'forge':
    RESULT_FN = '/home/claire/Research/Output/controlled/AD_optimize_result_used_by_base_0.yaml'
    OUT_DIR = '/home/claire/Research/Output/heatwave_grid/'
else:
    RESULT_FN = '/ocean/projects/ees260009p/cwilson4/Output/AD_optimize/AD_optimize_result.yaml'
    OUT_DIR = '/ocean/projects/ees260009p/cwilson4/Output/heatwave_grid/'
SPINUP_FP = OUT_DIR + 'spinup_0/'  # PEBSI appends _0 to a new output_fp


def design():
    """(center, duration, intensity, remove_precip) of each run; run 0 has no event."""
    runs = [(None, 0, 0, False)]
    runs += [(c, d, i, False) for c, d, i in itertools.product(CENTERS, DURATIONS, INTENSITIES)]
    runs += [(c, d, 1, True) for c, d in itertools.product(CENTERS, DURATIONS)]
    return runs


def build_config(config_fn, end_date, output_fp, kp=None, wind_factor=None, event=None):
    """Writes the run's config: calibrated parameters, no-heatwave forcing and the event."""
    configs = dict(BASE_CONFIG)
    configs.update(
        rgi_ids=[translate_rgi[g]['6'] for g in GLACIERS],
        method_distribute='adaptive',
        start_date=START_DATE,
        end_date=end_date,
        output_freq='daily',
        store_vars=STORE_VARS,
        store_data=True,
        debug=True,
        progress_bar=False,
        kp=baseline['kp'] if kp is None else [float(v) for v in kp],
        wind_factor=baseline['wind_factor'] if wind_factor is None else [float(v) for v in wind_factor],
        deposition_data='UKESM',
        ukesm_fp='../UKESM/dr401_GFED/',
        secondary_climate_string='_no_heatwaves',
        secondary_vars=['temp', 'SWin', 'LWin', 'rh'],
        output_fp=output_fp,
        **{k: v for k, v in HOST_PATHS[host].items() if k != 'output_fp'},
    )
    if event is not None and event[1] > 0:
        center, duration, intensity, remove_precip = event
        configs.update(
            heatwave_start=str((pd.Timestamp(center) - pd.Timedelta(days=duration // 2)).date()),
            heatwave_duration=duration,
            heatwave_intensity=intensity,
            heatwave_remove_precip=remove_precip,
        )
    with open(config_fn, 'w') as f:
        yaml.dump(configs, f, sort_keys=False)


def calibrated_params(config_fn):
    """Per-point (kp, wind_factor) from RESULT_FN, on the mesh config_fn builds."""
    with open(RESULT_FN) as f:
        result = yaml.safe_load(f)
    best = {translate_rgi[g]['6']: (kp, wf) for g, kp, wf
            in zip(result['glaciers'], result['kp'], result['wind_factor'])}
    args = sim.get_args(parse=False).parse_args([])
    args.config_fn = config_fn
    rgiid_n = np.asarray(Terrain(sim.PEBSI(args).params).rgiid_n, dtype=str)
    return [best[g][0] for g in rgiid_n], [best[g][1] for g in rgiid_n]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if RUN == 'spinup':
        assert not os.path.exists(SPINUP_FP), f'{SPINUP_FP} exists; delete it to rerun'
        run_fp, end_date, event = OUT_DIR + 'spinup', SPINUP_END, None
        config_fn = f'heatwave_grid_{RUN}.yaml'
    else:
        run_fp, end_date, event = OUT_DIR + f'run_{int(RUN):03d}/', END_DATE, design()[int(RUN)]
        assert not os.path.exists(run_fp), f'{run_fp} exists; delete it to rerun'
        shutil.copytree(SPINUP_FP, run_fp)
        # PEBSI reads <resume_from>/config.yaml when resuming, so this run's config goes there
        config_fn = os.path.join(run_fp, 'config.yaml')

    build_config(config_fn, end_date, run_fp)
    kp, wind_factor = calibrated_params(config_fn)
    build_config(config_fn, end_date, run_fp, kp, wind_factor, event)
    print(f'run {RUN}  event {event}', flush=True)

    args = sim.get_args(parse=False).parse_args([])
    args.config_fn = config_fn
    if RUN != 'spinup':
        args.resume_from = run_fp
    sim.PEBSI(args).run()

    if RUN == 'spinup':
        os.remove(config_fn)


if __name__ == '__main__':
    main()
