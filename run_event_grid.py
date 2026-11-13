"""
Synthetic heatwave or wildfire grid on the five calibration glaciers in summer 1999, at the
parameters AD_optimize.py calibrated for each glacier, on the no-heatwave forcing.

EVENT in the environment selects the experiment ('heatwave' or 'wildfire'), and RUN the run:
  RUN=spinup  1998-04-01 to 1999-04-30, saving the checkpoint every event run starts from
  RUN=0       no event
  RUN=1-64    every event center x duration x intensity
  RUN=65-80   heatwaves only: every center x duration at PRECIP_INTENSITY, precipitation removed
  then        each glacier's largest real event (REAL_EVENTS), heatwaves 81-85, wildfires 65-69,
              then its most damaging one where different (DAMAGING_EVENTS), wildfires 70-73
Each event run copies the spin-up's folder into its own and resumes from it, so its
output.zarr holds the spin-up followed by May-Sep 1999.

PEBSI adds the event on the fly (heatwave_* / wildfire_* in pebsi/defaults.py), from the
average event of each MERRA-2 cell: heatwaves from extreme_weather_analysis/prepare_no_heatwaves.py
on the GFED deposition, wildfires from project/synthetic/prepare_synthetic_wildfires.py on the
UK-ESM no-fires deposition.
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

EVENT = os.environ['EVENT']
RUN = os.environ['RUN']

START_DATE = '1998-04-01'
SPINUP_END = '1999-04-30 23:00'
END_DATE = '1999-09-30 23:00'
STORE_VARS = ['mass_balance', 'accumulation', 'melt', 'albedo', 'snowdepth']

CENTERS = ['1999-05-15', '1999-06-15', '1999-07-15', '1999-08-15']
if EVENT == 'heatwave':
    DURATIONS = [5, 10, 15, 20]  # days
    INTENSITIES = [0.5, 0.8, 1.1, 1.4]  # multiple of each cell's average heatwave; real events reached 0.49-1.37
    PRECIP_INTENSITY = 1.1  # intensity of the precipitation-removed runs
    UKESM_FP = '../UKESM/dr401_GFED/'
elif EVENT == 'wildfire':
    DURATIONS = [3, 6, 10, 15]  # days; real fires at these cells lasted 3-16 days, median 3-4
    INTENSITIES = [0.5, 1, 2, 3]  # multiple of each cell's average fire; real events reached 0.32-2.8
    UKESM_FP = '../UKESM/dw068_nofires/'

if host == 'forge':
    RESULT_FN = '/home/claire/Research/Output/AD_optimize_result.yaml'
    OUT_DIR = f'/home/claire/Research/Output/{EVENT}_grid/'
else:
    RESULT_FN = '/ocean/projects/ees260009p/cwilson4/Output/AD_optimize/AD_optimize_result.yaml'
    OUT_DIR = f'/ocean/projects/ees260009p/cwilson4/Output/{EVENT}_grid/'
SPINUP_FP = OUT_DIR + 'spinup_0/'  # PEBSI appends _0 to a new output_fp


# each glacier's largest real event of 1997-2015 at the MERRA-2 cell nearest its centroid, by duration x
# intensity (heatwaves) or fire-attributable deposition (wildfires), as (start month-day, duration, intensity);
# heatwaves: gulkana 2004, lemon_creek 2015, wolverine 1997, kennicott 2004, kahiltna 2004;
# wildfires: gulkana 2004, lemon_creek 2005, wolverine 2010, kennicott 2009, kahiltna 2015
REAL_EVENTS = {
    'heatwave': {'gulkana': ('08-13', 12, 1.21), 'lemon_creek': ('05-12', 12, 1.15),
                 'wolverine': ('06-21', 10, 1.32), 'kennicott': ('06-17', 17, 1.29),
                 'kahiltna': ('08-14', 12, 1.37)},
    'wildfire': {'gulkana': ('08-15', 16, 2.10), 'lemon_creek': ('08-14', 5, 3.37),
                 'wolverine': ('07-24', 8, 2.56), 'kennicott': ('07-05', 16, 1.59),
                 'kahiltna': ('07-18', 15, 2.73)},
}[EVENT]

# each glacier's real event of 1997-2015 with the largest effect estimated from the grid (deposition x the
# grid's effect per mg m-2 at the event's midpoint date), where that differs from REAL_EVENTS (wildfires
# only: gulkana 2004, kahiltna 2015, wolverine 2003, lemon_creek 2004)
DAMAGING_EVENTS = {
    'heatwave': {},
    'wildfire': {'gulkana': ('06-26', 11, 2.36), 'kahiltna': ('06-16', 11, 3.62),
                 'wolverine': ('05-08', 5, 0.94), 'lemon_creek': ('07-07', 7, 1.45)},
}[EVENT]


def design():
    """
    (center, duration, intensity, remove_precip) of each run; run 0 has no event. After the grid
    come the real events of REAL_EVENTS, then of DAMAGING_EVENTS, inserted into 1999 on the same
    dates, in their order.
    """
    runs = [(None, 0, 0, False)]
    runs += [(c, d, i, False) for c, d, i in itertools.product(CENTERS, DURATIONS, INTENSITIES)]
    if EVENT == 'heatwave':
        runs += [(c, d, PRECIP_INTENSITY, True) for c, d in itertools.product(CENTERS, DURATIONS)]
    for start, d, i in list(REAL_EVENTS.values()) + list(DAMAGING_EVENTS.values()):
        center = str((pd.Timestamp(f'1999-{start}') + pd.Timedelta(days=d // 2)).date())
        runs.append((center, d, i, False))
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
        ukesm_fp=UKESM_FP,
        secondary_climate_string='_no_heatwaves',
        secondary_vars=['temp', 'SWin', 'LWin', 'rh'],
        output_fp=output_fp,
        **{k: v for k, v in HOST_PATHS[host].items() if k != 'output_fp'},
    )
    if event is not None and event[1] > 0:
        center, duration, intensity, remove_precip = event
        start = str((pd.Timestamp(center) - pd.Timedelta(days=duration // 2)).date())
        if EVENT == 'heatwave':
            configs.update(heatwave_start=start, heatwave_duration=duration,
                           heatwave_intensity=intensity, heatwave_remove_precip=remove_precip)
        else:
            configs.update(wildfire_start=start, wildfire_duration=duration, wildfire_intensity=intensity)
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
        config_fn = f'{EVENT}_grid_{RUN}.yaml'
    else:
        run_fp, end_date, event = OUT_DIR + f'run_{int(RUN):03d}/', END_DATE, design()[int(RUN)]
        assert not os.path.exists(run_fp), f'{run_fp} exists; delete it to rerun'
        shutil.copytree(SPINUP_FP, run_fp)
        # PEBSI reads <resume_from>/config.yaml when resuming, so this run's config goes there
        config_fn = os.path.join(run_fp, 'config.yaml')

    build_config(config_fn, end_date, run_fp)
    kp, wind_factor = calibrated_params(config_fn)
    build_config(config_fn, end_date, run_fp, kp, wind_factor, event)
    print(f'{EVENT} run {RUN}  event {event}', flush=True)

    args = sim.get_args(parse=False).parse_args([])
    args.config_fn = config_fn
    if RUN != 'spinup':
        args.resume_from = run_fp
    sim.PEBSI(args).run()

    if RUN == 'spinup':
        os.remove(config_fn)


if __name__ == '__main__':
    main()
