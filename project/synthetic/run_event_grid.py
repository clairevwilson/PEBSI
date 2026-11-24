"""
Synthetic heatwave or wildfire grid on the five calibration glaciers in summer YEAR, at the
parameters and settings of the MERRA-2 deposition controlled runs (run_controlled_experiment.py with
DEPOSITION=MERRA2), on the no-heatwave forcing. YEAR is the 2000-2025 summer most typical in
temperature with the fewest extreme events, including fire deposition
(extreme_weather_analysis/select_baseline_year.py).

EVENT in the environment selects the experiment ('heatwave' or 'wildfire'), and RUN the run:
  RUN=spinup  April of the year before YEAR to the end of April YEAR, saving the checkpoint every
              event run starts from
  RUN=0       no event
  RUN=1-N     every event center x duration x intensity (heatwaves 1-64, wildfires 1-96)
  RUN=65-80   heatwaves only: every center x duration at PRECIP_INTENSITY, precipitation removed
  then        each glacier's largest real event (REAL_EVENTS), heatwaves 81-85, wildfires 97-101,
              then its most damaging one where different (DAMAGING_EVENTS, none yet)
  RUN=split_0-11  heatwaves only, in split_<n>/: the average heatwave (1x, SPLIT_DURATION days) at each
              center with all its anomalies, only temperature, or only shortwave (split_design), from a
              copy of the anomaly file with the other anomalies zeroed
Each event run copies the spin-up's folder into its own and resumes from it, so its
output.zarr holds the spin-up followed by May-Sep YEAR.

PEBSI adds the event on the fly (heatwave_* / wildfire_* in pebsi/defaults.py), from the
average event of each MERRA-2 cell: heatwaves from extreme_weather_analysis/prepare_no_heatwaves.py
on the MERRA-2 deposition, wildfires from `project/synthetic/prepare_synthetic_wildfires.py anomaly
merra2` on the MERRA-2 no-wildfire deposition (reg01_no_either).
"""
import os
import shutil
import itertools
import yaml
import numpy as np
import pandas as pd
import xarray as xr

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import simulation as sim
from pebsi.defaults import heatwave_anomaly_fn
from pebsi.io.terrain import Terrain
from project.parameters import host, HOST_PATHS, BASE_CONFIG, GLACIERS, translate_rgi, baseline

EVENT = os.environ['EVENT']
RUN = os.environ['RUN']

YEAR = 2021
START_DATE = f'{YEAR - 1}-04-01'
SPINUP_END = f'{YEAR}-04-30 23:00'
END_DATE = f'{YEAR}-09-30 23:00'
STORE_VARS = ['mass_balance', 'accumulation', 'melt', 'albedo', 'snowdepth', 'surftype']  # surftype: 0 snow, 1 firn, 2 ice

CENTERS = [f'{YEAR}-{m:02d}-15' for m in range(5, 9)]
MERRA2_DEP_VARS = ['bcdry', 'bcwet', 'ocdry', 'ocwet']
if EVENT == 'heatwave':
    DURATIONS = [5, 10, 15, 20]  # days
    INTENSITIES = [0.5, 0.8, 1.1, 1.4]  # multiple of each cell's average heatwave; real events reached 0.49-1.37
    PRECIP_INTENSITY = 1.1  # intensity of the precipitation-removed runs
    SPLIT_DURATION = 10  # days, of the split runs
    SPLIT_KEEP = {'all': None, 'temp': 'T2M', 'sw': 'SWGDN'}  # anomaly each split run keeps (None: all)
    # no-heatwave weather on the MERRA-2 deposition (fires kept)
    SECONDARY = dict(secondary_climate_string='_no_heatwaves', secondary_vars=['temp', 'SWin', 'LWin', 'rh'])
elif EVENT == 'wildfire':
    DURATIONS = [3, 6, 10, 15]  # days; real MERRA-2 fires at these cells last up to 8-20 days (90% <= 3-4)
    # multiple of each cell's average fire; 95% of real MERRA-2 events reached <= 2.2-3.2, the largest 5-16
    INTENSITIES = [0.5, 1, 2, 3, 5, 10]
    # no-heatwave weather on the MERRA-2 no-wildfire deposition
    SECONDARY = dict(secondary_climate_string='_no_either',
                     secondary_vars=['temp', 'SWin', 'LWin', 'rh'] + MERRA2_DEP_VARS,
                     wildfire_anomaly_fn='MERRA2/reg01_no_wildfires/wildfire_anomaly.nc')

if host == 'forge':
    RESULT_FN = '/home/claire/Research/Output/AD_optimize/AD_optimize_result_current.yaml'
    OUT_DIR = f'/home/claire/Research/Output/{EVENT}_grid_current/'
else:
    RESULT_FN = '/ocean/projects/ees260009p/cwilson4/Output/AD_optimize/AD_optimize_result_current.yaml'
    OUT_DIR = f'/ocean/projects/ees260009p/cwilson4/Output/{EVENT}_grid_current/'
# og_kappa is held fixed during the fit, so it is not in the result YAML (see CALIBRATION_NOTES.md)
OG_KAPPA_FN = os.path.join(os.path.dirname(RESULT_FN), 'og_kappa_kp2.yaml')
SPINUP_FP = OUT_DIR + 'spinup_0/'  # PEBSI appends _0 to a new output_fp


# each glacier's largest real event of 2000-2025 at the MERRA-2 cell nearest its centroid, by duration x
# intensity (heatwaves) or fire-attributable MERRA-2 deposition (wildfires, runs of the 1-day wildfire mask),
# as (start month-day, duration, intensity);
# heatwaves: gulkana 2004, lemon_creek 2019, wolverine 2019, kennicott 2004, kahiltna 2004;
# wildfires: gulkana 2004, lemon_creek 2004, wolverine 2004, kennicott 2016, kahiltna 2022
REAL_EVENTS = {
    'heatwave': {'gulkana': ('08-13', 12, 1.22), 'lemon_creek': ('06-25', 16, 1.18),
                 'wolverine': ('06-23', 16, 1.62), 'kennicott': ('06-17', 17, 1.29),
                 'kahiltna': ('08-14', 12, 1.37)},
    'wildfire': {'gulkana': ('08-16', 20, 2.09), 'lemon_creek': ('06-26', 6, 5.17),
                 'wolverine': ('06-27', 8, 6.55), 'kennicott': ('07-17', 9, 6.65),
                 'kahiltna': ('07-09', 5, 9.65)},
}[EVENT]

# each glacier's real event with the largest effect estimated from the grid (deposition x the grid's effect
# per mg m-2 at the event's midpoint date), where that differs from REAL_EVENTS; to be filled in from this grid
DAMAGING_EVENTS = {
    'heatwave': {},
    'wildfire': {},
}[EVENT]


def design():
    """
    (center, duration, intensity, remove_precip) of each run; run 0 has no event. After the grid
    come the real events of REAL_EVENTS, then of DAMAGING_EVENTS, inserted into YEAR on the same
    dates, in their order.
    """
    runs = [(None, 0, 0, False)]
    runs += [(c, d, i, False) for c, d, i in itertools.product(CENTERS, DURATIONS, INTENSITIES)]
    if EVENT == 'heatwave':
        runs += [(c, d, PRECIP_INTENSITY, True) for c, d in itertools.product(CENTERS, DURATIONS)]
    for start, d, i in list(REAL_EVENTS.values()) + list(DAMAGING_EVENTS.values()):
        center = str((pd.Timestamp(f'{YEAR}-{start}') + pd.Timedelta(days=d // 2)).date())
        runs.append((center, d, i, False))
    return runs


def split_design():
    """(center, part) of each split run: every center with all anomalies, then only temperature, then only
    shortwave."""
    return [(c, p) for p in SPLIT_KEEP for c in CENTERS]


def split_anomaly_fn(part):
    """heatwave_anomaly_fn of a split run: the average heatwave, or a copy next to it (written if missing)
    keeping only part's anomaly. The temperature-only copy has no shortwave cap, so the baseline shortwave
    stays untouched."""
    if SPLIT_KEEP[part] is None:
        return heatwave_anomaly_fn
    fn = heatwave_anomaly_fn.replace('.nc', f'_{part}_only.nc')
    path = HOST_PATHS[host]['climate_fp'] + fn
    if not os.path.exists(path):
        ds = xr.load_dataset(HOST_PATHS[host]['climate_fp'] + heatwave_anomaly_fn)
        for v in ['T2M', 'SWGDN', 'LWGAB', 'QV2M']:
            if v != SPLIT_KEEP[part]:
                ds[v] = ds[v] * 0
        if part == 'temp':
            ds['SWGDN_cap'] = ds['SWGDN_cap'] * 0 + np.inf
        ds.to_netcdf(path)
    return fn


def build_config(config_fn, end_date, output_fp, params=None, event=None, anomaly_fn=None):
    """Writes the run's config: calibrated parameters (kp, wind_factor, precgrad, og_kappa per point),
    no-heatwave forcing, MERRA-2 deposition and the event."""
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
        kp=baseline['kp'],
        wind_factor=baseline['wind_factor'],
        deposition_data=None,
        method_turbulent='OG_hybrid',
        **SECONDARY,
        output_fp=output_fp,
        **{k: v for k, v in HOST_PATHS[host].items() if k != 'output_fp'},
    )
    if params is not None:
        for k, v in zip(['kp', 'wind_factor', 'precgrad', 'og_kappa'], params):
            configs[k] = [float(x) for x in v]
    if event is not None and event[1] > 0:
        center, duration, intensity, remove_precip = event
        start = str((pd.Timestamp(center) - pd.Timedelta(days=duration // 2)).date())
        if EVENT == 'heatwave':
            configs.update(heatwave_start=start, heatwave_duration=duration,
                           heatwave_intensity=intensity, heatwave_remove_precip=remove_precip)
            if anomaly_fn is not None:
                configs.update(heatwave_anomaly_fn=anomaly_fn)
        else:
            configs.update(wildfire_start=start, wildfire_duration=duration, wildfire_intensity=intensity)
    with open(config_fn, 'w') as f:
        yaml.dump(configs, f, sort_keys=False)


def calibrated_params(config_fn):
    """Per-point (kp, wind_factor, precgrad, og_kappa) from RESULT_FN and OG_KAPPA_FN, on the mesh
    config_fn builds."""
    with open(RESULT_FN) as f:
        result = yaml.safe_load(f)
    with open(OG_KAPPA_FN) as f:
        og = yaml.safe_load(f)
    og = dict(zip(og['glaciers'], og['og_kappa']))
    best = {translate_rgi[g]['6']: (kp, wf, pg, og[g]) for g, kp, wf, pg
            in zip(result['glaciers'], result['kp'], result['wind_factor'], result['precgrad'])}
    args = sim.get_args(parse=False).parse_args([])
    args.config_fn = config_fn
    rgiid_n = np.asarray(Terrain(sim.PEBSI(args).params).rgiid_n, dtype=str)
    return [[best[g][i] for g in rgiid_n] for i in range(4)]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if RUN == 'spinup':
        assert not os.path.exists(SPINUP_FP), f'{SPINUP_FP} exists; delete it to rerun'
        run_fp, end_date, event = OUT_DIR + 'spinup', SPINUP_END, None
        config_fn = f'{EVENT}_grid_{RUN}.yaml'
    else:
        anomaly_fn, end_date = None, END_DATE
        if RUN.startswith('split_'):
            center, part = split_design()[int(RUN.split('_')[1])]
            run_fp, event = OUT_DIR + f'{RUN}/', (center, SPLIT_DURATION, 1.0, False)
            anomaly_fn = split_anomaly_fn(part)
        else:
            run_fp, event = OUT_DIR + f'run_{int(RUN):03d}/', design()[int(RUN)]
        assert not os.path.exists(run_fp), f'{run_fp} exists; delete it to rerun'
        shutil.copytree(SPINUP_FP, run_fp)
        # PEBSI reads <resume_from>/config.yaml when resuming, so this run's config goes there
        config_fn = os.path.join(run_fp, 'config.yaml')

    build_config(config_fn, end_date, run_fp)
    build_config(config_fn, end_date, run_fp, calibrated_params(config_fn), event,
                 None if RUN == 'spinup' else anomaly_fn)
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
