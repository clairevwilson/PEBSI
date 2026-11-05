"""
Synthetic heatwaves and wildfires at Kennicott with fixed timing: every event lasts DURATION days
from one of START_DATES, and only its intensity varies, so at each start date the runs differ in
event size alone (the same days, so the same precipitation removed for heatwaves). Run 0 has no
event; runs 1-30 go through START_DATES in order, with INTENSITIES at each. Heatwave runs 31-33
have zero intensity at each start date: they remove the precipitation over the event days but add
no warming, so subtracting them leaves the loss from the warming alone.

The forcing and deposition are built exactly as in prepare_synthetic_heatwaves.py and
prepare_synthetic_wildfires.py, only from this design and into their own folders.
"""
import os
import numpy as np
import pandas as pd

import prepare_synthetic_heatwaves as hw
import prepare_synthetic_wildfires as wf

START_DATES = ['2006-06-15', '2006-07-15', '2006-08-15']
DURATION = 10  # days
HW_INTENSITIES = np.linspace(0.5, 2, 10)  # multiple of the average Kennicott heatwave
WF_INTENSITIES = np.geomspace(0.5, 10, 10)  # multiple of the average Kennicott fire

HW_RUN_FP = hw.ROOT + 'reg01_kennicott_hwfixed{run:03d}/{v}_reg01_kennicott_hwfixed{run:03d}.zarr'
WF_RUN_FP = wf.ROOT + 'UKESM/kennicott_firefixed{run:03d}/sum_{sp}_{t}deposition_kgm-2s-1.nc'
HW_OUT_DIR = '/ocean/projects/ees260009p/cwilson4/Output/fixed_heatwaves/'
WF_OUT_DIR = '/ocean/projects/ees260009p/cwilson4/Output/fixed_wildfires/'


def fixed_design(intensities):
    """Run 0 without an event, then every start date at every intensity."""
    events = pd.DataFrame([(pd.Timestamp(s), DURATION, k) for s in START_DATES for k in intensities],
                          columns=['start', 'duration', 'intensity'])
    no_event = pd.DataFrame({'start': [pd.NaT], 'duration': [0], 'intensity': [0.0]})
    design = pd.concat([no_event, events], ignore_index=True)
    design.index.name = 'run'
    return design


def heatwave_design():
    """The fixed design, then one zero-intensity run per start date."""
    no_warming = pd.DataFrame([(pd.Timestamp(s), DURATION, 0.0) for s in START_DATES],
                              columns=['start', 'duration', 'intensity'])
    design = pd.concat([fixed_design(HW_INTENSITIES), no_warming], ignore_index=True)
    design.index.name = 'run'
    return design


if __name__ == '__main__':
    os.makedirs(HW_OUT_DIR, exist_ok=True)
    os.makedirs(WF_OUT_DIR, exist_ok=True)
    hw.main(heatwave_design(), HW_RUN_FP, os.path.join(HW_OUT_DIR, 'fixed_heatwave_design.csv'))
    wf.main(fixed_design(WF_INTENSITIES), WF_RUN_FP, os.path.join(WF_OUT_DIR, 'fixed_wildfire_design.csv'))
