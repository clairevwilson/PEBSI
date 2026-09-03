"""
Shared config for Bayesian calibration of (kp, wind_factor): glacier
list, parameter bounds/baseline, the physics/filepath config block
every calibration run builds on, and the daily-output date-alignment
helper. Used by project/grid_calibrate.py (the grid-based calibration
pipeline) and bayes_pointscaletest.py (timing tests).

@author: clairevwilson
"""
import pandas as pd

from project.glacierwide_loss import translate_rgi

OUTPUT_ROOT = '/ocean/projects/ees260009p/cwilson4/Output/bayes_calibrate/'

GLACIERS = ['gulkana', 'kahiltna', 'kennicott', 'wolverine', 'lemon_creek']

# baseline/bounds mirror sensitivity_gulkana.py's kp/wind_factor entries
baseline = {'kp': 2.5, 'wind_factor': 3}
bounds = {'kp': (0.5, 4), 'wind_factor': (0.5, 5)}

BASE_CONFIG = {
    # PHYSICS
    'option_ice_albedo_tif': True,
    'option_windmaps': True,
    'option_accel_grains': True,
    'option_flat_plates': True,
    'option_dynamics': False,
    'constant_freshgrainsize': 54.5,
    'constant_irrwater': True,
    'precgrad': 0.000100,

    # CONFIGURATION
    'debug': False,
    'store_data': True,
    'progress_bar': False,
    'output_freq': 'daily',
    'temporal_chunk_years': 1,
    'store_vars': ['mass_balance', 'albedo', 'surftype', 'total_water'],
    'bias_vars': ['temp'],

    # FILEPATHS
    'climate_fp': '/ocean/projects/ees260009p/cwilson4/climate_data/',
    'rgi_fp': '/ocean/projects/ees260009p/cwilson4/RGI/rgi60/00_rgi60_attribs/',
    'cop30_vrt_path': '/ocean/projects/ees260009p/cwilson4/data/dems/COP30/COP30_reg01.vrt',
    'shading_fp': '/ocean/projects/ees260009p/cwilson4/data/shading/',
    'ice_albedo_fn': '/ocean/projects/ees260009p/cwilson4/data/ice_albedo/{gid}_albedo.tif',
    'thickness_fn': '/ocean/projects/ees260009p/cwilson4/data/ice_thickness/RGI60-01/RGI60-{gid}_thickness.tif',
    'windmap_fn': '/ocean/projects/ees260009p/cwilson4/data/windmapper/{gid}.nc',
}


def align_end_date_for_daily_output(start_date, end_date):
    """
    output_freq='daily' requires len(date_range(start_date, end_date,
    freq='h')) (inclusive) to be a multiple of 24 (pebsi/config.py's
    validate_config). Nudges end_date backward by the minimum number of
    hours needed to satisfy that, rather than requiring every caller to
    hand-align it (e.g. same start/end time-of-day never satisfies this,
    since the inclusive count is always 24*D + 1).
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    hours_between = int((end - start) / pd.Timedelta(hours=1))
    remainder = (hours_between + 1) % 24
    if remainder:
        end -= pd.Timedelta(hours=remainder)
    return str(end)
