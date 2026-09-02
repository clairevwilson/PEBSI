"""
Timing test: ONE (kp, wind_factor) parameter set (no walker/grid
batching) at 30,000 scattered points, run for 6 months. temporal_chunk_years
in BASE_CONFIG is 1, so a 6-month run is exactly one chunk -- gives a
clean "time per chunk at this point count" number that should scale
~linearly to the full ~26-chunk, 25.7-year run, without having to
actually run the full 25.7 years to find out.

@author: clairevwilson
"""
import os
import yaml

import simulation as sim
from project.bayes_calibrate import BASE_CONFIG, baseline, align_end_date_for_daily_output
from project.scatter_sites import get_scattered_sites
from project.glacierwide_loss import translate_rgi

GLACIER = 'gulkana'
N_POINTS = 30000

if __name__ == '__main__':
    rgi_id = translate_rgi[GLACIER]['6']
    site_names, metadata_fn = get_scattered_sites(GLACIER, rgi_id, N_POINTS)

    start_date = '2019-01-01 00:00'
    end_date = align_end_date_for_daily_output(start_date, '2021-01-01 00:00')

    configs = dict(BASE_CONFIG)
    
    configs['n_spinup_years'] = 0 
    configs['temporal_chunk_years'] = 5

    configs['start_date'] = start_date
    configs['end_date'] = end_date
    configs['metadata_fn'] = metadata_fn
    configs['output_fp'] = '/ocean/projects/ees260009p/cwilson4/Output/bayes_pointscaletest/'
    configs['sites'] = site_names
    configs['rgi_ids'] = [rgi_id] * len(site_names)
    configs['n_points'] = len(site_names)
    configs['kp'] = baseline['kp']
    configs['wind_factor'] = baseline['wind_factor']

    out_config_fn = '_bayes_pointscaletest.yaml'
    with open(out_config_fn, 'w') as f:
        yaml.dump(configs, f, sort_keys=False)

    args = sim.get_args()
    args.config_fn = out_config_fn
    sim.PEBSI(args).run()
    os.remove(out_config_fn)
