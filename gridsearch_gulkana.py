"""
Grid search over dust_factor, kp, and wind_factor for the small
set of Gulkana sites defined in config.yaml.

Builds a single config file where each site is repeated once per
(dust_factor, kp, wind_factor) combination, so all site x parameter
points are run together as one set of model points.

@author: clairevwilson
"""
import itertools
import yaml
import os

out_config_fn = 'config_gridsearch_gulkana.yaml'

rgi_id = '01.00570'
sites = ['A', 'AU', 'B', 'D', 'T']

dust_factors = [1, 5, 20]
kps = [0.5, 1, 2]
wind_factors = [1, 2, 3]

configs = {}

# PHYSICS
configs['option_ice_albedo_tif'] = True
configs['option_windmaps'] = True
configs['option_accel_grains'] = True
configs['option_flat_plates'] = True
configs['option_dynamics'] = False
configs['constant_freshgrainsize'] = 54.5
configs['precgrad'] = 0.000100

# CONFIGURATION
configs['start_date'] = '2014-03-01 00:00'
configs['end_date'] = '2025-08-20 00:00'
configs['debug'] = True
configs['store_data'] = True
configs['progress_bar'] = False
configs['store_vars'] = ['climate', 'MB', 'EB', 'surface']
configs['bias_vars'] = ['temp']

# FILEPATHS
configs['climate_fp'] = '/ocean/projects/ees260009p/cwilson4/climate_data/'
configs['rgi_fp'] = '/ocean/projects/ees260009p/cwilson4/RGI/rgi60/00_rgi60_attribs/'
configs['output_fp'] = '/ocean/projects/ees260009p/cwilson4/Output/gridsearch_gulkana/'
configs['cop30_vrt_path'] = '/ocean/projects/ees260009p/cwilson4/data/dems/COP30/COP30_reg01.vrt'
configs['shading_fp'] = '/ocean/projects/ees260009p/cwilson4/data/shading/'
configs['ice_albedo_fn'] = '/ocean/projects/ees260009p/cwilson4/data/ice_albedo/{gid}_albedo.tif'
configs['thickness_fn'] = '/ocean/projects/ees260009p/cwilson4/data/ice_thickness/RGI60-01/RGI60-{gid}_thickness.tif'
configs['windmap_fn'] = '/ocean/projects/ees260009p/cwilson4/data/windmapper/{gid}.nc'

param_combos = list(itertools.product(dust_factors, kps, wind_factors))

gids = []
site_list = []
dust_factor_list = []
kp_list = []
wind_factor_list = []

for site in sites:
    for dust_factor, kp, wind_factor in param_combos:
        gids.append(rgi_id)
        site_list.append(site)
        dust_factor_list.append(dust_factor)
        kp_list.append(kp)
        wind_factor_list.append(wind_factor)

# POINTS
configs['rgi_ids'] = gids
configs['sites'] = site_list
configs['dust_factor'] = dust_factor_list
configs['kp'] = kp_list
configs['wind_factor'] = wind_factor_list
configs['n_points'] = len(site_list)

with open(out_config_fn, 'w') as f:
    yaml.dump(configs, f, sort_keys=False)

print(f'Wrote {out_config_fn} with {len(site_list)} points '
      f'({len(sites)} sites x {len(param_combos)} param combos)')

if __name__ == '__main__':
    import simulation as sim

    args = sim.get_args()
    args.config_fn = out_config_fn

    model = sim.PEBSI(args)
    model.run()

    os.remove(out_config_fn)
