"""
Grid search over wind_factor and precipitation factor (kp) for
every site in project/sites.pkl.

Builds a single config file where each site is repeated once per
(wind_factor, kp) combination, so all site x parameter points are
run together as one set of model points.

@author: clairevwilson
"""
import itertools
import pickle
import yaml
import os

sites_fn = 'project/sites.pkl'
out_config_fn = 'config_gridsearch.yaml'

wind_factors = [0.3, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]
kps = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5]

# initialize parameter storage
configs = {}

# PHYSICS
configs['option_accel_grains'] = True 
configs['option_flat_plates'] = True
configs['option_ice_albedo_tif'] = True
configs['constant_freshgrainsize'] = 54.5

# CONFIGURATION
configs['start_date'] = '2000-04-01'
configs['end_date'] = '2025-09-01'
configs['temporal_chunk_years'] = 4
configs['debug'] = True
configs['store_data'] = True
configs['progress_bar'] = False
configs['store_vars'] = ['minimal']
configs['bias_vars'] = ['temp']

# FILEPATHS
configs['climate_fp'] = '/ocean/projects/ees260009p/cwilson4/climate_data/'
configs['rgi_fp'] = '/ocean/projects/ees260009p/cwilson4/RGI/rgi60/00_rgi60_attribs/'
configs['output_fp'] = '/ocean/projects/ees260009p/cwilson4/Output/gridsearch/'
configs['cop30_vrt_path'] = '/ocean/projects/ees260009p/cwilson4/data/dems/COP30/COP30_reg01.vrt'
configs['shading_fp'] = '/ocean/projects/ees260009p/cwilson4/data/shading/'
configs['ice_albedo_fn'] = '/ocean/projects/ees260009p/cwilson4/data/ice_albedo/{gid}_albedo.tif'

# load the sites: dict of {rgi_id: [site, site, ...]}
with open(sites_fn, 'rb') as f:
    sites_by_id = pickle.load(f)

param_combos = list(itertools.product(wind_factors, kps))

gids = []
sites = []
wind_factor_list = []
kp_list = []

for gid, site_list in sites_by_id.items():
    for site in site_list:
        for wind_factor, kp in param_combos:
            gids.append(gid)
            sites.append(site)
            wind_factor_list.append(wind_factor)
            kp_list.append(kp)

# POINTS
configs['rgi_ids'] = gids
configs['sites'] = sites
configs['n_points'] = len(sites)
configs['wind_factor'] = wind_factor_list
configs['kp'] = kp_list
configs['n_points'] = len(sites)

with open(out_config_fn, 'w') as f:
    yaml.dump(configs, f, sort_keys=False)

print(f'Wrote {out_config_fn} with {len(sites)} points '
      f'({sum(len(v) for v in sites_by_id.values())} sites x {len(param_combos)} param combos)')

if __name__ == '__main__':
    import simulation as sim

    args = sim.get_args()
    args.config_fn = out_config_fn

    model = sim.PEBSI(args)
    model.run()

# remove config file since it was copied to output directory
os.remove(out_config_fn)