"""
Grid search over wind_factor and precipitation factor (kp) for
every site in project/sites.pkl.

Builds a single config file where each site is repeated once per
(wind_factor, kp) combination, so all site x parameter points are
run together as one (large) set of model points. Every other option
is copied from config_redos.yaml.

@author: clairevwilson
"""
import itertools
import pickle

import yaml

sites_fn = 'project/sites.pkl'
base_config_fn = 'config_redos.yaml'
out_config_fn = 'config_gridsearch.yaml'

wind_factors = [1, 1.5, 2, 2.5, 3]
kps = [0.5, 1, 1.5, 2, 2.5, 3, 3.5]

# load the base config to carry over every other option
with open(base_config_fn, 'r') as f:
    configs = yaml.safe_load(f)

# drop leftover per-site lists tied to the old (smaller) site set -
# they won't align with the new grid search points
configs.pop('albedo_ice', None)

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

configs['rgi_ids'] = gids
configs['sites'] = sites
configs['n_points'] = len(sites)
configs['wind_factor'] = wind_factor_list
configs['kp'] = kp_list

configs['temporal_chunks'] = 43800
configs['progress_bar'] = False

configs['climate_fp'] = '/ocean/projects/ees260009p/cwilson4/climate_data/'
configs['rgi_fp'] = '/ocean/projects/ees260009p/cwilson4/RGI/rgi60/00_rgi60_attribs/'
configs['output_fp'] = '/ocean/projects/ees260009p/cwilson4/Output/gridsearch/'
configs['cop30_vrt_path'] = '/ocean/projects/ees260009p/cwilson4/data/dems/COP30/COP30_reg01.vrt'
configs['shading_fp'] = '/ocean/projects/ees260009p/cwilson4/data/shading/'
configs['ice_albedo_fn'] = '/ocean/projects/ees260009p/cwilson4/data/ice_albedo/{gid}_albedo.tif'

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
