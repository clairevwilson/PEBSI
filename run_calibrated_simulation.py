"""
Runs a simulation using the parameters stored in 
project/calibrated_grid.pkl which stores all of
the model outputs.

@author: clairevwilson
"""
import pickle
import yaml

params_fn = 'project/calibrated_grid.pkl'
out_config_fn = 'config_calibrated.yaml'

configs = {}

# load the sites: dict of {rgi_id: [site, site, ...]}
with open(params_fn, 'rb') as f:
    site_params = pickle.load(f)

gids = []
sites = []
wind_factor_list = []
kp_list = []

for gid, g_sites in site_params.items():
    for site, cal_dict in g_sites.items():
        gids.append(gid)
        sites.append(site)
        wind_factor_list.append(float(cal_dict['kw']))
        kp_list.append(float(cal_dict['kp']))

# POINTS
configs['rgi_ids'] = gids
configs['sites'] = sites
configs['n_points'] = len(sites)
configs['wind_factor'] = wind_factor_list
configs['kp'] = kp_list
configs['n_points'] = len(sites)

# PHYSICS
configs['option_accel_grains'] = True 
configs['option_flat_plates'] = True
configs['option_ice_albedo_tif'] = True
configs['constant_freshgrainsize'] = 54.5

# CONFIGURATION
configs['start_date'] = '2000-04-01'
configs['end_date'] = '2025-09-01'
configs['temporal_chunks'] = 43800
configs['debug'] = True
configs['store_data'] = True
configs['progress_bar'] = False
configs['store_vars'] = ['MB','EB','layers']

# FILEPATHS
configs['climate_fp'] = '/ocean/projects/ees260009p/cwilson4/climate_data/'
configs['rgi_fp'] = '/ocean/projects/ees260009p/cwilson4/RGI/rgi60/00_rgi60_attribs/'
configs['output_fp'] = '/ocean/projects/ees260009p/cwilson4/Output/calibrated_all/'
configs['cop30_vrt_path'] = '/ocean/projects/ees260009p/cwilson4/data/dems/COP30/COP30_reg01.vrt'
configs['shading_fp'] = '/ocean/projects/ees260009p/cwilson4/data/shading/'
configs['ice_albedo_fn'] = '/ocean/projects/ees260009p/cwilson4/data/ice_albedo/{gid}_albedo.tif'

with open(out_config_fn, 'w') as f:
    yaml.dump(configs, f, sort_keys=False)

if __name__ == '__main__':
    import simulation as sim

    args = sim.get_args()
    args.config_fn = out_config_fn

    model = sim.PEBSI(args)
    model.run()