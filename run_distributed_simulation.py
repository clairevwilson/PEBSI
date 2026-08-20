"""
Runs a simulation using the parameters stored in 
project/calibrated_grid.pkl which stores all of
the model outputs.

@author: clairevwilson
"""
import pickle
import yaml
import os

configs = {}

run = 'precipwind2x'
configs['wind_factor'] = 2
configs['kp'] = 2
configs['precgrad'] = 0.0001

# POINTS
configs['rgi_ids'] = ['01.00570']
configs['n_points'] = 300

# PHYSICS
configs['option_accel_grains'] = True 
configs['option_flat_plates'] = True
configs['option_ice_albedo_tif'] = True
configs['option_windmaps'] = True
configs['constant_freshgrainsize'] = 54.5

# CONFIGURATION
configs['start_date'] = '2000-04-01'
configs['end_date'] = '2015-04-01'
configs['temporal_chunk_years'] = 5
configs['debug'] = True
configs['store_data'] = True
configs['progress_bar'] = False
configs['store_vars'] = ['MB','EB','climate']
configs['bias_vars'] = ['temp']

# FILEPATHS
configs['climate_fp'] = '/ocean/projects/ees260009p/cwilson4/climate_data/'
configs['rgi_fp'] = '/ocean/projects/ees260009p/cwilson4/RGI/rgi60/00_rgi60_attribs/'
configs['cop30_vrt_path'] = '/ocean/projects/ees260009p/cwilson4/data/dems/COP30/COP30_reg01.vrt'
configs['shading_fp'] = '/ocean/projects/ees260009p/cwilson4/data/shading/'
configs['ice_albedo_fn'] = '/ocean/projects/ees260009p/cwilson4/data/ice_albedo/{gid}_albedo.tif'
configs['windmap_fn'] = '/ocean/projects/ees260009p/cwilson4/data/windmapper/{gid}.nc'

configs['deposition_data'] = 'UKESM'

# SPECIAL SIMS
sim_type = os.environ.get('SIM_TYPE', 'base')

if sim_type == 'base':
    configs['ukesm_fp'] = '../UKESM/dr401_GFED/'

elif sim_type == 'no_fires':
    configs['ukesm_fp'] = '../UKESM/dw068_nofires/'

elif sim_type == 'no_heatwaves':
    configs['ukesm_fp'] = '../UKESM/dr401_GFED/'
    configs['secondary_climate_string'] = '_no_heatwaves'
    configs['secondary_vars'] = ['temp','SWin','LWin']
    
elif sim_type == 'no_either':
    configs['ukesm_fp'] = '../UKESM/dw068_nofires/'
    configs['secondary_climate_string'] = '_no_heatwaves'
    configs['secondary_vars'] = ['temp','SWin','LWin']

configs['output_fp'] = f'/ocean/projects/ees260009p/cwilson4/Output/{run}/{sim_type}/'
out_config_fn = f'{sim_type}.yaml'

with open(out_config_fn, 'w') as f:
    yaml.dump(configs, f, sort_keys=False)

if __name__ == '__main__':
    import simulation as sim

    args = sim.get_args()
    args.config_fn = out_config_fn

    model = sim.PEBSI(args)
    model.run()

# remove config file since it was copied to output directory
os.remove(out_config_fn)