# main model functions
import run_simulation as sim
import pandas as pd
import array as xr
# internal imports
import yaml, os
import itertools
import pickle
from concurrent.futures import ProcessPoolExecutor

base_fp = '/trace/group/rounce/cvwilson/'

# glacier names and their associated sites
site_dict = {
    'wolverine':['EC'],
    'kahiltna':['KPS','KQU',],
    'gulkana':['T','Z'],
    # 'gulkana':['T']
}

# parameters to calibrate
vars_dict = {
             'temperature':[0, 0.5, 1, 2],
             'precipitation':[1, 1.05, 1.1, 1.2],
             }
keys = list(vars_dict.keys())
values = list(vars_dict.values())

# open calibrated parameters dict
with open('project/best_firn_params.pkl', 'rb') as f:
    params_dict = pickle.load(f)

# params_table = pd.read_csv('firn_params.csv', index_col=0)

def initialize_simulation(input):
    global base_fp
    i, glacier, site, perturb_var, perturb_val = input

    # get file names
    config_fn = base_fp + f'configs/config_{i}.yaml'
    climate_fp = base_fp + 'climate_data/'
    rgi_fp = base_fp + '../shared/RGI/rgi60/00_rgi60_attribs/'
    out_fp = base_fp + f'Output/paper2/{glacier}{site}_sensitivity/'
    if perturb_var == 'temperature':
        param = 'temp_perturb'
        param_str = 'temp+' + str(perturb_val)
    elif perturb_var == 'precipitation':
        param = 'tp_perturb'
        param_str = 'tpx' + str(perturb_val)
    out_fn = f'{glacier}{site}_{param_str}_fixedgrains_'

    # define bias vars
    if glacier != 'kahiltna':
        bias_vars = ['temp','rh','SWin','wind']
    else:
        bias_vars = ['temp','SWin','wind']
    
    # create dict
    config_dict = {
        # Simulation info
        'store_data':True,
        'task_id':i,
        'start_date':'1980-04-01',
        'end_date':'2025-09-01',
        'bias_vars':bias_vars,
        'option_accel_grains':True,

        # Glacier info
        'glac_name':glacier,
        'site':site,

        # Filepaths
        'climate_fp':climate_fp,
        'output_fn':out_fn,
        'output_fp':out_fp,
        'rgi_fp':rgi_fp,

        # Parameters
        param: perturb_val,
        'constant_freshgrainsize': 54.5,
        # 'kp': float(params_table.loc[site, 'kp']),
        # 'lapse_rate': float(params_table.loc[site, 'lr'])
    }

    # add calibrated parameters to config
    for key in params_dict[glacier][site]:
        config_dict[key] = params_dict[glacier][site][key]

    # dump config to yaml
    with open(config_fn, 'w') as f:
        yaml.dump(config_dict, f, sort_keys=False)

    # read command-line args and specify this config file
    args = sim.get_args()
    args.config_fn = config_fn
    args.use_config = True

    # initialize the model
    climate, args = sim.initialize_model(args)
    return climate, args

def run_single_simulation(input):
    # unpack climate and args
    climate, args = input 

    success = False
    try:
        # run the model
        ds = sim.run_model(climate, args)
        success = True
    except Exception as e:
        # print failure message if an error occurred
        print(args.glac_name, args.site, f'failed with {e}')
    finally:
        # remove temp file even if failed 
        if os.path.exists(args.config_fn):
            os.remove(args.config_fn)

    print('success?', success)
    if success:
        print('so were here then, wtf?')
        # process the dataset for CFM input
        timeres='1d'
        forcing_fp = '/trace/group/rounce/cvwilson/Firn/Forcings/'
        forcing_fn = glacier.lower() + site + '/' + args.output_fn.replace('.nc','.csv')

        # get sublimation from any negative vaporsolid mass fluxes in m w.e.
        ds['vaporsolid'][ds['vaporsolid'] > 0] = 0
        ds['sublim'] = ds['vaporsolid']

        # change units of surftemp from C to K
        ds['surftemp'] += 273.15

        # resample to the specified resolution with sum (mass balance terms) and mean (surface temp)
        ds_mb = ds[['melt','accum','rainfall','sublim']].resample(time=timeres).sum()
        ds_mb *= 1000   # convert m w.e. to kg m-2
        ds_notmb = ds[['surftemp']].resample(time=timeres).mean()

        # merge datasets and rename
        data_in = xr.merge([ds_mb, ds_notmb])
        data_in = data_in.rename_vars({'melt':'SMELT', 'rainfall':'RAIN', 
                                        'surftemp':'TS', 'accum':'BDOT',
                                        'sublim':'SUBLIM'}) # , 'surfdens':'RHOS'

        # store data as a .csv       
        df = data_in[['BDOT','RAIN','TS','SMELT','SUBLIM']].to_dataframe()
        print('STORING TO', forcing_fp + forcing_fn)
        df.to_csv(forcing_fp + forcing_fn)
    return

if __name__ == '__main__':
    # initialize storage for tasks
    tasks = []
    i = 0

    # loop glaciers and sites
    for glacier in site_dict:
        for site in site_dict[glacier]:
            # loop parameter combinations
            for key in keys:
                for val in vars_dict[key]:
                    # initialize the model in series
                    initial_input = (i, glacier, site, key, val)
                    sim_inputs = initialize_simulation(initial_input)
                    i += 1

                    # append the initialized climate and args
                    tasks.append(sim_inputs)

    # execute the model in parallel
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        executor.map(run_single_simulation, tasks)