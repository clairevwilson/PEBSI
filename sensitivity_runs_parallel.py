# import main model functions
import run_simulation as sim
import pandas as pd
# internal imports
import yaml, os
import itertools
from concurrent.futures import ProcessPoolExecutor
# data handling
from project.data_handling import MassBalance

base_fp = '/trace/group/rounce/cvwilson/'

# glacier names and their associated sites
site_dict = {
    'wolverine':['EC'],
    # 'kahiltna':['KPS','KQU',],
    # 'gulkana':['T','Z'],
    # 'gulkana':['T']
}

# parameters to calibrate
vars_dict = {
             'temperature':[0, 0.5, 1, 2],
             'precipitation':[1, 1.05, 1.1, 1.2],
             }
keys = list(vars_dict.keys())
values = list(vars_dict.values())

df_params = pd.read_csv('firn_params.csv', index_col=0)

def initialize_simulation(input):
    global base_fp
    i, glacier, site, perturb_var, perturb_val = input

    # get file names
    config_fn = base_fp + f'configs/config_{i}.yaml'
    climate_fp = base_fp + 'climate_data/'
    out_fp = base_fp + f'Output/paper2/{glacier}{site}_sensitivity/'
    if perturb_var == 'temperature':
        param = 'temp_perturb'
        param_str = 'temp+' + str(perturb_val)
    elif perturb_var == 'precipitation':
        param = 'tp_perturb'
        param_str = 'tpx' + str(perturb_val)
    out_fn = f'{glacier}{site}_{param_str}_redo2_'

    # define bias vars
    if glacier != 'kahiltna':
        bias_vars = ['temp','wind','rh','SWin']
    else:
        bias_vars = ['temp','wind','rh']

    # extract calibrated params
    kp = float(df_params.loc[site, 'kp'])
    lapse_rate = float(df_params.loc[site, 'lr'])
    
    # create dict
    config_dict = {
        # Simulation info
        'store_data':True,
        'task_id':i,
        'start_date':'1980-04-01',
        'end_date':'2025-09-01',
        'bias_vars':bias_vars,

        # Glacier info
        'glac_name':glacier,
        'site':site,

        # Filepaths
        'climate_fp':climate_fp,
        'output_fn':out_fn,
        'output_fp':out_fp,

        # Parameters
        'kp': kp,
        'lapse_rate':lapse_rate,
        param: perturb_val,
        'constant_freshgrainsize': 54.5
    }

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

    try:
        # run the model
        sim.run_model(climate, args)
    except Exception as e:
        # print failure message if an error occurred
        print(args.glac_name, args.site, f'failed with {e}')
    finally:
        # remove temp file even if failed 
        if os.path.exists(args.config_fn):
            os.remove(args.config_fn)
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