# import main model functions
import run_simulation as sim
# internal imports
import yaml, os
import itertools
from concurrent.futures import ProcessPoolExecutor
# data handling
from project.data_handling import MassBalance

base_fp = '/trace/group/rounce/cvwilson/'

# glacier names and their associated sites
site_dict = {
    'kahiltna':['K17b','K53',], # KAHILTNA
    'kennicott':['GTH','KC31','GTL'], # KENNICOTT   
    'gulkana':['AU','B','D'], # GULKANA
    'wolverine':['N','B','EC'], # WOLVERINE   
    'lemon_creek':['C','B','D'], # LEMON CREEK
    'taku':['NWB1','TKG3'], # TAKU       
}

# parameters to calibrate
params = {'wind_factor':[1, 1.2, 1.5, 2, 2.5, 3]}
keys = list(params.keys())
values = list(params.values())

def initialize_simulation(input):
    global base_fp
    i, glacier, site, param_keys, param_values = input

    # get file names
    config_fn = base_fp + f'configs/config_{i}.yaml'
    climate_fp = base_fp + 'climate_data/'
    out_fp = base_fp + 'Output/ddf/calibrate_wind/'
    params_str = '_'.join([p.replace('_','') + str(v) for p, v in zip(param_keys, param_values)])
    out_fn = f'grid_{glacier}_{site}_{params_str}_'

    # check what years the mass balance data covers
    mb = MassBalance(glacier, site)
    start_year = max(2000, mb.start_year) - 1 # start one year before for spinup
    end_year = mb.end_year
    
    # create dict
    config_dict = {
        # Simulation info
        'store_data':True,
        'task_id':i,
        'start_date':f'{start_year}-04-01',
        'end_date':f'{end_year}-09-01',

        # Glacier info
        'glac_name':glacier,
        'site':site,

        # Filepaths
        'climate_fp':climate_fp,
        'output_fn':out_fn,
        'output_fp':out_fp,
    }

    # add parameters to config
    for param_key, param_value in zip(param_keys, param_values):
        config_dict[param_key] = param_value

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
            for param_values in itertools.product(*values):
                # initialize the model in series
                initial_input = (i, glacier, site, keys, param_values)
                sim_inputs = initialize_simulation(initial_input)
                i += 1

                # append the initialized climate and args
                tasks.append(sim_inputs)

    # execute the model in parallel
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        executor.map(run_single_simulation, tasks)