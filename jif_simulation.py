# import main functionality
import run_simulation as sim
import yaml
import os
from concurrent.futures import ProcessPoolExecutor

base_fp = '/trace/group/rounce/cvwilson/'

# rgi_ids and their associated sites
site_dict = {
    '01.01390': ['Taku-1'], # TAKU
    # '01.00704': ['Gilkey-2', 'Gilkey-3'], # GILKEY
    '01.00709': ['Mend-2', 'Mend-3'], # MENDENHALL
}

def run_single_simulation(params):
    i, rgi_id, site, dust_factor, ksp_BC, base_fp = params

    # get file names
    config_fn = base_fp + f'configs/config_{site}_{dust_factor}_{ksp_BC}.yaml'
    climate_fp = base_fp + 'climate_data/'
    out_fp = base_fp + 'Output/bc_dust/'
    out_fn = f'grid_{site}_{dust_factor}_{ksp_BC}_'

    # load in the input data
    initial_lap_fn = f'../data/Nagorski/{site}_initial_laps.csv'
    
    # create dict
    config_dict = {
        # Simulation info
        'rgi_id':rgi_id,
        'site':site,
        'start_date':'2016-05-11 14:00',
        'end_date':'2016-07-19',
        'task_id':i,

        # Filepaths
        'climate_fp':climate_fp,
        'initial_LAP_fn':initial_lap_fn,

        # Output
        'store_data':True,
        'output_fn':out_fn,
        'output_fp':out_fp,

        # Parameters
        'ksp_BC':ksp_BC,
        'dust_factor':dust_factor,
    }

    # dump config to yaml
    with open(config_fn, 'w') as f:
        yaml.dump(config_dict, f, sort_keys=False)

    try:
        # read command-line args and specify this config file
        args = sim.get_args()
        args.config_fn = config_fn
        args.use_config = True

        # initialize the model
        climate, args = sim.initialize_model(args)
        print(args.config_fn)
            
        # run the model
        sim.run_model(climate,args)
    except Exception as e:
        print(e)
    finally:
        if os.path.exists(config_fn):
            os.remove(config_fn)

if __name__ == '__main__':
    tasks = []
    i=0
    # loop glaciers and sites
    for rgi_id in site_dict:
        for site in site_dict[rgi_id]:
            for dust_factor in [1, 10, 20, 50, 100]:
                for ksp_BC in [0.1, 0.5, 1]:
                    # tasks.append()
                    task = (i, rgi_id, site, dust_factor, ksp_BC, base_fp)
                    run_single_simulation(task)
                    i += 1

    # with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
    #     executor.map(run_single_simulation, tasks)
                    
