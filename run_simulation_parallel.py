"""
This script executes parallel runs for multiple sites.

@author: clairevwilson
"""

# Built-in libraries
import os
import time
import copy
import pickle
from multiprocessing import Pool
# External libraries
import pandas as pd
# Internal libraries
import run_simulation as sim
import pebsi.massbalance as mb
import pebsi.input as prms

n_runs_ahead = 0    # Step if you're going to run this script more than once

# Read command line args
args = sim.get_args()

# Edit these
args.startdate = '2018-04-20 00:00'
args.enddate = '2025-08-20 00:00'
args.use_AWS = True
args.dates_from_data = True

with open('project/best_datasets.pkl','rb') as f:
    params = pickle.load(f)

# sites to run in parallel
site_dict = {
    '01.22193':['K17b','K53'], # KAHILTNA     'KPS',
    '01.15645':['GTH','KC31','GTL'], # KENNICOTT
    '01.00570':['AU','B','D'], # GULKANA
    '01.09162':['N','B','EC'], # WOLVERINE
    '01.01104':['B','C','D'], # LEMON CREEK
    '01.01390':['MG1','NWB1','TKG3'], # TAKU
    #  '02.06675':[], # ATHABASCA
    #  '02.05098':[], # PEYTO
    #  '02.17023':[], # SPERRY
    #  '02.18778':[], # SOUTH CASCADE
}
glac_nos = list(site_dict.keys())
 
# Probably do not edit
args.store_data = True             # Ensures output is stored
run_date = str(pd.Timestamp.today()).replace('-','_')[:10]
if 'trace' in prms.machine:
    prms.output_fp = '/trace/group/rounce/cvwilson/Output/ddf/'

# Determine number of runs for each process
n_processes = sum([len(site_dict[gn]) for gn in glac_nos])
args.n_processes = n_processes

def pack_vars():
    # Parse list for inputs to Pool function
    packed_vars = [[] for _ in range(n_processes)]
    run_no = 0
    for glac_no in glac_nos:
        args_glac = copy.deepcopy(args)
        args_glac.glac_no = glac_no
        sites = site_dict[args_glac.glac_no]

        if glac_no == '01.01390':
            args_glac.qm_glac_name = 'lemon_creek'
        elif glac_no == '01.15645':
            args_glac.qm_glac_name = 'gulkana'

        for site in sites:
            # Get current site args
            args_run = copy.deepcopy(args_glac)
            args_run.site = site

                        # Output name
            df_meta = pd.read_csv('data/glacier_metadata.csv',index_col=0,converters={0: str})
            glac = df_meta.loc[args_run.glac_no,'name']
            args_run.out = f'{glac}{site}_{run_date}_base_'

            # AWS fn for albedo timeseries
            args_run.AWS_fn = f'../climate_data/AWS/albedo/{glac}{site}_S2albedo.csv'

            # Set parameters from calibration
            if site not in params[glac]:
                continue
            args_run.ksp_BC = params[glac][site]['ksp_BC']
            args_run.Sr = params[glac][site]['Sr']

            # Set task ID for SNICAR input file
            args_run.task_id = run_no + n_runs_ahead*n_processes

            # Store model inputs
            climate, args_run = sim.initialize_model(args_run)
            # climate.cds['ocwet'] *= 0
            # climate.cds['ocdry'] *= 0
            # climate.cds['bcwet'] *= 0
            # climate.cds['bcdry'] *= 0

            # Store model parameters
            store_attrs = {'ksp_BC':args_run.ksp_BC, 'Sr': args_run.Sr,
                           'kp':args_run.kp}

            packed_vars[run_no].append((args_run,climate,store_attrs))

            # Advance counter
            run_no += 1
    return packed_vars

def run_model_parallel(list_inputs):
    # Loop through the variable sets
    for inputs in list_inputs:
        try:
            # Unpack inputs
            args,climate,store_attrs = inputs

            # Get file name
            args = sim.get_output_name(args, climate)
            if os.path.exists(args.out):
                print(args.out, 'already exists')
                return
            
            # Start timer
            start_time = time.time()

            # Run the model
            massbal = mb.massBalance(args,climate)
            massbal.main()

            # Completed model run: end timer
            time_elapsed = time.time() - start_time

            # Store output
            massbal.output.add_vars()
            massbal.output.add_basic_attrs(args,time_elapsed,climate)
            massbal.output.add_attrs(store_attrs)
        except Exception as e:
            print(f'Simulation failed for {inputs[0].out}: {e}')
            continue
    return

# Run model in parallel
if __name__ == '__main__':
    packed_vars = pack_vars()
    with Pool(n_processes) as pool:
        pool.map(run_model_parallel,packed_vars)