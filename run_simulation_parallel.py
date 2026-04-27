"""
This script executes parallel runs for multiple sites.

@author: clairevwilson
"""

# Built-in libraries
import os
import time
import copy
import pickle
import traceback
from multiprocessing import Pool
# External libraries
import pandas as pd
# Internal libraries
import run_simulation as sim
import pebsi.massbalance as mb
import util.params as prms
prms.use_config = True
prms.config_fn = 'config_parallel.yaml'

# Step if you're going to run this script more than once simultaneously
n_runs_ahead = 0

# Date for output filename
run_date = str(pd.Timestamp.today()).replace('-','_')[:10]

# Read command line args
parser = sim.get_args(parse=False)
parser.add_argument('-n','--n_simultaneous_processes',default=1)
cmd_args = parser.parse_args()
out_str = 'dust50x'

# Sites to run in parallel
site_dict = {
    '01.22193':['K17b','K53',], # KAHILTNA
    '01.15645':['GTH','KC31','GTL'], # KENNICOTT   
    '01.00570':['AU','B','D'], # GULKANA
    '01.09162':['N','B','EC'], # WOLVERINE   
    '01.01104':['C','B','D'], # LEMON CREEK
    '01.01390':['MG1','NWB1','TKG3'], # TAKU       
}
rgi_ids = list(site_dict.keys())

# Determine number of runs for each process
n_processes = sum([len(site_dict[gn]) for gn in rgi_ids])
cmd_args.n_processes = n_processes

def pack_vars():
    # Parse list for inputs to Pool function
    packed_vars = [[] for _ in range(n_processes)]
    run_no = 0
    # Glacier loop
    for rgi_id in rgi_ids:
        # Copy args
        args_glac = copy.deepcopy(cmd_args)

        # Add glacier to args and get sites for this glacier
        args_glac.rgi_id = rgi_id
        sites = site_dict[args_glac.rgi_id]

        # Handle QM glacier for special cases
        if rgi_id == '01.01390': # Taku
            args_glac.qm_glac_name = 'lemon_creek'
        elif rgi_id == '01.15645': # Kennicott
            args_glac.qm_glac_name = 'gulkana'

        # Site loop
        for site in sites:
            # Copy args again and store site
            args_run = copy.deepcopy(args_glac)
            args_run.site = site

            # Output name
            df_meta = pd.read_csv('data/glacier_metadata.csv',index_col=0,converters={0: str})
            glac = df_meta.loc[args_run.rgi_id,'name']
            args_run.output_fn = f'{glac}{site}_{run_date}_{out_str}_'

            # Set task ID for SNICAR input file
            args_run.task_id = run_no + (n_runs_ahead+1)*n_processes

            # Store model inputs
            climate, args_run = sim.initialize_model(args_run)

            # Manipulate climate if desired
            # climate.cds['ocwet'] *= 0
            # climate.cds['ocdry'] *= 0
            # climate.cds['bcwet'] *= 0
            # climate.cds['bcdry'] *= 0
            climate.cds['dustwet'] *= 50
            climate.cds['dustdry'] *= 50

            # Store model parameters
            store_attrs = {'ksp_BC':args_run.ksp_BC, 'ksp_OC':args_run.ksp_OC, 'Sr': args_run.Sr,
                           'kp':args_run.kp, 'wet_C':args_run.wet_grain_C, 'a_ice':args_run.albedo_ice}

            # Pack function ro execute in parallel
            packed_vars[run_no].append((args_run,climate,store_attrs))

            # Advance counter for task ID
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
            if os.path.exists(args.output_fn):
                print(args.output_fn, 'already exists')
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
            print(f'Simulation failed for {inputs[0].output_fn}: {e}')
            traceback.print_exc()
            continue
    return

# Run model in parallel
if __name__ == '__main__':
    packed_vars = pack_vars()
    with Pool(n_processes) as pool:
        pool.map(run_model_parallel,packed_vars)