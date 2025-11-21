"""
This script executes parallel runs for multiple sites.

@author: clairevwilson
"""

# Built-in libraries
import sys
import time
import copy
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
args.startdate = '2004-04-20 00:00'
args.enddate = '2005-08-20 00:00'
args.glac_no = '01.00570'
args.use_AWS = False
sites = ['B'] # Sites to run in parallel

# Probably do not edit
args.store_data = True              # Ensures output is stored
run_date = str(pd.Timestamp.today()).replace('-','_')[:10]
if 'trace' in prms.machine:
    prms.output_fp = '/trace/group/rounce/cvwilson/Output/'

# Determine number of runs for each process
n_processes = len(sites)
args.n_processes = n_processes

def pack_vars():
    # Parse list for inputs to Pool function
    packed_vars = [[] for _ in range(n_processes)]
    run_no = 0
    for site in sites:
        # Get current site args
        args_run = copy.deepcopy(args)
        args_run.site = site

        # Output name
        df_meta = pd.read_csv('data/glacier_metadata.csv', index_col=0, dtype=str, 
                              converters={0: str})
        glac = df_meta.loc[args.glac_no,'name']
        args_run.out = f'{glac}{site}_{run_date}_'

        # Store model parameters
        store_attrs = {'kp':str(args_run.kp), 'c5':str(args_run.Boone_c5),
                       'lr':str(args_run.lapse_rate)}

        # Set task ID for SNICAR input file
        args_run.task_id = run_no + n_runs_ahead*n_processes

        # Store model inputs
        climate, args_run = sim.initialize_model(args_run)
        packed_vars[run_no].append((args_run,climate,store_attrs))

        # Advance counter
        run_no += 1
    return packed_vars

def run_model_parallel(list_inputs):
    # Loop through the variable sets
    for inputs in list_inputs:
        # Unpack inputs
        args,climate,store_attrs = inputs

        # Get file name
        args = sim.get_output_name(args, climate)
        
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
    return

# Run model in parallel
if __name__ == '__main__':
    packed_vars = pack_vars()
    with Pool(n_processes) as processes_pool:
        processes_pool.map(run_model_parallel,packed_vars)