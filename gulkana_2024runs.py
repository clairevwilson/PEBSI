"""
This script executes parallel runs for the 2024 melt season
across all sites for a single parameter combination.

@author: clairevwilson
"""

# Built-in libraries
import time
import copy
# External libraries
import pandas as pd
from multiprocessing import Pool
# Internal libraries
import run_simulation as sim
import pebsi.massbalance as mb
import pebsi.input as eb_prms

# User info
sites = ['AB','ABB','B','BD','D','T'] # Sites to run in parallel     
# False or filename of parameters .csv for run, relative to PyGEM-EB/
params_fn = False # '../Output/params/11_08.csv'
run_date = str(pd.Timestamp.today()).replace('-','_')[5:10]
n_runs_ahead = 0    # Step if you're going to run the model more than once at a time

# Read command line args
args = sim.get_args()
args.startdate = '2024-04-17 18:00'
args.enddate = '2024-08-20 00:00'
args.store_data = True              # Ensures output is stored
args.debug = False                  # Don't need debug prints
args.use_AWS = True                 # Use AWS and set filepath
eb_prms.glac_no = ['01.00570']
# Set AWS filename
eb_prms.AWS_fn = eb_prms.AWS_fp + 'Preprocessed/gulkana2024.csv'   
if 'trace' in eb_prms.machine:
    eb_prms.output_filepath = '/trace/group/rounce/cvwilson/Output/'

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

        # Set parameters filename (relative to PyGEM-EB/)
        if params_fn:
            args_run.params_fn = params_fn
            params = pd.read_csv(params_fn,index_col=0)
            kp = params.loc['kp',args_run.site].astype(float)
            kw = params.loc['kw',args_run.site].astype(float)
            a_ice = params.loc['a_ice',args_run.site].astype(float)
            c5 = params.loc['Boone_c5',args_run.site].astype(float)
            # Command line args override params input
            if args_run.kp == eb_prms.kp:
                args_run.kp = kp
            if args_run.kw == eb_prms.wind_factor:
                args_run.kw = kw
            if args_run.a_ice == eb_prms.albedo_ice:
                args_run.a_ice = a_ice
            if args_run.Boone_c5 == eb_prms.Boone_c5:
                args_run.Boone_c5 = c5
            store_attrs = {'params_fn':params_fn,'site':site,
                        'kp':str(args_run.kp),'kw':str(args_run.kw),
                            'AWS':eb_prms.AWS_fn,'c5':str(c5)}
        else:
            store_attrs = {'site':site}

        # Output info
        args_run.out = f'Gulkana_{run_date}_2024{site}_'

        # Set task ID for SNICAR input file
        args_run.task_id = run_no + n_runs_ahead*n_processes

        # Store model inputs
        climate, args_run = sim.initialize_model(args_run.glac_no,args_run)
        packed_vars[run_no].append((args_run,climate,store_attrs))

        # Advance counter
        run_no += 1
    return packed_vars

def run_model_parallel(list_inputs):
    # Loop through the variable sets
    for inputs in list_inputs:
        # Unpack inputs
        args,climate,store_attrs = inputs
        
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