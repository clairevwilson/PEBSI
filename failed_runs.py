# Built-in libraries
import os
import time
import copy
import warnings
import traceback
# External libraries
import numpy as np
import pandas as pd
import xarray as xr
from multiprocessing import Pool
# Internal libraries
import pebsi.input as eb_prms
import run_simulation as sim
import pebsi.massbalance as mb

# Suppress UserWarning messages
warnings.filterwarnings('ignore', category=UserWarning)

# OPTIONS
date = '09_15'
idx = '0'

# Read command line args
args = sim.get_args()
n_processes = args.n_simultaneous_processes

# Replace date
if args.site in ['EC','KPS']:
    date = '09_16'

# Create output directory
if 'trace' in eb_prms.machine:
    eb_prms.output_filepath = '/trace/group/rounce/cvwilson/Output/'
out_fp = f'{date}_{args.site}_{idx}/'

# Load in failed dataset
missing_fn = eb_prms.output_filepath + out_fp + 'missing.txt'
failed = np.genfromtxt(missing_fn,delimiter=',',dtype=str)
if len(failed) < 1:
    print('Successfully finished all runs!')
    quit()
if type(failed[0]) == np.str_:
    failed = [failed]

# Determine number of runs for each process
n_runs = len(failed)
print(f'Beginning {n_runs} model runs on {n_processes} CPUs')

# Check if we need multiple (serial) runs per (parallel) set
if n_runs <= n_processes:
    n_runs_per_process = 0
    n_process_with_extra = 0
else:
    n_runs_per_process = n_runs // n_processes  # Base number of runs per CPU
    n_process_with_extra = n_runs % n_processes    # Number of CPUs with one extra run

# Force some args
args.store_data = True  # Ensures output is stored

# Special dates for low sites
if args.site == 'Z':
    args.glac_no = '01.00570'
    args.startdate = '2021-08-01'
    args.enddate = '2025-05-01'
    eb_prms.bias_vars = ['wind','temp','rh','SWin']
if args.site == 'T':
    args.glac_no = '01.00570'
    args.startdate = '2012-08-01'
    args.enddate = '2025-05-01'
    eb_prms.bias_vars = ['wind','temp','rh','SWin']
if args.site == 'EC':
    args.glac_no = '01.09162'
    args.startdate = '2015-08-01'
    args.enddate = '2025-05-01'
    eb_prms.bias_vars = ['wind','temp','rh','SWin']
if args.site == 'KPS':
    args.glac_no = '01.22193'
    args.startdate = '2015-08-01'
    args.enddate = '2025-05-01'
    eb_prms.bias_vars = ['wind','temp','rh']

# Parse list for inputs to Pool function
packed_vars = [[] for _ in range(n_processes)]
run_no = 0  # Counter for runs added to each set
set_no = 0  # Index for the parallel process

# Loop through the failed runs
for param in failed:
    # Unpack the parameters 
    site, lr, kp, out = param

    # Get args for the current run
    args_run = copy.deepcopy(args)

    # Get the climate
    climate_run, args_run = sim.initialize_model(args_run.glac_no,args_run)

    # Set parameters
    args_run.site = site
    args_run.lapse_rate = lr
    args_run.kp = kp

    # Set identifying output filename
    args_run.out = out

    # Specify attributes for output file
    store_attrs = {'lapse_rate':lr,'kp':kp,'site':site}

    # Set task ID for SNICAR input file
    args_run.task_id = set_no
    args_run.run_id = run_no

    # Store model inputs
    packed_vars[set_no].append((args_run,climate_run,store_attrs))

    # Check if moving to the next set of runs
    n_runs_set = n_runs_per_process + (1 if set_no < n_process_with_extra else 0)
    if run_no == n_runs_set - 1:
        set_no += 1
        run_no = -1

    # Advance counter
    run_no += 1

def run_model_parallel(list_inputs):
    global outdict
    # Loop through the variable sets
    for inputs in list_inputs:
        # Unpack inputs
        args,climate,store_attrs = inputs
        print(args.out)

        try:
            # Start timer
            start_time = time.time()

            # Initialize the mass balance / output
            massbal = mb.massBalance(args,climate)

            # Add attributes to output file in case it crashes
            massbal.output.add_attrs(store_attrs)

            # Run the model
            massbal.main()

            # End timer
            time_elapsed = time.time() - start_time

            # Store output
            massbal.output.add_vars()
            massbal.output.add_basic_attrs(args,time_elapsed,climate)
        except Exception as e:
            print('An error occurred at site',args.site,'with lapserate =',args.lapse_rate,'kp =',args.kp,' ... removing',args.out)
            traceback.print_exc()
            os.remove(eb_prms.output_filepath + args.out + '0.nc')

    return

# Run model in parallel
with Pool(n_processes) as processes_pool:
    processes_pool.map(run_model_parallel,packed_vars)

# missing = []
# for run in failed:
#     fn = run[4]
#     if not os.path.exists(eb_prms.output_filepath + fn + '0.pkl'):
#         missing.append(run)

# # Store missing as .txt
# np.savetxt(missing_fn,np.array(missing),fmt='%s',delimiter=',')
# print(f'Finished param_set_parallel, saved missing to {missing_fn}')