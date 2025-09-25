"""
This script executes a grid search in parallel over
multiple parameters. These parameters can be specified
in the params dict below. It is set up to perform the
search over two parameters for ***Paper 1*** 
(Boone c5 densification parameter and kp precipitation
factor) but with minor edits more parameters can be added.

@author: clairevwilson
"""

# Built-in libraries
import os
import time
import copy
import traceback
import pickle
from multiprocessing import Pool
# External libraries
import pandas as pd
import xarray as xr
# Internal libraries
import run_simulation as sim
import pebsi.input as prms
import pebsi.massbalance as mb
from objectives import *

# OPTIONS
repeat_run = False   # True if restarting an already begun run
# Define sets of parameters
# params = {'Boone_c5':[0.018,0.02,0.022,0.024,0.026,0.028,0.03], # 
#           'kp':[1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.25,3.5]} # 
params = {# 'Boone_c5':[0.014,0.016,0.018,0.02,0.022,0.024], # 
          'kp':[1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.25,3.5],
          'lapse_rate':[-3.5,-4,-4.5,-5,-5.5,-6,-6.5,-7,-7.5,-8,-8.5]} # 

# Read command line args
parser = sim.get_args(parse=False)
args = parser.parse_args()
n_processes = args.n_simultaneous_processes

# Determine number of runs for each process
n_runs = 1
for param in list(params.keys()):
    n_runs *= len(params[param])
print(f'Beginning {n_runs} model runs on {n_processes} CPUs')

# Check if we need multiple (serial) runs per (parallel) set
if n_runs <= n_processes:
    n_runs_per_process = 1
    n_process_with_extra = 0
else:
    n_runs_per_process = n_runs // n_processes  # Base number of runs per CPU
    n_process_with_extra = n_runs % n_processes    # Number of CPUs with one extra run

# Create output directory
if 'trace' in prms.machine:
    prms.output_fp = '/trace/group/rounce/cvwilson/Output/'
# Store all variables
args.store_data = True
prms.store_vars = ['MB','layers','temp','EB']

if repeat_run:
    date = '08_01' if args.run_type == 'long' else '08_02'
    print('Forcing run date to be', date)
    n_today = '0'
    out_fp = f'{date}_{args.site}_{n_today}/'
    if not os.path.exists(prms.output_fp + out_fp):
        os.mkdir(prms.output_fp + out_fp)
else:
    date = str(pd.Timestamp.today()).replace('-','_')[5:10]
    n_today = 0
    out_fp = f'{date}_{args.site}_{n_today}/'
    while os.path.exists(prms.output_fp + out_fp):
        n_today += 1
        out_fp = f'{date}_{args.site}_{n_today}/'
    os.mkdir(prms.output_fp + out_fp)

# Transform params to strings for comparison
for key in params:
    for v,value in enumerate(params[key]):
        params[key][v] = str(value)

# Parse list for inputs to Pool function
packed_vars = [[] for _ in range(n_processes)]
run_no = 0  # Counter for runs added to each set
set_no = 0  # Index for the parallel process

# Storage for failed runs
all_runs = []
missing_fn = prms.output_fp + out_fp + 'missing.txt'

# Dates depend on the site
if args.site == 'Z':
    args.glac_no = '01.00570'
    args.startdate = '2021-08-01'
    args.enddate = '2025-05-01'
    prms.bias_vars = ['wind','temp','rh','SWin']
if args.site == 'T':
    args.glac_no = '01.00570'
    args.startdate = '2012-08-01'
    args.enddate = '2025-05-01'
    prms.bias_vars = ['wind','temp','rh','SWin']
if args.site == 'EC':
    args.glac_no = '01.09162'
    args.startdate = '2015-08-01'
    args.enddate = '2025-05-01'
    prms.bias_vars = ['wind','temp','rh','SWin']
if args.site == 'KPS':
    args.glac_no = '01.22193'
    args.startdate = '2015-08-01'
    args.enddate = '2025-05-01'
    prms.bias_vars = ['wind','temp','rh']

# Loop through parameters
for kp in params['kp']:
    for lr in params['lapse_rate']:
        # Copy over args
        args_run = copy.deepcopy(args)

        # Set parameters
        args_run.lapse_rate = lr
        args_run.kp = kp

        # Set identifying output filename
        args_run.out = out_fp + f'grid_{date}_set{set_no}_run{run_no}_'
        all_runs.append((args_run.site, lr, kp, args_run.out))

        # Get the climate
        climate_run, args_run = sim.initialize_model(args_run.glac_no,args_run)

        # Specify attributes for output file
        store_attrs = {'lapse_rate':lr,'kp':kp}

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

        # Check if model run should be performed
        if not os.path.exists(prms.output_fp + args.out + '0.nc'):
            try:
                # Start timer
                start_time = time.time()

                # Initialize the mass balance / output
                massbal = mb.massBalance(args,climate)

                # Add attributes to output file in case it crashes
                if args.store_data:
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
                os.remove(prms.output_fp + args.out + '0.nc')
    return

# Run model in parallel
with Pool(n_processes) as processes_pool:
    processes_pool.map(run_model_parallel,packed_vars)
    
missing = []
for run in all_runs:
    fn = run[-1]
    if not os.path.exists(prms.output_fp + fn + '0.nc'):
        missing.append(run)
n_missing = len(missing)

# Store missing as .txt
np.savetxt(missing_fn,np.array(missing),fmt='%s',delimiter=',')
print(f'Finished grid search at site {args.site} with {n_missing} failed: saved to {missing_fn}')