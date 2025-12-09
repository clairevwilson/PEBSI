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
repeat_run = True   # True if restarting an already begun run
# Define sets of parameters
params = {'c5':[ 0.014,0.016,0.018,0.02,0.022,0.024],               # Gulkana-only grid search for paper 1
          'kp':[1,1.25,1.5,1.75,2,2.25,2.5,2.75,3]}
# params = {# 'Boone_c5':[0.014,0.016,0.018,0.02,0.022,0.024], # 
        #   'kp':[1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.25,3.5],
        #   'lapse_rate':[-3.5,-4,-4.5,-5,-5.5,-6,-6.5,-7,-7.5,-8,-8.5]} # 
param_1, param_2 = list(params.keys())

# Read command line args
parser = sim.get_args(parse=False)
parser.add_argument('-run_type', default='long', type=str)
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

# Force some args
if args.run_type == '2024': # Short AWS run
    args.use_AWS = True
    prms.AWS_fn = '../climate_data/AWS/Processed/gulkana2024.csv'
    prms.store_vars = ['MB','EB','layers','temp']
    args.startdate = pd.to_datetime('2024-04-18 00:00:00')
    args.enddate = pd.to_datetime('2024-08-20 00:00:00')
else: # Long MERRA-2 run
    args.use_AWS = False
    prms.store_vars = ['MB','layers','temp','EB']
    args.startdate = pd.to_datetime('2000-04-15 00:00:00')
    args.enddate = pd.to_datetime('2024-08-20 00:00:00')

if repeat_run:
    date = '12_09' # if args.run_type == 'long' else '08_02'
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
if args.run_type == 'long':
    if args.site == 'A':
        args.enddate = pd.to_datetime('2015-05-20 00:00:00')
    elif args.site == 'AU':
        args.startdate = pd.to_datetime('2012-04-20 00:00:00')

# Loop through parameters
for p1 in params[param_1]:
    for p2 in params[param_2]:
        # Copy over args
        args_run = copy.deepcopy(args)

        # Set parameters MANUALLY
        args_run.lapse_rate = -6.5
        args_run.kp = p2
        args_run.Boone_c5 = p1

        # Set identifying output filename
        args_run.out = out_fp + f'grid_{date}_set{set_no}_run{run_no}_'
        all_runs.append((args_run.site, p1, p2, args_run.out))

        # Get the climate
        climate_run, args_run = sim.initialize_model(args_run)

        # Specify attributes for output file
        store_attrs = {'lapse_rate':-6.5, 'c5':p1,'kp':p2}

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

                # Get unique filename
                args = sim.get_output_name(args, climate)

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
                print('An error occurred at site',args.site,'with c5 =',args.Boone_c5,'kp =',args.kp,' ... removing',args.out)
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