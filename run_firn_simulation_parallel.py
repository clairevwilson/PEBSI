# Built-in libraries
import time
import copy
from multiprocessing import Pool
# External libraries
import pandas as pd
# Internal libraries
import run_simulation as sim
import pebsi.massbalance as mb
import pebsi.input as prms

# User info
sites = ['T','Z','EC','KPS'] # Sites to run in parallel 
run_date = str(pd.Timestamp.today()).replace('-','_')[:10]
n_runs_ahead = 1    # Step if you're going to run this script more than once

# Read command line args
args = sim.get_args()
args.startdate = '1980-04-15 00:00'
args.enddate = '2025-05-20 12:00'
args.store_data = True              # Ensures output is stored
args.use_AWS = False
if 'trace' in prms.machine:
    prms.output_filepath = '/trace/group/rounce/cvwilson/Output/'

# !!! CHANGE THESE
prms.bias_vars = []
test_run = False

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

        # Parse different glaciers
        if site == 'EC':
            # Wolverine
            # prms.bias_vars = ['wind','temp','SWin','rh']
            args_run.glac_no = '01.09162'
            args_run.kp = 1.650 # 1.75
            args_run.lapse_rate = -6.5 # -8.5
            glacier = 'Wolverine'
            if test_run:
                args_run.startdate = '2015-08-01'
                args_run.enddate = '2025-05-01'
        elif site == 'KPS':
            # Kahiltna
            # prms.bias_vars = ['wind','temp','rh']
            args_run.glac_no = '01.22193'
            args_run.kp = 2.470 # 2
            args_run.lapse_rate = -6.5 # -4.5
            glacier = 'Kahiltna'
            if test_run:
                args_run.startdate = '2015-08-01'
                args_run.enddate = '2025-05-01'
        else:
            # Gulkana
            # prms.bias_vars = ['wind','temp','SWin','rh']
            args_run.glac_no = '01.00570'
            if site == 'T':
                args_run.kp = 3.665 # 3.5
            elif site == 'Z':
                args_run.kp = 3.774
            args_run.lapse_rate = -6.5 # -5
            glacier = 'Gulkana'
            if test_run and args_run.site == 'Z':
                args_run.startdate = '2021-08-01'
                args_run.enddate = '2025-05-01'
            elif test_run:
                args_run.startdate = '2012-08-01'
                args_run.enddate = '2025-05-01'

        # Output name
        args_run.out = f'{glacier}_{run_date}_long{site}_noqm_'

        # Store model parameters
        store_attrs = {'kp':str(args_run.kp), 'lapse_rate':str(args_run.lapse_rate),
                       'c5':str(args_run.Boone_c5)}

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