# Built-in libraries
import time
import copy
from multiprocessing import Pool
import os
# External libraries
import pandas as pd
import xarray as xr
# Internal libraries
import run_simulation as sim
import pebsi.input as prms
import pebsi.massbalance as massBalance
from pebsi.climate import Climate

# Loop info
sites = ['T','Z','EC','KPS','KQU'] # Sites to run in parallel 
vars_adjust = ['temperature','precipitation']
vars_dict = {
             'temperature':[0, 0.5, 1, 2],
             'precipitation':[1, 1.05, 1.1, 1.2],
             }

# Run date for filenames
run_date = str(pd.Timestamp.today()).replace('-','_')[:10]

# ===================================================
# This script has two functions: running the model and
# processing the output runs from the model. 
# Toggle process_runs = FALSE first to run simulations, 
# then rerun with process_runs = TRUE
process_runs = True
# ===================================================

# Load .csv with params for each site
df_sites = pd.read_csv('gridsearch_params.csv', index_col=0)
params_by = 'site'

# Read command line args
args = sim.get_args()
args.startdate = '1980-04-15 00:00'
# args.enddate = '1980-04-16 00:00'
args.enddate = '2025-06-01 00:00'
args.store_data = True              # Ensures output is stored
args.use_AWS = False
if 'trace' in prms.machine:
    prms.output_fp = '/trace/group/rounce/cvwilson/Output/'
for site in sites:
    glacier = 'wolverine' if site == 'EC' else 'gulkana' if site in ['Z','T'] else 'kahiltna'
    if not os.path.exists(prms.output_fp + glacier + site + '_sensitivity'):
        os.mkdir(prms.output_fp + glacier + site + '_sensitivity')

# Determine number of runs for each process
n_runs = len(sites)
n_runs *= len(vars_dict[vars_adjust[0]]) + len(vars_dict[vars_adjust[1]])
n_processes = args.n_simultaneous_processes

# Check if we need multiple (serial) runs per (parallel) set
if n_runs <= n_processes:
    n_runs_per_process = 1
    n_process_with_extra = 0
else:
    n_runs_per_process = n_runs // n_processes  # Base number of runs per CPU
    n_process_with_extra = n_runs % n_processes    # Number of CPUs with one extra run
print(n_runs, 'on', n_processes, 'with', n_runs_per_process, 'per process and',n_process_with_extra,'with one extra')

def pack_vars(args):
    # Parse list for inputs to Pool function
    packed_vars = [[] for _ in range(n_processes)]
    run_no = 0
    set_no = 0
    for site in sites:
        # Get current site args
        args_site = copy.deepcopy(args)
        args_site.site = site

        # Load site specific params
        glacier = 'wolverine' if site == 'EC' else 'kahiltna' if site in ['KPS','KQU'] else 'gulkana'
        args_site.kp = df_sites.loc[site, 'kp']
        args_site.lapse_rate = df_sites.loc[site, 'lr']

        # Parse different glaciers
        if site == 'EC':
            # Wolverine
            prms.bias_vars = ['wind','temp','SWin','rh']
            args_site.glac_no = '01.09162'
            glacier = 'wolverine'
        elif site in ['KPS', 'KQU']:
            # Kahiltna
            prms.bias_vars = ['wind','temp','rh']
            args_site.glac_no = '01.22193'
            glacier = 'kahiltna'
        elif site in ['Z','T']:
            # Gulkana
            prms.bias_vars = ['wind','temp','SWin','rh']
            args_site.glac_no = '01.00570'
            glacier = 'gulkana'

        ############################################################
        # INITIALIZE MODEL CLIMATE AND ARGS
        # NOTE: this code comes from initialize_model, but the
        # order is changed such that temperature is perturbed
        # BEFORE updating longwave radiation
        ############################################################
        
        if not process_runs:
            # check inputs are there
            args_site = sim.check_inputs(args_site)

            # initialize the climate class
            climate_site = Climate(args_site)

            # check if already loaded cds
            if not climate_site.loaded_climate:
                # load in reanalysis data
                climate_site.get_reanalysis(climate_site.all_vars)

            # check all the climate variables are there and do bias correction
            climate_site.check_ds()

        ############################################################
        # END initialize_model: got ds in correct units
        # Still need to update the elevation-dependent cds variables
        ############################################################

        # Loop through perturbations
        for var in vars_adjust:
            for value in vars_dict[var]:
                # Get label for this adjustmnet factor
                if var == 'temperature':
                    var_str = 'temp+'+str(value) if value >= 0 else 'temp'+str(value)
                elif var == 'precipitation':
                    var_str = 'tpx'+str(value)

                if not process_runs:
                    # Create copies for this particular run
                    climate_run = copy.deepcopy(climate_site)
                    args_run = copy.deepcopy(args_site)

                    # Get output name
                    run_date = '2025_10_21'
                    args_run.out = f'{glacier}{site}_sensitivity/{glacier}{site}_{run_date}_{var_str}_'

                    # Make climate adjustments
                    if var == 'temperature':
                        climate_run.cds.temp.values += value
                    elif var == 'precipitation':
                        climate_run.cds.tp.values *= value

                    # Update elevation-dependent terms
                    climate_run.adjust_to_elevation(LWin=False) # Don't update LWin
                    
                    # Update LWin with MERRA-2 temperature set to the original (unadjusted)
                    LAPSE_RATE = float(args_run.lapse_rate) / 1000 # in K m-1
                    LW_elev = climate_run.AWS_elev if 'LWin' in climate_run.measured_vars else climate_run.reanalysis_elev
                    climate_run.LWin_to_elevation(temp_LW_elev = climate_run.cds.temp.values - value + LAPSE_RATE*(LW_elev - climate_run.elev))
                    print(site, var, 'change by', value, 'mean LWin', climate_run.cds.LWin.mean().values / 3600)

                    # Store model parameters
                    store_attrs = {'kp':str(args_run.kp), 'lapse_rate':str(args_run.lapse_rate),
                                    var:str(value)}

                    # Set task ID for SNICAR input file
                    args_run.task_id = set_no
                    args_run.run_id = run_no

                    # Store model inputs
                    packed_vars[set_no].append((args_run,climate_run,store_attrs))

                    # Print for sanity check
                    mean_temp = climate_run.cds.temp.mean().values
                    sum_tp = climate_run.cds.tp.sum().values
                    print(f'Beginning {args_run.out} with {mean_temp} temp, {sum_tp} tp')

                else:
                    out_fn = f'{glacier}{site}_sensitivity/{glacier}{site}_2025_10_06_{var_str}_0.nc'
                    packed_vars[set_no].append((out_fn, var_str, glacier, site))

                # Check if moving to the next set of runs
                n_runs_set = n_runs_per_process + (1 if set_no < n_process_with_extra else 0)
                if run_no == n_runs_set - 1:
                    set_no += 1
                    run_no = -1

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

        # get a unique filename to store the output
        args = sim.get_output_name(args, climate)
        if os.path.exists(prms.output_fp+args.out):
            os.remove(prms.output_fp+args.out)

        # Run the model
        massbal = massBalance.massBalance(args,climate)
        massbal.main()

        # Completed model run: end timer
        time_elapsed = time.time() - start_time

        # Store output
        massbal.output.add_vars()
        massbal.output.add_basic_attrs(args,time_elapsed,climate)
        massbal.output.add_attrs(store_attrs)
    return

def process_runs_parallel(list_inputs):
    for input in list_inputs:
        out_fn, var_str, glacier, site = input
        ds = xr.open_dataset(prms.output_fp + out_fn)
        timeres='1d'
        forcing_fn = f'/trace/group/rounce/cvwilson/Firn/Forcings/{glacier.lower()}{site}/{glacier.lower()}{site}_{timeres}_{var_str}_forcings.csv'

        # get sublimation from any negative vaporsolid mass fluxes in m w.e.
        ds['vaporsolid'][ds['vaporsolid'] > 0] = 0
        ds['sublim'] = ds['vaporsolid']

        # change units of surftemp
        ds['surftemp'] += 273.15

        # resample to the specified resolution with sum (mass balance terms) and mean (surface temp)
        ds_mb = ds[['melt','accum','rainfall','sublim']].resample(time=timeres).sum()
        ds_mb *= 1000   # convert m w.e. to kg m-2
        ds_other = ds[['surftemp']].resample(time=timeres).mean()

        # merge datasets and rename
        data_in = xr.merge([ds_mb, ds_other])
        data_in = data_in.rename_vars({'melt':'SMELT', 'rainfall':'RAIN', 'surftemp':'TS', 'accum':'BDOT','sublim':'SUBLIM'}) # ,'surfdens':'RHOS'

        # store data as a .csv       
        df = data_in[['BDOT','RAIN','TS','SMELT','SUBLIM']].to_dataframe()
        print(forcing_fn, 'stored')
        df.to_csv(forcing_fn)

# Run model in parallel
if __name__ == '__main__':
    packed_vars = pack_vars(args)
    with Pool(n_processes) as processes_pool:
        if not process_runs:
            processes_pool.map(run_model_parallel,packed_vars)
        else:
            processes_pool.map(process_runs_parallel, packed_vars)