"""
This script processes grid search outputs in parallel.

@author: clairevwilson
"""
import pebsi.processing.gridsearch_processing as gsproc
from multiprocessing import Pool
import argparse
import time
import os

parser = argparse.ArgumentParser()
parser.add_argument('-n','--n_processes',default=10,type=int)
parser.add_argument('-redo',default=True,type=bool)
args = parser.parse_args()

n_processes = args.n_processes

def process_run(runs):
    for run in runs:
        run_type,fn = run
        gsproc.process_run(run_type, fn)

for run_type in ['long','2024']:
    for site in gsproc.sitedict[run_type]:
        start_site = time.time()
        date = gsproc.run_info[run_type]['date']
        idx = gsproc.run_info[run_type]['idx']
        fp = gsproc.base_fp+f'{date}_{site}_{idx}/'

        if os.path.exists(fp + f'grid_{date}_set0_run0_0.pkl') and not args.redo:
            continue
        
        check_date = gsproc.run_info['long']['date']
        check_idx = gsproc.run_info['long']['idx']
        all_nc = [f for f in os.listdir(fp) if 'nc' in f]
        n_runs = len(all_nc)

        # Parse list for inputs to Pool function
        packed_vars = [[] for _ in range(n_processes)]
        run_no = 0  # Counter for runs added to each set
        set_no = 0  # Index for the parallel process

        if n_runs <= n_processes:
            n_runs_per_process = 1
            n_process_with_extra = 0
        else:
            n_runs_per_process = n_runs // n_processes     # Base number of runs per CPU
            n_process_with_extra = n_runs % n_processes    # Number of CPUs with one extra run

        for i in range(n_runs):
            fn = fp + f'grid_{date}_set{i}_run0_0.nc'
            packed_vars[set_no].append((run_type,fn))

            # Check if moving to the next set of runs
            n_runs_set = n_runs_per_process + (1 if set_no < n_process_with_extra else 0)
            if run_no == n_runs_set - 1:
                set_no += 1
                run_no = -1
            
            run_no += 1

        with Pool(n_processes) as processes_pool:
            processes_pool.map(process_run,packed_vars)

        elapsed = time.time() - start_site
        print(f'Finished site {site} in {elapsed:.0f}')

print('Done!')