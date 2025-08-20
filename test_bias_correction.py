"""
This script runs the model for multiple sites using 
1) weather station data
2) original MERRA-2 data
3) bias-corrected MERRA-2 data
These simulates are done in series and thus this script
is inefficient but is only used for a figure in ***Paper 1***

@author: clairevwilson
"""

# Built-in libraries
import os
import time
# External libraries
import pandas as pd
import pickle
from multiprocessing import Pool
# Internal libraries
import pebsi.input as eb_prms
import run_simulation as sim
import pebsi.massbalance as mb
from objectives import *

# Read command line args
args = sim.get_args()
args.store_data = True
eb_prms.AWS_fn = '../climate_data/AWS/Preprocessed/gulkana2024.csv'
eb_prms.store_vars = ['MB','EB']
args.startdate = pd.to_datetime('2024-04-18 00:00:00')
args.enddate = pd.to_datetime('2024-08-20 00:00:00')
args.kp = 2.25
args.Boone_c5 = 0.016

# Create output directory
eb_prms.output_filepath = '/trace/group/rounce/cvwilson/Output/bias_correct/'

results = {'AU':{},'AB':{},'B':{},'D':{},'T':{}}
# 

# Loop through sites and model options
for site in results:
    args.site = site
    for climate_option in ['og','bc','aws']:
        if climate_option == 'aws':
            args.use_AWS = True
        else:
            args.use_AWS = False
        
        if climate_option == 'og':
            eb_prms.bias_vars = []
        else:
            eb_prms.bias_vars = ['wind','SWin','temp','rh']

        # Get climate
        climate, args = sim.initialize_model(args.glac_no,args)

        name = f'{site}_{climate_option}_'
        args.out = name

        # Start timer
        start = time.time()

        # Initialize the mass balance / output
        massbal = mb.massBalance(args,climate)

        # Run the model
        massbal.main()

        # Store the mass balance
        ds = massbal.output.get_output()
        internal_acc = ds.isel(time=-2).cumrefreeze.values
        summer_mb = ds.accum.sum().values + ds.refreeze.sum().values - ds.melt.sum().values 
        results[site][climate_option] = summer_mb - internal_acc
        print(climate_option, site, 'wind',np.mean(climate.cds.wind.values), 'mb', summer_mb)
        # os.remove(eb_prms.output_filepath + args.out + '0.nc')

        # Print time
        timer = time.time() - start
        print(f'Time elapsed: {timer:.0f} seconds')

with open(eb_prms.output_filepath+'bias_correction_test.pkl', 'wb') as f:
    pickle.dump(results, f)
    print('Done! Saved to ',eb_prms.output_filepath+'bias_correction_test.pkl')