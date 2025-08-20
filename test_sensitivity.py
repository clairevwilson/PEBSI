"""
This script runs the model for different values of 
parameters for a sensitivity test. These simulations 
are done in series and thus this script is inefficient
but is only used for a figure in ***Paper 1***

@author: clairevwilson
"""

# Built-in libraries
import os
import time
import copy
import traceback
from multiprocessing import Pool
import pickle
# External libraries
import pandas as pd
import xarray as xr
# Internal libraries
import run_simulation as sim
import pebsi.input as eb_prms
import pebsi.massbalance as mb
from objectives import *

# Read command line args
args = sim.get_args()
args.store_data = True
args.use_AWS = False
args.site = 'B'
eb_prms.store_vars = ['MB','EB','layers','climate']
args.startdate = pd.to_datetime('2023-04-18 00:00:00')
args.enddate = pd.to_datetime('2024-04-20 00:00:00')

# Create output directory
eb_prms.output_filepath = '/trace/group/rounce/cvwilson/Output/sensitivity/'

# Get climate
climate, args = sim.initialize_model(args.glac_no,args)

# Place for results
result_dict = {'base':{},'kp':{},'Boone_c5':{},'lapserate':{},
               'roughness_fresh_snow':{},'roughness_aged_snow':{},
               'roughness_ice':{},'albedo_ground':{},
               'ksp_BC':{},'ksp_OC':{},'ksp_dust':{}}

def model_run(name,climate,args):
    # Name the run
    args.out = name
    print(name.split())

    # Start timer
    start = time.time()

    # Initialize the mass balance / output
    massbal = mb.massBalance(args,climate)

    # Run the model
    massbal.main()

    # Print time
    timer = time.time() - start
    print(f'Time elapsed: {timer:.0f} seconds')

model_run('base_',climate,args)

# SENSITIVITY
args.kp = 1
climate,args = sim.initialize_model(args.glac_no,args)
print('kp=1',np.sum(climate.cds.tp.values))
model_run('kp_-20_',climate,args)
args.kp = 3
climate,args = sim.initialize_model(args.glac_no,args)
print('kp=3',np.sum(climate.cds.tp.values))
model_run('kp_+20_',climate,args)
args.kp = 2
climate,args = sim.initialize_model(args.glac_no,args)
print('kp normal',np.sum(climate.cds.tp.values))

args.Boone_c5 = 0.014
model_run('Boone-c5_-20_',climate,args)
args.Boone_c5 = 0.024
model_run('Boone-c5_+20_',climate,args)
args.Boone_c5 = 0.018

eb_prms.lapserate = -0.0055
climate,args = sim.initialize_model(args.glac_no,args)
model_run('lapserate_-20_',climate,args)
eb_prms.lapserate = -0.007
climate,args = sim.initialize_model(args.glac_no,args)
model_run('lapserate_+20_',climate,args)
eb_prms.lapserate = -0.0065
climate,args = sim.initialize_model(args.glac_no,args)

eb_prms.roughness_fresh_snow = 0.1
model_run('roughness-fresh-snow_-20_',climate,args)
eb_prms.roughness_fresh_snow = 1
model_run('roughness-fresh-snow_+20_',climate,args)
eb_prms.roughness_fresh_snow = 0.24

eb_prms.roughness_aged_snow = 5
model_run('roughness-aged-snow_-20_',climate,args)
eb_prms.roughness_aged_snow = 20
model_run('roughness-aged-snow_+20_',climate,args)
eb_prms.roughness_aged_snow = 10

eb_prms.roughness_ice = 10
model_run('roughness-ice_-20_',climate,args)
eb_prms.roughness_ice = 40
model_run('roughness-ice_+20_',climate,args)
eb_prms.roughness_ice = 20

eb_prms.albedo_ground = 0.1
model_run('albedo-ground_-20_',climate,args)
eb_prms.albedo_ground = 0.3
model_run('albedo-ground_+20_',climate,args)
eb_prms.albedo_ground = 0.2

eb_prms.ksp_BC = 0.1
model_run('ksp-BC_-20_',climate,args)
eb_prms.ksp_BC = 1.2
model_run('ksp-BC_+20_',climate,args)
eb_prms.ksp_BC = 0.9

eb_prms.ksp_OC = 0.1
model_run('ksp-OC_-20_',climate,args)
eb_prms.ksp_OC = 1.2
model_run('ksp-OC_+20_',climate,args)
eb_prms.ksp_OC = 0.9

eb_prms.ksp_dust = 0.001
model_run('ksp-dust_-20_',climate,args)
eb_prms.ksp_dust = 0.2
model_run('ksp-dust_+20_',climate,args)

print('Done!')