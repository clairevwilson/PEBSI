"""
Main script to execute PEBSI

Parses the command-line arguments, checks the 
inputs of the model, runs the shading model if 
the inputs are incomplete, initializes the
climate dataset, and runs the model for a
single point.

@author: clairevwilson
"""
# Built-in libraries
import argparse
import time
import os
import netCDF4
# External libraries
import numpy as np
import xarray as xr
import pandas as pd
# Internal libraries
import util.params as prms
from util.config import Config
from util.spatial import SpatialData
from util.output import Output
from pebsi.climate import Climate
from pebsi.massbalance import massBalance

try:
    import cupy as cp 
    xp = cp
    from shading.gpu_shading import Shading
except:
    xp = np
    from shading.shading import Shading

# START TIMER
start_time = time.time()

def get_args(parse=True):
    """
    Defines config class

    Parameters
    ==========
    parse : Bool
        If True, parses the command line (returns args)
        If False, returns the parser
    """    
    parser = argparse.ArgumentParser(description='energy balance model runs')

    # CONFIG FILE (any command-line args will overwrite the config file)
    parser.add_argument('-c', '--use_config', action='store_true', default=prms.use_config,
                        help='load settings from config file?')
    parser.add_argument('-cf', '--config_fn', type=str, default=None,
                        help='filename of config yaml file')
    
    # GLACIERS
    parser.add_argument('-rgi_ids', type=str, nargs='+', default=None,
                        help='List of RGI IDs to run')
    parser.add_argument('-rgi_region', type=int, default=prms.rgi_region,
                        help='RGI O1 region to run')
    
    # MODEL TIME
    parser.add_argument('-start','--start_date', type=str, default=None,
                        help='pass str like datetime of model run start')
    parser.add_argument('-end','--end_date', default=None,
                        help='pass str like datetime of model run end')
    
    # USER OPTIONS
    parser.add_argument('-store_data', action='store_true',
                        help='store the model output?')
    parser.add_argument('-debug', action='store_true',
                        help='print debug statements?')
    parser.add_argument('-pb','--progress_bar', action='store_true',
                        help='show progress bar for main loop?')
    parser.add_argument('-out','--output_fn',type=str,default=None,
                        help='output file name excluding extension')
    
    # CLIMATE OPTOINS
    parser.add_argument('-use_aws', action='store_true',
                        help='use AWS or just reanalysis?')

    if parse:
        args = parser.parse_args()
        return args
    else:
        return parser

class PEBSI():
    def __init__(self, args):
        # parse args from config file, command line and default params
        self.config = Config(args)
        args = self.config.args

        # initialize spatial class for handling distributed data
        sd = SpatialData(args)

        # create spatial points and load the DEM info for points
        sd.get_points()
        sd.load_dem_info()

        # check if shading model needs to be run anywhere
        os.makedirs(args.shading_fp, exist_ok=True)
        existing_shading = os.listdir(args.shading_fp)
        missing_shading = [f for f in args.rgi_ids if f + '.nc' not in existing_shading]
        if len(missing_shading) > 0:
            # *** run shading here (need fast implementation)
            pass

        # validate spatial inputs
        sd.validate_spatial_data()
        self.sd = sd
        return

    def get_climate(self):
        """
        Loads climate data for the simulation
        """
        # initialize the climate class
        climate = Climate(self.config.args, self.sd)
        climate.get_data()

        # validate the climate inputs
        climate.check_ds()

        # adjust elevation-dependent variables
        climate.adjust_to_elevation()

        # load data for emulator
        if self.config.args.method_snicar == 'emulator':
            climate.get_emulator_inputs()

        return climate

    def run(self, store_attrs=None):
        """
        Executes model functions and stores 
        output data.

        Parameters
        ==========
        store_attrs : dict
            Dictionary of additional metadata to store 
            in the model output .nc
        """
        # ===== INITIALIZE THE INPUTS =====
        self.args = args = self.config.args 
        self.climate = self.get_climate()

        # ===== INITIALIZE THE OUTPUTS =====
        dates = self.climate.dates
        self.output = Output(dates, args, self.sd)

        # ===== PRINT MODEL RUN INFO =====
        start = pd.to_datetime(args.start_date)
        end = pd.to_datetime(args.end_date)
        n_months = xp.round((end-start)/pd.Timedelta(days=30))
        start_fmtd = start.month_name()+', '+str(start.year)
        if self.sd.n == 1:
            id = args.rgi_id[0]
            elev = self.sd.elev_n[0]
            print(f'~ Running {id} at {elev} m a.s.l. for {n_months} months starting in {start_fmtd} ~')
        else:
            print(f'~ Running {self.sd.n} points in region {args.rgi_region} for {n_months} months starting in {start_fmtd} ~')

        # ===== RUN ENERGY BALANCE =====
        massbal = massBalance(self)
        massbal.main()
        # ==============================

        # get final model run time
        end_time = time.time()
        time_elapsed = end_time-start_time
        if args.debug:
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print(f'~ Model run complete in {time_elapsed:.1f} seconds ~')

        # store metadata in netcdf and save result
        if args.store_data:
            model.output.add_vars()
            model.output.add_basic_attrs(args,time_elapsed,self.climate)
            model.output.add_attrs(store_attrs)
            out = model.output.get_output()
        else:
            print('~ Success: data was not saved ~')
            out = None

        # delete any temporary files
        model.output.delete_temp_files()
        
        # print the final mass balance
        if isinstance(out, xr.Dataset) and args.debug:
            mb_out = out.accum + out.refreeze - out.melt
            print(f'Net mass balance: {mb_out.sum().values:.3f} m w.e.')
        
        return out

# execute the model if this script is called from command line
if __name__ == '__main__':
    # get command-line args
    args = get_args()

    # initialize the model
    model = PEBSI(args)

    # run the model
    model.run()