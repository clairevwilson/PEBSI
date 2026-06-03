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
import jax.numpy as jnp
# Internal libraries
import util.params as prms
from util.config import Config
from util.terrain import Terrain
from util.output import Output
from util.climate import Climate
from util.layers import Layers
from pebsi.state import *
# from pebsi.massbalance import run_mass_balance_model

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
        self.args = self.config.args 
        return
    
    def prepare_inputs(self):
        """
        Initializes the spatial (elevation, shading, etc.;
        vertical layer temperature, density, etc.) and
        temporal (air temperature, precipitation, etc.) 
        inputs to the model.
        """

        # =========== SPATIAL DATA HANDLING ===========
        terrain = Terrain(self.config.args)

        # run DEM functions for shading, elevation, etc.
        terrain.run_dem_functions()

        # validate spatial inputs
        terrain.validate_terrain_data()

        # ====== TEMPORAL (CLIMATE) DATA HANDLING ======
        climate = Climate(self.config.args, terrain)
        climate.get_data()

        # validate the climate inputs
        climate.check_ds()

        # adjust elevation-dependent variables
        climate.adjust_to_elevation()

        # load data for emulator
        if self.config.args.method_snicar == 'emulator':
            climate.get_emulator_inputs()

        # ================== SHADING ==================
        terrain.load_shading(climate.dates_UTC)

        # ============================================
        self.terrain = terrain 
        self.climate = climate
        return
    
    def prepare_initial_state(self):
        # ================== LAYERS ==================
        layers = Layers(args, self.climate, self.terrain)

        # initialize layer properties
        layers.initialize_layers()

        # initialize LAPs (black carbon, organic carbon, and dust)
        layers.initialize_LAPs()
        
        # ============================================
        self.layers = layers
        return
    
    def pack_states(self):
        # CONSTANTS
        self.N_POINTS = N_POINTS = self.terrain.N_POINTS
        self.N_LAYERS = self.layers.N_LAYERS
        CTOK = self.args.celsius_to_kelvin

        # ================== CLIMATE ==================
        forcings = ClimateState(
            # basic climate variables
            tempC=jnp.array(self.climate.temp, dtype=jnp.float32),
            tempK=jnp.array(self.climate.temp, dtype=jnp.float32) + CTOK,
            wind=jnp.array(self.climate.wind, dtype=jnp.float32),
            rh=jnp.array(self.climate.rh, dtype=jnp.float32),
            sp=jnp.array(self.climate.sp, dtype=jnp.float32),
            tcc=jnp.array(self.climate.tcc, dtype=jnp.float32),

            # deposition fluxes for light-absorbing particles
            bcwet=jnp.array(self.climate.bcwet, dtype=jnp.float32),
            bcdry=jnp.array(self.climate.bcdry, dtype=jnp.float32),
            ocwet=jnp.array(self.climate.ocwet, dtype=jnp.float32),
            ocdry=jnp.array(self.climate.ocdry, dtype=jnp.float32),
            dustwet=jnp.array(self.climate.dustwet, dtype=jnp.float32),
            dustdry=jnp.array(self.climate.dustdry, dtype=jnp.float32),

            # radiation terms
            shortwave_in=jnp.array(self.climate.SWin, dtype=jnp.float32),
            longwave_in=jnp.array(self.climate.LWin, dtype=jnp.float32),
            shadow_mask=jnp.array(self.terrain.shadow_mask, dtype=bool),
        )

        # ================== GLACIERS ==================
        grid = GlacierState(
            step_idx=jnp.array(0, dtype=jnp.int32),

            # surface properties
            albedo=jnp.full((N_POINTS,), self.args.albedo_fresh_snow, dtype=jnp.float32),
            albedo_surr=jnp.full((N_POINTS,), self.args.albedo_fresh_snow, dtype=jnp.float32),
            surftemp=jnp.zeros((N_POINTS,), 0.0, dtype=jnp.float32),

            # these may not need to be stored to state --- they will be passed to func
            # previous_mass=jnp.array(self.layers.mass, dtype=jnp.float32),
            # previous_ice=jnp.array(self.layers.mass_ice, dtype=jnp.float32),
            # previous_water=jnp.array(self.layers.mass_water, dtype=jnp.float32),

            # trackers
            delayed_snow=jnp.zeros((N_POINTS,), dtype=jnp.float32),
            annual_firn_converted=jnp.zeros((N_POINTS,), dtype=bool),
            annual_min_albedo=jnp.ones((N_POINTS,), dtype=jnp.float32),
            annual_max_snow=jnp.array(self.layers.max_snow, dtype=jnp.float32),
            days_since_snowfall=jnp.zeros((N_POINTS,), dtype=jnp.int8),

            # layer properties
            lheight=jnp.array(self.layers.lheight, dtype=jnp.float32),
            ldepth=jnp.array(self.layers.ldepth, dtype=jnp.float32),
            ltype=jnp.array(self.layers.ltype, dtype=jnp.int8),
            lice=jnp.array(self.layers.lice, dtype=jnp.float32),
            lwater=jnp.array(self.layers.lwater, dtype=jnp.float32),
            ltemp=jnp.array(self.layers.ltemp, dtype=jnp.float32),
            ldensity=jnp.array(self.layers.ldensity, dtype=jnp.float32),
            lage=jnp.array(self.layers.lage, dtype=jnp.int8),
            lgrainsize=jnp.array(self.layers.lgrainsize, dtype=jnp.float32),
            lrefreeze=jnp.array(self.layers.lrefreeze, dtype=jnp.float32),
            dlrefreeze=jnp.array(self.layers.dlrefreeze, dtype=jnp.float32),
            lBC=jnp.array(self.layers.lBC, dtype=jnp.float32),
            lOC=jnp.array(self.layers.lOC, dtype=jnp.float32),
            ldust=jnp.array(self.layers.ldust, dtype=jnp.float32),
        )

        point_attrs = PointAttributes(
            elevation=jnp.array(self.sd.elev_n, dtype=jnp.float32),
            slope=jnp.array(self.sd.slope_n, dtype=jnp.float32),
            aspect=jnp.array(self.sd.aspect_n, dtype=jnp.float32),
            timezone=jnp.array(self.sd.tz_n, dtype=jnp.float32)
        )
        return forcings, grid, point_attrs

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
        # ======== INITIALIZE THE INPUTS ========
        self.prepare_inputs()
        self.prepare_initial_state()
        forcings, grid, point_attrs = self.pack_states()

        # ======== INITIALIZE THE OUTPUTS ========
        # dates = self.climate.dates
        # self.output = Output(dates, args, self.terrain)

        # ========== RUN ENERGY BALANCE ==========
        self.start_prints()
        final_state, record = run_mass_balance_model(
            grid, forcings, point_attrs, self.args
        )
        self.stop_prints(model, store_attrs)
        # ========================================
        return
    
    def start_prints(self):
        # get info about the simulation time
        start = pd.to_datetime(self.args.start_date)
        end = pd.to_datetime(self.args.end_date)
        n_months = np.round((end-start) / pd.Timedelta(days=30))
        start_fmtd = start.month_name()+', '+str(start.year)

        # print starting statement
        if self.terrain.N_POINTS == 1:
            id = self.args.rgi_id[0]
            elev = self.terrain.elev_n[0]
            print(f'~ Running {id} at {elev} m a.s.l. for {n_months} months starting in {start_fmtd} ~')
        else:
            print(f'~ Running {self.terrain.N_POINTS} points in region {self.args.rgi_region} for {n_months} months starting in {start_fmtd} ~')
        return

    def stop_prints(self, model, store_attrs):
        # get final model run time
        end_time = time.time()
        time_elapsed = end_time-start_time

        # print timer
        if self.args.debug:
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print(f'~ Model run complete in {time_elapsed:.1f} seconds ~')

        # store metadata in netcdf and save result
        if self.args.store_data:
            pass
            # model.output.add_vars()
            # model.output.add_basic_attrs(self.args,time_elapsed,self.climate)
            # model.output.add_attrs(store_attrs)
            # out = model.output.get_output()
        else:
            print('~ Success: data was not saved ~')
            out = None
        
        # print the final mass balance
        if isinstance(out, xr.Dataset) and self.args.debug:
            mb_out = out.accum + out.refreeze - out.melt
            print(f'Net mass balance: {mb_out.sum().values:.3f} m w.e.')
        return out

# execute the model if this script is called from command line
if __name__ == '__main__':
    # get command-line args
    args = get_args()

    # initialize and run the model
    model = PEBSI(args)
    model.run()