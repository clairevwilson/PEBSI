"""
Main script to execute PEBSI

Parses the command-line arguments, loads all
of the initial states and forcing data,
and runs the main() function.

@author: clairevwilson
"""
# Built-in libraries
import os
import argparse
import time
import warnings
# External libraries
import jax 
import zarr
import numpy as np
import pandas as pd
import jax.numpy as jnp
import netCDF4
# Internal libraries
import util.defaults as defaults
from util.config import *
from util.terrain import Terrain
from util.output import Output
from util.climate import Climate
from util.layers import Layers
from pebsi.state import *
from pebsi.main import main

os.umask(0o000) # make sure files are created with universal permissions
os.environ["JAX_TRACEBACK_FILTERING"] = "off" # show full error statements
jax.config.update("jax_enable_x64", True) # enable floaf64 storage

warnings.filterwarnings("ignore", category=FutureWarning, module="jax")
warnings.filterwarnings('ignore', category=zarr.errors.ZarrUserWarning)

def get_args(parse=True):
    """
    Loads command-line arguments

    Parameters
    ==========
    parse : Bool
        If True, parses the command line (returns args)
        If False, returns the parser
    """    
    parser = argparse.ArgumentParser(description='energy balance model runs')

    # CONFIG FILE (any command-line args will overwrite the config file)
    parser.add_argument('-c', '--use_config', action='store_true', 
                        default=defaults.use_config,
                        help='load settings from config file?')
    parser.add_argument('-cf', '--config_fn', type=str, default=None,
                        help='filename of config yaml file')
    
    # GLACIERS
    parser.add_argument('-rgi_ids', type=str, nargs='+', default=None,
                        help='List of RGI IDs to run (overrides rgi_region)')
    parser.add_argument('-rgi_region', type=int, 
                        default=defaults.rgi_region,
                        help='RGI O1 region to run (all glaciers in this region if rgi_ids not specified)')
    
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
        """
        Initializes the configuration for a simulation.
        """
        self.start_time = time.time() 

        # parse args in order of defaults < config yaml < command line
        self.config = Config(args)

        # params contains both static and dynamic parameters
        self.params = self.config.params 
        return
    
    def prepare_inputs(self):
        """
        Initializes the spatial (elevation, shading, etc.) and
        temporal (air temperature, precipitation, etc.) 
        inputs to the model.
        """
        params = self.params

        # =========== SPATIAL DATA HANDLING ===========
        terrain = Terrain(params)

        # run DEM functions for shading, elevation, etc.
        terrain.run_dem_functions()

        # validate spatial inputs
        terrain.validate_terrain_data()

        # ====== TEMPORAL (CLIMATE) DATA HANDLING ======
        climate = Climate(params, terrain)
        climate.get_data()

        # process the dataset for elevation, bias, etc.
        climate.process_climate()

        # adjust elevation-dependent variables
        climate.adjust_to_elevation()

        # precompute the upcoming snowfall amounts
        climate.precompute_upcoming_snow()

        # ================== SHADING ==================
        terrain.load_shading(climate.dates_UTC)

        # ============================================
        self.terrain = terrain 
        self.climate = climate
        return
    
    def prepare_initial_state(self):
        """
        Initializes the vertical structure (layer 
        temperature, density, etc.) across points.
        """
        # ================== LAYERS ==================
        layers = Layers(
            self.params, self.climate, self.terrain
        )

        # initialize layer properties
        layers.initialize_layers()
        
        # ============================================
        self.layers = layers
        return
    
    def pack_states(self):
        """
        Packs the terrain, layer, and climate inputs into 
        JAX-compatible states.
        """
        params = self.params 

        # CONSTANTS
        self.N_POINTS = N_POINTS = self.terrain.N_POINTS
        self.N_LAYERS = self.layers.N_LAYERS
        CTOK = self.params.celsius_to_kelvin
        SPH = self.params.seconds_per_hour

        # get timestamps in Pandas format
        dates = pd.to_datetime(self.climate.dates)
        N_YEARS = len(np.unique(dates.year))

        # ================== CLIMATE ==================
        forcings = ClimateState(
            time_idx=jnp.array(jnp.arange(len(dates)), dtype=jnp.int32),
            year=jnp.array(dates.year, dtype=jnp.int32),
            month=jnp.array(dates.month, dtype=jnp.int32),
            day=jnp.array(dates.day, dtype=jnp.int32),
            hour=jnp.array(dates.hour, dtype=jnp.int32),
            doy=jnp.array(dates.day_of_year, dtype=jnp.int32),

            # basic climate variables
            tempC=jnp.array(self.climate.temp, dtype=jnp.float64).T,
            tempK=jnp.array(self.climate.temp, dtype=jnp.float64).T + CTOK,
            tp=jnp.array(self.climate.tp, dtype=jnp.float64).T,
            prec=jnp.array(self.climate.tp, dtype=jnp.float64).T / SPH,
            wind=jnp.array(self.climate.wind, dtype=jnp.float64).T,
            winddir=jnp.array(self.climate.winddir, dtype=jnp.float64).T,
            rh=jnp.array(self.climate.rh, dtype=jnp.float64).T,
            sp=jnp.array(self.climate.sp, dtype=jnp.float64).T,
            tcc=jnp.array(self.climate.tcc, dtype=jnp.float64).T,
            upcoming_snow=jnp.array(self.climate.upcoming_snow, dtype=jnp.float64).T,

            # deposition fluxes for light-absorbing particles
            bcwet=jnp.array(self.climate.bcwet, dtype=jnp.float64).T,
            bcdry=jnp.array(self.climate.bcdry, dtype=jnp.float64).T,
            ocwet=jnp.array(self.climate.ocwet, dtype=jnp.float64).T,
            ocdry=jnp.array(self.climate.ocdry, dtype=jnp.float64).T,
            dustwet=jnp.array(self.climate.dustwet, dtype=jnp.float64).T,
            dustdry=jnp.array(self.climate.dustdry, dtype=jnp.float64).T,

            # radiation terms
            shortwave_in=jnp.array(self.climate.SWin, dtype=jnp.float64).T,
            longwave_in=jnp.array(self.climate.LWin, dtype=jnp.float64).T,
            shadow_mask=jnp.array(self.terrain.shadow_mask, dtype=bool).T,
            solar_azimuth=jnp.array(self.terrain.solar_azimuth, dtype=jnp.float64).T,
            solar_zenith=jnp.array(self.terrain.solar_zenith, dtype=jnp.float64).T,
        )

        # ================== GLACIERS ==================

        # time-varying spatial and layer attributes
        glacier_state = GlacierState(
            # surface properties
            albedo=jnp.full((N_POINTS,), params.albedo_fresh_snow, dtype=jnp.float64),
            albedo_surr=jnp.full((N_POINTS,), params.albedo_fresh_snow, dtype=jnp.float64),
            surftemp=jnp.full((N_POINTS,), params.surftemp_guess, dtype=jnp.float64),
            roughness=jnp.full((N_POINTS,), params.roughness_fresh_snow, dtype=jnp.float64),
            last_snow=jnp.zeros((N_POINTS,), dtype=jnp.int32),

            # trackers
            delayed_snow=jnp.zeros((N_POINTS,), dtype=jnp.float64),
            annual_firn_converted=jnp.zeros((N_POINTS,), dtype=bool),
            annual_min_albedo=jnp.ones((N_POINTS, N_YEARS), dtype=jnp.float64),
            annual_max_snow=jnp.array(self.layers.max_snow, dtype=jnp.float64),
            days_since_snowfall=jnp.zeros((N_POINTS,), dtype=jnp.int32),
            cum_mass_error=jnp.zeros((N_POINTS), dtype=jnp.float64),
            basal_reservoir=jnp.zeros((N_POINTS), dtype=jnp.float64),

            # layer properties
            lheight=jnp.array(self.layers.lheight, dtype=jnp.float64),
            ldepth=jnp.array(self.layers.ldepth, dtype=jnp.float64),
            snow_mask=jnp.array(self.layers.snow_mask, dtype=bool),
            firn_mask=jnp.array(self.layers.firn_mask, dtype=bool),
            ice_mask=jnp.array(self.layers.ice_mask, dtype=bool),
            ltype=jnp.array(self.layers.ltype, dtype=jnp.int32),
            lice=jnp.array(self.layers.lice, dtype=jnp.float64),
            lwater=jnp.array(self.layers.lwater, dtype=jnp.float64),
            ltemp=jnp.array(self.layers.ltemp, dtype=jnp.float64),
            ldensity=jnp.array(self.layers.ldensity, dtype=jnp.float64),
            lage=jnp.array(self.layers.lage, dtype=jnp.int32),
            lgrainsize=jnp.array(self.layers.lgrainsize, dtype=jnp.float64),
            lrefreeze=jnp.array(self.layers.lrefreeze, dtype=jnp.float64),
            ldrefreeze=jnp.array(self.layers.drefreeze, dtype=jnp.float64),
            lBC=jnp.array(self.layers.lBC, dtype=jnp.float64),
            lOC=jnp.array(self.layers.lOC, dtype=jnp.float64),
            ldust=jnp.array(self.layers.ldust, dtype=jnp.float64),
        
        )

        # time-invariant spatial attributes
        point_attrs = PointAttributes(
            elevation=jnp.array(self.terrain.elev_n, dtype=jnp.float64),
            slope=jnp.array(self.terrain.slope_n, dtype=jnp.float64),
            aspect=jnp.array(self.terrain.aspect_n, dtype=jnp.float64),
            timezone=jnp.array(self.terrain.tz_n, dtype=jnp.float64),
            sky_view_factor=jnp.array(self.terrain.sky_view_factor, dtype=jnp.float64),
        )
        return glacier_state, forcings, point_attrs
    
    def run(self):
        """
        Executes model functions and stores the
        output data. The main() function is run in
        chunks so JAX only has to recompile at most
        two times (once for the main temporal_chunks
        size, and once for the remainder.)
        """
        static_args = self.config.static_args
        dynamic_args = self.config.dynamic_args 
        params = self.config.params

        # ======== INITIALIZE THE INPUTS ========
        self.prepare_inputs()
        self.prepare_initial_state()
        initial_state, all_forcings, point_attrs = self.pack_states()

        # ======== INITIALIZE THE OUTPUTS ========
        dates = self.climate.dates
        model_output = Output(params, self.terrain)

        self.start_print()

        # ========== RUN ENERGY BALANCE ==========
        total_steps = len(dates)
        state = initial_state
        chunk_size = params.temporal_chunks

        # start model timer
        total_chunks = (total_steps + chunk_size - 1) // chunk_size
        model_timer = ProgressTimer(total_chunks)

        for start in range(0, total_steps, chunk_size):
            # crop forcings to the temporal subset
            end = min(start + chunk_size, total_steps)
            actual_length = end - start
            chunk_forcings = jax.tree.map(lambda x: x[start:end], all_forcings)
           
           # pad the forcings so the shape matches temporal_chunks
            if actual_length < chunk_size:
                pad_amt = chunk_size - actual_length
                chunk_forcings = jax.tree.map(
                    lambda x: jnp.pad(x, ((0, pad_amt),) + ((0, 0),) * (x.ndim - 1)), 
                    chunk_forcings
                )

            # RUN MODEL FOR ONE TIME CHUNK
            before = time.time()
            state, chunk_records = main(
                state, chunk_forcings, point_attrs, static_args, dynamic_args
            )
            jax.effects_barrier()
            if params.debug:
                print(f'. . . chunk {start / chunk_size + 1} /  in {time.time() - before:.1f} s')

            # remove the padded garbage
            if actual_length < chunk_size:
                chunk_records = jax.tree.map(lambda x: x[:actual_length], chunk_records)

            # store this chunk (append it onto the zarr)
            if params.store_data:
                model_output.store_chunk(chunk_records, dates[start:end], start)
            
            # clear chunk records from RAM
            del chunk_records

        # ============== END TIMER ===============
        time_elapsed = time.time() - self.start_time
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(f'~ Simulation completed in {time_elapsed:.2f} seconds ~')

        # ============ STORE OUTPUT ==============
        if self.params.store_data:
            model_output.close_out(params, time_elapsed, self.climate)
        else:
            print('~ Success: data was not saved ~')
        return state
    
    def start_print(self):

        # get info about the simulation time
        start = pd.to_datetime(self.params.start_date)
        end = pd.to_datetime(self.params.end_date)
        n_months = np.round((end-start) / pd.Timedelta(days=30))
        start_fmtd = start.month_name()+', '+str(start.year)

        # print starting statement
        if self.terrain.N_POINTS == 1:
            id = self.params.rgi_id[0]
            elev = self.terrain.elev_n[0]
            print(f'~ Running {id} at {elev} m a.s.l. for {n_months} months starting in {start_fmtd} ~')
        else:
            print(f'~ Running {self.terrain.N_POINTS} points in region {self.params.rgi_region} for {n_months} months starting in {start_fmtd} ~')
        return
    
# execute the model if this script is called from command line
if __name__ == '__main__':
    # get command-line args
    args = get_args()

    # initialize and run the model
    model = PEBSI(args)
    model.run()