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
import itertools
import threading
import time
import warnings
# External libraries
import jax
import numpy as np
import pandas as pd
import jax.numpy as jnp
import netCDF4
from tqdm import tqdm
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
os.environ["JAX_TRACEBACK_FILTERING"] = "off"   # show full error statements
jax.config.update("jax_enable_x64", True)       # enable floaf64 storage

warnings.filterwarnings("ignore", category=FutureWarning, module="jax")

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
    
    def prepare_spatial_inputs(self):
        """
        Initializes the spatial (elevation, shading, etc.)
        inputs to the model.
        """
        params = self.params
        all_dates_UTC = pd.date_range(params.start_date, params.end_date, freq='h')
        self.dates = all_dates_UTC

        # =========== SPATIAL DATA HANDLING ===========
        terrain = Terrain(params)

        # run DEM functions for shading, elevation, etc.
        terrain.run_dem_functions()

        # validate spatial inputs
        terrain.validate_terrain_data()

        # ================== SHADING ==================
        # grab one full year of shading data for indexing
        year_end = self.dates[0] + pd.Timedelta(days=366)
        terrain.load_shading(
            pd.date_range(self.dates[0], year_end, freq='h')
        )

        self.terrain = terrain
        return
    
    def prepare_initial_state(self):
        """
        Initializes the vertical structure (layer 
        temperature, density, etc.) across points.
        """
        # ================== LAYERS ==================
        layers = Layers(self.params, self.terrain)

        # initialize layer properties
        layers.initialize_layers()

        self.layers = layers
        return
    
    def pack_states(self):
        """
        Packs the terrain and layer inputs into 
        JAX-compatible states.
        """
        params = self.params 

        # CONSTANTS
        self.N_POINTS = N_POINTS = self.terrain.N_POINTS
        self.N_LAYERS = self.layers.N_LAYERS

        # get number of years in the simulation
        N_YEARS = len(np.unique(self.dates.year))

        # get number of steps to look back for accumulation start
        N_PAST_STEPS = params.new_snow_days * 24

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
            past_snow=jnp.zeros((N_POINTS, N_PAST_STEPS), dtype=jnp.float64),
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
            latitude=jnp.array(self.terrain.lat_n, dtype=jnp.float64),
            longitude=jnp.array(self.terrain.lon_n, dtype=jnp.float64),
            elevation=jnp.array(self.terrain.elev_n, dtype=jnp.float64),
            slope=jnp.array(self.terrain.slope_n, dtype=jnp.float64),
            aspect=jnp.array(self.terrain.aspect_n, dtype=jnp.float64),
            sky_view_factor=jnp.array(self.terrain.sky_view_factor, dtype=jnp.float64),
        )
        return glacier_state, point_attrs
    
    def pack_forcings(self, params, dates, start):
        """
        Packs the climate forcings for a temporal
        chunk into JAX-compatible state.
        """
        # initiate the climate class for these dates
        climate = Climate(dates, params, self.terrain)
        climate.get_data()

        # process the dataset for bias, elevation, etc.
        climate.process_climate()
        self.climate = climate
        
        # define conversion factors 
        CTOK = self.params.celsius_to_kelvin
        SPH = self.params.seconds_per_hour

        # slice solar inputs from terrain by matching (dayofyear, hour)
        date_mask = self.terrain.shading_dates.isin(dates)
        shadow_mask = self.terrain.shadow_mask[:, date_mask]
        solar_azimuth = self.terrain.solar_azimuth[:, date_mask]
        solar_zenith = self.terrain.solar_zenith[:, date_mask]

        # ================== CLIMATE ==================
        # per-point local solar hour: (N_TIME, N_POINTS)
        tz_offset = np.round(self.terrain.lon_n / 15).astype(int)  # (N_POINTS,)
        local_hour = (dates.hour.values[:, None] + tz_offset[None, :]) % 24

        forcings = ClimateState(
            time_idx=jnp.array(jnp.arange(start, start + len(dates)), dtype=jnp.int32),
            year=jnp.array(dates.year, dtype=jnp.int32),
            month=jnp.array(dates.month, dtype=jnp.int32),
            day=jnp.array(dates.day, dtype=jnp.int32),
            hour=jnp.array(dates.hour, dtype=jnp.int32),
            local_hour=jnp.array(local_hour, dtype=jnp.int32),
            doy=jnp.array(dates.day_of_year, dtype=jnp.int32),

            # basic climate variables
            tempC=jnp.array(climate.temp, dtype=jnp.float64).T,
            tempK=jnp.array(climate.temp, dtype=jnp.float64).T + CTOK,
            tp=jnp.array(climate.tp, dtype=jnp.float64).T,
            prec=jnp.array(climate.tp, dtype=jnp.float64).T / SPH,
            wind=jnp.array(climate.wind, dtype=jnp.float64).T,
            winddir=jnp.array(climate.winddir, dtype=jnp.float64).T,
            rh=jnp.array(climate.rh, dtype=jnp.float64).T,
            sp=jnp.array(climate.sp, dtype=jnp.float64).T,
            tcc=jnp.array(climate.tcc, dtype=jnp.float64).T,

            # deposition fluxes for light-absorbing particles
            bcwet=jnp.array(climate.bcwet, dtype=jnp.float64).T,
            bcdry=jnp.array(climate.bcdry, dtype=jnp.float64).T,
            ocwet=jnp.array(climate.ocwet, dtype=jnp.float64).T,
            ocdry=jnp.array(climate.ocdry, dtype=jnp.float64).T,
            dustwet=jnp.array(climate.dustwet, dtype=jnp.float64).T,
            dustdry=jnp.array(climate.dustdry, dtype=jnp.float64).T,

            # radiation terms
            shortwave_in=jnp.array(climate.SWin, dtype=jnp.float64).T,
            longwave_in=jnp.array(climate.LWin, dtype=jnp.float64).T,
            shadow_mask=jnp.array(shadow_mask, dtype=bool).T,
            solar_azimuth=jnp.array(solar_azimuth, dtype=jnp.float64).T,
            solar_zenith=jnp.array(solar_zenith, dtype=jnp.float64).T,
        )

        return forcings
    
    def _run_chunk(self, state, point_attrs, static_args, dynamic_args, chunk_dates, start):
        """
        Packs forcings for one temporal chunk, runs main(), and returns
        the updated state and trimmed output records.
        """
        chunk_size = self.params.temporal_chunks
        actual_length = len(chunk_dates)

        chunk_forcings = self.pack_forcings(self.params, chunk_dates, start)

        if actual_length < chunk_size:
            pad_amt = chunk_size - actual_length
            chunk_forcings = jax.tree.map(
                lambda x: jnp.pad(x, ((0, pad_amt),) + ((0, 0),) * (x.ndim - 1)),
                chunk_forcings
            )

        state, chunk_records = main(state, chunk_forcings, point_attrs, static_args, dynamic_args)
        jax.effects_barrier()

        if actual_length < chunk_size:
            chunk_records = jax.tree.map(lambda x: x[:actual_length], chunk_records)

        return state, chunk_records

    def run(self):
        """
        Executes model functions and stores the output data.
        Runs a one-year spin-up before the main loop to allow
        initialization to stabilize; spin-up output is discarded.
        The main() function is chunked so JAX compiles once and
        memory use stays bounded.
        """
        static_args = self.config.static_args
        dynamic_args = self.config.dynamic_args
        params = self.config.params

        self.prepare_spatial_inputs()
        self.prepare_initial_state()
        initial_state, point_attrs = self.pack_states()

        model_output = Output(params, self.terrain)
        self.start_print()

        chunk_size = params.temporal_chunks

        # ========== SPIN-UP ==========
        spinup_dates = pd.date_range(
            self.dates[0], self.dates[0] + pd.Timedelta(days=365), freq='h'
        )
        spinup_steps = range(0, len(spinup_dates), chunk_size)
        dots = ['.  ', '.. ', '...', '   ']

        state = initial_state
        print('~ Spinning up ', end='', flush=True)
        for i, start in enumerate(spinup_steps):
            chunk_dates = spinup_dates[start:start + chunk_size]
            state, _ = self._run_chunk(state, point_attrs, static_args, dynamic_args, chunk_dates, start)
            print(f'\r~ Spinning up {dots[i % len(dots)]}', end='', flush=True)
        print(f'\r~ Spin-up complete ({time.time() - self.start_time:.0f} s elapsed)   ')

        # reset annual trackers: spin-up and year 0 are the same calendar year
        state = state._replace(
            annual_min_albedo=jnp.ones_like(state.annual_min_albedo),
            annual_firn_converted=jnp.zeros_like(state.annual_firn_converted),
        )

        # ========== MAIN LOOP ==========
        total_steps = len(self.dates)
        total_chunks = (total_steps + chunk_size - 1) // chunk_size
        chunk_iter = tqdm(
            range(0, total_steps, chunk_size),
            desc='~ Simulating',
            unit='chunk',
            disable=not params.progress_bar,
        )

        for start in chunk_iter:
            chunk_dates = self.dates[start:start + chunk_size]
            chunk_start = time.time()

            state, chunk_records = self._run_chunk(state, point_attrs, static_args, dynamic_args, chunk_dates, start)

            elapsed = time.time() - chunk_start
            if params.progress_bar:
                chunk_iter.set_postfix(chunk_t=f'{elapsed:.1f}s')
            elif params.debug:
                chunk_num = start // chunk_size + 1
                print(f'. . . chunk {chunk_num}/{total_chunks} in {elapsed:.1f}s')

            if params.store_data:
                model_output.store_chunk(chunk_records, chunk_dates, start)

            del chunk_records

        time_elapsed = time.time() - self.start_time
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(f'~ Simulation completed in {time_elapsed:.2f} seconds ~')

        if self.params.store_data:
            model_output.close_out(params, time_elapsed, self.climate)
        else:
            print('~ Success: data was not saved ~')
        return state
    
    def start_print(self):
        """Command-line printout when a simulation is started"""
        # get info about the simulation time
        start = pd.to_datetime(self.params.start_date)
        end = pd.to_datetime(self.params.end_date)
        n_months = np.round((end-start) / pd.Timedelta(days=30))
        start_fmtd = start.month_name()+', '+str(start.year)

        # print starting statement
        if self.terrain.N_POINTS == 1:
            id = self.params.rgi_ids[0]
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