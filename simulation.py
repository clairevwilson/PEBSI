"""
Main script to execute PEBSI

Parses the command-line arguments, loads all
of the initial states and forcing data,
and runs the main() function.

@author: clairevwilson
"""
# Built-in libraries
import os
import sys
import shutil
import argparse
import itertools
import threading
import time
import warnings
import yaml
# External libraries
import jax
import numpy as np
import pandas as pd
import jax.numpy as jnp
import netCDF4
# Internal libraries
import pebsi.defaults as defaults
from pebsi.config import *
from pebsi.io.terrain import Terrain
from pebsi.io.output import Output
from pebsi.io.climate import Climate
from pebsi.physics.layers import Layers
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
    parser.add_argument('-ids', '--rgi_ids', type=str, nargs='+', default=None,
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
    
    # RESUME
    parser.add_argument('-resume_from', type=str, default=None,
                        help='output directory of a crashed run to resume from')

    # CLIMATE OPTOINS
    parser.add_argument('-use_aws', action='store_true',
                        help='use AWS or just reanalysis?')

    # TESTING
    parser.add_argument('--testing', action='store_true',
                        help='run the test glacier (RGI ID 00.00000) with sample data?')

    if parse:
        args = parser.parse_args()
        return args
    else:
        return parser

class PEBSI():
    """
    Functions which prepare the inputs, execute the model,
    and store the outputs. 
    """
    # ----------------------------------------------------------------- #
    #                        INITIALIZATION
    # ----------------------------------------------------------------- #
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
        if params.option_ice_albedo_tif:
            albedo_ice_n = terrain.get_ice_albedo()
            self.params.albedo_ice = albedo_ice_n
            self.config.dynamic_args = self.config.dynamic_args._replace(
                albedo_ice=jnp.array(albedo_ice_n, dtype=jnp.float64)
            )
        if params.option_windmaps:
            terrain.get_wind_fields()

        # validate spatial inputs
        terrain.validate_terrain_data()

        # ================== SHADING ==================
        terrain.load_shading()

        # ============ GLACIER DYNAMICS (optional) ============
        if params.option_dynamics:
            terrain.get_initial_ice_thickness()

        # ============= CLIMATE DATA GRID =============
        # build a lightweight Climate object to get cell mapping
        _cl = Climate.__new__(Climate)
        _cl.params = self.params
        _cl.terrain = terrain
        _cl.get_vardict()
        _cl.get_unique_cells()

        # save to self
        self._cl = _cl
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
    
    # ----------------------------------------------------------------- #
    #                       PACKING JAX ARRAYS
    # ----------------------------------------------------------------- #
    
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
            surftype=jnp.zeros((N_POINTS,), dtype=jnp.int32),
            roughness=jnp.full((N_POINTS,), params.roughness_fresh_snow / 1000, dtype=jnp.float64),
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
        # per-cell elevation arrays are zero here; filled in pack_forcings()

        N_UNIQUE = self._cl.N_UNIQUE
        N_POINTS = self.terrain.N_POINTS

        if params.option_windmaps:
            wind_spdup = jnp.array(self.terrain.spdup_n, dtype=jnp.float64)
            wind_directions = jnp.array(self.terrain.wind_directions, dtype=jnp.float64)
        else:
            wind_spdup = jnp.ones((N_POINTS, 1), dtype=jnp.float64)
            wind_directions = jnp.zeros((1,), dtype=jnp.float64)

        point_attrs = PointAttributes(
            # per-point (N_POINTS,)
            latitude=jnp.array(self.terrain.lat_n, dtype=jnp.float64),
            longitude=jnp.array(self.terrain.lon_n, dtype=jnp.float64),
            elevation=jnp.array(self.terrain.elev_n, dtype=jnp.float64),
            slope=jnp.array(self.terrain.slope_n, dtype=jnp.float64),
            aspect=jnp.array(self.terrain.aspect_n, dtype=jnp.float64),
            sky_view_factor=jnp.array(self.terrain.sky_view_factor, dtype=jnp.float64),
            median_elev=jnp.array(self.terrain.median_elev_n, dtype=jnp.float64),
            cell_idx=jnp.array(self._cl.point_to_cell_idx, dtype=jnp.int32),
            wind_spdup=wind_spdup,
            wind_directions=wind_directions,

            # per-cell (N_UNIQUE,) dummy arrays
            gcm_elev=jnp.zeros(N_UNIQUE, dtype=jnp.float64),
            temp_elev=jnp.zeros(N_UNIQUE, dtype=jnp.float64),
            sp_elev=jnp.zeros(N_UNIQUE, dtype=jnp.float64),
            LWin_elev=jnp.zeros(N_UNIQUE, dtype=jnp.float64),
        )

        # store point_attrs to self since they are invariant
        self.point_attrs = point_attrs
        return glacier_state
    
    def pack_forcings(self, params, dates, start, spinup=False):
        """
        Packs the climate forcings for a temporal
        chunk into JAX-compatible state.
        """
        if not spinup and not params.progress_bar and hasattr(self, '_chunk_label'):
            ci, cn = self._chunk_label
            print(f'\033[2K\r~ Loading climate  [{ci}/{cn}] ~', end='', flush=True)

        # initiate the climate class for these dates
        climate = Climate(dates, params, self.terrain)
        climate.get_data()

        # process the dataset for bias, elevation, etc.
        climate.process_climate()
        self.climate = climate
        self.params.climate_measured_vars = climate.measured_vars
        self.params.climate_all_vars = climate.all_vars

        # update self.point_attrs with the real per-cell elevation
        if not getattr(self, '_cell_elevs_set', False):
            self.point_attrs = self.point_attrs._replace(
                gcm_elev=jnp.array(climate.terrain.gcm_elev_n, dtype=jnp.float64),
                temp_elev=jnp.array(climate.temp_elev, dtype=jnp.float64),
                sp_elev=jnp.array(climate.sp_elev, dtype=jnp.float64),
                LWin_elev=jnp.array(climate.LWin_elev, dtype=jnp.float64),
            )
            self._cell_elevs_set = True

        # slice solar inputs from terrain by (day of year, hour)
        shading_idx = [self.terrain.shading_lookup[(d, h)]
                       for d, h in zip(dates.dayofyear, dates.hour)]
        shadow_mask = self.terrain.shadow_mask[:, shading_idx]
        solar_azimuth = self.terrain.solar_azimuth[:, shading_idx]
        solar_zenith = self.terrain.solar_zenith[:, shading_idx]

        # ================== CLIMATE ==================
        # per-cell local solar hour: (N_TIME, N_UNIQUE)
        tz_offset = np.round(climate.unique_lons / 15).astype(int)  # (N_UNIQUE,)
        local_hour = (dates.hour.values[:, None] + tz_offset[None, :]) % 24

        # climate variables are (N_UNIQUE, N_TIME) in climate object;
        # transpose to (N_TIME, N_UNIQUE) for JAX scan
        forcings = ClimateState(
            time_idx=jnp.array(jnp.arange(start, start + len(dates)), dtype=jnp.int32),
            year=jnp.array(dates.year, dtype=jnp.int32),
            month=jnp.array(dates.month, dtype=jnp.int32),
            day=jnp.array(dates.day, dtype=jnp.int32),
            hour=jnp.array(dates.hour, dtype=jnp.int32),
            local_hour=jnp.array(local_hour, dtype=jnp.int32),
            doy=jnp.array(dates.day_of_year, dtype=jnp.int32),

            # basic climate variables
            temp=jnp.array(climate.temp, dtype=jnp.float64).T,
            tp=jnp.array(climate.tp, dtype=jnp.float64).T,
            wind=jnp.array(climate.wind, dtype=jnp.float64).T,
            winddir=jnp.array(climate.winddir, dtype=jnp.float64).T,
            rh=jnp.array(climate.rh, dtype=jnp.float64).T,
            sp=jnp.array(climate.sp, dtype=jnp.float64).T,
            tcc=jnp.array(np.nan_to_num(climate.tcc, nan=0.0), dtype=jnp.float64).T,

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
    
    # ----------------------------------------------------------------- #
    #                        CHECKPOINTS
    # ----------------------------------------------------------------- #
    
    def save_checkpoint(self, state, next_start, output_fp):
        """
        Saves GlacierState and loop position inside the 
        output directory, atomically.
        """
        path = os.path.join(output_fp, 'checkpoint.npz')
        tmp_path = os.path.join(output_fp, 'checkpoint.tmp.npz')

        # convert data to numpy arrays and store as .npz
        arrays = {field: np.array(getattr(state, field)) for field in state._fields}
        arrays['_next_start'] = np.array(next_start)
        np.savez(tmp_path, **arrays)
        os.replace(tmp_path, path)

    def load_checkpoint(self, resume_from):
        """
        Loads checkpoint.npz from the given output directory.
        Returns (GlacierState, next_start) or raises if not found.
        """
        path = os.path.join(resume_from, 'checkpoint.npz')
        if not os.path.exists(path):
            raise FileNotFoundError(f'No checkpoint found in {resume_from}')
        
        # load the data and re-build glacier state
        data = np.load(path)
        next_start = int(data['_next_start'])
        state = GlacierState(**{
            field: jnp.array(data[field])
            for field in GlacierState._fields
        })

        print(f'~ Resuming from step {next_start} in {resume_from} ~')
        return state, next_start
    
    # ----------------------------------------------------------------- #
    #                      SPIN-UP AND SIMULATION
    # ----------------------------------------------------------------- #

    def _run_chunk(self, state,
                   static_args, dynamic_args,
                   chunk_dates, start, chunk_size, spinup=False):
        """
        Packs forcings for one temporal chunk, runs main(), and returns
        the updated state and trimmed output records.

        chunk_size is a free variable for the case where temporal_chunk_hours
        are longer than the spinup period. In this case, the spinup runs one 
        temporal chunk of one year.
        """
        actual_length = len(chunk_dates)

        # pack the forcings for this temporal chunk
        chunk_forcings = self.pack_forcings(self.params, chunk_dates, start, spinup)

        if not spinup and not self.params.progress_bar and hasattr(self, '_chunk_label'):
            ci, cn = self._chunk_label
            print(f'\033[2K\r~ Running chunk    [{ci}/{cn}] ~', end='', flush=True)

        # check if we need to pad forcings with NaNs
        if actual_length < chunk_size:
            pad_amt = chunk_size - actual_length
            chunk_forcings = jax.tree.map(
                lambda x: jnp.pad(x, ((0, pad_amt),) + ((0, 0),) * (x.ndim - 1)),
                chunk_forcings
            )

        # run main for this temporal chunk
        state, chunk_records = main(state, chunk_forcings, self.point_attrs,
                                    static_args, dynamic_args)
        jax.effects_barrier()

        # crop the output records to the actual chunk size (in output periods,
        # not hours, since records may be aggregated to daily/monthly resolution)
        if actual_length < chunk_size:
            steps_per_output = static_args.steps_per_output
            n_periods = -(-actual_length // steps_per_output)  # ceil division
            chunk_records = jax.tree.map(lambda x: x[:n_periods], chunk_records)

        return state, chunk_records
    
    def _start_spinner(self, message_fn):
        """Starts a braille spinner in a background thread. Returns (done_event, thread)."""
        done = threading.Event()
        frames = itertools.cycle(['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏'])
        def _animate():
            while not done.is_set():
                print(f'\r{message_fn(next(frames))}', end='', flush=True)
                done.wait(0.1)
        t = threading.Thread(target=_animate, daemon=True)
        t.start()
        return done, t

    def spinup(self, state):
        """
        Spins up the model for the user-prescribed duration
        by repeating forcings beginning on the simulation 
        start date. This minimizes the error from prescribed
        initial conditions and if a long enough period is 
        defined, can initialize firnpack in accumulation areas.
        """
        params = self.params
        n_spinup_years = params.n_spinup_years
        n_spinup_steps = n_spinup_years * 8760

        # avoid a short final chunk, which _run_chunk would pad with zeros
        chunk_size = params.temporal_chunk_hours
        if n_spinup_steps > 0 and (chunk_size > n_spinup_steps or n_spinup_steps % chunk_size != 0):
            chunk_size = n_spinup_steps

        spinup_dates = pd.date_range(self.dates[0], periods=n_spinup_steps, freq='h')

        if n_spinup_years == 0:
            # skip if the user specified no spin-up
            self.spinup_time = 30.0
            return state

        # initialize spinup animation variables
        n_spinup_chunks = n_spinup_steps // chunk_size
        spinup_start = time.time()
        spinup_chunk = [0]  # list so the animation thread can read updates
        if sys.stdout.isatty():
            spinup_done, spinup_thread = self._start_spinner(
                lambda f: f'~ Spinning up {f} ({spinup_chunk[0]}/{n_spinup_chunks} chunks)'
            )
        else:
            print(f'~ Spinning up ({n_spinup_chunks} chunks) ~', flush=True)
            spinup_done, spinup_thread = threading.Event(), None

        # loop through temporal chunks for first year
        for i, start in enumerate(range(0, len(spinup_dates), chunk_size)):
            chunk_dates = spinup_dates[start:start + chunk_size]
            state, _ = self._run_chunk(state,
                                       self.config.static_args, self.config.dynamic_args,
                                       chunk_dates, start, chunk_size, spinup=True)
            spinup_chunk[0] = i + 1

        # print final spinup timer
        spinup_done.set()
        if spinup_thread is not None:
            spinup_thread.join()
        self.spinup_time = time.time() - spinup_start
        msg = f'~ Spun up and compiled in {self.spinup_time:.1f} s ~'
        print(f'\r{msg:<70}')

        # reset annual trackers: spin-up and year 0 are the same calendar year
        state = state._replace(
            annual_min_albedo=jnp.ones_like(state.annual_min_albedo),
            annual_firn_converted=jnp.zeros_like(state.annual_firn_converted),
            last_snow=state.last_snow - n_spinup_steps,
        )

        return state

    def initialize(self):
        """
        Prepares spatial inputs, layer structure, and initial JAX state.
        Results are stored on self so they can be accessed directly for
        optimization or other uses without going through run().
        """
        self.prepare_spatial_inputs()
        self.prepare_initial_state()
        self.initial_state = self.pack_states()

        # glacier dynamics (GLIDE) coupling, if enabled
        if self.params.option_dynamics:
            from pebsi.connectors.pebsi_glide_connector import GlideCoupler
            self.glide_coupler = GlideCoupler(self.terrain, self.params)
            self._glide_t_years = 0.0
            
            # accumulates mass balance + elapsed hours across chunks
            self._glide_pending_mb = np.zeros(self.terrain.N_POINTS)
            self._glide_pending_hours = 0

    def run(self):
        """
        Executes model functions and stores the output data.
        Runs a one-year spin-up before the main loop to allow
        initialization to stabilize; spin-up output is discarded.
        The main() function is chunked so JAX compiles once and
        memory use stays bounded.
        """
        params = self.config.params

        # prepare all the inputs
        self.initialize()
        initial_state = self.initial_state

        # grab static/dynamic args
        static_args = self.config.static_args
        dynamic_args = self.config.dynamic_args

        # initialize climate metadata on params so close_out can read it directly
        if params.use_aws:
            aws_cols = set(pd.read_csv(params.aws_fn, nrows=0).columns)
            all_known = list(params.climate_all_vars) + list(params.climate_optional_vars)
            params.climate_measured_vars = [v for v in all_known if v in aws_cols]
        else:
            params.climate_measured_vars = []

        self.start_print()

        # ========== CHECKPOINT / SPIN-UP ==========
        resume_from = getattr(params, 'resume_from', None)
        if resume_from:
            assert not params.option_dynamics, (
                'Resuming a run (-resume_from) with'
                'option_dynamics=True is not supported'
            )
            state, start_from = self.load_checkpoint(resume_from)
            resume_fp = resume_from
            self.spinup_time = 30.0  # fallback seed for progress bar
        else:
            state = self.spinup(initial_state)
            start_from = 0
            resume_fp = None

        # initialize output after the checkpoint check so resume_fp is known
        model_output = Output(params, self.terrain, resume_fp=resume_fp)

        # copy config to the output directory
        if params.store_data and params.use_config and resume_fp is None:
            # open the config
            with open(params.config_fn, 'r') as f:
                yaml_data = yaml.safe_load(f)

            # copy in any args specified in command line 
            for key in params.cmd_args:
                yaml_data[key] = getattr(params, key)

            # dump new dict into output filepath
            config_dest = os.path.join(model_output.output_fp, defaults.config_fn)
            with open(config_dest, 'w') as f:
                yaml.dump(yaml_data, f, default_flow_style=False)

        # =============== CHUNKING SETUP ===============
        total_steps = len(self.dates)

        if params.output_freq == 'monthly':
            # chunks stay ~temporal_chunk_hours long, snapped to whole months
            month_periods = self.dates.to_period('M').values
            change_idx = np.flatnonzero(np.r_[True, month_periods[1:] != month_periods[:-1]])
            month_bounds = list(zip(change_idx.tolist(), change_idx[1:].tolist() + [total_steps]))

            target = params.temporal_chunk_hours
            groups, current, current_len = [], [], 0
            for s, e in month_bounds:
                if current and current_len + (e - s) > target:
                    groups.append(current)
                    current, current_len = [], 0
                current.append((s, e))
                current_len += e - s
            if current:
                groups.append(current)

            groups = [g for g in groups if g[0][0] >= start_from]
            chunk_starts = [g[0][0] for g in groups]
            chunk_lengths = {g[0][0]: sum(e - s for s, e in g) for g in groups}
            chunk_month_lengths = {g[0][0]: tuple(e - s for s, e in g) for g in groups}
        else:
            chunk_size = params.temporal_chunk_hours
            chunk_starts = list(range(start_from, total_steps, chunk_size))
            chunk_lengths = {s: chunk_size for s in chunk_starts}
            chunk_month_lengths = {}

        n_chunks = len(chunk_starts)

        # single-chunk runs: spinner only (progress bar is meaningless with no prior timing)
        if n_chunks == 1 and params.progress_bar:
            if sys.stdout.isatty():
                sim_done, sim_thread = self._start_spinner(lambda f: f'~ Simulating {f}')
            else:
                print('~ Simulating ~', flush=True)
                sim_done, sim_thread = threading.Event(), None

        # seed duration estimate from spin-up
        prev_duration = self.spinup_time * 3
        progress = ChunkProgress(total_steps, params.progress_bar and n_chunks > 1, prev_duration)

        # ================== MAIN SIMULATION ==================
        chunk_i = 0
        self._chunk_label = (0, n_chunks)  # (current, total) for use in sub-methods
        for start in chunk_starts:
            chunk_i += 1
            self._chunk_label = (chunk_i, n_chunks)
            call_chunk_size = chunk_lengths[start]
            # get dates in this chunk
            chunk_dates = self.dates[start:start + call_chunk_size]
            actual_size = len(chunk_dates)

            # monthly: pass this chunk's per-month segment lengths
            chunk_static_args = static_args
            if params.output_freq == 'monthly':
                chunk_static_args = static_args._replace(month_lengths=chunk_month_lengths[start])

            # start timer / progress bar
            progress.start_chunk(start, actual_size, prev_duration)
            chunk_start = time.time()

            # simulate one chunk
            state, chunk_records = self._run_chunk(state, chunk_static_args, dynamic_args, chunk_dates, start, call_chunk_size)

            # update progress bar
            chunk_end = time.time()
            prev_duration = chunk_end - chunk_start
            progress.finish_chunk(start, actual_size)

            # glacier dynamics (GLIDE) coupling
            if params.option_dynamics:
                chunk_mb = np.array(
                    chunk_records.accumulation + chunk_records.refreeze - chunk_records.melt
                ).sum(axis=0)
                self._glide_pending_mb += chunk_mb
                self._glide_pending_hours += actual_size

                period_hours = params.dynamics_period_years * 8760
                if self._glide_pending_hours >= period_hours:
                    dt_years = self._glide_pending_hours / 8760
                    self.terrain, self.point_attrs, state = self.glide_coupler.couple_step(
                        self.terrain, self.point_attrs, state,
                        self._glide_pending_mb, self._glide_t_years, dt_years
                    )
                    self._glide_t_years += dt_years
                    self._glide_pending_mb = np.zeros(self.terrain.N_POINTS)
                    self._glide_pending_hours = 0

            # store data before checkpointing so the two are always in sync
            if params.store_data:
                if not self.params.progress_bar:
                    print(f'\033[2K\r~ Storing output   [{chunk_i}/{n_chunks}] ~', end='', flush=True)
                # one timestamp (period start) per output row
                if params.output_freq == 'monthly':
                    offsets = np.cumsum((0,) + chunk_month_lengths[start][:-1])
                    record_dates = chunk_dates[offsets]
                else:
                    record_dates = chunk_dates[::chunk_static_args.steps_per_output]
                model_output.store_chunk(chunk_records, record_dates, start)
                self.save_checkpoint(state, start + actual_size, model_output.output_fp)
            del chunk_records

        # ================== CLOSING OUT ==================
        if n_chunks == 1 and params.progress_bar:
            sim_done.set()
            if sim_thread is not None:
                sim_thread.join()
            print('\033[2K', end='\r', flush=True)

        if not self.params.progress_bar:
            print()  # end the overwriting line before the summary

        time_elapsed = time.time() - self.start_time

        progress.close()

        if self.params.store_data:
            model_output.close_out(params, time_elapsed)
        else:
            print('~ Success: data was not saved ~')

        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(f'~ Simulation completed in {time_elapsed:.1f} seconds ~')
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

    if args.testing:
        args.rgi_ids = ['00.00000']

    # initialize and run the model
    model = PEBSI(args)
    model.run()