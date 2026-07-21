"""
Configuration for PEBSI

Contains functions which load and map the
configuration YAML file to the args used
by the model.

@author: clairevwilson
"""
# Internal libraries
import os
from collections import namedtuple 
import types
import time
import threading
# External libraries
from tqdm import tqdm
import yaml
import xarray as xr
import numpy as np
import pandas as pd
import xarray as xr
import jax
from jax.scipy.interpolate import RegularGridInterpolator
# Local libraries
import pebsi.defaults as defaults

# Fields must be broken into static and dynamic for JAX. 
# Statics args are needed to initialize the model and
# make choices in the compilation (e.g., options / methods)

static_fields = [
    'max_nlayers', 'albedo_TOD', 'bias_vars', 'n_heat_steps', 
    'store_vars', 'differentiable',
    
    'intensive_vars','extensive_vars', 'all_layer_vars', 'cmd_args',

    'method_turbulent', 'method_stability', 'method_diffuse',
    'method_heateq', 'method_densification', 'method_cooling',
    'method_ground', 'method_conductivity',

    'option_SWpen', 'option_accel_grains', 
    'option_uniform_ice', 'option_uniform_snow',
    'option_flat_plates',

    'constant_snowfall_density','constant_freshgrainsize',
    'constant_drdry','constant_irrwater',
]

# Dynamic args are parameters / constants which are 
# allowed to be passed as scalars or as (N_POINTS, ) arrays.

dynamic_fields = ['kp','wind_factor','precgrad',            
                'dust_factor','lapse_rate',
                'albedo_ice','albedo_firn','albedo_fresh_snow',
                'temp_depth','roughness_aging_rate',
                'roughness_fresh_snow', 'roughness_aged_snow',
                'roughness_firn','roughness_ice',
                'ksp_BC', 'ksp_OC', 'ksp_dust',
                'initial_snow_depth', 'initial_firn_depth']

# External args are those which are only needed in the
# CPU intiialization / output functions but are not used 
# within the model itself.

external_fields = [
    'start_date', 'end_date', 'rgi_ids', 'sites',
    'bias_vars', 'station_elevation',
    'use_config', 'rgi_region', 'use_aws', 'store_data',
    'testing', 'progress_bar', 'debug'
]

# Any other parameters from defaults.py not listed here:
# - strings go to static args
# - non-strings go to dynamic args

class ConfigError(Exception):
    """Raised when an expected crash
    ends the simulation."""
    pass

class Config():
    def __init__(self, cmd_args):
        """
        Loads the model configuration in the following order.
        1. Fills in all variables present in util.defaults.
        2. Overwrites the variables present in config.yaml.
        3. Overwrites the variables present in the command line (cmd_args).
        """
        args = types.SimpleNamespace()
        valid = 'Please check pebsi/defaults.py for valid variable names.'

        # if config filename was specified, make sure use_config is True
        if cmd_args.config_fn is not None:
            cmd_args.use_config = True
            args.config_fn = cmd_args.config_fn

        # 1: add all default attributes to args
        for key in dir(defaults):
            # ignore internal python stuff
            key_start = not key.startswith('__')
            # check if we're on config_fn
            config_var = key == 'config_fn'
            # check if config_fn is specified
            no_config = cmd_args.config_fn is None

            if config_var and not no_config:
                continue
            elif key_start:
                val = getattr(defaults, key)
                if isinstance(val, types.ModuleType):
                    continue
                else:
                    setattr(args, key, val)
                    
        # 2: fill out variables from yaml file, if specified
        yaml_fn = args.config_fn
        if cmd_args.use_config:
            # open yaml
            with open(yaml_fn, 'r') as f:
                yaml_data = yaml.safe_load(f)

            # loop through directories and subdirectories in .yaml
            for key, value in yaml_data.items():
                # if nested, flatten
                if isinstance(value, dict):
                    for sub_key, sub_val in value.items():
                        if not hasattr(args, sub_key):
                            raise ConfigError(f'Unknown config key: {sub_key} found in {key}\n{valid}')
                        setattr(args, sub_key, sub_val)
                else:
                    if not hasattr(args, key):
                        raise ConfigError(f'Unknown config key: {key}\n{valid}')
                    setattr(args, key, value)

        # 3: overwrite anything specified in the command line
        args.cmd_args = []
        for key, value in vars(cmd_args).items():
            # overwrite variables that are not None in command line
            if value is not None:
                if isinstance(value, bool):
                    # if the value is a Boolean, only override if it's True
                    if value:
                        setattr(args, key, value)
                        args.cmd_args.append(key)
                else:
                    # strings, numbers, etc. 
                    setattr(args, key, value) 
                    args.cmd_args.append(key)     

        # make sure rgi_region agrees with rgi_ids 
        if args.rgi_ids is not None:
            # rgi_id was specified: store its region to rgi_region
            all_regions = np.unique([f[:2] for f in args.rgi_ids])
            assert len(all_regions) < 2, 'PEBSI is only set up to run a single region'  
            args.rgi_region = int(all_regions[0])

        else:
            # no rgi_ids specified: open the RGI data for this region
            region = str(args.rgi_region).zfill(2)
            all_regions = os.listdir(args.rgi_fp)
            csv_path = [f for f in all_regions if f.startswith(region) and f.endswith('csv')][0]
            df = pd.read_csv(args.rgi_fp + csv_path)

            min_area = args.min_area
            df = df.loc[df['Area'] > min_area]
            n = len(df.index)
            print(f'No RGI IDs were specified: running {n} glaciers over {min_area} km 2')

            # find all glaciers in this region
            all_ids = [f.split('-')[-1] for f in df['RGIId']]
            args.rgi_ids = all_ids

        # test glacier (region 00) always uses the sample AWS file
        if args.rgi_region == 0:
            args.use_aws = True
            args.aws_elev = 1232

        # set method_distribute to points if specified sites
        if args.sites is not None:
            args.method_distribute = 'sites'

        # configure last items
        self.args = args
        self.configure_lookups()
        self.args.start_year = pd.to_datetime(self.args.start_date).year

        # validate
        self.validate_config()

        # FINALLY: convert args into a JAX-compatible NamedTuple (immutable)
        self.convert_to_jax_safe(self.args)

        # print debug statement
        if self.params.debug and self.params.use_config:
            print(f'~ Loaded configs from {self.params.config_fn}')
        return
    
    def configure_lookups(self):
        args = self.args 

        # load grainsize lookup table
        grainsize_fn = args.grainsize_fn.format(s=str(args.initSSA))
        ds = xr.open_dataset(grainsize_fn).load()

        # get dimensions
        grain_size_dims = (ds.TVals.values, 
                        ds.DTDZVals.values, 
                        ds.DENSVals.values)
        
        # create interpolation functions for each lookup variable
        args.interp_tau = RegularGridInterpolator(
            grain_size_dims, ds.taumat.values, method='linear', fill_value=0.0)
        args.interp_kap = RegularGridInterpolator(
            grain_size_dims, ds.kapmat.values, method='linear', fill_value=1.0)
        args.interp_dr0 = RegularGridInterpolator(
            grain_size_dims, ds.dr0mat.values, method='linear', fill_value=0.0)

        self.args = args
        return
    
    def validate_config(self):
        """
        Checks that all configurations are valid.
        """
        # MODEL OPTIONS 
        # make sure temporal chunks is a fairly even multiplier of 1 year
        temporal_chunks = getattr(self.args, 'temporal_chunks')
        threshold_hours = 10 * 24
        hours_in_year = 365 * 24
        smaller, larger = sorted([temporal_chunks, hours_in_year])
        remainder = larger % smaller
        to_next = smaller - remainder if remainder != 0 else 0
        assert min(remainder, to_next) <= threshold_hours, \
            f'Temporal chunks should be an ~ even multiplier of 8760'

        # PHYSICAL CONSTANTS
        # parameters that must be positive
        must_be_positive = ['kp','wind_factor','dust_factor',
                            'initial_snow_depth','roughness_aging_rate']
        for var in must_be_positive:
            var_data = np.array(getattr(self.args, var))
            assert np.all(var_data > 0), f'{var} must be positive'
        
        must_be_negative = ['lapse_rate']
        for var in must_be_negative:
            var_data = np.array(getattr(self.args, var))
            assert np.all(var_data < 0), f'{var} must be negative'
        
        must_be_0_or_positive = ['initial_firn_depth', 'precgrad']
        for var in must_be_0_or_positive:
            var_data = np.array(getattr(self.args, var))
            assert np.all(var_data >= 0), f'{var} must be 0 or positive'

        # make sure albedo terms are between 0 and 1
        must_be_0_1 = ['albedo_ice','albedo_firn','albedo_fresh_snow']
        for var in must_be_0_1:
            var_data = np.array(getattr(self.args, var))
            assert np.all((0 < var_data) & (var_data < 1))

        if self.args.debug and len(self.args.bias_vars) > 0:
            print('~ Applying quantile mapping for:',self.args.bias_vars)

        return
    
    def convert_to_jax_safe(self, args):
        # 1. Convert static arguments to a frozen NamedTuple
        # sort args as a dictionary
        raw_config_dict = vars(args)

        # define custom class for dictionaries so JAX sees them as immutable
        @jax.tree_util.register_static
        class FrozenDict(dict):
            """A read-only, hashable dictionary that JAX treats as a static constant."""
            def __hash__(self):
                # Hash the sorted items so that identical dictionaries share the same hash
                return hash(frozenset(self.items()))
            def __setitem__(self, key, value):
                raise TypeError("FrozenDict objects are immutable")
            def __delitem__(self, key):
                raise TypeError("FrozenDict objects are immutable")

        def freeze_object(obj):
            """Recursively converts lists to tuples and dicts to FrozenDicts."""
            if isinstance(obj, list):
                return tuple(freeze_object(item) for item in obj)
            elif isinstance(obj, np.ndarray):
                return tuple(freeze_object(item) for item in obj.tolist())
            elif isinstance(obj, dict):
                # Recursively freeze nested elements, then pack them into our Registered FrozenDict
                frozen_inner = {k: freeze_object(v) for k, v in obj.items()}
                return FrozenDict(frozen_inner)
            else:
                return obj
            
        def to_array_if_numeric(v, k):
            if k in dynamic_fields:
                if isinstance(v, (int, float)):
                    return np.atleast_1d(np.array(v))
                elif isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
                   return np.atleast_1d(np.array(v))
            return v  # leave anything not in dynamic_fields list as-is
        
        # deep freeze every item inside the dictionary
        static_dict = {
            k: freeze_object(v)
            for k, v in raw_config_dict.items()
            if (k in static_fields or isinstance(v, str)) and \
                (k not in external_fields)
        }
        
        # create new namedtuple for static arguments
        StaticArgs = namedtuple('StaticArgs', static_dict.keys())
        self.static_args = StaticArgs(**static_dict)

        # 2. Convert dynamic arguments into arrays
        dynamic_dict = {
            k: to_array_if_numeric(v, k)
            for k, v in raw_config_dict.items()
            if (k not in static_fields and not isinstance(v, str)) and \
                (k not in external_fields)
        }

        DynamicArgs = namedtuple('DynamicArgs', dynamic_dict.keys())
        self.dynamic_args = DynamicArgs(**dynamic_dict)

        # 3. Stack all parameters including external ones into a namespace
        all_params = {
            k: to_array_if_numeric(v, k)
            for k, v in raw_config_dict.items()
        }
        
        self.params = types.SimpleNamespace(**all_params)
        self.params.static_args = self.static_args
        self.params.dynamic_args = self.dynamic_args
        return
    
class ChunkProgress:
    """
    Interpolating tqdm progress bar for long chunks.

    Since JAX blocks Python for the entire chunk, a background thread
    linearly extrapolates progress using the previous chunk's wall-clock
    duration, snapping to the true value when each chunk completes.
    """

    def __init__(self, total, enabled, seed_duration):
        self._enabled = enabled
        self._done = threading.Event()
        self._info = {'t0': time.time(), 'step0': 0, 'size': 1, 'duration': seed_duration}
        self.pbar = tqdm(total=total, desc='~ Simulating', unit='step',
                         miniters=0, mininterval=0, disable=not enabled)
        if enabled:
            threading.Thread(target=self._refresh, daemon=True).start()

    def _refresh(self):
        while not self._done.is_set():
            frac = min((time.time() - self._info['t0']) / self._info['duration'], 0.99)
            self.pbar.n = self._info['step0'] + int(frac * self._info['size'])
            self.pbar.refresh()
            self._done.wait(0.1)

    def start_chunk(self, step0, size, duration):
        self._info.update(t0=time.time(), step0=step0, size=size, duration=duration)

    def finish_chunk(self, step0, actual_size):
        self.pbar.n = step0 + actual_size
        self.pbar.refresh()

    def close(self):
        self._done.set()
        self.pbar.close()