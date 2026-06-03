"""
Configuration for PEBSI

Contains functions which load and map the
configuration YAML file to the args used
by the model.

@author: clairevwilson
"""
import yaml, os
import types
import util.params as prms

import xarray as xr
import rioxarray as rxr
import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from collections import namedtuple 
from types import SimpleNamespace
from types import MappingProxyType
import jax

class ConfigError(Exception):
    """Raised when an expected crash
    ends the simulation."""
    pass

class Config():
    def __init__(self, cmd_args):
        """
        Loads the model configuration in the following order.
        1. Fills in all variables present in util.params (prms).
        2. Overwrites the variables present in config.yaml.
        3. Overwrites the variables present in the command line (cmd_args).
        """
        args = SimpleNamespace()
        valid = 'Please check pebsi/params.py for valid variable names.'

        # if config filename was specified, make sure use_config is True
        if cmd_args.config_fn is not None:
            cmd_args.use_config = True
            args.config_fn = cmd_args.config_fn

        # 1: add all prms default attributes to args
        for key in dir(prms):
            # ignore internal python stuff
            key_start = not key.startswith('__')
            # check if we're on config_fn
            config_var = key == 'config_fn'
            # check if config_fn is specified
            no_config = cmd_args.config_fn is None

            if config_var and not no_config:
                continue
            elif key_start:
                val = getattr(prms, key)
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
            df = df.loc[(df['Area'] > 14) & (df['Area'] < 15)]

            # find all glaciers in this region
            all_ids = [f.split('-')[-1] for f in df['RGIId']]
            args.rgi_ids = all_ids

        # configure last items
        self.args = args
        self.configure_lookups()
        self.args.ice_spectrum_fn = self.configure_SNICAR()

        # FINALLY: convert args into a JAX-compatible NamedTuple (immutable)
        self.convert_to_jax_safe(self.args)

        # print debug statement
        if self.args.debug and self.args.use_config:
            print(f'~ Loaded configs from {args.config_fn}')
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
            grain_size_dims, ds.taumat.values, method='linear')
        args.interp_kap = RegularGridInterpolator(
            grain_size_dims, ds.kapmat.values, method='linear')
        args.interp_dr0 = RegularGridInterpolator(
            grain_size_dims, ds.dr0mat.values, method='linear')

        # load ML algorithm for albedo
        # if args.method_snicar == 'emulator':
        #     import joblib
        #     args.SNICAR_emulator = joblib.load(args.emulator_fn)

        # define wind reference height [m]
        args.wind_ref_height = 10 if args.climate_source in ['ERA5-hourly'] else 2

        self.args = args
        return

    def configure_SNICAR(self):
        args = self.args 

        # 1: INPUT FILE
        # get filename of input file
        if args.method_snicar == 'bioSNICAR':
            base_fn = args.biosnicar_input_fn
        elif args.method_snicar == 'SNICARfx':
            base_fn = args.snicarfx_input_fn
        else:
            # nothing to initialize for emulator method
            return

        # open the input and copy it to memory
        with open(base_fn, 'r') as f:
            input = yaml.safe_load(f)

        # store the SNICAR input dict to args
        args.snicar_inputs = input

        # 2: COPY BACKGROUND ICE SPECTRUM FILE
        df_clean_ice = pd.read_csv(args.clean_ice_fn,names=[''])

        # find albedo of the base spectrum from the filename
        albedo_string = args.clean_ice_fn.split('bba')[-1].split('.')[0]
        bba = int(albedo_string) / (10 ** len(albedo_string))

        # scale the new spectrum by the ice albedo
        ice_point_spectrum = df_clean_ice * args.albedo_ice / bba

        # create new name for ice spectrum
        clean_ice_fn = args.clean_ice_fn.split('/')[-1]
        ice_spectrum_fn = args.clean_ice_fn.replace(clean_ice_fn,f'ice_spectrum_{args.task_id}{args.site}.csv')

        # store new spectrum (will be deleted after run completion)
        df_spectrum = pd.DataFrame(ice_point_spectrum)
        df_spectrum.to_csv(ice_spectrum_fn, index=False, header=False)
        return ice_spectrum_fn
    
    def convert_to_jax_safe(self, args):
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
        
        # deep freeze every item inside the dictionary
        frozen_config_dict = {k: freeze_object(v) for k, v in raw_config_dict.items()}
        
        # create new dynamic named tuple
        JaxSafeConfig = namedtuple('JaxSafeConfig', frozen_config_dict.keys())

        # instantiate it with the SimpleNamespace args
        self.args = JaxSafeConfig(**frozen_config_dict)
        return