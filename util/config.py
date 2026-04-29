"""
Configuration for PEBSI

Contains functions which load and map the
configuration YAML file to the args used
by the model.

@author: clairevwilson
"""
import yaml
import types
import util.params as prms

import xarray as xr
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

class Config:
    def __init__(self):
        return

class ConfigError(Exception):
    """Raised when an expected crash
    ends the simulation."""
    pass

def get_config(cmd_args):
    """
    Loads the model configuration in the following order.
    1. Fills in all variables present in util.params (prms).
    2. Overwrites the variables present in config.yaml.
    3. Overwrites the variables present in the command line (cmd_args).
    """
    args = Config()
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
    for key, value in vars(cmd_args).items():
        # overwrite non-Boolean variables that are not None in command line
        if value is not None and not isinstance(value, bool):
            setattr(args, key, value)

        # special case: qm_glac_name can be None (climate.py handles this)
        elif key == 'qm_glac_name':
            setattr(args, key, value)
        
        # if the value is a Boolean, only override if it's True
        elif isinstance(value, bool) and value is True:
            setattr(args, key, value)

    # load grainsize lookup table
    grainsize_fn = args.grainsize_fn.format(s=str(args.initSSA))
    ds = xr.open_dataset(grainsize_fn).load()
    grain_size_dims = (ds.TVals.values, 
                       ds.DTDZVals.values, 
                       ds.DENSVals.values)
    args.interp_tau = RegularGridInterpolator(
        grain_size_dims, ds.taumat.values, method='linear')
    args.interp_kap = RegularGridInterpolator(
        grain_size_dims, ds.kapmat.values, method='linear')
    args.interp_dr0 = RegularGridInterpolator(
        grain_size_dims, ds.dr0mat.values, method='linear')
    
    # store args.wvs as a numpy array
    args.wvs = np.array(args.wvs)

    # print debug statement
    if args.debug and args.use_config:
        print(f'~ Loaded configs from {args.config_fn}')

    return args