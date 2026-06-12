"""
Runs PEBSI preprocessing steps separately from a 
full simulation. 

This takes the same command line args / config.yaml
input structure as the main model.
It can be completely skipped IF the working environment
simultaneously handles JAX and CuPY.

On AMD GPUs this can get complicated, so this script
is here to simplify. One environment can be used
to generate the shading masks using CuPY, and then
a separate environment can be used to run the model
using JAX.

"""
import os 
import jax
os.environ["JAX_TRACEBACK_FILTERING"] = "off"
jax.config.update("jax_enable_x64", True)
# Built-in libraries
import argparse
import time
# Internal libraries
import util.defaults as defaults
from util.config import Config
from util.terrain import Terrain

# START TIMER
start_time = time.time()

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
    
    parser.add_argument('-testing', action='store_true',
                        help='test a single function?')

    if parse:
        args = parser.parse_args()
        return args
    else:
        return parser
    
def run_preprocessing():
    # parse command-line args 
    args = get_args()

    # parse args from config file, command line and default params
    config = Config(args)
    args = config.args 

    # load the terrain
    terrain = Terrain(config.args)

    # run DEM functions for shading, elevation, etc.
    terrain.run_dem_functions()

    # validate spatial inputs
    terrain.validate_terrain_data()
    return

if __name__ == '__main__':
    run_preprocessing()