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
# External libraries
import numpy as np
import xarray as xr
import pandas as pd
# Internal libraries
import util.params as prms
import util.config as config
from pebsi.climate import Climate
from pebsi.massbalance import massBalance
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
    
    # SITE INFORMATION
    parser.add_argument('-id','--rgi_id', type=str, default=None,
                        help='RGI glacier ID')
    parser.add_argument('-site', type=str, default=None,
                        help='site name')
    
    # MODEL TIME
    parser.add_argument('-start','--start_date', type=str, default=None,
                        help='pass str like datetime of model run start')
    parser.add_argument('-end','--end_date', default=None,
                        help='pass str like datetime of model run end')
    parser.add_argument('-dfd','--dates_from_data',action='store_true',
                        help='use dates from input AWS data?')
    
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
    parser.add_argument('-cds','--input_cds',action='store_true',
                        help='use existing cds?')
    parser.add_argument('-qm_glac_name',type=str,default=None,
                        help='glacier for quantile mapping climate data')
    
    # UTILITY
    parser.add_argument('-task_id',default=-1,type=int,
                        help='unique task ID for this job')
    parser.add_argument('-f', '--fff', help='Dummy arg to fool Jupyter', default='1')

    if parse:
        args = parser.parse_args()
        return args
    else:
        return parser
    
def get_site_table(site_df, rgi_df, args):
    """
    Loads the table for site locations on the
    glacier of interest and stores them in args

    Parameters
    ==========
    site_df : pd.DataFrame
        Table containing the glacier point information
    rgi_df : pd.DataFrame
        Table from RGI containing glacier metadata
    args : config class
    """
    site = args.site 

    # core variables
    core_vars = ['lat','lon','elevation','slope','aspect','sky_view']
    for var in core_vars:
        current_val = getattr(args, var)
        default_val = getattr(prms, var)

        # if current and default matches, user didn't specify
        if current_val == default_val:
            # pull from site_table
            assert var in site_df.columns, f'Specify {var} in config file or command line'
            setattr(args, var, site_df.loc[site, var])

    # optional variables
    # SNOW DEPTH
    if args.initial_snow_depth == prms.initial_snow_depth:
        if 'snowdepth' in site_df.columns and ~np.isnan(site_df.loc[site,'snowdepth']):
            args.initial_snow_depth = site_df.loc[site,'snowdepth']

    # FIRN DEPTH
    if args.initial_firn_depth == prms.initial_firn_depth:
        if 'firndepth' in site_df.columns and ~np.isnan(site_df.loc[site,'firndepth']):
            args.initial_firn_depth = site_df.loc[site,'firndepth']
        # firn depth is not in the table: estimate if the site should have firn
        elif len(rgi_df) == 0 or args.elevation <= rgi_df.loc[args.rgi_id, 'Zmed']:
            # below median glacier elevation: no firn
            args.initial_firn_depth = 0
        else:
            # above median glacier elevation: initialize with firn
            args.initial_firn_depth = args.initial_firn_depth

    # ICE ALBEDO
    if args.albedo_ice == prms.albedo_ice:
        if 'a_ice' in site_df.columns and not np.isnan(site_df.loc[site,'a_ice']):
            args.albedo_ice = site_df.loc[site,'a_ice']

    # PRECIPITATION FACTOR
    if args.kp == prms.kp:
        if 'kp' in site_df.columns and not np.isnan(site_df.loc[site,'kp']):
            args.kp = site_df.loc[site,'kp']

    # get fns for the initialization data
    for prop in ['temp', 'density', 'grains', 'LAP']:
        # check if args contains default value for this initialization type
        attr_name = f'initial_{prop}_fn'
        current_fn = getattr(args, attr_name)
        default_fn = getattr(prms, attr_name)
        
        # if current and default matches, user didn't specify
        if current_fn == default_fn:
            # check if we have the data available for this site
            fn_prop = f'{prop}/{args.glac_name}{args.site}{prop}.csv'
            if os.path.exists(os.path.join(args.glac_fp, fn_prop)):
                setattr(args, attr_name, os.path.join(args.glac_fp, fn_prop))
    return args

def get_shading(args, coords_match):
    """
    Runs the shading model for a given lat/lon on the 
    glacier which produces the shading file and two plots 
    which can be inspected in shading/plots. 

    Parameters
    ==========
    args : config class
    """
    if coords_match:
        # shading file does not exist: warn the user
        print(f'! Shading file was not found for {args.glac_name} {args.site}')
    else:
        print(f'! Shading file coordinate mismatch')

    # specify shading model arguments
    args.site_by = 'latlon'
    args.plot = ['result','search']
    args.store = ['result','result_plot','search_plot']

    # run the shading model
    print(f'~ Running shading model at [{args.lat:.5f}, {args.lon:.5f}] ...')
    start_shading = time.time()
    shading_model = Shading(args)
    shading_model.main()

    # store the data and print the time to run the shading model
    shading_model.store_site_info()
    shading_elapsed_time = time.time() - start_shading
    print(f'~ Calculated shading for {args.glac_name} {args.site} in {shading_elapsed_time:.1f} seconds ~')
    return args
    
def check_inputs(args):
    """
    Checks that the glacier point has all required inputs.
    - Name and timezone (must either be in the metadata file
        or specified in config/args)
    - Lat, lon and elevation (if not found in the site file
        or in config/args, model takes the RGI centerpoint)
    - Shading file (if not found, the shading model is run)
    - Start and end time of the simulation
    
    Parameters
    ==========
    args : config class
    """
    # get rgi_id from args
    rgi_id = args.rgi_id 

    # open the RGI dataframe
    rgi_region = args.rgi_id.split('.')[0]
    if rgi_region != '00':
        for fn in os.listdir(args.rgi_fp):
            # open the attributes .csv for the correct region
            if fn[:2] == rgi_region and fn[-3:] == 'csv':
                # open the RGI attributes dataframe
                rgi_df = pd.read_csv(args.rgi_fp + fn)
                rgi_df.index = [f.split('-')[-1] for f in rgi_df['RGIId']]

                # get filepath where .shp would be stored
                raw_fp = os.path.join(args.rgi_fp,'../',fn.replace('.csv',''))
                reg_fp = os.path.normpath(raw_fp)

                # tell model if shapefile exists for plotting shade
                args.shapefile_exists = os.path.exists(reg_fp)    
                if not args.shapefile_exists:
                    print('! Warning: shapefile not found')
                    print(f'  Recommended to add RGI O1 shapefile to: {reg_fp}')  
    else:
        rgi_df = pd.DataFrame([])

    # BASIC METADATA
    all_df = pd.read_csv(args.metadata_fn,index_col=0,converters={0: str})
    if rgi_id not in all_df.index:
        # this RGI ID was not found in the metadata file
        if args.glac_name is None:
            args.glac_name = args.rgi_id
            if args.debug:
                print('! glac_name not specified: using RGI ID')
        if args.timezone is None:
            lon_est = rgi_df.loc[args.rgi_id,'CenLon']
            args.timezone = round(lon_est / 15)
            if args.debug:
                print('! timezone not specified: estimating from RGI CenLon')

        # add rgi_id to metadata file
        all_df.loc[rgi_id] = None
        all_df.loc[rgi_id, 'name'] = args.glac_name 
        all_df.loc[rgi_id, 'timezone'] = args.timezone

        # store the csv
        all_df.to_csv(args.metadata_fn) 
        
    # load the metadata for the glacier
    args.timezone = pd.Timedelta(hours=int(all_df.loc[rgi_id,'timezone']))
    args.glac_name = all_df.loc[rgi_id, 'name']
    
    # AWS FILENAME
    if args.glac_name == 'test' or (args.use_aws and args.aws_fn is None):
        # if aws_fn was not supplied, try to read it from all_df
        if rgi_id in all_df.index and args.glac_name != 'test':
            raise config.ConfigError('Specify aws_fn in config file')

        args.aws_fn = all_df.loc[rgi_id,'AWS_fn']
        args.use_aws = True
        if args.debug and args.glac_name == 'test':
            print('~ Test glacier: using sample AWS data')

    # specify other filepaths to args
    args.shading_fn = args.shading_fn.format(g=args.glac_name, s=args.site)
    args.dem_fn = args.dem_fn.format(g=args.glac_name)
    args.glac_fp = args.glac_fp.format(g=args.glac_name)
    args.site_fn = args.glac_fp + args.site_fn

    # open the site dataframe
    site_df = None
    if os.path.exists(args.site_fn):
        site_df = pd.read_csv(args.site_fn,index_col='site')

    # check if we have lat, lon, and elevation information from config
    if None in [args.lat, args.lon]:
        # check if site constants table exists
        if site_df is not None:
            # check if the site is already in site_df
            assert args.site in site_df.index, f'Specify lat/lon for {args.site} in config file'

            # fill missing pieces to args from the site_df
            if args.lat is None: args.lat = site_df.loc[args.site, 'lat']
            if args.lon is None: args.lon = site_df.loc[args.site, 'lon']
            if args.elevation is None: args.elevation = site_df.loc[args.site, 'elevation']

        else:
            # does not exist: override site with RGI cenlat and cenlon
            args.lat = rgi_df.loc[args.rgi_id,'CenLat']
            args.lon = rgi_df.loc[args.rgi_id,'CenLon']
            args.elevation = rgi_df.loc[args.rgi_id,'Zmed']
            if args.site != 'center':
                args.site = 'center'
                print('~ Using centerpoint lat/lon: changed site name to \"center\"')

    # check if we have a coordinate mismatch in the existing shading file
    coords_match = True
    if site_df is not None and args.site in site_df.index:
        db_lat = site_df.loc[args.site, 'lat']
        db_lon = site_df.loc[args.site, 'lon']
        if not (np.isclose(args.lat, db_lat) and np.isclose(args.lon, db_lon)):
            # site lat/lon was changed: need to rerun shading
            coords_match = False

    # check if shading model should be run
    if not os.path.exists(args.shading_fn) or not coords_match:
        # run shading model
        args = get_shading(args, coords_match)

    # reload site table (updated in get_shading)
    site_df = pd.read_csv(args.site_fn, index_col='site')

    # update args from the site table
    args = get_site_table(site_df, rgi_df, args)

    # check if time should be taken from AWS data
    if args.dates_from_data and args.use_aws:
        assert os.path.exists(args.aws_fn), f'AWS data not found ({args.aws_fn})'
        cdf = pd.read_csv(args.aws_fn,index_col=0)
        cdf.index = pd.to_datetime(cdf.index)

        # take start and end time from the climate dataframe
        startdate = pd.to_datetime(cdf.index[0])
        enddate = pd.to_datetime(cdf.index.to_numpy()[-1])

        # add dates to args
        args.start_date = startdate
        args.end_date = enddate
    
    if args.debug:
        print('~ Inputs verified ~')
    return args

def initialize_model(args):
    """
    Loads glacier table and climate dataset for one
    glacier to initialize the model inputs.

    Parameters
    ==========
    args : command-line arguments
    
    Returns
    -------
    climate
        Class object from climate.py
    """
    # ===== LOAD CONFIGURATION =====
    args = config.get_config(args)

    # ===== CHECK GLACIER INPUTS (LAT,LON,ELEV,...) =====
    args = check_inputs(args)

    # ===== GET GLACIER CLIMATE =====
    # initialize the climate class
    climate = Climate(args)

    # check if already loaded cds
    if not climate.loaded_climate:
        # load in available AWS data, then reanalysis
        if args.use_aws:
            need_vars = climate.get_aws(args.aws_fn)
            if len(need_vars) > 1:
                climate.get_reanalysis(need_vars)
        else:
            climate.get_reanalysis(climate.all_vars)

    # check the climate dataset is ready to go
    climate.check_ds()

    # adjust elevation-dependent variables
    climate.adjust_to_elevation()

    # ===== PRINT MODEL RUN INFO =====
    start = pd.to_datetime(args.start_date)
    end = pd.to_datetime(args.end_date)
    n_months = np.round((end-start)/pd.Timedelta(days=30))
    start_fmtd = start.month_name()+', '+str(start.year)
    print(f'~ Running {args.rgi_id} at {args.elevation} m a.s.l. for {n_months} months starting in {start_fmtd} ~')
    return climate, args

def get_output_name(args, climate):
    """
    Finds a unique filename to store the output.
    If store_climate is specified in input.py,
    or if the args.cds = True and the cds does
    not already exist for this simulation, stores
    the climate dataset.

    Parameters
    ==========
    args : command-line arguments
    climate
        Class object from climate.py
    """
    # get output name and store the climate data
    if args.output_fn is None:
        model_run_date = str(pd.Timestamp.today()).replace('-','_')[0:10]
        args.output_fn = f'{args.glac_name}{args.site}_{model_run_date}_'
    # make file name unique by adding an indexer
    i = 0
    while os.path.exists(args.output_fp + args.output_fn + f'{i}.nc'):
        i += 1
    args.output_fn += str(i) + '.nc'

    # store climate if specified
    if climate.store_cds:
        climate.store()
    return args

def run_model(climate,args,store_attrs=None):
    """
    Executes model functions in parallel or series and
    stores output data.

    Parameters
    ==========
    climate
        Class object from pebsi.climate
    args : config class
    store_attrs : dict
        Dictionary of additional metadata to store 
        in the model output .nc
    """
    # get a unique filename to store the output
    args = get_output_name(args,climate)

    # ===== RUN ENERGY BALANCE =====
    model = massBalance(args,climate)
    model.main()
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
        model.output.add_basic_attrs(args,time_elapsed,climate)
        model.output.add_attrs(store_attrs)
        out = model.output.get_output()
    else:
        print('~ Success: data was not saved ~')
        out = None
    
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
    climate, args = initialize_model(args)
    
    # run the model
    run_model(climate,args)