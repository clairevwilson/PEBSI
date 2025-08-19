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
import pebsi.input as prms
from pebsi.climate import Climate
from pebsi.massbalance import massBalance
from shading.shading import Shading

# START TIMER
start_time = time.time()

def get_args(parse=True):
    """
    Defines command line arguments

    Parameters
    ==========
    parse : Bool
        If True, parses the command line (returns args)
        If False, returns the parser
    """    
    parser = argparse.ArgumentParser(description='energy balance model runs')
    
    # GLACIER INFORMATION
    parser.add_argument('-glac_no', action='store', default=prms.glac_no,
                        help='RGI glacier ID')
    parser.add_argument('-site',action='store',default='center',type=str,
                        help='Site name')
    parser.add_argument('-s0','--initial_snow_depth',default=prms.initial_snow_depth,
                        help='Initial snow depth in m',action='store',type=float)
    parser.add_argument('-f0','--initial_firn_depth',default=prms.initial_firn_depth,
                        help='Initial firn depth in m',action='store',type=float)
    
    # MODEL TIME
    parser.add_argument('-start','--startdate', action='store', type=str, 
                        default=prms.startdate,
                        help='pass str like datetime of model run start')
    parser.add_argument('-end','--enddate', action='store', type=str,
                        default=prms.enddate,
                        help='pass str like datetime of model run end')
    parser.add_argument('-dfd','--dates_from_data',action='store_true',
                        help='use dates from input AWS data?')
    
    # USER OPTIONS
    parser.add_argument('-use_AWS', action='store_true',
                        default=prms.use_AWS,help='use AWS or just reanalysis?')
    parser.add_argument('-store_data', action='store_true',
                        default=prms.store_data, help='store the model output?')
    parser.add_argument('-debug', action='store_true',
                        default=prms.debug, help='print debug statements?')
    parser.add_argument('-out',action='store',type=str,default='',
                        help='output file name excluding extension (.nc)')
    
    # ALBEDO SWITCHES
    parser.add_argument('-switch_LAPs',action='store', type=int,
                        default=prms.switch_LAPs, help='')
    parser.add_argument('-switch_melt',action='store', type=int, 
                        default=prms.switch_melt, help='')
    parser.add_argument('-switch_snow',action='store', type=int,
                        default=prms.switch_snow, help='')
    
    # CALIBRATED PARAMETERS
    parser.add_argument('-kp',default=prms.kp,action='store',type=float,
                        help='Multiplicative precipitation factor')
    parser.add_argument('-Boone_c5',default=prms.Boone_c5,action='store',type=float,
                        help='Parameter for Boone densification scheme')
    parser.add_argument('-a_ice',default=prms.albedo_ice,action='store',type=float,
                        help='Bare ice albedo')
    
    # FILEPATHS
    parser.add_argument('-initial_temp_fp',default=prms.initial_temp_fp,type=str,
                        action='store',help='Filepath for initializing temperature')
    parser.add_argument('-initial_density_fp',default=prms.initial_density_fp,type=str,
                        action='store',help='Filepath for initializing density')
    parser.add_argument('-initial_grains_fp',default=prms.initial_grains_fp,type=str,
                        action='store',help='Filepath for initializing grain size')
    parser.add_argument('-initial_LAP_fp',default=prms.initial_LAP_fp,type=str,
                        action='store',help='Filepath for initializing LAPs')
    
    # PARALLELIZATION
    parser.add_argument('-n','--n_simultaneous_processes',default=1,type=int,
                        help='Number of parallel processes to run')
    parser.add_argument('-task_id',default=-1,type=int,
                        help='Task ID if submitted as batch job')
    
    # FOR JUPYTER NOTEBOOKS
    parser.add_argument('-f', '--fff', help='Dummy arg to fool ipython', default='1')

    if parse:
        args = parser.parse_args()
        return args
    else:
        return parser
    
def get_site_table(site_df, args):
    """
    Loads the table for site locations on the
    glacier of interest and stores them in args

    Parameters
    ==========
    site_df : pd.DataFrame
        Table containing the glacier point information
    args : command line arguments
    """
    # get site-specific variables
    site = args.site
    args.lat = site_df.loc[args.site,'lat']
    args.lon = site_df.loc[args.site,'lon']
    args.elev = site_df.loc[site]['elevation']
    args.slope = site_df.loc[site]['slope']
    args.aspect = site_df.loc[site]['aspect']
    args.sky_view = site_df.loc[site]['sky_view']

    # snow and firn depth may be specified in the site_constants table
    if 'snowdepth' in site_df.columns:
        if not np.isnan(site_df.loc[site,'snowdepth']):
            args.initial_snow_depth = site_df.loc[site,'snowdepth']
    if 'firndepth' in site_df.columns:
        if not np.isnan(site_df.loc[site,'firndepth']):
            args.initial_firn_depth = site_df.loc[site,'firndepth']

    # Override site lat/lon/elevation with the AWS site
    if args.use_AWS and prms.use_AWS_site:
        metadata_df = pd.read_csv(prms.AWS_metadata_fn, sep='\t', index_col='glacier')
        args.lat = metadata_df.loc[args.glac_name, 'latitude']
        args.lon = metadata_df.loc[args.glac_name, 'longitude']
        args.elev = metadata_df.loc[args.glac_name, 'elevation']
        if args.debug:
            print(f'~ Using AWS site: ({args.lat}, {args.lon}) at {args.elev} m a.s.l.')

    # *****Special HARD-CODED handling for Gulkana*****
    if args.glac_name == 'gulkana' and args.site != 'center':
        # set scaling albedo
        slope = (0.485 - 0.315)/(site_df.loc['B','elevation'] - site_df.loc['A','elevation'])
        intercept = 0.315
        args.a_ice = intercept + (args.elev - site_df.loc['A','elevation'])*slope
        args.a_ice = min(0.485,args.a_ice)

        # set initial density profile from measurements
        if args.site not in ['AB','ABB','BD']:
            if pd.to_datetime(args.startdate) > pd.to_datetime('2023-12-31'):
                args.initial_density_fp = f'data/by_glacier/gulkana/density/gulkana{args.site}density24.csv'
            else:
                args.initial_density_fp = f'data/by_glacier/gulkana/density/gulkana{args.site}meandensity.csv'
        elif args.site in ['ABB','BD']:
            args.initial_density_fp = 'data/by_glacier/gulkana/density/gulkanaBdensity24.csv'
        elif args.site in ['AB']:
            args.initial_density_fp = 'data/by_glacier/gulkana/density/gulkanaAUdensity24.csv'
    return args

def get_shading(args):
    """
    Runs the shading model for a given lat/lon on the 
    glacier which produces the shading file and two plots 
    which can be inspected in shading/plots. 
    
    If no lat/lon is provided, the model defaults to 
    the RGI CenLat and CenLon.

    Parameters
    ==========
    args : command line arguments
    """
    # shading file does not exist: warn the user
    print(f'! Shading file was not found for {args.glac_name} {args.site}')

    # specify shading model arguments
    args.site_by = 'latlon'
    args.plot = ['result','search']
    args.store = ['result','result_plot','search_plot']

    # check if we can index the lat/lon for this site
    site_fp = prms.site_fp.replace('GLACIER', args.glac_name)
    if os.path.exists(site_fp):
        # open site constants file and check if our site is there
        site_df = pd.read_csv(site_fp,index_col='site')
        assert args.site in site_df.index, f'Add lat/lon for {args.site} in site_constants.csv'
        
        # grab the lat/lon from site_constants
        args.lat = site_df.loc[args.site,'lat']
        args.lon = site_df.loc[args.site,'lon']
    else:
        # no site constants file: use RGI cenlat and cenlon
        RGI_region = args.glac_no.split('.')[0]
        for fn in os.listdir(prms.RGI_fp):
            # open the attributes .csv for the correct region
            if fn[:2] == RGI_region and fn[-3:] == 'csv':
                RGI_df = pd.read_csv(prms.RGI_fp + fn)
                RGI_df.index = [f.split('-')[-1] for f in RGI_df['RGIId']]
        # grab the lat/lon from RGI
        args.lat = RGI_df.loc[args.glac_no,'CenLat']
        args.lon = RGI_df.loc[args.glac_no,'CenLon']
        if args.site != 'center':
            print('~ Using centerpoint lat/lon: changed site name to \"center\"')

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
    
def check_inputs(glac_no, args):
    """
    Checks that the glacier point has all required inputs.
    - Shading file: if not found, executes the shading 
        model which requires a DEM 
    - Loads site_constants (created by shading) and finds
        the lat/lon/elevation/slope/aspect for the point
    - Specifies the model start and end date
    - Names the output filepath
    
    Parameters
    ==========
    glac_no : str
        Individual glacier RGI ID
    args : command line arguments
    """
    # check if the RGI ID is in the metadata file
    all_df = pd.read_csv(prms.metadata_fp,index_col=0,converters={0: str})
    assert glac_no in all_df.index, f'Add {glac_no} to data/glacier_metadata.csv'

    # load the metadata for the glacier
    args.timezone = pd.Timedelta(hours=int(all_df.loc[glac_no,'timezone']))
    args.glac_name = all_df.loc[glac_no,'name']
    
    # load AWS filepath for test glacier or general case
    if args.glac_name == 'test':
        args.AWS_fn = all_df.loc[glac_no,'AWS_fn']
        args.use_AWS = True
        if args.debug:
            print('Test glacier: using sample AWS data')
    else:
        args.AWS_fn = prms.AWS_fp + args.glac_name + '/' + all_df.loc[glac_no,'AWS_fn']

    # specify filepaths to args
    args.shading_fp = prms.shading_fp.replace('GLACIER',args.glac_name).replace('SITE',args.site)
    args.dem_fp = prms.dem_fp.replace('GLACIER', args.glac_name)

    # check if the shading file exists
    if not os.path.exists(args.shading_fp):
        args = get_shading(args)
    
    # load site constants table
    site_fp = prms.site_fp.replace('GLACIER', args.glac_name)
    site_df = pd.read_csv(site_fp,index_col='site')

    # update args from the site table
    args = get_site_table(site_df, args)

    # check if time should be taken from AWS data
    if args.dates_from_data:
        cdf = pd.read_csv(args.AWS_fn,index_col=0)
        cdf.index = pd.to_datetime(cdf.index)

        # take start and end time from the climate dataframe
        startdate = pd.to_datetime(cdf.index[0])
        enddate = pd.to_datetime(cdf.index.to_numpy()[-1])

        # have to offset by 30 minutes for MERRA-2
        if prms.reanalysis == 'MERRA2' and startdate.minute != 30:
            startdate += pd.Timedelta(minutes=30)
            enddate -= pd.Timedelta(minutes=30)

        # add dates to args
        args.startdate = startdate
        args.enddate = enddate

    # create model run name
    if args.out == '':
        model_run_date = str(pd.Timestamp.today()).replace('-','_')[0:10]
        args.out = f'{args.glac_name}{args.site}_{model_run_date}_'
    
    if args.debug:
        print('~ Inputs verified ~')
    return args

def initialize_model(glac_no,args):
    """
    Loads glacier table and climate dataset for one
    glacier to initialize the model inputs.

    Parameters
    ==========
    glac_no : str
        RGI glacier ID
    args : command-line arguments
    
    Returns
    -------
    climate
        Class object from climate.py
    """
    # ===== CHECK GLACIER INPUTS (LAT,LON,ELEV,...) =====
    args = check_inputs(glac_no, args)

    # ===== GET GLACIER CLIMATE =====
    # initialize the climate class
    climate = Climate(args)

    # load in available AWS data, then reanalysis
    if args.use_AWS:
        need_vars = climate.get_AWS(args.AWS_fn)
        if len(need_vars) > 1:
            climate.get_reanalysis(need_vars)
    else:
        climate.get_reanalysis(climate.all_vars)

    # check the climate dataset is ready to go
    climate.check_ds()

    # ===== PRINT MODEL RUN INFO =====
    start = pd.to_datetime(args.startdate)
    end = pd.to_datetime(args.enddate)
    n_months = np.round((end-start)/pd.Timedelta(days=30))
    start_fmtd = start.month_name()+', '+str(start.year)
    print(f'~ Running {glac_no} at {args.elev} m a.s.l. for {n_months} months starting in {start_fmtd} ~')
    return climate, args

def run_model(climate,args,store_attrs=None):
    """
    Executes model functions in parallel or series and
    stores output data.

    Parameters
    ==========
    climate
        Class object from pebsi.climate
    args : command line arguments
    store_attrs : dict
        Dictionary of additional metadata to store 
        in the model output .nc
    """
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
    # get command line arguments
    args = get_args()

    # initialize the model
    climate, args = initialize_model(args.glac_no,args)
    
    # run the model
    run_model(climate,args)