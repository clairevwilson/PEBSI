import xarray as xr 
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import copy
from mapping_fxns import *

##########################
###### START INPUTS ######
##########################

# ========= WEATHER STATION INFO =========
# Station location
glacier = 'kahiltna'
for glacier in ['kahiltna','gulkana','wolverine']:
    if glacier == 'kahiltna':
        # KAHILTNA
        glacier = 'kahiltna'               # Glacier name consistent with glacier_metadata file in PEBSI/data
        lat = 63.07	                       # Station latitude [decimal degrees]
        lon = -151.17                      # Station ongitude [decimal degrees]
        elev_AWS = 3053                    # Station elevation [m a.s.l.] USGS: 2377, NPS: 3053
        timezone = 0                       # Station time zone compared to UTC [hrs] (if the AWS time is in UTC, this should be 0)
        time_col = 0

        # AWS file information
        fp_AWS = '../climate_data/AWS/'       # Filepath to processed AWS data
        # fn_AWS = fp_AWS + 'Raw/kahiltna/combined.csv'          # Filename of AWS data to use in mapping
        fn_AWS = fp_AWS + 'Raw/kahiltna/NPS/preprocessed_2018.csv'          # Filename of AWS data to use in mapping

    else:
        # BENCHMARK
        benchmark_dict = {'gulkana':{'lat':63.285514, 'lon':-145.410, 'elev':1725, 'time_col':'UTC_time', 'timezone':0},
                        'wolverine':{'lat':60.381923, 'lon':-148.939662 , 'elev':990, 'time_col':'local_time', 'timezone':-8}}
        lat = benchmark_dict[glacier]['lat']                    # Station latitude [decimal degrees]     WOLV: 60.381923     GULKANA:63.285514
        lon = benchmark_dict[glacier]['lon']                     # Station ongitude [decimal degrees]     WOLV: -148.939662   GULKANA: -145.410
        elev_AWS = benchmark_dict[glacier]['elev']                    # Station elevation [m a.s.l.]            WOLV: 990, 1420     GULKANA: 1725
        timezone = benchmark_dict[glacier]['timezone']                       # Station time zone compared to UTC [hrs] (if the AWS time is in UTC, this should be 0)
        time_col = benchmark_dict[glacier]['time_col']

        # AWS file information
        fp_AWS = '../climate_data/AWS/'       # Filepath to processed AWS data
        if glacier == 'gulkana':
            fn_AWS = fp_AWS + f'Raw/Benchmark/{glacier}/{elev_AWS}/LVL2/{glacier}{elev_AWS}_hourly_LVL2_ALL.csv'          # Filename of AWS data to use in mapping
        else:
            fn_AWS = fp_AWS + f'Raw/Benchmark/{glacier}/LVL2/{glacier}{elev_AWS}_hourly_LVL2.csv'

    # ========= MERRA-2 INFO =========
    start_MERRA2_data = pd.to_datetime('1980-01-01 00:30')  # First timestamp in MERRA-2 data
    end_MERRA2_data = pd.to_datetime('2025-06-30 00:30')    # Last timestamp in MERRA-2 data
    MERRA2_filetag = glacier+'_alltime.nc'                  # False to use lat/lon indexed files, otherwise string to follow 'MERRA2_VAR_'

    # ========= VARIABLE =========
    # Define name of variable
    var = 'temp'                          # Name of var as referenced in PEBSI (see var_dict keys below if unsure)
    if glacier == 'kahiltna':
        var_AWS = 'AirTemp_C_Avg'                  # Name of var in the AWS data
    else:
        var_AWS = 'site_temp_USGS'

    # Define unit conversion for units of MERRA-2 and AWS data
    def MERRA2_unit_conversion(data):
        if var == 'temp':
            return data - 273.15
        elif var == 'SWin':
            return data * 3600
        else:
            return data
    def AWS_unit_conversion(data):
        if var == 'SWin':
            return data * 3600
        else:
            return data

    # ======== VISUALIZATION ========
    get_scatter_plot = True         # *** Plots 1:1 AWS vs. MERRA-2 data
    get_quantile_plot = True        # *** Plots distribution of AWS and MERRA-2 data before and after correction

    ##########################
    ####### END INPUTS #######
    ##########################

    # ========= FILEPATHS =========
    fp_base = os.getcwd() + '/../../'    # Base filepath to direct files to
    fn_AWS = fp_base + fn_AWS
    assert os.path.exists(fn_AWS), f'AWS dataset not found at {fn_AWS}'
    # MERRA-2
    fp_MERRA = fp_base + '../climate_data/MERRA2/'             # Filepath to MERRA-2 data
    fn_MERRA = fp_MERRA + 'VAR/MERRA2_VAR_LAT_LON.nc'       # Formattable file name for MERRA-2 variable data
    # OUTPUT
    fn_store_quantiles = fp_base + f'data/bias_adjustment/quantile_mapping_{glacier}_VAR.csv'     # Filename to store quantiles
    fn_store_fig = fp_base + f'../Output/{glacier}_{var}_quantile_mapping.png'                                          # Filename to store figures

    # Get MERRA-2 elevation (gepotential)
    ds_elev = xr.open_dataarray(fp_MERRA + 'MERRA2constants.nc4')
    elev_MERRA2 = ds_elev.sel(lat=lat,lon=lon,method='nearest').values[0] / 9.81
    print(f'MERRA2 cell lies at {elev_MERRA2} m a.s.l.')

    # Update MERRA-2 filepath
    flat = str(int(np.floor(lat/10)*10))        # Latitude rounded to 10 degrees to find the right file
    flon = str(int(np.floor(lon/10)*10))        # Longitude rounded to 10 degrees to find the right file
    if not MERRA2_filetag:
        fn_MERRA = fn_MERRA.replace('LAT', str(flat)).replace('LON', str(flon))
    else:
        fn_MERRA = fn_MERRA.replace('LAT_LON.nc', MERRA2_filetag)

    # ======== LOAD DATA ========
    # Dictionary for each variable that may be mapped to data
    var_dict = {'SWin':{'MERRA2_var':'SWGDN', 'label':'Incoming shortwave (W m-2)','lims':[0,1400*3600]},
                'LWin':{'MERRA2_var':'LWGAB', 'label':'Incoming shortwave (W m-2)','lims':[0,500*3600]},
                'temp':{'MERRA2_var':'T2M', 'label':'Air temperature ($^{\circ}$C)','lims':[-60,50]},
                'wind':{'MERRA2_var':'U2M', 'label':'Wind speed (m s-1)','lims':[0,70]},
                'rh':{'MERRA2_var':'RH2M', 'label':'Relative humidity (%)','lims':[0,100]},
                'sp':{'MERRA2_var':'PS', 'label':'Surface pressure (Pa)','lims':[6e4,1.2e4]}}

    # Select variable name and label from var_dict
    var_MERRA2 = var_dict[var]['MERRA2_var']        # Name of var in the MERRA-2 data
    var_label = var_dict[var]['label']              # Label for plotting
    lims = var_dict[var]['lims']

    # Open datasets
    data_MERRA2 = xr.open_dataarray(fn_MERRA.replace('VAR',var_MERRA2))
    data_AWS = pd.read_csv(fn_AWS, index_col=time_col)[var_AWS]

    # Wind is treated differently since it is broken into east and west components by MERRA-2
    either_wind = ['U2M','V2M']
    if var_MERRA2 in either_wind:
        either_wind.remove(var_MERRA2)
        other_variable = either_wind[0]
        data_MERRA2_other = xr.open_dataarray(fn_MERRA.replace('VAR',other_variable))
        data_MERRA2.values = np.sqrt(data_MERRA2.values**2 + data_MERRA2_other.values**2)

    # Update units to match the AWS
    data_MERRA2 = MERRA2_unit_conversion(data_MERRA2)
    data_AWS = AWS_unit_conversion(data_AWS)

    # Clip MERRA-2 dataset to the right lat/lon
    if not MERRA2_filetag:
        data_MERRA2 = data_MERRA2.sel(lat=lat,lon=lon, method='nearest')
    else:
        data_MERRA2 = data_MERRA2

    # Create copy of AWS data
    data_AWS_copy = copy.deepcopy(data_AWS)
    data_MERRA2_copy = copy.deepcopy(data_MERRA2)

    for lapse_rate in [0]:
        # ======== ELEVATION DEPENDENCE ========
        if var == 'temp':
            # lapse_rate = -6.5           # Lapse rate [K km-1]
            data_AWS = data_AWS_copy + (elev_MERRA2 - elev_AWS) * lapse_rate / 1000

        # ======== CHECK VALUES ========
        # Check for unreasonable values
        upper_lim = lims[1]
        lower_lim = lims[0]
        data_AWS = data_AWS[(data_AWS.values >= lower_lim) & (data_AWS.values <= upper_lim)]

        # ======== TIME ========
        # Define all_MERRA2 variable
        all_MERRA2 = data_MERRA2_copy.copy(deep=True).values

        # Make sure AWS data is using datetime index
        data_AWS.index = pd.to_datetime(data_AWS.index)

        # Convert AWS time to UTC
        data_AWS.index -= pd.Timedelta(hours=timezone)

        # Find dates where AWS recorded temperature
        dates_data = pd.to_datetime(data_AWS.index[~np.isnan(data_AWS)])

        # Filter out dates that are outside of MERRA-2 range
        dates_data = dates_data[dates_data >= start_MERRA2_data]
        dates_data = dates_data[dates_data <= end_MERRA2_data]

        # Make sure there are not repeated values
        dates_data = np.unique(dates_data)

        # Select dates where AWS has data
        data_MERRA2_copy['time'] = pd.to_datetime(data_MERRA2_copy['time'].values)
        data_MERRA2 = data_MERRA2_copy.interp(time=pd.to_datetime(dates_data).tz_localize(None))
        data_AWS = data_AWS.loc[dates_data]

        # Separate data into train and test
        X_train = data_AWS.values # [data_AWS.values > 0]
        y_train = data_MERRA2.values # [data_AWS.values > 0]

        # Update filepaths with lapse rate
        fn = fn_store_quantiles.replace('VAR',var).replace('.csv',f'_{lapse_rate}.csv')
        fn_fig = fn_store_fig.replace('.png',f'_{lapse_rate}.png')

        # Get quantile mapping and store it
        sorted, mapping = quantile_mapping(X_train, y_train, fn)
        print('Mean shift:', np.mean(np.interp(y_train, sorted, mapping) - y_train))

        # Plot scatter plot
        if get_scatter_plot:
            fig, axes = plot_scatter(X_train, y_train, lims, fn, plot_kde=False)
            fig.supylabel(var_label,x=-0.03)
            plt.savefig(fn_fig.replace('.png', '_1to1.png'), dpi=200,bbox_inches='tight')
            plt.close()
        
        # Select time range for plot_quantile subplot
        time = select_random_48hr_window(data_AWS)
        
        # Create the timed variable for quantile function
        raw = data_MERRA2.sel(time=time)
        adj = np.interp(raw, sorted, mapping)
        aws = data_AWS.loc[time]
        
        # Convert back to local time for plotting
        time += pd.Timedelta(hours=timezone)
        timed_tuple = (time, raw, adj, aws)
        
        # Plot quantiles
        if get_quantile_plot:
            fig, axes = plot_quantile(X_train, y_train, all_MERRA2, timed_tuple, fn)
            axes[-1].set_ylabel(var_label,fontsize=12)
            plt.savefig(fn_fig,dpi=200,bbox_inches='tight')
            plt.close()

    print('DONE', glacier)
