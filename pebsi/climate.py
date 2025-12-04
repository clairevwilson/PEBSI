"""
Climate class for PEBSI

Handles generation of climate dataset for
the model run duration and adjusts to
the site elevation

@author: clairevwilson
"""
# Built-in libraries
import threading
import os,sys
import time
# External libraries
import pandas as pd
import numpy as np
import xarray as xr
# Local libraries
import pebsi.input as prms

class Climate():
    """
    Climate-related functions which build the 
    climate dataset for a single simulation.

    If use_AWS = True in the input, the climate 
    dataset will be filled with all variables in
    the AWS dataset before turning to reanalysis 
    data to fill the remaining variables.

    If use_AWS = False, only reanalysis data will 
    be used.
    """
    def __init__(self,args):
        """
        Initializes glacier information and creates
        the dataset where climate data will be stored.

        Parameters
        ==========
        args : command line arguments
        """
        # start timer
        self.start_time = time.time()

        # load args and run information
        self.args = args
        self.dates = pd.date_range(args.startdate,args.enddate,freq='h')
        self.dates_UTC = self.dates - args.timezone

        # specify glacier and time information
        self.lat = args.lat
        self.lon = args.lon
        self.n_time = len(self.dates)
        self.elev = args.elev

        # list all required variables
        self.all_vars = ['temp','tp','rh','uwind','vwind','sp','SWin','LWin',
                            'bcwet','bcdry','ocwet','ocdry','dustwet','dustdry']

        # find median elevation of the glacier from RGI for precip gradient
        RGI_region = args.glac_no.split('.')[0]
        if float(RGI_region) > 0:
            for fn in os.listdir(prms.RGI_fp):
                # open the attributes .csv for the correct region
                if fn[:2] == RGI_region and fn[-3:] == 'csv':
                    RGI_df = pd.read_csv(prms.RGI_fp + fn)
                    RGI_df.index = [f.split('-')[-1] for f in RGI_df['RGIId']]
            self.median_elev = RGI_df.loc[args.glac_no,'Zmed']
        else:
            self.median_elev = self.elev

        # find elevation of temperature data
        if 'temp' in prms.bias_vars:
            self.temp_elev = prms.station_elevation[self.args.glac_name]

        # check if storing the cds
        self.store_cds = prms.store_climate 

        # get data from existing .nc or from AWS/MERRA-2
        if str(bool(args.input_climate)) != 'False':
            # open data from existing .nc
            cds_input_fn = prms.cds_input_fn.replace('GLACIER', args.glac_name)
            fn_data = prms.climate_fp + cds_input_fn.replace('SITE', args.site)
            if not os.path.exists(fn_data):
                print(f'Climate data not found: getting new cds and saving to {fn_data}')
                self.store_cds = True
                self.cds_output_fn = fn_data
            else:
                # load data and tell the model to skip getting climate
                self.loaded_climate = True
                cds = xr.open_dataset(fn_data)

                # replace dates with dates from cds
                args.startdate = pd.to_datetime(cds.time.values[0])
                args.enddate = pd.to_datetime(cds.time.values[-1])
                self.dates = pd.date_range(args.startdate,args.enddate,freq='h')
                self.dates_UTC = self.dates - args.timezone
                self.n_time = len(self.dates)
                self.cds = cds

                # check which variables are stored as measured in cds
                self.measured_vars = [v for v in cds.variables if 'measured' in cds[v].attrs]
                return

        # did not load dataset so need to make it
        self.loaded_climate = False

        # create dictionary containing reanalysis filenames
        self.get_vardict()
        if not self.args.use_AWS:
            self.measured_vars = []

        # create empty dataset
        nans = np.ones(self.n_time)*np.nan
        self.cds = xr.Dataset(data_vars = dict(
                SWin = (['time'],nans,{'units':'J m-2'}),
                SWout = (['time'],nans,{'units':'J m-2'}),
                albedo = (['time'],nans,{'units':'-'}),
                LWin = (['time'],nans,{'units':'J m-2'}),
                LWout = (['time'],nans,{'units':'J m-2'}),
                NR = (['time'],nans,{'units':'J m-2'}),
                tcc = (['time'],nans,{'units':'-'}),
                rh = (['time'],nans,{'units':'%'}),
                uwind = (['time'],nans,{'units':'m s-1'}),
                vwind = (['time'],nans,{'units':'m s-1'}),
                wind = (['time'],nans,{'units':'m s-1'}),
                winddir = (['time'],nans,{'units':'o'}),
                bcdry = (['time'],nans,{'units':'kg m-2 s-1'}),
                bcwet = (['time'],nans,{'units':'kg m-2 s-1'}),
                ocdry = (['time'],nans,{'units':'kg m-2 s-1'}),
                ocwet = (['time'],nans,{'units':'kg m-2 s-1'}),
                dustdry = (['time'],nans,{'units':'kg m-2 s-1'}),
                dustwet = (['time'],nans,{'units':'kg m-2 s-1'}),
                temp = (['time'],nans,{'units':'C'}),
                tp = (['time'],nans,{'units':'m'}),
                sp = (['time'],nans,{'units':'Pa'})
                ),
                coords = dict(time=(['time'],self.dates)))
            
        return
    
    def get_AWS(self,fp):
        """
        Loads available AWS data and determines which
        variables need come from reanalysis data.

        Parameters
        ==========
        fp : str
            Filepath to the AWS dataset
        """
        # load data
        df = pd.read_csv(fp,index_col=0)
        df = df.set_index(pd.to_datetime(df.index))

        # check dates of data match input dates
        data_start = pd.to_datetime(df.index.to_numpy()[0])
        data_end = pd.to_datetime(df.index.to_numpy()[-1])
        assert self.dates[0] >= data_start, f'Check input dates: start date before range of AWS data ({data_start})'
        assert self.dates[len(self.dates)-1] <= data_end, f'Check input dates: end date after range of AWS data ({data_end})'
        
        # reindex in case of MERRA-2 half-hour timesteps
        new_index = pd.DatetimeIndex(self.dates)
        index_joined = df.index.join(new_index, how='outer')
        df = df.reindex(index=index_joined).interpolate().reindex(new_index)

        # get AWS elevation
        metadata_df = pd.read_csv(prms.AWS_metadata_fn, sep='\t', index_col='glacier')
        self.AWS_elev = metadata_df.loc[self.args.glac_name, 'elevation']
        # can have duplicates for a glacier
        if '__iter__' in dir(self.AWS_elev):
            station = fp.split(self.args.glac_name)[-1].split('.csv')[0]
            assert station in metadata_df['station'].values, f'specify station name as {station} in aws_metadata.txt'
            glac_df = metadata_df.loc[self.args.glac_name]
            self.AWS_elev = glac_df.loc[glac_df['station'] == station, 'elevation'].values[0]

        # get the available variables
        all_AWS_vars = ['temp','tp','rh','uwind','vwind','sp','SWin','SWout','albedo',
                        'NR','LWin','LWout','bcwet','bcdry','ocwet','ocdry','dustwet','dustdry']
        AWS_vars = df.columns
        self.measured_vars = list(set(all_AWS_vars) & set(AWS_vars))

        # check if wind direction can be calculated
        uwind_measured = 'uwind' in AWS_vars
        vwind_measured = 'vwind' in AWS_vars
        if uwind_measured ^ vwind_measured:
            self.wind_direction = False
            # print('! Wind speed was input as a scalar. Wind shading is not handled')
        else:
            self.wind_direction = True
        
        # extract and store data
        for var in self.measured_vars:
            self.cds[var].values = df[var].astype(float)

        # determine which data variables are still needed from reanalysis
        need_vars = [e for e in self.all_vars if e not in AWS_vars]

        # if net radiation was measured, don't need LWin
        if 'NR' in self.measured_vars:
            need_vars.remove('LWin')

        # if wind was input as a scalar, don't need the other direction of wind
        if not self.wind_direction:
            if uwind_measured:
                self.cds['vwind'].values = np.zeros(self.n_time)
                need_vars.remove('vwind')
            elif vwind_measured:
                self.cds['uwind'].values = np.zeros(self.n_time)
                need_vars.remove('uwind')
        self.need_vars = need_vars
        return need_vars
    
    def get_reanalysis(self,vars):
        """
        Fetches reanalysis climate data variables.

        Parameters
        ==========
        vars : list-like
            Variables to be fetched from reanalysis data
        """
        # load time and point data
        dates = self.dates_UTC
        lat = self.lat
        lon = self.lon
        
        # interpolate data if time was input on the hour instead of half-hour
        self.interpolate = dates[0].minute != 30 and prms.reanalysis == 'MERRA2'
        
        # get reanalysis data geopotential
        z_fp = self.reanalysis_fp + self.var_dict['elev']['fn']
        zds = xr.open_dataarray(z_fp)
        zds = zds.sel({self.lat_vn:lat,self.lon_vn:lon},method='nearest')
        zds = self.check_units('elev',zds)
        self.reanalysis_elev = zds.isel(time=0).values.ravel()[0]
        
        # initiate variables
        all_data = {}
        # loop through vars
        for var in vars:
            # gather data for each var and add to all_data
            fn = self.reanalysis_fp + self.var_dict[var]['fn']
            all_data = self.get_var_data(fn,var,all_data)

        # store data
        for var in vars:
            self.cds[var].values = all_data[var].ravel()
        return
    
    def get_var_data(self, fn, var, result_dict):
        # get dates
        dates = self.dates_UTC

        # special check for RH: must be calculated from QV
        if var == 'rh' and not os.path.exists(fn):
            assert prms.reanalysis == 'MERRA2', 'RH conversion not yet set up for ERA-5'
            self.create_rh2m_ds(fn)

        # open and check units of climate data
        ds = xr.open_dataset(fn)

        # get variable names
        vn = self.var_dict[var]['vn'] 
        lat_vn,lon_vn = [self.lat_vn,self.lon_vn]

        # light-absorbing particles always come from MERRA-2 and need special treatment
        if 'bc' in var or 'oc' in var or 'dust' in var:
            if prms.reanalysis == 'ERA5-hourly':
                lat_vn,lon_vn = ['lat','lon']

        # index by lat and lon
        if ds.coords[lat_vn].values.size > 1:
            ds = ds.sel({lat_vn:self.lat,lon_vn:self.lon}, method='nearest')[vn]
        else:
            ds = ds[vn]

        # check the units
        ds = self.check_units(var,ds)

        # for time-varying variables, select/interpolate to the model time
        if var != 'elev':
            dep_var = 'bc' in var or 'dust' in var or 'oc' in var
            if not dep_var and prms.reanalysis == 'ERA5-hourly':
                assert dates[0] >= pd.to_datetime(ds.time.values[0])
                assert dates[-1] <= pd.to_datetime(ds.time.values[-1])
                ds = ds.interp(time=dates)
            elif self.interpolate:
                ds = ds.interp(time=dates)
            else:
                ds = ds.sel(time=dates)
        
        # make sure the correct grid cell was accessed
        assert np.abs(ds.coords[lat_vn].values - float(self.lat)) <= 0.5, 'Wrong grid cell was accessed'
        assert np.abs(ds.coords[lon_vn].values - float(self.lon)) <= 0.625, 'Wrong grid cell was accessed'

        # store result
        result_dict[var] = ds.values.ravel()
        ds.close()

        # return the result dict
        return result_dict
    
    def check_ds(self):
        """
        Calculates wind speed and direction from u and v,
        bias-corrects reanalysis data with quantile mapping,
        adjusts elevation-dependent variables, and
        checks that all required variables are filled.
        """
        # calculate wind speed and direction from u and v components
        # ***WINDMAPPER GOES HERE***
        uwind = self.cds['uwind'].values
        vwind = self.cds['vwind'].values
        wind = np.sqrt(np.power(uwind,2)+np.power(vwind,2))
        winddir = np.arctan2(-uwind,-vwind) * 180 / np.pi
        self.cds['wind'].values = wind
        self.cds['winddir'].values = winddir

        if not self.loaded_climate:
            if prms.reanalysis == 'MERRA2':
                # correct MERRA-2 variables in inputs list
                self.bias_vars = prms.bias_vars
                if self.args.debug and len(self.bias_vars) > 0:
                    print('~ Applying quantile mapping for:',self.bias_vars)
                for var in self.bias_vars:
                    from_MERRA = True if not self.args.use_AWS else var in self.need_vars
                    if from_MERRA:
                        self.bias_adjust_qm(var)
            
            # adjust MERRA-2 deposition by reduction coefficient
            if prms.reanalysis == 'MERRA2' and prms.adjust_deposition:
                self.adjust_dep()

        # check all variables are there
        failed = []
        for var in self.all_vars:
            data = self.cds[var].values
            if np.any(np.isnan(data)):
                failed.append(var)

        # can input net radiation instead of incoming LW radiation
        if 'LWin' in failed and 'NR' in self.measured_vars:
            failed.remove('LWin')

        # print any missing data
        if len(failed) > 0:
            print('Missing data from',failed)
            self.exit()
        
        # done getting climate
        time_elapsed = time.time()-self.start_time
        if self.args.debug:
            print(f'~ Loaded climate dataset in {time_elapsed:.1f} seconds ~')
        return

    def adjust_to_elevation(self, temp=True, precip=True, sp=True, LWin=True):
        """
        Adjusts elevation-dependent climate variables 
        (temperature, precip, surface pressure, and
        incoming longwave radiation).
        
        Vars can be toggled using
        temp / precip / sp / LWin = False
        """
        # Set copies of un-edited variables
        self.original_temp = self.cds.temp.copy(deep=True).values
        self.original_tp = self.cds.tp.copy(deep=True).values
        self.original_sp = self.cds.sp.copy(deep=True).values
        self.original_LWin = self.cds.LWin.copy(deep=True).values

        # TEMPERATURE: correct according to lapse rate
        if temp:
            self.temp_to_elevation()
            
        # PRECIP: correct according to precipitation gradient
        if precip:
            self.precip_to_elevation()

        # SURFACE PRESSURE: correct according to barometric law
        if sp:
            self.sp_to_elevation()

        # LONGWAVE: correct with elevation-dependent emissivity 
        if LWin:
            self.LWin_to_elevation()  
        return
    
    def check_units(self,var,ds):
        """
        Checks the units for a meteorological
        variable and puts them in the correct units.

        Parameters
        ==========
        var : str
            Variable to check
        ds : xr.Dataset
            Climate dataset

        Returns
        -------
        ds : xr.Dataset
            Updated climate dataset
        """
        # CONSTANTS
        SPH = prms.seconds_per_hour
        CTOK = prms.celsius_to_kelvin
        GRAVITY = prms.gravity
        DENSITY_WATER = prms.density_water

        # define the units the model needs
        model_units = {'temp':'C','uwind':'m s-1','vwind':'m s-1',
                       'rh':'%','sp':'Pa','tp':'m s-1','elev':'m',
                       'SWin':'J m-2', 'LWin':'J m-2', 'NR':'J m-2', 'tcc':'-',
                       'bcdry':'kg m-2 s-1', 'bcwet':'kg m-2 s-1',
                       'ocdry':'kg m-2 s-1', 'ocwet':'kg m-2 s-1',
                       'dustdry':'kg m-2 s-1', 'dustwet':'kg m-2 s-1'}
        
        # get the current variable's units
        units_in = ds.attrs['units'].replace('*','')
        units_out = model_units[var]

        # check and make replacements
        if units_in != units_out:
            if var == 'temp' and units_in == 'K':
                ds = ds - CTOK
            elif var == 'rh' and units_in in ['-','0-1']:
                ds  = ds * 100
            elif var == 'tp':
                if units_in == 'kg m-2 s-1':
                    ds = ds / DENSITY_WATER * SPH
                elif units_in == 'm':
                    ds = ds / SPH
            elif var in ['SWin','LWin','NR'] and units_in == 'W m-2':
                ds = ds * SPH
            elif var == 'elev' and units_in in ['m+2 s-2','m2 s-2']:
                ds = ds / GRAVITY
            else:
                print(f'WARNING! Units did not match for {var} but were not updated')
                print(f'Previously {units_in}; should be {units_out}')
                print('Make a manual change in check_units (climate.py)')
                self.exit()
        return ds

    def store(self):
        # set output filename for storing .nc
        if prms.cds_output_fn == 'default':
            cds_fn = self.args.out.replace('.nc','_climate.nc')
            self.cds_output_fn = prms.climate_fp + cds_fn
        else:
            self.cds_output_fn = prms.climate_fp + prms.cds_output_fn

        # add measured boolean to output
        for var in self.cds.variables:
            if var in self.measured_vars:
                self.cds[var].attrs['measured'] = 'True'
        
        # store cds
        self.cds.to_netcdf(self.cds_output_fn)
        print(f'Climate dataset saved to {self.cds_output_fn}')

    def adjust_dep(self):
        """
        Updates deposition based on preprocessed 
        reduction coefficients
        """
        print('***Hard-coded MERRA-2 to UK-ESM filepath for deposition adjustment***')
        fn = self.reanalysis_fp + 'merra2_to_ukesm_conversion_map_MERRAgrid.nc'
        ds_f = xr.open_dataarray(fn)
        ds_f = ds_f.sel({self.lat_vn:self.lat,self.lon_vn:self.lon},method='nearest')
        f = ds_f.mean('time').values.ravel()[0]
        # To do time-moving monthly factors:
        # for date in ds_f.time.values:
        #     # select the reduction coefficient of the current month
        #     f = ds_f.sel(time=date).values[0]
        #     # index the climate dataset by the month and year
        #     month = pd.to_datetime(date).month
        #     year = pd.to_datetime(date).year
        #     idx_month = np.where(self.cds.coords['time'].dt.month.values == month)[0]
        #     idx_year = np.where(self.cds.coords['time'].dt.year.values == year)[0]
        #     idx = list(set(idx_month)&set(idx_year))
        #     # update dry and wet BC deposition
        #     self.cds['bcdry'][{'time':idx}] = self.cds['bcdry'][{'time':idx}] * f
        #     self.cds['bcwet'][{'time':idx}] = self.cds['bcwet'][{'time':idx}] * f
        self.cds['bcdry'].values *= f
        self.cds['bcwet'].values *= f
        return

    def temp_to_elevation(self):
        """
        Corrects air temperature at the site elevation
        based on a linear lapse rate
        """
        # CONSTANTS
        LAPSE_RATE = float(self.args.lapse_rate) / 1000 # in K m-1

        # get elevation of the original temperature data
        if 'temp' in prms.bias_vars and 'temp' not in self.measured_vars:
            # if temperature was a bias-corrected variable, use pre-set temp_elev
            temp_elev = self.temp_elev
        else:
            temp_elev = self.AWS_elev if 'temp' in self.measured_vars else self.reanalysis_elev
        new_temp = self.original_temp + LAPSE_RATE*(self.elev - temp_elev)

        # update temperature in the cds
        self.cds.temp.values = new_temp.ravel()
        return

    def precip_to_elevation(self):
        """
        Corrects precipitation at the site elevation
        based on a % gradient
        """
        # CONSTANTS
        if self.args.glac_name in prms.precgrads:
            PREC_GRAD = prms.precgrads[self.args.glac_name]
        else:
            PREC_GRAD = prms.precgrad

        # get elevation of the precipitation data
        tp_elev = self.median_elev
        new_tp = self.original_tp*(1+PREC_GRAD*(self.elev-tp_elev))

        # update precip in the cds
        self.cds.tp.values = new_tp.ravel()
        return

    def sp_to_elevation(self):
        """
        Corrects surface pressure according to barometric law
        """
        # CONSTANTS
        LAPSE_RATE = float(self.args.lapse_rate) / 1000 # in K m-1
        GRAVITY = prms.gravity
        R_GAS = prms.R_gas
        MM_AIR = prms.molarmass_air
        CTOK = prms.celsius_to_kelvin
        
        # get elevation of surface pressure data
        sp_elev = self.AWS_elev if 'sp' in self.measured_vars else self.reanalysis_elev

        # adjust temperature from elevation of the site to elevation of the sp data
        new_temp = self.cds.temp.values
        temp_sp_elev = new_temp + LAPSE_RATE*(sp_elev - self.elev) + CTOK

        # calculate new surface pressure with barometric law
        exponent = -GRAVITY*MM_AIR/(R_GAS*LAPSE_RATE)
        ratio = ((new_temp + CTOK) / temp_sp_elev) ** (exponent)
        new_sp = self.original_sp * ratio

        # update surface pressure in the cds
        self.cds.sp.values = new_sp.ravel()
        return

    def LWin_to_elevation(self, temp_LW_elev=False):
        """
        Corrects incoming longwave by determining a
        theoretical difference in longwave under clear 
        sky conditions using the Brutsaert (1975) 
        parameterization of emissivity (based on 
        temperature and vapor pressure) and applying
        this difference to the MERRA-2 longwave data.

        Temperature sensitivity runs require input temp 
        at the MERRA-2 elevation. Base PEBSI does not use
        arg temp_LW_elev
        """
        # CONSTANTS
        SIGMA_SB = prms.sigma_SB
        LAPSE_RATE = float(self.args.lapse_rate) / 1000 # in K m-1
        SPH = prms.seconds_per_hour
        CTOK = prms.celsius_to_kelvin

        # get temperature and RH data at the site and data location
        rh = self.cds.rh.values             # RH assumed constant with elevation
        temp_site = self.cds.temp.values    # Temperature already updated to self.elev
        LW_elev = self.AWS_elev if 'LWin' in self.measured_vars else self.reanalysis_elev
        if type(temp_LW_elev) == bool and not temp_LW_elev:
            temp_LW_elev = temp_site + LAPSE_RATE*(LW_elev - self.elev)

        # store temperature in Kelvin
        temp_site_K = temp_site + CTOK
        temp_LW_elev_K = temp_LW_elev + CTOK

        # calculate emissivity from temperature at each elevation
        eps_site = self.emissivity_brutsaert(temp_site, rh)
        eps_LW_elev = self.emissivity_brutsaert(temp_LW_elev, rh)

        # compute clear-sky longwave radiation at each elevation [W m-2]
        LWin_clear_site = eps_site * SIGMA_SB * temp_site_K**4
        LWin_clear_MERRA2 = eps_LW_elev * SIGMA_SB * temp_LW_elev_K**4

        # apply difference in clear-sky radiation to longwave data
        delta_LW = (LWin_clear_site - LWin_clear_MERRA2) * SPH
        new_LWin = self.original_LWin + delta_LW

        # Update surface pressure in the cds
        self.cds.LWin.values = new_LWin.ravel()
        return
    
    def bias_adjust_qm(self,var):
        """
        Applies preprocessed quantile mapping to
        a reanalysis climate variable.

        Parameters
        ==========
        var : str
            Variable to bias correct
        """
        # get quantile mapping .csv filename
        bias_fn = prms.bias_fn.replace('METHOD','quantile_mapping').replace('VAR',var)
        bias_fn = bias_fn.replace('GLACIER', self.args.glac_name)

        # need to use file generated without a lapse rate for temperature
        if var == 'temp':
            bias_fn = bias_fn.replace('.csv','_0.0.csv')

        assert os.path.exists(bias_fn), f'Quantile mapping file does not exist for {var}'
        bias_df = pd.read_csv(bias_fn)
        
        # interpolate values according to quantile mapping
        values = self.cds[var].values
        adjusted = np.interp(values, bias_df['sorted'], bias_df['mapping'])

        # update values
        self.cds[var].values = adjusted
        return
    
    def create_rh2m_ds(self, fn):
        """
        Creates an RH2M (2 m relative humidity) 
        dataset from specific humidity, air 
        temperature, and surface pressure datasets
        from MERRA-2.

        Parameters
        ==========
        fn : str
            Filename of the RH2M dataset to create
        """
        # CONSTANTS
        CTOK = prms.celsius_to_kelvin

        # get variable names
        rh_vn = self.var_dict['rh']['vn']
        temp_vn = self.var_dict['temp']['vn']
        sp_vn = self.var_dict['sp']['vn']
        qv_vn = 'QV2M'

        ds_qv = xr.open_dataset(fn.replace(rh_vn, qv_vn))
        ds_temp = xr.open_dataset(fn.replace(rh_vn, temp_vn))
        ds_sp = xr.open_dataset(fn.replace(rh_vn, sp_vn))

        # calculate saturation pressure from air temperature
        esat = self.sat_vapor_pressure(ds_temp[temp_vn].values - CTOK)

        # saturation and actual specific humidity vapor pressure
        ws = 0.622*esat / (ds_sp[sp_vn].values - esat)
        w = ds_qv[qv_vn].values / (1 - ds_qv[qv_vn].values)

        # relative humidity as a percentage of saturation humidity
        rh = w / ws * 100

        # create copy dataset and fill with RH data
        ds_rh = ds_qv.copy(deep=True)
        ds_rh['RH2M'] = xr.DataArray(
                                        rh,
                                        dims=['time'],
                                        coords={'time': ds_rh['time'].values},
                                        attrs={'units': '%', 'long_name': '2-meter_relative_humidity'}
                                    )

        # drop QV data and store the RH dataset
        ds_rh = ds_rh.drop_vars('QV2M')
        ds_rh.to_netcdf(fn)
        return

    def sat_vapor_pressure(self,airtemp,method='ARM'):
        """
        Returns saturation vapor pressure [Pa] 
        from air temperature 

        Parameters
        ==========
        airtemp : float
            Air temperature [C]
        """
        # CONSTANTS
        CTOK = prms.celsius_to_kelvin

        # calculate saturation vapor pressure in kPa
        if method in ['ARM']:
            P = 0.61094*np.exp(17.625*airtemp/(airtemp+243.04)) # kPa
        elif method in ['Sonntag']:
            # follows COSIPY
            airtemp += CTOK
            if airtemp > CTOK: # over water
                P = 0.6112*np.exp(17.67*(airtemp-CTOK)/(airtemp-29.66))
            else: # over ice
                P = 0.6112*np.exp(22.46*(airtemp-CTOK)/(airtemp-0.55))

        # return vapor pressure in Pa
        return P*1000
    
    def emissivity_brutsaert(self, airtemp, rh):
        """
        Returns Brutsaert (1975) clear-sky atmospheric
        emissivity from temperature and relative humidity
        
        Parameters
        ==========
        airtemp : float or np.array
            Air temperature [C]
        rh : float or np.array
            Relative humidity [%]
        """
        # CONSTANTS
        CTOK = prms.celsius_to_kelvin

        # get saturation vapor pressure
        esat = self.sat_vapor_pressure(airtemp)

        # convert to actual vapor pressure (in hPa)
        e_hPa = esat * (rh / 100) / 100

        return 1.24 * (e_hPa / (airtemp + CTOK)) ** (1/7)
    
    def dew_point(self,vap):
        """
        Returns dewpoint air temperature from 
        vapor pressure.

        Parameters
        ==========
        vap : float
            Vapor pressure [Pa]
        """
        return 243.04*np.log(vap/610.94)/(17.625-np.log(vap/610.94))

    def get_vardict(self):
        """
        Fills a dictionary with the reanalysis
        file and variable names.
        """
        # determine filetag for MERRA2 lat/lon file
        assert os.path.exists(prms.merra2_eg_fn), f'Store global geopotential file to {prms.merra2_eg_fn}'
        ds_global = xr.open_dataset(prms.merra2_eg_fn)
        ds_closest = ds_global.sel(lat=self.lat, lon=self.lon, method='nearest')
        flat = str(ds_closest.lat.values)
        flon = str(ds_closest.lon.values)
        tag = prms.MERRA2_filetag if prms.MERRA2_filetag else f'{flat}_{flon}'

        # update filenames for MERRA-2 (need grid lat/lon)
        self.reanalysis_fp = prms.climate_fp
        self.var_dict = {'temp':{'fn':[],'vn':[]},
            'rh':{'fn':[],'vn':[]},'sp':{'fn':[],'vn':[]},
            'tp':{'fn':[],'vn':[]},'tcc':{'fn':[],'vn':[]},
            'SWin':{'fn':[],'vn':[]},'LWin':{'fn':[],'vn':[]},
            'uwind':{'fn':[],'vn':[]},'vwind':{'fn':[],'vn':[]},
            'bcdry':{'fn':[],'vn':[]},'bcwet':{'fn':[],'vn':[]},
            'ocdry':{'fn':[],'vn':[]},'ocwet':{'fn':[],'vn':[]},
            'dustdry':{'fn':[],'vn':[]},'dustwet':{'fn':[],'vn':[]},
            'elev':{'fn':[],'vn':[]},'time':{'fn':'','vn':''},
            'lat':{'fn':'','vn':''}, 'lon':{'fn':'','vn':''}}
        if prms.reanalysis == 'MERRA2':
            self.reanalysis_fp += 'MERRA2/'
            self.var_dict['temp']['vn'] = 'T2M'
            self.var_dict['rh']['vn'] = 'RH2M'
            self.var_dict['sp']['vn'] = 'PS'
            self.var_dict['tp']['vn'] = 'PRECTOTCORR'
            self.var_dict['elev']['vn'] = 'PHIS'
            self.var_dict['tcc']['vn'] = 'CLDTOT'
            self.var_dict['SWin']['vn'] = 'SWGDN'
            self.var_dict['LWin']['vn'] = 'LWGAB'
            self.var_dict['uwind']['vn'] = 'U2M'
            self.var_dict['vwind']['vn'] = 'V2M'
            self.var_dict['bcwet']['vn'] = 'BCWT002'
            self.var_dict['bcdry']['vn'] = 'BCDP002'
            self.var_dict['ocwet']['vn'] = 'OCWT002'
            self.var_dict['ocdry']['vn'] = 'OCDP002'
            self.var_dict['dustwet']['vn'] = 'DUWT003'
            self.var_dict['dustdry']['vn'] = 'DUDP003'
            self.time_vn = 'time'
            self.lat_vn = 'lat'
            self.lon_vn = 'lon'
            self.elev_vn = self.var_dict['elev']['vn']

            # Variable filenames
            self.var_dict['temp']['fn'] = f'{tag}/T2M_{tag}.nc'
            self.var_dict['rh']['fn'] = f'{tag}/RH2M_{tag}.nc'
            self.var_dict['sp']['fn'] = f'{tag}/PS_{tag}.nc'
            self.var_dict['tcc']['fn'] = f'{tag}/CLDTOT_{tag}.nc'
            self.var_dict['LWin']['fn'] = f'{tag}/LWGAB_{tag}.nc'
            self.var_dict['SWin']['fn'] = f'{tag}/SWGDN_{tag}.nc'
            self.var_dict['vwind']['fn'] = f'{tag}/V2M_{tag}.nc'
            self.var_dict['uwind']['fn'] = f'{tag}/U2M_{tag}.nc'
            self.var_dict['tp']['fn'] = f'{tag}/PRECTOTCORR_{tag}.nc'
            self.var_dict['elev']['fn'] = f'MERRA2constants.nc4'
            self.var_dict['bcwet']['fn'] = f'{tag}/BCWT002_{tag}.nc'
            self.var_dict['bcdry']['fn'] = f'{tag}/BCDP002_{tag}.nc'
            self.var_dict['ocwet']['fn'] = f'{tag}/OCWT002_{tag}.nc'
            self.var_dict['ocdry']['fn'] = f'{tag}/OCDP002_{tag}.nc'
            self.var_dict['dustwet']['fn'] = f'{tag}/DUWT003_{tag}.nc'
            self.var_dict['dustdry']['fn'] = f'{tag}/DUDP003_{tag}.nc'
        elif prms.reanalysis == 'ERA5-hourly':
            self.reanalysis_fp += 'ERA5/ERA5_hourly/'

            # Variable names for energy balance
            self.var_dict['temp']['vn'] = 't2m'
            self.var_dict['rh']['vn'] = 'rh'
            self.var_dict['sp']['vn'] = 'sp'
            self.var_dict['tp']['vn'] = 'tp'
            self.var_dict['elev']['vn'] = 'z'
            self.var_dict['tcc']['vn'] = 'tcc'
            self.var_dict['SWin']['vn'] = 'ssrd'
            self.var_dict['LWin']['vn'] = 'strd'
            self.var_dict['uwind']['vn'] = 'u10'
            self.var_dict['vwind']['vn'] = 'v10'
            self.var_dict['bcwet']['vn'] = 'BCWT002'
            self.var_dict['bcdry']['vn'] = 'BCDP002'
            self.var_dict['ocwet']['vn'] = 'OCWT002'
            self.var_dict['ocdry']['vn'] = 'OCDP002'
            self.var_dict['dustwet']['vn'] = 'DUWT003'
            self.var_dict['dustdry']['vn'] = 'DUDP003'
            self.time_vn = 'time'
            self.lat_vn = 'latitude'
            self.lon_vn = 'longitude'
            self.elev_vn = self.var_dict['elev']['vn']

            # Variable filenames
            self.var_dict['temp']['fn'] = 'ERA5_temp_hourly.nc'
            self.var_dict['rh']['fn'] = 'ERA5_rh_hourly.nc'
            self.var_dict['sp']['fn'] = 'ERA5_sp_hourly.nc'
            self.var_dict['tcc']['fn'] = 'ERA5_tcc_hourly.nc'
            self.var_dict['LWin']['fn'] = 'ERA5_LWin_hourly.nc'
            self.var_dict['SWin']['fn'] = 'ERA5_SWin_hourly.nc'
            self.var_dict['vwind']['fn'] = 'ERA5_vwind_hourly.nc'
            self.var_dict['uwind']['fn'] = 'ERA5_uwind_hourly.nc'
            self.var_dict['tp']['fn'] = 'ERA5_tp_hourly.nc'
            self.var_dict['elev']['fn'] = 'ERA5_geopotential_2000.nc'
            self.var_dict['bcwet']['fn'] = f'./../../MERRA2/{tag}/BCWT002_{tag}.nc'
            self.var_dict['bcdry']['fn'] = f'./../../MERRA2/{tag}/BCDP002_{tag}.nc'
            self.var_dict['ocwet']['fn'] = f'./../../MERRA2/{tag}/OCWT002_{tag}.nc'
            self.var_dict['ocdry']['fn'] = f'./../../MERRA2/{tag}/OCDP002_{tag}.nc'
            self.var_dict['dustwet']['fn'] = f'./../../MERRA2/{tag}/DUWT003_{tag}.nc'
            self.var_dict['dustdry']['fn'] = f'./../../MERRA2/{tag}/DUDP003_{tag}.nc'

    def exit(self):
        sys.exit()