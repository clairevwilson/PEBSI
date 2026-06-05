"""
Climate class for PEBSI

Handles generation of climate dataset for
the model run duration and adjusts to
the site elevation

@author: clairevwilson
"""
# Built-in libraries
import os,sys
import time
# External libraries
import pandas as pd
import numpy as np
import xarray as xr
# Internal libraries
from util.config import ConfigError

class Climate():
    """
    Climate-related functions which build the 
    climate dataset for a single simulation.

    If use_aws = True in the input, the climate 
    dataset will be filled with all variables in
    the AWS dataset before turning to reanalysis 
    data to fill the remaining variables.

    If use_aws = False, only reanalysis data will 
    be used.
    """
    def __init__(self, args, terrain):
        """
        Initializes glacier information and creates
        the dataset where climate data will be stored.

        Parameters
        ==========
        args : command line arguments
        """
        # start timer for loading climate data
        self.start_time = time.time()

        # load args and simulation information
        self.args = args
        self.terrain = terrain
        self.get_spatial_temporal_info()

        # list all required variables
        self.all_vars = ['temp','tp','rh','uwind','vwind','sp','SWin','LWin',
                            'bcwet','bcdry','ocwet','ocdry','dustwet','dustdry']
        self.optional_vars = ['SWout','LWout','tcc','NR','albedo']
        self.carbon_vars = ['bcwet','bcdry','ocwet','ocdry']

        # create dictionary containing reanalysis filenames
        self.get_vardict()
        self.bias_vars = []
        if not self.args.use_aws:
            self.measured_vars = []
            self.need_vars = self.all_vars.copy()
        else:
            assert ~self.args.use_aws, 'NOT SET UP FOR AWS DATA YET'
            pass
        return
    
    def get_spatial_temporal_info(self):
        self.dates = pd.date_range(self.args.start_date,self.args.end_date,freq='h')

        # handle timezones
        tz = self.terrain.tz_n
        if len(np.unique(tz)) == 1:
            # if all glaciers are in the same timezone, dates_UTC can be 1D
            self.dates_UTC = self.dates - pd.to_timedelta(int(tz[0]), unit='h')
        else:
            # else, need dates_UTC to be a 2D array (points, time)
            timedelta_col = pd.to_timedelta(tz, unit='h')[:, np.newaxis]
            self.dates_UTC = self.dates[np.newaxis, :] - timedelta_col

        # specify spatial and temporal information
        self.N_TIME = len(self.dates)
        self.shape = (self.terrain.N_POINTS, self.N_TIME)
        return
    
    # def get_aws(self,fp):
    #     """
    #     Loads available AWS data and determines which
    #     variables need come from reanalysis data.

    #     Parameters
    #     ==========
    #     fp : str
    #         Filepath to the AWS dataset
    #     """
    #     # load data
    #     df = pd.read_csv(fp,index_col=0)
    #     df = df.set_index(pd.to_datetime(df.index))

    #     # check dates of data match input dates
    #     data_start = pd.to_datetime(df.index.to_numpy()[0])
    #     data_end = pd.to_datetime(df.index.to_numpy()[-1])
    #     assert self.dates[0] >= data_start, f'Check input dates: start date before range of AWS data ({data_start})'
    #     assert self.dates[len(self.dates)-1] <= data_end, f'Check input dates: end date after range of AWS data ({data_end})'
        
    #     # reindex in case of MERRA-2 half-hour timesteps
    #     new_index = pd.DatetimeIndex(self.dates)
    #     index_joined = df.index.join(new_index, how='outer')
    #     df = df.reindex(index=index_joined).interpolate().reindex(new_index)

    #     # get AWS elevation
    #     if self.args.aws_elev is not None:
    #         self.aws_elev = self.args.aws_elev

    #     # get the available variables
    #     all_aws_vars = ['temp','tp','rh','uwind','vwind','sp','SWin','SWout','albedo',
    #                     'NR','LWin','LWout','bcwet','bcdry','ocwet','ocdry','dustwet','dustdry']
    #     aws_vars = df.columns
    #     self.measured_vars = list(set(all_aws_vars) & set(aws_vars))

    #     # check if wind direction can be calculated
    #     uwind_measured = 'uwind' in aws_vars
    #     vwind_measured = 'vwind' in aws_vars
    #     if uwind_measured ^ vwind_measured:
    #         self.wind_direction = False
    #         # print('! Wind speed was input as a scalar. Wind shading is not handled')
    #     else:
    #         self.wind_direction = True
        
    #     # extract and store data
    #     for var in self.measured_vars:
    #         self.cds[var].values = df[var].astype(float)

    #     # determine which data variables are still needed from reanalysis
    #     need_vars = [e for e in self.all_vars if e not in aws_vars]

    #     # if net radiation was measured, don't need LWin
    #     if 'NR' in self.measured_vars:
    #         need_vars.remove('LWin')

    #     # if wind was input as a scalar, don't need the other direction of wind
    #     if not self.wind_direction:
    #         if uwind_measured:
    #             self.cds['vwind'].values = np.zeros(self.n_time)
    #             need_vars.remove('vwind')
    #         elif vwind_measured:
    #             self.cds['uwind'].values = np.zeros(self.n_time)
    #             need_vars.remove('uwind')
    #     self.need_vars = need_vars
    #     return need_vars
    
    def get_data(self):
        """
        Fetches reanalysis climate data variables.
        """
        # load time and point data
        dates = self.dates_UTC
        sd = self.terrain
        
        # interpolate data if time was input on the hour instead of half-hour
        self.interpolate = dates[0].minute != 30 and self.args.climate_source == 'MERRA2'

        # get lat/lon DataArrays for indexing spatial data
        lat_n = sd.lat_n.get() if hasattr(sd.lat_n, 'get') else sd.lat_n
        lon_n = sd.lon_n.get() if hasattr(sd.lon_n, 'get') else sd.lon_n
        self.point_lats = xr.DataArray(lat_n, dims='point')
        self.point_lons = xr.DataArray(lon_n, dims='point')
        
        # get GCM data geopotential
        z_fp = self.GCM_fp + self.var_dict['elev']['fn']
        with xr.open_dataarray(z_fp) as zds:
            zds = zds.sel({self.lat_vn: self.point_lats, 
                           self.lon_vn: self.point_lons},method='nearest')
            zds = self.check_units('elev',zds)
            self.terrain.gcm_elev_n = zds.isel(time=0).values
        
        # loop through vars
        for var in self.need_vars:
            # gather data for each var and add to all_data
            region = 'gulkana' # 'reg' + str(self.args.rgi_region).zfill(2)
            fn = self.GCM_fp + self.var_dict[var]['fn'].format(r=region)
            if var == self.need_vars[0]:
                print('hard coded gulkana region for testing')
            self.get_var_data(fn, var)
        return
    
    def get_var_data(self, fn, var):
        # get dates
        dates = self.dates_UTC
        args = self.args

        # special check for RH: must be calculated from QV
        if var == 'rh' and not os.path.exists(fn):
            assert args.climate_source == 'MERRA2', 'RH conversion is only set up for MERRA2'
            self.create_rh2m_ds(fn)

        # special check for deposition variables
        dep_var = 'dry' in var or 'wet' in var

        # open the dataset for this variable
        if 'zarr' in fn:
            ds = xr.open_zarr(fn, decode_timedelta=False, consolidated=False)
        else:
            ds = xr.open_dataset(fn, decode_timedelta=False)

        # get variable names and lat/lon resolution
        vn = self.var_dict[var]['vn'] 
        lat_vn, lon_vn = (self.lat_vn, self.lon_vn)
        lat_res, lon_res = (args.merra_lat_res, args.merra_lon_res)

        # light-absorbing particles need special treatment
        if dep_var:
            if args.climate_source == 'ERA5-hourly':
                # tell lat/lon to use MERRA2 lat/lon names
                lat_vn,lon_vn = ['lat','lon']
            
            if args.deposition_data == 'UKESM' and var in self.carbon_vars:
                # tell lat/lon to use UKESM lat/lon names for BC/OC
                lat_vn,lon_vn = ['latitude','longitude']
                lat_res = np.diff(ds.isel({lat_vn:range(2)})[lat_vn].values)[0]
                lon_res = np.diff(ds.isel({lon_vn:range(2)})[lon_vn].values)[0]

                # convert longitude from 0-360 to -180-180 and re-sort
                ds = ds.assign_coords({lon_vn: ((ds[lon_vn] + 180) % 360) - 180})
                ds = ds.sortby(lon_vn)

        # index by lat and lon and select variable
        if lat_vn not in ds.dims or lon_vn not in ds.dims:
            # extract the single cell and replicate it for additional points
            n_points = len(self.point_lats)
            ds = ds[vn].expand_dims(point=n_points)
        else:
            # extract lats/lons across full dataset
            ds = ds.sel({lat_vn: self.point_lats, 
                         lon_vn: self.point_lons}, method='nearest')[vn]

        if var in self.carbon_vars and args.deposition_data:
            # turn daily into hourly
            ds = ds.resample(time='h').ffill()

        # check the units
        ds = self.check_units(var,ds)

        # for time-varying variables, select/interpolate to the model time
        if var != 'elev':
            assert dates[0] >= pd.to_datetime(ds.time.values[0]), f'Check input times and {var} data'
            assert dates[-1] <= pd.to_datetime(ds.time.values[-1]), f'Check input times and {var} data'
            if not dep_var and args.climate_source == 'ERA5-hourly':
                ds = ds.interp(time=dates)
            elif self.interpolate:
                ds = ds.interp(time=dates)
            else:
                ds = ds.sel(time=dates)
        
        # make sure the correct grid cell was accessed
        assert np.all(np.abs(ds.coords[lat_vn].values - self.point_lats.values) <= lat_res), 'Wrong grid cell was accessed'
        assert np.all(np.abs(ds.coords[lon_vn].values - self.point_lons.values) <= lon_res), 'Wrong grid cell was accessed'

        # store result
        data_array = ds.transpose('point','time').values
        setattr(self, var, data_array.astype(np.float32))
        ds.close()
        return
    
    def check_ds(self):
        """
        Calculates wind speed and direction from u and v,
        bias-corrects reanalysis data with quantile mapping,
        adjusts elevation-dependent variables, and
        checks that all required variables are filled.
        """
        args = self.args

        # calculate wind speed and direction from u and v components
        uwind = self.uwind
        vwind = self.vwind
        wind = np.sqrt(np.power(uwind,2)+np.power(vwind,2))
        winddir = np.arctan2(-uwind,-vwind) * 180 / np.pi
        self.wind = wind
        self.winddir = winddir

        # ****** FIGURE OUT QM
        # if args.climate_source == 'MERRA2':
        #     # correct MERRA-2 variables in bias_vars list
        #     self.bias_vars = [v for v in self.args.bias_vars if v not in self.measured_vars]
        #     if self.args.debug and len(self.bias_vars) > 0:
        #         print('~ Applying quantile mapping for:',self.bias_vars)
        #     for var in self.bias_vars:
        #         from_MERRA = True if not self.args.use_aws else var in self.need_vars
        #         if from_MERRA:
        #             self.bias_adjust_qm(var)
        
        # apply coefficients to adjust deposition
        if not args.deposition_data:
            # MERRA-2 representative bin --> total deposition
            self.apply_merra_dep_ratio()
        else:
            # UKESM --> MERRA-2
            self.apply_ukesm_dep_ratio()

        # apply climatic perturbations
        self.apply_perturbations()

        # check all variables are full
        failed = []
        for var in self.all_vars + self.optional_vars:
            if not hasattr(self, var ) and var in self.optional_vars:
                setattr(self, var, np.full(self.shape, np.nan))
                continue
            data = getattr(self, var)
            if np.any(np.isnan(data)) or data.shape != self.shape:
                failed.append(var)

        # can input net radiation instead of incoming LW radiation
        if 'LWin' in failed and 'NR' in self.measured_vars:
            failed.remove('LWin')

        # print any missing data
        if len(failed) > 0:
            failed_str = ', '.join(failed)
            raise ConfigError(f'Climate is missing data from {failed_str}')
        
        # done getting climate
        time_elapsed = time.time()-self.start_time
        if self.args.debug:
            print(f'~ Loaded climate data in {time_elapsed:.1f} seconds ~')
        return
    
    def apply_perturbations(self):
        """
        Applies additive temperature perturbation
        or multiplicative precipitation perturbation.
        """
        self.temp += self.args.temp_perturb
        self.tp *= self.args.tp_perturb
        return

    def adjust_to_elevation(self):
        """
        Adjusts elevation-dependent climate variables 
        (temperature, precip, surface pressure, and
        incoming longwave radiation).
        """
        # Set copies of un-edited variables
        self.original_temp = self.temp
        self.original_tp = self.tp
        self.original_sp = self.sp
        self.original_LWin = self.LWin

        # TEMPERATURE: correct according to lapse rate
        self.temp_to_elevation()
            
        # PRECIP: correct according to precipitation gradient
        self.precip_to_elevation()

        # SURFACE PRESSURE: correct according to barometric law
        self.sp_to_elevation()

        # LONGWAVE: correct with elevation-dependent emissivity 
        if self.args.temp_perturb > 0:
            # account for perturbed air temperature
            LAPSE_RATE = self.args.lapse_rate / 1000
            elev_change = LAPSE_RATE*(self.terrain.gcm_elev_n - self.temp_elev)
            temp_LW_elev = self.original_temp - self.args.temp_perturb + elev_change
            self.LWin_to_elevation(temp_LW_elev)
        else:
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
        args = self.args

        # CONSTANTS
        SPH = args.seconds_per_hour
        CTOK = args.celsius_to_kelvin
        GRAVITY = args.gravity
        DENSITY_WATER = args.density_water

        # define the units the model needs
        model_units = {'temp':'C','uwind':'m s-1','vwind':'m s-1',
                       'rh':'%','sp':'Pa','tp':'m s-1','elev':'m',
                       'SWin':'J m-2', 'LWin':'J m-2', 'NR':'J m-2', 'tcc':'1',
                       'bcdry':'kg m-2 s-1', 'bcwet':'kg m-2 s-1',
                       'ocdry':'kg m-2 s-1', 'ocwet':'kg m-2 s-1',
                       'dustdry':'kg m-2 s-1','dustwet':'kg m-2 s-1'}
        
        # get the current variable's units
        units_in = ds.attrs['units'].replace('*','')
        units_out = model_units[var]

        # check and make replacements for units mismatches
        if units_in != units_out:
            # TEMPERATURE
            if var == 'temp' and units_in == 'K':
                ds = ds - CTOK

            # RELATIVE HUMIDITY
            elif var == 'rh' and units_in in ['-','0-1']:
                ds  = ds * 100

            # PRECIPITATION
            elif var == 'tp':
                if units_in == 'kg m-2 s-1':
                    ds = ds / DENSITY_WATER * SPH
                elif units_in == 'm':
                    ds = ds / SPH

            # RADIATION
            elif var in ['SWin','LWin','NR'] and units_in == 'W m-2':
                ds = ds * SPH

            # ELEVATION
            elif var == 'elev' and units_in in ['m+2 s-2','m2 s-2']:
                ds = ds / GRAVITY

            # PARTICLE DEPOSITION 
            elif 'dry' in var or 'wet' in var:
                if units_in == 'm-2.kg.s-1':
                    pass

            # TOTAL CLOUD COVER
            elif var == 'tcc' and units_in in ['%']:
                ds = ds / 100

            # OTHER MISMATCH NOT LISTED
            else:
                print(f'WARNING! Units did not match for {var} but were not updated')
                print(f'Previously {units_in}; should be {units_out}')
                print('Make a manual change in check_units (climate.py)')
                raise ConfigError('Unit mismatch')
        return ds

    # UNUSED: from Coleman's work
    # def adjust_dep(self):
    #     """
    #     Updates deposition based on preprocessed 
    #     reduction coefficients
    #     """

    #     print('***Hard-coded MERRA-2 to UK-ESM filepath for deposition adjustment***')
    #     fn = self.GCM_fp + 'merra2_to_ukesm_conversion_map_MERRAgrid.nc'
    #     ds_f = xr.open_dataarray(fn)
    #     ds_f = ds_f.sel({self.lat_vn:self.lat,self.lon_vn:self.lon},method='nearest')
    #     f = ds_f.mean('time').values.ravel()[0]
    #     # To do time-moving monthly factors:
    #     # for date in ds_f.time.values:
    #     #     # select the reduction coefficient of the current month
    #     #     f = ds_f.sel(time=date).values[0]
    #     #     # index the climate dataset by the month and year
    #     #     month = pd.to_datetime(date).month
    #     #     year = pd.to_datetime(date).year
    #     #     idx_month = np.where(self.cds.coords['time'].dt.month.values == month)[0]
    #     #     idx_year = np.where(self.cds.coords['time'].dt.year.values == year)[0]
    #     #     idx = list(set(idx_month)&set(idx_year))
    #     #     # update dry and wet BC deposition
    #     #     self.cds['bcdry'][{'time':idx}] = self.cds['bcdry'][{'time':idx}] * f
    #     #     self.cds['bcwet'][{'time':idx}] = self.cds['bcwet'][{'time':idx}] * f
    #     self.bcdry *= f
    #     self.bcwet *= f
    #     return
    
    def apply_merra_dep_ratio(self):
        args = self.args
        RATIO_DUST = args.ratio_DU3_DUtot

        region = str(self.args.rgi_region).zfill(2)
        fn_bc = args.merra2_laps_fn.format(sp='BC', r=region)
        fn_oc = args.merra2_laps_fn.format(sp='OC', r=region)

        # open ratio dataset
        ds_bc = xr.open_dataset(args.climate_fp + fn_bc)
        ds_oc = xr.open_dataset(args.climate_fp + fn_oc)

        # select ratio at the correct lat/lon
        ratio_bc = (ds_bc['ratio'].sel(lat=self.point_lats, 
                                      lon=self.point_lons, 
                                      method='nearest')).values.reshape(-1, 1)
        ratio_oc = ds_oc['ratio'].sel(lat=self.point_lats, 
                                      lon=self.point_lons, 
                                      method='nearest').values.reshape(-1, 1)

        # apply to dry deposition
        self.bcdry *= ratio_bc
        self.ocdry *= ratio_oc
        self.dustdry *= RATIO_DUST
        self.dustwet *= RATIO_DUST

        # close datasets
        ds_bc.close()
        ds_oc.close()
        return
    
    def apply_ukesm_dep_ratio(self):
        args = self.args
        reg = self.args.rgi_id[:2]
        reg_fn = args.ukesm_merra_laps_fn.format(r=reg)

        for species in ['bc','oc']:
            for deptype in ['wet','dry']:
                fn = reg_fn.format(sp=species, t=deptype)
                
                # open ratio dataset
                ds_bc = xr.open_dataarray(args.climate_fp + fn)

                # select ratio at the correct lat/lon
                ratio = ds_bc.sel(lat=self.point_lats, 
                                  lon=self.point_lons, method='nearest').values

                # apply to climate dataset
                self.cds[species+deptype] *= ratio 

                # close the dataset
                ds_bc.close()
        return

    def temp_to_elevation(self):
        """
        Corrects air temperature at the site elevation
        based on a linear lapse rate
        """
        # CONSTANTS
        LAPSE_RATE = self.args.lapse_rate / 1000 # in K m-1

        # get elevation of the original temperature data
        if 'temp' in self.args.bias_vars and 'temp' not in self.measured_vars:
            # if temperature was a bias-corrected variable, use pre-set temp_elev
            temp_elev = self.temp_elev
        else:
            temp_elev = self.aws_elev if 'temp' in self.measured_vars else self.terrain.gcm_elev_n

        # format temp and point elev as (, n) arrays
        self.temp_elev = temp_elev[:, np.newaxis]
        point_elev = self.terrain.elev_n[:, np.newaxis]

        # apply lapse rate
        new_temp = self.original_temp + LAPSE_RATE*(point_elev - self.temp_elev)

        # update temperature in the cds
        self.temp = new_temp
        return

    def precip_to_elevation(self):
        """
        Corrects precipitation at the site elevation
        based on a % gradient
        """
        # CONSTANTS
        if self.args.glac_name in self.args.precgrads:
            PREC_GRAD = self.args.precgrads[self.args.glac_name]
        else:
            PREC_GRAD = self.args.precgrad

        # format tp and point elev as (, n) arrays
        tp_elev = self.terrain.median_elev_n[:, np.newaxis]
        point_elev = self.terrain.elev_n[:, np.newaxis]

        # apply precipitation gradient
        new_tp = self.original_tp*(1+PREC_GRAD*(point_elev-tp_elev))

        # update precip in the cds
        self.tp = new_tp
        return

    def sp_to_elevation(self):
        """
        Corrects surface pressure according to barometric law
        """
        # CONSTANTS
        LAPSE_RATE = self.args.lapse_rate / 1000 # in K m-1
        GRAVITY = self.args.gravity
        R_GAS = self.args.R_gas
        MM_AIR = self.args.molarmass_air
        CTOK = self.args.celsius_to_kelvin
        
        # get elevation of surface pressure data
        sp_elev = self.aws_elev if 'sp' in self.measured_vars else self.terrain.gcm_elev_n

        # format sp and point elev as (, n) arrays
        sp_elev = sp_elev[:, np.newaxis]
        point_elev = self.terrain.elev_n[:, np.newaxis]

        # adjust temperature from elevation of the site to elevation of the sp data
        new_temp = self.temp.copy()
        temp_sp_elev = new_temp + LAPSE_RATE*(sp_elev - point_elev) + CTOK

        # calculate new surface pressure with barometric law
        exponent = -GRAVITY*MM_AIR/(R_GAS*LAPSE_RATE)
        ratio = ((new_temp + CTOK) / temp_sp_elev) ** (exponent)
        new_sp = self.original_sp * ratio

        # update surface pressure array
        self.sp = new_sp
        return

    def LWin_to_elevation(self, temp_LW_elev=False):
        """
        Corrects incoming longwave by determining a
        theoretical difference in longwave under clear 
        sky conditions using the Brutsaert (1975) 
        parameterization of emissivity (based on 
        temperature and vapor pressure) and applying
        this difference to the MERRA-2 longwave data.

        If the temperature has been perturbed, the
        temperature fed here should be at the
        elevation of the MERRA-2 grid cell.
        """
        # CONSTANTS
        SIGMA_SB = self.args.sigma_SB
        LAPSE_RATE = self.args.lapse_rate / 1000 # in K m-1
        SPH = self.args.seconds_per_hour
        CTOK = self.args.celsius_to_kelvin

        # get temperature and RH data at the site and data location
        rh = self.rh                     # RH assumed constant with elevation
        temp_site = self.temp            # Temperature already updated to self.elev

        # get elevation of longwave data
        LW_elev = self.aws_elev if 'LWin' in self.measured_vars else self.terrain.gcm_elev_n
        if type(temp_LW_elev) == bool and not temp_LW_elev:
            temp_LW_elev = temp_site + LAPSE_RATE*(LW_elev - self.terrain.elev_n)[:, np.newaxis]

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
        self.LWin = new_LWin
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
        method = 'quantile_mapping'
        map_glac = self.mapping_glacier
        bias_fn = self.args.bias_fn.format(m=method, v=var, g=map_glac)

        # need to use file generated without a lapse rate for temperature
        if var == 'temp':
            bias_fn = bias_fn.replace('.csv','_0.csv')

        assert os.path.exists(bias_fn), f'Quantile mapping file does not exist for {var}'
        bias_df = pd.read_csv(bias_fn)
        
        # interpolate values according to quantile mapping
        values = getattr(self, var)
        adjusted = np.interp(values, bias_df['sorted'], bias_df['mapping'])

        # update values
        setattr(self, var, adjusted)
        return
    
    def precompute_upcoming_snow(self):
        temp = self.temp 
        tp = self.tp

        T_LOW = self.args.snow_threshold_low
        T_HIGH = self.args.snow_threshold_high

        rain_scale = np.linspace(1, 0, 20)
        temp_scale = np.linspace(T_LOW, T_HIGH, 20)
        snow_fraction = np.interp(temp, temp_scale, rain_scale)
        hourly_snow = tp * snow_fraction

        look_ahead_steps = int(self.args.new_snow_days * 24)
        
        # calculate the running total on time axis
        running_total = np.cumsum(hourly_snow, axis=1)

        # define storage for upcoming snow
        upcoming_snow = np.zeros_like(hourly_snow)
    
        # calculate snow upcoming in upcoming days
        upcoming_snow[:, :-look_ahead_steps] = (
            running_total[:, look_ahead_steps:] - \
                running_total[:, :-look_ahead_steps]
        )

        # for final hours where can't look into the future,
        # subtract the current toal from the absolute final total
        final_total = running_total[:, -1:]
        upcoming_snow[:, -look_ahead_steps:] = final_total - running_total[:, -look_ahead_steps:]
        
        self.upcoming_snow = upcoming_snow
        return
        
    def precompute_emulator_inputs(self):
        """
        Loads the SNICAR emulator inputs.
        """
        # sum dry and wet deposition
        bctot = self.bcwet + self.bcdry

        cds = xr.Dataset(
            data_vars={
                'bc': (['point', 'time'], bctot),
                'temp': (['point', 'time'], self.temp),
                'SWin': (['point', 'time'], self.SWin)
            },
            coords={
                'time': self.dates,
                'point': range(self.terrain.N_POINTS)
            }
        )

        # PDDs
        positive_temp_sum = (cds['temp']
                             .where(cds['temp']>0, other=0)
                             .resample({'time': '1d'})
                             .sum() / 24) # Celcius
        
        # rolling BC
        bc_rolling = (cds['bc']
                       .resample(time='1d').sum()
                       .rolling(time=5, min_periods=1).sum() * 3600) # kg m-2

        # incoming shortwave
        SWin = (cds['SWin']
                .resample(time='1d')
                .sum()) # J m-2
        
        # cumulative PDDs
        water_year = xr.where(
            positive_temp_sum['time.month'] >= 10, 
            positive_temp_sum['time.year'] + 1, 
            positive_temp_sum['time.year']
        )
        PDDs_cumsum = (positive_temp_sum.groupby(water_year).cumsum())

        self.SW_emulator_input = SWin.values
        self.PDD_emulator_input = PDDs_cumsum.values
        self.BC_emulator_input = bc_rolling.values
        return
    
    def create_rh2m_ds(self, fn):
        """
        Calculates 2-m relative humidity from
        specific humidity (set up for MERRA-2)

        Parameters
        ==========
        fn : str
            Filename of the RH2M dataset to create
        """
        # CONSTANTS
        CTOK = self.args.celsius_to_kelvin

        # get variable names
        rh_vn = self.var_dict['rh']['vn']
        temp_vn = self.var_dict['temp']['vn']
        sp_vn = self.var_dict['sp']['vn']
        qv_vn = 'QV2M'

        if '.nc' in fn:
            ds_qv = xr.open_dataset(fn.replace(rh_vn, qv_vn))
            ds_temp = xr.open_dataset(fn.replace(rh_vn, temp_vn))
            ds_sp = xr.open_dataset(fn.replace(rh_vn, sp_vn))
        elif '.zarr' in fn:
            ds_qv = xr.open_zarr(fn.replace(rh_vn, qv_vn), consolidated=False)
            ds_temp = xr.open_zarr(fn.replace(rh_vn, temp_vn), consolidated=False)
            ds_sp = xr.open_zarr(fn.replace(rh_vn, sp_vn), consolidated=False)

        # calculate saturation pressure from air temperature
        esat = self.sat_vapor_pressure(ds_temp[temp_vn].values - CTOK)

        # saturation and actual specific humidity vapor pressure
        ws = 0.622 * esat / (ds_sp[sp_vn].values - esat)
        w = ds_qv[qv_vn].values / (1 - ds_qv[qv_vn].values)

        # relative humidity as a percentage of saturation humidity
        rh = w / ws * 100

        # create copy dataset and fill with RH data
        ds_rh = ds_qv.copy(deep=True)
        ds_rh['RH2M'] = (ds_qv[qv_vn].dims, rh)
        ds_rh['RH2M'].attrs = {
            'units': '%', 
            'long_name': '2-meter_relative_humidity'
        }

        # close datasets
        ds_qv.close()
        ds_temp.close()
        ds_sp.close()

        # drop QV data and store the RH dataset
        ds_rh = ds_rh.drop_vars('QV2M')

        # clear encoding
        ds_rh.encoding = {}
        for var in ds_rh.variables: ds_rh[var].encoding = {}

        if 'zarr' in fn:
            import zarr
            ds_rh.to_zarr(fn, mode='w', consolidated=False)
            store = zarr.DirectoryStore(fn)
            zarr.consolidate_metadata(store)
        else:
            ds_rh.to_netcdf(fn)
        if self.args.debug:
            print(f'RH dataset created at {fn}')
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
        CTOK = self.args.celsius_to_kelvin

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
        CTOK = self.args.celsius_to_kelvin

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
        args = self.args 

        # update filenames for MERRA-2 (need grid lat/lon)
        self.GCM_fp = args.climate_fp
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
        if args.climate_source == 'MERRA2':
            self.GCM_fp += 'MERRA2/'
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
            self.var_dict['temp']['fn'] = '{r}/T2M_{r}.zarr'
            self.var_dict['rh']['fn'] = '{r}/RH2M_{r}.zarr'
            self.var_dict['sp']['fn'] = '{r}/PS_{r}.zarr'
            self.var_dict['tcc']['fn'] = '{r}/CLDTOT_{r}.zarr'
            self.var_dict['LWin']['fn'] = '{r}/LWGAB_{r}.zarr'
            self.var_dict['SWin']['fn'] = '{r}/SWGDN_{r}.zarr'
            self.var_dict['vwind']['fn'] = '{r}/V2M_{r}.zarr'
            self.var_dict['uwind']['fn'] = '{r}/U2M_{r}.zarr'
            self.var_dict['tp']['fn'] = '{r}/PRECTOTCORR_{r}.zarr'
            self.var_dict['elev']['fn'] = 'MERRA2_constants.nc'
            self.var_dict['bcwet']['fn'] = '{r}/BCWT002_{r}.zarr'
            self.var_dict['bcdry']['fn'] = '{r}/BCDP002_{r}.zarr'
            self.var_dict['ocwet']['fn'] = '{r}/OCWT002_{r}.zarr'
            self.var_dict['ocdry']['fn'] = '{r}/OCDP002_{r}.zarr'
            self.var_dict['dustwet']['fn'] = '{r}/DUWT003_{r}.zarr'
            self.var_dict['dustdry']['fn'] = '{r}/DUDP003_{r}.zarr'
        elif args.climate_source == 'ERA5-hourly':
            self.GCM_fp += 'ERA5/ERA5_hourly/'

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

        # functionality for independent deposition datasets
        if args.deposition_data:
            if args.deposition_data == 'UKESM':
                # Define names used in UKESM data
                sp_oc = 'particulate_organic_matter'
                sp_bc = 'elemental_carbon'
                vn = args.ukesm_vn

                # Variable names 
                self.var_dict['bcwet']['vn'] = vn.format(sp=sp_bc, t='wet')
                self.var_dict['bcdry']['vn'] = vn.format(sp=sp_bc, t='dry')
                self.var_dict['ocwet']['vn'] = vn.format(sp=sp_oc, t='wet')
                self.var_dict['ocdry']['vn'] = vn.format(sp=sp_oc, t='dry')

                # Variable filenames
                fp = args.ukesm_fp
                self.var_dict['bcwet']['fn'] = fp + args.ukesm_fn.format(sp='bc',t='wet')
                self.var_dict['bcdry']['fn'] = fp + args.ukesm_fn.format(sp='bc',t='dry')
                self.var_dict['ocwet']['fn'] = fp + args.ukesm_fn.format(sp='oc',t='wet')
                self.var_dict['ocdry']['fn'] = fp + args.ukesm_fn.format(sp='oc',t='dry')