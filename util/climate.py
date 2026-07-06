"""
Climate class for PEBSI

Loads the climate dataset and processes it
for the simulation, including:
- Bias correction (i.e., quantile mapping)
- Elevation adjustments
- Spatial adjustments (e.g., precipitation factor)
- Perturbations (e.g., additive temperature factor)

@author: clairevwilson
"""
# Built-in libraries
import os
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
    def __init__(self, dates, params, terrain):
        """
        Initializes glacier information and creates
        the dataset where climate data will be stored.
        """
        # start timer for loading climate data
        self.start_time = time.time()

        # load params and simulation information
        self.params = params
        self.terrain = terrain
        self.get_spatial_temporal_info(dates)

        # list all required variables
        self.all_vars = ['temp','tp','rh','uwind','vwind','sp','SWin','LWin',
                            'bcwet','bcdry','ocwet','ocdry','dustwet','dustdry']
        self.optional_vars = ['SWout','LWout','tcc','NR','albedo']
        self.carbon_vars = ['bcwet','bcdry','ocwet','ocdry']

        # create dictionary containing reanalysis filenames
        self.get_vardict()
        self.bias_vars = []
        if not self.params.use_aws:
            self.measured_vars = []
            self.need_vars = self.all_vars.copy()
        else:
            self.measured_vars = []
            self.need_vars = []
            self.get_aws()
        return
    
    def get_spatial_temporal_info(self, dates):
        """
        Loads metadata about the points and 
        dates in the simulation.
        """
        self.dates = pd.to_datetime(dates).to_numpy()
            
        # specify spatial and temporal information
        self.N_TIME = len(self.dates)
        self.shape = (self.terrain.N_POINTS, self.N_TIME)
        return
    
    def get_aws(self):
        """
        Loads available AWS data and determines which
        variables need come from reanalysis data.

        All variables present in the AWS .csv are broadcast
        across all simulation points (same value everywhere).
        AWS variables are then elevation adjusted using
        self.aws_elev.
        """
        # load data
        df = pd.read_csv(self.params.aws_fn, index_col=0, parse_dates=True)

        # check dates of data match input dates
        data_start = pd.to_datetime(df.index.to_numpy()[0])
        data_end = pd.to_datetime(df.index.to_numpy()[-1])
        assert self.dates[0] >= data_start, \
            f'Check input dates: start date before range of AWS data ({data_start})'
        assert self.dates[len(self.dates)-1] <= data_end, \
            f'Check input dates: end date after range of AWS data ({data_end})'
        df = df.loc[self.dates]

        # store AWS elevation as a (N_POINTS,) array for elevation adjustment methods
        assert self.params.aws_elev is not None, 'aws_elev must be set in config when use_aws=True'
        self.aws_elev = np.full(self.terrain.N_POINTS, self.params.aws_elev)

        # get the available variables (intersection of data columns and known model vars)
        all_aws_vars = self.all_vars + self.optional_vars
        aws_vars = df.columns
        self.measured_vars = list(set(all_aws_vars) & set(aws_vars))

        # check if wind direction can be calculated from u/v components
        uwind_measured = 'uwind' in aws_vars
        vwind_measured = 'vwind' in aws_vars
        if uwind_measured ^ vwind_measured:
            self.wind_direction = False
        else:
            self.wind_direction = True

        # broadcast each AWS time series to (N_POINTS, N_TIME)
        for var in self.measured_vars:
            time_series = df[var].astype(float).values               # (N_TIME,)
            data = np.tile(time_series, (self.terrain.N_POINTS, 1))  # (N_POINTS, N_TIME)
            setattr(self, var, data)

        # determine which variables are still needed from reanalysis
        need_vars = [e for e in self.all_vars if e not in aws_vars]

        # if net radiation was measured, don't need LWin
        if 'NR' in self.measured_vars:
            need_vars.remove('LWin')

        # if wind was input as a scalar, fill the missing component with zeros
        if not self.wind_direction:
            if uwind_measured:
                self.vwind = np.zeros(self.shape)
                need_vars.remove('vwind')
            elif vwind_measured:
                self.uwind = np.zeros(self.shape)
                need_vars.remove('uwind')

        self.need_vars = need_vars
        return need_vars
    
    def get_data(self):
        """
        Loads all the raw climate data.
        """
        # load time and point data
        terrain = self.terrain
        
        # get lat/lon DataArrays for indexing spatial data
        lat_n = terrain.lat_n
        lon_n = terrain.lon_n
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
            region = 'reg' + str(self.params.rgi_region).zfill(2)
            fn = self.GCM_fp + self.var_dict[var]['fn'].format(r=region)
            
            self.get_var_data(fn, var)
        return
    
    def get_var_data(self, fn, var):
        """
        Loads reanalysis data for a single variable.
        """
        # get dates
        dates = self.dates
        params = self.params

        # special check for RH: must be calculated from QV
        if var == 'rh' and not os.path.exists(fn):
            assert params.climate_source == 'MERRA2', 'RH conversion is only set up for MERRA2'
            self.create_rh2m_ds(fn)

        # special check for deposition variables
        dep_var = 'dry' in var or 'wet' in var
        non_merra_dep_var = (var in self.carbon_vars) and (params.deposition_data)

        # open the dataset for this variable
        if 'zarr' in fn:
            ds = xr.open_zarr(fn, decode_timedelta=False, consolidated=False, chunks={})
        else:
            ds = xr.open_dataset(fn, decode_timedelta=False)

        # get variable names and lat/lon resolution
        vn = self.var_dict[var]['vn'] 
        lat_vn, lon_vn = (self.lat_vn, self.lon_vn)
        lat_res, lon_res = (params.merra_lat_res, params.merra_lon_res)

        # light-absorbing particles need special treatment
        if dep_var:
            if params.climate_source == 'ERA5-hourly':
                # tell lat/lon to use MERRA2 lat/lon names
                lat_vn,lon_vn = ['lat','lon']
            
            if params.deposition_data == 'UKESM' and non_merra_dep_var:
                # tell lat/lon to use UKESM lat/lon names for BC/OC
                lat_vn,lon_vn = ['latitude','longitude']
                lat_res = np.diff(ds.isel({lat_vn:range(2)})[lat_vn].values)[0]
                lon_res = np.diff(ds.isel({lon_vn:range(2)})[lon_vn].values)[0]

                # convert longitude from 0-360 to -180-180 and re-sort
                ds = ds.assign_coords({lon_vn: ((ds[lon_vn] + 180) % 360) - 180})
                ds = ds.sortby(lon_vn)

        # check the units
        da = self.check_units(var, ds[vn])

        # slice time to chunk window before spatial sel to minimize data pulled
        N_TIME = len(dates)
        N_POINTS = len(self.point_lats)
        data_array = np.empty((N_POINTS, N_TIME), dtype=np.float64)

        # time bounds
        t_start = dates[0]
        t_end = dates[-1]

        # MERRA-2 data is on the half hour; shift timestamps up
        if self.params.climate_source == 'MERRA2' and not non_merra_dep_var:
            t_start = t_start + pd.Timedelta(minutes=30)
            t_end = t_end + pd.Timedelta(minutes=30)

        # slice by time
        if non_merra_dep_var:
            # non-MERRA2 deposition data can be daily; forward fill to hourly
            # shift timestamps back to midnight
            shifted = da.assign_coords(time=da['time'] - pd.Timedelta(hours=12))

            # add a day at the end to make sure all hours are there
            last = pd.Timestamp(shifted['time'].values[-1]) + pd.Timedelta(hours=23)
            new_index = pd.date_range(shifted['time'].values[0], last, freq='h')
            hourly = shifted.reindex(time=new_index, method='ffill')

            da_sliced = hourly.sel(time=slice(t_start, t_end))
        else:
            da_sliced = da.sel(time=slice(t_start, t_end))

        # make sure the time is actually there
        assert t_start in da_sliced.time.values, f'Dates out of range: {t_start}'
        assert t_end in da_sliced.time.values, f'Dates out of range: {t_end}'

        # slice by lat and lon points
        if lat_vn not in da_sliced.dims or lon_vn not in da_sliced.dims:
            da_sliced = da_sliced.expand_dims(point=N_POINTS)
        else:
            da_sliced = da_sliced.sel(
                {lat_vn: self.point_lats, lon_vn: self.point_lons},
                method='nearest')

        data_array[:] = da_sliced.transpose('point', 'time').values

        # make sure the correct grid cells were accessed
        lat_check = np.all(np.abs(da_sliced.coords[lat_vn].values - self.point_lats.values) <= lat_res)
        lon_check = np.all(np.abs(da_sliced.coords[lon_vn].values - self.point_lons.values) <= lon_res)
        bbox = (self.point_lats.min().values, self.point_lats.max().values,
                self.point_lons.min().values, self.point_lons.max().values)
        assert lat_check & lon_check, \
            f'Wrong grid cell was accessed: climate data may not cover whole region ({bbox})'

        # store result
        setattr(self, var, data_array.astype(np.float64))
        return
    
    def process_climate(self):
        """
        Processes raw climate data into the format expected
        by the model.

        - Calculates wind speed and direction vectors from u and v
        - Corrects biases using quantile mapping
        - Adjusts elevation-dependent variables
        - Validates that all variables are filled
        """
        params = self.params

        # calculate wind speed and direction from u and v components
        uwind = self.uwind
        vwind = self.vwind
        wind = np.sqrt(np.power(uwind,2)+np.power(vwind,2))
        winddir = np.arctan2(-uwind,-vwind) * 180 / np.pi
        self.wind = wind
        self.winddir = winddir

        # bias correction
        if params.climate_source == 'MERRA2':
            # do not adjust variables that were measured
            self.bias_vars = [v for v in self.params.bias_vars if v not in self.measured_vars]

            for var in self.bias_vars:
                self.bias_correct_qm(var)
        
        # apply coefficients to adjust deposition
        if not params.deposition_data:
            # MERRA-2 representative bin --> total deposition
            self.apply_merra_dep_ratio()
        else:
            # UKESM --> MERRA-2
            self.apply_ukesm_dep_ratio()

        # apply climatic perturbations
        self.apply_perturbations()

        # apply parameters (precipitation / wind / dust factors)
        self.apply_parameters()

        # apply elevation correction
        self.adjust_to_elevation()

        # check all required variables are full
        failed = []
        for var in self.all_vars:
            data = getattr(self, var)
            if np.any(np.isnan(data)) or data.shape != self.shape:
                failed.append(var)

        # optional variables: fill with NaN if absent or invalid
        for var in self.optional_vars:
            if not hasattr(self, var):
                setattr(self, var, np.full(self.shape, np.nan))
            else:
                data = getattr(self, var)
                if data.shape != self.shape:
                    setattr(self, var, np.full(self.shape, np.nan))

        # can input net radiation instead of incoming LW radiation
        if 'LWin' in failed and 'NR' in self.measured_vars:
            failed.remove('LWin')

        # print any missing data
        if len(failed) > 0:
            failed_str = ', '.join(failed)
            for var in failed:
                data = getattr(self, var)
            raise ConfigError(f'Climate is missing data from {failed_str}')
        
        return
    
    def check_units(self,var,ds):
        """
        Checks the units for a meteorological
        variable and puts them in the correct units.
        Takes in the raw dataset from MERRA-2.
        """
        params = self.params

        # CONSTANTS
        SPH = params.seconds_per_hour
        CTOK = params.celsius_to_kelvin
        GRAVITY = params.gravity
        DENSITY_WATER = params.density_water

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
    
    def apply_merra_dep_ratio(self):
        """
        Applies pre-computed factors to adjust a
        representative bin (e.g., dust size bin 3)
        to total deposition of a given particle type.
        """
        params = self.params
        RATIO_DUST = params.ratio_DU3_DUtot

        # load the pre-computed maps of BC/OC factors
        region = str(self.params.rgi_region).zfill(2)
        fn_bc = params.merra2_laps_fn.format(sp='BC', r=region)
        fn_oc = params.merra2_laps_fn.format(sp='OC', r=region)

        ds_bc = xr.open_dataset(params.climate_fp + fn_bc)
        ds_oc = xr.open_dataset(params.climate_fp + fn_oc)

        # select ratio at the correct lats/lons
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
        """
        Applies pre-computed factors to adjust
        UK-ESM deposition data to MERRA-2.
        """
        params = self.params
        reg = str(self.params.rgi_region).zfill(2)
        lap_fn = params.ukesm_merra_laps_fn

        for species in ['bc','oc']:
            for deptype in ['wet','dry']:
                fn = lap_fn.format(r=reg, sp=species, t=deptype)
                
                # open ratio dataset
                ds_bc = xr.open_dataarray(params.climate_fp + fn)

                # select ratio at the correct lat/lon
                ratio = ds_bc.sel(lat=self.point_lats, 
                                  lon=self.point_lons, method='nearest').values

                # apply to data
                data = getattr(self, species+deptype)
                data *= ratio[:, np.newaxis]
                setattr(self, species+deptype, data)

                # close the dataset
                ds_bc.close()

        return
    
    def apply_perturbations(self):
        """
        Applies additive temperature perturbation
        or multiplicative precipitation perturbation.
        """
        self.temp += self.params.temp_perturb
        self.tp *= self.params.tp_perturb
        return
    
    def apply_parameters(self):
        """
        Applies climate parameters.
        """
        self.tp *= self.params.kp[:, None]
        self.wind *= self.params.wind_factor[:, None]
        self.dustdry *= self.params.dust_factor[:, None]
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
        if self.params.temp_perturb > 0:
            # account for perturbed air temperature
            lapse_rate = self.params.lapse_rate / 1000
            elev_change = lapse_rate*(self.terrain.gcm_elev_n - self.temp_elev)
            temp_LW_elev = self.original_temp - self.params.temp_perturb + elev_change
            self.LWin_to_elevation(temp_LW_elev)
        else:
            self.LWin_to_elevation()
        return

    def temp_to_elevation(self):
        """
        Corrects air temperature at the site elevation
        using a linear lapse rate.
        """
        # CONSTANTS
        lapse_rate = self.params.lapse_rate / 1000 # in K m-1

        # get elevation of the original temperature data
        if 'temp' in self.params.bias_vars and 'temp' not in self.measured_vars:
            # if temperature was a bias-corrected variable, temp_elev was already set
            temp_elev = self.temp_elev
        else:
            temp_elev = self.aws_elev if 'temp' in self.measured_vars else self.terrain.gcm_elev_n

        # format temp and point elev as (, n) arrays
        self.temp_elev = temp_elev[:, np.newaxis]
        point_elev = self.terrain.elev_n[:, np.newaxis]

        # apply lapse rate
        new_temp = self.original_temp + lapse_rate*(point_elev - self.temp_elev)

        # update temperature in the cds
        self.temp = new_temp
        return

    def precip_to_elevation(self):
        """
        Corrects precipitation at the site elevation
        using a precipitation gradient in % / m.
        """
        # CONSTANTS
        prec_grad = self.params.precgrad

        # format tp and point elev as (, n) arrays
        tp_elev = self.terrain.median_elev_n[:, np.newaxis]
        point_elev = self.terrain.elev_n[:, np.newaxis]

        # apply precipitation gradient
        new_tp = self.original_tp*(1+prec_grad*(point_elev-tp_elev))

        # update precip in the cds
        self.tp = new_tp
        return

    def sp_to_elevation(self):
        """
        Corrects surface pressure according to barometric law.
        """
        # CONSTANTS
        lapse_rate = self.params.lapse_rate / 1000 # in K m-1
        GRAVITY = self.params.gravity
        R_GAS = self.params.R_gas
        MM_AIR = self.params.molarmass_air
        CTOK = self.params.celsius_to_kelvin
        
        # get elevation of surface pressure data
        sp_elev = self.aws_elev if 'sp' in self.measured_vars else self.terrain.gcm_elev_n

        # format sp and point elev as (, n) arrays
        sp_elev = sp_elev[:, np.newaxis]
        point_elev = self.terrain.elev_n[:, np.newaxis]

        # adjust temperature from elevation of the site to elevation of the sp data
        new_temp = self.temp.copy()
        temp_sp_elev = new_temp + lapse_rate*(sp_elev - point_elev) + CTOK

        # calculate new surface pressure with barometric law
        exponent = -GRAVITY*MM_AIR/(R_GAS*lapse_rate)
        ratio = ((new_temp + CTOK) / temp_sp_elev) ** (exponent)
        new_sp = self.original_sp * ratio

        # update surface pressure array
        self.sp = new_sp
        return

    def LWin_to_elevation(self, temp_LW_elev=False):
        """
        Corrects incoming longwave to point elevations.
         
        Determines a theoretical difference in longwave 
        radiation between the air temperature at the point 
        and air temperature at the elevation of the longwave
        data. Uses the Brutsaert (1975) parameterization for
        clear-sky emissivity. This difference is then applied
        to the raw longwave data.

        With a temperature perturbation, the "temp_LW_elev" 
        should first be adjusted back to the elevation of the 
        MERRA-2 grid cell (this is done in adjust_to_elevation).
        """
        # CONSTANTS
        SIGMA_SB = self.params.sigma_SB
        lapse_rate = self.params.lapse_rate / 1000 # in K m-1
        SPH = self.params.seconds_per_hour
        CTOK = self.params.celsius_to_kelvin

        # get temperature and RH data at the site and data location
        rh = self.rh                     # RH assumed constant with elevation
        temp_site = self.temp            # Temperature already updated to self.elev

        # get elevation of longwave data
        LW_elev = self.aws_elev if 'LWin' in self.measured_vars else self.terrain.gcm_elev_n
        if type(temp_LW_elev) == bool and not temp_LW_elev:
            temp_LW_elev = temp_site + lapse_rate*(LW_elev - self.terrain.elev_n)[:, np.newaxis]

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
    
    def bias_correct_qm(self,var):
        """
        Applies preprocessed quantile mapping to
        reanalysis climate data for a single variable.
        """
        # open the dataset for this group
        ds = xr.open_dataset(self.params.bias_fn, group=var)

        # load lats/lons from quantile mapping data and simulation points
        data_lat = ds.lat.values[None, :]
        glacier_lat = self.terrain.lat_n[:, None]
        data_lon = ds.lon.values[None, :]
        glacier_lon = self.terrain.lon_n[:, None]

        # find nearest station to each grid cell
        coslat = np.cos(np.deg2rad(glacier_lat))
        dist = ((data_lat - glacier_lat))**2 + ((data_lon - glacier_lon) * coslat)**2
        station_idx = dist.argmin(axis=1)          # (N_POINTS,) which station each cell uses

        # load data
        to_correct = getattr(self, var)
        corrected = np.empty_like(to_correct, dtype=float)

        # storage for elevation (needed for temp)
        temp_elev = np.empty(self.terrain.N_POINTS)

        # load CDF from the pre-processed quantile mapping data
        merra_cdf = ds.merra_cdf.values     # (N_STATIONS, N_QUANTILES)
        aws_cdf = ds.aws_cdf.values         # (N_STATIONS, N_QUANTILES)
        elevation = ds.elevation.values     # (N_STATIONS, )

        # loop through unique stations
        for s in np.unique(station_idx):
            m = station_idx == s
            corrected[m] = np.interp(to_correct[m], merra_cdf[s], aws_cdf[s])
            temp_elev[m] = elevation[s]

            # make sure zeros stay as zeros in shortwave radiation
            if var == 'SWin':
                corrected[m] = np.wnere(to_correct[m] < 5, 0, corrected[m])

        if var == 'temp':
            self.temp_elev = temp_elev

        # update values
        setattr(self, var, corrected)

        ds.close()
        return
    
    def create_rh2m_ds(self, fn):
        """
        Calculates 2-m relative humidity in % from
        specific humidity in kg kg-1 and stores the 
        dataset to the passed fn.

        Only set up for MERRA-2.
        """
        print('~ Did not find RH2M data: calculating . . .')
        # CONSTANTS
        CTOK = self.params.celsius_to_kelvin

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
            ds_qv = xr.open_zarr(fn.replace(rh_vn, qv_vn), consolidated=False).chunk({'time': 1000})
            ds_temp = xr.open_zarr(fn.replace(rh_vn, temp_vn), consolidated=False).chunk({'time': 1000})
            ds_sp = xr.open_zarr(fn.replace(rh_vn, sp_vn), consolidated=False).chunk({'time': 1000})

        # calculate saturation pressure from air temperature
        esat = self.sat_vapor_pressure(ds_temp[temp_vn] - CTOK)

        # saturation and actual specific humidity vapor pressure
        ws = 0.622 * esat / (ds_sp[sp_vn] - esat)
        w = ds_qv[qv_vn] / (1 - ds_qv[qv_vn])

        # relative humidity as a percentage of saturation humidity
        rh = w / ws * 100

        # create copy dataset and fill with RH data
        ds_rh = ds_qv.copy(deep=True)
        ds_rh[rh_vn] = rh
        ds_rh[rh_vn].attrs = {
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
            n_time = len(ds_qv.time)
            time_chunk = 1000

            first_chunk = ds_rh.isel(time=slice(0, time_chunk)).compute()
            first_chunk.to_zarr(fn, mode='w', consolidated=False)
            del first_chunk

            # append remaining chunks
            for i in range(time_chunk, n_time, time_chunk):
                chunk = ds_rh.isel(time=slice(i, i + time_chunk)).compute()
                chunk.drop_vars([v for v in chunk.coords if 'time' not in chunk[v].dims])
                chunk.to_zarr(fn, append_dim='time', consolidated=False)
                del chunk

            zarr.consolidate_metadata(fn)
        else:
            ds_rh.to_netcdf(fn)
        if self.params.debug:
            print(f'RH dataset created at {fn}')
        return

    def sat_vapor_pressure(self,airtemp,method='ARM'):
        """
        Takes in air temperature in C and returns
        saturation vapor pressure in Pa
        """
        # CONSTANTS
        CTOK = self.params.celsius_to_kelvin

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
        Takes in air temperature in C and relative humidity
        in % and returns Brutsaert (1975) clear-sky atmospheric
        emissivity (unitless).
        """
        # CONSTANTS
        CTOK = self.params.celsius_to_kelvin

        # get saturation vapor pressure
        esat = self.sat_vapor_pressure(airtemp)

        # convert to actual vapor pressure (in hPa)
        e_hPa = esat * (rh / 100) / 100

        return 1.24 * (e_hPa / (airtemp + CTOK)) ** (1/7)
    
    def dew_point(self,vap):
        """
        Returns dewpoint air temperature in K from 
        vapor pressure in Pa.

        """
        return 243.04*np.log(vap/610.94)/(17.625-np.log(vap/610.94))

    def get_vardict(self):
        """
        Fills a dictionary with the reanalysis file 
        and variable names.
        """
        params = self.params 
        self.GCM_fp = params.climate_fp

        # variables every climate source must define
        data_vars = [
            'temp', 'rh', 'sp', 'tp', 'tcc', 'SWin', 'LWin',
            'uwind', 'vwind', 'bcdry', 'bcwet', 'ocdry', 'ocwet',
            'dustdry', 'dustwet', 'elev',
        ]
        coord_vars = ['time', 'lat', 'lon']

        # empty skeleton: guarantees every key exists for any source
        self.var_dict = {v: {'fn': '', 'vn': ''} for v in data_vars + coord_vars}

        if params.climate_source == 'MERRA2':
            self.GCM_fp += 'MERRA2/'

            merra2_vn = {
                'temp': 'T2M', 'rh': 'RH2M', 'sp': 'PS',
                'tp': 'PRECTOTCORR', 'tcc': 'CLDTOT',
                'SWin': 'SWGDN', 'LWin': 'LWGAB',
                'uwind': 'U2M', 'vwind': 'V2M',
                'bcwet': 'BCWT002', 'bcdry': 'BCDP002',
                'ocwet': 'OCWT002', 'ocdry': 'OCDP002',
                'dustwet': 'DUWT003', 'dustdry': 'DUDP003',
            }
            for var, vn in merra2_vn.items():
                self.var_dict[var] = {'vn': vn, 'fn': f'{{r}}/{vn}_{{r}}.zarr'}
            self.var_dict['elev'] = {'vn': 'PHIS', 'fn': 'MERRA2_constants.nc'}

            for coord in coord_vars:
                self.var_dict[coord]['vn'] = coord

            self.time_vn, self.lat_vn, self.lon_vn = 'time', 'lat', 'lon'
            self.elev_vn = self.var_dict['elev']['vn']

        # elif params.climate_source == 'ERA5':
        #     self.GCM_fp += 'ERA5/'
        #     # fill the same keys with ERA5 names/filenames here
        #     ...

        else:
            raise ValueError(f'Unsupported climate_source: {params.climate_source}')

        # functionality for independent deposition datasets
        if params.deposition_data:
            if params.deposition_data == 'UKESM':
                # define names used in UKESM data
                sp_oc = 'particulate_organic_matter'
                sp_bc = 'elemental_carbon'
                vn = params.ukesm_vn

                # variable names 
                self.var_dict['bcwet']['vn'] = vn.format(sp=sp_bc, t='wet')
                self.var_dict['bcdry']['vn'] = vn.format(sp=sp_bc, t='dry')
                self.var_dict['ocwet']['vn'] = vn.format(sp=sp_oc, t='wet')
                self.var_dict['ocdry']['vn'] = vn.format(sp=sp_oc, t='dry')

                # variable filenames
                fp = params.ukesm_fp
                self.var_dict['bcwet']['fn'] = fp + params.ukesm_fn.format(sp='bc',t='wet')
                self.var_dict['bcdry']['fn'] = fp + params.ukesm_fn.format(sp='bc',t='dry')
                self.var_dict['ocwet']['fn'] = fp + params.ukesm_fn.format(sp='oc',t='wet')
                self.var_dict['ocdry']['fn'] = fp + params.ukesm_fn.format(sp='oc',t='dry')