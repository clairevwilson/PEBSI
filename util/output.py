# Built-in libraries
import os
# External libraries
import numpy as np
import pandas as pd
import xarray as xr
# Enable GPU/CPU fluidity
try:
    import cupy as cp 
    xp = cp
except:
    xp = np

class Output():
    """
    Output class which stores the data during the
    simulation and saves it to a netcdf file upon
    run completion.
    """
    def __init__(self, time, params, terrain):
        """
        Creates netcdf file where the model output 
        will be saved.

        Parameters
        ==========
        time : list-like
            List of times used in the simulation
        params : command-line params
        """
        self.params = params
        self.terrain = terrain

        # define all the variables to be saved
        all_variables = {
            'EB':['shortwave_in','shortwave_ref','longwave_in','longwave_out',
                  'rain_heat','ground_heat', 'sensible_heat','latent_heat',
                   'melt_energy','albedo','surftemp'],
            'MB':['melt','refreeze','runoff','cumrefreeze','dh',
                  'sublimation','deposition','evaporation','condensation',
                  'accumulation','rainfall','error'],
            'layers':['layertemp','layerdensity','layerwater','layerheight',
                      'layerage','layertype','layergrainsize','layerrefreeze',
                      'layerBC','layerOC','layerdust'],
            'SW':['vis_albedo','SWin_sky','SWin_terr'],
            'climate':['airtemp','rh','wind','winddir','sp','tp']
        }

        # associate each 1D variable with an index
        self.var_idx = {}
        i = 0
        for otype in ['EB','MB','SW','climate']:
            for var in all_variables[otype]:
                self.var_idx[var] = i
                i += 1 

        # associate each 2D variable with an index
        self.layer_idx = {}
        j = 0
        for var in all_variables['layers']:
            self.layer_idx[var] = j 
            j += 1

        # make sure the requested storage variables are valid
        for v in params.store_vars: assert v in all_variables, f'Invalid output group: {v}'

        # extract the actual variables to store
        self.store = [v for g in params.store_vars for v in all_variables[g]]

        # generate dummy variables of the correct shape
        N_TIME = self.n_timesteps = len(time)
        N_LAYERS = self.max_nlayers = params.max_nlayers
        N_POINTS = self.n_points = terrain.N_POINTS
        
        # Keep tracking variables
        initial_height = params.initial_ice_depth + params.initial_firn_depth + params.initial_snow_depth
        self.last_height = initial_height
        
        # create file to store outputs
        zeros = np.zeros([N_TIME, N_LAYERS])
        ds_template = xr.Dataset(data_vars = dict(
                # ENERGY BALANCE
                surftemp = (['time'],zeros[:,0],{'units':'C'}),
                shortwave_in = (['time'],zeros[:,0],{'units':'W m-2'}),
                shortwave_ref = (['time'],zeros[:,0],{'units':'W m-2'}),
                longwave_in = (['time'],zeros[:,0],{'units':'W m-2'}),
                longwave_out = (['time'],zeros[:,0],{'units':'W m-2'}),
                rain_heat = (['time'],zeros[:,0],{'units':'W m-2'}),
                ground_heat = (['time'],zeros[:,0],{'units':'W m-2'}),
                sensible_heat = (['time'],zeros[:,0],{'units':'W m-2'}),
                latent_heat = (['time'],zeros[:,0],{'units':'W m-2'}),
                melt_energy = (['time'],zeros[:,0],{'units':'W m-2'}),
                albedo = (['time'],zeros[:,0],{'units':'0-1'}),

                # # SHORTWAVE DETAILED
                # vis_albedo = (['time'],zeros[:,0],{'units':'0-1'}),
                # SWin_sky = (['time'],zeros[:,0],{'units':'W m-2'}),
                # SWin_terr = (['time'],zeros[:,0],{'units':'W m-2'}),

                # MASS BALANCE
                melt = (['time'],zeros[:,0],{'units':'m w.e.'}),
                refreeze = (['time'],zeros[:,0],{'units':'m w.e.'}),
                cumrefreeze = (['time'],zeros[:,0],{'units':'m w.e.'}),
                runoff = (['time'],zeros[:,0],{'units':'m w.e.'}),
                accumulation = (['time'],zeros[:,0],{'units':'m w.e.'}),
                rainfall = (['time'],zeros[:,0],{'units':'m w.e.'}),
                sublimation = (['time'],zeros[:,0],{'units':'m w.e.'}),
                deposition = (['time'],zeros[:,0],{'units':'m w.e.'}),
                evaporation = (['time'],zeros[:,0],{'units':'m w.e.'}),
                condensation = (['time'],zeros[:,0],{'units':'m w.e.'}),
                dh = (['time'],zeros[:,0],{'units':'m'}),
                error = (['time'],zeros[:,0],{'units':'m w.e.'}),

                # CLIMATE
                airtemp = (['time'],zeros[:,0],{'units':'C'}),
                rh = (['time'],zeros[:,0],{'units':'%'}),
                wind = (['time'],zeros[:,0],{'units':'m s-1'}),
                winddir = (['time'],zeros[:,0],{'units':'o'}),
                sp = (['time'],zeros[:,0],{'units':'Pa'}),
                tp = (['time'],zeros[:,0],{'units':'m w.e.'}),
                
                # LAYERS
                layertemp = (['time','layer'],zeros,{'units':'C'}),
                layerwater = (['time','layer'],zeros,{'units':'kg m-2'}),
                layerrefreeze = (['time','layer'],zeros,{'units':'kg m-2'}),
                layerheight = (['time','layer'],zeros,{'units':'m'}),
                layerdensity = (['time','layer'],zeros,{'units':'kg m-3'}),
                layerBC = (['time','layer'],zeros,{'units':'ppb'}),
                layerOC = (['time','layer'],zeros,{'units':'ppb'}),
                layerdust = (['time','layer'],zeros,{'units':'ppm'}),
                layergrainsize = (['time','layer'],zeros,{'units':'um'}),
                layerage = (['time','layer'],zeros,{'units':'days'}),
                layertype = (['time','layer'],zeros,{'units':'-'}),
                ), coords=dict(
                    time=(['time'],time),
                    layer=(['layer'],np.arange(N_LAYERS))
                ))
        
        # find the filepath to store the data 
        self.get_output_names()

        # create the empty file to store output
        if params.store_data:
            os.makedirs(self.output_fp, exist_ok=True)
            self.out_files = []
            for i, g in enumerate(terrain.rgiid_n):
                out_fn = os.path.join(self.output_fp, self.output_fn.format(g=g, i=i))
                ds_template[self.store].to_netcdf(out_fn)
                self.out_files.append(out_fn)
        return
    
    def get_output_names(self):
        """
        Creates the filepath/name to store the output.

        Parameters
        ==========
        params : command-line arguments
        """
        params = self.params

        # specify individual file output name format
        self.output_fn = '{g}_{i}.nc'

        # crop the trailing /
        if str(params.output_fp).endswith('/'):
            output_fp_compare = params.output_fp[:-1]
        else:
            output_fp_compare = params.output_fp

        # make it end with a _ for clean naming
        if not str(output_fp_compare).endswith('_'):
            if output_fp_compare is not None:
                output_fp_compare += '_'

        # get output name and store the climate data
        if output_fp_compare is None:
            model_run_date = str(pd.Timestamp.today()).replace('-','_')[0:10]
            self.output_fp = f'RGI{params.rgi_region}_{model_run_date}_'
        
        # make file name unique by adding an indexer
        i = 0
        while os.path.exists(output_fp_compare+ str(i)):
            i += 1
        self.output_fp = output_fp_compare + str(i) + '/'
        return

    def store_data(self, records):
        """
        Saves all data to point netCDF files.
        """

        # loop through glaciers 
        for i, out_fn in enumerate(self.out_files):

            # open and populate the dataset
            with xr.open_dataset(out_fn) as dataset:
                ds = dataset.load()

                # iterate through only the user-requested storage variables
                for var in self.store:
                    # 1D time variables
                    if 'layer' not in var:
                        if hasattr(records, var):
                            ds[var].values = getattr(records, var)[:, i]
                    
                    # 2D layer / time variables
                    else:
                        if hasattr(records, var):
                            ds[var].values = getattr(records, var)[:, i, :]

                # add some helpful variables
                ds = self.add_vars(ds, records, i)

            # save the full dataset back to its output
            ds.to_netcdf(out_fn)
        return
    
    def add_vars(self, ds, records, i):
        """
        Calculates additional variables from other
        existing variables in the output dataset.
        - Surface height change dh [m]
        - Net shortwave radiation flux SWnet [W m-2]
        - Net longwave radiation flux LWnet [W m-2]
        - Net radiation NetRad [W m-2]
        - Net mass balance MB [m w.e.]
        - Summed snow, firn and ice heights [m]
        """
        # add surface height change 
        if 'dh' in self.store:
            total_heights = np.sum(records.layerheight[:, i, :], axis=1)
            initial_height = np.sum(records.layerheight[0, i, :])

            # prepend initial height to compute differences accurately
            padded_heights = np.insert(total_heights, 0, initial_height)

            # different total_heights from initial_height
            ds['dh'].values = np.diff(padded_heights)

        # add summed radiation terms
        if np.all([f in self.store for f in ['SWin','SWout','LWin','LWout']]):
            SWnet = ds['SWin'] + ds['SWout']
            LWnet = ds['LWin'] + ds['LWout']
            NetRad = SWnet + LWnet
            ds['SWnet'] = (['time'],SWnet.values,{'units':'W m-2'})
            ds['LWnet'] = (['time'],LWnet.values,{'units':'W m-2'})
            ds['NetRad'] = (['time'],NetRad.values,{'units':'W m-2'})

        # add summed mass balance term
        if np.all([f in self.store for f in ['accum','refreeze','melt']]):
            MB = ds['accum'] + ds['refreeze'] - ds['melt']
            ds['MB'] = (['time'],MB.values,{'units':'m w.e.'})

        # add snow, firn, and ice depth
        if 'layertype' in self.store and 'layerheight' in self.store:
            snowdepth = ds.layerheight.where(ds.layertype == 0).sum(dim='layer')
            firndepth = ds.layerheight.where(ds.layertype == 1).sum(dim='layer')
            icedepth = ds.layerheight.where(ds.layertype == 2).sum(dim='layer')
            ds['snowdepth'] = (['time'],snowdepth.values,{'units':'m'})
            ds['firndepth'] = (['time'],firndepth.values,{'units':'m'})
            ds['icedepth'] = (['time'],icedepth.values,{'units':'m'})
        return ds
    
    def add_basic_attrs(self,params,time_elapsed,climate):
        """
        Adds informational attributes to the output dataset.
        - glacier name, site, and elevation
        - length of the simulation (time_elapsed)
        - simulation dates (run_start and run_end)
        - list of variables from AWS/reanalysis
        - AWS and reanalysis dataset names
        - model run date
        - machine that ran the simulation (machine) 
        
        Parameters
        ==========
        params : command line arguments
        time_elapsed : float
            Run time for the whole simulation
        climate
            Class object from pebsi.climate
        """
        # get elapsed time
        time_elapsed = f'{time_elapsed:.1f} s'

        # get inforation about climate data sources
        which_re = params.climate_source
        re_str = ''
        if params.use_aws:
            measured = climate.measured_vars
            AWS_name = params.glac_name
            AWS_elev = climate.aws_elev
            which_AWS = f'{AWS_name} {AWS_elev}'
            AWS_str = ', '.join(measured)
            re_vars = [e for e in climate.all_vars if e not in measured]
            if 'vwind' in re_vars and not 'uwind' in re_vars:
                re_vars.remove('vwind')
            if 'uwind' in re_vars and not 'vwind' in re_vars:
                re_vars.remove('uwind')
            re_str += ', '.join(re_vars)
        else:
            re_str += 'all'
            AWS_str = 'none'
            which_AWS = 'none'
        
        # get information about bias correction
        if params.use_aws:
            corr_vars = [v for v in climate.bias_vars if v not in measured]
        else:
            corr_vars = climate.bias_vars
        corr_str = ', '.join(corr_vars)
        corr_str = 'none' if corr_str == '' else corr_str
        
        # load the output dataset
        for i, out_fn in enumerate(self.out_files):
            with xr.open_dataset(out_fn) as dataset:
                ds = dataset.load()

            elev = f'{self.terrain.elev_n[i]:.1f} m a.s.l.'
            rgiid = str(self.terrain.rgiid_n[i])
            lat = str(self.terrain.lat_n[i])
            lon = str(self.terrain.lon_n[i])

            # store basic attributes
            ds = ds.assign_attrs(
                elevation=elev,
                id=rgiid,
                lat=lat,
                lon=lon,
                from_AWS=AWS_str,
                which_AWS=which_AWS,
                from_reanalysis=re_str,
                which_reanalysis=which_re,
                bias_corrected=corr_str,
                sim_start=str(params.start_date),
                sim_end=str(params.end_date),
                model_run_date=str(pd.Timestamp.today()),
                time_elapsed=time_elapsed,
                run_by=params.machine
            )

            if params.task_id > -1:
                ds = ds.assign_attrs(task_id=str(params.task_id))

            # list variables from config that can be skipped
            skip = ['store_data','progress_bar','debug', 'fff',
                        'dates_from_data','climate_source',
                        'bias_vars', 'aws_elev', 'output_fn',
                        'rgiids','start_date','end_date','machine']
            skip_in_config = skip + list(params.cmd_params)

            # add params that were specified in config file
            if params.use_config:
                import yaml
                with open(params.config_fn) as f:
                    config_inputs = yaml.safe_load(f)
        
                new_attrs = {}
                for key, value in config_inputs.items():
                    if key not in skip_in_config:
                        if type(value) == list:
                            if len(value) == self.n_points:
                                store = value[i]
                            else:
                                store = ', '.join(str(v) for v in value)
                        elif type(value) == bool:
                            store = str(value)
                        else:
                            store = value
                        new_attrs[key] = store
                ds = ds.assign_attrs(**new_attrs)

            # add params that were specified in command line
            cmd_attrs = {}
            for key in params.cmd_params:
                if key not in skip:
                    value = getattr(params, key)
                    if type(value) == list:
                        if len(value) == self.n_points:
                            store = value[i]
                        else:
                            store = ', '.join(str(v) for v in value)
                    elif type(value) == bool:
                        store = str(value)
                    else:
                        store = value
                    cmd_attrs[key] = store
            ds = ds.assign_attrs(**cmd_attrs)

            # save NetCDF
            ds.to_netcdf(out_fn)

        # success printout
        print(f"~ Successfully saved model outputs across {self.n_points} points in {self.output_fp} ~")
        return

    def add_attrs(self,new_attrs):
        """
        Adds new attributes as a dict to the output dataset.

        Parameters
        ==========
        new_attrs : dict
            Attributes to store
        """
        if not new_attrs:
            return 
        
        for out_fn in self.out_files:
            with xr.open_dataset(out_fn) as dataset:
                ds = dataset.load()
                ds = ds.assign_attrs(new_attrs)
            ds.to_netcdf(out_fn)
        return
    
    def get_output(self, i):
        """
        Returns the output dataset for a given 
        index in the points array.
        """
        return xr.open_dataset(self.out_files[i])
    
    def delete_temp_files(self):
        """
        Deletes any temporary files that were created
        for parallel runs.
        """
        # delete ice spectrum file
        if os.path.exists(self.params.ice_spectrum_fn):
            os.remove(self.params.ice_spectrum_fn)
        return