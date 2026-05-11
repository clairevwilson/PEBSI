
# Built-in libraries
import os, sys
import time
# External libraries
import numpy as np
import pandas as pd
import xarray as xr

class Output():
    """
    Output class which stores the data during the
    simulation and saves it to a netcdf file upon
    run completion.
    """
    def __init__(self,time,args):
        """
        Creates netcdf file where the model output 
        will be saved.

        Parameters
        ==========
        time : list-like
            List of times used in the simulation
        args : command-line args
        """
        # get filename
        self.out_fn = args.output_fp + args.output_fn
        self.args = args

        # info needed to create the output file
        self.n_timesteps = len(time)
        zeros = np.zeros([self.n_timesteps,args.max_nlayers])

        # create variable name dict
        vn_dict = {'EB':['SWin','SWout','LWin','LWout','rain','ground',
                         'sensible','latent','meltenergy','albedo'],
                   'MB':['melt','refreeze','runoff','cumrefreeze','dh',
                         'vaporsolid','vaporliquid','accum','rainfall'],
                   'temp':['airtemp','surftemp'],
                   'layers':['layertemp','layerdensity','layerwater','layerheight',
                             'layerage','layertype','layergrainsize','layerrefreeze',
                             'layerBC','layerOC','layerdust'],
                    'SW':['vis_albedo','SWin_sky','SWin_terr']}
        
        # create file to store outputs
        all_variables = xr.Dataset(data_vars = dict(
                SWin = (['time'],zeros[:,0],{'units':'W m-2'}),
                SWout = (['time'],zeros[:,0],{'units':'W m-2'}),
                SWin_sky = (['time'],zeros[:,0],{'units':'W m-2'}),
                SWin_terr = (['time'],zeros[:,0],{'units':'W m-2'}),
                LWin = (['time'],zeros[:,0],{'units':'W m-2'}),
                LWout = (['time'],zeros[:,0],{'units':'W m-2'}),
                rain = (['time'],zeros[:,0],{'units':'W m-2'}),
                ground = (['time'],zeros[:,0],{'units':'W m-2'}),
                sensible = (['time'],zeros[:,0],{'units':'W m-2'}),
                latent = (['time'],zeros[:,0],{'units':'W m-2'}),
                meltenergy = (['time'],zeros[:,0],{'units':'W m-2'}),
                albedo = (['time'],zeros[:,0],{'units':'0-1'}),
                vis_albedo = (['time'],zeros[:,0],{'units':'0-1'}),
                melt = (['time'],zeros[:,0],{'units':'m w.e.'}),
                refreeze = (['time'],zeros[:,0],{'units':'m w.e.'}),
                cumrefreeze = (['time'],zeros[:,0],{'units':'m w.e.'}),
                runoff = (['time'],zeros[:,0],{'units':'m w.e.'}),
                accum = (['time'],zeros[:,0],{'units':'m w.e.'}),
                rainfall = (['time'],zeros[:,0],{'units':'m w.e.'}),
                vaporliquid = (['time'],zeros[:,0],{'units':'m w.e.'}),
                vaporsolid = (['time'],zeros[:,0],{'units':'m w.e.'}),
                airtemp = (['time'],zeros[:,0],{'units':'C'}),
                surftemp = (['time'],zeros[:,0],{'units':'C'}),
                layertemp = (['time','layer'],zeros,{'units':'C'}),
                layerwater = (['time','layer'],zeros,{'units':'kg m-2'}),
                layerrefreeze = (['time','layer'],zeros,{'units':'kg m-2'}),
                layerheight = (['time','layer'],zeros,{'units':'m'}),
                layerdensity = (['time','layer'],zeros,{'units':'kg m-3'}),
                layerBC = (['time','layer'],zeros,{'units':'ppb'}),
                layerOC = (['time','layer'],zeros,{'units':'ppb'}),
                layerdust = (['time','layer'],zeros,{'units':'ppm'}),
                layergrainsize = (['time','layer'],zeros,{'units':'um'}),
                layerage = (['time','layer'],zeros),
                layertype = (['time','layer'],zeros),
                dh = (['time'],zeros[:,0],{'units':'m'})
                ),
                coords=dict(
                    time=(['time'],time),
                    layer=(['layer'],np.arange(args.max_nlayers))
                    ))
        # select variables from the specified input
        vars_list = vn_dict[args.store_vars[0]]
        for var in args.store_vars[1:]:
            assert var in vn_dict, 'Choose store_vars from [MB,EB,temp,layers,SW]'
            vars_list.extend(vn_dict[var])
        self.vars_list = vars_list
        
        # create the netcdf file to store output
        if args.store_data:
            if not os.path.exists(args.output_fp):
                try:
                    os.mkdir(args.output_fp)
                except:
                    assert os.path.exists(args.output_fp), f'Create output folder at {args.output_filepath}'
            all_variables[self.vars_list].to_netcdf(self.out_fn)

        # ENERGY BALANCE OUTPUTS
        self.SWin_output = []       # incoming shortwave [W m-2]
        self.SWout_output = []      # outgoing shortwave [W m-2]
        self.LWin_output = []       # incoming longwave [W m-2]
        self.LWout_output = []      # outgoing longwave [W m-2]
        self.rain_output = []       # rain energy [W m-2]
        self.ground_output = []     # ground energy [W m-2]
        self.sensible_output = []   # sensible energy [W m-2]
        self.latent_output = []     # latent energy [W m-2]
        self.meltenergy_output = [] # melt energy [W m-2]
        self.albedo_output = []     # surface broadband albedo [-]

        # TEMPERATURE OUTPUTS
        self.airtemp_output = []    # downscaled air temperature [C]
        self.surftemp_output = []   # surface temperature [C]

        # MASS BALANCE OUTPUTS
        self.melt_output = []           # melt by timestep [m w.e.]
        self.refreeze_output = []       # refreeze by timestep [m w.e.]
        self.cumrefreeze_output = []    # cumulative refreeze by timestep [m w.e.]
        self.accum_output = []          # accumulation by timestep [m w.e.]
        self.rainfall_output = []          # accumulation by timestep [m w.e.]
        self.runoff_output = []         # runoff by timestep [m w.e.]
        self.dh_output = []             # surface height change by timestep [m]
        self.vaporliquid_output = []    # liquid-vapor mass flux [m w.e.]  
        self.vaporsolid_output = []     # solid-vapor mass flux [m w.e.]  

        # DETAILED SHORTWAVE OUTPUTS
        self.SWin_sky_output = []   # incoming sky shortwave [W m-2]
        self.SWin_terr_output = []  # incoming terrain shortwave [W m-2]
        self.vis_albedo_output = [] # surface visible albedo [-]

        # LAYER OUTPUTS
        self.layertemp_output = dict()      # layer temperature [C]
        self.layerwater_output = dict()     # layer water content [kg m-2]
        self.layerdensity_output = dict()   # layer density [kg m-3]
        self.layerheight_output = dict()    # layer height [m]
        self.layerBC_output = dict()        # layer black carbon content [ppb]
        self.layerOC_output = dict()        # layer organic carbon content [ppb]
        self.layerdust_output = dict()      # layer dust content [ppm]
        self.layergrainsize_output = dict() # layer grain size [um]
        self.layerrefreeze_output = dict()  # layer refreeze [kg m-2]
        self.layerage_output = dict()       # layer age [datetime]
        self.layertype_output = dict()      # layer type [-]
        self.last_height = args.initial_ice_depth+args.initial_firn_depth+args.initial_snow_depth
        return
    
    def store_timestep(self,massbal,enbal,surface,layers,step):
        """
        Appends the current values to each output list.

        Parameters
        ==========
        massbal
            Class object from pebsi.massbalance
        enbal
            Class object from pebsi.energybalance
        surface
            Class object from pebsi.surface
        layers
            Class object from pebsi.layers
        step : pd.Datetime
            Current timestamp
        """
        # CONSTANTS
        DENSITY_WATER = self.args.density_water
        step = str(step)

        # ENERGY BALANCE OUTPUTS
        self.SWin_output.append(float(enbal.SWin))
        self.SWout_output.append(float(enbal.SWout))
        self.LWin_output.append(float(enbal.LWin))
        self.LWout_output.append(float(enbal.LWout))
        self.rain_output.append(float(enbal.rain))
        self.ground_output.append(float(enbal.ground))
        self.sensible_output.append(float(enbal.sens))
        self.latent_output.append(float(enbal.lat))
        self.meltenergy_output.append(float(surface.Qm))
        self.albedo_output.append(float(surface.bba))
        
        # TEMPERATURE OUTPUTS
        self.surftemp_output.append(float(surface.stemp))
        self.airtemp_output.append(float(enbal.tempC))

        # MASS BALANCE OUTPUTS
        self.melt_output.append(float(massbal.melt))
        self.refreeze_output.append(float(massbal.refreeze))
        self.cumrefreeze_output.append(float(np.sum(layers.lrefreeze))/DENSITY_WATER)
        self.runoff_output.append(float(massbal.runoff))
        self.accum_output.append(float(massbal.accum))
        self.rainfall_output.append(float(massbal.rainfall))
        self.dh_output.append(np.sum(layers.lheight)-self.last_height)
        self.last_height = np.sum(layers.lheight)
        self.vaporliquid_output.append(massbal.vapor_liquid/DENSITY_WATER)
        self.vaporsolid_output.append(massbal.vapor_solid/DENSITY_WATER)

        # LAYER OUTPUTS
        self.layertemp_output[step] = layers.ltemp.copy()
        self.layerwater_output[step] = layers.lwater.copy()
        self.layerheight_output[step] = layers.lheight.copy()
        self.layerdensity_output[step] = layers.ldensity.copy()
        self.layerBC_output[step] = layers.lBC / layers.lheight * 1e6
        self.layerOC_output[step] = layers.lOC / layers.lheight * 1e6
        self.layerdust_output[step] = layers.ldust / layers.lheight * 1e3
        self.layergrainsize_output[step] = layers.lgrainsize.copy()
        self.layerrefreeze_output[step] = layers.lrefreeze.copy()
        self.layerage_output[step] = layers.lage.copy()
        mapping = {'snow': 0, 'firn': 1, 'ice': 2}
        self.layertype_output[step] = [mapping[l] for l in layers.ltype]

        # DETAILED SHORTWAVE OUTPUTS
        self.vis_albedo_output.append(float(surface.vis_a))
        self.SWin_sky_output.append(float(enbal.SWin_sky))
        self.SWin_terr_output.append(float(enbal.SWin_terr))
        return

    def store_data(self):
        """
        Saves all data in the netcdf file.
        """
        args = self.args 

        # load output dataset
        with xr.open_dataset(self.out_fn) as dataset:
            ds = dataset.load()
            # store variables
            if 'EB' in args.store_vars:
                ds['SWin'].values = self.SWin_output
                ds['SWout'].values = self.SWout_output
                ds['LWin'].values = self.LWin_output
                ds['LWout'].values = self.LWout_output
                ds['rain'].values = self.rain_output
                ds['ground'].values = self.ground_output
                ds['sensible'].values = self.sensible_output
                ds['latent'].values = self.latent_output
                ds['meltenergy'].values = self.meltenergy_output
                ds['albedo'].values = self.albedo_output
            if 'MB' in args.store_vars:
                ds['melt'].values = self.melt_output
                ds['refreeze'].values = self.refreeze_output
                ds['runoff'].values = self.runoff_output
                ds['accum'].values = self.accum_output
                ds['rainfall'].values = self.rainfall_output
                ds['dh'].values = self.dh_output
                ds['cumrefreeze'].values = self.cumrefreeze_output
                ds['vaporliquid'].values = self.vaporliquid_output
                ds['vaporsolid'].values = self.vaporsolid_output
            if 'temp' in args.store_vars:
                ds['airtemp'].values = self.airtemp_output
                ds['surftemp'].values = self.surftemp_output
            if 'SW' in args.store_vars:
                ds['SWin_sky'].values = self.SWin_sky_output
                ds['SWin_terr'].values = self.SWin_terr_output
                ds['vis_albedo'].values = self.vis_albedo_output
            if 'layers' in args.store_vars:
                layertemp_output = pd.DataFrame.from_dict(self.layertemp_output,orient='index')
                layerdensity_output = pd.DataFrame.from_dict(self.layerdensity_output,orient='index')
                layerheight_output = pd.DataFrame.from_dict(self.layerheight_output,orient='index')
                layerwater_output = pd.DataFrame.from_dict(self.layerwater_output,orient='index')
                layerBC_output = pd.DataFrame.from_dict(self.layerBC_output,orient='index')
                layerOC_output = pd.DataFrame.from_dict(self.layerOC_output,orient='index')
                layerdust_output = pd.DataFrame.from_dict(self.layerdust_output,orient='index')
                layergrainsize_output = pd.DataFrame.from_dict(self.layergrainsize_output,orient='index')
                layerrefreeze_output = pd.DataFrame.from_dict(self.layerrefreeze_output,orient='index')
                layerage_output = pd.DataFrame.from_dict(self.layerage_output,orient='index')
                layertype_output = pd.DataFrame.from_dict(self.layertype_output,orient='index')

                if len(layertemp_output.columns) < args.max_nlayers:
                    n_columns = len(layertemp_output.columns)
                    # Build the missing column names
                    missing_cols = [str(i) for i in range(n_columns, args.max_nlayers)]

                    # Build a DataFrame of NaNs once
                    nan_block = pd.DataFrame(
                        np.full((self.n_timesteps, len(missing_cols)), np.nan),
                        columns=missing_cols,index=layertemp_output.index,
                    )

                    # Now append to each output DataFrame in one shot
                    layertemp_output = pd.concat([layertemp_output, nan_block], axis=1)
                    layerdensity_output = pd.concat([layerdensity_output, nan_block], axis=1)
                    layerheight_output = pd.concat([layerheight_output, nan_block], axis=1)
                    layerwater_output = pd.concat([layerwater_output, nan_block], axis=1)
                    layerBC_output = pd.concat([layerBC_output, nan_block], axis=1)
                    layerOC_output = pd.concat([layerOC_output, nan_block], axis=1)
                    layerdust_output = pd.concat([layerdust_output, nan_block], axis=1)
                    layergrainsize_output = pd.concat([layergrainsize_output, nan_block], axis=1)
                    layerrefreeze_output = pd.concat([layerrefreeze_output, nan_block], axis=1)
                    layerage_output = pd.concat([layerage_output, nan_block], axis=1)
                    layertype_output = pd.concat([layertype_output, nan_block], axis=1)

                else:
                    n = len(layertemp_output.columns)
                    assert 1==0, f'Need to increase max_nlayers: currently have {n} layers'

                ds['layertemp'].values = layertemp_output
                ds['layerheight'].values = layerheight_output
                ds['layerdensity'].values = layerdensity_output
                ds['layerwater'].values = layerwater_output
                ds['layerBC'].values = layerBC_output
                ds['layerOC'].values = layerOC_output
                ds['layerdust'].values = layerdust_output
                ds['layergrainsize'].values = layergrainsize_output
                ds['layerrefreeze'].values = layerrefreeze_output
                ds['layertype'].values = layertype_output

                # handle datetime object for layerage
                arr = np.asarray(layerage_output, dtype=object)
                arr[pd.isna(arr)] = np.datetime64("NaT")
                ds['layerage'].values = arr.astype("datetime64[ns]")

        encoding = {
            "layerage": {
                "units": "seconds since 1970-01-01 00:00:00",
                "calendar": "standard",
                "dtype":"float64"
            }
        }

        # save NetCDF
        ds.to_netcdf(self.out_fn, encoding=encoding)

        return ds
    
    def add_vars(self):
        """
        Calculates additional variables from other
        existing variables in the output dataset.
        - Net shortwave radiation flux SWnet [W m-2]
        - Net longwave radiation flux LWnet [W m-2]
        - Net radiation NetRad [W m-2]
        - Net mass balance MB [m w.e.]
        """
        if 'SWin' in self.vars_list:
            with xr.open_dataset(self.out_fn) as dataset:
                ds = dataset.load()

                # add summed radiation terms
                SWnet = ds['SWin'] + ds['SWout']
                LWnet = ds['LWin'] + ds['LWout']
                NetRad = SWnet + LWnet
                ds['SWnet'] = (['time'],SWnet.values,{'units':'W m-2'})
                ds['LWnet'] = (['time'],LWnet.values,{'units':'W m-2'})
                ds['NetRad'] = (['time'],NetRad.values,{'units':'W m-2'})

                # add summed mass balance term
                MB = ds['accum'] + ds['refreeze'] - ds['melt']
                ds['MB'] = (['time'],MB.values,{'units':'m w.e.'})

                # add snow, firn, and ice depth
                snowdepth = ds.layerheight.where(ds.layertype == 0).sum(dim='layer')
                firndepth = ds.layerheight.where(ds.layertype == 1).sum(dim='layer')
                icedepth = ds.layerheight.where(ds.layertype == 2).sum(dim='layer')
                ds['snowdepth'] = (['time'],snowdepth.values,{'units':'m'})
                ds['firndepth'] = (['time'],firndepth.values,{'units':'m'})
                ds['icedepth'] = (['time'],icedepth.values,{'units':'m'})

            # save NetCDF 
            ds.to_netcdf(self.out_fn)
        return
    
    def add_basic_attrs(self,args,time_elapsed,climate):
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
        args : command line arguments
        time_elapsed : float
            Run time for the whole simulation
        climate
            Class object from pebsi.climate
        """
        time_elapsed = f'{time_elapsed:.1f} s'
        elev = str(args.elevation)+' m a.s.l.'

        # get information on variable sources (AWS or reanalysis)
        which_re = args.reanalysis
        re_str = ''
        if args.use_aws:
            measured = climate.measured_vars
            AWS_name = args.glac_name
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
        if args.use_aws:
            corr_vars = [v for v in climate.bias_vars if v not in measured]
        else:
            corr_vars = climate.bias_vars
        corr_str = ', '.join(corr_vars)
        corr_str = 'none' if corr_str == '' else corr_str
        
        # load the output dataset
        with xr.open_dataset(self.out_fn) as dataset:
            ds = dataset.load()

        # store basic attributes
        ds = ds.assign_attrs(glacier=args.glac_name,
                                id=args.rgi_id,
                                elevation=elev,
                                site=args.site,
                                from_AWS=AWS_str,
                                which_AWS=which_AWS,
                                from_reanalysis=re_str,
                                which_reanalysis=which_re,
                                bias_corrected=corr_str,
                                sim_start=str(args.start_date),
                                sim_end=str(args.end_date),
                                model_run_date=str(pd.Timestamp.today()),
                                time_elapsed=time_elapsed,
                                run_by=args.machine)
        if args.task_id > -1:
            ds = ds.assign_attrs(task_id=str(args.task_id))

        # list variables from config that can be skipped
        skip = ['store_data','progress_bar','debug', 'fff',
                    'dates_from_data','reanalysis',
                    'bias_vars', 'aws_elev', 'output_fn',
                    'glac_name', 'site', 'rgi_id',
                    'start_date','end_date','machine']
        skip_in_config = skip + args.cmd_args

        # add args that were specified in config file
        if args.use_config:
            import yaml
            with open(args.config_fn) as f:
                config_inputs = yaml.safe_load(f)
    
            new_attrs = {}
            for key, value in config_inputs.items():
                if key not in skip_in_config:
                    if type(value) == list:
                        store = ', '.join(value)
                    elif type(value) == bool:
                        store = str(value)
                    else:
                        store = value
                    new_attrs[key] = store
            ds = ds.assign_attrs(**new_attrs)

        # add args that were specified in command line
        cmd_attrs = {}
        for key in args.cmd_args:
            if key not in skip:
                value = getattr(args, key)
                if type(value) == list:
                    store = ', '.join(value)
                elif type(value) == bool:
                    store = str(value)
                else:
                    store = value
                cmd_attrs[key] = store
        ds = ds.assign_attrs(**cmd_attrs)

        # save NetCDF
        ds.to_netcdf(self.out_fn)

        # success printout
        print(f'~ Saved {args.glac_name.capitalize()} {args.site} model output to {self.out_fn} ~')
        return
    
    def add_attrs(self,new_attrs):
        """
        Adds new attributes as a dict to the output dataset.

        Parameters
        ==========
        new_attrs : dict
            Attributes to store
        """
        with xr.open_dataset(self.out_fn) as dataset:
            ds = dataset.load()
            if not new_attrs:
                return ds
            ds = ds.assign_attrs(new_attrs)
        ds.to_netcdf(self.out_fn)
        return
    
    def get_output(self):
        """
        Returns the output dataset.
        """
        return xr.open_dataset(self.out_fn)
    
    def delete_temp_files(self):
        """
        Deletes any temporary files that were created
        for parallel runs.
        """
        # delete ice spectrum file
        if os.path.exists(self.args.ice_spectrum_fn):
            os.remove(self.args.ice_spectrum_fn)
        return