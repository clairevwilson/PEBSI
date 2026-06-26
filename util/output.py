"""
Output class for PEBSI

Contains functions which initialize the 
datasets for storage, fills them with data
after the simulation, and adds metadata
attributes to the files.
"""
# Built-in libraries
import os
# External libraries
import numpy as np
import pandas as pd
import xarray as xr
import zarr

class Output():
    """
    Output class which stores the data during the
    simulation and saves it to a .zarr file upon
    run completion.
    """
    def __init__(self, params, terrain):
        """
        Creates list of filenames and variable names
        to store throughout the simulation.
        """
        self.params = params
        self.terrain = terrain

        self.N_LAYERS = params.max_nlayers
        self.N_POINTS = terrain.N_POINTS

        # define all the variables to be saved
        self.all_vars = all_variables = {
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

        # make sure the requested storage variables are valid
        for v in params.store_vars: assert v in all_variables, f'Invalid output group: {v}'

        # extract the actual variables to store
        self.store = [v for g in params.store_vars for v in all_variables[g]]

        # find the filepath to store the data 
        self.get_output_names()

        # generate the filenames for the outputs
        if params.store_data:
            os.makedirs(self.output_fp, exist_ok=True)
            self.out_files = []
            for i, g in enumerate(terrain.rgiid_n):
                out_fn = os.path.join(self.output_fp, self.output_fn.format(g=g, i=i))
                self.out_files.append(out_fn)
        return
    
    def get_output_names(self):
        """
        Creates the filepath/names to store the output.

        Parameters
        ==========
        params : command-line arguments
        """
        params = self.params

        # specify individual file output name format
        self.output_fn = params.output_fn

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

    def store_chunk(self, chunk_records, chunk_dates, chunk_idx):
        """
        Saves a temporal subset of the output to a 
        the .zarr store for each point. 
        """
        # loop through points
        for i, out_fn in enumerate(self.out_files):
            data_vars = {}

            # loop through storage variables and pull from chunk_records
            for var in self.store:
                if not hasattr(chunk_records, var):
                    continue
                vals = np.array(getattr(chunk_records, var))

                # store values to dict with the indexing dims
                if 'layer' not in var:
                    data_vars[var] = (['time'], vals[:, i])
                else:
                    data_vars[var] = (['time', 'layer'], vals[:, i, :])

            # create dataset of this chunk
            ds_chunk = xr.Dataset(data_vars, coords={
                'time': chunk_dates,
                'layer': np.arange(self.N_LAYERS)
            })

            if chunk_idx == 0:
                # make sure time is chunked in reasonable chunk
                encoding = {var: {'chunks': (24 * 7, )} for var in ds_chunk.data_vars}
                ds_chunk.to_zarr(out_fn, mode='w', consolidated=False, encoding=encoding)
            else:
                ds_chunk.to_zarr(out_fn, append_dim='time', consolidated=False)

    def close_out(self, params, time_elapsed, climate):
        """
        Closes out the datasets, storing last helpful things:
          - Units for each variable
          - Additional variables like net radiation
          - Metadata attributes
        """
        self.add_units()
        self.add_vars()
        self.add_basic_attrs(params, time_elapsed, climate)
        for out_fn in self.out_files:
            zarr.consolidate_metadata(out_fn)
        return 
    
    def add_units(self):
        """
        Stores units alongside each data variable
        """
        output_units = {
            'surftemp':'C', 'albedo':'-', 'airtemp': 'C',
            'rh':'%', 'wind':'m s-1', 'winddir': 'o',
            'sp': 'Pa', 'tp': 'm w.e.',
            'layertemp': 'C', 'layerwater': 'kg m-2',
            'layerrefreeze': 'kg m-2', 'layerheight':'m',
            'layerdensity': 'kg m-3', 'layerBC':'ppb',
            'layerOC': 'ppb', 'layerdust':'ppm',
            'layergrainsize': 'um', 'layerage': 'days',
            'layertype': '-',
        }

        for out_fn in self.out_files:
            z = zarr.open(out_fn, mode='a')
            for var in self.store:
                if var not in z:
                    continue
                if var in output_units:
                    z[var].attrs['units'] = output_units[var]
                elif var in self.all_vars['EB']:
                    z[var].attrs['units'] = 'W m-2'
                elif var in self.all_vars['MB']:
                    z[var].attrs['units'] = 'm w.e.'
                
        return
    
    def add_vars(self):
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
        for fn in self.out_files:
            # load dataset and create new dataset for additional vars
            ds = xr.open_zarr(fn, consolidated=False).compute()
            new_vars = xr.Dataset()

            # add surface height change 
            if np.all([f in self.store for f in ['dh','layerheight']]):
                total_heights = np.sum(ds.layerheight.values, axis=1)
                initial_height = np.sum(ds.layerheight.values[0, :])

                # prepend initial height to compute differences accurately
                padded_heights = np.insert(total_heights, 0, initial_height)

                # difference total_heights from initial_height
                new_vars['dh'] = (['time'], np.diff(padded_heights),{'units':'m'})

            # add summed radiation terms
            rad_terms = ['shortwave_in','shortwave_ref','longwave_in','longwave_out']
            if np.all([f in self.store for f in rad_terms]):
                SWnet = ds['shortwave_in'] + ds['shortwave_ref']
                LWnet = ds['longwave_in'] + ds['longwave_out']
                NetRad = SWnet + LWnet
                new_vars['shortwave_net'] = (['time'],SWnet.values,{'units':'W m-2'})
                new_vars['longwave_net'] = (['time'],LWnet.values,{'units':'W m-2'})
                new_vars['net_radiation'] = (['time'],NetRad.values,{'units':'W m-2'})

            # add summed mass balance term
            if np.all([f in self.store for f in ['accumulation','refreeze','melt']]):
                MB = ds['accumulation'] + ds['refreeze'] - ds['melt']
                ds['mass_balance'] = (['time'],MB.values,{'units':'m w.e.'})

            # add snow, firn, and ice depth
            if np.all([f in self.store for f in ['layertype','layerheight']]):
                snowdepth = ds.layerheight.where(ds.layertype == 0).sum(dim='layer')
                firndepth = ds.layerheight.where(ds.layertype == 1).sum(dim='layer')
                icedepth = ds.layerheight.where(ds.layertype == 2).sum(dim='layer')
                new_vars['snowdepth'] = (['time'],snowdepth.values,{'units':'m'})
                new_vars['firndepth'] = (['time'],firndepth.values,{'units':'m'})
                new_vars['icedepth'] = (['time'],icedepth.values,{'units':'m'})
            
            new_vars = new_vars.assign_coords(time=ds.time)
            new_vars.attrs = ds.attrs
            new_vars.to_zarr(fn, mode='a')
        return
    
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
            which_AWS = params.aws_fn
            AWS_elev = f'{params.aws_elev:.1f} m a.s.l.'
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
            AWS_elev = '-'
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
            z = zarr.open(out_fn, mode='a')

            elev = f'{self.terrain.elev_n[i]:.1f} m a.s.l.'
            rgiid = str(self.terrain.rgiid_n[i])
            lat = str(self.terrain.lat_n[i])
            lon = str(self.terrain.lon_n[i])

            # store basic attributes
            attrs = dict(
                elevation=elev,
                id=rgiid,
                lat=lat,
                lon=lon,
                from_AWS=AWS_str,
                which_AWS=which_AWS,
                elev_AWS=AWS_elev,
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
                attrs['task_id'] = str(params.task_id)

            # list variables from config that can be skipped
            skip = ['store_data','progress_bar','debug', 'fff',
                        'dates_from_data','climate_source',
                        'bias_vars', 'aws_elev', 'output_fn',
                        'rgiids','start_date','end_date','machine']
            skip_in_config = skip + list(params.cmd_args)

            # add params that were specified in config file
            if params.use_config:
                import yaml
                with open(params.config_fn) as f:
                    config_inputs = yaml.safe_load(f)
        
                for key, value in config_inputs.items():
                    if key not in skip_in_config:
                        if type(value) == list:
                            if len(value) == self.N_POINTS:
                                store = value[i]
                            else:
                                store = ', '.join(str(v) for v in value)
                        elif type(value) == bool:
                            store = str(value)
                        else:
                            store = value
                        attrs[key] = store

            # add params that were specified in command line
            for key in params.cmd_args:
                if key not in skip:
                    value = getattr(params, key)
                    if type(value) == list:
                        if len(value) == self.N_POINTS:
                            store = value[i]
                        else:
                            store = ', '.join(str(v) for v in value)
                    elif type(value) == bool:
                        store = str(value)
                    else:
                        store = value
                    attrs[key] = store
            
            z.attrs.update(attrs)
            zarr.consolidate_metadata(out_fn)

        # success printout
        print(f"~ Successfully saved model outputs across {self.N_POINTS} points in {self.output_fp} ~")
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
            z = zarr.open(out_fn, mode='a')
            z.attrs.update(new_attrs)
        return
    
    def get_output(self, i):
        """
        Returns the output dataset for a given 
        index in the points array.
        """
        return xr.open_zarr(self.out_files[i]).compute()
    
    def delete_temp_files(self):
        """
        Deletes any temporary files that were created
        for parallel runs.
        """
        # delete ice spectrum file
        if os.path.exists(self.params.ice_spectrum_fn):
            os.remove(self.params.ice_spectrum_fn)
        return