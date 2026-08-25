"""
Output class for PEBSI

Contains functions which initialize the
datasets for storage, fills them with data
after the simulation, and adds metadata
attributes to the files.
"""
# Built-in libraries
import os
import shutil
# External libraries
import numpy as np
import pandas as pd
import xarray as xr
import zarr
# Local libraries
from pebsi.state import OUTPUT_GROUPS

class Output():
    """
    Output class which stores the data during the
    simulation and saves it to a .zarr file upon
    run completion.
    """
    def __init__(self, params, terrain, resume_fp=None):
        """
        Creates the output path and store list of
        variable names to store throughout the simulation.
        """
        self.params = params
        self.terrain = terrain
        self.resume_fp = resume_fp

        self.N_LAYERS = params.max_nlayers
        self.N_POINTS = terrain.N_POINTS

        self.all_vars = OUTPUT_GROUPS

        # make sure the requested storage variables are valid
        all_vars_flat = [vv for vvs in OUTPUT_GROUPS.values() for vv in vvs]
        for v in params.store_vars: 
            if v not in OUTPUT_GROUPS and v not in all_vars_flat:
                assert v in OUTPUT_GROUPS, f'Invalid output group or variable: {v}'

        # extract the actual variables to store
        self.store = [
            v for g in params.store_vars 
            for v in (OUTPUT_GROUPS[g] if g in OUTPUT_GROUPS else [g])
        ]

        # find the filepath to store the data
        self.get_output_name()

        # single combined store for all points
        if params.store_data:
            os.makedirs(self.output_fp, exist_ok=True)
            self.out_fn = os.path.join(self.output_fp, 'output.zarr')
        return

    def get_output_name(self):
        """
        Creates the filepath/names to store the output.
        """
        params = self.params

        # when resuming, reuse the original output directory without incrementing
        if self.resume_fp is not None:
            self.output_fp = self.resume_fp
            return

        # crop the trailing `/``
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
        Saves a temporal subset of the output, across all
        points at once, to the combined .zarr store.
        """
        data_vars = {}

        # loop through storage variables and pull from chunk_records
        for var in self.store:
            if not hasattr(chunk_records, var):
                continue
            vals = np.array(getattr(chunk_records, var))

            # store values to dict with the indexing dims
            if 'layer' not in var:
                data_vars[var] = (['time', 'point'], vals)
            else:
                data_vars[var] = (['time', 'point', 'layer'], vals)

        # create dataset of this chunk, across all points
        ds_chunk = xr.Dataset(data_vars, coords={
            'time': chunk_dates,
            'point': np.arange(self.N_POINTS),
            'layer': np.arange(self.N_LAYERS)
        })

        if chunk_idx == 0:
            # make sure time is chunked in a reasonable chunk; keep point/layer whole
            encoding = {}
            for var in ds_chunk.data_vars:
                if ds_chunk[var].ndim == 2:
                    encoding[var] = {'chunks': (24 * 7, self.N_POINTS)}
                else:
                    encoding[var] = {'chunks': (24 * 7, self.N_POINTS, self.N_LAYERS)}
            ds_chunk.to_zarr(self.out_fn, mode='w', consolidated=False, encoding=encoding)
            
            # write site info right away
            self.add_site_info()
        else:
            ds_chunk.to_zarr(self.out_fn, append_dim='time', consolidated=False)

    def close_out(self, params, time_elapsed):
        """
        Closes out the dataset, storing last helpful things:
          - Units for each variable
          - Additional variables like net radiation
          - Metadata attributes
        """
        self.add_units()
        self.add_vars()
        self.add_basic_attrs(params, time_elapsed)
        self.rechunk_final()
        return

    def rechunk_final(self):
        """
        Rewrites the store chunked along both time (~10yr) and point
        (~10pts) instead of the whole-point chunks used while writing
        incrementally, so per-point and per-time-window reads don't
        each have to decompress the whole dataset.
        """
        ds = xr.open_zarr(self.out_fn, consolidated=False, chunks={})

        n_time = ds.sizes['time']
        if n_time > 1:
            dt = pd.Timestamp(ds.time.values[1]) - pd.Timestamp(ds.time.values[0])
            time_chunk = min(n_time, max(1, int(10 * pd.Timedelta(days=365.25) / dt)))
        else:
            time_chunk = n_time
        point_chunk = min(ds.sizes['point'], 10)

        chunk_sizes = {'time': time_chunk, 'point': point_chunk, 'layer': self.N_LAYERS}
        chunk_sizes = {d: s for d, s in chunk_sizes.items() if d in ds.dims}

        ds = ds.chunk(chunk_sizes)
        encoding = {var: {'chunks': tuple(chunk_sizes.get(d, -1) for d in ds[var].dims)}
                    for var in ds.data_vars}

        tmp_fn = self.out_fn.rstrip('/') + '_rechunk_tmp'
        if os.path.exists(tmp_fn):
            shutil.rmtree(tmp_fn)
        ds.to_zarr(tmp_fn, mode='w', consolidated=False, encoding=encoding)

        shutil.rmtree(self.out_fn)
        shutil.move(tmp_fn, self.out_fn)

        zarr.consolidate_metadata(self.out_fn)
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

        z = zarr.open(self.out_fn, mode='a')
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
        # open lazily (dask-backed) so derived variables stay chunked instead
        # of materializing the entire store (all points x time x layers) in RAM
        ds = xr.open_zarr(self.out_fn, consolidated=False, chunks={})
        new_vars = {}

        # add surface height change
        if np.all([f in self.store for f in ['dh','layerheight']]):
            total_heights = ds.layerheight.sum(dim='layer')  # (time, point)
            initial_height = total_heights.isel(time=0)

            # difference total_heights from the previous step; first step is 0
            # since it's diffed against its own initial value
            diffed = total_heights.diff(dim='time')
            first_step = xr.zeros_like(initial_height).expand_dims(time=[ds.time.values[0]])
            dh = xr.concat([first_step, diffed], dim='time')
            new_vars['dh'] = dh.assign_attrs(units='m')

        # add summed radiation terms
        rad_terms = ['shortwave_in','shortwave_ref','longwave_in','longwave_out']
        if np.all([f in self.store for f in rad_terms]):
            SWnet = ds['shortwave_in'] + ds['shortwave_ref']
            LWnet = ds['longwave_in'] + ds['longwave_out']
            NetRad = SWnet + LWnet
            new_vars['shortwave_net'] = SWnet.assign_attrs(units='W m-2')
            new_vars['longwave_net'] = LWnet.assign_attrs(units='W m-2')
            new_vars['net_radiation'] = NetRad.assign_attrs(units='W m-2')

        # add summed mass balance term
        if np.all([f in self.store for f in ['accumulation','refreeze','melt']]):
            MB = ds['accumulation'] + ds['refreeze'] - ds['melt']
            new_vars['mass_balance'] = MB.assign_attrs(units='m w.e.')

        # add snow, firn, and ice depth
        if np.all([f in self.store for f in ['layertype','layerheight']]):
            snowdepth = ds.layerheight.where(ds.layertype == 0).sum(dim='layer')
            firndepth = ds.layerheight.where(ds.layertype == 1).sum(dim='layer')
            icedepth = ds.layerheight.where(ds.layertype == 2).sum(dim='layer')
            new_vars['snowdepth'] = snowdepth.assign_attrs(units='m')
            new_vars['firndepth'] = firndepth.assign_attrs(units='m')
            new_vars['icedepth'] = icedepth.assign_attrs(units='m')

        if len(new_vars) == 0:
            return

        new_vars = xr.Dataset(new_vars)
        new_vars = new_vars.assign_coords(time=ds.time, point=ds.point)
        new_vars.attrs = ds.attrs
        new_vars.to_zarr(self.out_fn, mode='a')
        return

    def add_site_info(self):
        """
        Stores spatially-varying site info (one value per point)
        alongside the point dimension, rather than duplicating it
        as global attributes.
        """
        site_vars = {
            'rgiid': (['point'], np.array(self.terrain.rgiid_n, dtype=str)),
            'lat': (['point'], np.array(self.terrain.lat_n, dtype=float)),
            'lon': (['point'], np.array(self.terrain.lon_n, dtype=float)),
            'elev': (['point'], np.array(self.terrain.elev_n, dtype=float))
        }

        ds_site = xr.Dataset(site_vars, coords={'point': np.arange(self.N_POINTS)})
        ds_site.to_zarr(self.out_fn, mode='a')
        return

    def add_basic_attrs(self, params, time_elapsed):
        """
        Adds informational attributes to the output dataset.
        - length of the simulation (time_elapsed)
        - simulation dates (run_start and run_end)
        - list of variables from AWS/reanalysis
        - AWS and reanalysis dataset names
        - model run date
        - machine that ran the simulation (machine)
        Any config/command-line parameter that varies per point
        (i.e. is a list of length N_POINTS) is stored as a
        per-point data variable instead of a global attribute.

        Parameters
        ==========
        params : command line arguments
        time_elapsed : float
            Run time for the whole simulation
        """
        # get elapsed time
        time_elapsed = f'{time_elapsed:.1f} s'

        # get information about climate data sources
        # params.climate_measured_vars and params.climate_all_vars are set in
        # PEBSI.run() before the main loop (and updated each chunk in pack_forcings)
        which_re = params.climate_source
        measured = params.climate_measured_vars
        if params.use_aws:
            which_AWS = params.aws_fn
            AWS_elev = f'{params.aws_elev:.1f} m a.s.l.'
            AWS_str = ', '.join(measured)
            re_vars = [e for e in params.climate_all_vars if e not in measured]
            if 'vwind' in re_vars and 'uwind' not in re_vars:
                re_vars.remove('vwind')
            if 'uwind' in re_vars and 'vwind' not in re_vars:
                re_vars.remove('uwind')
            if 'NR' in measured and 'LWin' in re_vars:
                re_vars.remove('LWin')
            re_str = ', '.join(re_vars)
            corr_vars = [v for v in params.bias_vars if v not in measured]
        else:
            re_str = 'all'
            AWS_str = 'none'
            AWS_elev = '-'
            which_AWS = 'none'
            corr_vars = list(params.bias_vars)

        corr_str = ', '.join(corr_vars)
        corr_str = 'none' if corr_str == '' else corr_str

        # store basic attributes
        attrs = dict(
            n_points=self.N_POINTS,
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

        # list variables from config that can be skipped
        skip = ['rgi_ids', 'store_data','progress_bar','debug',
                    'start_date','end_date','testing']

        # values that vary per point become per-point data variables
        per_point_vars = {}

        def _record(key, value):
            if type(value) == list and len(value) == self.N_POINTS:
                per_point_vars[key] = value
            elif type(value) == list:
                attrs[key] = ', '.join(str(v) for v in value)
            elif type(value) == bool:
                attrs[key] = str(value)
            else:
                attrs[key] = value

        # add params that were specified in command line
        for key in params.cmd_args:
            if key not in skip:
                _record(key, getattr(params, key))

        z = zarr.open(self.out_fn, mode='a')
        z.attrs.update(attrs)

        if per_point_vars:
            ds_params = xr.Dataset(
                {k: (['point'], np.array(v)) for k, v in per_point_vars.items()},
                coords={'point': np.arange(self.N_POINTS)}
            )
            ds_params.to_zarr(self.out_fn, mode='a')

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

        z = zarr.open(self.out_fn, mode='a')
        z.attrs.update(new_attrs)
        return

    def get_output(self, i=None):
        """
        Returns the output dataset, optionally sliced
        to a single point index in the points array.
        """
        ds = xr.open_zarr(self.out_fn).compute()
        if i is not None:
            return ds.isel(point=i)
        return ds

    def delete_temp_files(self):
        """
        Deletes any temporary files that were created
        for parallel runs.
        """
        # delete ice spectrum file
        if os.path.exists(self.params.ice_spectrum_fn):
            os.remove(self.params.ice_spectrum_fn)
        return
