import os
import socket
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pyproj import Transformer
import rioxarray
from scipy.stats import gaussian_kde

if 'trace' in socket.gethostname():
    base_fp = '/trace/group/rounce/cvwilson/'
    home_fp = '/trace/home/cvwilson/research/'
elif 'bridges' in socket.gethostname():
    base_fp = '/ocean/projects/ees260009p/cwilson4/'
    home_fp = '/jet/home/cwilson4/'
elif 'lantern' in socket.gethostname():
    base_fp = '/Users/cvw/local/'
    home_fp = '/Users/cvw/local/'
elif 'campfire' in socket.gethostname():
    base_fp = '/home/claire/Local/Output/'
    home_fp = '/home/claire/Local/'
else:
    base_fp = 'C:/Users/cvw30/Research/Output/ddf/'
    home_fp = 'C:/Users/cvw30/Research/'

translate_rgi = {
                 'gulkana':{'6': '01.00570', '7':'01.05299'}, # GULKANA
                 'kahiltna':{'6':'01.22193','7':'01.04282'}, # KAHILTNA
                 'kennicott':{'6':'01.15645','7':'01.05740'}, # KENNICOTT
                 'wolverine':{'6':'01.09162','7':'01.11350'}, # WOLVERINE
                 'lemon_creek':{'6':'01.01104','7':'01.19406'}, # LEMON CREEK
                 'taku':{'6':'01.01390','7':'01.19709'}, # TAKU
                 }

class Albedo():
    def __init__(self, name, use='s2'):
        """
        Grabs the distributed timeseries of remotely sensed albedo
        to compare against a glacier-wide model run.

        Parameters
        name : str
            Glacier name
        use : str
            Dataset to use from ['l8','l9','s2','all']
        """
        # store input attributes
        self.name = name
        self.use = use

        # define dictionary for labeling datasets
        self.dataset_dict = {'l8':'Landsat 8', 'l9':'Landsat 9', 's2':'Sentinel-2',
                             'mean':'mean of Landsat/Sentinel scenes'}

        # get RGI7 glacier ID number
        glac_no = translate_rgi[name]['7']
        self.glac_no = glac_no

        # open dataframes
        self.albedo_fp = base_fp + 'data/albedo/'

        # grab the distributed data
        self.get_glacier_albedo()

        # put time into numpy format
        self.time = np.array(pd.to_datetime(self.time))
        return

    def get_glacier_albedo(self):
        num = str(self.glac_no[3:])

        if self.use == 'all':
            use_list = ['s2','l8','l9']
        else:
            if type(self.use) != list:
                use_list = [self.use]
            else:
                use_list = self.use

        # get filename for this glac_no
        albedo_fns = [self.albedo_fp + f'{num}/RGI2000-v7.0-G-01-{num}_{data}.nc' for data in use_list]

        # build dataset
        self.time = []
        self.dtype = []
        self.ds_meas = None
        for albedo_fn, dtype in zip(albedo_fns, use_list):
            # open the dataset and get the proper CRS
            ds = xr.open_dataset(albedo_fn)
            self.crs = ds.spatial_ref.attrs['crs_wkt']

            # convert coordinates
            ds = ds['albedo'].rio.write_crs(self.crs).reset_coords(drop=True).to_dataset()
            ds['dtype'] = ('time', np.array([dtype]*len(ds.time.values)).flatten())
            if self.ds_meas is None:
                self.ds_meas = ds
            else:
                self.ds_meas = xr.concat([self.ds_meas, ds], dim='time')

        self.time = np.array(self.time)
        self.dtype = np.array(self.dtype)

        self.ds_meas = self.ds_meas.squeeze('band')

        # get rid of duplicates by averaging 
        ds_mean = self.ds_meas.groupby("time").mean()
        s = self.ds_meas["dtype"].to_series()
        dupes = s.index.duplicated(keep=False)
        s.loc[dupes] = "mean"
        dtype_da = xr.DataArray(
            s.groupby(level="time").first().values.astype(object),
            dims=["time"],
            coords={"time": s.groupby(level="time").first().index},
        )
        ds_mean["dtype"] = dtype_da
        self.ds_meas = ds_mean.sortby('time')
        self.ds_meas.attrs['crs'] = self.crs
        return

    def get_model_albedo(self, model_ds, months_range=list(range(3, 10))):
        """
        Selects the remotely sensed albedo nearest to each point in a
        glacier-wide model run, and matches the model to those times.

        Parameters
        model_ds : xr.Dataset
            Glacier-wide model output with a 'point' dimension and
            'lon'/'lat' coordinates giving each simulation point's location.
        """
        proj = Transformer.from_crs('EPSG:4326', self.crs, always_xy=True)
        x, y = proj.transform(model_ds.lon.values, model_ds.lat.values)

        measured = (self.ds_meas
                    .sel(
                        x=xr.DataArray(x, dims='point'),
                        y=xr.DataArray(y, dims='point'),
                        method='nearest')
                    .drop_duplicates('time'))

        valid_times = measured.time.where(measured.time.dt.month.isin(months_range), drop=True)
        measured = measured.sel(time=valid_times)

        if measured.time.size == 0:
            print('! No valid measurements')
            return

        meas_period = f'{measured.time.values[0]} : {measured.time.values[-1]}'
        mod_period = f'{model_ds.time.values[0]} : {model_ds.time.values[-1]}'
        assert valid_times[0] <= model_ds.time.values[-1], f'Measurements start after model period\nModel: {mod_period}\nMeasured: {meas_period}'
        assert valid_times[-1] >= model_ds.time.values[0], f'Measurements end before model period\nModel: {mod_period}\nMeasured: {meas_period}'

        modeled = model_ds.sel(time=measured.time, method='nearest').drop_duplicates('time')

        self.mod = modeled['albedo'].values
        self.meas = measured['albedo'].values
        self.ds_meas_matched = measured
        self.model_ds = model_ds
        self.format = 'values'
        return

    def get_deltas(self, method='march_mean'):
        """
        choose method from 'max', 'march_mean'
        """
        dates = pd.to_datetime(self.ds_meas_matched.time.values)
        years = np.unique(dates.year)

        mod = self.mod.copy()
        meas = self.meas.copy()

        if method == 'max':
            for year in years:
                idx = np.where(dates.year == year)[0]
                mod[idx] -= np.nanmax(mod[idx], axis=0)
                meas[idx] -= np.nanmax(meas[idx], axis=0)

        elif method == 'march_mean':
            all_meas_march = []
            all_mod_march = []
            for year in years:
                march_idx = np.where((dates.year == year) & (dates.month == 3))[0]
                if len(march_idx) > 0:
                    march_meas = np.nanmean(meas[march_idx], axis=0)
                else:
                    march_meas = np.full(meas.shape[1], np.nan)

                march_mod = self.model_ds.sel(time=slice(f'{year}-03-01', f'{year}-04-01'))['albedo'].mean(dim='time').values

                all_meas_march.append(march_meas)
                all_mod_march.append(march_mod)

            all_meas_march = np.array(all_meas_march)
            all_mod_march = np.array(all_mod_march)

            mean_meas = np.nanmean(all_meas_march, axis=0)
            nan_meas = np.isnan(all_meas_march)
            all_meas_march[nan_meas] = np.broadcast_to(mean_meas, all_meas_march.shape)[nan_meas]

            mean_mod = np.nanmean(all_mod_march, axis=0)
            nan_mod = np.isnan(all_mod_march)
            all_mod_march[nan_mod] = np.broadcast_to(mean_mod, all_mod_march.shape)[nan_mod]

            for i, year in enumerate(years):
                idx = np.where(dates.year == year)[0]
                mod[idx] -= all_mod_march[i]
                meas[idx] -= all_meas_march[i]

        self.mod = mod
        self.meas = meas
        self.format = 'deltas'
        return

    def mae(self, mod=None, meas=None):
        mod = self.mod if mod is None else mod
        meas = self.meas if meas is None else meas
        return np.nanmean(np.abs(mod - meas))

    def bias(self, mod=None, meas=None):
        mod = self.mod if mod is None else mod
        meas = self.meas if meas is None else meas
        return np.nanmean(mod - meas)

    def rmse(self, mod=None, meas=None):
        mod = self.mod if mod is None else mod
        meas = self.meas if meas is None else meas
        return np.sqrt(np.nanmean(np.square(mod - meas)))

    def plot_1to1(self):
        mod = np.array(self.mod).flatten()
        meas = np.array(self.meas).flatten()

        no_nan = (~np.isnan(mod)) & (~np.isnan(meas))
        mod, meas = mod[no_nan], meas[no_nan]

        fig, ax = plt.subplots()
        
        xy = np.vstack([mod, meas])
        kde = gaussian_kde(xy)
        z = kde(xy)

        idx = z.argsort()
        x, y, z = meas[idx], mod[idx], z[idx]

        minn, maxx = (-0.8, 0) if self.format == 'deltas' else (0.1, 0.9)

        ax.scatter(x, y, c=z, cmap='magma')
        ax.plot([minn, maxx], [minn, maxx], 'k--')

        bias = np.mean(mod - meas)
        MAE = np.mean(np.abs((mod - meas)))
        ax.text(0.02, 0.95, f'Bias: {bias:.3f}', transform=ax.transAxes)
        ax.text(0.02, 0.90, f'MAE: {MAE:.3f}', transform=ax.transAxes)

        ax.set_xlabel('Remotely sensed albedo differences')
        ax.set_xlim(minn, maxx)

        ax.set_ylabel('Modeled albedo differences')
        ax.set_ylim(minn, maxx)
        ax.tick_params(length=5)

        # plt.savefig('../figs/new_albedo_agreement.png', dpi=300)
        plt.show()

class SnowlineMelt():
    def __init__(self, name, direction='Ascending'):
        self.name = name
        self.direction = direction

        # load the DEM and note its CRS
        self.dem_obj = DEM(name)
        self.crs = self.dem_obj.dem.rio.crs.to_wkt()

        # find rgi7 glacier number
        rgi7id = translate_rgi[name]['7']
        folder = base_fp + 'data/sar/' + rgi7id + '/'

        # load dataframe containing path/row pairs
        pathframe_fn = folder + '../Vertex_Path_Frame_info.csv'
        df_pathframe = pd.read_csv(pathframe_fn, dtype=str)
        df_pathframe['Path'] = df_pathframe['Path'].apply(lambda x: f"{int(x):03d}")
        df_pathframe['Frame'] = df_pathframe['Frame'].apply(lambda x: f"{int(x):03d}")
        
        for fn in os.listdir(folder):
            if 'melt_extent_elev_percentile' in fn and not 'ea' in fn and not 'eos' in fn:
                fn_melt = fn 
            if 'snowline_elev_percentile' in fn and not 'ea' in fn and not 'eos' in fn:
                fn_snow = fn 

        for fn in os.listdir(folder):
            # filter out extraneous files
            if 'elev_percentile' in fn and not 'ea' in fn and not 'eos' in fn:
                # find the path and frame number of this scene
                path = fn.split('ile_')[1][:3]
                frame = fn.split(path)[1].split('.csv')[0][1:]
                if len(frame) > 3:
                    frames = [str(f) for f in frame.split('_')]
                else:
                    frames = [str(frame)]

                # select the pathframe dataset at this path and frame(s)
                df_path = df_pathframe.loc[df_pathframe['Path'] == path]
                df_frame = df_path.loc[df_path['Frame'].isin(frames)]
                
                # determine direction of this path and frame(s)
                dir_frame = df_frame['Direction'].values
                if len(dir_frame) > 1 and len(np.unique(dir_frame)) > 1:
                    assert 1==0, 'Frames have mismatched direction! Yikes. Ask Albin awwells@cmu.edu'
                else:
                    dir_frame = dir_frame[0]

                # only use this fn if the direction is Ascending
                if dir_frame == direction and 'snow' in fn:
                    fn_snow = fn
                elif dir_frame == direction and 'melt' in fn:
                    fn_melt = fn 
        
        df_snow = pd.read_csv(folder + fn_snow, parse_dates=True, index_col=0)
        df_melt = pd.read_csv(folder + fn_melt, parse_dates=True, index_col=0)

        # reindex to a continuous daily record
        self.df_snow = df_snow.reindex(pd.date_range(df_snow.index[0], df_snow.index[-1])).ffill()
        self.df_melt = df_melt.reindex(pd.date_range(df_melt.index[0], df_melt.index[-1])).ffill()

        # build distributed boolean masks on the DEM grid from the elevation thresholds
        self.get_snowline_melt()
        return

    def get_snowline_melt(self):
        dem = self.dem_obj.dem
        elev = dem.values

        snow_elev = self.df_snow['snowline_elev_m'].values
        melt_elev = self.df_melt['melt_extent_elev_m'].values

        # a cell is snow-covered when its elevation is above the snowline,
        # and melting when its elevation is below the melt extent
        snow_bool = elev[None, :, :] >= snow_elev[:, None, None]
        melt_bool = elev[None, :, :] <= melt_elev[:, None, None]

        self.ds_meas_snow = xr.DataArray(
            snow_bool, dims=('time', 'y', 'x'),
            coords={'time': self.df_snow.index, 'y': dem.y, 'x': dem.x},
        )
        self.ds_meas_melt = xr.DataArray(
            melt_bool, dims=('time', 'y', 'x'),
            coords={'time': self.df_melt.index, 'y': dem.y, 'x': dem.x},
        )
        return

    def get_model_snow(self, model_ds):
        if 'layerheight' in model_ds.variables:
            daily_snow_depth = (model_ds['layerheight']
                                .where(model_ds['layertype'] < 2)
                                .sum(dim='layer')
                                .resample(time='1d').min())
            mod_snow = daily_snow_depth > 0.05
        elif 'surftype' in model_ds.variables:
            mod_snow = model_ds['surftype'].resample(time='1d').max() == 0 


        if 'layerwater' in model_ds.variables:
            daily_layer_water = model_ds['layerwater'].sum(dim='layer').resample(time='1d').min()
            mod_melt = daily_layer_water > 0.05
        elif 'surftemp' in model_ds.variables:
            mod_melt = model_ds['surftemp'].resample(time='1d').min() == 0.0

        # select the measured boolean grid at each simulation point
        proj = Transformer.from_crs('EPSG:4326', self.crs, always_xy=True)
        x, y = proj.transform(model_ds.lon.values, model_ds.lat.values)

        meas_snow = self.ds_meas_snow.sel(
            x=xr.DataArray(x, dims='point'), y=xr.DataArray(y, dims='point'), method='nearest')
        meas_melt = self.ds_meas_melt.sel(
            x=xr.DataArray(x, dims='point'), y=xr.DataArray(y, dims='point'), method='nearest')

        # clip to the overlapping time range
        start = max(mod_snow.time.values[0], meas_snow.time.values[0])
        end = min(mod_snow.time.values[-1], meas_snow.time.values[-1])

        self.mod_snow = mod_snow.sel(time=slice(start, end)).values
        self.mod_melt = mod_melt.sel(time=slice(start, end)).values
        self.meas_snow = meas_snow.sel(time=slice(start, end)).values
        self.meas_melt = meas_melt.sel(time=slice(start, end)).values
        self.time = pd.date_range(start, end, freq='1d')

        assert self.mod_snow.shape == self.meas_snow.shape
        assert self.mod_melt.shape == self.meas_melt.shape
        return

    def bernoulli_loss(self, mod_snow=None, meas_snow=None, mod_melt=None, meas_melt=None, eps=1e-7):
        def bce(mod, meas):
            p = np.clip(np.asarray(mod, dtype=float), eps, 1 - eps)
            y = np.asarray(meas, dtype=float)
            return -np.nanmean(y * np.log(p) + (1 - y) * np.log(1 - p))

        mod_snow = self.mod_snow if mod_snow is None else mod_snow
        meas_snow = self.meas_snow if meas_snow is None else meas_snow
        mod_melt = self.mod_melt if mod_melt is None else mod_melt
        meas_melt = self.meas_melt if meas_melt is None else meas_melt

        snow_loss = bce(mod_snow, meas_snow)
        melt_loss = bce(mod_melt, meas_melt)
        return snow_loss, melt_loss

class DEM():
    def __init__(self, name, epsg='default'):
        # store input attributes
        self.name = name

        # get RGI7 glacier ID number
        glac_no = translate_rgi[name]['6']
        self.glac_no = glac_no

        # get DEM filename
        dem_fp = base_fp + 'data/dems/RGI1_DEM/'
        dem_fn = dem_fp + f'RGI60-{self.glac_no}_dem.tif'

        # open DEM
        dem = xr.open_dataarray(dem_fn, engine='rasterio').squeeze()

        # ensure both datasets are on the same crs
        if epsg != 'default':
            dem = dem.rio.reproject(f'EPSG:{epsg}')
        self.dem = dem

        # open shapefile
        region = glac_no[:2]
        rgi_fp = base_fp + 'RGI/rgi60/'
        for fn in os.listdir(rgi_fp):
            if region in fn and 'Zone' not in fn:
                reg_name = fn
        rgi_fn = rgi_fp + f'{reg_name}/{reg_name}.shp'
        shp_reg = gpd.read_file(rgi_fn)
        self.shp = shp_reg[shp_reg['RGIId'] == f'RGI60-{glac_no}']

        # get min and max elevation
        self.min_elev = float(dem.min().values)
        self.max_elev = float(dem.max().values)
        return

    def reproject_to(self, da_sample):
        # reproject to match the grid of the da_sample
        self.shp = self.shp.to_crs(da_sample.crs)
        self.dem = self.dem.rio.reproject_match(da_sample)
        return