import os
import glob
import xarray as xr
import numpy as np
import pandas as pd
pd.options.mode.string_storage = "python"
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pyproj import Transformer
import socket
if 'trace' in socket.gethostname():
    base_fp = '/trace/group/rounce/cvwilson/Output/ddf/'
    home_fp = '/trace/home/cvwilson/research/'
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

class MassBalance():
    def __init__(self, name, site, use = '', min_n_winter = 3):
        """ 
        Grabs the timeseries of data to
        compare a model run.

        Parameters
        name : str
            Glacier name
        site : str
            Site name
        use : str
            Dataset to use from ['WGMS','benchmark']
        """
        # open dataframes
        metadata_df = pd.read_csv(home_fp + 'PEBSI/data/glacier_metadata.csv', index_col='name')
        wgms_df = pd.read_csv(home_fp + 'data/wgms/data/mass_balance_point.csv', parse_dates=True)
        benchmark_fp = '/trace/group/rounce/cvwilson/MB_data/' # home_fp + 'MB_data/'
        glacier_fp = home_fp + 'PEBSI/data/by_glacier/'

        # store input attributes
        self.name = name
        self.site = site 
        self.use = use

        # open site df to find the lat/lon used in the model run
        # site_df = pd.read_csv(glacier_fp + f'{self.name}/site_constants.csv', index_col='site')
        # model_lat = site_df.loc[site, 'lat']
        # model_lon = site_df.loc[site, 'lon']

        # types of data we have
        benchmark_glaciers = [f.lower() for f in os.listdir(benchmark_fp)]
        wgms_glaciers = wgms_df['glacier_name'].unique()

        # determine which type of data we have and put it in standard format
        if self.name.upper() in wgms_glaciers or 'wgms' in use:
            self.get_wgms_data(wgms_df)
        if self.name.replace('_','') in benchmark_glaciers or 'benchmark' in use:
            self.get_benchmark_data(benchmark_fp, min_n_winter)

        # get years
        self.start_year = self.period_starts.year[0]
        self.end_year = self.period_ends.year[-1]
        
        # ensure everything is in array format
        self.period_starts = np.array(pd.to_datetime(self.period_starts))
        self.period_ends = np.array(pd.to_datetime(self.period_ends))
        self.data = np.array(self.data)
        return 
    
    def get_model_mb(self, ds):
        # grab only periods within the dataset
        valid_periods = np.where((self.period_starts >= ds.time.values[0]) & 
                                 (self.period_ends <= ds.time.values[-1]))[0]
        
        if len(valid_periods) < 2:
            self.mod = [np.nan]
            self.meas = [np.nan]
            return
        
        self.period_starts = self.period_starts[valid_periods]
        self.period_ends = self.period_ends[valid_periods]
        meas = self.data[valid_periods]

        mod = []
        for start, end in zip(self.period_starts, self.period_ends):
            if 'MB' not in ds.variables:
                MB = ds['accum'] + ds['refreeze'] - ds['melt']
                ds['MB'] = (['time'],MB.values,{'units':'m w.e.'})
            mb_mod = ds.sel(time=slice(start, end)).MB.sum()
            mod.append(mb_mod)
        mod = np.array(mod)

        if self.dataset == 'seasonal':
            switch_idx = np.where(np.diff(self.period_starts) < pd.Timedelta(days=0))[0][0] + 1
            self.idx_summer = range(switch_idx)
            self.idx_winter = range(switch_idx, len(self.period_starts))

        self.mod = np.array(mod)
        self.meas = np.array(meas)
        assert len(mod) == len(meas)
        return

    def mae(self, seasonal=False):
        if not seasonal:
            return np.mean(np.abs(self.mod - self.meas))
        else:
            summer = np.mean(np.abs(self.mod[self.idx_summer] - self.meas[self.idx_summer]))
            winter = np.mean(np.abs(self.mod[self.idx_winter] - self.meas[self.idx_winter]))
            return summer, winter
    
    def rmse(self, seasonal=False):
        if not seasonal:
            return np.sqrt(np.mean(np.square(self.mod - self.meas)))
        else:
            summer = np.sqrt(np.mean(np.square(self.mod[self.idx_summer] - self.meas[self.idx_summer])))
            winter = np.sqrt(np.mean(np.square(self.mod[self.idx_winter] - self.meas[self.idx_winter])))
            return summer, winter
        
    def bias(self, seasonal=False):
        if not seasonal:
            return np.nanmean(np.square(self.mod - self.meas))
        else:
            summer = np.nanmean(self.mod[self.idx_summer] - self.meas[self.idx_summer])
            winter = np.nanmean(self.mod[self.idx_winter] - self.meas[self.idx_winter])
            return summer, winter
    
    def get_wgms_data(self, wgms_df):
        # WGMS glacier
        name_fmtd = self.name.upper().replace('_',' ')
        df = wgms_df.loc[wgms_df['glacier_name'] == name_fmtd]
        self.df = df.loc[df['original_id'] == self.site]

        self.period_starts = self.df['begin_date']
        self.period_ends = self.df['end_date']
        self.data = self.df['balance']
        self.dataset = 'wgms'
        self.elevation = self.df['elevation'].mean()
        return

    def get_benchmark_data(self, benchmark_fp, min_n_winter = 3):
        # BENCHMARK glacier
        name_parts = [f.capitalize() for f in self.name.split('_')]
        name_fmtd = ''
        for f in name_parts:
            name_fmtd += f
        data_fn = benchmark_fp + f'{name_fmtd}/Input_{name_fmtd}_Glaciological_Data.csv'
        assert os.path.exists(data_fn), f'benchmark data was not found for {self.name}'
        df = pd.read_csv(data_fn, parse_dates=True)

        # find all spring/fall dates across sites
        years = np.unique(df['Year'])
        dates = {}
        for year in years:
            dates[year] = []
            raw_spring = df.loc[df['Year'] == year, 'spring_date'].values
            clean_spring = raw_spring[~pd.isna(raw_spring)]

            if clean_spring.size > 0:
                s_vals, s_counts = np.unique(clean_spring.astype(str), return_counts=True)
                spring_date_mode = s_vals[np.argmax(s_counts)]
                spring_date = spring_date_mode if isinstance(spring_date_mode, str) else f'{year}-04-01'
            else:
                spring_date = f'{year}-04-01'
            
            raw_fall = df.loc[df['Year'] == year, 'fall_date'].values
            clean_fall = raw_fall[~pd.isna(raw_fall)]
            if clean_fall.size > 0:
                vals, counts = np.unique(clean_fall.astype(str), return_counts=True)
                fall_date_mode = vals[np.argmax(counts)]
                fall_date = fall_date_mode if isinstance(fall_date_mode, str) else f'{year}-09-01'
            else:
                fall_date = f'{year}-09-01'
            dates[year].append(spring_date)
            dates[year].append(fall_date)
        # for site in df['site_name'].unique():
        #     print(site, df.loc[df['site_name'] == site, 'ba'].count())
        df = df.loc[df['site_name'] == self.site]

        # determine if there are sufficient seasonal data to compare summer/winter
        n_winter_obs = df['bw'].count()
        if n_winter_obs < min_n_winter or 'annual' in self.use:
            # insufficient data to compare seasonal, so grab annual periods
            index_data = np.where(~np.isnan(df['ba']))[0]
            if len(index_data) > 1:
                index_data = index_data[1:]
            check_starts = pd.to_datetime(df.iloc[index_data - 1]['fall_date'].values)
            annual_ends = pd.to_datetime(df.iloc[index_data]['fall_date'].values)
            annual_data = df.iloc[index_data]['ba'].values

            # fill in any yeras where there is a gap with the correct fall date
            annual_starts = []
            for start, end in zip(check_starts, annual_ends):
                if end - start > pd.Timedelta(days=380):
                    year = end.year
                    if year-1 in dates:
                        annual_starts.append(dates[year - 1][1])
                    else:
                        annual_starts.append(f'{year-1}/09/01')
                else:
                    annual_starts.append(start) 
            annual_starts = pd.to_datetime(annual_starts)

            no_nans = np.where((~np.isnan(annual_starts) &
                                ~np.isnan(annual_ends)))[0]
            self.period_starts = annual_starts[no_nans]
            self.period_ends = annual_ends[no_nans]
            self.data = np.array(annual_data[no_nans])
            self.dataset = 'annual'
            self.elev_annual = df.iloc[index_data]['elevation'].values[no_nans]

        else:
            # sufficient winter data to separate winter and summer periods
            index_data = np.where((~np.isnan(df['ba'])) & (~np.isnan(df['bw'])))[0]
            df_mb = df.iloc[index_data]

            # if data starts at the beginning of the dataframe, need to sort out the previous year
            if index_data[0] == 0:
                # define year_0 which is the year before first data year (needed for bw)
                year_0 = df_mb.iloc[0]['Year'] - 1
                if year_0 in df['Year']:
                    # if that year is in the dataframe, select it
                    first_last = df.loc[df['Year'] == year_0]
                    df_last = pd.concat([first_last, df.iloc[index_data[1:] - 1]])
                elif year_0 in dates:

                    first_last = pd.DataFrame(df_mb.iloc[[0]].values, columns=df_mb.columns)
                    first_last.at[0, 'fall_date'] = dates[year_0][1]
                    
                    df_last = pd.concat([first_last, df.iloc[index_data[1:] - 1]])
                else:
                    # otherwise, need to guess
                    df_last = df.iloc[index_data[1:] - 1]
                    df_mb = df.iloc[index_data[1:]]
            else:
                df_last = df.iloc[index_data - 1]
                if df_last.iloc[0]['Year'] < df_mb.iloc[0]['Year'] - 1:
                    # need to clip the dataframe
                    df_last = df_last.iloc[1:]
                    df_mb = df_mb.iloc[1:]

            # pull out winter ablation/summer accumulation
            this_winter_abl = df_mb['winter_ablation'].values.astype(float)
            past_summer_acc = df_last['summer_accumulation'].values.astype(float)
            this_summer_acc = df_mb['summer_accumulation'].values.astype(float)
            past_summer_acc[np.isnan(past_summer_acc)] = 0
            this_summer_acc[np.isnan(this_summer_acc)] = 0
            this_winter_abl[np.isnan(this_winter_abl)] = 0

            # summer periods
            summer_starts = pd.to_datetime(df_mb['spring_date'].values)
            summer_ends = pd.to_datetime(df_mb['fall_date'].values)
            summer_data = df_mb['ba'].values - df_mb['bw'].values + this_summer_acc
            no_summer_nans = np.where((~np.isnan(summer_starts) &
                                        ~np.isnan(summer_ends)))[0]

            # winter periods
            winter_starts = pd.to_datetime(df_last['fall_date'].values, format='mixed')
            winter_ends = pd.to_datetime(df_mb['spring_date'].values, format='mixed')
            winter_data = df_mb['bw'].values - past_summer_acc + this_winter_abl
            no_winter_nans = np.where((~np.isnan(winter_starts) &
                                        ~np.isnan(winter_ends)))[0]

            self.period_starts = pd.to_datetime(np.append(summer_starts[no_summer_nans],
                                            winter_starts[no_winter_nans]))
            self.period_ends = pd.to_datetime(np.append(summer_ends[no_summer_nans],
                                            winter_ends[no_winter_nans]))
            self.data = np.append(summer_data[no_summer_nans],
                                    winter_data[no_winter_nans])
            self.n_summer = len(summer_starts[no_summer_nans])
            self.n_winter = len(winter_starts[no_winter_nans])
            self.dataset = 'seasonal'

        self.df = df
        if df['elevation'].count() > 0:
            self.elevation = df['elevation'].mean()
        return
    
    def plot_mb(self, mod_label='Modeled', savefig=False):
        self.colors = colors = ['#63c4c7','#fcc02e','#4D559C','#60C252','#BF1F6A',
              '#F77808','#298282','#999999','#FF89B0','#427801']
        
        fig, ax = plt.subplots()
        colors = self.colors
        starts = self.period_starts
        ends = self.period_ends
        mod = self.mod
        meas = self.meas

        if self.dataset == 'seasonal':
            idx_summer = self.idx_summer 
            idx_winter = self.idx_winter

            ax.plot(ends[idx_summer], mod[idx_summer], color=colors[4], marker='.')
            ax.plot(ends[idx_summer], meas[idx_summer], color='k', linestyle='--', marker='.')
            ax.plot(ends[idx_winter], mod[idx_winter], color=colors[0], marker='.')
            ax.plot(ends[idx_winter], meas[idx_winter], color='k', linestyle='--', marker='.')
        else:
            ax.plot(ends, mod, color=colors[0], marker='.')
            ax.plot(ends, meas, color='k', linestyle='--', marker='.')

        ax.plot(np.nan, np.nan, color='k', linestyle='--', label='Measured')
        ax.plot(np.nan, np.nan, color=colors[0], label=mod_label)
        ax.set_title(f'{self.name} {self.site} {self.dataset} mass balance')
        ax.legend()
        if savefig:
            plt.savefig(savefig, dpi=300, bbox_inches='tight')
        return fig, ax

class SnowMelt():
    def __init__(self, name, site, direction='Ascending'):
        self.name = name 
        self.site = site 
        self.direction = direction

        # find site elevation 
        glacier_fp = home_fp + 'PEBSI/data/by_glacier/'
        site_df = pd.read_csv(glacier_fp + f'{self.name}/site_constants.csv', index_col='site')
        self.elevation = site_df.loc[site, 'elevation']

        # find rgi7 glacier number
        rgi7id = translate_rgi[name]['7']
        folder = base_fp + '../../rs/sar/' + rgi7id + '/'

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
                    assert 1==0, 'Frames have mismatched direction! Yikes. awwells@cmu.edu'
                else:
                    dir_frame = dir_frame[0]

                # only use this fn if the direction is Ascending
                if dir_frame == direction and 'snow' in fn:
                    fn_snow = fn
                elif dir_frame == direction and 'melt' in fn:
                    fn_melt = fn 
        
        df_snow = pd.read_csv(folder + fn_snow, parse_dates=True, index_col=0)
        df_melt = pd.read_csv(folder + fn_melt, parse_dates=True, index_col=0)
        
        # reindex
        self.df_snow = df_snow.reindex(pd.date_range(df_snow.index[0], df_snow.index[-1])).ffill()
        self.df_melt = df_melt.reindex(pd.date_range(df_melt.index[0], df_melt.index[-1])).ffill()

        self.sar_snow = {
            'min':self.elevation >= self.df_snow['snowline_elev_min_m'],  # True when the snowline is above the site
            'max':self.elevation >= self.df_snow['snowline_elev_max_m'],  # (True = snow)
            'med':self.elevation >= self.df_snow['snowline_elev_m'],
        }

        self.sar_melt = {
            'min':self.elevation <= self.df_melt['melt_extent_elev_min_m'],  # True when the melt extent is above the site
            'max':self.elevation <= self.df_melt['melt_extent_elev_max_m'],  # (True = melting)
            'med':self.elevation <= self.df_melt['melt_extent_elev_m'],
        }
        return
    
    def get_model_snow(self, ds):
        daily_snow_depth = ds['layerheight'].where(ds['layertype'] < 2).sum(dim='layer').resample(time='1d').min()
        self.mod_snow = daily_snow_depth > 0.05

        # daily_melt = ds['melt'].resample(time='1d').sum()
        # daily_surface_type = ds['layertype'].sel(layer=0).resample(time='1d').max()
        # self.mod_melt = (daily_melt >= 0.01) & (daily_surface_type < 2)
        daily_layer_water = ds['layerwater'].sum(dim='layer').resample(time='1d').min()
        self.mod_melt = daily_layer_water > 0.05

        mod_start = daily_snow_depth.time.values[0]
        mod_end = daily_snow_depth.time.values[-1]
        sar_start = self.sar_snow['med'].index[0]
        sar_end = self.sar_snow['med'].index[-1]
        start = max(mod_start, sar_start)
        end = min(mod_end, sar_end)

        for level in ['min','med','max']:
            self.sar_melt[level] = self.sar_melt[level].loc[slice(start, end)].values
            self.sar_snow[level] = self.sar_snow[level].loc[slice(start, end)].values
        self.mod_melt = self.mod_melt.sel(time=slice(start, end)).values
        self.mod_snow = self.mod_snow.sel(time=slice(start, end)).values
        self.time = pd.date_range(start, end, freq='1d')

        assert len(self.sar_melt['med']) == len(self.mod_melt)
        assert len(self.sar_snow['med']) == len(self.mod_snow)
        return 
    
    def transition_indices(self, arr, transition_type='any'):
        """Return indices where the boolean array flips state."""
        arr = arr.astype(int)
        d = np.diff(arr)
        if transition_type == 'any':
            return np.where(d != 0)[0] + 1   # all transitions (onsets + retreats)
        else:
            onset_idx = np.where(d == 1)[0] + 1      # +1 because diff shifts left
            retreat_idx = np.where(d == -1)[0] + 1

            if transition_type=='onset':
                return onset_idx 
            if transition_type=='offset':
                return retreat_idx

    def model_error_metric(self, model_bool, obs_bool, time, tol_days=np.inf,
                           penalty_missed=0, penalty_false_positive=0,
                           transition_type='any'):
        """
        Returns a single scalar error metric combining:
        - timing error for matched events

        """
        # 1. Extract transitions
        mod_idx = self.transition_indices(model_bool, transition_type)
        obs_idx = self.transition_indices(obs_bool, transition_type)

        mod_t = time[mod_idx]
        obs_t = time[obs_idx]

        # 2. Match each observed event to nearest model event
        timing_errors = []
        matched_model = set()

        for t_obs in obs_t:
            if len(mod_t) == 0:
                continue
            i = np.argmin(np.abs(mod_t - t_obs))
            dt = int(abs((mod_t[i] - t_obs) / np.timedelta64(1, "D")))

            if dt <= tol_days:
                timing_errors.append(dt)
                matched_model.add(i)

        # 3. Count unmatched events
        false_positives = len(mod_t) - len(matched_model)
        false_positive_percent = false_positives / len(mod_t) * 100
        missed_events = len(obs_t) - len(timing_errors)
        missed_events_percent = missed_events / len(mod_t) * 100

        # 4. Combine into one scalar
        if len(timing_errors) > 0:
            timing_component = np.median(timing_errors)
        else:
            timing_component = tol_days  # worst-case penalty

        # Weighting: timing error + penalties for unmatched events
        error = timing_component + penalty_false_positive * false_positive_percent + \
                                   penalty_missed * missed_events_percent
        return error

class Albedo():
    def __init__(self, name, site, use='s2'):
        """ 
        Grabs the timeseries of data to
        compare a model run.

        Parameters
        name : str
            Glacier name
        site : str
            Site name
        """
        # store input attributes
        self.name = name
        self.site = site 
        self.use = use

        # define dictionary for labeling datasets
        self.dataset_dict = {'l8':'Landsat 8', 'l9':'Landsat 9', 's2':'Sentinel-2',
                             'mean':'mean of Landsat/Sentinel scenes'}

        # get RGI7 glacier ID number
        glac_no = translate_rgi[name]['7']
        self.glac_no = glac_no
        if name == 'kahiltna' and site == 'K14k':
            self.glac_no = '01.06469'

        # open dataframes
        metadata_df = pd.read_csv(home_fp + 'PEBSI/data/glacier_metadata.csv', index_col='name')
        glacier_fp = home_fp + 'PEBSI/data/by_glacier/'
        self.albedo_fp = base_fp + '../../rs/albedo/' # updated/

        # find the site location lat/lon
        self.site_df = pd.read_csv(glacier_fp + f'{self.name}/site_constants.csv', index_col='site')
        self.lat = self.site_df.loc[site, 'lat']
        self.lon = self.site_df.loc[site, 'lon']

        # grab the data
        self.get_point_albedo()
        
        # ensure everything is in array format
        self.time = np.array(pd.to_datetime(self.time))
        self.data = np.array(self.data)
        return
    
    def get_point_albedo(self):
        num = str(self.glac_no[3:])

        if self.use == 'all':
            use_list = ['s2','l8','l9']
        else:
            if type(self.use) != list:
                use_list = [self.use]
            else:
                use_list = self.use

        # get filename for this glac_no
        albedo_fns = [self.albedo_fp + f'{num}/{num}_{data}.nc' for data in use_list] # 'RGI2000-v7.0-G-01-

        # build dataset
        self.data = []
        self.time = []
        self.dtype = []
        self.ds = None
        for albedo_fn, dtype in zip(albedo_fns, use_list):
            # open the dataset and get the proper CRS
            ds = xr.open_dataset(albedo_fn)
            crs = ds.spatial_ref.attrs['crs_wkt']
            self.epsg = crs.split('AUTHORITY["EPSG","')[-1].split('"]')[0]

            # filter to the glacier extent
            # self.mask = ds['dem_shadow_mask'].astype(bool)
            # ds = ds.where(self.mask)

            # convert coordinates
            ds = ds['albedo'].rio.write_crs(crs).reset_coords(drop=True).to_dataset()
            ds['dtype'] = ('time', np.array([dtype]*len(ds.time.values)).flatten())    
            if self.ds is None:
                self.ds = ds 
            else:
                self.ds = xr.concat([self.ds, ds], dim='time')

            # select the point on the glacier 
            proj = Transformer.from_crs('EPSG:4326', f'EPSG:{self.epsg}', always_xy=True)
            x, y = proj.transform(self.lon, self.lat)
            da = ds['albedo'].sel(x=x,y=y, method='nearest')
            distance = np.sqrt(np.square(da.x.values - x) + np.square(da.y.values - y))
            assert distance < 100, 'Point selected is more than 100 m away'
            self.da = da
            
            # get the data and timeseries
            da = da.dropna(dim='time')
            for time, value in zip(da.time.values, da.values):
                self.data.append(value)
                self.time.append(time)
                self.dtype.append(dtype)
        self.data = np.array(self.data)
        self.time = np.array(self.time)
        self.dtype = np.array(self.dtype)

        self.ds = self.ds.squeeze('band')

        # get rid of duplicates
        ds_mean = self.ds.groupby("time").mean()
        s = self.ds["dtype"].to_series()
        dupes = s.index.duplicated(keep=False)
        s.loc[dupes] = "mean"
        dtype_da = xr.DataArray(
            s.groupby(level="time").first().values.astype(object),
            dims=["time"],
            coords={"time": s.groupby(level="time").first().index},
        )
        ds_mean["dtype"] = dtype_da
        self.ds = ds_mean.sortby('time')
        self.ds.attrs['crs'] = f'EPSG:{self.epsg}'
        return

    def get_model_albedo(self, ds, months_range=list(range(4, 9)), snow_only=False):
        valid_steps = np.where((self.time >= ds.time.values[0]) & 
                                 (self.time <= ds.time.values[-1]) &
                                 (pd.to_datetime(self.time).month.isin(months_range)))[0]
        
        if len(valid_steps) == 0:
            self.mod = np.nan 
            self.meas = np.nan
            return
        
        self.data = self.data[valid_steps]
        self.time = self.time[valid_steps]
        self.dtype = self.dtype[valid_steps]
        self.mod = ds.sel(time=self.time, method='nearest').albedo.values
        self.meas = self.data

        if snow_only:
            self.snow_only()
        self.format = 'values'
        return
    
    def get_deltas(self, method='max'):
        """
        choose method from 'first', 'max'
        """
        dates = self.time 

        for year in np.unique(pd.to_datetime(dates).year):
            idx = np.where(pd.to_datetime(dates).year == year)[0]
            if method == 'first':
                first_mod = self.mod[idx[0]]
                first_meas = self.meas[idx[0]]
                self.mod[idx] -= first_mod
                self.meas[idx] -= first_meas

            elif method == 'max':
                max_mod = max(self.mod[idx])
                max_meas = max(self.meas[idx])
                self.mod[idx] -= max_mod 
                self.meas[idx] -= max_meas 
        self.format = 'deltas'
        return
    
    def snow_only(self):
        snowline = SnowMelt(self.name, self.site, 'Descending')
        df = snowline.sar_snow['max']

        df.index = pd.to_datetime(df.index).normalize()
        albedo_dates = pd.to_datetime(self.time).normalize()
        idx_snow = df.reindex(albedo_dates).fillna(False).values.astype(bool)

        self.time = self.time[idx_snow]
        self.mod = self.mod[idx_snow]
        self.meas = self.meas[idx_snow]

    def get_dates(self, dates):
        # reshape dates and format
        if len(dates) == 1:
            dates = np.array([dates])
        if len(dates.shape) == 1:
            dates = np.array(dates).reshape(-1, 1)
        
        # find closest index in self.time to each date in dates
        idx_closest = np.argmin(np.abs(dates - self.time), axis=1)
        idx_closest = np.unique(idx_closest)

        mod_idx = self.mod[idx_closest]
        meas_idx = self.meas[idx_closest]
        time_idx = self.time[idx_closest]
        return time_idx, mod_idx, meas_idx

    def mae(self):
        return np.nanmean(np.abs(self.mod - self.meas))
    
    def bias(self):
        return np.nanmean(self.mod - self.meas)
    
    def rmse(self):
        return np.sqrt(np.nanmean(np.square(self.mod - self.meas)))

    def plot_map(self, time='mean', full=False, savefig=False,
                plot_sites = [], full_threshold=0.9):
        ds = self.ds

        # grab the dataarray
        if time == 'mean':
            ds = ds.mean(dim='time')
            if self.use == 'all':
                dataset = self.dataset_dict['mean']
            else:
                dataset = self.dataset_dict[self.use]
        else:
            time = pd.to_datetime(time)
            if full:
                valid_count = ds['albedo'].notnull().sum(dim=('x','y'))
                max_count = self.mask.sum().values

                # identify time steps that meet the requirement
                good_times = valid_count / max_count >= full_threshold
                assert good_times.sum() > 0, f'No {full_threshold*100}% full timesteps were found'

                # extract the subset of times that pass the filter
                filtered_times = ds.time.where(good_times, drop=True)

                # select nearest image among the filtered times
                ds = ds.sel(time=filtered_times.sel(time=time, method="nearest"))
            else:
                ds = ds.sel(time=time, method='nearest')

            dataset = self.dataset_dict[ds.dtype.values.item()]

        fig, ax = plt.subplots(figsize=(6, 5))
        rect = mpl.patches.Rectangle(
            (0, 0), 1, 1,
            transform=ax.transAxes,
            facecolor='none',
            edgecolor='darkgray',
            hatch='///',
            linewidth=0
        )
        ax.add_patch(rect)
        
        im = ds['albedo'].plot(ax=ax, cmap='Grays_r', vmin=0.2, vmax=0.9)
        im.colorbar.set_ticks([0.2, 0.4, 0.6, 0.8, 0.9])
        im.colorbar.ax.tick_params(length=5, labelsize=10)

        for site in plot_sites:
            lat = self.site_df.loc[site, 'lat']
            lon = self.site_df.loc[site, 'lon']
            proj = Transformer.from_crs('EPSG:4326', f'EPSG:{self.epsg}', always_xy=True)
            x, y = proj.transform(lon, lat)
            xrange = ax.get_xlim()[1] - ax.get_xlim()[0]
            yrange = ax.get_ylim()[1] - ax.get_ylim()[0]
            ax.scatter(x, y, color='r', s=50, marker='+') # , facecolor=None)
            ax.text(x + xrange*0.02, y + yrange*0.02, site, c='r',
                            bbox=dict(facecolor='white', edgecolor='none',
                                      pad=1, alpha=0.8))
            
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.set_xlabel('Easting [m]')
        ax.set_ylabel('Northing [m]')
        ax.tick_params(length=5)
        
        ax.set_aspect('equal')
        if time == 'mean':
            time_fmtd = 'time mean'
        else:
            time_fmtd = pd.to_datetime(ds.time.values).strftime('%d %b %Y')
        ax.set_title(f'{self.name.capitalize()} Glacier ({time_fmtd})\nImage from {dataset}')

        if savefig:
            time_fmtd = time_fmtd.replace(' ','-')
            plt.savefig(base_fp + f'{self.name}_{time_fmtd}.png', dpi=300, bbox_inches='tight')

        plt.show()
        return fig, ax
    
    def plot_timeseries(self, years=None):
        if years is None:
            years = np.unique(pd.to_datetime(self.time).year)
        cmap = mpl.cm.get_cmap('viridis')
        norm = mpl.colors.Normalize(vmin=min(years),vmax=max(years))

        fig, ax = plt.subplots()
        ax.plot(np.nan, np.nan, marker='+', color='gray', label='Modeled')
        if self.use == 'all':
            ax.plot(np.nan, np.nan, marker='^', color='gray', label='Landsat-8', linestyle='--')
            ax.plot(np.nan, np.nan, marker='>', color='gray', label='Landsat-9', linestyle='--')
            ax.plot(np.nan, np.nan, marker='.', color='gray', label='Sentinel', linestyle='--')
        else:
            ax.plot(np.nan, np.nan, marker='.', color='gray', label=self.dataset_dict[self.use], linestyle='--')

        for year in years:
            idx = np.where(pd.to_datetime(self.time).year == year)[0]
            if len(idx) < 3:
                continue
            doy = np.array(pd.to_datetime(self.time[idx]).day_of_year)

            if self.use == 'all':
                idx_landsat8 = np.where(self.dtype[idx] == 'l8')[0]
                idx_landsat9 = np.where(self.dtype[idx] == 'l9')[0]
                idx_sentinel = np.where(self.dtype[idx] == 's2')[0]
                ax.plot(doy[idx_landsat8],np.array(self.meas[idx])[idx_landsat8],color=cmap(norm(year)), marker='^', linestyle='--')
                ax.plot(doy[idx_landsat9],np.array(self.meas[idx])[idx_landsat9],color=cmap(norm(year)), marker='>', linestyle='--')
                ax.plot(doy[idx_sentinel],np.array(self.meas[idx])[idx_sentinel],color=cmap(norm(year)), marker='.', linestyle='--')
            else:
                ax.scatter(doy,np.array(self.meas[idx]),color=cmap(norm(year)), marker='.', linestyle='--')

            order = np.argsort(doy)
            doy_sorted = doy[order]
            mod_sorted = self.mod[idx][order]
            ax.plot(doy_sorted, mod_sorted, marker='+', color=cmap(norm(year)), label=str(year))
        ax.set_ylabel('Albedo [-]')
        ax.set_xlabel('Day of year')
        ax.legend(bbox_to_anchor=(1.2, 0.5), loc='center')
        ax.set_title(f'{self.name} {self.site}')
        plt.show()
        return fig, ax
    
    def plot_1to1(self):
        years = np.unique(pd.to_datetime(self.time).year)
        cmap = mpl.cm.get_cmap('viridis')
        norm = mpl.colors.Normalize(vmin=min(years),vmax=max(years))

        fig, ax = plt.subplots(figsize=(3.5, 3.5))

        if self.use == 'all':
            ax.scatter(np.nan, np.nan, marker='^', color='gray', label='Landsat-8')
            ax.scatter(np.nan, np.nan, marker='>', color='gray', label='Landsat-9')
            ax.scatter(np.nan, np.nan, marker='.', color='gray', label='Sentinel')

        for year in years:
            idx = np.where(pd.to_datetime(self.time).year == year)[0]
            mod = np.array(self.mod[idx]).ravel()

            if self.use == 'all':
                idx_landsat8 = np.where(self.dtype[idx] == 'l8')[0]
                idx_landsat9 = np.where(self.dtype[idx] == 'l9')[0]
                idx_sentinel = np.where(self.dtype[idx] == 's2')[0]
                ax.scatter(np.array(self.meas[idx])[idx_landsat8],mod[idx_landsat8],color=cmap(norm(year)), marker='^')
                ax.scatter(np.array(self.meas[idx])[idx_landsat9],mod[idx_landsat9],color=cmap(norm(year)), marker='>')
                ax.scatter(np.array(self.meas[idx])[idx_sentinel],mod[idx_sentinel],color=cmap(norm(year)), marker='.', label=str(year))
            else:
                ax.scatter(self.meas[idx],mod,color=cmap(norm(year)), marker='.', label=str(year))

        if self.format =='values':
            minval, maxval = (0.2, 0.9)
        else:
            minval, maxval = (-0.8, 0)
        ax.plot([minval, maxval],[minval, maxval],'k--')
        ax.set_xlim(minval, maxval)
        ax.set_ylim(minval, maxval)
        ax.set_ylabel('Modeled albedo [-]')
        ax.set_xlabel('Observed albedo [-]')
        ax.tick_params(length=5)
        ax.legend(bbox_to_anchor=(1.2, 0.5), loc='center')
        ax.text(0.03, 0.92, f'Bias: {self.bias():.3f}',transform=ax.transAxes)
        ax.text(0.03, 0.85, f'MAE: {self.mae():.3f}',transform=ax.transAxes)
        ax.set_title(f'{self.name} {self.site}')
        plt.show()
        # return fig, ax

    def plot_map_snowline(self, time, SnowMelt, DEM, which='ice', full=True, savefig=False,
                plot_sites=[], full_threshold=0.9, snowline_var='snowline_elev_min_m'):
        # grab the dataarray
        ds = self.ds
        time = pd.to_datetime(time)

        # find the closest full date to the time requested
        if full:
            valid_count = ds['albedo'].notnull().sum(dim=('x','y'))
            max_count = valid_count.max().values

            # identify time steps that meet the requirement
            good_times = valid_count / max_count >= full_threshold

            # extract the subset of times that pass the filter
            filtered_times = ds.time.where(good_times, drop=True)

            # select nearest image among the filtered times
            ds = ds.sel(time=filtered_times.sel(time=time, method="nearest"))
            time_used = pd.to_datetime(ds.time.values)
        # or just forcibly grab the nearest time, regardless of how much of the image is full
        else:
            ds = ds.sel(time=time, method='nearest')
            time_used = pd.to_datetime(ds.time.values)

        # figure out which dataset this image came from
        dataset = self.dataset_dict[ds.dtype.values.item()]

        # get the snowline on this date
        df_snowline = SnowMelt.df_snow[snowline_var]
        snow_elev = df_snowline.loc[time_used]

        # find which day this snowline actually comes from
        df_up_to_now = df_snowline.loc[:time_used]
        different_values = df_up_to_now[df_up_to_now != snow_elev]
        if not different_values.empty:
            # The source date is the very next entry after the last 'different' value
            source_idx = df_snowline.index.get_loc(different_values.index[-1]) + 1
            source_time = df_snowline.index[source_idx]
        else:
            # If there are no different values, it's the very first date in the dataset
            source_time = df_snowline.index[0]

        # mask it?
        # if which == 'ice':
        #     mask = DEM.dem < snow_elev
        # else:
        #     mask = DEM.dem >= snow_elev
        albedo_range = np.arange(0.1, 0.95, 0.05)

        ar = (ds.x.max() - ds.x.min()) / (ds.y.max() - ds.y.min())
        base_height = 5
        panel_width = base_height * ar
        fig, axes = plt.subplots(1, 2, figsize=(panel_width*2+0.5, base_height))
        for ax in axes:
            rect = mpl.patches.Rectangle(
                (0, 0), 1, 1,
                transform=ax.transAxes,
                facecolor='none',
                edgecolor='darkgray',
                hatch='///',
                linewidth=0
            )
            ax.add_patch(rect)
        
        p1 = ds['albedo'].plot(ax=axes[0], cmap='Grays_r', 
                                vmin=albedo_range[0], vmax=albedo_range[-1],
                                add_colorbar=False)

        elev_range = np.arange(round(DEM.min_elev, -2), round(DEM.max_elev, -2), 100)
        p2 = DEM.dem.plot(ax=axes[1], cmap='terrain',
                           vmin=elev_range[0], vmax=elev_range[-1],
                           add_colorbar=False)
        DEM.shp.plot(ax=axes[1], facecolor='none', edgecolor='k', linewidth=1.5, zorder=10)

        # add contour for snowline
        X = DEM.dem.x.values
        Y = DEM.dem.y.values
        Z = DEM.dem.values
        for ax in axes:
            contour = ax.contour(X, Y, Z, levels=[snow_elev], 
                     colors='red', linewidths=1.5, zorder=11)

        for site in plot_sites:
            lat = self.site_df.loc[site, 'lat']
            lon = self.site_df.loc[site, 'lon']
            proj = Transformer.from_crs('EPSG:4326', f'EPSG:{self.epsg}', always_xy=True)
            x, y = proj.transform(lon, lat)
            xrange = ax.get_xlim()[1] - ax.get_xlim()[0]
            yrange = ax.get_ylim()[1] - ax.get_ylim()[0]
            axes[0].scatter(x, y, color='r', s=50, marker='+') # , facecolor=None)
            axes[0].text(x + xrange*0.02, y + yrange*0.02, site, c='r',
                            bbox=dict(facecolor='white', edgecolor='none',
                                      pad=1, alpha=0.8))
        
        axes[0].set_aspect('equal')
        time_fmtd = time_used.strftime('%d %b %Y')
        source_time_fmtd = source_time.strftime('%d %b %Y')
        fig.suptitle(f'{self.name.capitalize()} Glacier ({time_fmtd})\nImage from {dataset}; snowline from {SnowMelt.direction.lower()} scene on {source_time_fmtd}')

        for ax in axes:
            ax.set_ylabel('Northing [m]')
            ax.set_xlabel('Easting [m]')
            ax.set_title('')
        cax1 = fig.add_axes([0.125, 0, 0.35, 0.03])
        cax2 = fig.add_axes([0.55, 0, 0.35, 0.03])
        if len(albedo_range) > 6:
            albedo_range_ticks = albedo_range[::2]
        cb1 = fig.colorbar(p1, cax=cax1, orientation='horizontal', label='Albedo (-)', ticks=albedo_range_ticks)
        cb1.ax.set_xticklabels([f'{b:.1f}' for b in albedo_range_ticks])
        if len(elev_range) > 6:
            elev_range_clipped = elev_range[::2]
            while len(elev_range_clipped) > 6:
                elev_range_clipped = elev_range_clipped[::2]
        cb2 = fig.colorbar(p2, cax=cax2, orientation='horizontal', label='Elevation (m)', ticks=elev_range_clipped)
        cb2.ax.set_xticklabels([f'{b:.0f}' for b in elev_range_clipped])

        if savefig:
            time_fmtd = time_fmtd.replace(' ','-')
            plt.savefig(base_fp + f'{self.name}_{time_fmtd}.png', dpi=300, bbox_inches='tight')

        plt.show()
        return fig, ax
    
# import os
# import numpy as np
# import pandas as pd
# import rasterio
# import geopandas as gpd
# from rasterio.mask import mask as rio_mask
# from rasterio.warp import reproject, Resampling
# from scipy.stats import binned_statistic

# # file paths
# albedo_fp = '/trace/group/rounce/cvwilson/gulkana_albedo/albedo/'
# mask_fp   = '/trace/group/rounce/cvwilson/gulkana_masks/masks/'
# dem_fp    = '/trace/home/cvwilson/research/data/dems/gulkana_dem.tif'
# shp_fp    = '/trace/group/rounce/cvwilson/dems/gulkana_shapefile.shp'

# # load DEM once
# with rasterio.open(dem_fp) as dem_src:
#     dem_data = dem_src.read(1)
#     dem_profile = dem_src.profile
#     dem_crs = dem_src.crs

# # elevation bins
# min_elev = int(np.nanmin(dem_data) // 100 * 100)
# max_elev = int(np.nanmax(dem_data) // 100 * 100 + 100)
# bins = np.arange(min_elev, max_elev + 100, 100)

# # load shapefile
# shp = gpd.read_file(shp_fp)

# # sort file lists to ensure matching order
# albedo_files = sorted(os.listdir(albedo_fp))
# mask_files   = sorted(os.listdir(mask_fp))

# # dataframe to store results
# bin_centers = (bins[:-1] + bins[1:]) / 2
# df = pd.DataFrame(index=bin_centers)

# # loop through all albedo/mask pairs
# results = []
# for i, (fmask, falbedo) in enumerate(zip(mask_files, albedo_files)):
#     with rasterio.open(os.path.join(albedo_fp, falbedo)) as albedo_ds, rasterio.open(os.path.join(mask_fp, fmask)) as mask_ds:

#         date = pd.to_datetime(falbedo[:8])
#         if i < len(mask_files) - 1:
#             date_next = pd.to_datetime(albedo_files[i+1][:8])
#             last_of_year = date_next.year > date.year 
#         else:
#             last_of_year = False
        
#         # if date.year < 2023:
#         #     continue

#         # clip albedo to shapefile
#         shp_proj = shp.to_crs(albedo_ds.crs)
#         geom = [shp_proj.union_all().__geo_interface__]
#         albedo_clipped, albedo_transform = rio_mask(albedo_ds, geom, crop=True)
#         albedo_arr = albedo_clipped[0].astype(float)

#         # clip mask to shapefile
#         mask_clipped, mask_transform = rio_mask(mask_ds, geom, crop=True)
#         mask_arr = mask_clipped[0].astype(float)

#         # resample mask to albedo grid
#         mask_resampled = np.empty(albedo_arr.shape, dtype=np.float32)
#         reproject(
#             source=mask_arr,
#             destination=mask_resampled,
#             src_transform=mask_transform,
#             src_crs=mask_ds.crs,
#             dst_transform=albedo_transform,
#             dst_crs=albedo_ds.crs,
#             resampling=Resampling.nearest
#         )

#         # apply mask to albedo
#         albedo_arr = np.where(mask_resampled == 1.0, albedo_arr, np.nan)

#         # resample DEM to albedo grid (inside loop!)
#         dem_clipped, dem_transform = rio_mask(rasterio.open(dem_fp), geom, crop=True)
#         dem_arr = dem_clipped[0].astype(float)
#         dem_resampled = np.empty(albedo_arr.shape, dtype=np.float32)
#         reproject(
#             source=dem_arr,
#             destination=dem_resampled,
#             src_transform=dem_transform,
#             src_crs=dem_crs,
#             dst_transform=albedo_transform,
#             dst_crs=albedo_ds.crs,
#             resampling=Resampling.nearest
#         )

#         # apply mask to DEM as well
#         dem_resampled = np.where(mask_resampled == 1.0, dem_resampled, np.nan)
#         dem_resampled = np.where(dem_resampled > 0, dem_resampled, np.nan)

#         # mask NaNs before binning
#         # valid = ~np.isnan(albedo_arr) & ~np.isnan(dem_resampled)
#         albedo_binned, bin_edges, bin_number = binned_statistic(
#             dem_resampled.flatten(),
#             albedo_arr.flatten(),
#             bins=bins,
#             statistic='mean'
#         )

#         # store results
#         results.append(pd.Series(albedo_binned, index=bin_centers, name=date))
#         if last_of_year:
#             date_plus_10 = date + pd.Timedelta(days=10)
#             results.append(pd.Series(np.ones_like(albedo_binned)*np.nan, index=bin_centers, name=date_plus_10))

# # build DataFrame in one go
# df = pd.concat(results, axis=1)

# def plot_db_heatmap(db_bin, dates, bins_center, set_ymin, set_ymax, cmap='Grays_r',
#                     cbar_label='Broadband albedo', ylabel='Elevation [m a.s.l.]',
#                     figsize=(9,6),save_fn=None, years=None,
#                     bins2plot_lowerquantile=2, bins2plot_upperquantile=98):
#     """" Heatmap plotting function """

#     if years is None:
#         years = np.unique(dates.year)
#     nyears = len(years)
#     fig, axes = plt.subplots(1, nyears, figsize=figsize, sharey=True)

#     for ax, year in zip(axes, years):
#         dates_windows = pd.to_datetime(dates[dates.year == year])
#         dates_str = [x.strftime('%Y%m%d') for x in dates_windows]
#         db_bin_full = np.full((db_bin.shape[0], len(dates_windows)), np.nan)

#         for ndate, date in enumerate(dates_str):
#             date_np = np.datetime64(f'{date[:4]}-{date[4:6]}-{date[6:]}').astype('datetime64[ns]')
#             if date_np in dates:
#                 date_idx = np.where(dates == date_np)[0][0]
#                 db_bin_full[:, ndate] = db_bin[:, date_idx]

#         dbmin = np.nanpercentile(db_bin, bins2plot_lowerquantile)
#         dbmax = np.nanpercentile(db_bin, bins2plot_upperquantile)

#         bin_sizes = np.diff(bins_center)
#         if ylabel == 'Elevation [m a.s.l.]':
#             assert np.all(bin_sizes == bin_sizes[0]), 'Elevation bins are not regularly spaced.'

#         x = mpl.dates.date2num(dates_windows)
#         y = bins_center

#         # build edges: start at each timestamp, end at the next
#         x_edges = np.concatenate([
#             x,                                   # each timestamp as a left edge
#             [x[-1] + 10]                         # last edge = 10 days after last timestamp
#         ])

#         # db_bin_full shape: (n_bins, n_dates)
#         mesh = ax.pcolormesh(x_edges, y, db_bin_full, cmap=cmap, vmin=0.2, vmax=0.9, shading='auto')
#         ax.set_xlim(x[0], x[-1])

#         ax.set_title(str(year))
#         ax.set_xlim(pd.to_datetime(f'{year}-04-10'), pd.to_datetime(f'{year}-09-10'))
#         ax.set_ylim([set_ymin, set_ymax])
#         ax.xaxis_date()
#         ax.set_xticks(pd.date_range(f'{year}-04-10', f'{year}-09-10', freq='2MS'))
#         ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%b-%d'))
#         ax.xaxis.set_minor_locator(mpl.dates.MonthLocator(2))
#         ax.tick_params(length=3, which='minor')
#         ax.set_facecolor('#FFF6C9')
#         ax.tick_params(length=5)
#         for label in ax.get_xticklabels():
#             label.set_rotation(45)   # or 90 for vertical
#             label.set_ha('right') 
#     axes[0].set_ylabel('Elevation (m a.s.l.)')
    
#     cax = fig.add_axes((0.95, 0.1, 0.02, 0.8))
#     cb = fig.colorbar(mesh, cax=cax, orientation='vertical')
#     cb.set_label('Broadband albedo')

#     # cax.axis('off')
#     # cb = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap),
#     #                   boundaries=np.arange(0.2, 0.9, 0.1),
#     #             cax=cax, orientation='vertical')
#     # cb.ax.tick_params(labelsize=10,direction='inout',length=8)
#     # cb.ax.set_title('Albedo')
#     # fig.colorbar(cax, orientation='vertical', label=cbar_label)
#     if save_fn:
#         plt.savefig(save_fn, dpi=300, bbox_inches='tight')
#     plt.close(fig)
#     return fig

# plot_db_heatmap(df.to_numpy(), df.columns, bins, set_ymin=1200, set_ymax=2400, figsize=(6, 3), years=np.arange(2019, 2024),
#                 save_fn='/trace/group/rounce/cvwilson/Output/albedo_heatmap.png')

class DEM():
    def __init__(self, name, epsg='default'):
        # store input attributes
        self.name = name

        # get RGI7 glacier ID number
        glac_no = translate_rgi[name]['6']
        self.glac_no = glac_no

        # get DEM filename
        dem_fp = base_fp + '../../dems/RGI1_DEM/'
        dem_fn = dem_fp + f'RGI60-{self.glac_no}_dem.tif'

        # open DEM
        dem = xr.open_dataarray(dem_fn, engine='rasterio').squeeze()

        # ensure both datasets are on the same crs
        if epsg != 'default':
            dem = dem.rio.reproject(f'EPSG:{epsg}')
        self.dem = dem

        # open shapefile
        region = glac_no[:2]
        rgi_fp = home_fp + 'RGI/rgi60/'
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

class LAPs():
    def __init__(self, site):
        fn = home_fp + 'data/Nagorski/bcdust.csv'
        df = pd.read_csv(fn, index_col=0)
        df = df.rename(index={'Taku':'Taku-1'})
        df = df.loc[site]

        self.bc_july = df['Surface_BC_July']
        self.dust_july = df['Surface_dust_July']
        return
    
    def get_model(self, ds):
        date = pd.to_datetime('2024-06-01 00:00')
        diff = pd.Timedelta(hours=12)
        date_range = pd.date_range(date - diff, date + diff)
        self.mod_bc_july = ds.sel(time=date_range, layer=0).layerBC.values
        self.mod_dust_june = ds.sel(time=date_range, layer=0).layerdust.values
        return
