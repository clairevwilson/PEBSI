import os
import glob
import xarray as xr
import numpy as np
import pandas as pd
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

translate_rgi = {'gulkana':'01.05299', # GULKANA
                 'kahiltna':'01.04282', # KAHILTNA
                 'kennicott':'01.05740', # KENNICOTT
                 'wolverine':'01.11350', # WOLVERINE
                 'lemon_creek':'01.19406', # LEMON CREEK
                 'taku':'01.19709', # TAKU
                 }

class MassBalance():
    def __init__(self, name, site, use = ''):
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
        benchmark_fp = home_fp + 'MB_data/'
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
            self.get_benchmark_data(benchmark_fp)
        
        # ensure everything is in array format
        self.period_starts = np.array(pd.to_datetime(self.period_starts))
        self.period_ends = np.array(pd.to_datetime(self.period_ends))
        self.data = np.array(self.data)
        return 
    
    def get_model_mb(self, ds):
        # grab only periods within the dataset
        valid_periods = np.where((self.period_starts >= ds.time.values[0]) & 
                                 (self.period_ends <= ds.time.values[-1]))[0]
        
        if len(valid_periods) == 0:
            self.mod = [np.nan ]
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

        self.mod = mod 
        self.meas = meas
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
            spring_dates = df.loc[(df['Year'] == year), 'spring_date'].values
            spring_date_mode = spring_dates[np.argmax(np.unique(spring_dates.astype(str), return_counts=True)[1])]
            spring_date = spring_date_mode if type(spring_date_mode) != str else f'{year}-04-01'
            
            fall_dates =  df.loc[(df['Year'] == year), 'fall_date'].values
            fall_date_mode = fall_dates[np.argmax(np.unique(fall_dates.astype(str), return_counts=True)[1])]
            fall_date = fall_date_mode if type(fall_date_mode) != str else f'{year}-09-01'
            dates[year].append(spring_date)
            dates[year].append(fall_date)

        # for site in df['site_name'].unique():
        #     print(site, df.loc[df['site_name'] == site, 'ba'].count())

        df = df.loc[df['site_name'] == self.site]

        # determine if there are sufficient seasonal data to compare summer/winter
        n_winter_obs = df['bw'].count()
        if n_winter_obs < min_n_winter or 'annual' in self.use:
            # insufficient data to compare seasonal, so grab annual periods
            index_data = np.where(~np.isnan(df['ba']))[0][1:]
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
                        annual_starts.append(f'{year-1}-09-01')
                else:
                    annual_starts.append(start) 
            annual_starts = pd.to_datetime(annual_starts)

            no_nans = np.where((~np.isnan(annual_starts) &
                                ~np.isnan(annual_ends)))[0]
            self.period_starts = annual_starts[no_nans]
            self.period_ends = annual_ends[no_nans]
            self.data = np.array(annual_data[no_nans])
            self.dataset = 'annual'

        else:
            # sufficient winter data to separate winter and summer periods
            index_data = np.where((~np.isnan(df['ba'])) & (~np.isnan(df['bw'])))[0]
            df_mb = df.iloc[index_data]
            if index_data[0] == 0:
                year_0 = df_mb.iloc[0]['Year'] - 1
                if year_0 in df['Year']:
                    first_last = df.loc[df['Year'] == year_0]
                    df_last = pd.concat([first_last, df.iloc[index_data[1:] - 1]])
                elif year_0 in dates:
                    first_last = df_mb.iloc[[0]].copy()
                    first_last.iloc[0, :] = np.nan
                    first_last.loc[first_last.index[0], 'fall_date'] = dates[year_0][1]
                    df_last = pd.concat([first_last, df.iloc[index_data[1:] - 1]])
                else:
                    df_last = df.iloc[index_data[1:] - 1]
                    df_mb = df.iloc[index_data[1:]]
            else:
                df_last = df.iloc[index_data - 1]

            # pull out winter ablation/summer accumulation
            this_winter_abl = df_mb['winter_ablation'].values
            past_summer_acc = df_last['summer_accumulation'].values
            this_summer_acc = df_mb['summer_accumulation'].values
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
    
    def plot_mb(self):
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

            ax.plot(ends[idx_summer], mod[idx_summer], color=colors[4])
            ax.plot(ends[idx_summer], meas[idx_summer], color='k', linestyle='--')
            ax.plot(ends[idx_winter], mod[idx_winter], color=colors[0])
            ax.plot(ends[idx_winter], meas[idx_winter], color='k', linestyle='--')
        else:
            ax.plot(ends, mod, color=colors[0])
            ax.plot(ends, meas, color='k', linestyle='--')

        ax.plot(np.nan, np.nan, color=colors[0], label='Modeled')
        ax.plot(np.nan, np.nan, color='k', linestyle='--', label='Measured')
        ax.set_title(f'{self.name} {self.site} {self.dataset} mass balance')
        ax.legend()
        return fig, ax

class SnowMelt():
    def __init__(self, name, site):
        self.name = name 
        self.site = site 

        # find site elevation 
        glacier_fp = home_fp + 'PEBSI/data/by_glacier/'
        site_df = pd.read_csv(glacier_fp + f'{self.name}/site_constants.csv', index_col='site')
        self.elevation = site_df.loc[site, 'elevation']

        # find rgi7 glacier number
        rgi7id = translate_rgi[name]
        folder = base_fp + rgi7id + '/'
        
        for fn in os.listdir(folder):
            if 'melt_extent_elev_percentile' in fn and not 'ea' in fn and not 'eos' in fn:
                fn_melt = fn 
            if 'snowline_elev_percentile' in fn and not 'ea' in fn and not 'eos' in fn:
                fn_snow = fn 
        
        df_snow = pd.read_csv(folder + fn_snow, parse_dates=True, index_col=0)
        df_melt = pd.read_csv(folder + fn_melt, parse_dates=True, index_col=0)
        
        # reindex
        df_snow = df_snow.reindex(pd.date_range(df_snow.index[0], df_snow.index[-1])).ffill()
        df_melt = df_melt.reindex(pd.date_range(df_melt.index[0], df_melt.index[-1])).ffill()

        self.sar_snow = {
            'min':self.elevation >= df_snow['snowline_elev_min_m'],  # True when the snowline is above the site
            'max':self.elevation >= df_snow['snowline_elev_max_m'],  # (True = snow)
            'med':self.elevation >= df_snow['snowline_elev_m'],
        }

        self.sar_melt = {
            'min':self.elevation <= df_melt['melt_extent_elev_min_m'],  # True when the melt extent is above the site
            'max':self.elevation <= df_melt['melt_extent_elev_max_m'],  # (True = melting)
            'med':self.elevation <= df_melt['melt_extent_elev_m'],
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
    
    def plot_snow_bool(self):
        fig, (ax, lax) = plt.subplots(2, height_ratios=(4, 1), figsize=(6, 4))

        data = np.vstack([self.mod_snow, self.sar_snow['min'],
                          self.sar_snow['med'], self.sar_snow['max']]) # 
        time_edges = np.concatenate([self.time, self.time[-1:] + (self.time[1] - self.time[0])])
        y_edges = np.array([0,1,1.333,1.6667,2])
                          
        mesh = ax.pcolormesh(
            time_edges,
            y_edges,
            data,
            cmap="gray",
            vmin=0,
            vmax=1,
            shading="flat",
        )

        ax.set_yticks([0.5, 1.16667, 1.5, 1.833333])
        ax.set_yticklabels(['Model','SAR min', 'SAR med','SAR max'])
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y'))
        for y in y_edges:
            ax.axhline(y, color='k', linewidth=0.3)

        lax.bar(np.nan, np.nan, color='k', label='Ice / Firn')
        lax.bar(np.nan, np.nan, color='white', edgecolor='k', linewidth=1, label='Snow')
        lax.legend(loc='center', bbox_to_anchor= (0.5,0.5))
        lax.axis('off')

        # ax.set_xticks([])
        plt.show()

    def plot_melt_bool(self):
        fig, (ax, lax) = plt.subplots(2, height_ratios=(4, 1), figsize=(6, 4))

        data = ~np.vstack([self.mod_melt, self.sar_melt['min'],
                          self.sar_melt['med'], self.sar_melt['max']]) # 
        time_edges = np.concatenate([self.time, self.time[-1:] + (self.time[1] - self.time[0])])
        y_edges = np.array([0,1,1.333,1.6667,2])
                          
        mesh = ax.pcolormesh(
            time_edges,
            y_edges,
            data,
            cmap="gray",
            vmin=0,
            vmax=1,
            shading="flat",
        )
        
        ax.set_yticks([0.5, 1.16667, 1.5, 1.833333])
        ax.set_yticklabels(['Model','SAR min', 'SAR med','SAR max'])
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y'))
        for y in y_edges:
            ax.axhline(y, color='k', linewidth=0.3)

        lax.bar(np.nan, np.nan, color='k', label='Wet snow / firn')
        lax.bar(np.nan, np.nan, color='white', edgecolor='k', linewidth=1, label='Dry')
        lax.legend(loc='center', bbox_to_anchor= (0.5,0.5))
        lax.axis('off')

        # ax.set_xticks([])
        plt.show()

class Albedo():
    def __init__(self, name, site, use='S2'):
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

        # get RGI7 glacier ID number
        glac_no = translate_rgi[name]
        self.glac_no = glac_no
        if name == 'kahiltna' and site == 'K14k':
            self.glac_no = '01.06469'

        # open dataframes
        metadata_df = pd.read_csv(home_fp + 'PEBSI/data/glacier_metadata.csv', index_col='name')
        glacier_fp = home_fp + 'PEBSI/data/by_glacier/'
        self.albedo_fp = base_fp + '../../rs/albedo/'

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
        # get filename for this glac_no
        if self.use == 'S2':
            albedo_fns = [self.albedo_fp + f'data_cube_s2_{self.glac_no[3:]}.nc']
            dtypes = [self.use]
        elif self.use == 'L8':
            albedo_fns = [self.albedo_fp + f'data_cube_l8_{self.glac_no[3:]}.nc']
            dtypes = [self.use]
        elif self.use == 'both':
            albedo_fns = [self.albedo_fp + f'data_cube_s2_{self.glac_no[3:]}.nc',
                          self.albedo_fp + f'data_cube_l8_{self.glac_no[3:]}.nc']
            dtypes = ['S2','L8']

        self.data = []
        self.time = []
        self.dtype = []
        for albedo_fn, dtype in zip(albedo_fns, dtypes):
            # open the dataset and get the proper CRS
            ds = xr.open_dataset(albedo_fn)
            crs = ds.spatial_ref.attrs['crs_wkt']
            self.epsg = crs.split('AUTHORITY["EPSG","')[-1].split('"]')[0]

            ds = ds['albedo'].rio.write_crs(crs).reset_coords(drop=True)
            self.ds = ds

            # select the point on the glacier 
            proj = Transformer.from_crs('EPSG:4326', f'EPSG:{self.epsg}', always_xy=True)
            x, y = proj.transform(self.lon, self.lat)
            da = ds.sel(x=x,y=y, method='nearest')
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
        return

    def get_model_albedo(self, ds):
        valid_steps = np.where((self.time >= ds.time.values[0]) & 
                                 (self.time <= ds.time.values[-1]))[0]
        
        if len(valid_steps) == 0:
            self.mod = np.nan 
            self.meas = np.nan
            return
        
        self.data = self.data[valid_steps]
        self.time = self.time[valid_steps]
        self.dtype = self.dtype[valid_steps]
        self.mod = ds.sel(time=self.time, method='nearest').albedo.values
        self.meas = self.data
        return

    def mae(self):
        return np.nanmean(np.abs(self.mod - self.meas))
    
    def bias(self):
        return np.nanmean(self.mod - self.meas)
    
    def rmse(self):
        return np.sqrt(np.nanmean(np.square(self.mod - self.meas)))

    def plot_map(self, time='mean', full=False, savefig=False,
                plot_sites = []):
        ds = self.ds
        ds = ds.squeeze('band')

        # grab the dataarray
        if time == 'mean':
            da = ds.mean(dim='time')
        else:
            if full:
                valid_count = ds.notnull().sum(dim=('x','y'))
                max_count = valid_count.max().values
                threshold = 0.9

                # identify time steps that meet the requirement
                good_times = valid_count / max_count >= threshold

                # extract the subset of times that pass the filter
                filtered_times = ds.time.where(good_times, drop=True)

                # select nearest image among the filtered times
                da = ds.sel(time=filtered_times.sel(time=time, method="nearest"))

            else:
                da = self.ds.sel(time=time, method='nearest')

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
        
        da.plot(ax=ax, cmap='Grays_r', vmin=0.2, vmax=0.9)

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
        
        ax.set_aspect('equal')
        if time == 'mean':
            time_fmtd = 'time mean'
        else:
            time_fmtd = pd.to_datetime(da.time.values).strftime('%d %b %Y')
        ax.set_title(f'{self.name.capitalize()} Glacier ({time_fmtd})')

        if savefig:
            time_fmtd = time_fmtd.replace(' ','-')
            plt.savefig(base_fp + f'{self.name}_{time_fmtd}.png', dpi=300, bbox_inches='tight')

        plt.show()
        return fig, ax
    
    def plot_timeseries(self):
        years = np.unique(pd.to_datetime(self.time).year)
        cmap = mpl.cm.get_cmap('viridis')
        norm = mpl.colors.Normalize(vmin=min(years),vmax=max(years))

        fig, ax = plt.subplots()
        ax.plot(np.nan, np.nan, linestyle='--', marker='.', color='gray', label='Observed')
        ax.plot(np.nan, np.nan, marker='.', color='gray', label='Modeled')
        for year in years:
            idx = np.where(pd.to_datetime(self.time).year == year)[0]
            doy = pd.to_datetime(self.time[idx]).day_of_year

            if self.use == 'both':
                idx_landsat = np.where(self.dtype[idx] == 'L8')[0]
                idx_sentinel = np.where(self.dtype[idx] == 'S2')[0]
                ax.plot(np.array(doy)[idx_landsat],np.array(self.meas[idx])[idx_landsat],color=cmap(norm(year)), marker='+', linestyle='--')
                ax.plot(np.array(doy)[idx_sentinel],np.array(self.meas[idx])[idx_sentinel],color=cmap(norm(year)), marker='^', linestyle='--')
            else:
                ax.plot(np.array(doy),np.array(self.meas[idx]),color=cmap(norm(year)), marker='*', linestyle='--')

            ax.plot(doy, self.mod[idx], marker='.', color=cmap(norm(year)), label=str(year))
        ax.set_ylabel('Albedo [-]')
        ax.set_xlabel('Day of year')
        ax.legend()
        ax.set_title(f'{self.name} {self.site}')
        plt.show()
        return fig, ax
    
    def plot_1to1(self):
        years = np.unique(pd.to_datetime(self.time).year)
        cmap = mpl.cm.get_cmap('viridis')
        norm = mpl.colors.Normalize(vmin=min(years),vmax=max(years))

        fig, ax = plt.subplots(figsize=(3.5, 3.5))

        if self.use == 'both':
            ax.scatter(np.nan, np.nan, marker='+', color='gray', label='Landsat')
            ax.scatter(np.nan, np.nan, marker='.', color='gray', label='Sentinel')

        for year in [2019,2022]: # years:
            idx = np.where(pd.to_datetime(self.time).year == year)[0]
            doy = pd.to_datetime(self.time[idx]).day_of_year
            mod = np.array(self.mod[idx]).ravel()

            if self.use == 'both':
                idx_landsat = np.where(self.dtype[idx] == 'L8')[0]
                idx_sentinel = np.where(self.dtype[idx] == 'S2')[0]
                ax.scatter(np.array(self.meas[idx])[idx_landsat],mod[idx_landsat],color=cmap(norm(year)), marker='+')
                ax.scatter(np.array(self.meas[idx])[idx_sentinel],mod[idx_sentinel],color=cmap(norm(year)), marker='.', label=str(year))
            else:
                ax.scatter(self.meas[idx],mod,color=cmap(norm(year)), marker='.', label=str(year))

        ax.plot([0, 1],[0,1],'k--')
        ax.set_xlim(0.2, 0.9)
        ax.set_ylim(0.2, 0.9)
        ax.set_ylabel('Modeled albedo [-]')
        ax.set_xlabel('Measured (RS) albedo [-]')
        ax.tick_params(length=5)
        ax.legend(bbox_to_anchor=(1.2, 0.5), loc='center')
        ax.set_title(f'{self.name} {self.site}')
        plt.show()
        # return fig, ax