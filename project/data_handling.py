import os
import glob
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
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
            self.mod = np.nan 
            self.meas = np.nan
            return
        
        self.period_starts = self.period_starts[valid_periods]
        self.period_ends = self.period_ends[valid_periods]
        meas = self.data[valid_periods]

        mod = []
        for start, end in zip(self.period_starts, self.period_ends):
            mb_mod = ds.sel(time=slice(start, end)).MB.sum()
            mod.append(mb_mod)
        mod = np.array(mod)

        self.mod = mod 
        self.meas = meas
        return

    def mae(self):
        return np.mean(np.abs(self.mod - self.meas))
    
    def rmse(self):
        return np.sqrt(np.mean(np.square(self.mod - self.meas)))
    
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

    def get_benchmark_data(self, benchmark_fp, min_n_winter = 5):
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
            index_data = np.where((~np.isnan(df['ba'])) & (~np.isnan(df['bw'])))[0][1:]
            df_mb = df.iloc[index_data]
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
            winter_starts = pd.to_datetime(df_last['fall_date'].values)
            winter_ends = pd.to_datetime(df_mb['spring_date'].values)
            winter_data = df_mb['bw'].values - past_summer_acc + this_winter_abl
            no_winter_nans = np.where((~np.isnan(winter_starts) &
                                        ~np.isnan(winter_ends)))[0]

            self.period_starts = np.append(summer_starts[no_summer_nans],
                                            winter_starts[no_winter_nans])
            self.period_ends = np.append(summer_ends[no_summer_nans],
                                            winter_ends[no_winter_nans])
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
            switch_idx = np.where(np.diff(starts) < pd.Timedelta(days=0))[0][0] + 1
            idx_summer = range(switch_idx)
            idx_winter = range(switch_idx, len(starts))

            ax.plot(ends[idx_summer], mod[idx_summer], color=colors[4])
            ax.plot(ends[idx_summer], meas[idx_summer], color='k', linestyle='--')
            ax.plot(ends[idx_winter], mod[idx_winter], color=colors[0])
            ax.plot(ends[idx_winter], meas[idx_winter], color='k', linestyle='--')
        else:
            ax.plot(ends, mod, color=colors[0])
            ax.plot(ends, meas, color='k', linestyle='--')

        ax.plot(np.nan, np.nan, color=colors[0], label='Modeled')
        ax.plot(np.nan, np.nan, color='k', linestyle='--', label='Measured')
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

        self.sar_snow = self.elevation > df_snow['snowline_elev_m']  # True when snowline is above the site
        self.sar_melt = self.elevation > df_melt['melt_extent_elev_m']  # True when the melt extent is above the site
        return
    
    def get_model_snow(self, ds):
        daily_snow_depth = ds['layerheight'].where(ds['layertype'] < 2).sum(dim='layer').resample(time='1d').min()
        self.mod_snow = daily_snow_depth > 0.05

        daily_melt = ds['melt'].resample(time='1d').sum()
        self.mod_melt = daily_melt > 0.01

        mod_start = daily_melt.time.values[0]
        mod_end = daily_melt.time.values[-1]
        sar_start = self.sar_snow.index[0]
        sar_end = self.sar_snow.index[-1]
        start = max(mod_start, sar_start)
        end = min(mod_end, sar_end)

        self.sar_melt = self.sar_melt.loc[slice(start, end)].values
        self.sar_snow = self.sar_snow.loc[slice(start, end)].values
        self.mod_melt = self.mod_melt.sel(time=slice(start, end)).values
        self.mod_snow = self.mod_snow.sel(time=slice(start, end)).values
        self.time = pd.date_range(start, end, freq='1d')

        assert len(self.sar_melt) == len(self.mod_melt)
        assert len(self.sar_snow) == len(self.mod_snow)
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
        - penalty for unmatched model events (false positives)
        - penalty for missed observed events
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