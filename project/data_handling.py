import os
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

class Data():
    def __init__(self, name, site,
                 use = None):
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

        # find site attributes used in the model run 
        self.name = name
        self.site = site 
        site_df = pd.read_csv(glacier_fp + f'{self.name}/site_constants.csv', index_col='site')
        model_lat = site_df.loc[site, 'lat']
        model_lon = site_df.loc[site, 'lon']

        # types of data we have
        benchmark_glaciers = [f.lower() for f in os.listdir(benchmark_fp)]
        wgms_glaciers = wgms_df['glacier_name'].unique()

        # determine which type of data we have and put it in standard format
        if self.name.upper() in wgms_glaciers or use == 'wgms':
            # WGMS glacier
            name_fmtd = self.name.upper().replace('_',' ')
            df = wgms_df.loc[wgms_df['glacier_name'] == name_fmtd]
            self.df = df.loc[df['original_id'] == site]

            self.period_starts = self.df['begin_date']
            self.period_ends = self.df['end_date']
            self.data = self.df['balance']
            
        if self.name.replace('_','') in benchmark_glaciers or use == 'benchmark':
            # BENCHMARK glacier
            name_fmtd = sum([f.capitalize() for f in self.name.split('_')])
            data_fn = benchmark_fp + f'{name_fmtd}/Input_{name_fmtd}_Glaciological_Data.csv'
            df = pd.read_csv(data_fn, parse_dates=True)
            df = df.loc[df['site_name'] == site]

            # determine if there are sufficient seasonal data to compare summer/winter
            n_winter_obs = df['bw'].count()
            if n_winter_obs < 5:
                # insufficient data to compare seasonal, so grab annual periods
                index_fall = np.where(~np.isnan(df['ba']))[0][1:]
                annual_starts = df.loc[index_fall - 1, 'fall_date']
                annual_ends = df.loc[index_fall, 'fall_date']
                annual_data = df.loc[index_fall, 'ba']

                no_nans = np.where((~np.isnan(annual_starts) &
                                    ~np.isnan(annual_ends)))[0]
                self.period_starts = annual_starts[no_nans]
                self.period_ends = annual_ends[no_nans]
                self.data = annual_data[no_nans]

            else:
                # sufficient winter data to separate winter and summer periods
                index_fall = np.where(~np.isnan(df['ba']))[0][1:]
                # self.period_starts = pd.to_datetime()
                # self.period_ends = 

        self.period_starts = pd.to_datetime(self.period_starts)
        self.period_ends = pd.to_datetime(self.period_ends)

        return 
    
    def get_seasonal_mb(self, ds):
        # grab only periods within the dataset
        valid_periods = np.where((self.period_starts >= ds.time.values[0]) & 
                                 (self.period_ends <= ds.time.values[-1]))[0]
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
    