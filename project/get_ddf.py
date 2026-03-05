import os
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import statsmodels.api as sm
import socket
if 'trace' in socket.gethostname():
    base_fp = '/trace/group/rounce/cvwilson/Output/ddf/'
    home_fp = '/trace/home/cvwilson/research/'
else:
    base_fp = 'C:/Users/cvw30/Research/Output/ddf/'
    home_fp = 'C:/Users/cvw30/Research/'

colors = ['#63c4c7','#fcc02e','#4D559C','#60C252','#BF1F6A',
              '#F77808','#298282','#999999','#FF89B0','#427801']

site_dict = {'kahiltna':['K53','K17b'], #  ,'K14C','KPS'
             'kennicott':['GTH','GTL'], # KENNICOTT   'KC31',
             'gulkana':['AU','B','D'], # GULKANA 
             'wolverine':['N','B','EC'], # WOLVERINE     
             'lemoncreek':['B','C','D'], # LEMON CREEK
             'taku':['MG1','NWB1','TKG3'], # TAKU
             }
coord_dict = {'wolverine':'60.5_-148.7',
                'kahiltna':'63.0_-151.2',
                'kennicott':'61.5_-143.1',
                'taku':'58.5_-134.3',
                'lemoncreek':'58.5_-134.3',
                'gulkana':'63.5_-145.625'}
date_dict = {'wolverine':'02_06',
                'kahiltna':'02_05',
                'kennicott':'02_09',
                'taku':'02_10',
                'lemoncreek':'', # DONT HAVE
                'gulkana':'02_03'}

glaciers = ['gulkana','wolverine','taku','kennicott']
include_OC = False

def get_ddf_df(ds_in, ds_bc, output_fn=None, plot_corr=False, savefig=False,
               n_rolling_bc = 7, n_rolling_acc = 3, n_rolling_pdds = 3):

    # Resample to daily
    time_res = '1d'
    daily_snow_depth = ds_in['layerheight'].where(ds_in['layertype'] < 2).sum(dim='layer').resample(time=time_res).min()
    daily_snow_temp = ds_in['layertemp'].where(ds_in['layertype'] < 2).isel(layer=slice(0,5)).min(dim='layer').resample(time=time_res).min()
    daily_melt = ds_in['melt'].resample({'time': time_res}).sum() * 1000
    daily_pdds = ds_in['airtemp'].resample({'time': time_res}).mean().where(lambda x: x > 0)
    daily_acc = ds_in['accum'].resample({'time': time_res}).sum() * 1000
    daily_rain = ds_in['rainfall'].resample({'time': time_res}).sum() * 1000
    daily_albedo = ds_in['albedo'].resample({'time': time_res}).min()
    snow_BC = ds_in['layerBC'].where(ds_in['layertype'] < 1).isel(layer=range(5)).max(dim='layer').resample(time=time_res).max()

    # Days since accumulation
    last = np.maximum.accumulate(np.where(daily_acc > 1e-3, np.arange(len(daily_acc)), -1))
    days_since_acc = np.arange(len(daily_acc)) - last
    days_since_acc = xr.DataArray(days_since_acc, dims=['time'], coords={'time': daily_acc.time})

    # calculate rolling DDF
    melt_rolling = daily_melt.rolling(time=5).sum()
    pdd_rolling = daily_pdds.rolling(time=5).sum()
    ddf = melt_rolling / pdd_rolling

    # clip to reasonable bounds
    ddf = ddf.where(pdd_rolling > np.nanquantile(pdd_rolling.values, 0.2)) # avoids small PDDs in early summer
    ddf = ddf.where(np.isfinite(ddf))  # avoids nans and infinity
    ddf = ddf.where(daily_melt > 1) # avoids small melt days
    ddf = ddf.where(daily_snow_depth > 1e-8) # avoids days with ice surface
    # ddf = ddf.where(daily_snow_temp > -1) # avoids days where snow is not ripe
    
    # start to build the dataset
    ds = xr.Dataset({
        'melt':daily_melt,
        'pdds':daily_pdds,
        'ddf':ddf,
        'accum':daily_acc,
        'days_since_accum':days_since_acc,
        'snow_BC':snow_BC,
        # 'snow_depth':daily_snow_depth
        'rain':daily_rain,
        'albedo':daily_albedo
        })

    # add deposition to dataset
    ds['bc_dep'] = ds_bc.resample(time=time_res).sum() * 3600 # kg m-2
    if include_OC:
        ds['oc_dep'] = ds_oc.resample(time=time_res).sum()

    # define variables to drop and to take cumsum
    drop_vars = ['lat','lon'] # ,'melt'
    cum_vars = ['bc_dep','accum','pdds']
    if include_OC:
        cum_vars += ['oc_dep']

    # add cumulative variables
    water_year = xr.where(ds['time.month'] >= 10, ds['time.year'] + 1, ds['time.year'])
    for var in cum_vars: 
        ds[f'{var}_cumsum'] = (ds[var].groupby(water_year).cumsum())
    drop_vars += cum_vars

    # add weekly rolling deposition (n days prior to each timestep)
    ds[f'bc_{n_rolling_bc}d_rolling'] = ds['bc_dep'].rolling(time=n_rolling_bc).sum()
    if include_OC:
        ds[f'oc_{n_rolling_bc}d_rolling'] = ds['oc_dep'].rolling(time=n_rolling_bc).sum()
    rolling_accum = f'accum_{n_rolling_acc}d_rolling'
    rolling_pdds = f'pdds_{n_rolling_pdds}d_rolling'
    ds[rolling_accum] = ds['accum'].rolling(time=n_rolling_acc).sum()
    ds[rolling_pdds] = ds['pdds'].rolling(time=n_rolling_pdds).sum()

    # clip to where PDDs have built up
    ds = ds.where(ds['pdds_cumsum'] > 50)
    ds = ds.where(ds['ddf'] < 100)

    # create dataframe and drop variables
    df = ds.to_dataframe().drop(columns=drop_vars)

    # rename and reorder columns
    df = df.rename(columns={'daily_snow_depth':'snow_depth'})
    first = ['ddf','days_since_accum','accum_cumsum',rolling_accum,'pdds_cumsum',rolling_pdds] # 'ddf_wBCOC', 'ddf_noBCOC', 
    df = df[first + [c for c in df.columns if c not in first]]

    # crop to usable days
    df = df[~df['ddf'].isna()]

    # save to csv
    if output_fn is not None:
        df.to_csv(output_fn)

    # plot the heatmap
    corr = df.corr()
    if plot_corr:
        plt.figure()
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        glacier = ds_in.attrs['glacier']
        site = ds_in.attrs['site']
        plt.title(f'Correlation Matrix for {glacier.capitalize()} {site}')
        plt.tight_layout()
        if savefig:
            plt.savefig(base_fp + f'{glacier}{site}_correlation.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    return df


if '__main__' in __name__:
    for glacier in glaciers:
        coords = coord_dict[glacier]
        date = date_dict[glacier]
        sites = site_dict[glacier]

        ds_bcd = xr.open_dataset(f'../../climate_data/MERRA2/{coords}/BCDP002_{coords}.nc')
        ds_bcw = xr.open_dataset(f'../../climate_data/MERRA2/{coords}/BCWT002_{coords}.nc')
        ds_bc = ds_bcd['BCDP002'] + ds_bcw['BCWT002']

        if include_OC:
            ds_ocd = xr.open_dataset(f'../../climate_data/MERRA2/{coords}/OCDP002_{coords}.nc')
            ds_ocw = xr.open_dataset(f'../../climate_data/MERRA2/{coords}/OCWT002_{coords}.nc')
            ds_oc = ds_ocd['OCDP002'] + ds_ocw['OCWT002'] 

        for site in sites:
            df_fn = base_fp+f'{glacier}{site}_df.csv'
            ds_in = xr.open_dataset(base_fp + f'{glacier}{site}_2026_{date}_base_long_0.nc') 
            df = get_ddf_df(ds_in, df_fn)