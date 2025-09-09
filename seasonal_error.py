"""
Contains functions to get the seasonal mass
balance error and plot the seasonal mass
balance for a given glacier and site.

@author: clairevwilson
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

# User options
date_form = mpl.dates.DateFormatter('%b %d')
mpl.style.use('seaborn-v0_8-white')
USGS_fp = '../MB_data/GLACIER/Input_GLACIER_Glaciological_Data.csv'

all_colors = ['#63c4c7','#fcc02e','#4D559C','#60C252','#BF1F6A',
              '#F77808','#298282','#999999','#FF89B0','#427801']

# Objective function
def objective(model,data,method):
    if method == 'MSE':
        return np.nanmean(np.square(model - data))
    elif method == 'RMSE':
        return np.sqrt(np.nanmean(np.square(model - data)))
    elif method == 'MAE':
        return np.nanmean(np.abs(model - data))
    elif method == 'MdAE':
        return np.nanmedian(np.abs(model - data))
    elif method == 'ME':
        return np.nanmean(model - data)
    
# ========== 1. SEASONAL MASS BALANCE ==========
def seasonal_mass_balance(ds,method='MAE',out=None):
    """
    Compares seasonal mass balance measurements from
    USGS stake surveys to a model output.

    The stake data comes directly from the USGS Data
    Release (Input_Glaciological_Data.csv).
    """
    # Determine the site
    site = ds.attrs['site']
    glacier = ds.attrs['glacier'].capitalize()
    USGS_fp = f'../MB_data/{glacier}/Input_{glacier}_Glaciological_Data.csv'

    # Load dataset
    df_mb = pd.read_csv(USGS_fp)
    df_mb = df_mb.loc[df_mb['site_name'] == site]
    df_mb.index = df_mb['Year']

    # Get overlapping years
    years_model = np.unique(pd.to_datetime(ds.time.values).year)
    # if pd.to_datetime(f'{years_model[-1]}-08-01') not in ds.time.values:
    #     years_model = years_model[:-1]
    if pd.to_datetime(f'{years_model[0]-1}-08-01') not in ds.time.values:
        years_model = years_model[1:]

    years_measure = np.unique(df_mb.index)
    years = np.sort(list(set(years_model) & set(years_measure)))

    # Retrieve the model data
    mb_dict = {'bw':[],'bs':[],'ba':[]}
    for year in years:
        # Get sample dates
        spring_date = df_mb.loc[year,'spring_date']
        fall_date = df_mb.loc[year,'fall_date']
        if year-1 in df_mb.index:
            last_fall_date = df_mb.loc[year-1,'fall_date']
        else:
            last_fall_date = np.nan

        # Fill nans
        if str(spring_date) == 'nan':
            spring_date = str(year)+'-04-20 00:00'
        if str(fall_date) == 'nan':
            fall_date = str(year)+'-08-20 00:00'
        if str(last_fall_date) == 'nan':
            last_fall_date = str(year-1)+'-08-20 00:00'

        # Split into winter and summer
        summer_dates = pd.date_range(spring_date,fall_date,freq='h')
        winter_dates = pd.date_range(last_fall_date,spring_date,freq='h')
        annual_dates = pd.date_range(last_fall_date,fall_date,freq='h')
        if pd.to_datetime(ds.time.values[0]).minute == 30:
            summer_dates += pd.Timedelta(minutes=30)
            winter_dates += pd.Timedelta(minutes=30)
            annual_dates += pd.Timedelta(minutes=30)

        # Sum model mass balance
        if summer_dates[-1] not in ds.time.values:
            years_summer = years[:-1]
            summer_dates = pd.date_range(summer_dates[0], ds.time.values[-1],freq='h')
            annual_dates = pd.date_range(annual_dates[0], ds.time.values[-1],freq='h')
        else:
            years_summer = years
        wds = ds.sel(time=winter_dates).sum()
        sds = ds.sel(time=summer_dates).sum()
        ads = ds.sel(time=annual_dates).sum()
        winter_mb = wds.accum + wds.refreeze - wds.melt
        internal_acc = ds.sel(time=summer_dates[-2]).cumrefreeze.values
        summer_mb = sds.accum + sds.refreeze - sds.melt - internal_acc
        annual_mb = ads.accum + ads.refreeze - ads.melt - internal_acc
        mb_dict['bw'].append(winter_mb.values)
        mb_dict['bs'].append(summer_mb.values)
        mb_dict['ba'].append(annual_mb.values)

    # Index mass balance data
    df_mb = df_mb.loc[years]
    this_winter_abl_data = df_mb['winter_ablation'].values
    past_summer_acc_data = np.append(np.zeros(1), df_mb['summer_accumulation'].values[:-1])
    this_summer_acc_data = df_mb['summer_accumulation'].values
    past_summer_acc_data[np.isnan(past_summer_acc_data)] = 0
    this_summer_acc_data[np.isnan(this_summer_acc_data)] = 0
    this_winter_abl_data[np.isnan(this_winter_abl_data)] = 0
    winter_data = df_mb['bw'].values - past_summer_acc_data + this_winter_abl_data
    summer_data = df_mb['ba'].values - df_mb['bw'].values + this_summer_acc_data
    annual_data = winter_data + summer_data

    # Clean up arrays
    winter_model = np.array(mb_dict['bw'])
    summer_model = np.array(mb_dict['bs'])
    annual_model = np.array(mb_dict['ba'])
    assert winter_model.shape == winter_data.shape
    assert summer_model.shape == summer_data.shape    
    assert annual_model.shape == annual_data.shape   

    # Check if summer is missing for last summer
    if len(years_summer) != len(years):
        summer_model = summer_model[:-1] 
        summer_data = summer_data[:-1]

    # Assess error
    if isinstance(method, str) and out is None:
        winter_error = objective(winter_model,winter_data,method) 
        summer_error = objective(summer_model,summer_data,method) 
        annual_error = objective(annual_model,annual_data,method)
        return winter_error, summer_error, annual_error
    elif out == 'data':
        return years,winter_model,winter_data,summer_model,summer_data,annual_model,annual_data
    else:
        out_dict = {'winter':[],'summer':[],'annual':[]}
        for mm in method:
            out_dict['winter'].append(objective(winter_model,winter_data,mm))
            out_dict['summer'].append(objective(summer_model,summer_data,mm))
            out_dict['annual'].append(objective(annual_model,annual_data,mm))
        return out_dict

def plot_seasonal_mass_balance(ds,plot_ax=False,plot_var='mb',color='default'):
    """
    plot_var : 'mb' (default), 'bw','bs','ba'
    """
    # Determine the site
    site = ds.attrs['site']
    glacier = ds.attrs['glacier'].capitalize()
    USGS_fp = f'../MB_data/{glacier}/Input_{glacier}_Glaciological_Data.csv'

    # Make or get plot ax
    if plot_ax:
        ax = plot_ax
    else:
        fig,ax = plt.subplots()
    
    # Load dataset
    df_mb = pd.read_csv(USGS_fp)
    df_mb = df_mb.loc[df_mb['site_name'] == site]
    df_mb.index = df_mb['Year']

    # Get overlapping years
    years_model = np.unique(pd.to_datetime(ds.time.values).year)
    if pd.to_datetime(f'{years_model[0]-1}-08-01') not in ds.time.values:
        years_model = years_model[1:]
    years_measure = np.unique(df_mb.index)
    years = np.sort(list(set(years_model) & set(years_measure)))

    # Retrieve the model data
    mb_dict = {'bw':[],'bs':[],'ba':[]}
    previous_internal = 0
    for year in years:
        spring_date = df_mb.loc[year,'spring_date']
        fall_date = df_mb.loc[year,'fall_date']
        if year-1 in df_mb.index:
            last_fall_date = df_mb.loc[year-1,'fall_date']
        else:
            last_fall_date = np.nan
        # Fill nans
        if str(spring_date) == 'nan':
            spring_date = str(year)+'-04-20 00:00'
        if str(fall_date) == 'nan':
            fall_date = str(year)+'-08-20 00:00'
        if str(last_fall_date) == 'nan':
            last_fall_date = str(year-1)+'-08-20 00:00'
        # Split into winter and summer
        summer_dates = pd.date_range(spring_date,fall_date,freq='h')
        winter_dates = pd.date_range(last_fall_date,spring_date,freq='h')
        annual_dates = pd.date_range(last_fall_date,fall_date,freq='h')
        if pd.to_datetime(ds.time.values[0]).minute == 30:
            summer_dates += pd.Timedelta(minutes=30)
            winter_dates += pd.Timedelta(minutes=30)
            annual_dates += pd.Timedelta(minutes=30)

        # Sum model mass balance
        if summer_dates[-1] not in ds.time.values:
            years_summer = years[:-1]
            summer_dates = pd.date_range(summer_dates[0], ds.time.values[-1],freq='h')
            annual_dates = pd.date_range(annual_dates[0], ds.time.values[-1],freq='h')
        else:
            years_summer = years
        wds = ds.sel(time=winter_dates).sum()
        sds = ds.sel(time=summer_dates).sum()
        ads = ds.sel(time=annual_dates).sum()
        winter_mb = wds.accum + wds.refreeze - wds.melt
        internal_acc = ds.sel(time=summer_dates[-2]).cumrefreeze.values
        internal_acc = ds.sel(time=summer_dates[-2]).cumrefreeze.values - previous_internal
        previous_internal = internal_acc
        summer_mb = sds.accum + sds.refreeze - sds.melt - internal_acc
        annual_mb = ads.accum + ads.refreeze - ads.melt - internal_acc
        mb_dict['bw'].append(winter_mb.values)
        mb_dict['bs'].append(summer_mb.values)
        mb_dict['ba'].append(annual_mb.values)

    # Index mass balance data
    df_mb = df_mb.loc[years]
    this_winter_abl_data = df_mb['winter_ablation'].values
    past_summer_acc_data = np.append(np.zeros(1), df_mb['summer_accumulation'].values[:-1])
    this_summer_acc_data = df_mb['summer_accumulation'].values
    past_summer_acc_data[np.isnan(past_summer_acc_data)] = 0
    this_summer_acc_data[np.isnan(this_summer_acc_data)] = 0
    this_winter_abl_data[np.isnan(this_winter_abl_data)] = 0
    winter_data = df_mb['bw'].values - past_summer_acc_data + this_winter_abl_data
    summer_data = df_mb['ba'].values - df_mb['bw'].values + this_summer_acc_data
    annual_data = winter_data + summer_data

    cannual = 'orchid'
    if color == 'default' and plot_var == 'mb':
        cwinter = 'turquoise'
        csummer = 'orange'
    elif color != 'default':
        cwinter = color
        csummer = color
    elif plot_var == 'bw':
        cwinter = color
    elif plot_var == 'bs':
        csummer = color

    if len(years_summer) != len(years):
        summer_data = summer_data[:-1]
        mb_dict['bs'] = np.array(mb_dict['bs'])[:-1]
    if plot_var in ['mb','bw']:
        ax.plot(years,mb_dict['bw'],label='Winter',color=cwinter,linewidth=2)
        ax.plot(years,winter_data,'o--',color=cwinter,)
    if plot_var in ['mb','bs']:
        ax.plot(years_summer,mb_dict['bs'],label='Summer',color=csummer,linewidth=2)
        ax.plot(years_summer,summer_data,'o--',color=csummer)
    if plot_var in ['ba']:
        ax.plot(years,mb_dict['ba'],color=cannual,linewidth=2)
        ax.plot(years,annual_data,color=cannual,linestyle='--')
    ax.axhline(0,color='grey',linewidth=0.5)
    if plot_var in ['mb','bw','bs']:
        min_all = np.nanmin(np.concat([mb_dict['bw'],mb_dict['bs'],winter_data,summer_data]))
        max_all = np.nanmax(np.concat([mb_dict['bw'],mb_dict['bs'],winter_data,summer_data]))
    else:
        min_all = np.nanmin(np.array([mb_dict['ba'],annual_data]))
        max_all = np.nanmax(np.array([mb_dict['ba'],annual_data]))
    ax.set_xticks(np.arange(years[0],years[-1],4))
    ax.set_yticks(np.arange(np.round(min_all,0),np.round(max_all,0)+1,1))
    ax.set_ylabel('Seasonal mass balance (m w.e.)',fontsize=14)
    ax.plot(np.nan,np.nan,linestyle='--',color='grey',label='Data')
    ax.plot(np.nan,np.nan,color='grey',label='Modeled')
    ax.legend(fontsize=12,ncols=2)
    ax.tick_params(labelsize=12,length=5,width=1)
    if len(years) > 5:
        ax.set_xlim(years[0],years[-1])
    else:
        ax.set_xlim(years[0]-1, years[-1]+1)
    ax.set_ylim(min_all-0.5,max_all+0.5)
    ax.set_xticks(np.arange(years[0],years[-1],4))
    if plot_ax:
        return ax
    else:
        return fig,ax