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
import os

# User options
date_form = mpl.dates.DateFormatter('%b %d')
mpl.style.use('seaborn-v0_8-white')
USGS_fp = '../MB_data/GLACIER/Input_GLACIER_Glaciological_Data.csv'

all_colors = ['#63c4c7','#fcc02e','#4D559C','#60C252','#BF1F6A',
              '#F77808','#298282','#999999','#FF89B0','#427801']

# Objective function
def objective(model,data,method):
    if len(model) > 0:
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
    else:
        return []

def get_glacier(site):
    if site in ['KQU','KPS']:
        return 'kahiltna'
    elif site in ['Z','T']:
        return 'gulkana'
    elif site in ['EC']:
        return 'wolverine'
    
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
        spring_date = df_mb.loc[year,'spring_date']
        fall_date = df_mb.loc[year,'fall_date']
        if year-1 in df_mb.index:
            last_fall_date = df_mb.loc[year-1,'fall_date']
        else:
            last_fall_date = np.nan
        
        # Fill nans
        if str(spring_date) == 'nan':
            if glacier == 'Gulkana':
                spring_date = str(year)+'-04-20 00:00'
            else:
                spring_date = str(year) + '-05-20 00:00'
        if str(fall_date) == 'nan':
            if glacier == 'Gulkana':
                fall_date = str(year)+'-08-20 00:00'
            else:
                fall_date = str(year)+ '-09-20 00:00'
        if str(last_fall_date) == 'nan':
            if glacier == 'Gulkana':
                last_fall_date = str(year-1)+'-08-20 00:00'
            else:
                last_fall_date = str(year-1) + '-09-20 00:00'
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
        if winter_dates[0] not in ds.time.values:
            winter_dates = pd.date_range(ds.time.values[0], winter_dates[-1], freq='h')
        elif winter_dates[-1] not in ds.time.values:
            winter_dates = pd.date_range(winter_dates[0], ds.time.values[-1], freq='h')
        wds = ds.sel(time=winter_dates).sum()
        if len(summer_dates) == 0:
            sds = ds.isel(time=-1)
            internal_acc = sds.layerrefreeze.sum(dim='layer').max().values / 1000
        else:
            sds = ds.sel(time=summer_dates).sum()
            internal_acc = ds.sel(time=summer_dates).layerrefreeze.sum(dim='layer').max().values / 1000
        ads = ds.sel(time=annual_dates).sum()
        winter_mb = wds.accum + wds.refreeze - wds.melt
        # internal_acc = ds.sel(time=summer_dates[-2]).cumrefreeze.values
        # print(site, 'winter days',(winter_dates[-1] - winter_dates[0]).days, winter_dates[0], winter_dates[-1],'summer', (summer_dates[-1] - summer_dates[0]).days)
        # internal_acc = ds.sel(time=summer_dates[-2]).cumrefreeze.values - previous_internal
        # previous_internal = internal_acc
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
        annual_model = annual_model[:-1]
        annual_data = annual_data[:-1]

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
            if glacier == 'Gulkana':
                spring_date = str(year)+'-04-20 00:00'
            else:
                spring_date = str(year) + '-05-20 00:00'
        if str(fall_date) == 'nan':
            if glacier == 'Gulkana':
                fall_date = str(year)+'-08-20 00:00'
            else:
                fall_date = str(year)+ '-09-20 00:00'
        if str(last_fall_date) == 'nan':
            if glacier == 'Gulkana':
                last_fall_date = str(year-1)+'-08-20 00:00'
            else:
                last_fall_date = str(year-1) + '-09-20 00:00'
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
        if winter_dates[0] not in ds.time.values:
            winter_dates = pd.date_range(ds.time.values[0], winter_dates[-1], freq='h')
        elif winter_dates[-1] not in ds.time.values:
            winter_dates = pd.date_range(winter_dates[0], ds.time.values[-1], freq='h')
        wds = ds.sel(time=winter_dates).sum()
        if len(summer_dates) == 0:
            sds = ds.isel(time=-1)
            internal_acc = sds.layerrefreeze.sum(dim='layer').max().values / 1000
        else:
            sds = ds.sel(time=summer_dates).sum()
            internal_acc = ds.sel(time=summer_dates).layerrefreeze.sum(dim='layer').max().values / 1000
        ads = ds.sel(time=annual_dates).sum()
        winter_mb = wds.accum + wds.refreeze - wds.melt
        # internal_acc = ds.sel(time=summer_dates[-2]).cumrefreeze.values
        # print(site, 'winter days',(winter_dates[-1] - winter_dates[0]).days, winter_dates[0], winter_dates[-1],'summer', (summer_dates[-1] - summer_dates[0]).days)
        # internal_acc = ds.sel(time=summer_dates[-2]).cumrefreeze.values - previous_internal
        # previous_internal = internal_acc
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
        annual_model = annual_model[:-1]
        annual_data = annual_data[:-1]
        summer_mb = sds.accum + sds.refreeze - sds.melt - internal_acc
        annual_mb = ads.accum + ads.refreeze - ads.melt - internal_acc
        mb_dict['bw'].append(winter_mb.values)
        mb_dict['bs'].append(summer_mb.values)
        mb_dict['ba'].append(annual_mb.values)

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

    if plot_var in ['mb','bw']:
        if len(winter_model) > len(years):
            winter_model = winter_model[:-1]
        ax.plot(years,winter_model,label='Winter',color=cwinter,linewidth=2,marker='^')
        ax.plot(years,winter_data,'o--',color=cwinter,)
    if plot_var in ['mb','bs']:
        if len(summer_model) > len(summer_model):
            summer_model= summer_model[:-1]
        ax.plot(years_summer,summer_model,label='Summer',color=csummer,linewidth=2,marker='^')
        ax.plot(years_summer,summer_data,'o--',color=csummer)
    if plot_var in ['ba']:
        ax.plot(years_summer,mb_dict['ba'],color=cannual,linewidth=2,marker='^')
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
    ax.plot(np.nan,np.nan,color='grey',label='Modeled',marker='^')
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

def firn_cores(ds, out='mean_error', method='MAE'):
    all_years = np.arange(1980,2026)
    years_model = np.unique(pd.to_datetime(ds.time.values).year)
    year_0 = pd.to_datetime(ds.time.values[0]).year
    if pd.to_datetime(f'{year_0}-04-20') not in ds.time.values:
        years_model = years_model[1:]
    potential_years = np.sort(list(set(years_model) & set(all_years)))

    # Open the cores for this site
    site = ds.attrs['site']
    glacier = get_glacier(site)
    cores_fp = '/trace/home/cvwilson/INTERN/CommunityFirnModel/Data/cores/'
    cores_fp += glacier + '/'
    all_files = os.listdir(cores_fp)

    # list the dates of cores
    core_dates = {}
    core_years = []
    for fn in all_files:
        # filter out any mass balance sheets / snowdepth timeseries
        # and make sure it's the right site
        if 'snowdepth' not in fn and 'csv' in fn and site in fn:
            date = fn.split(site)[-1][1:-4]
            # only append spring cores
            if '_04_' in date or '_05_' in date:
                core_years.append(int(date[:4]))
                core_dates[int(date[:4])] = date

    # Storage for errors
    if out != 'data':
        error_dict = {}
        if type(method) == str:
            method = [method]
        for var in ['firndensity_']:
            for m in method:
                error_dict[var+m] = {}

    # Loop through years with cores
    data = {'firndensity_mod':[],'firndensity_meas':[], 'depths':[]}
    years = np.sort(list(set(potential_years) & set(core_years)))
    for year in years:
        # get date in that year
        date = core_dates[year]
        assert str(year) in date, 'got the wrong date'

        # Load data
        core = pd.read_csv(cores_fp + f'{glacier}{site}_{date}.csv')
        sbd = core['SBD']
        depth_meas = sbd - core['length'] / 2
        dens_meas = core['density']

        # Load dataset on the date the snowpit was taken
        date_time = pd.to_datetime(date.replace('_','/'))
        if date_time in ds.time.values:
            dsyear = ds.sel(time=date_time)
        else:
            dsyear = ds.sel(time=date_time, method='nearest')

        # Calculate layer density
        ldz = dsyear.layerheight.values
        depth_mod = np.array([np.sum(ldz[:i+1])-(ldz[i]/2) for i in range(len(ldz))])
        dens_mod = dsyear['layerdensity'].values

        # Interpolate density to the measured depths
        dens_interp = np.interp(depth_meas,depth_mod,dens_mod)

        # Store in timeseries
        data['depths'].append(np.array(depth_meas.values))
        data['firndensity_meas'].append(np.array(dens_meas.values))
        data['firndensity_mod'].append(np.array(dens_interp))

        # Calculate error from mass and density 
        if out != 'data':
            for m in method:
                density_error = objective(dens_interp, dens_meas, m) # kg/m3
                error_dict['firndensity_'+m][date] = density_error

    if out != 'data':
        # Aggregate to time mean
        for var in error_dict:
            data_arr = list(error_dict[var].values())
            error_dict[var]['mean'] = np.mean(data_arr)

    if out == 'data':
        return list(core_dates.values()), data
    elif out == 'mean_error':
        dict_out = {}
        for var in error_dict:
            dict_out[var] = error_dict[var]['mean']
        return dict_out
    else:
        return error_dict

def plot_firn_cores(dates, data, site):
    n_panels = len(dates) + 1
    fig, all_axes = plt.subplots(1, n_panels, sharex=True, sharey=True, figsize=(n_panels*1.5, 4))
    idx = np.arange(len(dates))
    cax = all_axes[-1]
    axes = all_axes[:-1]
    
    max_depths =[]
    for i, ax, date in zip(idx, axes, dates):
        depth = data['depths'][i]
        density_meas = data['firndensity_meas'][i]
        density_mod = data['firndensity_mod'][i]

        ax.plot(density_meas, depth, color='k', linestyle='--')
        ax.plot(density_mod, depth, color=all_colors[i])
        ax.invert_yaxis()
        ax.set_title(date.replace('_','-'))
        ax.tick_params(length=5)
        max_depths.append(max(depth))
        ax.set_xticks([300, 600])
    ax.set_ylim(max(max_depths), 0)
    cax.plot(np.nan,np.nan,linestyle='--',color='k',label='Data')
    cax.plot(np.nan,np.nan,color='grey',label='Modeled')
    cax.legend(loc='center')
    cax.axis('off')
    axes[0].set_ylabel('Depth below surface [m]')
    fig.supxlabel('Density [kg m$^{-3}$]', y=-0.03)
    glacier = get_glacier(site)
    fig.suptitle(glacier.capitalize() +' '+site)
    return fig, axes