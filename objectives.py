"""
Contains functions to compare a model output dataset (ds)
to a field dataset. The field datasets included:

1. Seasonal mass balance from stake measurements
2. Surface height change (cumulative mass balance) from GNSS-IR
3. Snow temperatures from iButtons
4. Albedo timeseries from automatic weather station
5. Spectral albedo from FieldSpec

All functions use the following notation for the arguments:
    site : str
        Field site being examined
    ds : xarray.Dataset
        Output dataset from PyGEM-EB
    method : str, default 'MAE'
        Choose between 'RMSE','MAE','MdAE','ME','MSE'
    plot : Bool, default False
        Plot the result

@author: clairevwilson
"""
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import pickle
import itertools

# User options
date_form = mpl.dates.DateFormatter('%b %d')
mpl.style.use('seaborn-v0_8-white')
GNSSIR_fp = '../MB_data/Stakes/gulkanaSITE24_GNSSIR.csv'
USGS_fp = '../MB_data/Gulkana/Input_Gulkana_Glaciological_Data.csv'

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

    # Load dataset
    df_mb = pd.read_csv(USGS_fp)
    df_mb = df_mb.loc[df_mb['site_name'] == site]
    df_mb.index = df_mb['Year']

    # Get overlapping years
    years_model = np.unique(pd.to_datetime(ds.time.values).year)
    if pd.to_datetime(f'{years_model[-1]}-08-01') not in ds.time.values:
        years_model = years_model[:-1]

    years_measure = np.unique(df_mb.index)
    years = np.sort(list(set(years_model) & set(years_measure)))

    # Retrieve the model data
    mb_dict = {'bw':[],'bs':[],'ba':[]}
    for year in years[1:]:
        # Get sample dates
        spring_date = df_mb.loc[year,'spring_date']
        fall_date = df_mb.loc[year,'fall_date']
        last_fall_date = df_mb.loc[year-1,'fall_date']

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
            summer_dates = pd.date_range(summer_dates[0], ds.time.values[-1],freq='h')
            annual_dates = pd.date_range(annual_dates[0], ds.time.values[-1],freq='h')
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
    this_winter_abl_data = df_mb['winter_ablation'].iloc[1:].values
    past_summer_acc_data = df_mb['summer_accumulation'].iloc[:-1].values
    this_summer_acc_data = df_mb['summer_accumulation'].iloc[1:].values
    past_summer_acc_data[np.isnan(past_summer_acc_data)] = 0
    this_summer_acc_data[np.isnan(this_summer_acc_data)] = 0
    this_winter_abl_data[np.isnan(this_winter_abl_data)] = 0
    winter_data = df_mb['bw'].iloc[1:] - past_summer_acc_data + this_winter_abl_data
    summer_data = df_mb['ba'].iloc[1:] - df_mb['bw'].iloc[1:] + this_summer_acc_data
    annual_data = winter_data + summer_data

    # Clean up arrays
    winter_model = np.array(mb_dict['bw'])
    summer_model = np.array(mb_dict['bs'])
    annual_model = np.array(mb_dict['ba'])
    assert winter_model.shape == winter_data.shape
    assert summer_model.shape == summer_data.shape    
    assert annual_model.shape == annual_data.shape    

    # Assess error
    if isinstance(method, str) and out is None:
        winter_error = objective(winter_model,winter_data,method) 
        summer_error = objective(summer_model,summer_data,method) 
        annual_error = objective(annual_model,annual_data,method)
        return winter_error, summer_error, annual_error
    elif out == 'data':
        return years[1:],winter_model,winter_data,summer_model,summer_data,annual_model,annual_data
    else:
        out_dict = {'winter':[],'summer':[],'annual':[]}
        for mm in method:
            out_dict['winter'].append(objective(winter_model,winter_data,mm))
            out_dict['summer'].append(objective(summer_model,summer_data,mm))
            out_dict['annual'].append(objective(annual_model,annual_data,mm))
        return out_dict

def plot_seasonal_mass_balance(ds,plot_ax=False,label=None,plot_var='mb',color='default'):
    """
    plot_var : 'mb' (default), 'bw','bs','ba'
    """
    # Determine the site
    site = ds.attrs['site']

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
    years_measure = np.unique(df_mb.index)
    years = np.sort(list(set(years_model) & set(years_measure)))
    if site == 'A':
        years = years[:-1]

    # Retrieve the model data
    mb_dict = {'bw':[],'bs':[],'ba':[]}
    previous_internal = 0
    for year in years[1:]:
        spring_date = df_mb.loc[year,'spring_date']
        fall_date = df_mb.loc[year,'fall_date']
        last_fall_date = df_mb.loc[year-1,'fall_date']
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
            summer_dates = pd.date_range(summer_dates[0], ds.time.values[-1],freq='h')
            annual_dates = pd.date_range(annual_dates[0], ds.time.values[-1],freq='h')
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
    this_winter_abl_data = df_mb['winter_ablation'].iloc[1:].values
    past_summer_acc_data = df_mb['summer_accumulation'].iloc[:-1].values
    this_summer_acc_data = df_mb['summer_accumulation'].iloc[1:].values
    past_summer_acc_data[np.isnan(past_summer_acc_data)] = 0
    this_summer_acc_data[np.isnan(this_summer_acc_data)] = 0
    this_winter_abl_data[np.isnan(this_winter_abl_data)] = 0
    winter_data = df_mb['bw'].iloc[1:] - past_summer_acc_data + this_winter_abl_data
    summer_data = df_mb['ba'].iloc[1:] - df_mb['bw'].iloc[1:] + this_summer_acc_data
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

    years = years[1:]
    if plot_var in ['mb','bw']:
        ax.plot(years,mb_dict['bw'],label='Winter',color=cwinter,linewidth=2)
        ax.plot(years,winter_data,color=cwinter,linestyle='--')
    if plot_var in ['mb','bs']:
        ax.plot(years,mb_dict['bs'],label='Summer',color=csummer,linewidth=2)
        ax.plot(years,summer_data,color=csummer,linestyle='--')
    if plot_var in ['ba']:
        ax.plot(years,mb_dict['ba'],color=cannual,linewidth=2)
        ax.plot(years,annual_data,color=cannual,linestyle='--')
    ax.axhline(0,color='grey',linewidth=0.5)
    if plot_var in ['mb','bw','bs']:
        min_all = np.nanmin(np.array([mb_dict['bw'],mb_dict['bs'],winter_data,summer_data]))
        max_all = np.nanmax(np.array([mb_dict['bw'],mb_dict['bs'],winter_data,summer_data]))
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
    ax.set_xlim(years[0],years[-1])
    ax.set_ylim(min_all-0.5,max_all+0.5)
    ax.set_xticks(np.arange(years[0],years[-1],4))
    if plot_ax:
        return ax
    else:
        return fig,ax

# ========== 3. END-OF-WINTER SNOWPACK ==========
def snowpits(ds,method='MAE',out=None):
    all_years = np.arange(2000,2025)
    years_model = np.unique(pd.to_datetime(ds.time.values).year)
    year_0 = pd.to_datetime(ds.time.values[0]).year
    if pd.to_datetime(f'{year_0}-04-01') not in ds.time.values:
        years_model = years_model[1:]
    years = np.sort(list(set(years_model) & set(all_years)))

    # Open the mass balance
    with open('../MB_data/pits.pkl', 'rb') as file:
        site_profiles = pickle.load(file)

    # Determine and load the site
    site = ds.attrs['site']
    profiles = site_profiles[site]

    # Storage to determine error
    error_dict = {}
    if type(method) == str:
        method = [method]
    for var in ['snowmass_','snowdepth_','snowdensity_']:
        for m in method:
            error_dict[var+m] = {}

    # Loop through years
    data = {'snowdepth_mod':[],'snowdepth_meas':[],'snowdensity_mod':[],'snowdensity_meas':[]}
    for year in years:
        # Some years don't have data: skip
        if year in profiles['sbd']:
            # Load data
            sbd = profiles['sbd'][year]
            dens_meas = profiles['density'][year]

            # Load dataset on the date the snowpit was taken
            sample_date = profiles['date'][year]
            dsyear = ds.sel(time=pd.to_datetime(f'{year}-{sample_date}'))

            # Calculate layer density
            ldz = dsyear.layerheight.values
            depth_mod = np.array([np.sum(ldz[:i+1])-(ldz[i]/2) for i in range(len(ldz))])
            dens_mod = dsyear['layerdensity'].values

            # Find the snow depth
            snowdepth_mod = dsyear.snowdepth.values
            snowdepth_pit = sbd[~np.isnan(sbd)][-1]

            # Only consider modeled density up to the minimum snow depth
            min_depth = min(snowdepth_mod, snowdepth_pit)
            if np.any(sbd <= min_depth):
                dens_meas = dens_meas[sbd <= min_depth]
                sbd = sbd[sbd <= min_depth]
            else:
                # if the modeled snow depth is too small, take average measured density
                sbd = [min_depth]
                dens_meas = [np.mean(dens_meas)]

            # Interpolate modeled density to the snowpit depths
            dens_interp = np.interp(sbd,depth_mod,dens_mod)

            # Calculate mass of snow
            snow_idx = np.where(depth_mod < dsyear.snowdepth.values)[0]
            snowmass_mod = np.sum(dsyear.layerheight.values[snow_idx] * dsyear.layerdensity.values[snow_idx]) / 1000
            sample_heights = np.append(np.array([profiles['sbd'][year][0]]),np.diff(np.array(profiles['sbd'][year])))
            snowmass_meas = np.sum(profiles['density'][year] * sample_heights) / 1000

            # Store in timeseries
            data['snowdensity_meas'].append(np.array(dens_meas))
            data['snowdensity_mod'].append(np.array(dens_interp))
            data['snowdepth_meas'].append(snowdepth_pit)
            data['snowdepth_mod'].append(snowdepth_mod)

            # Calculate error from mass and density 
            for m in method:
                density_error = objective(dens_interp, dens_meas, m) # kg/m3
                mass_error = objective(snowmass_mod, snowmass_meas, m) # m w.e.
                depth_error = objective(snowdepth_mod, snowdepth_pit, m) # m

                # Store
                error_dict['snowmass_'+m][str(year)] = mass_error
                error_dict['snowdepth_'+m][str(year)] = depth_error
                error_dict['snowdensity_'+m][str(year)] = density_error
        else:
            data['snowdensity_meas'].append(np.array(np.nan))
            data['snowdensity_mod'].append(np.array(np.nan))
            data['snowdepth_meas'].append(np.nan)
            data['snowdepth_mod'].append(np.nan)

    # Aggregate to time mean
    for var in error_dict:
        data_arr = list(error_dict[var].values())
        error_dict[var]['mean'] = np.mean(data_arr)

    if out == 'data':
        for item in data:
            data[item] = data[item]
        return years, data
    elif out == 'mean_error':
        dict_out = {}
        for var in error_dict:
            dict_out[var] = error_dict[var]['mean']
        return dict_out
    else:
        return error_dict

# ========== 3. CUMULATIVE MASS BALANCE ==========
def cumulative_mass_balance(ds,method='MAE',out=None):
    """
    Compares cumulative mass balance measurements from
    a stake to a model output. 

    The stake data should be formatted as a .csv with two 
    columns: 'Date' and 'CMB' where 'CMB' is the surface 
    height change in meters.
    """
    # Load the site
    site = ds.attrs['site']

    # Update filepath
    fp_gnssir = GNSSIR_fp.replace('SITE',site)
    fp_stake = fp_gnssir.replace('GNSSIR','stake')

    # Load GNSSIR daily MB
    df_mb_dict = {}
    if os.path.exists(fp_gnssir):
        df_mb_daily = pd.read_csv(fp_gnssir)
        df_mb_daily.index = pd.to_datetime(df_mb_daily['Date'])
        df_mb_daily['CMB'] -= df_mb_daily['CMB'].iloc[0]
        df_mb_dict['GNSS_IR'] = df_mb_daily.sort_index()
    if os.path.exists(fp_stake):
        df_mb_daily = pd.read_csv(fp_stake)
        df_mb_daily.index = pd.to_datetime(df_mb_daily['Date'])
        df_mb_dict['stake'] = df_mb_daily.sort_index()

    if 'stake' in df_mb_dict:
        df_mb_daily = df_mb_dict['stake']
    if 'GNSS_IR' in df_mb_dict:
        df_mb_daily = df_mb_dict['GNSS_IR']

    # Get the summer mass balance
    if out== 'mbs':
        # Load USGS seasonal MB
        if site not in ['ABB','BD']:
            year = pd.to_datetime(ds.time.values[0]).year
            df_mb = pd.read_csv(USGS_fp)
            df_mb = df_mb.loc[df_mb['site_name'] == site]
            mba = df_mb.loc[df_mb['Year'] == year,'ba'].values[0]
            mbw = df_mb.loc[df_mb['Year'] == year,'bw'].values[0]
            mbsa = df_mb.loc[df_mb['Year'] == year,'summer_accumulation'].values[0]
            mbs_measured = mba - mbw + mbsa

            spring_date = df_mb.loc[df_mb['Year'] == year,'spring_date'].values[0]
            fall_date = df_mb.loc[df_mb['Year'] == year,'fall_date'].values[0]
            if fall_date not in ds.time.values:
                fall_date = ds.time.values[-1]
            if spring_date not in ds.time.values:
                spring_date = ds.time.values[0]

            # Retrieve modeled summer MB
            melt_dates = pd.date_range(spring_date,fall_date,freq='h')
            ds_summer = ds.sel(time=melt_dates)
            mbs_ds = ds_summer.accum + ds_summer.refreeze - ds_summer.melt
            internal_acc = ds.sel(time=melt_dates[-2]).cumrefreeze.values
            if internal_acc > 1e-5:
                print(f'Site {site} internal acc: {internal_acc:.5f} m w.e.')
            mbs_modeled = mbs_ds.sum().values - internal_acc

    if os.path.exists(fp_gnssir) or os.path.exists(fp_stake):
        # Retrieve the dates
        start = df_mb_daily.index[0]
        end = df_mb_daily.index[-1]
        assert ds.time.values[0] < end, 'Model run begins after field date period'
        assert ds.time.values[-1] > start, 'Model run ends before field date period'
        if start < ds.time.values[0]:
            start = pd.to_datetime(ds.time.values[0])
        if end > ds.time.values[-1]:
            end = pd.to_datetime(ds.time.values[-1])

        # Index daily data
        df_mb_daily = df_mb_daily.loc[start:end]
        df_mb_daily.index = pd.to_datetime(df_mb_daily.index)
        idx_data = []
        for i,date in enumerate(pd.date_range(start,end)):
            date = pd.to_datetime(date.date())
            if date in df_mb_daily.index:
                if ~np.isnan(df_mb_daily.loc[date,'CMB']):
                    idx_data.append(i)
                
        # Index model data
        if pd.to_datetime(ds.time.values[0]).minute == 30:
            if start.minute != pd.to_datetime(ds.time.values[0]).minute:
                start += pd.Timedelta(minutes=30)
            if end.minute != pd.to_datetime(ds.time.values[0]).minute:
                end -= pd.Timedelta(minutes=30)
        ds = ds.sel(time=pd.date_range(start,end,freq='h'))

        # Accumulation area sites: need only dh above stake depth
        # DONT NEED THIS IF INITIAL FIRN IS 5 M
        # if site in ['D','T']:
        #     dh = []
        #     stake_depth = 8.97 if site == 'T' else 7.856
        #     for hour in pd.date_range(start,end,freq='h'):
        #         ds_now = ds.sel(time=hour)
        #         lheight = ds_now.layerheight.values 
        #         ldepth = np.array([np.sum(lheight[:i+1])-(lheight[i]/2) for i in range(len(lheight))])
        #         layers = np.where(ldepth < stake_depth)[0]
        #         print(layers,stake_depth)
        #         height_now = np.sum(lheight[layers])
        #         if hour == start:
        #             height_before = height_now
        #         if len(layers) > 0:
        #             i = -1  # Start searching backwards
        #             while np.abs(height_now - height_before) > 0.5:
        #                 height_now = np.sum(lheight[layers[:i]] if i < 0 else lheight[:layers[-1] + i])
        #                 if (i < 0 and len(layers[:i]) < 1):
        #                     i = 0
        #                 elif (i > 0 and i >= 20):  # Stop if bounds are exceeded
        #                     break
        #                 i = i - 1 if i < 0 else i + 1
        #         dh.append(height_now - height_before)
        #         height_before = height_now
        #         stake_depth += ds_now.dh.values
        #     ds['dh'].values = dh

        # Cumululative sum
        ds = ds.sel(time=pd.date_range(start,end,freq='h'))
        ds['dh'].values = ds.dh.cumsum().values - ds.dh.isel(time=0).values
        # Select data daily
        ds = ds.sel(time=pd.date_range(start,end)).dh

        # Clean up arrays
        model = ds.values[idx_data]
        data = df_mb_daily['CMB'].values
        if model.shape != data.shape:
            idx_data = np.where(~np.isnan(df_mb_daily['CMB'].values))[0]
            data = df_mb_daily.iloc[idx_data]
            data = data['CMB'].values
            model = ds.values[idx_data]
        assert model.shape == data.shape

        # Assess error
        error = objective(model,data,method)
    else:
        print('No data available to compare')
        return

    if out == 'mbs':
        return mbs_modeled,mbs_measured
    elif out == 'data':
        return ds.time.values[idx_data], model, data
    else:
        return error
        
def plot_2024_mass_balance(ds,plot_ax=False,label='Model',color='default'):
    # Determine the site
    site = ds.attrs['site']

    # Update filepath
    fp_gnssir = GNSSIR_fp.replace('SITE',site)
    fp_stake = fp_gnssir.replace('GNSSIR','stake')
    year = '2024'

    # Load GNSSIR daily MB
    df_mb_dict = {}
    if os.path.exists(fp_gnssir):
        df_mb_daily = pd.read_csv(fp_gnssir)
        df_mb_daily.index = pd.to_datetime(df_mb_daily['Date'])
        df_mb_daily['CMB'] -= df_mb_daily['CMB'].iloc[0]
        df_mb_dict['GNSS_IR'] = df_mb_daily.sort_index()
    if os.path.exists(fp_stake):
        df_mb_daily = pd.read_csv(fp_stake)
        df_mb_daily.index = pd.to_datetime(df_mb_daily['Date'])
        df_mb_dict['stake'] = df_mb_daily.sort_index()

    # Retrieve the dates
    start = df_mb_dict[list(df_mb_dict.keys())[0]].index[0]
    if start < ds.time.values[0]:
        start = pd.to_datetime(ds.time.values[0])
    end = pd.to_datetime(ds.time.values[-1])

    # Get the summer mass balance
    # Load USGS seasonal MB
    if site not in ['ABB','BD']:
        year = pd.to_datetime(ds.time.values[0]).year
        df_mb = pd.read_csv(USGS_fp)
        df_mb = df_mb.loc[df_mb['site_name'] == site]
        mba = df_mb.loc[df_mb['Year'] == year,'ba'].values[0]
        mbw = df_mb.loc[df_mb['Year'] == year,'bw'].values[0]
        mbs_measured = mba - mbw

    # Retrieve modeled summer MB
    spring_date = str(year)+'-04-20 00:00'
    fall_date = str(year)+'-08-20 00:00'
    melt_dates = pd.date_range(spring_date,fall_date,freq='h')
    ds_summer = ds.sel(time=melt_dates)
    mbs_ds = ds_summer.accum + ds_summer.refreeze - ds_summer.melt
    internal_acc = ds.sel(time=melt_dates[-2]).cumrefreeze.values
    mbs_modeled = mbs_ds.sum().values - internal_acc

    # Cumululative sum of dh
    ds = ds.sel(time=pd.date_range(start,end,freq='h'))
    ds['dh'].values = ds.dh.cumsum().values - ds.dh.isel(time=0).values
    # Select data daily
    ds = ds.sel(time=pd.date_range(start,end)).dh

    # Plot
    if not plot_ax:
        fig,ax = plt.subplots(figsize=(3,6))
    else:
        ax = plot_ax
    
    # Plot model
    if color == 'default':
        color = plt.cm.Dark2(0)
    if label is None:
        ax.plot(ds.time.values,ds.values,color=color)
    else:
        ax.plot(ds.time.values,ds.values,color=color,label=label)
    
    # Plot gnssir
    if 'GNSS_IR' in df_mb_dict:
        df_mb_daily = df_mb_dict['GNSS_IR']
        ax.plot(df_mb_daily.index,df_mb_daily['CMB'],label='GNSS-IR',linestyle='--',color='black')
        # error bounds
        lower = df_mb_daily['CMB'] - df_mb_daily['sigma']
        upper = df_mb_daily['CMB'] + df_mb_daily['sigma']
        ax.fill_between(df_mb_daily.index,lower,upper,alpha=0.2,color='gray')

    # Plot stake
    if 'stake' in df_mb_dict:
        df_stake_daily = df_mb_dict['stake']
        df_stake_daily.index = pd.to_datetime(df_stake_daily.index)
        df_stake_daily['CMB'] -= df_stake_daily['CMB'].iloc[0]
        df_stake_daily = df_stake_daily.sort_index()
        ax.plot(df_stake_daily.index,df_stake_daily['CMB'],label='Banded stake',linestyle=':',color='gray')

    # Beautify
    ax.legend(fontsize=12)
    ax.xaxis.set_major_formatter(date_form)
    ax.set_xticks(pd.date_range(start,end,freq='MS'))
    ax.tick_params(labelsize=12,length=5,width=1)
    ax.set_xlim(start,end)
    ax.set_ylabel('Surface height change (m)',fontsize=14)
    if not plot_ax:
        return fig, ax
    else:
        return ax

# ========== 3. SNOW TEMPERATURES ==========
def snow_temperature(ds,method='RMSE',plot=False,plot_heights=[0.5]):
    """
    Compares modeled snow temperatures to measurements from
    iButton temperature sensors.

    The iButton data should be formatted as a .csv with 
    datetime in the first column and each sensor labeled 
    by its initial height above the ice in cm (eg."50").

    ** Additional option for method: 'ripe'
    Error returned is the difference in days between modeled 
    and measured ripening date (measured - modeled)

    ** Additional argument plot_heights:
    List of heights above the ice in meters to plot the timeseries.
    """
    # Determine the site
    site = ds.attrs['site']

    # Load dataset
    year = pd.to_datetime(ds.time.values[0]).year
    data_fp = f'../Data/iButtons/iButtons_{year}_{site}.csv'
    temp_df = pd.read_csv(data_fp,index_col=0)
    temp_df.index = pd.to_datetime(temp_df.index)
    temp_df = temp_df.resample('30min').interpolate()
    h0 = np.array(temp_df.columns).astype(float)/100
    h0 = np.max(h0) - h0

    # Retrieve the dates and index stake data
    start = pd.to_datetime(temp_df.index[0])
    end = pd.to_datetime(temp_df.index[-1])
    assert ds.time.values[0] < end, 'Model run begins after field date period'
    assert ds.time.values[-1] > start, 'Model run ended before field data period'
    start = max(start,ds.time.values[0])
    end = min(end,ds.time.values[-1])
    time = pd.date_range(start,end,freq='h')
    if time[0].minute != 30 and pd.to_datetime(ds.time.values[0]).minute == 30:
        time += pd.Timedelta(minutes=30)
    temp_df = temp_df.loc[time]

    # Initialize time loop
    store = {'measured':[],'modeled':[],'measure_plot':[],'model_plot':[]}
    for j,hour in enumerate(time):
        buried = np.arange(len(temp_df.columns))
        current_meas = temp_df.loc[hour].values
        if np.any(current_meas > 0):
            n_exposed = len(np.where(current_meas > 0)[0])
            buried = buried[n_exposed:]
            if len(buried) < 1:
                break
        # get temperatures of buried iButtons
        temp_measure = np.flip(current_meas[buried])
        height_measure = np.flip(h0[buried])
        
        # get modeled temperatures
        # index snow layers
        dens_model = ds.sel(time=hour)['layerdensity'].values
        dens_model[np.where(np.isnan(dens_model))[0]] = 1e5
        snow = np.where(dens_model < 700)[0]
        # include one extra layer for interpolating (will index out when stored)
        snow = np.append(snow,snow[-1]+1).ravel()
        # get height above ice
        lheight = ds.sel(time=hour)['layerheight'].values[snow]
        icedepth = np.sum(lheight[:-1]) + lheight[-2] / 2
        # get property and absolute depth
        ldepth = np.array([np.sum(lheight[:i+1])-(lheight[i]/2) for i in range(len(lheight))])
        temp_model = np.flip(ds.sel(time=hour)['layertemp'].values[snow])
        height_model = np.flip(icedepth - ldepth)
        
        # interpolate measured temperatures to model heights
        temp_measure_interp = np.interp(height_model[:-1],height_measure,temp_measure)
        
        # store x height above the ice for plotting
        temp_measure_plot = np.interp(plot_heights,height_measure,temp_measure)
        temp_model_plot = np.interp(plot_heights,height_model,temp_model)

        # store
        store['measured'].append(temp_measure_interp)
        store['modeled'].append(temp_model[1:])
        store['measure_plot'].append(temp_measure_plot)
        store['model_plot'].append(temp_model_plot)
    
    # Clean up outputs
    time = time[:j]
    flatten = lambda xss: np.array([x for xs in xss for x in xs])
    data = flatten(store['measured'])
    model = flatten(store['modeled'])
    assert model.shape == data.shape

    # Error
    error = objective(model,data,method)

    # Plot
    if plot:
        # get plotting data
        measure_plot = np.array(store['measure_plot'])
        model_plot = np.array(store['model_plot'])

        # initialize plots
        n = len(plot_heights)
        fig,ax = plt.subplots(figsize=(6,4),dpi=200,
                            layout='constrained',sharex=True,sharey=True)
        norm = mpl.colors.Normalize(vmin=0, vmax=n)
        cmap = mpl.cm.get_cmap('viridis')
        for i in range(n):
            ax.plot(time,measure_plot[:,i],linestyle='--',color=cmap(norm(i)))
            ax.plot(time,model_plot[:,i],color=cmap(norm(i)))
            ax.xaxis.set_major_formatter(date_form)
            ax.tick_params(length=5,labelsize=11)
            # ax.set_title(f'Initial height: {plot_heights[i]}',loc='right')
            ax.set_xlim(start,time[-1])
            ax.plot(np.nan,np.nan,color=cmap(norm(i)),label=f'{plot_heights[i]}')
        ax.plot(np.nan,np.nan,linestyle='--',color='gray',label='Measured')
        ax.plot(np.nan,np.nan,color='gray',label='Modeled')
        ax.legend(title='Initial height above ice (m)')
        fig.supylabel('Snow Temperature ($^{\circ}$C)',fontsize=12)
        fig.suptitle(f'{method} = {error:.3e} '+'$^{\circ}$C')
        plt.show()

    return error

# ========== 4. ALBEDO ==========
def daily_albedo(bds,method='MAE',out=None):
    df = pd.read_csv('/trace/home/cvwilson/research/climate_data/AWS/Processed/gulkana2024_bothalbedo.csv',index_col=0)
    df.index = pd.to_datetime(df.index) # - pd.Timedelta(hours=8)
    
    # Sample dataset to daily
    bds = bds.resample(time='d').mean()

    # Store albedo values
    daily_albedo = []

    # Manually specify summer dates and ice dates
    dates = pd.date_range('2024-04-20','2024-08-20',freq='d')
    for date in dates:
        start = pd.to_datetime(str(date.date())+' 12:00')
        end = pd.to_datetime(str(date.date())+' 16:00')
        # print(df.loc[start:end,'albedo'].values)
        daily_albedo.append(np.mean(df.loc[start:end,'albedo']))
    
    model = bds['albedo'].sel(time=dates).values
    data = np.array(daily_albedo)
    
    if out == 'data':
        return dates, model, data
    else:
        return objective(model, data, method)

def grid_plot(params_dict,summer_result,winter_result):
    """
    Parameters
    ----------
    params_dict : dict
        dict formatted as 'param_name':[option_1,option_2,...]
    result_summer and reuslt_winter : np.array
        should be size (N1xN2x...Nn) where:
            n is the number of param sets
            N is the number of options in each set
    """
    # Parse parameters
    param_names = list(params_dict.keys())
    combos = list(itertools.combinations(param_names,2))
    n_combos = len(combos)
    mid_idx = n_combos / 2
    if mid_idx % 1 != 0:
        mid_idx = int(mid_idx)

    # Create plot
    fig = plt.figure(figsize=(3*n_combos, 5))
    ratios = np.append(np.array([1]*n_combos).flatten(),0.3)
    gs = fig.add_gridspec(2,n_combos+1, wspace=1,hspace=0.5, width_ratios=ratios)

    # Normalize loss values across all three plots for shared colorbar
    all_summer = summer_result[~np.isnan(summer_result)].flatten()
    all_winter = winter_result[~np.isnan(winter_result)].flatten()
    norm_summer = plt.Normalize(vmin=np.min(all_summer), vmax=np.max(all_summer))
    norm_winter = plt.Normalize(vmin=np.min(all_winter), vmax=np.max(all_winter))
    cmap = 'viridis_r'

    # Create the scatter plots
    for j,pair in enumerate(combos):
        # Unpack the parameters
        param_1 = params_dict[pair[0]]
        param_2 = params_dict[pair[1]]

        # If parameter is a string, need to parse differently
        ylabels = False
        xlabels = False
        if type(param_1[0]) == str:
            xlabels = param_1
            param_1 = np.arange(len(param_1))
        if type(param_2[0]) == str:
            ylabels = param_2
            param_2 = np.arange(len(param_2))
        
        # Create meshgrid
        x,y = np.meshgrid(param_1,param_2)
        for i,season in enumerate(['winter','summer']):
            ax = fig.add_subplot(gs[i,j])
    
            if n_combos > 1:
                slices = [slice(None)] * n_combos
                slices[np.flip(np.arange(len(combos)))[j]] = mid_idx
                if season == 'summer':
                    result = summer_result[tuple(slices)]
                    norm = norm_summer
                elif season == 'winter':
                    result = winter_result[tuple(slices)]
                    norm = norm_winter
            else:
                if season == 'summer':
                    result = summer_result
                    norm = norm_summer
                elif season == 'winter':
                    result = winter_result
                    norm = norm_winter

            ax.scatter(x,y,c=result.T,cmap=cmap,s=500,norm=norm)
            ax.set_xlabel(pair[0],fontsize=12)
            ax.set_ylabel(pair[1],fontsize=12)
            ax.set_xticks(param_1)
            ax.set_yticks(param_2)
            ax.grid(True)
            if type(xlabels) != bool:
                ax.set_xticklabels(xlabels)
            if type(ylabels) != bool:
                ax.set_yticklabels(ylabels)

    # Add colorbars to each row
    cax1 = fig.add_subplot(gs[0,n_combos])
    cax2 = fig.add_subplot(gs[1,n_combos])
    axes = np.array(fig.get_axes()).reshape(2,n_combos+1)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm_winter, cmap=cmap),ax=axes[0,:-1], orientation='vertical',cax=cax1)
    cbar.set_label('Winter MAE',loc='top',fontsize=12)

    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm_summer, cmap=cmap), ax=axes[1,:-1], orientation='vertical',cax=cax2)
    cbar.set_label('Summer MAE',loc='top',fontsize=12)
    return fig, axes