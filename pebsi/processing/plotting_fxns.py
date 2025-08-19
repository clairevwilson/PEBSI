import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mean_squared_error = lambda model,data: np.mean(np.square(model - data))

mpl.style.use('seaborn-v0_8-white')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] 
all_colors = ['#63c4c7','#fcc02e','#4D559C','#60C252','#BF1F6A',
              '#F77808','#298282','#999999','#FF89B0','#427801']
glac_props = {'01.00570':{'name':'Gulkana',
                            'AWS_fn':'gulkana1725_hourly.csv'},
            '01.01104':{'name':'Lemon Creek',
                            'AWS_fn':'LemonCreek1285_hourly.csv'},
            '01.16195':{'name':'South',
                            'AWS_fn':'Preprocessed/south/south2280_hourly_2008_wNR.csv'},
            '08.00213':{'name':'Storglaciaren',
                            'AWS_fn':'Storglaciaren/SITES_MET_TRS_SGL_dates_15MIN.csv'},
            '11.03674':{'name':'Saint-Sorlin',
                            'AWS_fn':'Preprocessed/saintsorlin/saintsorlin_hourly.csv'},
            '16.02444':{'name':'Artesonraju',
                            'AWS_fn':'Preprocessed/artesonraju/Artesonraju_hourly.csv'}}

varprops = {'surftemp':{'label':'Temperature','type':'Temperature','units':'C'},
            'airtemp':{'label':'Temperature','type':'Temperature','units':'C'},
           'melt':{'label':'Mass balance','type':'MB','units':'m w.e.'},
           'runoff':{'label':'Mass balance','type':'MB','units':'m w.e.'},
           'accum':{'label':'Mass balance','type':'MB','units':'m w.e.'},
           'refreeze':{'label':'Mass balance','type':'MB','units':'m w.e.'},
           'MB':{'label':'Mass balance','type':'MB','units':'m w.e.'},
           'meltenergy':{'label':'Heat fluxes','type':'Flux','units':'W m$^{-2}$'},
           'SWin':{'label':'Heat fluxes','type':'Flux','units':'W m$^{-2}$'},
           'SWout':{'label':'Heat fluxes','type':'Flux','units':'W m$^{-2}$'},
           'SWin_sky':{'label':'Heat fluxes','type':'Flux','units':'W m$^{-2}$'},
           'SWin_terr':{'label':'Heat fluxes','type':'Flux','units':'W m$^{-2}$'},
           'LWin':{'label':'Heat fluxes','type':'Flux','units':'W m$^{-2}$'},
           'LWout':{'label':'Heat fluxes','type':'Flux','units':'W m$^{-2}$'},
           'SWnet':{'label':'Heat fluxes','type':'Flux','units':'W m$^{-2}$'},
           'LWnet':{'label':'Heat fluxes','type':'Flux','units':'W m$^{-2}$'},
           'NetRad':{'label':'Heat fluxes','type':'Flux','units':'W m$^{-2}$'},
           'sensible':{'label':'Heat fluxes','type':'Flux','units':'W m$^{-2}$'},
           'latent':{'label':'Heat fluxes','type':'Flux','units':'W m$^{-2}$'},
           'rain':{'label':'Heat fluxes','type':'Flux','units':'W m$^{-2}$'},
           'ground':{'label':'Heat fluxes','type':'Flux','units':'W m$^{-2}$'},
           'layertemp':{'label':'Temperature','type':'Layers','units':'C'},
           'layerdensity':{'label':'Density','type':'Layers','units':'kg m$^{-3}$'},
           'layerwater':{'label':'Water content','type':'Layers','units':'kg m$^{-2}$'},
           'layerBC':{'label':'Black carbon','type':'Layers','units':'ppb'},
           'layerOC':{'label':'Organic carbon','type':'Layers','units':'ppb'},
           'layerdust':{'label':'Dust','type':'Layers','units':'ppm'},
           'layergrainsize':{'label':'Grain size','type':'Layers','units':'um'},
           'layerheight':{'label':'Layer height','type':'Layers','units':'m'},
           'layerrefreeze':{'label':'Layer refreeze','type':'Layers','units':'kg m-2'},
           'snowdepth':{'label':'Snow depth','type':'MB','units':'m'},
           'dh':{'label':'Surface height change','type':'MB','units':'m$'},
           'albedo':{'label':'Albedo','type':'Albedo','units':'-'},}
AWS_vars = {'temp':{'label':'Temperature','units':'C'},
             'wind':{'label':'Wind Speed','units':'m s$^{-1}$'},
             'rh':{'label':'Relative Humidity','units':'%'},
            'SWin':{'label':'Shortwave In','units':'W m$^{-2}$'},
             'LWin':{'label':'Longwave In','units':'W m$^{-2}$'},
             'sp':{'label':'Surface Pressure','units':'Pa'}}

def getds(file):
    ds = xr.open_dataset(file)
    start = pd.to_datetime(ds.indexes['time'].to_numpy()[0])
    end = pd.to_datetime(ds.indexes['time'].to_numpy()[-1])
    return ds,start,end

def simple_plot(ds,time,vars,res='d',t='',cumMB=True,
                skinny=True,save_fig=False,new_y=['None'],date_form=None):
    """
    Returns a simple timeseries plot of the variables as lumped in the input.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset object containing the model output
    vars : list-like
        List of strings where the variables to be plotted together are nested together
        e.g. [['airtemp','surftemp'],['SWnet','LWnet','sensible','latent']]
    time : list-like   
        Either len-2 list of start date, end date, or a list of datetimes
    res : str
        Abbreviated time resolution (e.g. '12h' or 'd')
    t : str
        Title for the figure
    skinny : Bool
        True or false, defines the height of each panel
    save_fig : Bool
        False or filepath to save the image
    new_y : list-like
        List of variables in vars that should be plotted on a new y-axis
    """
    h = 2 if skinny else 4
    fig,axes = plt.subplots(len(vars),1,figsize=(8,h*len(vars)),sharex=True,layout='constrained')

    if len(time) == 2:
        start = pd.to_datetime(time[0])
        end = pd.to_datetime(time[1])
        time = pd.date_range(start,end,freq='h')
    ds = ds.sel(time=time)
    ds_mean = ds.resample(time=res).mean(dim='time',keep_attrs='units')
    ds_sum = ds.resample(time=res).sum(dim='time',keep_attrs='units')
    c_iter = iter([plt.cm.Dark2(i) for i in range(8)])
    for i,v in enumerate(vars):
        if len(vars) > 1:
            axis = axes[i]
        else:
            axis = axes

        vararray = np.array(v)
        for var in vararray:
            try:
                c = next(c_iter)
            except:
                c_iter = iter([plt.cm.Dark2(i) for i in range(8)])
                c = next(c_iter)
        
            if var in ['melt','runoff','accum','refreeze','dh','MB'] and cumMB:
                var_to_plot = ds_sum[var].cumsum()
            elif 'layer' in var:
                var_to_plot = ds_mean[var].isel(layer=0)
            else:
                var_to_plot = ds_mean[var]

            if var in new_y:
                newaxis = axis.twinx()
                newaxis.plot(ds_mean.coords['time'],var_to_plot,color=c,label=var)
                newaxis.grid(False)
                newaxis.set_ylabel({varprops[var]['label']})
                newaxis.legend(bbox_to_anchor=(1.01,1.1),loc='upper left')
            else:
                axis.plot(ds_mean.coords['time'],var_to_plot,color=c,label=var)
                axis.set_ylabel(varprops[var]['label'])
        axis.tick_params(length=5)
        axis.legend(bbox_to_anchor=(1.01,1),loc='upper left')
    if date_form is None:
        date_form = mpl.dates.DateFormatter('%d %b')
    elif type(date_form) == str:
        date_form = mpl.dates.DateFormatter(date_form)
    axis.xaxis.set_major_formatter(date_form)
    fig.suptitle(t)
    axis.set_xlim(start,end)
    if save_fig:
        plt.savefig(save_fig,dpi=150)
    return fig, axes

def plot_hours(ds,vars,skinny=False,time=False):
    h = 1.5 if skinny else 2
    fig,axes = plt.subplots(len(vars),1,figsize=(7,h*len(vars)),sharex=True,layout='constrained')
    ds['hour'] = (['time'],pd.to_datetime(ds['time'].values).hour)

    if not time:
        start = ds.time.values[0]
        end = ds.time.values[-1]
        time = pd.date_range(start,end,freq='h')
    elif len(time) == 2:
        start = pd.to_datetime(time[0])
        end = pd.to_datetime(time[1])
        time = pd.date_range(start,end,freq='h')
    ds = ds.sel(time=time)

    for i,v in enumerate(vars):
        if len(vars) > 1:
            axis = axes[i]
        else:
            axis = axes
        vararray = np.array(v)
        c_iter = iter([all_colors[i] for i in range(8)])
        for v,var in enumerate(vararray):
            try:
                c = next(c_iter)
            except:
                c_iter = iter([all_colors[v] for v in range(8)])
                c = next(c_iter)
        
            var_hourly = []
            for hour in np.arange(24):
                ds_hour = ds.where(ds['hour'] == hour,drop=True)
                if 'layer' in var:
                    vardata = ds_hour.isel(layer=0)[var].to_numpy()
                else:
                    vardata = ds_hour[var].to_numpy()
                hourly_mean = np.mean(vardata)
                if 'melt' in vararray or 'MB' in vararray:
                    hourly_mean *= 1000
                var_hourly.append(hourly_mean)
            axis.plot(np.arange(24),var_hourly,label=var,color=c)
            ncols = 2 if len(vararray) > 4 else 1
            axis.legend(ncols=ncols)
            axis.set_xlim(0,23)
            if 'SWnet' in vararray or 'SWin' in vararray:
                axis.set_ylabel('Flux (W m$^{-2}$)')
            elif 'melt' in vararray or 'MB' in vararray:
                axis.set_ylabel('Mass balance (mm w.e.)')
            elif 'surftemp' in vararray or 'airtemp' in vararray:
                axis.set_ylabel('Temperature ($^{\circ}$C)')
    axis.set_xlabel('Hour of Day')

def dh_vs_stake(stake_df,ds_list,time,labels=['Model'],t='Surface Height Change Comparison'):
    """
    Returns a comparison of snow depth from the output datasets to stake data

    Parameters
    ----------
    stake_df : pd.DataFrame
        DataFrame object containing stake MB data
    ds_list : list of xr.Datasets
        List of model output datasets to plot melt
    time : list-like   
        Either len-2 list of start date, end date, or a list of datetimes
    labels : list of str
        List of same length as ds_list containing labels to plot
    """
    # plt.style.use('bmh')
    fig,ax = plt.subplots(figsize=(4,5.5),sharex=True,layout='constrained')
    stake_df = stake_df.set_index(pd.to_datetime(stake_df['Date']))

    if len(time) == 2:
        start = pd.to_datetime(time[0])
        end = pd.to_datetime(time[1])
        time = pd.date_range(start,end,freq='h')
        days = pd.date_range(start,end,freq='d')

    stake_df = stake_df.loc[days-pd.Timedelta(minutes=30)]
    stake_df['CMB'] -= stake_df['CMB'].iloc[0]
    for i,ds in enumerate(ds_list):
        c = plt.cm.Dark2(i)
        ds = ds.sel(time=time).resample(time='d').sum()
        dh = ds.dh.cumsum().to_numpy() - ds.dh.to_numpy()[0]
        ax.plot(ds.coords['time'],dh,label=labels[i],color=c)
    ax.plot(stake_df.index,stake_df['CMB'].to_numpy(),label='Stake',linestyle='--',color='black')
    date_form = mpl.dates.DateFormatter('%b %d')
    ax.xaxis.set_major_formatter(date_form)
    ax.set_xticks(pd.date_range(start,end,periods=5))
    ax.set_xlim(start,end)
    ax.tick_params(labelsize=12,length=5,width=1)
    ax.legend(fontsize=12,loc='upper right')
    ax.grid(False)
    ax.set_ylabel('Surface Height Change (m)',fontsize=14)
    fig.suptitle(t)

def snowdepth_vs_stake(stake_df,ds_list,time,labels,t='Snow Depth Comparison'):
    """
    Returns a comparison of snow depth from the output datasets to stake data

    Parameters
    ----------
    stake_df : pd.DataFrame
        DataFrame object containing stake MB data
    ds_list : list of xr.Datasets
        List of model output datasets to plot melt
    time : list-like   
        Either len-2 list of start date, end date, or a list of datetimes
    labels : list of str
        List of same length as ds_list containing labels to plot
    """
    fig,ax = plt.subplots(figsize=(4,6),sharex=True,layout='constrained')
    stake_df = stake_df.set_index(pd.to_datetime(stake_df['Date']))

    if len(time) == 2:
        start = pd.to_datetime(time[0])
        end = pd.to_datetime(time[1])
        time = pd.date_range(start,end,freq='h')
        days = pd.date_range(start,end,freq='d')
        if days[0].minute == 30:
            days = days - pd.Timedelta(minutes=30)
    stake_df = stake_df.loc[days]
    for i,ds in enumerate(ds_list):
        c = plt.cm.Dark2(i)
        ds = ds.sel(time=time)
        ax.plot(ds.coords['time'],ds.snowdepth.to_numpy()*100,label=labels[i],color=c)
    ax.plot(stake_df.index,stake_df['snow_depth'].to_numpy(),label='Stake',linestyle='--',color='black')
    date_form = mpl.dates.DateFormatter('%d %b')
    ax.xaxis.set_major_formatter(date_form)
    ax.legend()
    ax.set_ylabel('Snow Depth (cm)')
    fig.suptitle(t)

def albedo_vs_CNR4(cnr4_df,ds_list,time,labels=['Model'],t='Albedo Comparison'):
    
    fig,ax = plt.subplots(figsize=(4,6),sharex=True,layout='constrained')
    cnr4_df = cnr4_df.set_index(pd.to_datetime(cnr4_df['Datetime']))

    if len(time) == 2:
        startdate = pd.to_datetime(time[0])
        enddate = pd.to_datetime(time[1]) - pd.Timedelta(days=1)
        start = str(startdate.date()) + ' 12:30:00'
        end = str(enddate.date()) + ' 12:30:00'
        time = pd.date_range(start,end,freq='d')
        start = str(startdate.date()) + ' 12:00:00'
        end = str(enddate.date()) + ' 12:00:00'
        days = pd.date_range(start,end,freq='d')
    cnr4_df = cnr4_df.loc[days]
    cnr4_df['albedo'] = cnr4_df['sw_up_Avg'] / cnr4_df['sw_down_Avg']
    cnr4_df['albedo'] = cnr4_df['albedo'].mask(cnr4_df['albedo']>1,1)
    for i,ds in enumerate(ds_list):
        c = plt.cm.Dark2(i)
        ds = ds.sel(time=time)
        ax.plot(ds.coords['time'],ds['albedo'],label=labels[i],color=c)
    ax.plot(cnr4_df.index,cnr4_df['albedo'].to_numpy(),label='CNR4',linestyle='--',color='black')
    date_form = mpl.dates.DateFormatter('%d %b')
    ax.xaxis.set_major_formatter(date_form)
    ax.legend()
    ax.set_ylabel('Snow Depth (cm)')
    fig.suptitle(t)
    

def plot_stake_ablation(stake_df,ds_list,time,labels,t='Stake Comparison'):
    """
    Returns a comparison of melt from the output datasets to stake data

    Parameters
    ----------
    stake_df : pd.DataFrame
        DataFrame object containing stake MB data
    ds_list : list of xr.Datasets
        List of model output datasets to plot melt
    time : list-like   
        Either len-2 list of start date, end date, or a list of datetimes
    labels : list of str
        List of same length as ds_list containing labels to plot
    """
    fig,ax = plt.subplots(figsize=(4,6),sharex=True,layout='constrained')
    stake_df = stake_df.set_index(pd.to_datetime(stake_df['Date']))

    if len(time) == 2:
        start = pd.to_datetime(time[0])
        end = pd.to_datetime(time[1])
        time = pd.date_range(start,end,freq='h')
        days = pd.date_range(start,end,freq='d')
    stake_df = stake_df.loc[days - pd.Timedelta(minutes=30)]
    for i,ds in enumerate(ds_list):
        c = plt.cm.Dark2(i)
        ds = ds.sel(time=time)
        melt = ds.melt.cumsum().to_numpy()
        rfz = ds.refreeze.cumsum().to_numpy()
        ax.plot(ds.coords['time'],melt-rfz,label=labels[i],color=c)

    ax.plot(stake_df.index,np.cumsum(stake_df['melt'].to_numpy()),label='Stake',linestyle='--',c='black')
    ax.xaxis.set_major_locator(mpl.dates.MonthLocator())

    date_form = mpl.dates.DateFormatter('%d %b')
    ax.xaxis.set_major_formatter(date_form)
    
    ax.legend()
    ax.set_ylabel('Cumulative Melt (m w.e.)')
    fig.suptitle(t)

def plot_stake_accumulation(stake_df,ds_list,time,labels,t=''):
    """
    Returns a comparison of accumulation from the output datasets to stake data

    Parameters
    ----------
    stake_df : pd.DataFrame
        DataFrame object containing stake MB data
    ds_list : list of xr.Datasets
        List of model output datasets to plot melt
    time : list-like   
        Either len-2 list of start date, end date, or a list of datetimes
    labels : list of str
        List of same length as ds_list containing labels to plot
    """

    fig,ax = plt.subplots(figsize=(4,6),sharex=True,layout='constrained')
    stake_df = stake_df.set_index(pd.to_datetime(stake_df['Date']))

    if len(time) == 2:
        start = pd.to_datetime(time[0])
        end = pd.to_datetime(time[1])
        time = pd.date_range(start,end,freq='h')
        days = pd.date_range(start,end,freq='d')
    stake_df = stake_df.loc[days]
    for i,ds in enumerate(ds_list):
        c = plt.cm.Dark2(i)
        ds = ds.sel(time=time)
        ax.plot(ds.coords['time'],ds.accum,color=c,label=labels[i])
    snow_depth = stake_df['snow_depth'].to_numpy() / 100
    previous_depth = snow_depth[0]
    accum = []
    for depth in snow_depth:
        if depth > previous_depth:
            accum.append((depth - previous_depth)/100)
        else:
            accum.append(0)
    ax.plot(stake_df.index,accum,label='Stake',linestyle='--')

    date_form = mpl.dates.DateFormatter('%d %b')
    ax.xaxis.set_major_formatter(date_form)
    ax.legend()
    ax.set_ylabel('Accumulation (m w.e.)')
    fig.suptitle(t)
        
def compare_runs(ds_list,time,labels,var,res='d',t=''):
    """
    Returns a comparison of different model runs

    Parameters
    ----------
    ds_list : list of xr.Datasets
        List of model output datasets to plot melt
    labels : list of str
        List of same length as ds_list containing labels to plot
    time : list-like   
        Either len-2 list of start date, end date, or a list of datetimes
    var : str
        Variable to plot as named in ds
    res : str
        Abbreviated time resolution to plot (e.g. '12h' or 'd')
    t : str
        Title of plot
    """
    fig,ax = plt.subplots(figsize=(6,3))
    start = pd.to_datetime(time[0])
    end = pd.to_datetime(time[1])
    time_sel = pd.date_range(start,end,freq='h')
    if len(time) == 2 and res != 'd':
        # start += pd.Timedelta(minutes=30)
        # end -= pd.Timedelta(minutes=30)
        time = pd.date_range(start,end,freq=res)
        # time = pd.to_datetime(time,format='%Y-%m-%d')
        # res = time[1] - time[0]
        # res = str(int(res.total_seconds() / 3600))+'h'
    else:
        time = pd.date_range(start,end,normalize=True)
    for i,ds in enumerate(ds_list):
        ds = ds.sel(time=time_sel)
        c = plt.cm.Dark2(i)
        if res != 'h':
            if var in ['melt','runoff','refreeze','accum','MB','dh']:
                ds_resampled = ds[var].resample(time=res).sum()
                to_plot = ds_resampled.sel(time=time).cumsum()
            elif 'layer' in var:
                ds_resampled = ds.resample(time=res).mean()
                to_plot = ds_resampled[var].sel(time=time,layer=0)
            else:
                ds_resampled = ds.resample(time='d').mean()
                to_plot = ds_resampled[var].sel(time=time)
        else:
            if var in ['melt','runoff','refreeze','accum','MB']:
                to_plot = ds[var].sel(time=time).cumsum()
            elif 'layer' in var:
                to_plot = ds[var].sel(time=time,layer=0)
            else:
                to_plot = ds[var].sel(time=time)     
        ax.plot(to_plot.time,to_plot,label=labels[i],color=c)
        ax.set_ylim(np.min(to_plot),np.max(to_plot)+0.01*np.max(to_plot))
    date_form = mpl.dates.DateFormatter('%d %b')
    ax.xaxis.set_major_formatter(date_form)
    ax.set_ylabel(var)
    ax.legend()
    ax.tick_params(length=5)
    ax.set_xlim(time[0],time[-1])
    fig.suptitle(t)
    return fig,ax

def plot_by(ds,time,vars,t='Monthly EB Outputs',by='doy'):
    h = 1.5
    fig,axes = plt.subplots(len(vars),1,figsize=(7,h*len(vars)),sharex=True,layout='constrained')
    if len(vars) == 1:
        axes = [axes]
    
    if by == 'month':
        ds[by] = (['time'],pd.to_datetime(ds['time'].values).month)
        time_list = np.arange(1,13)
    elif by == 'hour':
        ds[by] = (['time'],pd.to_datetime(ds['time'].values).hour)
        time_list = np.arange(0,24)
    elif by == 'doy':
        ds[by] = (['time'],pd.to_datetime(ds['time'].values).day_of_year)
        time_list = np.arange(1,366)

    if len(time) == 2:
        start = pd.to_datetime(time[0])
        end = pd.to_datetime(time[1])
        time = pd.date_range(start,end,freq='h')
    ds = ds.sel(time=time)
    c_iter = iter([plt.cm.Dark2(i) for i in range(8)])
    for i,v in enumerate(vars):
        axis = axes[i]
        vararray = np.array(v)
        for var in vararray:
            try:
                c = next(c_iter)
            except:
                c_iter = iter([plt.cm.Dark2(i) for i in range(8)])
                c = next(c_iter)
        
            var_out = []
            for time in time_list:
                ds_sel = ds.where(ds[by] == time,drop=True)
                if 'layer' in var:
                    vardata = ds_sel.isel(layer=0)[var].to_numpy()
                else:
                    vardata = ds_sel[var].to_numpy()
                if by == 'doy':
                    nyrs = int(vardata.shape[0] / 24)
                    try:
                        vardata = np.sum(vardata.reshape(24,nyrs),axis=0)
                    except:
                        if time != 366:
                            print('Must index dates exactly 1 year - 1 day apart')
                        vardata = np.array([0])
                    out = np.mean(vardata)
                else:
                    out = np.mean(vardata)
                var_out.append(out)
            axis.plot(time_list,np.cumsum(var_out),label=var,color=c)
            axis.legend()
            if by == 'doy':
                axis.axvline(111,color='green')
                axis.axvline(232,color='red')
    if by == 'month':
        months = pd.date_range('2024-01-01','2024-12-31',freq='MS')
        month_names = [date.month_name()[:3] for date in months]
        axis.set_xticks(np.arange(1,13),month_names)
    if by == 'doy':
        axis.set_xlabel('Day of year')
    axis.set_ylabel('Cumulative accum.')
    axis.tick_params(length=5)
    fig.suptitle(t)

def panel_dh_compare(ds_list,time,labels,units,stake_df,rows=2,t=''):
    """
    Returns a comparison of different model runs

    Parameters
    ----------
    ds_list : list of xr.Datasets
        List of model output datasets to plot melt
    labels : list of str
        List of same length as ds_list containing labels to plot
    time : list-like   
        Either len-2 list of start date, end date, or a list of datetimes
    var : str
        List of vars to plot as named in ds
    t : str
        Title of plot
    """
    w = 2 # width of each plot
    n = int(np.ceil(len(ds_list)/2))
    n = 2 if n == 1 else n

    # Initialize plots
    fig,ax = plt.subplots(rows,int(n/rows),sharex=True,sharey=True,
                              figsize=(w*n/rows,6),layout='constrained')
    for j in range(rows):
        ax[j,0].set_ylabel('Surface Height Change (m)')
    ax = ax.flatten()
    
    # Initialize time and comparison dataset
    if len(time) == 2:
        if pd.to_datetime(time[0]).minute == 30:
            start = pd.to_datetime(time[0] - pd.Timedelta(minutes=30))
            end = pd.to_datetime(time[1] - pd.Timedelta(minutes=30))
        else:
            start = pd.to_datetime(time[0])
            end = pd.to_datetime(time[1])
        time = pd.date_range(start,end,freq='d')
    stake_df = stake_df.loc[time]
    daily_cum_dh_DATA = stake_df['CMB'].to_numpy()

    c_iter = iter(plt.cm.Dark2(np.linspace(0,1,8)))
    date_form = mpl.dates.DateFormatter('%d %b')
    plot_idx = 0
    for i,ds in enumerate(ds_list):
        # get variable and value for labeling
        var,val = labels[i].split('=')

        # get RMSE
        daily_dh_MODEL = ds.resample(time='d').sum().sel(time=time)
        daily_cum_dh_MODEL = daily_dh_MODEL['dh'].cumsum().to_numpy()
        # melt_mse = mean_squared_error(daily_cum_melt_DATA,daily_cum_melt_MODEL)
        # melt_rmse = np.mean(melt_mse)
        diff = daily_cum_dh_MODEL[-1] - daily_cum_dh_DATA[-1]
        label = f'{val}{units[i]}: {diff:.2f} m'

        # get color (loops itself)
        try:
            c = next(c_iter)
        except:
            c_iter = iter([plt.cm.Dark2(i) for i in range(8)])
            c = next(c_iter)

        # plot stake_df once per plot
        if i % 2 == 0:
            ax[plot_idx].plot(stake_df.index,daily_cum_dh_DATA,label='Stake',linestyle='--')

        # plot daily melt
        ax[plot_idx].plot(time,daily_cum_dh_MODEL,label=label,color=c,linewidth=0.8)
        ax[plot_idx].set_title(var)
        ax[plot_idx].xaxis.set_major_locator(mpl.dates.MonthLocator())
        ax[plot_idx].xaxis.set_major_formatter(date_form)
        ax[plot_idx].legend(fontsize=8)

        if i % 2 != 0:
            plot_idx += 1
    fig.autofmt_xdate()
    fig.suptitle(t)
    plt.show()
    return

def temp_vs_iButton(dslist,temp_df,time,plot_heights=[0.05,0.5],
                    t='Modeled and measured snow temperatures',labels=['Model'],
                    ax_titles=False):
    if not ax_titles:
        ax_titles = np.array(plot_heights).astype(str)
    # define height above ice of iButtons initially
    h0 = 3.5 - np.array([.1,.4,.8,1.2,1.6,2,2.4,2.8,3.2,3.49])
    melt_out = pd.to_datetime(['2023-05-20 05:30:00','2023-05-23 14:30:00',
                            '2023-06-11 10:30:00','2023-06-28 15:30:00',
                            '2023-07-03 09:30:00','2023-07-11 17:30:00',
                            '2023-07-18 12:30:00','2023-07-20 10:30:00',
                            '2023-07-25 18:30:00','2023-07-28 15:30:00'])
    if len(time) == 2:
        start = pd.to_datetime(time[0])
        end = pd.to_datetime(time[1])
        end = min(pd.to_datetime('2023-05-23 00:00'),end)
        time = pd.date_range(start,end,freq='h')
    df = temp_df.resample('30min').interpolate().loc[time]

    fig,axes = plt.subplots(len(plot_heights),figsize=(6,1.5*len(plot_heights)),
                            layout='constrained',sharex=True,sharey=True,dpi=200)

    for j,ds in enumerate(dslist):
        store = {'measured':[],'modeled':[],'measure_plot':[],'model_plot':[]}
        c = plt.cm.Dark2(j)
        ds = ds.sel(time=time)
        for hour in time:
            # get temperatures of buried iButtons
            buried = np.where(melt_out > hour)[0]
            temp_measure = np.flip(df.loc[hour].to_numpy()[buried])
            height_measure = np.flip(h0[buried])
            
            # get modeled temperatures
            # index snow layers
            dens_model = ds.sel(time=hour)['layerdensity'].to_numpy()
            dens_model[np.where(np.isnan(dens_model))[0]] = 1e5
            snow = np.where(dens_model < 700)[0]
            # include one extra layer for interpolating (will index out when stored)
            snow = np.append(snow,snow[-1]+1).ravel()
            # get height above ice
            lheight = ds.sel(time=hour)['layerheight'].to_numpy()[snow]
            icedepth = np.sum(lheight[:-1]) + lheight[-2] / 2
            # get property and absolute depth
            ldepth = np.array([np.sum(lheight[:i+1])-(lheight[i]/2) for i in range(len(lheight))])
            temp_model = np.flip(ds.sel(time=hour)['layertemp'].to_numpy()[snow])
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
        # get RMSE
        def flatten(xss):
            return np.array([x for xs in xss for x in xs])
        measured = flatten(store['measured'])
        modeled = flatten(store['modeled'])
        mse = mean_squared_error(measured,modeled)
        rmse = np.sqrt(mse)

        # get plotting data
        measure_plot = np.array(store['measure_plot'])
        model_plot = np.array(store['model_plot'])

        # plot
        date_form = mpl.dates.DateFormatter('%b %d')
        for i,ax in enumerate(axes):
            ax.set_title(ax_titles[i],loc='right')
            if j == 0:
                ax.plot(time,measure_plot[:,i],color='black',linestyle='--',label='iButtons')
            ax.plot(time,model_plot[:,i],label=labels[j],color=c)
            ax.xaxis.set_major_formatter(date_form)
            ax.grid(False)
            ax.tick_params(length=5,labelsize=11)
    fig.supylabel('Snow Temperature (C)',fontsize=14)
    ax.set_xticks(pd.date_range(start,end,freq='W'))
    ax.set_xlim(start,end)
    axes[0].legend(fontsize=12,loc='lower right')
    fig.suptitle(t) #+f'\n RMSE = {rmse:.3f}')

def panel_temp_compare(ds_list,time,labels,temp_df,rows=2,t=''):
    """
    Returns a comparison of different model runs

    Parameters
    ----------
    ds_list : list of xr.Datasets
        List of model output datasets to plot melt
    labels : list of str
        List of same length as ds_list containing labels to plot
    time : list-like   
        Either len-2 list of start date, end date, or a list of datetimes
    t : str
        Title of plot
    """
    w = 2 # width of each plot
    n = int(np.ceil(len(ds_list)/2))
    n = 2 if n == 1 else n

    # Initialize plots
    fig,ax = plt.subplots(rows,int(n/rows),sharex=True,figsize=(w*n/rows,6),layout='constrained')
    ax = ax.flatten()

    # Initialize time and comparison dataset
    if len(time) == 2:
        start = pd.to_datetime(time[0])
        end = pd.to_datetime(time[1])
        time = pd.date_range(start,end,freq='h')
    temp_df = temp_df.set_index(pd.to_datetime(temp_df['Datetime']))
    temp_df = temp_df.drop(columns='Datetime')
    height_DATA = 3.5 - np.array([.1,.4,.8,1.2,1.6,2,2.4,2.8,3.2,3.49])

    c_iter = iter(plt.cm.Dark2(np.linspace(0,1,8)))
    date_form = mpl.dates.DateFormatter('%d %b')
    plot_idx = 0
    for i,ds in enumerate(ds_list):
        # get variable and value for labeling
        var,val = labels[i].split('=')

        # Need to interpolate data for comparison to model depths -- loop through timesteps
        all_MODEL = np.array([])
        all_DATA = np.array([])
        all_TIME = np.array([])
        plot_MODEL = np.array([])
        plot_DATA = np.array([])
        for hour in time:
            # Extract layer heights
            lheight = ds.sel(time=hour)['layerheight'].to_numpy()
            # Index snow layers
            density = ds.sel(time=hour)['layerdensity'].to_numpy()
            density[np.where(np.isnan(density))[0]] = 1e5
            full_layers = np.where(density < 700)[0]
            if len(full_layers) < 1:
                break
            lheight = lheight[full_layers]
            icedepth = np.sum(lheight) + lheight[-1] / 2

            # Get property and absolute depth
            temp_MODEL = ds.sel(time=hour)['layertemp'].to_numpy()[full_layers]
            ldepth = np.array([np.sum(lheight[:i+1])-(lheight[i]/2) for i in range(len(lheight))])
            height_above_ice = icedepth - ldepth

            # Interpolate temperature data to model heights
            temp_at_iButtons = temp_df.loc[hour].to_numpy().astype(float)
            temp_DATA = np.interp(height_above_ice,height_DATA,temp_at_iButtons)
            all_MODEL = np.append(all_MODEL,temp_MODEL)
            all_DATA = np.append(all_DATA,temp_DATA)
            all_TIME = np.append(all_TIME,hour)

            # Extract mean snow column temperature to plot
            temp_no_above_0 = temp_df.mask(temp_df>=0.2,None).loc[hour].to_numpy().astype(float)
            plot_MODEL = np.append(plot_MODEL,np.average(temp_MODEL,weights=lheight))
            plot_DATA = np.append(plot_DATA,np.mean(temp_no_above_0))
        temp_mse = mean_squared_error(all_DATA,all_MODEL)
        temp_rmse = np.mean(temp_mse)
        label = f'{val}: {temp_rmse:.3f}'

        # get color (loops itself)
        try:
            c = next(c_iter)
        except:
            c_iter = iter([plt.cm.Dark2(i) for i in range(8)])
            c = next(c_iter)

        # plot temp_df once per plot
        if i % 2 == 0:
            ax[plot_idx].plot(all_TIME,plot_DATA,label='iButtons',linestyle='--')

        # plot daily melt
        time = pd.date_range(time[0],end,freq='h')
        ax[plot_idx].plot(all_TIME,plot_MODEL,label=label,color=c,linewidth=0.8)
        ax[plot_idx].set_title(var)
        ax[plot_idx].xaxis.set_major_formatter(date_form)
        ax[plot_idx].set_ylabel('Average Snow Temperature (C)')
        ax[plot_idx].legend()

        if i % 2 != 0:
            plot_idx += 1
    fig.suptitle(t)
    plt.show()
    return

def plot_multiyear_mb(ds_list,mb_df,years,site):
    fig,ax = plt.subplots(figsize=(9,3),sharex=True,layout='constrained')
    mb_df = mb_df.loc[mb_df['site_name'] == site]
    mb_df.index = mb_df['Year']

    # plot mass balance data
    winter_mb_data = mb_df['bw'].loc[mb_df['Year'].isin(years)]
    summer_mb_data = mb_df['ba'].loc[mb_df['Year'].isin(years)] - mb_df['bw'].loc[mb_df['Year'].isin(years)]
    ax.plot(years,winter_mb_data,label='MB Data',color='black',linestyle='--')
    ax.plot(years,summer_mb_data,color='black',linestyle='--')

    for k,ds in enumerate(ds_list):
        mb_mod = {'bw':[],'bs':[]}
        for year in years:
            # spring_date = mb_df['spring_date'].loc[year]
            # fall_date = mb_df['fall_date'].loc[year]
            # next_spring_date = mb_df
            spring_date = str(year)+'-04-20 00:00'
            fall_date = str(year)+'-08-20 00:00'
            next_spring_date = str(year)+'-04-20 00:00'
            last_fall_date = str(year-1)+'-08-20 00:00'
            melt_dates = pd.date_range(spring_date,fall_date,freq='h')
            acc_dates = pd.date_range(last_fall_date,spring_date,freq='h')
            if pd.to_datetime(ds.time.values[0]).minute == 30:
                melt_dates = melt_dates + pd.Timedelta(minutes=30)
                acc_dates = acc_dates + pd.Timedelta(minutes=30)
            # sum mass balance
            wds = ds.sel(time=acc_dates).sum()
            sds = ds.sel(time=melt_dates).sum()
            winter_mb = wds.accum + wds.refreeze # - wds.melt
            summer_mb = sds.accum + sds.refreeze - sds.melt - wds.melt
            mb_mod['bw'].append(winter_mb.to_numpy())
            mb_mod['bs'].append(summer_mb.to_numpy())
        ax.plot(years,mb_mod['bw'],label='Winter MB Modeled'+str(k))
        ax.plot(years,mb_mod['bs'],label='Summer MB Modeled'+str(k))
    ax.legend()
    ax.axhline(0,color='white')
    ax.set_ylabel('Seasonal Mass Balance (m w.e.)')
    ax.set_xlim(years[0],years[-1])
    ax.set_xticks(np.arange(years[0],years[-1],2))
    fig.suptitle('Gulkana Mass Balance Comparison')
    # plt.show()
    # plt.savefig('20yrfig.png',dpi=200)
    return fig, ax

def build_RMSEs(ds_list,stake_df,time,labels,save='sensitivity.npy'):
    """
    save : str or False
        Filepath to save .npy file
    """
    # get stake data into right format
    stake_df = stake_df.set_index(pd.to_datetime(stake_df['Date']))
    stake_df = stake_df.loc[time[0]:time[1]]
    daily_cum_melt_DATA = np.cumsum(stake_df['melt'].to_numpy())
    sens_out = {}
    for i,ds in enumerate(ds_list):
        daily_melt_MODEL = ds.resample(time='d').sum()
        daily_cum_melt_MODEL = daily_melt_MODEL['melt'].cumsum().to_numpy()
        melt_mse = mean_squared_error(daily_cum_melt_DATA,daily_cum_melt_MODEL)
        melt_rmse = np.mean(melt_mse)
        sens_out[labels[i]] = melt_rmse
    if save:
        np.save(save,sens_out)
    return sens_out

def plot_iButtons(ds,dates,path=None):
    if not path:
        path = '/home/claire/research/MB_data/Gulkana/field_data/iButton_2023_all.csv'
    df = pd.read_csv(path,index_col=0)
    df = df.set_index(pd.to_datetime(df.index)- pd.Timedelta(hours=8))
    df = df[pd.to_datetime('04-18-2023 00:00'):]
    depth_0 = 3.5 - np.array([.1,.4,.8,1.2,1.6,2,2.4,2.8,3.2,3.5])

    fig,axes = plt.subplots(1,len(dates),sharey=True,sharex=True,figsize=(8,4)) #,sharex=True,sharey='row'
    for i,date in enumerate(dates):
        # Extract layer heights
        lheight = ds.sel(time=date)['layerheight'].to_numpy()
        # Index snow layers
        density = ds.sel(time=date)['layerdensity'].to_numpy()
        density[np.where(np.isnan(density))[0]] = 1e5
        full_layers = np.where(density < 700)
        
        # full_layers = np.array([not y for y in np.isnan(lheight)])
        lheight = lheight[full_layers]
        icedepth = np.sum(lheight) + lheight[-1] / 2
        # Get property and absolute depth
        lprop = ds.sel(time=date)['layertemp'].to_numpy()[full_layers]
        ldepth = np.array([np.sum(lheight[:i+1])-(lheight[i]/2) for i in range(len(lheight))])
        height_above_ice = icedepth - ldepth
        # Plot output data
        axes[i].plot(lprop,height_above_ice,label='Model')

        # Plot iButton data
        snowdepth = ds.sel(time=date)['snowdepth'].to_numpy()
        tempdata = df.loc[date].to_numpy()
        idx = np.where(depth_0 < snowdepth)
        axes[i].plot(tempdata[idx],depth_0[idx],label='iButton')

        axes[i].set_title(str(date)[:10])
    axes[0].legend()
    fig.supxlabel('Temperature (C)')
    axes[0].set_ylabel('Depth (m)')
    return

def stacked_eb_barplot(ds,time,res='d',t='',savefig=False):
    """
    Returns a barplot where energy fluxes are stacked

    Parameters
    ----------
    ds : xr.Dataset
        Dataset object containing the model output
    time : list-like   
        Either len-2 list of [start date, end date], or a list of datetimes
    res : str
        Abbreviated time resolution to plot (e.g. '12h' or 'd')
    t : str
        Title of plot
    """
    fig,ax = plt.subplots(figsize=(10,5))
    vars = ['latent','NetRad','sensible','rain']
    ds['all_but_shortwave'] = ds['LWnet'] + ds['latent']+ ds['sensible']+ ds['ground']+ ds['rain']

    if len(time) == 2:
        start = pd.to_datetime(time[0])
        end = pd.to_datetime(time[1])
        time = pd.date_range(start,end,freq='h')
    ds = ds.sel(time=time)
    ds = ds.resample(time=res).mean(dim='time')
    bottom=0
    for i,var in enumerate(np.array(vars)):
        vardata = ds[var].to_numpy().T[0]
        if i==0:
            ax.bar(ds.coords['time'],vardata,label='Net Shortwave')
        else:
            bottom = ds[vars[i-1]].to_numpy().T[0]+bottom
            bottom[np.where(bottom<0)] = 0
            ax.bar(ds.coords['time'],vardata,bottom=bottom,label='All Other Fluxes')
    # ax.plot(ds.coords['time'],ds['meltenergy'],label='melt energy',color='black',linewidth=.6,alpha=0.7)
    date_form = mpl.dates.DateFormatter('%d %b %Y')
    ax.xaxis.set_major_formatter(date_form)
    ax.set_ylabel('Fluxes (W/m2)')
    ax.legend(loc='upper left')
    fig.suptitle(t)
    if savefig:
        plt.savefig(savefig,dpi=300)
    plt.show()

def plot_yrs(file,nyr):
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    fig,axes = plt.subplots(3,nyr,sharey='row',sharex='col',figsize=(14,8))

    ds = xr.open_dataset(file)
    varnames_idx = ['SWin','SWout','LWin','LWout','sensible','latent','rain','meltenergy','surftemp','melt','runoff','refreeze','accum','snowdepth']
    varnames = ['SWnet','LWnet','sensible','latent','rain','meltenergy','melt','runoff','refreeze','accum','snowdepth']
    heat = ['SWnet','LWnet','sensible','latent','rain','meltenergy']
    temp = ['snowdepth']
    mb = ['melt','runoff','refreeze','accum','MB']

    df = ds[varnames_idx].to_pandas()
    df['SWnet'] = df['SWin'] + df['SWout']
    df['LWnet'] = df['LWin'] + df['LWout']
    df['MB'] = df['accum']+df['refreeze']-df['melt']

    # Loop through variables to get monthly averages and plot them
    for var in varnames:
        if var in ['melt','runoff','refreeze','accum','MB']:
            monthly = df[var].resample('M').sum()
        else:
            monthly = df[var].resample('M').mean()
        monthly_avg = monthly[:(nyr*12)].values.reshape((nyr,12))
        
        axis = np.piecewise(var,[var in heat, var in temp, var in mb],[0,1,2])
        for yr in range(nyr):
            axes[int(axis),yr].plot(months,monthly_avg[yr,:],label=var)
            axes[int(axis),yr].set_xlabel(str(pd.to_datetime(ds.coords['time'].values[0]).year+yr))
            axes[int(axis),yr].set_ylabel(var)
            axes[int(axis),0].legend()
            axes[1,0].set_xlim(0,6)
    return

def plot_AWS(df,vars,time,t=''):
    """
    Plots heatmap of AWS data in the specified time period

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing AWS data as input to the model
    vars : list
        List of variable names to plot
    time : list-like   
        Either len-2 list of [start date, end date], or a list of datetimes
    t : str
        Title of plot
    """
    df = df.set_index(pd.to_datetime(df.index))
    start = pd.to_datetime(time[0])
    end = pd.to_datetime(time[-1])
    df = df.loc[start:end+pd.Timedelta(hours=23)]
    days = df.resample('d').mean().index
    hours = np.arange(0,24)

    fig,axs = plt.subplots(len(vars),sharex=True,layout='constrained')
    for i,var in enumerate(vars):
        vardata = df[var].to_numpy().reshape((len(days),24))
        if var in ['SWin','LWin']:
            vardata = vardata * 3600
        pc = axs[i].pcolormesh(days,hours,vardata.T, cmap='RdBu_r')
        ticks = np.linspace(np.ceil(np.min(vardata)),np.floor(np.max(vardata)),3)
        if ticks[1]%1 > 0:
            ticks =  np.linspace(np.ceil(np.min(vardata)),np.floor(np.max(vardata))+1,3)
        clb = fig.colorbar(pc,ax=axs[i],ticks=ticks.astype(int),aspect=10,pad=0.02)
        clb.ax.set_title(AWS_vars[var]['units'])
        axs[i].set_title(AWS_vars[var]['label'])
        axs[i].set_ylabel('Hour')
        yticks = mpl.ticker.MultipleLocator(6)
        axs[i].yaxis.set_major_locator(yticks)
    date_form = mpl.dates.DateFormatter('%d %b')
    axs[i].xaxis.set_major_formatter(date_form)
    fig.suptitle(t)
    plt.show()

def compare_AWS(df_list,vars,time,labels=None,t='',res='d',y=''):
    fig,axs = plt.subplots(len(vars),sharex=True,layout='constrained')
    linestyles = ['-','--','-.',':']
    for i,df in enumerate(df_list):
        df = df.set_index(pd.to_datetime(df.index))
        df[['SWout','LWout']] = df[['SWout','LWout']] * -1
        start = pd.to_datetime(time[0])
        end = pd.to_datetime(time[-1])
        df = df.loc[time]
        # df = df.loc[start:end+pd.Timedelta(hours=23)]
        # df = df.resample(res).mean()

        for j,var in enumerate(vars):
            vardata = df[var].to_numpy()
            if var in ['tp']:
                vardata = np.cumsum(vardata)
            if not np.all(np.isnan(vardata)):
                axs[j].plot(df.index,vardata,label=var+': '+labels[i],linestyle=linestyles[i])
                axs[j].legend()
                axs[j].set_ylabel(var)
    date_form = mpl.dates.DateFormatter('%d %b')
    axs[j].xaxis.set_major_formatter(date_form)
    fig.suptitle(t)
    plt.show()

def plot_avg_layers(ds,nyr=False,res='365d'):
    """
    Plots layer temperature, density, water content and layer height averaged first across all layers,
    then averaged between years. 
    """
    # ds = xr.open_dataset(file)
    fig,axes = plt.subplots(2,3,sharex=True,figsize=(8,8)) #,sharex=True,sharey='row',figsize=(12,5)
    axes = axes.flatten()
    days =  np.arange(365)
    for j,var in enumerate(['layertemp','layerdensity','layerwater','snowdepth','layergrainsize','layerBC']):
        ax = axes[j]
        ax.set_title(var+'   '+ds[var].attrs['units'])
        snow = ds[var].to_pandas()
        loop = True
        i=29
        while loop:
            if np.isnan(snow.iloc[i]).all and var not in ['snowdepth']:
                snow.drop(i,axis=1)
            i -=1
            if i == 0:
                break

        if var in ['layertemp','layerdensity','layergrainsize']:
            snow = snow.mean(axis=1)
        elif var in ['snowdepth']:
            pass
        else:
            snow = snow.sum(axis=1)

        if var in ['layerwater']:
            snow = snow/1000 # to m w.e.
        
        if nyr:
            snowdaily = snow.resample('d').mean()
            snowdaily = np.mean(snow[:nyr*365].values.reshape((nyr,365)),axis=0)

            ax.plot(days,snowdaily,label=var)
        else:
            time = pd.date_range('2000-07-01 12:30','2020-07-01 12:30',freq=res)
            snow = snow.loc[time]
            ax.plot(time,snow)
            

    # axes[1,0].set_title('water content    m w .e')
    plt.gcf().autofmt_xdate()
    # plt.savefig('/home/claire/research/Output/EB/subsurfplot.png')
    plt.show()
    return

def plot_layers(ds,vars,dates):
    fig,axes = plt.subplots(len(vars),len(dates),sharey=True,sharex=True,figsize=(8,4)) #,sharex=True,sharey='row'
    for i,var in enumerate(vars):
        for j,date in enumerate(dates):
            lheight = ds.sel(time=date)['layerheight'].to_numpy()
            full_layers = np.array([not y for y in np.isnan(lheight)])
            full_layers = np.where(ds.sel(time=date)['layerdensity']<600)[0]
            lheight = lheight[full_layers]
            lprop = ds.sel(time=date)[var].to_numpy()[full_layers]
            ldepth = -1*np.array([np.sum(lheight[:i+1])-(lheight[i]/2) for i in range(len(lheight))])
            if len(vars) > 1:
                axes[i,j].plot(lprop,ldepth)
                axes[i,j].set_xlabel(var)
                axes[0,j].set_title(str(date)[:10])
                axes[i,0].legend()
                axes[i,0].set_ylabel('Depth (m)')
            else:
                axes[j].plot(lprop,ldepth)
                axes[j].set_xlabel(var)
                axes[j].set_title(str(date)[:10])
                axes[0].legend()
                axes[0].set_ylabel('Depth (m)')
        
    # fig.supxlabel(varprops[var]['label'])
    return

def visualize_layers(ds,dates,vars,force_layers=False,
                     t='',plot_ax=False,
                     plot_firn=True,plot_ice=False,ylim=False,
                     colorbar=True):
    """
    force_layers:
        Three options:
        - False, takes all snow layers
        - List of integers to select those layer indices
        - Depth in m 
    """
    # plt.style.use('bmh')
    # mpl.style.use('seaborn-v0_8-whitegrid')
    diff = dates[1] - dates[0]

    # Custom color function based on concentrations
    def get_color(value,bounds,color):
        min_val = bounds[0]
        max_val = bounds[1]
        # value = max(min_val, min(max_val, value))
        norm = mpl.colors.Normalize(vmin=min_val,vmax=max_val)
        cmap = mpl.cm.get_cmap(color)
        c = cmap(norm(value))
        return c

    fig,axes = plt.subplots(len(vars),figsize=(5,1.7*len(vars)),sharex=True,layout='constrained')
    if plot_ax:
        assert len(plot_ax) == len(vars)
        axes = plot_ax
    if len(vars) == 1 and '__iter__' not in dir(axes):
        axes = [axes]
    for i,var in enumerate(vars):
        if var in ['layerBC']:
            bounds = [-5,30]
        elif var in ['layerOC']:
            bounds = [-5,100]
        elif var in ['layerdust']:
            bounds = [0,2]
        elif var in ['layerdensity']:
            bounds = [50,800] if plot_firn else [0,500]
        elif var in ['layerwater']:
            bounds = [-1,6]
        elif var in ['layertemp']:
            bounds = [-10,0]
        elif var in ['layergrainsize']:
            bounds = [50,1500]
        elif var in ['layerrefreeze']:
            bounds = [0,0.4]
        dens_lim = 890 if plot_firn else 600
        dens_lim = 1000 if plot_ice else dens_lim
        assert 'layer' in var, 'choose layer variable'
        ax = axes[i]
        # if plot_ice:
        #     ax.set_yscale('log')
        first = False
        last = False
        max_snowdepth = 0
        for step in dates:
            height = ds.sel(time=step)['layerheight'].to_numpy()
            vardata = ds.sel(time=step)[var].to_numpy()
            dens = ds.sel(time=step)['layerdensity'].to_numpy()
            if type(force_layers) == bool:
                layers_to_plot = np.where(dens < dens_lim)[0]
            else:
                if '__iter__' in dir(force_layers):
                    layers_to_plot = force_layers
                else:
                    layers_to_plot = np.where(np.cumsum(height) < force_layers)[0]
            if plot_ice:
                layers_to_plot = np.arange(len(vardata))
            # flip order so they stack bottom to top
            height = np.flip(height[layers_to_plot])
            vardata = np.flip(vardata[layers_to_plot])
            if var in ['layerwater']:
                vardata = vardata / height / 1000 * 100
            if var in ['layerrefreeze']:
                vardata = vardata / (np.flip(dens[layers_to_plot]) * height)
            # if plot_ice:
            #     height = np.log(height)

            bottom = 0
            ctypes = {'layerBC':'Greys','layerOC':'Oranges','layerdust':'Reds',
                      'layertemp':'plasma','layerdensity':'Greens','layerwater':'Blues',
                      'layergrainsize':'PuRd','layerrefreeze':'Purples'}
            ctype = ctypes[var]
            if np.sum(height) < 0.05 and first and not last and step.month<9:
                last = step
            for [dh,data] in zip(height,vardata):
                if np.isnan(dh):
                    continue
                elif not first:
                    first = step
                color = get_color(data,bounds,ctype)
                if 'density' in var and data > 800:
                    color = '0.1'
                ax.bar(step,dh, bottom=bottom, width=diff, color=color,linewidth=0.5,edgecolor='none')
                bottom += dh  # Update bottom for the next set of bars
            max_snowdepth = max(max_snowdepth,np.sum(height))
            # if np.abs(step.day_of_year-244) < 6:
            #     ax.axvline(step,lw=0.7,color='red')
        # Add colorbar
        units = {'layerBC':'ppb','layerdust':'ppm','layerOC':'ppb','layertemp':'$^{\circ}$C',
                'layerdensity':'kg m$^{-3}$','layerwater':'%','layergrainsize':'um',
                'layerrefreeze':'kg m-2'}
        if colorbar:
            sm = mpl.cm.ScalarMappable(cmap=ctype,norm=plt.Normalize(bounds[0],bounds[1]))
            leg = plt.colorbar(sm,ax=ax,aspect=7)
            leg.ax.tick_params(labelsize=9)
            if 'BC' in var:
                leg.ax.set_ylim(0, 30)
                leg.ax.set_yticks([0,15,30])
            
            # leg.set_label(units[var],loc='top',rotation=0)
            # leg.yaxis.set_label_position('right')
            # leg.yaxis.set_label_coords(1.2,0)
            label = varprops[var]['label']+' ('+units[var]+')'
            # ax.set_ylabel(label,fontsize=10)
            leg.set_label(label,rotation=270,labelpad=27,fontsize=12)
        ax.grid(axis='y')
        ax.tick_params(length=5)
        if ylim:
            ax.set_ylim(ylim)
    # Customize plot     
    ylabel = 'Height above ice (m)'
    if len(axes) > 1:
        fig.supylabel(ylabel,)
    else:
        ax.set_ylabel(ylabel,)
    fig.suptitle(t,fontsize=14)
    # melt_out = last.strftime('%b %d')
    # axes[0].set_title(f'Max snowdepth of {max_snowdepth:.2f}m melted out on {melt_out}',fontsize=10)
    ax.set_xticks(dates)
    date_form = mpl.dates.DateFormatter('%b %d')
    ax.xaxis.set_major_formatter(date_form)
    ax.set_xticks(pd.date_range(dates[0],dates[len(dates)-1],freq='2MS'))
    ax.set_xlim([dates[0],dates[len(dates)-1]])

    if dates[-1] - dates[1] < pd.Timedelta(days=5):
        date_form = mpl.dates.DateFormatter('%m/%d %H:00')
        ax.xaxis.set_major_formatter(date_form)
        ax.set_xticks(pd.date_range(dates[0],dates[len(dates)-1],5))

    # Show plot
    # plt.show()
    if not plot_ax:
        return fig,ax
    else:
        plt.close()
        return axes

def plot_single_layer(ds,layer,vars,time,cumMB=False,t='',vline=None,res='h',resample=False):
    if len(time) == 2:
        start = pd.to_datetime(time[0])
        end = pd.to_datetime(time[1])
        time = pd.date_range(start,end,freq=res)
    
    fig,axes = plt.subplots(len(vars),sharex=True,figsize=(8,1.2*len(vars)),layout='constrained')
    if len(vars) == 1:
        axes = np.array([axes])

    for i,var in enumerate(vars):
        if vline:
            axes[i].axvline(vline,c='r',linewidth=0.6)
        if 'layer' in var:
            dsvar = ds.resample(time=res).mean() if resample else ds
            lprop = dsvar.sel(time=time,layer=layer)[var].to_numpy()
        else:
            if var in ['melt','runoff','accum','refreeze'] and cumMB:
                dsvar = ds.resample(time=res).sum() if resample else ds
                lprop = dsvar.sel(time=time)[var].cumsum().to_numpy()
            else:
                dsvar = ds.resample(time=res).sum() if resample else ds
                lprop = dsvar.sel(time=time)[var].to_numpy()
        axes[i].plot(time,lprop)
        axes[i].legend()
        axes[i].set_title(varprops[var]['label'])
        if 'Cum.' in varprops[var]['label'] and not cumMB:
            axes[i].set_title(varprops[var]['label'][5:])
        axes[i].set_ylabel(varprops[var]['units'])
    if end - start > pd.Timedelta(days=365):
        date_form = mpl.dates.DateFormatter('%d %b %y')
    else:
        date_form = mpl.dates.DateFormatter('%d %b')
    axes[i].xaxis.set_major_formatter(date_form)
    fig.suptitle(t)
    plt.show()
    return