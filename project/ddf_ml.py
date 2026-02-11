import os
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import statsmodels.api as sm
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.linear_model import LinearRegression
import socket
if 'trace' in socket.gethostname():
    base_fp = '/trace/group/rounce/cvwilson/Output/ddf/'
    home_fp = '/trace/home/cvwilson/research/'
else:
    base_fp = 'C:/Users/cvw30/Research/Output/ddf/'
    home_fp = 'C:/Users/cvw30/Research/'

colors = ['#63c4c7','#fcc02e','#4D559C','#60C252','#BF1F6A',
              '#F77808','#298282','#999999','#FF89B0','#427801']

site_dict = {'kahiltna':['KPS','K17b','K14C'], #  
             'kennicott':['KCH','KCO'], # KENNICOTT
             'gulkana':['AU','B','D','B_noqm'], # GULKANA
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

all_dfs = {}
all_df = None
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
        if os.path.exists(df_fn):
            df = pd.read_csv(df_fn, index_col=0, parse_dates=True)
        else:
            if 'noqm' in site:
                ds_in = xr.open_dataset(base_fp + f'../{glacier}B_2026_02_04_base_long_0.nc')
            else:
                ds_in = xr.open_dataset(base_fp + f'{glacier}{site}_2026_{date}_base_long_0.nc') 
            
            # Clip temperature to positive values
            positive_temp = ds_in['airtemp'].clip(min=0)

            # Resample to daily
            time_res = '1d'
            daily_snow_depth = ds_in['layerheight'].where(ds_in['layertype'] < 2).sum(dim='layer').resample(time=time_res).min()
            resampled_melt = ds_in['melt'].resample({'time': time_res}).sum() * 1000
            resampled_pdds = positive_temp.resample({'time': time_res}).sum()
            resampled_acc = ds_in['accum'].resample({'time': time_res}).sum() * 1000
            resampled_rain = ds_in['rainfall'].resample({'time': time_res}).sum() * 1000

            # Days since accumulation
            last = np.maximum.accumulate(np.where(resampled_acc > 1e-3, np.arange(len(resampled_acc)), -1))
            days_since_acc = np.arange(len(resampled_acc)) - last
            days_since_acc = xr.DataArray(days_since_acc, dims=['time'], coords={'time': resampled_acc.time})

            # calculate rolling DDF
            melt_rolling = resampled_melt.rolling(time=5).sum()
            pdd_rolling = resampled_pdds.rolling(time=5).sum()
            ddf = melt_rolling / pdd_rolling

            # clip to reasonable bounds
            ddf = ddf.where(pdd_rolling > np.nanquantile(pdd_rolling.values, 0.2)) # avoids small PDDs in early summer
            ddf = ddf.where(np.isfinite(ddf))  # avoids nans and infinity
            ddf = ddf.where(resampled_melt > 1) # avoids small melt days
            ddf = ddf.where(daily_snow_depth > 1e-8) # avoids days with ice surface
            ddf = ddf.where(ddf < 200) # avoids extremely high ddfs
            
            # start to build the dataset
            ds = xr.Dataset({
            'melt':resampled_melt,
            'pdds':resampled_pdds,
            'ddf':ddf,
            'accum':resampled_acc,
            'rainfall':resampled_rain,
            'days_since_accum':days_since_acc,
            'snow_depth':daily_snow_depth
            })

            # add deposition to dataset
            ds['bc_dep'] = ds_bc.resample(time=time_res).sum() * 3600 # kg m-2
            if include_OC:
                ds['oc_dep'] = ds_oc.resample(time=time_res).sum()

            # define variables to drop and to take cumsum
            drop_vars = ['lat','lon']
            cum_vars = ['bc_dep','melt','accum','pdds']
            if include_OC:
                cum_vars += ['oc_dep']

            # add cumulative variables
            water_year = xr.where(ds['time.month'] >= 10, ds['time.year'] + 1, ds['time.year'])
            for var in cum_vars: 
                ds[f'{var}_cumsum'] = (ds[var].groupby(water_year).cumsum())
            drop_vars += cum_vars

            # add weekly rolling deposition (7 days prior to each timestep)
            ds['bc_7d_rolling'] = ds['bc_dep'].rolling(time=7).sum()
            if include_OC:
                ds['oc_7d_rolling'] = ds['oc_dep'].rolling(time=7).sum()
            ds['accum_3d_rolling'] = ds['accum'].rolling(time=3).sum()

            # create dataframe and drop variables
            df = ds.to_dataframe().drop(columns=drop_vars)

            # rename and reorder columns
            df = df.rename(columns={'daily_snow_depth':'snow_depth'})
            first = ['ddf','snow_depth','days_since_accum','accum_cumsum','accum_3d_rolling','pdds_cumsum'] # 'ddf_wBCOC', 'ddf_noBCOC', 
            df = df[first + [c for c in df.columns if c not in first]]

            # save to csv
            df.to_csv(df_fn)

        # crop to usable days
        df = df[~df['ddf'].isna()]

        # plot the heatmap
        corr = df.corr()
        plt.figure()
        sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title(f'Correlation Matrix for {glacier.capitalize()} {site}')
        plt.tight_layout()
        plt.savefig(base_fp + f'{glacier}{site}_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()

        # store
        all_dfs[glacier+site] = df
        if 'noqm' in site:
            kk = 0
        elif all_df is not None:
            all_df = pd.concat([all_df, df])
        else:
            all_df = df

        print('got dataframe for', glacier, site)

# store df with all data
all_dfs['all'] = all_df

# define variables for regression
target = 'ddf'

# loop through sites and train models 
for glaciersite in ['all']:
    df = all_dfs[glaciersite]
    vars_use = np.array(df.columns[df.columns != target])

    n_features = []
    r2s = []
    maes = []
    best_r2 = -np.inf

    while len(vars_use) > 1:
        X = df[vars_use]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=500, random_state=42)
        model.fit(X_train, y_train)

        perm = sk.inspection.permutation_importance(model, X_test, y_test, n_repeats=10)
        importances = perm.importances_mean
        indices = np.argsort(importances)[::-1]

        y_pred = model.predict(X_test)
        r2 = sk.metrics.r2_score(y_test, y_pred)
        mae = np.mean(np.abs(y_pred - y_test))

        r2s.append(r2)
        maes.append(mae)
        n_features.append(len(vars_use))

        if r2 > best_r2:
            best_r2 = r2
            best_model = model 
            best_vars = vars_use 
            best_data = {'X_train':X_train, 'y_train':y_train,
                         'X_test':X_test, 'y_test':y_test}
        
        print('built model for', len(vars_use))
        least_important = vars_use[indices[-1]]
        vars_use = vars_use[vars_use != least_important]

    # plot the best model performance 
    y_pred_test = best_model.predict(best_data['X_test'])
    y_pred_train = best_model.predict(best_data['X_train'])
    y_test = best_data['y_test']
    y_train = best_data['y_train']
    r2_test = sk.metrics.r2_score(y_test, y_pred_test)
    r2_train = sk.metrics.r2_score(y_train, y_pred_train)
    n_features = len(best_vars)
    vars_joined = ', '.join(best_vars)
    if len(best_vars) > 3:
        vars_joined = ', '.join(best_vars[:2]) + ',\n' + ', '.join(best_vars[2:])

    print(f'Best model for {glaciersite} uses {n_features}: {vars_joined}')
    
    plt.figure()
    plt.scatter(y_train, y_pred_train, alpha=0.8, color='#63c4c7', label='Train')
    plt.scatter(y_test, y_pred_test, alpha=0.8, color='#BF1F6A', label='Test')
    plt.plot([-10, 100], [-10, 100], color='k', linestyle='--', label='1:1')
    plt.xlim(-5, 75)
    plt.ylim(-5, 75)
    plt.xlabel('Modeled degree-day factor')
    plt.ylabel('Predicted degree-day factor')
    plt.text(0.98, 0.05, f'Using {n_features} features\nR$^2$ (train)$=$ {r2_train:.3f}\nR$^2$ (test)$=$ {r2_test:.3f}',
             transform=plt.gca().transAxes,ha='right',va='bottom')
    plt.title(f'Best model for {glaciersite}\n{vars_joined}')
    plt.legend()
    plt.savefig(base_fp + f'{glaciersite}_predictions.png', dpi=300)
    plt.close()

    if glaciersite in ['gulkanaB','all']:
        for compare in all_dfs:
            if compare != glaciersite:
                X = all_dfs[compare][best_vars]
                y = all_dfs[compare][target]
                y_pred = best_model.predict(X)
                r2 = sk.metrics.r2_score(y, y_pred)

                plt.figure()
                plt.scatter(y, y_pred, alpha=0.8, color='#BF1F6A', label='Data')
                plt.plot([-10, 100], [-10, 100], color='k', linestyle='--', label='1:1')
                plt.xlim(-5, 75)
                plt.ylim(-5, 75)
                plt.xlabel('Modeled degree-day factor')
                plt.ylabel('Predicted degree-day factor')
                plt.text(0.98, 0.05, f'R$2=$ {r2:.3f}',transform=plt.gca().transAxes,ha='right',va='bottom')
                plt.title(f'Best model for {glaciersite}\napplied on {compare}')
                plt.savefig(base_fp + f'{glaciersite}_on{compare}_predictions.png', dpi=300)
                plt.close()
        
    # plt.figure()
    # plt.plot(n_features, maes)
    # plt.ylabel('MAE')
    # plt.xlabel('# of features')
    # plt.show()

    # plt.figure()
    # plt.plot(n_features, r2s)
    # plt.ylabel('R$^2$')
    # plt.xlabel('# of features')
    # plt.show()