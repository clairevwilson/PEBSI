# built-in libraries
import socket
import os
import argparse
import json
# external libraries
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import xarray as xr
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
import statsmodels.api as sm
from scipy.stats import gaussian_kde as kde
import matplotlib.pyplot as plt
import matplotlib as mpl
import joblib
# internal libraries
from data_handling import MassBalance

# ===========================================================================================
#                                       USER OPTIONS
# ===========================================================================================
surface = 'snow'        # ice, snow or both
hold_constant = ('SW_absorbed', 1.31211e-6)
reprocess_dfs = True   # reprocess individual dataframes?
group = 'all'           # all, coastal, continental, accumulation, or ablation
time_res = 'daily'      # daily or hourly
regression_vars = ['positive_temp','SW_absorbed']           # vars to use in melt regression
albedo_regression_vars = ['PDD_cumsum','log_bc_5d_rolling','days_since_acc','SWin']  # vars to use in albedo regression

# ===========================================================================================
#                                       LOAD INPUTS 
# ===========================================================================================
# parse arguments 
parser = argparse.ArgumentParser()
parser.add_argument('-g','--group', default=group, type=str,
                    help='group from options: [all, coastal, continental, accumulation, ablation]')
parser.add_argument('-t','--time_res', default=time_res, type=str,
                    help='time resolution from options: [daily, hourly]')
parser.add_argument('-r','--regression_vars', default=regression_vars, type=str, nargs='+',
                    help='listed regression vars from options: [positive_temp, bc_5d_rolling, SW_absorbed, accum, rain]')
parser.add_argument('-s','--surface', default=surface, type=str,
                    help='surface type [snow or ice]')
parser.add_argument('-a','--albedo_predicted', action='store_true', help='test on ML-derived albedo?')
parser.add_argument('-sf','--savefig', action='store_true', help='store figures?')
parser.add_argument('-plot_kde', action='store_true', help='plot KDE heatmap?')
parser.add_argument('-debug', action='store_true', help='print check-ins?')
args = parser.parse_args()

# define filepaths
if 'trace' in socket.gethostname():
    base_fp = '/trace/group/rounce/cvwilson/Output/'
    home_fp = '/trace/home/cvwilson/research/'
else:
    base_fp = 'C:/Users/cvw30/Research/Output/'
    home_fp = 'C:/Users/cvw30/Research/'
sim_fn = base_fp + 'ddf/GLACIERSITE_2026_04_17_ukesm_0.nc'
all_df_fn = base_fp + f'ddf/all_{args.surface}_{args.time_res}_df.csv' # UKESM_
temp_fn = base_fp + f'ddf/temp_GLACIERSITE_{args.surface}_{args.time_res}_df.csv'
groups_fn = base_fp + f'ddf/glacier_groups.json'
model_fn = base_fp + 'ddf/albedo_model.joblib'

# define colors for plotting
colors = ['#63c4c7','#fcc02e','#4D559C','#60C252','#BF1F6A',
              '#F77808','#298282','#999999','#FF89B0','#427801']

# define sites 
site_dict = {'wolverine':['N','B','EC'],
                'kahiltna':['K53','K17b'],
                'kennicott':['GTL','GTH','KC31'],
                'lemon_creek':['C','D'], # 'B',
                'taku':['NWB1','MG1','TKG3'],
                'gulkana':['AU','B','D']
                }

# define MERRA-2 coordinates 
coord_dict = {'wolverine':'60.5_-148.7',
                'kahiltna':'63.0_-151.2',
                'kennicott':'61.5_-143.1',
                'taku':'58.5_-134.3',
                'lemon_creek':'58.5_-134.3',
                'gulkana':'63.5_-145.6'}

# list potential distinctions to perform regression on
if not os.path.exists(groups_fn):
    # get list of sites in ablation and accumulation area based on measured annual mass balance
    list_accumulation = []
    list_ablation = []
    for glacier in site_dict:
        for site in site_dict[glacier]:
            data = MassBalance(glacier, site, use='benchmark annual')
            if np.mean(data.data) > 0:
                list_accumulation.append(glacier+site)
            else:
                list_ablation.append(glacier+site)

    # create all_groups dict  
    all_groups = {
            'all':['gulkana','wolverine','taku','kennicott','kahiltna','lemon_creek'], 
            'coastal':['wolverine','taku','lemon_creek'],
            'continental':['gulkana','kennicott','kahiltna'],
            'accumulation_area':list_accumulation,
            'ablation_area':list_ablation
          }
    
    # add individual sites to groups
    for site in site_dict:
        all_groups[site] = [site]
    with open(groups_fn, 'w') as f:
        json.dump(all_groups, f, indent=4)
else:
    with open(groups_fn, 'r') as f:
        all_groups = json.load(f)

# separately define glacier- and site-based groupings
glacier_based_groups = list(site_dict.keys()) + ['all','coastal','continental']
site_based_groups = ['accumulation_area','ablation_area']

# list strings to use for each regression variable in written equation
var_dict = {'positive_temp':{'const':r'f_T', 'var':r'T_+'},
            'bc_5d_rolling':{'const':r'f_{BC}', 'var':r'BC_{5d}'},
            'LWin':{'const':r'f_{LW}', 'var':r'LW_{in}'},
            'SW_absorbed':{'const':r'f_{SW}', 'var':r'SW_{in}*(1-\alpha)'},
            'rain':{'const':r'f_r', 'var':r'P_{liq}'},
            'accum':{'const':r'f_a', 'var':r'P_{solid}'},
            'PDD_cumsum':{'const':r'f_T', 'var':r'T_{+, \text{cumsum}}'}
            }

# ===========================================================================================
#             STEP 1: PROCESS OR LOAD DATAFRAME FROM RAW MODEL RUNS
# ===========================================================================================
if reprocess_dfs or not os.path.exists(all_df_fn):
    if args.debug:
        print('Processing datasets for all glaciers...')
    all_df = None
    for glacier in tqdm(site_dict, desc='glacier loop'):
        # load the glacier coordinates for the MERRA-2 data
        coords = coord_dict[glacier]

        # grab MERRA-2 data
        ds_bcd = xr.open_dataset(f'/trace/group/rounce/cvwilson/climate_data/MERRA2/{coords}/BCDP002_{coords}.nc')
        ds_bcw = xr.open_dataset(f'/trace/group/rounce/cvwilson/climate_data/MERRA2/{coords}/BCWT002_{coords}.nc')
        # ds_sw = xr.open_dataset(f'/trace/group/rounce/cvwilson/climate_data/MERRA2/{coords}/SWGDN_{coords}.nc')
        ds_lw = xr.open_dataset(f'/trace/group/rounce/cvwilson/climate_data/MERRA2/{coords}/LWGAB_{coords}.nc')

        # combine wet and dry deposition
        ds_bc = ds_bcd['BCDP002'] + ds_bcw['BCWT002']
        
        # convert time to be in the Alaska timezone
        ds_bc = ds_bc.assign_coords({'time':ds_bc.time.values - pd.Timedelta(hours=8)})
        # ds_sw = ds_sw.assign_coords({'time':ds_sw.time.values - pd.Timedelta(hours=8)})
        ds_lw = ds_lw.assign_coords({'time':ds_lw.time.values - pd.Timedelta(hours=8)})

        for site in tqdm(site_dict[glacier], desc=f'{glacier} site loop', leave=False):
            site_temp_fn = temp_fn.replace('GLACIER',glacier).replace('SITE', site)
            if os.path.exists(site_temp_fn):
                df = pd.read_csv(site_temp_fn, index_col=0, parse_dates=True)
            else:
                sim_fn_site = sim_fn.replace('GLACIER', glacier).replace('SITE', site)
                print(sim_fn_site)
                ds = xr.open_dataset(sim_fn_site)
                time_res_hours = 24 if args.time_res == 'daily' else 1

                if args.time_res != 'hourly':
                    # must be daily then
                    time_res = '1d'
                    # surface_type = ds['layer_type'].isel(layer=0).resample(time=time_res).min()
                    snow_depth = ds['layerheight'].where(ds['layertype'] < 2).sum(dim='layer').resample(time=time_res).min() # m
                    melt = ds['melt'].resample({'time': time_res}).sum() * 1000 # m w.e.
                    positive_temp = ds['airtemp'].resample({'time': time_res}).mean().where(lambda x: x > 0, other=0) # C
                    positive_temp_sum = ds['airtemp'].where(ds['airtemp']>0, other=0).resample({'time': time_res}).sum() / 24 # C
                    accum = ds['accum'].resample({'time': time_res}).sum() * 1000 # m w.e.
                    rain = ds['rainfall'].resample({'time': time_res}).sum() * 1000 # m w.e.
                    albedo = ds['albedo'].resample({'time': time_res}).min() # -
                    bc = ds_bc.resample(time=time_res).sum() * 3600 # kg m-2
                    SWin = ds['SWin'].resample(time=time_res).sum() * 3600 # J m-2
                    LWin = ds_lw['LWGAB'].resample(time=time_res).sum() * 3600 # J m-2

                    # days since accumulation
                    last = np.maximum.accumulate(np.where(accum > 1e-3, np.arange(len(accum)), -1))
                    days_since_acc = np.arange(len(accum)) - last
                    days_since_acc = xr.DataArray(days_since_acc, dims=['time'], coords={'time': accum.time}).astype(float)
                else:
                    snow_depth = ds['layerheight'].where(ds['layertype'] < 2).sum(dim='layer') # m
                    melt = ds['melt'] * 1000 # m w.e.
                    positive_temp = ds['airtemp'].where(lambda x: x > 0) # C
                    accum = ds['accum'] * 1000 # m w.e.
                    rain = ds['rainfall'] * 1000 # m w.e.
                    albedo = ds['albedo'] # -
                    bc = ds_bc * 3600 # kg m02
                    SWin = ds['SWin'] * 3600 # J m-2
                    LWin = ds_lw['LWGAB'] * 3600 # J m-2
                    days_since_acc = ds['melt'] * np.nan # PLACEHOLDER

                # build new dataset
                ds_out = xr.Dataset({
                    'melt': melt,
                    'positive_temp': positive_temp_sum,
                    'accum': accum,
                    'rain': rain,
                    'albedo':albedo,
                    'bc_dep':bc,
                    'SWin':SWin,
                    'LWin':LWin,
                    'days_since_acc':days_since_acc,
                })

                # clip to bounds to help in computational messiness
                # ds_out = ds_out.where(positive_temp > np.nanquantile(positive_temp.values, 0.2)) # avoids small PDDs in early summer
                # ds_out = ds_out.where(melt > 0.1 * time_res_hours) # avoids small melt timesteps
                if args.surface == 'snow':
                    ds_out = ds_out.where(snow_depth > 0.05) # only days with snow or firn
                elif args.surface == 'ice':
                    ds_out = ds_out.where(snow_depth < 1e-8) # only days with ice 
                # only include March through October
                ds_out = ds_out.where(ds_out.time.dt.month.isin(range(3, 11)), drop=True)

                # define lists of variables to sum cumulatively and drop from dataframe
                cum_vars = ['bc_dep', 'positive_temp']
                drop_vars = ['lat','lon']

                # add cumulative variables
                water_year = xr.where(ds_out['time.month'] >= 10, ds_out['time.year'] + 1, ds_out['time.year'])
                for var in cum_vars: 
                    ds_out[f'{var}_cumsum'] = (ds_out[var].groupby(water_year).cumsum())
                # drop_vars += cum_vars

                # add rolling deposition (n days prior to each timestep)
                n_rolling_bc = 5
                ds_out[f'bc_{n_rolling_bc}d_rolling'] = ds_out['bc_dep'].rolling(time=n_rolling_bc).sum()

                # create dataframe and drop variables
                df = ds_out.to_dataframe().drop(columns=drop_vars)
                df = df.rename(columns={'positive_temp_cumsum':'PDD_cumsum'})

                # crop out any nans
                df.dropna(inplace=True)

                # add glacier and site
                df['glacier'] = [glacier] * len(df.index)
                df['glaciersite'] = [glacier + site] * len(df.index)

                # store temp file
                df.to_csv(site_temp_fn)

            # concatenate with other data
            if all_df is None:
                all_df = df 
            else:
                all_df = pd.concat([all_df, df])

    # store the built .csv
    all_df.to_csv(all_df_fn)
    for glacier in site_dict:
        for site in site_dict[glacier]:
            os.remove(temp_fn.replace('GLACIER', glacier).replace('SITE', site))
else:
    # open dataframe processed in `get_ddf.py` containing hourly data for all sites
    all_df = pd.read_csv(all_df_fn, index_col=0, parse_dates=True)

print('stored UKESM results')
assert 1==0

# clip the dataframe to the sites of interest
if args.group in glacier_based_groups:
    glac_df = all_df.loc[all_df['glacier'].isin(all_groups[args.group])].copy()
elif args.group in site_based_groups:
    glac_df = all_df.loc[all_df['glaciersite'].isin(all_groups[args.group])].copy()

# add absorbed SW term using albedo
glac_df['SW_absorbed'] = glac_df['SWin'] * (1-glac_df['albedo'])
if 'positive_temp' not in glac_df.columns:
    glac_df['positive_temp'] = glac_df['pdds_1d_rolling']
if args.debug:
    print(f'Got dataframe! Running regression on {args.regression_vars}')

# get log term for BC
if 'log_bc_5d_rolling' not in glac_df.columns:
    glac_df['log_bc_5d_rolling'] = np.log1p(glac_df['bc_5d_rolling'] * 1e9)

# subset the dataframe differently for melt
albedo_df = glac_df.copy()
melt_lim = 2 if time_res == 'daily' else 0.1
glac_df = glac_df.loc[glac_df['melt'] > melt_lim]

albedo_df['doy'] = albedo_df.index.day_of_year
albedo_df = albedo_df.loc[albedo_df['doy'] > 90]

# ===========================================================================================
#                              STEP 2: RUN THE REGRESSION 
# ===========================================================================================
# define variables for regression
X = glac_df[args.regression_vars]
y = glac_df['melt']

if hold_constant is not None:
    fixed_var, C = hold_constant 
    X = glac_df[args.regression_vars].drop(columns=[fixed_var])
    y = glac_df['melt'] - (C * glac_df[fixed_var])
else:
    fixed_var = None

# train OLS model
model = sm.OLS(y, X).fit()
if args.debug:
    print(model.summary())

# extract model parameters
factors = []
for var in args.regression_vars:
    if var == fixed_var:
        factors.append(C)
    else:
        factors.append(model.params[var])

# evaluate the predicted melt
melt_predicted = np.sum(glac_df[args.regression_vars] * factors, axis=1)
melt_actual = glac_df['melt']

# if using absorbed_SW, test on predicted albedo
if args.albedo_predicted and 'SW_absorbed' in args.regression_vars:
    # get data to predict albedo
    X = albedo_df[albedo_regression_vars]
    y = albedo_df['albedo']

    # train random forest on albedo data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # separate variables
    bc_vars = [v for v in albedo_regression_vars if 'bc' in v]
    phys_vars = [v for v in albedo_regression_vars if 'bc' not in v]
    bc_var = bc_vars[0]
    constraints = [-1 if 'bc' in col.lower() else 0 for col in X_train.columns]

    # train model
    albedo_model = HistGradientBoostingRegressor(
        monotonic_cst=constraints,
        max_iter=100,
        max_depth=10, 
        random_state=42
    ) 
    # albedo_model = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42)
    albedo_model.fit(X_train, y_train)
    joblib.dump(albedo_model, model_fn)

    # FEATURE IMPORTANCE
    result = sk.inspection.permutation_importance(
        albedo_model, X_train, y_train, n_repeats=10, random_state=42, n_jobs=1
    )
    feature_importance = pd.Series(result.importances_mean, index=X_train.columns)
    feature_importance = feature_importance.sort_values(ascending=True)
    plt.figure(figsize=(6, 3))
    feature_importance.plot(kind='barh', color='teal')
    plt.title('Feature Importance')
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.savefig(base_fp + '../figs/ddf/albedo_features.png', bbox_inches='tight', dpi=300)
    plt.close()

    # PARTIAL DEPENDENCE PLOT
    fig, axes = plt.subplots(2, 2, figsize=(5, 5), gridspec_kw={'hspace':0.3})
    display = sk.inspection.PartialDependenceDisplay.from_estimator(
        albedo_model, 
        X_train, 
        features=albedo_regression_vars,
        kind="average", # Shows the global average impact
        ax=axes
    )
    plt.savefig(base_fp + '../figs/ddf/albedo_PDP.png', bbox_inches='tight', dpi=300)
    plt.close()

    # evaluate the predicted melt
    albedo_predicted = albedo_model.predict(X_test)
    albedo_actual = y_test

    # get error metrics
    meas = np.array(albedo_actual).ravel()
    mod = np.array(albedo_predicted).ravel()
    alb_bias = np.nanmean(meas - mod)
    alb_mae = np.nanmean(np.abs(meas - mod))
    alb_r2 = 1 - np.sum(np.square(meas - mod)) / np.sum(np.square(meas - np.mean(meas)))

# ===========================================================================================
#                                STEP 3: PLOT THE RESULTS
# ===========================================================================================
# FIGURE 1: melt regression on 1:1 space
fig, ax = plt.subplots(figsize=(4, 4))

# build stack for KDE plot
if args.plot_kde:
    x = melt_actual.values
    y = melt_predicted.values
    xy = np.vstack([x, y])
    z = kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx], y[idx], z[idx]

    # plot data
    ax.scatter(x, y, c=z, cmap='magma')
else:
    # plot data in one color
    ax.scatter(melt_actual, melt_predicted, c=colors[4], alpha=0.6)

# add error metrics
mae = np.nanmean(np.abs(melt_predicted - melt_actual))
bias = np.nanmean(melt_predicted - melt_actual)
r2_melt = model.rsquared

# if using albedo predictor, add that scatter plot
if args.albedo_predicted and 'SW_absorbed' in args.regression_vars:
    # first, gather normal predictions (with black carbon)
    X_wBC = glac_df[albedo_regression_vars]
    glac_df['albedo_ml'] = albedo_model.predict(X_wBC) # for just the days with melt
    albedo_df['albedo_ml']  = albedo_model.predict(albedo_df[albedo_regression_vars]) # for entire timeseries

    # calculate column for the (1-a)SWin term in the melt regression
    glac_df['SW_absorbed_ml'] = (1-glac_df['albedo_ml']) * glac_df['SWin']

    # build new X from predicted albedo
    regression_vars = list(args.regression_vars)
    idx = args.regression_vars.index('SW_absorbed')
    regression_vars[idx] = 'SW_absorbed_ml'
    X = glac_df[regression_vars]

    # predict melt using ML shortwave term
    melt_predicted = np.sum(X * factors, axis=1)

    # REPEAT with no BC
    # create copies of X to replace BC
    X_noBC = glac_df[albedo_regression_vars].copy() 
    X_noBC_all = albedo_df[albedo_regression_vars].copy()

    # replace BC with zeros
    X_noBC[bc_var] *= 0.1
    X_noBC_all[bc_var] *= 0.1

    # predict albedo without BC and get the (1-a)SWin term
    glac_df['albedo_ml_noBC'] = albedo_model.predict(X_noBC)
    albedo_df['albedo_ml_noBC']= albedo_model.predict(X_noBC_all)
    glac_df['SW_absorbed_ml_noBC'] = (1-glac_df['albedo_ml_noBC']) * glac_df['SWin']

    # build new X and predict melt from no-BC albedo
    regression_vars[idx] = 'SW_absorbed_ml_noBC'
    X_melt_noBC = glac_df[regression_vars]
    melt_predicted_noBC = np.sum(X_melt_noBC * factors, axis=1)

    print('Difference in albedo with - without BC:')
    print(f'    Mean: {np.mean(albedo_df['albedo_ml'] - albedo_df['albedo_ml_noBC']):.3f}')
    print(f'  Median: {np.median(albedo_df['albedo_ml'] - albedo_df['albedo_ml_noBC']):.3f}')
    print(f'     Max: {np.max(albedo_df['albedo_ml'] - albedo_df['albedo_ml_noBC']):.3f}')
    print(f'     Min: {np.min(albedo_df['albedo_ml'] - albedo_df['albedo_ml_noBC']):.3f}')
    print()
    print('Difference in melt with - without BC:')
    print(f'    Mean: {np.mean(melt_predicted - melt_predicted_noBC):.3f} mm w.e.')
    print(f'  Median: {np.median(melt_predicted - melt_predicted_noBC):.3f} mm w.e.')
    print(f'     Max: {np.max(melt_predicted - melt_predicted_noBC):.3f} mm w.e.')
    print(f'     Min: {np.min(melt_predicted - melt_predicted_noBC):.3f} mm w.e.')

    # plot the histogram of albedo with and without BC to understand the process
    fig2, ax2 = plt.subplots(figsize=(3, 3))
    albedo_bins = np.arange(0.1, 0.9, 0.05)
    diff_bins = np.arange(-0.3, 0.3, 0.02)
    ax2.set_title('Gradient boosting predictions\nWith BC $-$ Without BC')
    ax2.hist(albedo_df['albedo_ml'] - albedo_df['albedo_ml_noBC'], bins=diff_bins, color=colors[6], histtype='step', linewidth=2, label='With $-$ Without BC') # histtype='step', 
    ax2.set_xlabel('Residual')
    # ax2.hist(albedo_df['albedo_ml'], bins=albedo_bins, color=colors[6], histtype='step', linewidth=2, label='With BC') # histtype='step', 
    # ax2.hist(albedo_df['albedo_ml_noBC'], bins=albedo_bins, color=colors[5], histtype='step', linewidth=2, label='Without BC')
    # ax2.legend()
    # ax2.set_xlabel('Albedo')
    ax2.set_ylabel('Count')
    fig2.savefig(base_fp + '../figs/ddf/albedo_noBC_comparison_residuals.png', bbox_inches='tight', dpi=300)
    
    fig3, ax3 = plt.subplots(figsize=(3, 3))
    diff_bins = np.arange(-4, 4, 0.2)
    ax3.set_title('Melt predictions\nWith BC $-$ Without BC')
    ax3.hist(melt_predicted - melt_predicted_noBC, bins=diff_bins, color=colors[6], histtype='step', linewidth=2, label='With $-$ Without BC') # histtype='step', 
    ax3.set_xlabel('Residual')
    ax2.set_ylabel('Count')
    fig3.savefig(base_fp + '../figs/ddf/melt_noBC_comparison_residuals.png', bbox_inches='tight', dpi=300)

    if args.plot_kde:
        x = melt_actual.values
        y = melt_predicted.values
        xy = np.vstack([x, y])
        z = kde(xy)(xy)
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

        # plot data
        ax.scatter(x, y, c=z, cmap='magma')
    else:
        # plot data in one color
        ax.scatter(melt_actual, melt_predicted, c=colors[1], alpha=0.6, marker='^')

    # add error metrics
    meas_melt = np.array(melt_actual).ravel()
    mod_melt = np.array(melt_predicted).ravel()
    r2 = 1 - np.sum(np.square(meas_melt - mod_melt)) / np.sum(np.square(meas_melt))
    # ax.text(0.02, 0.88, r'R$^2$ with predicted $\alpha$: '+f'{r2:.3f}', transform=ax.transAxes, c=colors[1])
    r2_str = f'R$^2$: {r2_melt:.3f} ({r2:.3f} with '+ r'$\alpha_{pred}$)'
else:
    r2_str = f'R$^2$: {r2_melt:.3f}'
ax.text(0.02, 0.95, r2_str, transform=ax.transAxes, c=colors[4])
ax.text(0.02, 0.81, f'MAE: {mae:.3f} mm w.e.', transform=ax.transAxes, c=colors[4])
ax.text(0.02, 0.88, f'Bias: {bias:.3f} mm w.e.', transform=ax.transAxes, c=colors[4])

# plot 1:1 line 
min_value = 0
max_value = np.max([melt_actual, melt_predicted]) * 1.05
ax.plot([min_value, max_value],[min_value, max_value],'k--')

# beautify
# ax.legend()
ax.tick_params(length=5)
ax.set_xlim(min_value, max_value)
ax.set_ylim(min_value, max_value)
ax.set_xlabel('PEBSI melt (mm w.e.)')
ax.set_ylabel('Predicted melt (mm w.e.)')

# get equation string
equation = '$melt='
for var, param_value in zip(args.regression_vars, factors):
    if param_value < 1e-2:
        equation += f'{param_value:.3e}'.replace('e-0','e-')
    elif param_value < 100 :
        equation += f'{param_value:.3f}'
    else:
        equation += f'{param_value:.2e}'.replace('e+0','e')
    equation += '*' + var_dict[var]['var']
    if var != args.regression_vars[-1]:
        equation += '+'
equation += '$'
# ax.text(0.5, -0.1, equation, ha='center', transform=ax.transAxes)

vars_comma = ', '.join(args.regression_vars)
ax.set_title(f'Regression on {args.surface} of {args.group} glaciers with {args.time_res} data\n{equation}')
if args.savefig:
    vars_str = '_'.join(args.regression_vars)
    if args.albedo_predicted and 'SW_absorbed' in args.regression_vars:
        vars_str += '_predalbedo'
    fig.savefig(base_fp + f'../figs/ddf/{args.group}_{args.time_res}_{args.surface}_regression_{vars_str}.png', dpi=300, bbox_inches='tight')
if args.debug:
    print('Got Figure 1...')
# ===========================================================================================
# FIGURE 2: histogram of predicted melt 
fig, (ax1, ax2) = plt.subplots(2, figsize=(5, 4), gridspec_kw={'hspace':0.4})

# histogram of residuals (ax1)
if args.surface == 'snow':
    bins_diff = np.arange(-20.5, 20, 1)
else:
    bins_diff = np.arange(-30.5, 30, 1)
ax1.hist(melt_actual - melt_predicted, bins=bins_diff, histtype='step', edgecolor=colors[0])
ax1.axvline(0, c='k', linewidth=0.5)
ax1.set_title('Melt residuals (mm w.e.) (predicted $-$ PEBSI)')

# histogram of melt values (ax2)
bins = np.arange(0, 80, 2)
ax2.hist(melt_actual, histtype='step', label='PEBSI', bins=bins, edgecolor=colors[1])
ax2.hist(melt_predicted, histtype='step', label='Predicted', bins=bins, edgecolor=colors[4])
ax2.set_title('Melt (mm w.e.)')
ax2.legend()

# save figure
if args.savefig:
    plt.savefig(base_fp + f'../figs/ddf/melt_histogram_{args.surface}.png',dpi=300)
if args.debug:
    print('Got Figure 2...')
# ===========================================================================================
# FIGURE 3: albedo predictions on 1:1 space
if args.albedo_predicted and 'SW_absorbed' in args.regression_vars:
    fig, ax = plt.subplots(figsize=(4, 4))

    # build stack for KDE plot
    if args.plot_kde:
        xy = np.vstack([meas, mod])
        z = kde(xy)(xy)
        idx = z.argsort()
        x, y, z = meas[idx], mod[idx], z[idx]

        # plot data
        ax.scatter(x, y, c=z, cmap='magma')
    else:
        # plot data in one color
        ax.scatter(albedo_actual, albedo_predicted, c=colors[4], alpha=0.6)
    ax.plot([0,1],[0,1],'k--')

    # add error metrics
    ax.text(0.02, 0.95, f'Bias: {alb_bias:.3f}', transform=ax.transAxes)
    ax.text(0.02, 0.88, f'MAE: {alb_mae:.3f}', transform=ax.transAxes)
    ax.text(0.02, 0.81, f'R$^2$: {alb_r2:.3f}', transform=ax.transAxes)

    # beautify
    ax.tick_params(length=5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('PEBSI albedo')
    ax.set_ylabel('Predicted albedo')
    ax.set_title(f'Albedo predictions\n{args.group.capitalize()} glaciers with {args.time_res} data')
    if args.savefig:
        plt.savefig(base_fp + f'../figs/ddf/{args.group}_{args.time_res}_albedo_regression.png', dpi=300, bbox_inches='tight')
    if args.debug:
        print('Got Figure 3...')
    # ===========================================================================================
    # FIGURE 4: albedo residuals 
    fig, (ax1, ax2) = plt.subplots(2, figsize=(5, 4), gridspec_kw={'hspace':0.4})

    # histogram of residuals (ax1)
    bins_diff = np.arange(-0.21, 0.2, 0.02)
    less = glac_df.loc[glac_df['albedo'] < np.median(glac_df['albedo'])]
    more = glac_df.loc[glac_df['albedo'] >= np.median(glac_df['albedo'])]
    ax1.hist(less['albedo_ml'] - less['albedo'], bins=bins_diff, histtype='step', edgecolor=colors[0], label=r'$\alpha$ < $\alpha_{median}$')
    ax1.hist(more['albedo_ml'] - more['albedo'], bins=bins_diff, histtype='step', edgecolor=colors[2], label=r'$\alpha$ >= $\alpha_{median}$')
    ax1.set_title('Albedo residuals (predicted $-$ PEBSI)')
    ax1.axvline(0, c='k', linewidth=0.5)
    ax1.legend()

    # histogram of albedo values (ax1)
    bins = np.arange(0.1, 0.9, 0.02)
    ax2.hist(glac_df['albedo'], histtype='step', label='PEBSI', bins=bins, edgecolor=colors[1])
    ax2.hist(glac_df['albedo_ml'], histtype='step', label='Predicted', bins=bins, edgecolor=colors[4])
    ax2.set_xlim(0.1, 0.9)
    ax2.set_title('Albedo')
    ax2.legend()

    # add suplabels
    fig.supylabel('Count')
    n = len(glac_df.index)
    ax1.text(0.02, 0.9, f'$n={n}$', transform=ax1.transAxes)
    if args.savefig:
        plt.savefig(base_fp + '../figs/ddf/albedo_histogram.png', dpi=300)
    if args.debug:
        print('Got Figure 4...')
# ===========================================================================================
# FIGURE 5: correlation heatmap
if args.savefig:
    corr = glac_df[['melt','positive_temp','SWin','SW_absorbed','bc_dep_cumsum','bc_5d_rolling','albedo']].corr()
    plt.figure()
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'Correlation matrix for {args.group} glaciers')
    plt.tight_layout()
    plt.savefig(base_fp + f'../figs/ddf/{args.time_res}_{args.surface}_correlation.png', dpi=300)
    if args.debug:
        print('Got Figure 5...')

# save the dataframe containing all the data
albedo_df.to_csv(base_fp + f'ddf/{args.group}_{args.surface}_df_processed.csv')
# print(glac_df.loc[glac_df['melt'] > 100])
print(factors)