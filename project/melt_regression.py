# built-in libraries
import socket
import os
import argparse
# external libraries
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy.stats import gaussian_kde as kde
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# internal libraries
from data_handling import MassBalance

# ===============================================
# =================== OPTIONS ===================
# ===============================================
group = 'all'
time_res = 'daily'
regression_vars = ['positive_temp','SW_absorbed']
albedo_regression_vars = ['positive_temp','bc_dep_cumsum']

# ===============================================
# ==================== INPUTS ===================
# ===============================================
# parse arguments 
parser = argparse.ArgumentParser()
parser.add_argument('-g','--group', default=group, type=str,
                    help='group from options: [all, coastal, continental, accumulation, ablation]')
parser.add_argument('-t','--time_res', default=time_res, type=str,
                    help='time resolution from options: [daily, hourly]')
parser.add_argument('-r','--regression_vars', default=regression_vars, type=str, nargs='+',
                    help='listed regression vars from options: [positive_temp, bc_5d_rolling, SW_absorbed, accum, rain]')
parser.add_argument('-a','--albedo_predicted', action='store_true', help='test on ML-derived albedo?')
parser.add_argument('-sf','--savefig', action='store_true', help='store figures?')
parser.add_argument('-plot_kde', action='store_true', help='plot KDE heatmap?')
args = parser.parse_args()

# define filepaths
if 'trace' in socket.gethostname():
    base_fp = '/trace/group/rounce/cvwilson/Output/'
    home_fp = '/trace/home/cvwilson/research/'
else:
    base_fp = 'C:/Users/cvw30/Research/Output/'
    home_fp = 'C:/Users/cvw30/Research/'

# define colors for plotting
colors = ['#63c4c7','#fcc02e','#4D559C','#60C252','#BF1F6A',
              '#F77808','#298282','#999999','#FF89B0','#427801']

# define sites 
site_dict = {'wolverine':['N','B','EC'],
                'kahiltna':['K53','K17b'],
                'kennicott':['GTL','GTH','KC31'],
                'lemon_creek':['B','C','D'],
                'taku':['NWB1','MG1','TKG3'],
                'gulkana':['AU','B','D']
                }

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

# list potential distinctions to perform regression on
all_groups = {
          'all':['gulkana','wolverine','taku','kennicott','kahiltna','lemon_creek'], 
          'coastal':['wolverine','taku','lemon_creek'],
          'continental':['gulkana','kennicott','kahiltna'],
          'accumulation_area':list_accumulation,
          'ablation_area':list_ablation
          }

# list strings to use for each regression variable in written equation
var_dict = {'positive_temp':{'const':r'f_T', 'var':r'T_+'},
            'bc_5d_rolling':{'const':r'f_{BC}', 'var':r'BC_{5d}'},
            'SW_absorbed':{'const':r'f_{SW}', 'var':r'SW_{in}*(1-\alpha)'},
            'rain':{'const':r'f_r', 'var':r'P_{liq}'},
            'accum':{'const':r'f_a', 'var':r'P_{solid}'},
            }

# open dataframe processed in `get_ddf.py` containing hourly data for all sites
all_df = pd.read_csv(base_fp+f'ddf/all_{args.time_res}_df.csv')
if args.group in ['all','coastal','continental']:
    glac_df = all_df.loc[all_df['glacier'].isin(all_groups[args.group])].copy()
else:
    glac_df = all_df.loc[all_df['glaciersite'].isin(all_groups[args.group])].copy()
glac_df['SW_absorbed'] = glac_df['SWin'] * (1-glac_df['albedo'])
if 'positive_temp' not in glac_df.columns:
    glac_df['positive_temp'] = glac_df['pdds_1d_rolling']

# plot correlation heatmap
if args.savefig:
    corr = glac_df[['melt','positive_temp','SWin','SW_absorbed','bc_dep_cumsum','bc_5d_rolling','albedo']].corr()
    plt.figure()
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'Correlation matrix for {args.group} glaciers')
    plt.tight_layout()
    plt.savefig(base_fp + f'ddf/figs/{args.time_res}_correlation.png', dpi=300)
    plt.show()

# ===============================================
# ================ REGRESSION ===================
# ===============================================
# define variables for regression
X = glac_df[args.regression_vars] # 
y = glac_df['melt']

# train OLS model
model = sm.OLS(y, X).fit()
print(model.summary())

# extract model parameters
factors = [model.params[factor] for factor in args.regression_vars]

# evaluate the predicted melt
melt_predicted = np.sum(X * factors, axis=1)
melt_actual = glac_df['melt']

# if using absorbed_SW, test on predicted albedo
if args.albedo_predicted and 'SW_absorbed' in args.regression_vars:
    # get data to predict albedo
    X = glac_df[albedo_regression_vars]
    y = glac_df['albedo']

    # train random forest on albedo data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    albedo_model = RandomForestRegressor(n_estimators=100, n_jobs=-1, max_depth = 20, random_state=42)
    albedo_model.fit(X_train, y_train)

    # evaluate the predicted melt
    albedo_predicted = albedo_model.predict(X_train)
    albedo_actual = y_train

    # get error metrics
    meas = np.array(albedo_actual).ravel()
    mod = np.array(albedo_predicted).ravel()
    bias = np.nanmean(meas - mod)
    mae = np.nanmean(np.abs(meas - mod))
    r2 = 1 - np.sum(np.square(meas - mod)) / np.sum(np.square(meas - np.mean(meas)))

    # scatterplot of predictions on 1:1 space
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(albedo_actual, albedo_predicted)
    ax.plot([0,1],[0,1],'k--')

    # add error metrics
    ax.text(0.02, 0.95, f'Bias: {bias:.3f}', transform=ax.transAxes)
    ax.text(0.02, 0.88, f'MAE: {mae:.3f}', transform=ax.transAxes)
    ax.text(0.02, 0.81, f'R$^2$: {r2:.3f}', transform=ax.transAxes)

    # beautify
    # ax.legend()
    ax.tick_params(length=5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('PEBSI albedo')
    ax.set_ylabel('Predicted albedo')
    ax.set_title(f'Albedo random forest predictions\n{args.group.capitalize()} glaciers with {args.time_res} data')
    if args.savefig:
        plt.savefig(base_fp + f'ddf/figs/{args.group}_{args.time_res}_albedo_regression.png', dpi=300, bbox_inches='tight')
    plt.show()

# create figure
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
    ax.scatter(melt_actual, melt_predicted, c=z, cmap='magma')
else:
    # plot data in one color
    ax.scatter(melt_actual, melt_predicted, c=colors[4], alpha=0.6)

# add error metrics
mae = np.nanmean(np.abs(melt_predicted - melt_actual))
bias = np.nanmean(melt_predicted - melt_actual)
r2 = model.rsquared
ax.text(0.02, 0.95, f'R$^2$: {r2:.3f}', transform=ax.transAxes, c=colors[4])

# if using albedo predictor, add that scatter plot
if args.albedo_predicted and 'SW_absorbed' in args.regression_vars:
    glac_df['albedo_ml'] = albedo_model.predict(glac_df[albedo_regression_vars])

    # first calculate column for the (1-a)SWin term
    glac_df['SW_absorbed_ml'] = (1-glac_df['albedo_ml']) * glac_df['SWin']

    # build new X from predicted albedo
    regression_vars = list(args.regression_vars)
    idx = args.regression_vars.index('SW_absorbed')
    regression_vars[idx] = 'SW_absorbed_ml'
    X = glac_df[regression_vars]

    # evaluate the predicted melt
    melt_predicted = np.sum(X * factors, axis=1)

    if args.plot_kde:
        # plot data in black
        ax.scatter(melt_actual, melt_predicted, c='k', alpha=0.6, marker='^')
    else:
        # plot data in one color
        ax.scatter(melt_actual, melt_predicted, c=colors[1], alpha=0.6, marker='^')

    # add error metrics
    meas = np.array(melt_actual).ravel()
    mod = np.array(melt_predicted).ravel()
    r2 = 1 - np.sum(np.square(meas - mod)) / np.sum(np.square(meas))
    ax.text(0.02, 0.88, r'R$^2$ with predicted $\alpha$: '+f'{r2:.3f}', transform=ax.transAxes, c=colors[1])
else:
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
# ax.set_title(f'{args.group.capitalize()} glaciers\n$melt={ddf:.3f}*T_++{bcf:.2e}*'.replace('e+0','e')+r'BC_{5d}$')

# get equation string
equation = '$melt='
for var in args.regression_vars:
    param_value = model.params[var]
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
ax.set_title(f'Regression of {args.group} glaciers with {args.time_res} data\n{equation}')
if args.savefig:
    vars_str = '_'.join(args.regression_vars)
    if args.albedo_predicted and 'SW_absorbed' in args.regression_vars:
        vars_str += '_predalbedo'
    plt.savefig(base_fp + f'ddf/figs/{args.group}_{args.time_res}_regression_{vars_str}.png', dpi=300, bbox_inches='tight')
plt.show()