import os
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

site_dict = {'kahiltna':['K53','K17b'], #  ,'K14C','KPS'
             'kennicott':['GTH','GTL'], # KENNICOTT   'KC31',
             'gulkana':['AU','B','D'], # GULKANA 
             'wolverine':['N','EC'], # WOLVERINE     # 'B'
            #  'lemoncreek':['B','C','D'], # LEMON CREEK
             'taku':['MG1','NWB1'], # TAKU ,'TKG3'
             }


groups = {
          'all':['gulkana','wolverine','taku','kennicott','kahiltna','lemon_creek'], 
          'coastal':['wolverine','taku','lemon_creek'],
          'continental':['gulkana','kennicott','kahiltna'],
          }

all_dfs = {}
for group in groups:
    all_dfs[group] = None

for glacier in site_dict:
    sites = site_dict[glacier]

    for site in sites:
        df_fn = base_fp+f'{glacier}{site}_df.csv'
        df = pd.read_csv(df_fn, index_col=0, parse_dates=True)[['ddf','accum_cumsum','pdds_cumsum','bc_dep_cumsum']]

        # store
        all_dfs[glacier+site] = df

        # add this data to each grouped dataframe
        for group in groups:
            if glacier in groups[group]:
                if all_dfs[group] is not None:
                    all_dfs[group] = pd.concat([all_dfs[group], df])
                else:
                    all_dfs[group] = df 

# define variables for regression
target = 'ddf'

# loop through sites and train models 
for glaciersite in groups:
    df = all_dfs[glaciersite]
    vars_use = np.array(df.columns[df.columns != target])

    n_features = []
    r2s = []
    maes = []
    best_r2 = -np.inf

    while len(vars_use) > 2:
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
        
        print('built model for', len(vars_use), '    R2:', r2)
        least_important = vars_use[indices[-1]]
        vars_use = vars_use[vars_use != least_important]

    # plot the best model performance 
    y_pred_test = best_model.predict(best_data['X_test'])
    y_pred_train = best_model.predict(best_data['X_train'])
    y_test = best_data['y_test']
    y_train = best_data['y_train']
    r2_test = sk.metrics.r2_score(y_test, y_pred_test)
    r2_train = sk.metrics.r2_score(y_train, y_pred_train)
    n_features_best = len(best_vars)
    vars_joined = ', '.join(best_vars)
    if len(best_vars) > 3:
        vars_joined = ', '.join(best_vars[:2]) + ',\n' + ', '.join(best_vars[2:])

    print(f'Best model for {glaciersite} uses {n_features_best}: {vars_joined}')
    
    plt.figure()
    plt.scatter(y_train, y_pred_train, alpha=0.8, color='#63c4c7', label='Train')
    plt.scatter(y_test, y_pred_test, alpha=0.8, color='#BF1F6A', label='Test')
    plt.plot([-10, 100], [-10, 100], color='k', linestyle='--', label='1:1')
    plt.xlim(-5, 75)
    plt.ylim(-5, 75)
    plt.xlabel('Modeled degree-day factor')
    plt.ylabel('Predicted degree-day factor')
    plt.text(0.98, 0.05, f'Using {n_features_best} features\nR$^2$ (train)$=$ {r2_train:.3f}\nR$^2$ (test)$=$ {r2_test:.3f}',
             transform=plt.gca().transAxes,ha='right',va='bottom')
    plt.title(f'Best model for {glaciersite}\n{vars_joined}')
    plt.legend()
    plt.savefig(base_fp + f'{glaciersite}_predictions.png', dpi=300)
    plt.close()

    if glaciersite in groups:
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
    # plt.savefig(base_fp + f'{glaciersite}_nfeatures.png')
    # plt.show()