import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from pebsi.processing.plotting_fxns import *
from objectives import *
import pickle
import os
import copy
import socket, time
import itertools

# Get the base filepath
machine = socket.gethostname()
if 'trace' in machine:
    base_fp = '/trace/group/rounce/cvwilson/Output/'
elif os.path.exists('/mnt/d/grid_search'):
    base_fp = '/mnt/d/grid_search/'
else:
    base_fp = '/home/claire/research/Output/EB/'

# Create some dictionaries with useful information
labels = {'kp':'Precipitation factor','kw':'Wind factor','c5':'Densification c$_5$'}      # Labels for the different parameters we varied
methodlabels = {'MAE':'MAE','ME':'Bias','RMSE':'RMSE','MdAE':'MdAE'}                                          # Labels for the different error methods
errorlabels = {'seasonal':'Seasonal mass balance (m w.e.)',                                     # Labels for the different error metrics with units
               'winter':'Winter mass balance (m w.e.)', 
               'summer':'Summer mass balance (m w.e.)', 
               'annual':'Annual mass balance (m w.e.)',
               'snowdepth':'End-of-winter snow depth (m)',
               'snowmass':'End-of-winter snow mass (m w.e.)',
               'snowdensity':'End-of-winter snow density (kg m-3)',
               '2024':'2024 surface height change (m)'} 
shorterrorlabels = {'2024':'2024 surface height','snowdensity':'Snow density','snowdepth':'Snow depth',
                    'seasonal':'Seasonal MB','winter':'Winter MB','summer':'Summer MB','annual':'Annual MB'}
param_labels = {'kp':'Precipitation factor','c5':'Densification parameter'}
sitedict = {'2024':['AB','ABB','B','BD','D','T'],'long':['A','AU','B','D']}      # Dictionary of sites in the 2024 and long run
all_sites = sitedict['long']+sitedict['2024']+['mean','median']                  # List all sites

# USER OPTIONS
run_info = {'long':{'date':'08_01', 'idx':'0'},                     # Date and index of the grid search (12_04) (01_16) (02_11) (03_05)
            '2024':{'date':'08_02', 'idx':'0'}}                     # (12_06) (03_06)
params = {'c5':[ 0.014,0.016,0.018,0.02,0.022,0.024],  # 0.01, 0.012,
          'kp':[1,1.25,1.5,1.75,2,2.25,2.5,2.75,3]}  # 0.25,0.5,0.75,
for key in params:                                                  # Convert params to strings for processing
    for v,value in enumerate(params[key]):
        params[key][v] = str(value)

# Cutoffs for error to normalize on
error_lims = {'2024':0.5,'snowdensity':100,
              'winter':0.4,'summer':0.9,
              'snowdepth':1,'annual':1}

def get_any(result_dict,c5='0.018',kp='2',site='B',run_type='long',out='ds'):
    """
    Grabs the dataset for a given parameter combination and site
    """
    setno = result_dict[c5][kp][site]['set_no']
    runno = result_dict[c5][kp][site]['run_no']
    if run_type == '2024':
        date = run_info['2024']['date']
        idx = run_info['2024']['idx']
    else:
        date = run_info['long']['date']
        idx = run_info['long']['idx']
    if out == 'ds':
        ds,_,_ = getds(f'/trace/group/rounce/cvwilson/Output/{date}_{site}_{idx}/grid_{date}_set{setno}_run{runno}_0.nc')
        return ds
    elif out == 'fn':
        return f'/trace/group/rounce/cvwilson/Output/{date}_{site}_{idx}/grid_{date}_set{setno}_run{runno}_0.nc'
    else:
        assert 1==0, 'choose from out = ds or fn'

def get_percentile(result_dict, error, percentile=50, method='MAE',plot=False):
    all_error = []
    for c5 in params['c5']:
        for kp in params['kp']:
            if '2024' in error:
                sites = sitedict['2024'] + ['mean']
            else:
                sites = sitedict['long'] + ['mean']
            for site in sites:
                all_error.append(result_dict[c5][kp][site][error+'_'+method])
    if plot:
        fig, ax = plt.subplots(figsize=(2,2))
        sorted_list = np.sort(all_error)
        cdf = np.arange(1, len(sorted_list)+1) / len(sorted_list)
        ax.plot(sorted_list, cdf)
        ax.set_title(error)
        ax.set_ylim(0,1)
        ax.axhline(percentile/100,color='r')
        ax.tick_params(length=5)
        plt.show()
    upper = np.percentile(np.array(all_error), percentile)
    lower = np.min(all_error)
    return lower, upper

def process_run(run_type, fn):
    """
    Regenerates each individual run .pkl file which contains the
    timeseries and run information
    """
    results = {}
    ds = xr.open_dataset(fn)
    site = ds.attrs['site']
    print('Processing', fn)

    if run_type == 'long':
        # Seasonal mass balance
        years,wmod,wmeas,smod,smeas,amod,ameas = seasonal_mass_balance(ds, out='data')
        results['years'] = years
        results['winter_mod'] = wmod
        results['winter_meas'] = np.array(wmeas)
        results['summer_mod'] = smod
        results['summer_meas'] = np.array(smeas)
        results['annual_mod'] = amod
        results['annual_meas'] = np.array(ameas)

        # Seasonal error
        error_dict = seasonal_mass_balance(ds,method=['MAE','MdAE','ME'])
        winter_MAE, winter_MdAE, winter_ME = error_dict['winter']
        summer_MAE, summer_MdAE, summer_ME = error_dict['summer']
        annual_MAE, annual_MdAE, annual_ME = error_dict['annual']
        seasonal_MAE = np.mean([winter_MAE, summer_MAE])
        seasonal_MdAE = np.mean([winter_MdAE, summer_MdAE])
        seasonal_ME = np.mean([winter_ME, summer_ME])
        all_seasonal = {'winter_MAE':winter_MAE,'winter_MdAE':winter_MdAE,'winter_ME':winter_ME,
                    'summer_MAE':summer_MAE,'summer_MdAE':summer_MdAE,'summer_ME':summer_ME,
                    'seasonal_MAE':seasonal_MAE,'seasonal_MdAE':seasonal_MdAE,'seasonal_ME':seasonal_ME,
                    'annual_MAE':annual_MAE,'annual_MdAE':annual_MdAE,'annual_ME':annual_ME}
        for var in all_seasonal:
            results[var] = all_seasonal[var]

        # Snowpits
        years, data = snowpits(ds, out='data')
        for var in data:
            results[var] = data[var]

        # Snowpit error
        for method in ['MAE','ME','MdAE']:
            snowpit_dict = snowpits(ds,method=method, out='mean_error')
            for var in snowpit_dict:
                results[var] = snowpit_dict[var]

    elif run_type == '2024':
        # Cumulative mass balance
        dates, dhmod, dhmeas = cumulative_mass_balance(ds, out='data')
        results['dates'] = dates
        results['dh_mod'] = dhmod
        results['dh_meas'] = dhmeas

        # Error
        results['2024_MAE'] = cumulative_mass_balance(ds, method='MAE')
        results['2024_MdAE'] = cumulative_mass_balance(ds, method='MdAE')
        results['2024_ME'] = cumulative_mass_balance(ds, method='ME')

        # 2024 mass balance for sites where it's measured
        if site not in ['ABB','BD']:
            mbmod,mbmeas = cumulative_mass_balance(ds, out='mbs')
            results['mb2024_mod'] = mbmod
            results['mb2024_meas'] = mbmeas

        # albedo for site B
        if site == 'B':
            dates, albedomod, albedomeas = daily_albedo(ds, out='data')
            results['albedo_dates'] = dates
            results['albedo_mod'] = albedomod
            results['albedo_meas'] = albedomeas

            for method in ['MAE','ME','MdAE']:
                results[f'albedo_{method}'] = daily_albedo(ds, method=method)
        
    # Store the attributes in the results dict
    for attr in ds.attrs:
        results[attr] = ds.attrs[attr]
        if attr in ['c5','kp']:
            results[attr] = str(ds.attrs[attr])

    # Write to new pickle file
    fn_pickle = fn.replace('.nc','.pkl')
    if os.path.exists(fn_pickle):
        os.remove(fn_pickle)
    with open(fn_pickle, 'wb') as file:
        pickle.dump(results, file)

    return results

def create_dict(run_type):
    """
    Builds the results_dict from all individual runs and stores it
    to a .pkl file
    """
    date = run_info[run_type]['date']
    idx = run_info[run_type]['idx']
    
    # Create storage for the grid search results
    grid_dict = {}
    for c5 in params['c5']:
        grid_dict[c5] = {}
        for kp in params['kp']:
            grid_dict[c5][kp] = {}
            for site in sitedict[run_type]:
                grid_dict[c5][kp][site] = {}

    # Loop through all the sites
    for site in sitedict[run_type]:
        if run_type == 'long':
            date = '07_30' if site != 'D' else run_info[run_type]['date']
        for f in os.listdir(base_fp+f'{date}_{site}_{idx}/'):
            if 'pkl' in f:
                # Open individual output pickle 
                with open(base_fp + f'{date}_{site}_{idx}/' + f, 'rb') as file:
                    run_dict = pickle.load(file)
                set_no = f.split('_')[3].split('set')[1]
                run_no = f.split('_')[4].split('run')[1]
                c5 = str(run_dict['c5'])
                kp = str(run_dict['kp'])
                for var in run_dict:
                    if c5 in grid_dict and kp in grid_dict[c5]:
                        grid_dict[c5][kp][site][var] = run_dict[var]
                        grid_dict[c5][kp][site]['set_no'] = set_no
                        grid_dict[c5][kp][site]['run_no'] = run_no

    # Store compiled pickle
    with open(base_fp + f'{date}_{idx}_out.pkl', 'wb') as file:
        pickle.dump(grid_dict, file)

    return grid_dict

def add_site_means(result_dict):
    """
    Adds the 'mean' site to the overall error 
    (winter_MAE, summer_MAE, etc.)
    """
    # ===== Find site means of each error type =====
    # List out all error types
    all_error = list(result_dict['0.018']['2.5']['B'].keys())
        
    # Create dictionary to store site means
    sites_error_dict = {}
    # Loop through all parameters and sites
    for c5 in params['c5']:
        for kp in params['kp']:
            result_dict[c5][kp]['mean'] = {}
            result_dict[c5][kp]['median'] = {}
            for error_type in all_error:
                if '_M' in error_type and 'albedo' not in error_type:
                    sites_error_dict[error_type] = []
                    if '2024' in error_type:
                        for site in sitedict['2024']:
                            sites_error_dict[error_type].append(result_dict[c5][kp][site][error_type])
                    else:
                        for site in sitedict['long']:
                            try:
                                sites_error_dict[error_type].append(result_dict[c5][kp][site][error_type])
                            except:
                                print(site, kp, c5, error_type,' failed')
                    if len(sites_error_dict[error_type]) > 0:
                        result_dict[c5][kp]['mean'][error_type] = np.mean(sites_error_dict[error_type])
                        result_dict[c5][kp]['median'][error_type] = np.median(sites_error_dict[error_type])
    return result_dict

def get_result_dict(force_redo=False):
    """
    Grabs result dictionary. If force_redo or if the .pkl doesn't exist,
    runs create_dict to combine the individual run dictionaries.
    Adds site means for the overall error metrics.

    Format of dict:
    > c5
        > kp
            > site
                > timeseries
                    years, winter_mod, winter_meas, etc.
                > error stats for the entire timeseries
                    winter_MAE, winter_ME, etc.
                > run information
                    set_no, run_no, model_run_date, etc.
    """
    # ===== Open or parse output dictionary for both runs =====
    both_dict = {}
    parse_runs = ['long','2024']
    for run_type in parse_runs:  
        date = run_info[run_type]['date']
        idx = run_info[run_type]['idx']
        if not os.path.exists(base_fp + f'{date}_{idx}_out.pkl') or force_redo in [True, run_type]:
            # No compiled pickle exists: create dictionary
            grid_dict = create_dict(run_type)
        else:
            # Compiled pickle exists: load it
            with open(base_fp + f'{date}_{idx}_out.pkl', 'rb') as file:
                grid_dict = pickle.load(file)
        
        # Store the dictionary under the run type (long or 2024)
        both_dict[run_type] = grid_dict

    # Condense long and 2024 runs into to a single result_dict
    result_dict = both_dict['long']
    if '2024' in parse_runs:
        for c5 in params['c5']:
            for kp in params['kp']:
                # Add the 2024 error stats to the result_dict
                for site in sitedict['2024']:
                    # Some sites are different from long run, so add a slot for these runs
                    if site not in result_dict[c5][kp]:
                        result_dict[c5][kp][site] = {}
                    # Add the 2024 error stats
                    for var in both_dict['2024'][c5][kp][site]:
                        result_dict[c5][kp][site][var] = both_dict['2024'][c5][kp][site][var]
    
    result_dict = add_site_means(result_dict)
    return result_dict

def plot_error_cdf(result_dict, error_type, site='mean'):
    """Plots the cumulative distribution function for error metrics"""
    # Choose error type to plot
    fig,ax = plt.subplots(figsize=(4,3))

    # List out all errors in that type
    error_list = []
    for c5 in params['c5']:
        for kp in params['kp']:
            error_list.append(result_dict[c5][kp][site][error_type])

    # Evaluate CDF
    sorted_list = np.sort(error_list)
    cdf = np.arange(1, len(sorted_list)+1) / len(sorted_list)

    # Plot
    ax.plot(sorted_list, cdf)
    ax.set_xlabel('Value',fontsize=12)
    ax.set_ylabel('CDF',fontsize=12)
    ax.set_ylim(0,1)
    ax.set_xlim(sorted_list[0],sorted_list[-1])
    ax.tick_params(length=5)
    label = error_type.replace('_',' ').replace('snow','snow ')
    ax.set_title(f'Cumulative distribution of {label}')
    plt.show()

def not_dominated(sol, others):
    """
    Checks if an input solution is dominated by any others.

    sol : array of len(error_list)
    others: array of (len(error_list), len(all_parameter_combos))
    """
    others = others[~np.all(others == sol, axis=1)]
    return np.all(np.any(others > sol, axis=1) | np.any(others >= sol, axis=1))
        
def get_pareto_fronts_bootstrap(n_iterations, result_dict, error_list, 
                                site='mean', metric='MAE', split=0.7,
                                return_validation=False,return_optima=False):
    """
    Iteratively determines a random subset of the data, calculates
    error on this subset, and determines Pareto fronts, which are
    then aggregated into a list.
    If 'return_validation', returns a list of li
    """
    # Store all Pareto fronts from all bootstrap samples
    all_bootstrap_pareto = []
    all_calib = []
    all_valid = []
    optima = []
    result_copy = copy.deepcopy(result_dict)
    # import pygem_eb.processing.gridsearch_plotting as gsplot

    for iter in range(n_iterations):
        # Store results for each iteration
        iteration_dict = {'calib':{},'valid':{}}
        iter_start = time.time()

        # Sample random indices for this iteration
        calib_base = np.sort(np.random.choice(24, size=int(24*split), replace=False))
        valid_base = np.sort(np.delete(np.arange(24, dtype=int), calib_base))

        # Loop through all parameter combinations and evaluate on the random indices
        for c5 in params['c5']:
            for kp in params['kp']:
                # Store a list of all error metrics
                iteration_dict['calib'][c5+'_'+kp] = []
                iteration_dict['valid'][c5+'_'+kp] = []

                # Loop through error metrics
                for error in error_list:
                    # Determine if looping over all sites or just one
                    if site in ['mean','median']:
                        all_sites = sitedict['long']
                    else:
                        all_sites = [site]
                    
                    # If storing site means, need a list of all sites
                    site_error = {'calib':[], 'valid':[]}

                    # Loop through sites
                    for ss in all_sites:
                        # Load the timeseries for this combination of parameter, site and error
                        site_meas = result_dict[c5][kp][ss][error+'_meas']
                        site_mod = result_dict[c5][kp][ss][error+'_mod']

                        if ss == 'A':
                            calib_idx = calib_base[calib_base < 14]
                            valid_idx = valid_base[valid_base < 14]
                        elif ss == 'AU':
                            calib_idx = calib_base[calib_base > 12] - 12
                            valid_idx = valid_base[valid_base > 12] - 12
                        else:
                            calib_idx = copy.deepcopy(calib_base)
                            valid_idx = copy.deepcopy(valid_base)

                        for i, idx in enumerate([calib_idx, valid_idx]):
                            subset = ['calib','valid'][i]
                            if len(idx) > 0:
                                if error != 'snowdensity':
                                    # Calculate the error on the subset
                                    sample_mod = np.array(site_mod)[idx]
                                    sample_meas = np.array(site_meas)[idx]

                                    # Snow depth has missing years, so we need to filter those out
                                    if np.all(np.isnan(sample_mod)):
                                        error_value = np.nan
                                    else:
                                        error_value = objective(sample_mod, sample_meas, metric)

                                    if site == 'mean':
                                        # Store each site in a list if finding the mean of sites
                                        site_error[subset].append(error_value)
                                    else:
                                        # Otherwise just store the value from this site
                                        site_error[subset] = error_value
                                else:
                                    # Snow density contains yearly timeseries, so evaluate each
                                    all_years = []
                                    for year in idx:
                                        if np.all(np.isnan(site_mod[year])):
                                            all_years.append(np.nan)
                                        else:
                                            all_years.append(objective(site_mod[year], site_meas[year], metric))

                                    # Snow density has missing years, so we need to filter those out
                                    if np.all(np.isnan(all_years)):
                                        error_value = np.nan
                                    else:
                                        error_value = np.nanmean(all_years)

                                    if site == 'mean':
                                        # Store each site in a list if finding the mean of sites
                                        site_error[subset].append(error_value)
                                    else:
                                        # Otherwise just store the value from this site
                                        site_error[subset] = error_value
                                
                                # Store the calibration data to visualize heatmap (debugging)
                                if subset == 'calib':
                                    result_copy[c5][kp][ss][error+'_MAE'] = error_value
                            else:
                                # len(idx) can be zero for sites A and AU since years are limited
                                site_error[subset].append(np.nan)
                            
                    for subset in ['calib','valid']:
                        # Find the error for the site we're interested in
                        if site == 'mean':
                            site_value = np.nanmean(site_error[subset])
                        elif site == 'median':
                            site_value = np.nanmedian(site_error[subset])
                        else:
                            site_value = site_error[subset]
                    
                        # Store the value
                        iteration_dict[subset][c5+'_'+kp].append(site_value)

                        # Store the calibration data to visualize the heatmap
                        if subset == 'calib':
                            result_copy[c5][kp][site][error+'_MAE'] = site_value
       
        # Put error values into an array and normalize it
        solutions = np.array(list(iteration_dict['calib'].values()))
        min_vals = np.min(solutions, axis=0)
        max_vals = np.max(solutions, axis=0)
        solutions_norm = (solutions - min_vals) / (max_vals - min_vals)

        # Identify non-dominated solutions
        pareto_params = []
        for item in iteration_dict['calib']:
            sol = np.array(iteration_dict['calib'][item])
            sol_norm = (sol - min_vals) / (max_vals - min_vals)
            param_dup = (item.split('_')[0], item.split('_')[1])

            if not_dominated(sol_norm, solutions_norm):
                all_calib.append(sol)
                all_valid.append(np.array(iteration_dict['valid'][item]))
                pareto_params.append(param_dup)
                # cmap = mpl.colormaps.get_cmap('plasma')
                # norm = mpl.colors.Normalize(vmin=0,vmax=n_iterations)
                # plt.scatter(float(param_dup[1]), float(param_dup[0]), color=cmap(norm(iter)),s=(n_iterations-iter)*10)
        optima_iter = []
        for e in range(len(error_list)):
            argmin = np.argmin(solutions[:,e])
            item = list(iteration_dict['calib'].keys())[argmin]
            optima_iter.append((item.split('_')[0], item.split('_')[1]))
        optima.append(optima_iter)

        # DEBUGGING
        # for ss in ['mean']:
        #     gsplot.plot_pareto_heatmap(pareto_params, result_copy, error_list, optima=[optima_iter], bootstrap=False, site=ss, savefig=True) # 

        all_bootstrap_pareto.append(pareto_params)
        iter_timer = time.time() - iter_start
        if iter < 1:
            print(f'One iteration takes {iter_timer:.2f} seconds')

    if n_iterations == 1:
        return all_bootstrap_pareto, iteration_dict
    elif return_validation and not return_optima:
        return all_bootstrap_pareto, np.array(all_calib), np.array(all_valid)
    elif return_optima and not return_validation:
        return all_bootstrap_pareto, optima 
    elif return_validation and return_optima:
        return all_bootstrap_pareto, np.array(all_calib), np.array(all_valid), optima
    else:
        return all_bootstrap_pareto
    
def get_frequency(all_bootstrap_pareto):
    """
    Checks all the bootstrap parameter combinations and creates
    a dictionary that stores the frequency of appearance.
    Returns this dictionary and the parameter set that occurs
    at the highest frequency ('best')
    """
    frequency_dict = {}

    # Iterate over the list of lists
    for combination_list in all_bootstrap_pareto:
        for c5, kp in combination_list:
            # Check if c5 is already in the dictionary
            if c5 not in frequency_dict:
                frequency_dict[c5] = {}
            # Check if kp is already in the nested dictionary
            if kp not in frequency_dict[c5]:
                frequency_dict[c5][kp] = 0
            # Increment the frequency count for each (c5, kp) pair
            frequency_dict[c5][kp] += 1

    max_freq = 0
    for c5 in frequency_dict:
        for kp in frequency_dict[c5]:
            frequency = frequency_dict[c5][kp]
            if frequency > max_freq:
                max_freq = frequency
                best = (c5, kp)

    return frequency_dict, best


# =======================================================================================
# ========================= OUTDATED FUNCTIONS BELOW THIS POINT =========================
# =======================================================================================

def get_all_error(ds, run_type):
    if run_type == 'long':
        # seasonal mass balance
        error_dict = seasonal_mass_balance(ds,method=['MAE','MdAE','ME'])
        winter_MAE, winter_MdAE, winter_ME = error_dict['winter']
        summer_MAE, summer_MdAE, summer_ME = error_dict['summer']
        annual_MAE, annual_MdAE, annual_ME = error_dict['annual']
        seasonal_MAE = np.mean([winter_MAE, summer_MAE])
        seasonal_MdAE = np.mean([winter_MdAE, summer_MdAE])
        seasonal_ME = np.mean([winter_ME, summer_ME])
        results = {'winter_MAE':winter_MAE,'winter_MdAE':winter_MdAE,'winter_ME':winter_ME,
                   'summer_MAE':summer_MAE,'summer_MdAE':summer_MdAE,'summer_ME':summer_ME,
                   'seasonal_MAE':seasonal_MAE,'seasonal_MdAE':seasonal_MdAE,'seasonal_ME':seasonal_ME,
                   'annual_MAE':annual_MAE,'annual_MdAE':annual_MdAE,'annual_ME':annual_ME}

        # snowpits
        for method in ['MAE','ME','MdAE']:
            snowpit_dict = snowpits(ds,method=method)
            for var in snowpit_dict:
                results[var] = snowpit_dict[var]

    elif run_type == '2024':
        MAE = cumulative_mass_balance(ds,method='MAE')
        MdAE = cumulative_mass_balance(ds,method='MdAE')
        ME = cumulative_mass_balance(ds,method='ME')
        results = {'MAE':MAE,'ME':ME,'MdAE':MdAE}

    return results

def add_normalized(result_dict, error_lims=error_lims, pareto=False):
    """
    Normalizes error between 0-1 for the min-max of a given metric
    
    Returns result_dict with normalized error
    """

    # Grab list of all error metrics
    all_error = list(result_dict['0.018']['2.5']['B'].keys())
    all_error.remove('run_no')
    all_error.remove('set_no')

    # Divide each value by the minimum to get error_norm
    for c5 in params['c5']:
        for kp in params['kp']:
            if pareto:
                include = True if (c5,kp) in pareto else False
            else:
                include = True
            for site in all_sites:
                if site in result_dict[c5][kp]:
                    list_errors = copy.deepcopy(result_dict[c5][kp][site])
                    for error_type in list_errors:
                        if error_type.split('_M')[0] in error_lims:
                            current_value = result_dict[c5][kp][site][error_type]
                            min_value = error_lims[error_type.split('_M')[0]][0]
                            max_value = error_lims[error_type.split('_M')[0]][1]
                            
                            if include and np.abs(max_value - min_value) > 0:
                                scaled_value = (current_value - min_value) / (max_value - min_value)
                                result_dict[c5][kp][site][error_type+'_norm'] = scaled_value
                            else:
                                result_dict[c5][kp][site][error_type+'_norm'] = np.inf
    
    return result_dict

def pareto_sweep(points):
    """
    Returns the indices of the pareto front points in the passed array
    """
    # Sort points by x-coordinate, then by y-coordinate
    sorted_indices = np.argsort(points[:, 0])
    sorted_points = points[sorted_indices]
    
    pareto_indices = []
    current_min_y = np.inf

    for i, point in enumerate(sorted_points):
        if point[1] < current_min_y: # F
            pareto_indices.append(sorted_indices[i])  # Store the original index
            current_min_y = point[1]

    return np.array(pareto_indices)

def get_best_normalized(error_list, result_dict, add_weights=None, site='mean', prints=False):
    """
    Uses normalized error 
    """
    # Get weights arrays
    weights = []
    for i in range(len(error_list)):
        zeros = [0]*len(error_list)
        zeros[i] = 1
        weights.append(zeros)
    weights.append(np.ones(len(error_list)) * 1/len(error_list))
    if add_weights is not None:
        for item in add_weights:
            assert len(item) == len(error_list), 'Check formatting of add_weights'
            weights.append(item)

    # Loop through weights
    if prints:
        print('Weights:')
        print(' '.join(f'{x:>10}' for x in error_list),'    best')
    best_all = []
    for weight in weights:
        assert np.abs(np.sum(weight) - 1) < 1e-3
        weighted_list = [np.inf]
        for c5 in params['c5']:
            for kp in params['kp']:
                errors = []
                for error in error_list:
                    errors.append(result_dict[c5][kp][site][error+'_MAE_norm'])
                weighted_error = np.sum(np.array(weight) * np.array(errors))
                if len(weighted_list) > 0 and weighted_error < np.nanmin(weighted_list):
                    best = (c5,kp)
                weighted_list.append(weighted_error)
        if prints:
            print(' '.join(f'{x:>10}' for x in weight),'    ',best)
        best_all.append(best)
    return best_all