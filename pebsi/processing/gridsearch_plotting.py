import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
import socket
import copy
import os,sys
import pickle
mpl.rcParams['font.family'] = 'Liberation Sans'

# Get the base filepath
machine = socket.gethostname()
if 'trace' in machine:
    base_fp = '/trace/group/rounce/cvwilson/Output/'
    pygem_fp = '/trace/home/cvwilson/research/PyGEM-EB/'
elif os.path.exists('/mnt/d/grid_search'):
    base_fp = '/mnt/d/grid_search/'
    pygem_fp = '/home/claire/research/PyGEM-EB/'
else:
    base_fp = '/home/claire/research/Output/EB/'
    pygem_fp = '/home/claire/research/PyGEM-EB/'
sys.path.append(pygem_fp)
from objectives import *
from pebsi.processing.plotting_fxns import *
import pebsi.processing.gridsearch_processing as gsproc

# Create some dictionaries with useful information
labels = {'kp':'Precipitation factor','kw':'Wind factor','c5':'Densification c$_5$'}      # Labels for the different parameters we varied
methodlabels = {'MAE':'MAE','ME':'Bias','RMSE':'RMSE','MdAE':'mAE'}                                          # Labels for the different error methods
errorlabels = {'seasonal':'Seasonal mass balance (m w.e.)',                                     # Labels for the different error metrics with units
               'winter':'Winter mass\nbalance (m w.e.)', 
               'summer':'Summer mass\nbalance (m w.e.)', 
               'annual':'Annual mass\nbalance (m w.e.)',
               'snowdepth':'End-of-winter\nsnow depth (m)',
               'snowmass':'End-of-winter\nsnow mass (m w.e.)',
               'snowdensity':'End-of-winter snow\ndensity (kg m-3)',
               '2024':'2024 surface height change (m)',
               'weighted':'Weighted error (-)'} 
shorterrorlabels = {'2024':'2024 surface height','snowdensity':'Snow density (kg m$^{-3}$)','snowdepth':'Snow depth (m)','snowmass':'Snow mass',
                    'seasonal':'Seasonal MB (m w.e.)','winter':'Winter MB (m w.e.)','summer':'Summer MB (m w.e.)','annual':'Annual MB'}
units = {'2024':'m','snowdensity':'kg m$^{-3}$','snowdepth':'m','winter':'m w.e.','summer':'m w.e.','annual':'m w.e.'}
param_labels = {'kp':'Precipitation factor','c5':'Densification parameter'}
medians = {'kp':'2.6','c5':'0.024'}                                              # Median value of each of the parameters
sitedict = {'2024':['AB','ABB','B','BD','D','T'],'long':['A','AU','B','D']}            # Dictionary of sites in the 2024 and long run
all_sites = sitedict['long']+sitedict['2024']+['mean']                                 # List all sites
run_info = gsproc.run_info
params = gsproc.params
for key in params:                                                  # Convert params to strings for processing
    for v,value in enumerate(params[key]):
        params[key][v] = str(value)

# Load diverging cmaps
diverging_cmaps = []
for i in range(5):
    with open(pygem_fp + f'data/cmaps/custom_divergent_cmap_{i}.pkl','rb') as f:
        diverging_cmaps.append(pickle.load(f))
all_colors = ['#63c4c7','#fcc02e','#4D559C','#60C252','#BF1F6A',
              '#F77808','#298282','#999999','#FF89B0','#427801']

def modify_colormap(cmap_name, min=0, max=0.7):
    # Get the colormap
    cmap = plt.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    new_colors = [cmap(norm(min)), cmap(norm(max))]
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(f'{cmap_name}_sat', new_colors)
    return new_cmap

gradient_cmaps = []
for color in ['Greens','Purples','Blues','Reds','Greys','Oranges']:
    gradient_cmaps.append(modify_colormap(color))

def plot_any(c5,kp,site,result_dict,plot_type='seasonal'):
    if '2024' in plot_type:
        date = run_info['2024']['date']
        idx = run_info['2024']['idx']
        setno = result_dict[c5][kp][site]['2024_set_no']
        runno = result_dict[c5][kp][site]['2024_run_no']
    else:
        date = run_info['long']['date']
        idx = run_info['long']['idx']
        setno = result_dict[c5][kp][site]['set_no']
        runno = result_dict[c5][kp][site]['run_no']
    ds,_,_ = getds(f'/trace/group/rounce/cvwilson/Output/{date}_{site}_{idx}/grid_{date}_set{setno}_run{runno}_0.nc')
    if plot_type == 'seasonal':
        plot_seasonal_mass_balance(ds)
    elif plot_type == 'annual':
        plot_seasonal_mass_balance(ds, plot_var='ba')
    else:
        plot_2024_mass_balance(ds)
    plt.show()

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

def plot_pareto_fronts(error_list, result_dict, split=0.7,
                       site='mean', savefig=False):
    """
    Takes all combinations of error metrics and finds the optimal solutions
    for each pair of metrics
    """
    n_comb = len(list(itertools.combinations(error_list,2)))

    # Create the plot axes
    # fig = plt.figure(figsize=(7,6))
    n_rows = n_comb // 2
    n_rows = n_rows + 1 if n_comb % 2 != 0 else n_rows
    # gs = mpl.gridspec.GridSpec(2, 3, wspace=0.5, hspace=0.5)
    # axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(n_comb)]
    fig, axes = plt.subplots(2,3, figsize=(6.3,3.8)) #, gridspec_kw={'wspace':0.6, 'hspace':0.6})
    axes = axes.flatten()

    # Style
    # if colored_param == 'kp':
    #     vmin,vmax = (1,3.5)
    #     ip = 1
    # elif colored_param == 'c5':
    #     vmin,vmax = (0.018,0.028)
    #     ip = 0

    #     param_list = gsproc.params['c5']
    #     cbar_ax = fig.add_axes([0.85, 0.10, 0.02, 0.79])
    #     cmap = mpl.colormaps.get_cmap('viridis_r')
    #     norm =  mpl.colors.Normalize(vmin=0.01,vmax=0.024)
    #     param_arr = np.array(param_list).astype(float)
    #     boundaries = np.append(np.array([param_arr[0] - 0.002]), param_arr)
    #     labeled_ticks = [0.01,0.012,0.014,0.016,0.018,0.02,0.022,0.024]
    #     tick_locations = [f-0.05 for f in labeled_ticks]
    #     cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
    #                 cax=cbar_ax,
    #                 orientation='vertical',
    #                 boundaries=boundaries,ticks=tick_locations,
    #                 spacing='proportional')
    #     cb.ax.set_yticks(tick_locations)
    #     cb.ax.tick_params(labelsize=10,direction='inout',length=8)
    #     cb.ax.tick_params(which='minor', length=2)
    #     cb.ax.set_yticklabels(labeled_ticks)
    #     cb.ax.set_title('$c_5$')
    # norm = mpl.colors.Normalize(vmin=vmin,vmax=vmax)
    # cmap = mpl.colormaps.get_cmap('viridis_r')

    all_pareto_fronts, iteration_dict = gsproc.get_pareto_fronts_bootstrap(1, result_dict, error_list, site, split=split)
    iteration_dict = iteration_dict['calib']

    # Split error into lists
    all_error = np.array(list(iteration_dict.values()))
    # all_params = [float(item.split('_')[ip]) for item in iteration_dict]
    
    # Loop through combos of error metrics
    combos = itertools.combinations(np.arange(len(error_list)), 2)
    for i, (ix, iy) in enumerate(combos):
        ax = axes[i]
        ax.scatter(all_error[:, ix],all_error[:, iy],color='k') # cmap(norm(all_params))) 
        ax.set_xlabel(shorterrorlabels[error_list[ix]],fontsize=10,labelpad=0)
        ax.set_ylabel(shorterrorlabels[error_list[iy]],fontsize=10)  
        ax.tick_params(length=5)
        ax.set_xlim(0, ax.get_xlim()[1]*1.2)
        ax.set_ylim(0, ax.get_ylim()[1]*1.2)
    
        for j, (c5,kp) in enumerate(all_pareto_fronts[0]):
            ax.scatter(iteration_dict[c5+'_'+kp][ix], iteration_dict[c5+'_'+kp][iy],marker='*',color='red',s=12)
    plt.tight_layout()
    if savefig:
        plt.savefig(base_fp+f'pareto_front_scatter_MAE_site{site}.png',dpi=180,bbox_inches='tight')
    plt.show()

def plot_pareto_PMF(error_list, pareto_fronts, result_dict,
                    metric='MAE', plot_type='PMF', site='mean',
                    savefig=False):
    # Define dictionary to store all pareto error
    error_dict = {}
    for error in error_list:
        error_dict[error] = []
    
    # Store c5 and kp of pareto fronts
    params_list = {'c5':[],'kp':[]}
    for (c5,kp) in pareto_fronts:
        c5 = str(c5)
        kp = str(kp).replace('.0','')
        params_list['c5'].append(c5)
        params_list['kp'].append(kp)
        for error in error_dict:
            error_dict[error].append(result_dict[c5][kp][site][error+'_'+metric])
    
    # Create axes
    n_plots = len(error_list)
    n_rows = n_plots // 2
    n_rows = n_rows + 1 if n_plots % 2 != 0 else n_rows
    fig,axes = plt.subplots(n_rows,2,figsize=(5,n_rows*1.5),gridspec_kw={'hspace':0.6})
    axes = axes.flatten()
    for i,ax in enumerate(axes):
        errors = error_dict[list(error_dict.keys())[i]]
        if plot_type == 'CDF':
            sorted_list = np.sort(errors)
            cdf = np.arange(1, len(sorted_list)+1) / len(sorted_list)
            ax.plot(sorted_list, cdf)
        elif plot_type == 'PMF':
            ax.hist(errors, bins=10)
        elif plot_type == 'PDF':
            ax.hist(errors, bins=10,density=True)
        if metric == 'ME':
            ax.axvline(0,linewidth=0.5,color='black')
        ax.set_title(shorterrorlabels[list(error_dict.keys())[i]])
        ax.tick_params(length=5)
    name = 'bias' if metric == 'ME' else metric
    # fig.suptitle(f'{plot_type} of simulation {name} of Pareto front parameter sets',y=1.03)
    fig.supylabel('Count',x=0.06)
    if savefig:
        plt.savefig(base_fp+f'pareto_{metric}_{plot_type}_{site}.png',dpi=180,bbox_inches='tight')
    plt.show()

def plot_pareto_params(pareto_fronts, savefig=False):
    """Plots the locations of Pareto fronts in parameter space"""
    fig,ax = plt.subplots(gridspec_kw={'hspace':0.6},figsize=(6,3))
    kp_float = np.array(params['kp'], dtype=float)
    c5_float = np.array(params['c5'], dtype=float)
    x,y = np.meshgrid(kp_float, c5_float)
    ax.scatter(x.flatten(),y.flatten(),color='gray',s=3)

    dkp = 0.033
    dc5 = 0.00033
    for (c5,kp) in pareto_fronts: #[pareto_fronts[0]]:
        c5 = float(c5)
        kp = float(kp)
        # ax.scatter(kp+dkp, c5+dc5, s=50, marker='s', facecolors='blue',edgecolors='black')
        # ax.scatter(kp-dkp, c5+dc5, s=50, marker='s', facecolors='blue',edgecolors='black')
        # ax.scatter(kp+dkp, c5-dc5, s=50, marker='s', facecolors='blue',edgecolors='black')
        # ax.scatter(kp-dkp, c5-dc5, s=50, marker='s', facecolors='blue',edgecolors='black')
        ax.scatter(kp,c5,color='red',s=50)
    ax.set_xlabel('Precipitation factor $k_p$')
    ax.set_ylabel('Densification parameter $c_5$')
    ax.tick_params(length=5)
    ax.set_xlim(kp_float[0] - 0.15, kp_float[-1] + 0.15)
    ax.set_ylim(c5_float[0] - 0.001, c5_float[-1] + 0.001)
    # fig.suptitle('Pareto front parameter combinations')
    if savefig:
        plt.savefig(base_fp+f'pareto_params.png',dpi=180,bbox_inches='tight')
    plt.show()
    # Plotting PMF of pareto params:
    # for i,ax in enumerate(axes):
    #     param = list(params_list.keys())[i]
    #     errors = np.array(params_list[param],dtype=float)
    #     if plot_type == 'CDF':
    #         sorted_list = np.sort(errors)
    #         cdf = np.arange(1, len(sorted_list)+1) / len(sorted_list)
    #         ax.plot(sorted_list, cdf)
    #     elif plot_type == 'PMF':
    #         ax.hist(errors, bins=10)
    #     elif plot_type == 'PDF':
    #         ax.hist(errors, bins=10,density=True)
    #     ax.set_title(f'${param[0]}_{param[1]}$')
    #     ax.tick_params(length=5)
    # fig.suptitle(f'{plot_type} of Pareto front parameters',y=1.03)
    # fig.supylabel('Count')
    # plt.savefig(base_fp+f'pareto_parameter_{plot_type}.png',dpi=180,bbox_inches='tight')

def plot_pareto_heatmap(pareto_fronts, result_dict, error_names, bootstrap=True,
                        savefig=False, metric='MAE', site='mean', optima=None,
                        figaxes=None, legend=True):
    n_plots = len(error_names)
    if n_plots > 1:
        n_rows = n_plots // 2
        n_rows = n_rows + 1 if n_plots % 2 != 0 else n_rows
        n_cols = 2
    else:
        n_cols = n_rows = 1
    if figaxes is None:
        fig,axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3,n_rows*2),
                             sharex=True, gridspec_kw={'hspace':0.1, 'wspace':0.7}) #, constrained_layout=True)
        if n_cols > 1 or n_rows > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
    else:
        axes = figaxes[1]
        fig = figaxes[0]
    
    for ax in axes:
        ax.set_xticks([1.125,2.125,3.125])
        ax.set_xticklabels(['1','2','3'])
        ax.set_yticks([0.017,0.021,0.025])
        ax.set_yticklabels(['0.016','0.020','0.024'])
        ax.set_xticks([1.375,1.625,1.875,2.375,2.625,2.875,3.375,3.625], minor=True)
        ax.set_yticks([0.015,0.019,0.023],minor=True)
        ax.tick_params(which='major',length=5)
        ax.tick_params(which='minor',length=2)

    if 'AE' in metric:
        cmaps = gradient_cmaps
    else:
        cmaps = diverging_cmaps
    if len(error_names) < 2:
        cmaps = cmaps[3:]
    cmap_max = {'summer': 0.7, 'winter': 0.4, 'snowdensity':100, 'snowdepth':1,'snowmass':0.6, 'annual':0.9}
    all_c5 = [float(c) for c in params['c5']]
    all_kp = [float(k) for k in params['kp']]
    for e,error in enumerate(error_names):
        ax = axes[e]
        error += '_' + metric
        errors = np.empty((len(all_c5),len(all_kp)))
        max_value = 0
        min_value = np.inf
        for c,c5 in enumerate(all_c5):
            for k,kp in enumerate(all_kp):
                error_value = result_dict[str(c5)][str(kp).replace('.0','')][site][error]
                errors[c,k] = error_value

        min_value = np.min(errors)
        if 'AE' in metric:
            max_value = cmap_max[error.split('_')[0]]
        else:
            cmap_max_me = {'summer': 1, 'winter': 0.2, 'snowdensity':100, 'snowdepth':1.5,'snowmass':0.6}
            max_value = cmap_max_me[error.split('_')[0]]
            greater = max(np.abs(max_value), np.abs(min_value))
            max_value, min_value = (greater, -greater)
       
        kp_grid = all_kp + [all_kp[-1] + (all_kp[1] - all_kp[0])]
        c5_grid = all_c5 + [all_c5[-1] + (all_c5[1] - all_c5[0])]
        c = ax.pcolormesh(kp_grid, c5_grid, errors, cmap=cmaps[e], 
                          vmin=min_value, vmax=max_value)
        cbar = fig.colorbar(c, ax=ax)
        cbar.ax.tick_params(length=5)
        cbar.set_label(errorlabels[error.split('_')[0]], rotation=270, labelpad=27, fontsize=12)
        cbar.ax.yaxis.set_label_position('right')

        if not bootstrap:
            vertices = [((float(x2)),float(x1)) for x1,x2 in pareto_fronts]
            for kp, c5 in vertices:
                i_kp = params['kp'].index(str(kp).replace('.0',''))
                i_c5 = params['c5'].index(str(c5))
                if i_kp < len(params['kp']) - 1:
                    width = abs(kp - float(params['kp'][i_kp+1]))
                else:
                    width = 0.25
                if i_c5 < len(params['c5']) - 1:
                    height = abs(c5 - float(params['c5'][i_c5+1]))
                else:
                    height = 0.002
                rect = mpl.patches.Rectangle((kp, c5), width, height, linewidth=2, edgecolor='k', facecolor='none')
                ax.add_patch(rect)
        else:
            # Define colormap
            grays = modify_colormap('Grays',min=0.3,max=1)
            max_freq = max(max(kp_dict.values()) for kp_dict in pareto_fronts.values())
            min_freq = min(min(kp_dict.values()) for kp_dict in pareto_fronts.values())
            max_freq = 1000
            print('hard coded max frequency')
            grays_norm = mpl.colors.Normalize(vmin=0,vmax=max_freq)
            
            # Extract (c5, kp) pairs and their frequencies
            combinations = [(c5, kp, freq) for c5, kp_dict in pareto_fronts.items() for kp, freq in kp_dict.items()]

            # Sort the combinations by frequency
            sorted_combinations = sorted(combinations, key=lambda x: x[2])

            for c5, kp, freq in sorted_combinations:
                i_kp = params['kp'].index(str(kp).replace('.0',''))
                i_c5 = params['c5'].index(str(c5))
                kp = float(kp)
                c5 = float(c5)
                if i_kp < len(params['kp']) - 1:
                    width = abs(kp - float(params['kp'][i_kp+1]))
                else:
                    width = 0.25
                if i_c5 < len(params['c5']) - 1:
                    height = abs(c5 - float(params['c5'][i_c5+1]))
                else:
                    height = 0.002
                # bw(norm_bw(freq)), or (.1,.1,.1,alpha),
                alpha = 1 - (max_freq - freq)/max_freq
                rect = mpl.patches.Rectangle((kp, c5), width, height, linewidth=alpha*2, edgecolor=grays(grays_norm(freq)), facecolor='none')
                ax.add_patch(rect)
            ax.scatter(1.5,0.025, alpha=0.002,s=500)

        if optima is not None:
            all_combos = [opt[e] for opt in optima]
            all_combos_flat = list(set(all_combos))
            for c5, kp in all_combos_flat:
                i_kp = params['kp'].index(str(kp).replace('.0',''))
                i_c5 = params['c5'].index(str(c5))
                kp = float(kp)
                c5 = float(c5)
                if i_kp < len(params['kp']) - 1:
                    width = abs(kp - float(params['kp'][i_kp+1]))
                else:
                    width = 0.25
                if i_c5 < len(params['c5']) - 1:
                    height = abs(c5 - float(params['c5'][i_c5+1]))
                else:
                    height = 0.002
                rect = mpl.patches.Rectangle((kp, c5), width, height, linewidth=1.5, edgecolor='yellow', facecolor='none')
                # ax.add_patch(rect)

    y_pos = 1.0 + 0.08 * n_rows
    # fig.suptitle(f'Heatmap of {methodlabels[metric]} for Pareto front solutions', y=y_pos)
    fig.supylabel(param_labels['c5'],x=-0.01, fontsize=12)
    if len(axes) > 1:
        axes[-2].set_xlabel(param_labels['kp'], fontsize=12)
    axes[-1].set_xlabel(param_labels['kp'], fontsize=12)

    if bootstrap and legend:
        cax = fig.add_axes([0.25,-0.1,0.5,0.03])
        cmap = mpl.colormaps['Grays']
        norm =  mpl.colors.Normalize(vmin=0,vmax=5)
        boundaries = [1,2,3,4,5]
        for b in boundaries:
            alpha = 1 - (5 - b)/5
            rect = mpl.patches.Rectangle((b, 0), 1, 1, linewidth=alpha*2, edgecolor=cmap(norm(b)), facecolor='none')
            cax.add_patch(rect)
        border = 0.1
        cax.set_xlim(np.min(boundaries) - border, np.max(boundaries) + 1 + border)
        cax.set_ylim(-0.3,1.3)
        cax.axis('off')
        cax.text(0.5, -2.7, f'Less frequent ({min_freq})')
        cax.text(4.5, -2.7, f'More frequent ({max_freq})')
        cax.set_title('Pareto fronts')

    if savefig:
        plt.savefig(base_fp+f'pareto_params_heatmap_{metric}_site{site}_{error_names[-1]}.png',dpi=180,bbox_inches='tight')
    if not figaxes: 
        plt.show()
    else:
        return fig, axes

def plot_difference_by_param(best, result_dict, site='B', plot_vars=['2024','annual'],
                             include_best=False, savefig=False):
    if type(site) == str:
        site_dict = {'long':site,'2024':site}
    elif type(site) == dict:
        site_dict = site

    # Create fig and axes
    n_rows = len(plot_vars)+1
    height_ratios = [1]*n_rows
    height_ratios[-1] = 0.1
    fig, axes = plt.subplots(n_rows,3,figsize=(8,n_rows*1.5),
                             gridspec_kw={'wspace':0.28, 'hspace':0.4}, 
                             height_ratios=height_ratios)
    cmaps = [mpl.colormaps.get_cmap('viridis_r'),mpl.colormaps.get_cmap('magma_r')]
    both_vars = [params['c5'],params['kp'][::2]]

    # Top row is for colorbar; left column is empty
    axes[n_rows-1,0].axis('off')
    for j,param in enumerate(['c5','kp']):
        param_list = params[param]
        min,max = (np.min(np.array(param_list,dtype=float)),np.max(np.array(param_list,dtype=float)))
        cmap = cmaps[j]
        norm =  mpl.colors.Normalize(vmin=min,vmax=max)
        title = '$'+param[0]+'_'+param[1]+'$'
        value = '0.032' if param == 'c5' else '4'
        diff = 0.002 if param == 'c5' else 0.25
        param_arr = np.array(param_list).astype(float)
        boundaries = np.append(np.array([param_arr[0] - diff]), param_arr)
        if param == 'c5':
            labeled_ticks = param_list[::2]
            tick_locations = param_arr[::2] - 0.001
        else:
            labeled_ticks = param_list[::2]
            tick_locations = param_arr[::2] - 0.125
        tick_labels = [str(t) for t in labeled_ticks]
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                    cax=axes[n_rows-1,j+1],
                    orientation='horizontal',
                    boundaries=boundaries,ticks=boundaries,
                    spacing='proportional')
        cb.ax.minorticks_on()
        cb.ax.tick_params(which='minor', length=2)
        cb.ax.set_xticks(tick_locations)
        if param == 'kp':
            tick_locations = tick_locations[:-1]
        cb.ax.set_xticks(tick_locations + diff, minor=True)
        cb.ax.set_xticklabels(tick_labels)
        cb.ax.set_ylabel(title)

    # ===== 2024 panels =====
    if '2024' in plot_vars:
        row_2024 = plot_vars.index('2024')
        site_2024 = site_dict['2024']
        axes[row_2024,0].set_ylabel('2024 surface height\nchange (m)',fontsize=11)

        # Data filepath
        fp_gnssir = f'../MB_data/Stakes/gulkana{site}24_GNSSIR.csv'

        # Load GNSSIR daily MB
        df_mb_dict = {}
        if os.path.exists(fp_gnssir):
            df_mb_daily = pd.read_csv(fp_gnssir)
            df_mb_daily.index = pd.to_datetime(df_mb_daily['Date'])
            df_mb_daily['CMB'] -= df_mb_daily['CMB'].iloc[0]
            df_mb_dict['GNSS_IR'] = df_mb_daily.sort_index()

        if 'GNSS_IR' in df_mb_dict:
            df_mb_daily = df_mb_dict['GNSS_IR']
            axes[row_2024,0].plot(df_mb_daily.index,df_mb_daily['CMB'],label='GNSS-IR',linestyle='--',color='black')
            # error bounds
            lower = df_mb_daily['CMB'] - df_mb_daily['sigma']
            upper = df_mb_daily['CMB'] + df_mb_daily['sigma']
            axes[row_2024,0].fill_between(df_mb_daily.index,lower,upper,alpha=0.2,color='black')

        for j in range(2):
            ax = axes[row_2024,j+1]
            param_list = both_vars[j]
            min,max = (np.min(np.array(param_list,dtype=float)),np.max(np.array(param_list,dtype=float)))
            cmap = cmaps[j]
            norm =  mpl.colors.Normalize(vmin=min,vmax=max)
            for value in param_list:
                if j == 0:
                    c5 = value
                    kp = best[1]
                elif j == 1:
                    c5 = best[0]
                    kp = value
                ds = gsproc.get_any(c5,kp,site_2024,result_dict,'2024')
                time,model,data = cumulative_mass_balance(ds,out='data')
                diff = model - data
                ax.plot(time,diff,color=cmap(norm(float(value))),label=value)
            ax.axhline(0,color='k',linewidth=0.5)
            ax.set_ylabel('')
            ax.set_ylim(-1.2,0.5)

        # Best 2024 run
        if include_best:
            c5,kp = best
            ds = gsproc.get_any(c5,kp,site_2024,result_dict,'2024')
            time,model,data = cumulative_mass_balance(ds,out='data')
            axes[row_2024,0].plot(time, model, color='red',label= 'Best 2024 parameters')
            axes[row_2024,1].axhline(0,color='k',linewidth=0.5)

        for ax in axes[row_2024,:]:
            ax.set_xticks(pd.date_range('2024-04-20','2024-08-20',freq='MS'))
            ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%b-%d'))
            ax.set_xlim(pd.to_datetime('2024-04-20'),pd.to_datetime('2024-08-09'))

    # ===== Seasonal panels =====
    if 'winter' in plot_vars or 'summer' in plot_vars or 'annual' in plot_vars:
        site_long = site_dict['long']
        years = result_dict[params['c5'][0]][params['kp'][0]][site_long]['years']
        if 'winter' in plot_vars:
            row_winter = plot_vars.index('winter')
            winter_data = result_dict[params['c5'][0]][params['kp'][0]][site_long]['winter_meas']
            axes[row_winter,0].plot(years, winter_data, label='USGS',linestyle='--',color='black')
            axes[row_winter,0].set_ylabel('Winter mass\nbalance (m w.e.)',fontsize=11)
            for ax in axes[row_winter]:
                ax.set_xlim(2001,2024)
                ax.set_xticks([2005,2015])
            for ax in axes[row_winter,1:]:
                ax.axhline(0,color='k',linewidth=0.5)
        if 'summer' in plot_vars:
            row_summer = plot_vars.index('summer')
            summer_data = result_dict[params['c5'][0]][params['kp'][0]][site_long]['summer_meas']
            axes[row_summer,0].plot(years, summer_data, label='USGS',linestyle='--',color='black')
            axes[row_summer,0].set_ylabel('Summer mass\nbalance (m w.e.)',fontsize=11)
            for ax in axes[row_summer]:
                ax.set_xlim(2001,2024)
                ax.set_xticks([2005,2015])
            for ax in axes[row_summer,1:]:
                ax.axhline(0,color='k',linewidth=0.5)
        if 'annual' in plot_vars:
            row_annual = plot_vars.index('annual')
            annual_data = result_dict[params['c5'][0]][params['kp'][0]][site_long]['annual_meas']
            axes[row_annual,0].plot(years, annual_data, label='USGS',linestyle='--',color='black')
            axes[row_annual,0].set_ylabel('Annual mass\nbalance (m w.e.)',fontsize=11)
            for ax in axes[row_annual]:
                ax.set_xlim(2001,2024)
                ax.set_xticks([2005,2015])
            for ax in axes[row_annual,1:]:
                ax.axhline(0,color='k',linewidth=0.5)

        for j in range(2):
            param_list = both_vars[j]
            cmap = cmaps[j]
            norm =  mpl.colors.Normalize(vmin=0,vmax=len(param_list))
            for i,value in enumerate(param_list):
                if j == 0:
                    c5 = value
                    kp = best[1]
                elif j == 1:
                    c5 = best[0]
                    kp = value
                if 'winter' in plot_vars:
                    diff = result_dict[c5][kp][site_long]['winter_mod'] - winter_data
                    axes[row_winter,j+1].plot(years,diff,color=cmap(norm(i)))
                    axes[row_winter,j+1].set_ylim(-1.5,1.5)
                if 'summer' in plot_vars:
                    diff = result_dict[c5][kp][site_long]['summer_mod'] - summer_data
                    axes[row_summer,j+1].plot(years,diff,color=cmap(norm(i)))
                    axes[row_summer,j+1].set_ylim(-2.5,2.5)
                if 'annual' in plot_vars:
                    diff = result_dict[c5][kp][site_long]['annual_mod'] - annual_data
                    axes[row_annual,j+1].plot(years,diff,color=cmap(norm(i)))
                    axes[row_annual,j+1].set_ylim(-1.5,1.5)

        # Best annual run
        if include_best:
            assert 1==0, 'I didnt code this yet'

    # ===== Snowpit panels =====
    if 'snowdensity' in plot_vars or 'snowdepth' in plot_vars:
        site_long = site_dict['long']
        years = np.array(result_dict['0.018']['2'][site_long]['years'])
        density_meas = result_dict['0.018']['2'][site_long]['snowdensity_meas']
        density_meas = np.array([np.mean(d) for d in density_meas])
        depth_meas = result_dict['0.018']['2'][site_long]['snowdepth_meas']
        depth_meas = np.array([np.mean(d) for d in depth_meas])
        idx_data = np.arange(len(depth_meas)) # np.where(~np.isnan(density_meas))[0]
        if 'snowdensity' in plot_vars:
            row_density = plot_vars.index('snowdensity')
            axes[row_density,0].plot(years[idx_data], density_meas[idx_data], label='USGS',linestyle='--',color='black')
            axes[row_density,0].set_ylabel('End-of-winter snow\ndensity (kg m$^{-3}$)',fontsize=11)
            for ax in axes[row_density]:
                ax.set_xlim(2001,2024)
                ax.set_xticks([2005,2015])
            for ax in axes[row_density,1:]:
                ax.axhline(0,color='k',linewidth=0.5)
        if 'snowdepth' in plot_vars:
            row_depth = plot_vars.index('snowdepth')
            axes[row_depth,0].plot(years[idx_data], depth_meas[idx_data], label='USGS',linestyle='--',color='black')
            axes[row_depth,0].set_ylabel('End-of-winter snow\ndepth (m)',fontsize=11)
            for ax in axes[row_depth]:
                ax.set_xlim(2001,2024)
                ax.set_xticks([2005,2015])
            for ax in axes[row_depth,1:]:
                ax.axhline(0,color='k',linewidth=0.5)
        for j in range(2):
            param_list = both_vars[j]
            cmap = cmaps[j]
            norm =  mpl.colors.Normalize(vmin=0,vmax=len(param_list))
            for i,value in enumerate(param_list):
                if j == 0:
                    c5 = value
                    kp = best[1]
                elif j == 1:
                    c5 = best[0]
                    kp = value
                if 'snowdepth' in plot_vars:
                    depth_mod = result_dict[c5][kp][site_long]['snowdepth_mod']
                    depth_mod = np.array([np.mean(d) for d in depth_mod])
                    diff = depth_mod - np.array(depth_meas)
                    axes[row_depth,j+1].plot(years[idx_data],diff[idx_data],color=cmap(norm(i)))
                    axes[row_depth,j+1].set_ylim(-2.75,2.75)
                if 'snowdensity' in plot_vars:
                    density_mod = result_dict[c5][kp][site_long]['snowdensity_mod']
                    density_mod = np.array([np.mean(d) for d in density_mod])
                    diff = density_mod - np.array(density_meas)
                    axes[row_density,j+1].plot(years[idx_data],diff[idx_data],color=cmap(norm(i)))
                    axes[row_density,j+1].set_ylim(-275,275)
    
    letters = [chr(i) for i in range(ord('a'),ord('z')+1)]
    for a,ax in enumerate(axes.flatten()[:-3]):
        x = 0.05
        y = 0.1
        box_width = 0.1
        box_height = 0.2
                # bbox={'facecolor': 'white', 'edgecolor': 'black'})
        rect = plt.Rectangle((x - box_width / 2, y - box_height / 2), box_width, box_height,
                     linewidth=1.5, edgecolor='k', facecolor='white', zorder=5,transform=ax.transAxes,)
        ax.add_patch(rect)
        ax.text(x, y, letters[a],transform=ax.transAxes,fontsize=12,zorder=6,ha='center', va='center')
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        rect = plt.Rectangle((xlim[0], ylim[0]), xlim[1]-xlim[0], ylim[1]-ylim[0],
                     linewidth=1.5, edgecolor='k', facecolor='none', zorder=7)
        ax.add_patch(rect)
        
    axes[0,1].text(0.95,0.95, f'$c_5$ varies; $k_p={best[1]}$',transform=axes[0,1].transAxes,va='top',ha='right')
    axes[0,2].text(0.95,0.95, f'$k_p$ varies; $c_5={best[0]}$',transform=axes[0,2].transAxes,va='top',ha='right')
    for ax in axes.flatten():
        ax.tick_params(length=5)
    axes[0,0].set_title('Data',fontsize=12,y=1.01)
    fig.text(0.65,0.9,'Modeled $-$ Measured by parameter',ha='center',va='center',fontsize=12)
    if savefig:
        plt.savefig(base_fp+f'all_tradeoffs_{plot_vars[0]}.png',dpi=350,bbox_inches='tight')
    plt.show()

def plot_tradeoffs_2024(result_dict, best, site='B', savefig=False):
    fig, axes = plt.subplots(1,2)
    c5_best = best[0]
    kp_best = best[1]
    norm = mpl.colors.Normalize(vmin=0,vmax=len(params['c5'])-1)
    cmap = mpl.colormaps.get_cmap('viridis_r')
    for j,ax in enumerate(axes):
        list_plots = []
        if j == 0:
            param_list = params['c5']
        elif j == 1:
            param_list = params['kp'][::2]
        for i,value in enumerate(param_list):
            if j == 0:
                c5 = value
                kp = kp_best
            elif j == 1:
                c5 = c5_best
                kp = value
            date = run_info['2024']['date']
            idx = run_info['2024']['idx']
            best_setno = result_dict[c5][kp][site]['set_no']
            best_runno = result_dict[c5][kp][site]['run_no']
            ds,_,_ = getds(base_fp + f'{date}_{site}_{idx}/grid_{date}_set{best_setno}_run{best_runno}_0.nc')
            ax = plot_2024_mass_balance(ds,plot_ax=ax,label=c5,color=cmap(norm(i)))
            plot_empty, = ax.plot(np.nan,np.nan,color=cmap(norm(i)),label=value)
            list_plots.append(plot_empty)
        ax.get_legend().remove()
        title = '$c_5$' if j == 0 else '$k_p$'
        ax.legend(handles=list_plots,fontsize=10,title=title)
    ax.set_ylabel('')
    if savefig:
        plt.savefig(base_fp + f'tradeoffs_snowpits_site{site}.png',dpi=220,bbox_inches='tight')
    plt.show()

def plot_tradeoffs_annual(result_dict, best, site='B', savefig=False):
    fig, all_axes = plt.subplots(2,2,width_ratios=[1,0.2],figsize=(6,8))
    all_axes = all_axes.flatten()
    axes = [all_axes[0],all_axes[2]]
    site = 'B'

    params_order = ['c5','kp']
    for j,ax in enumerate(axes):
        leg_ax = all_axes[j+1]
        list_plots = []
        param = params_order[j]
        if param != 'c5':
            param_list = params[param][::2]
        else:
            param_list = params['c5']
        norm = mpl.colors.Normalize(vmin=0,vmax=len(param_list)-1)
        cmap = mpl.colormaps.get_cmap('viridis_r')
        for i,value in enumerate(param_list):
            c5 = best[0]
            kp = best[1]
            if param == 'c5':
                c5 = value
            elif param == 'kp':
                kp = value
            date = run_info['long']['date']
            idx = run_info['long']['idx']
            best_setno = result_dict[c5][kp][site]['set_no']
            best_runno = result_dict[c5][kp][site]['run_no']
            ds,_,_ = getds(base_fp + f'{date}_{site}_{idx}/grid_{date}_set{best_setno}_run{best_runno}_0.nc')
            ax = plot_seasonal_mass_balance(ds,plot_ax=ax,label=c5,color=cmap(norm(i)))
            plot_empty, = ax.plot(np.nan,np.nan,color=cmap(norm(i)),label=value)
            list_plots.append(plot_empty)
        ax.get_legend().remove()
        title = '$'+param[0]+'_'+param[1]+'$'
        k = 1 if j == 0 else 3
        all_axes[k].axis('off')
        all_axes[k].legend(handles=list_plots,fontsize=10,title=title) #,bbox_to_anchor=(0.9,0.5))
        ax.set_ylabel('')
    fig.supylabel('Seasonal mass balance (m w.e.)')
    if savefig:
        plt.savefig(base_fp + f'tradeoffs_annual_site{site}.png',dpi=220,bbox_inches='tight')
    plt.show()

def plot_tradeoffs_snowpits(result_dict, best, site='B', years=range(2014,2019),
                            savefig=False):
    fig,axes = plt.subplots(2,6,figsize=(6,4.5),sharex=True,sharey=True,gridspec_kw={'hspace':0.12,'wspace':0.15},width_ratios=[1,1,1,1,1,1.2])
    with open('../MB_data/pits.pkl', 'rb') as file:
        site_profiles = pickle.load(file)
    profiles = site_profiles[site]
    params_order = ['c5','kp']
    for j in range(2):
        list_plots = []
        param = params_order[j]
        param_list = params[param] if param == 'c5' else params[param][::2]

        # Create colormap
        norm = mpl.colors.Normalize(vmin=0,vmax=len(param_list)-1)
        cmap = mpl.colormaps.get_cmap('viridis_r')

        # Create legend
        leg_ax = axes[j,5]
        list_plots = []

        for i,value in enumerate(param_list):
            # Grab color for the value
            color = cmap(norm(i))

            # Load in the datasets for the site
            c5 = best[0]
            kp = best[1]
            if j == 0:
                c5 = value
            else:
                kp = value
            date = run_info['long']['date']
            idx = run_info['long']['idx']
            best_setno = result_dict[c5][kp][site]['set_no']
            best_runno = result_dict[c5][kp][site]['run_no']
            ds,_,_ = getds(base_fp + f'{date}_{site}_{idx}/grid_{date}_set{best_setno}_run{best_runno}_0.nc')

            # Dummy plot for legend
            plot, = leg_ax.plot(np.nan,np.nan,color=color,label=value)
            list_plots.append(plot)
            
            # Loop through years (subplots)
            for y,year in enumerate(years):
                ax = axes[j,y]

                # Some years don't have data: skip
                if year in profiles['sbd']:
                    # Load data
                    sbd = profiles['sbd'][year]
                    dens_meas = profiles['density'][year]

                    # Load dataset on the date the snowpit was taken
                    sample_date = profiles['date'][year]
                    dsyear = ds.sel(time=pd.to_datetime(f'{year}-{sample_date}'))

                    # Calculate layer density and determine snow indices
                    ldz = dsyear.layerheight.values
                    depth_mod = np.array([np.sum(ldz[:i+1])-(ldz[i]/2) for i in range(len(ldz))])
                    dens_mod = dsyear['layerdensity'].values
                    snow_idx = np.where(depth_mod < dsyear.snowdepth.values)[0]

                    # Interpolate modeled density to the snowpit depths
                    dens_interp = np.interp(sbd,depth_mod,dens_mod)
                    diff = dens_interp - dens_meas

                    # Plot the difference
                    ax.plot(diff,profiles['sbd'][year],color=color)

                    # Add year to the axis
                    ax.text(-560,3.7,str(year),fontsize=10)

                    # Find the snow depth
                    snowdepth_mod = dsyear.snowdepth.values
                    snowdepth_pit = sbd[~np.isnan(sbd)][-1]
                    
                    # Plot a bar of the snow depth
                    bar_width = 40
                    ax.bar(600-(i+1)*bar_width,snowdepth_mod,width=bar_width,color=color,align='edge')
                    if i == 0:
                        ax.axhline(snowdepth_pit,color='black',linestyle='--')

        title = '$'+param[0]+'_'+param[1]+'$'
        leg_ax.axis('off')
        leg_ax.legend(handles=list_plots,fontsize=10,title=title,bbox_to_anchor=(1.1,1.1))

    # Format axes
    axes = axes.flatten()
    for a,ax in enumerate(axes):
        if a % 6 != 5:
            ax.invert_yaxis()
            ax.set_ylim(4,0)
            ax.set_yticks([0,1,3])
            ax.tick_params(length=5,labelsize=10)
            ax.set_xlim(-600,600)
            ax.set_xticks([-350,0,350])
            if a % 5 != 0:
                ax.tick_params('y',length=0)
            ax.axvline(0,linewidth=0.5,color='black')
    fig.supxlabel('Difference in modeled density (kg m$^{-3}$)',fontsize=12,y=-0.01)
    fig.supylabel('Depth below surface (m)',fontsize=12,x=0.05)
    # fig.suptitle(f'Gulkana {site} end-of-winter snow density comparison',fontsize=13,y=0.93)
    if savefig:
        plt.savefig(base_fp + f'tradeoffs_snowpits_site{site}.png',dpi=220,bbox_inches='tight')
    plt.show()

def plot_tradeoffs(result_dict, error_names, site='mean',
                   metric='MAE', savefig=False):
    n_plots = len(error_names)
    n_rows = n_plots // 2
    n_rows = n_rows + 1 if n_plots % 2 != 0 else n_rows
    fig,axes = plt.subplots(n_rows, 2, figsize=(6,n_rows*2), sharex=True, layout='constrained')

    # Make colorbar
    param_list = params['c5']
    param_arr = np.array(param_list).astype(float)
    cbar_ax = fig.add_axes([1.05, 0.12, 0.02, 0.8])
    cmap = mpl.colormaps.get_cmap('viridis_r')
    norm =  mpl.colors.Normalize(vmin=min(param_arr),vmax=max(param_arr))
    boundaries = np.append(np.array([param_arr[0] - 0.002]), param_arr)
    labeled_ticks = param_list
    tick_locations = param_arr - 0.001
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=cbar_ax,
                orientation='vertical',
                boundaries=boundaries,ticks=tick_locations,
                spacing='proportional')
    cb.ax.set_yticks(tick_locations)
    cb.ax.tick_params(labelsize=10,direction='inout',length=8)
    # cb.ax.minorticks_on()
    # cb.ax.yaxis.set_minor_locator(mpl.ticker.FixedLocator([0.0145,0.0165]))
    cb.ax.tick_params(which='minor', length=2)
    cb.ax.set_yticklabels(labeled_ticks)
    cb.ax.set_title('$c_5$')
    
    axes = axes.flatten()
    for e,error_name in enumerate(error_names):
        ax = axes[e]
        # Parse run info
        error_name += '_' + metric

        errors = np.empty([len(params['kp']),len(params['c5'])])

        norm = mpl.colors.Normalize(vmin=0,vmax=len(params['c5'])-1)
        cmap = mpl.colormaps.get_cmap('viridis_r')
        for zz,c5 in enumerate(params['c5']):
            for xx,kp in enumerate(params['kp']):
                if error_name in result_dict[c5][kp][site]:
                    errors[xx,zz] = result_dict[c5][kp][site][error_name]
                else:
                    errors[xx,zz] = np.nan
            ax.plot(np.array(params['kp']).astype(float),errors[:,zz],color=cmap(norm(zz)),label=str(c5),marker='o',linewidth=0.7)
            if metric == 'ME':
                ax.axhline(0,color='k',linewidth=0.5)
        if '2024' in error_name:
            ax.legend(title=labels['Boone_c5'],fontsize=12,bbox_to_anchor=(1.5,0.5))
        error_label = errorlabels[error_name.split('_')[0]]
        method_label = methodlabels[metric].replace('B','b')
        title = error_label # .replace('(', method_label+' (').replace('\n',' ')
        if site not in ['mean','median']:
            title += f' at site {site}'
        ax.set_ylabel(title,fontsize=12)
        ax.tick_params(length=5,labelsize=11)
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(4))
        ax.tick_params(which='minor',length=2)
    if e < len(axes) - 1:
        axes[-1].axis('off')
    fig.supxlabel(labels['kp'])

    if savefig:
        plt.savefig(base_fp + f'tradeoffs_{metric}_site{site}.png',bbox_inches='tight',dpi=300)
    plt.show()

def plot_pareto_2024(all_pareto, result_dict, frequency_dict, best, savefig=False):
    site = 'B'
    fig, axes = plt.subplots(2, 2, figsize=(7,4), width_ratios=[1,1], gridspec_kw={'wspace':0.25}, sharex='col')

    param_list = gsproc.params['c5']
    param_arr = np.array(param_list).astype(float)
    cbar_ax = fig.add_axes([0.93, 0.10, 0.02, 0.79])
    cmap = mpl.colormaps.get_cmap('viridis_r')
    norm =  mpl.colors.Normalize(vmin=min(param_arr),vmax=max(param_arr))
    boundaries = np.append(np.array([param_arr[0] - 0.002]), param_arr)
    labeled_ticks = param_list
    tick_locations = param_arr - 0.001
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=cbar_ax,
                orientation='vertical',
                boundaries=boundaries,ticks=tick_locations,
                spacing='proportional')
    cb.ax.set_yticks(tick_locations)
    cb.ax.tick_params(labelsize=10,direction='inout',length=8)
    # cb.ax.minorticks_on()
    # cb.ax.yaxis.set_minor_locator(mpl.ticker.FixedLocator([0.0145,0.0165]))
    cb.ax.tick_params(which='minor', length=2)
    cb.ax.set_yticklabels(labeled_ticks)
    cb.ax.set_title('$c_5$')
    
    ax = axes[0,0]
    all_timeseries = {'dh':[], 'albedo':[]}
    for (c5,kp) in all_pareto:
        dates_dh = result_dict[c5][kp][site]['dates']
        dh_mod = result_dict[c5][kp][site]['dh_mod']
        dates_albedo = result_dict[c5][kp][site]['albedo_dates']
        albedo_mod = result_dict[c5][kp][site]['albedo_mod']
        all_timeseries['dh'].append(dh_mod)
        all_timeseries['albedo'].append(albedo_mod)
    min_series = np.min(np.array(all_timeseries['dh']), axis=0)
    max_series = np.max(np.array(all_timeseries['dh']), axis=0)
    dh_best =  result_dict[best[0]][best[1]][site]['dh_mod']
    dh_data = result_dict[best[0]][best[1]][site]['dh_meas']
    ax.plot(dates_dh, dh_data, color='black',linestyle='--', linewidth=1.2,label='Observations')
    ax.plot(dates_dh, dh_best, color='k', linewidth=1.5, label='Best parameters')
    ax.fill_between(dates_dh, min_series, max_series, alpha=0.8, color='gray', label='Pareto fronts')
    ax.tick_params(labelsize=10,direction='inout',length=8)

    ax.minorticks_on()
    ax.tick_params(which='minor', direction='in', length=4)
    ax.legend()
    ax.set_ylabel('Surface height\nchange (m)', fontsize=12)

    # SURFACE HEIGHT CHANGE TRADEOFFS
    ax = axes[0,1]
    cmap = mpl.colormaps.get_cmap('viridis_r')
    for c5 in frequency_dict:
        kp_list = []
        MAE_list = []
        for kp in frequency_dict[c5]:
            kp_list.append(float(kp))
            MAE_list.append(result_dict[c5][kp][site]['2024_MAE'])

        combined = list(zip(kp_list, MAE_list))
        combined_sorted = sorted(combined, key=lambda x: x[0])
        kp_list, MAE_list = zip(*combined_sorted)
        ax.plot(kp_list, MAE_list,'o-', color=cmap(norm(float(c5))))
    ax.tick_params(length=5)

    # ALBEDO TIMESERIES
    ax = axes[1,0]
    min_series = np.min(np.array(all_timeseries['albedo']), axis=0)
    max_series = np.max(np.array(all_timeseries['albedo']), axis=0)
    albedo_best =  result_dict[best[0]][best[1]][site]['albedo_mod']
    albedo_data = result_dict[best[0]][best[1]][site]['albedo_meas']
    
    # Plot albedo
    ax.plot(dates_albedo, albedo_data, color='k',linestyle='--', linewidth=1.2) #, label='Observations')
    ax.plot(dates_albedo, albedo_best, color='k', linewidth=1.5) #,label='Best parameters')
    ax.fill_between(dates_albedo, min_series, max_series, alpha=0.8, color='gray')
    # Plot ice exposure date
    when_modeled = dates_albedo[np.where(np.abs(albedo_best - np.min(albedo_best)) < 0.05)[0][0]]
    when_measured = pd.to_datetime('2024-07-21')
    ax.axvline(when_modeled, color='r', linewidth=1.5,label='Ice exposed')
    ax.axvline(when_measured, color='r', linestyle='--', linewidth=1.2)
    ax.legend()
    # ax.legend()
    ax.set_xlim(dates_albedo[0],dates_albedo[-1])
    ax.set_ylabel('Albedo (-)', fontsize=12)
    ax.set_xticks(pd.date_range('2024-04-20','2024-08-20',freq='2MS'))
    ax.tick_params(labelsize=10,direction='inout',length=8)
    ax.minorticks_on()
    ax.tick_params(which='minor', direction='in', length=4)
    ax.set_xticklabels(['May 1','Jul 1'],fontsize=11)
    ax.xaxis.set_minor_locator(mpl.dates.MonthLocator(interval=1))

    ax = axes[1,1]
    cmap = mpl.colormaps.get_cmap('viridis_r')
    for c5 in frequency_dict:
        kp_list = []
        MAE_list = []
        for kp in frequency_dict[c5]:
            kp_list.append(float(kp))
            MAE_list.append(result_dict[c5][kp][site]['albedo_MAE'])

        combined = list(zip(kp_list, MAE_list))
        combined_sorted = sorted(combined, key=lambda x: x[0])
        kp_list, MAE_list = zip(*combined_sorted)
        ax.plot(kp_list, MAE_list,'o-', color=cmap(norm(float(c5))))
    ax.set_xlim(1.75,3)
    ax.set_xticks([1.75,2,2.25,2.5,2.75,3])
    ax.tick_params(length=5)
    ax.set_xlabel('Precipitation factor, $k_p$',fontsize=12)
    axes[0,1].set_title('MAE',fontsize=12)
    axes[0,0].set_title('Timeseries',fontsize=12)
    axes = axes.flatten()
    for l, letter in enumerate(['a','b','c','d']):
        ax = axes[l]
        x = 0.95
        y = 0.9
        box_width = 0.1
        box_height = 0.2
                # bbox={'facecolor': 'white', 'edgecolor': 'black'})
        rect = plt.Rectangle((x - box_width / 2, y - box_height / 2), box_width, box_height,
                        linewidth=1.5, edgecolor='k', facecolor='white', zorder=5,transform=ax.transAxes,)
        ax.add_patch(rect)
        ax.text(x, y, letter,transform=ax.transAxes,fontsize=12,zorder=6,ha='center', va='center')
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        rect = plt.Rectangle((xlim[0], ylim[0]), xlim[1]-xlim[0], ylim[1]-ylim[0],
                        linewidth=1.5, edgecolor='k', facecolor='none', zorder=7)
        ax.add_patch(rect)

    # fig.suptitle('Validation of Pareto fronts on the 2024 melt season',fontsize=12,y=1)
    if savefig:
        plt.savefig(base_fp+'2024_pareto_comparison.png',bbox_inches='tight',dpi=300)
    plt.show()

def plot_best_seasonal(best, result_dict, savefig=False, include_B=False):
    fig, axes = plt.subplots(2,figsize=(8,4),gridspec_kw={'hspace':0.1},sharex=True)
    list_plots = []
    sites = ['A','AU','B','D']
    colors = np.flip(['#F0E442','#E69F00','#009E73','#0072B2']) #
    if not include_B:
        sites.remove('B')
        colors = colors[[0,1,3]]


    for ss,site in enumerate(sites):   
        color = colors[ss]
        years = result_dict[best[0]][best[1]][site]['years']
        wmod = result_dict[best[0]][best[1]][site]['winter_mod']
        wmeas = result_dict[best[0]][best[1]][site]['winter_meas']
        smod = result_dict[best[0]][best[1]][site]['summer_mod']
        smeas = result_dict[best[0]][best[1]][site]['summer_meas']
        axes[0].plot(years, wmeas, color=color, linestyle='--')
        axes[0].plot(years, wmod, color=color)
        axes[1].plot(years, smeas, color=color, linestyle='--')
        axes[1].plot(years, smod, color=color)
        dummy_site, = axes[1].plot(np.nan,np.nan,color=color)
        list_plots.append(dummy_site)
    dummy_model, = axes[1].plot(np.nan,np.nan,color='gray')
    dummy_meas, = axes[1].plot(np.nan,np.nan,color='gray',linestyle='--')
    list_plots.append(dummy_model)
    list_plots.append(dummy_meas)
    axes[0].set_ylabel('Winter',fontsize=12)
    axes[1].set_ylabel('Summer',fontsize=12)
    axes[0].set_ylim(-0.5,2.5)
    axes[1].set_ylim(-7,1)
    axes[1].set_yticks([0,-2,-4,-6])
    axes[0].set_xlim(2001,2024)
    axes[1].set_xlim(2001,2024)

    labels = ['Site '+sss for sss in sites]+['Modeled','Measured']
    fig.legend(list_plots,labels,bbox_to_anchor=(1.07,0.7),fontsize=10)
    fig.suptitle('Seasonal mass balance (m w.e.)',fontsize=12,y=0.96)
    if savefig:
        plt.savefig(base_fp + f'best_seasonal_mb.png',bbox_inches='tight',dpi=300)
    plt.show()

def plot_best_2024(best, result_dict, savefig=False):
    elev = {'AB':1542,'ABB':1608,'B':1682,'BD':1742,'D':1843,'T':1877}
    ylim = (-5,1)
    fig,axes = plt.subplots(1,6,figsize=(8,4),sharey=True,sharex=True,gridspec_kw={'wspace':0.2})
    c5 = best[0]
    kp = best[1]
    
    for i,site in enumerate(sitedict['2024']):
        dates = pd.to_datetime(result_dict[c5][kp][site]['dates'])
        axes[i].plot(dates, result_dict[c5][kp][site]['dh_meas'],label='Stake',linestyle='--',color='k')
        axes[i].plot(dates, result_dict[c5][kp][site]['dh_mod'],label='Model',color=all_colors[0])
        axes[i].set_title(f'Site {site}\n{str(elev[site])} m a.s.l.',fontsize=12,y=1.02)
        # axes[i].text(pd.to_datetime('2024-04-20'),,,fontsize=10,transform=axes[i].transAxes)

        # direction = '' if error < 0 else '+'
        # text = f'{direction}{error:.3f} m'
        # axes[i].text(enddate-pd.Timedelta(days=80),0.9,text,fontsize=10)
        # print(dates, result_dict[c5][kp][site]['dh_mod'])
        axes[i].set_xlim(dates[0],dates[-1])
        s = pd.to_datetime('2024-04-20')
        e = pd.to_datetime('2024-08-20')
        ticks = pd.date_range(s, e,freq='2MS')
        axes[i].set_xticks(ticks)
        axes[i].tick_params(labelsize=10,direction='inout',length=8)
        axes[i].minorticks_on()
        axes[i].tick_params(which='minor', direction='in', length=4)
        axes[i].set_xticklabels([f"{tick.strftime('%b')} {tick.day}" for tick in ticks])
        axes[i].xaxis.set_minor_locator(mpl.dates.MonthLocator(interval=1))

        twinax = axes[i].twinx()
        if site not in ['ABB','BD']:
            mod = twinax.scatter(e,result_dict[c5][kp][site]['mb2024_mod'],color='red',marker='x',s=100)
            meas = twinax.scatter(e,result_dict[c5][kp][site]['mb2024_meas'],color='red',marker='o',facecolors='none',s=100)
            meas.set_clip_on(False)
            mod.set_clip_on(False)
        twinax.set_ylim(ylim)
        twinax.yaxis.set_ticks_position('right')
        for s,spine in enumerate(twinax.spines.values()):
            if s == 1:
                spine.set_edgecolor('red')
        if site=='T':
            twinax.tick_params(labelright=True,labelsize=10,direction='inout',length=8,labelcolor='red',colors='red')
            twinax.set_ylabel('Summer mass balance (m w.e.)',fontsize=12,color='red')
        else:
            twinax.set_yticklabels([])
    for ax in axes:
        ax.set_xlim(pd.to_datetime('2024-04-20'),pd.to_datetime('2024-08-20'))
        ax.set_ylim((ylim))
    axes[0].set_ylabel('Surface height change (m)',fontsize=12)
    l1, = axes[-1].plot(np.nan,np.nan,color=all_colors[0])
    l2, = axes[-1].plot(np.nan,np.nan,color='black',linestyle='--')
    # l3, = axes[-1].plot(np.nan,np.nan,color='gray',linestyle=':')
    leg = fig.legend([l1,l2,mod,meas],['Model','GNSS-IR','Model','Stake survey'],ncols=2,fontsize=10,bbox_to_anchor=(0.7,0.05))
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_alpha(1)
    if savefig:
        plt.savefig(base_fp + f'best_2024_mb.png',bbox_inches='tight',dpi=300)
    plt.show()

def plot_best_2024_dh(best, result_dict, savefig=False):
    elev = {'AB':1542,'ABB':1608,'B':1682,'BD':1742,'D':1843,'T':1877}
    ylim = (-5,1)
    fig,axes = plt.subplots(1,6,figsize=(8,4),sharey=True,sharex=True,gridspec_kw={'wspace':0.2})
    c5 = best[0]
    kp = best[1]
    s = pd.to_datetime('2024-04-20')
    e = pd.to_datetime('2024-08-20')
    for i,site in enumerate(sitedict['2024']):
        # ds,s,e = getds(base_fp + f'{date}_{site}_{idx}/grid_{date}_set{best_setno}_run{best_runno}_0.nc')
        # axes[i] = plot_2024_mass_balance(ds,plot_ax=axes[i])
        dates = result_dict[c5][kp][site]['dates']
        axes[i].plot(dates, result_dict[c5][kp][site]['dh_meas'],label='Stake',linestyle='--',color='k')
        axes[i].plot(dates, result_dict[c5][kp][site]['dh_mod'],label='Model',color=all_colors[0])
        axes[i].set_title(f'Site {site}',fontsize=12,y=1.05)
        axes[i].text(pd.to_datetime('2024-04-20'),1.15,str(elev[site])+' m a.s.l.',fontsize=10)
        
        axes[i].set_xlim(s,e)
        axes[i].set_xticks(pd.date_range(s,e,freq='2MS'))
        axes[i].tick_params(labelsize=10,direction='inout',length=8)
        axes[i].minorticks_on()
        axes[i].tick_params(which='minor', direction='in', length=4)
        axes[i].set_xticklabels(['May 1','Jul 1'])
        axes[i].xaxis.set_minor_locator(mpl.dates.MonthLocator(interval=1))
    for ax in axes:
        ax.set_xlim(pd.to_datetime('2024-04-20'),pd.to_datetime('2024-08-20'))
        ax.set_ylim((ylim))
    axes[0].set_ylabel('Surface height change (m)',fontsize=12)
    l1, = axes[-1].plot(np.nan,np.nan,color=all_colors[0])
    l2, = axes[-1].plot(np.nan,np.nan,color='black',linestyle='--')
    # l3, = axes[-1].plot(np.nan,np.nan,color='gray',linestyle=':')
    leg = fig.legend([l1,l2],['Model','GNSS-IR'],ncols=3,fontsize=10,bbox_to_anchor=(0.7,0.05))
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_alpha(1)
    if savefig:
        plt.savefig(base_fp + f'best_2024_dh.png',bbox_inches='tight',dpi=300)
    plt.show()

def plot_pareto_2024_dh(result_dict, all_pareto, best, savefig=False):
    elev = {'AB':1542,'ABB':1608,'B':1682,'BD':1742,'D':1843,'T':1877}
    ylim = (-5,1)
    fig,all_axes = plt.subplots(2,5,figsize=(8,4),sharey='row',sharex='row',gridspec_kw={'wspace':0.2,'hspace':0.6},height_ratios=[1,0.3])
    start = pd.to_datetime('2024-04-20')
    end = pd.to_datetime('2024-08-20')
    axes = all_axes[0, :]
    axes_2 = all_axes[1, :]
    for s,site in enumerate(sitedict['2024'][1:]):
        ax = axes[s]
        dh_meas = result_dict[best[0]][best[1]][site]['dh_meas']
        all_timeseries = {'dh':[], 'mb':[], 'r2':[]}
        for (c5,kp) in all_pareto:
            dates_dh = result_dict[c5][kp][site]['dates']
            dh_mod = result_dict[c5][kp][site]['dh_mod']
            if site not in ['ABB','BD']:
                mb_mod = result_dict[c5][kp][site]['mb2024_mod']
                all_timeseries['mb'].append(mb_mod)
            ss_tot = np.sum((dh_meas - np.mean(dh_meas)) ** 2)
            ss_res = np.sum((dh_meas - dh_mod) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            all_timeseries['r2'].append(r_squared)
            all_timeseries['dh'].append(dh_mod)
        min_series = np.min(np.array(all_timeseries['dh']), axis=0)
        max_series = np.max(np.array(all_timeseries['dh']), axis=0)
        dh_best =  result_dict[best[0]][best[1]][site]['dh_mod']
        ax.plot(dates_dh, dh_meas, color='black',linestyle='--', linewidth=1.2,label='Observations')
        ax.plot(dates_dh, dh_best, color='k', linewidth=1.5, label='Best parameters')
        ax.fill_between(dates_dh, min_series, max_series, alpha=0.8, color='gray', label='Pareto fronts')
        ax.tick_params(labelsize=10,direction='inout',length=8)
        ax.set_xlim(start, end)
        ax.set_title(site+'\n'+str(elev[site])+' m a.s.l.', y=1.01)
        ax.set_ylim(-4,1)
        ax.set_xticks(pd.date_range(start, end, freq='2MS'))
        ax.tick_params(labelsize=10,direction='inout',length=8)
        ax.minorticks_on()
        ax.tick_params(which='minor', direction='in', length=4)
        ax.set_xticklabels(['May 1','Jul 1'])
        ax.xaxis.set_minor_locator(mpl.dates.MonthLocator(interval=1))

        ax = axes_2[s]
        ax.boxplot(all_timeseries['r2'],medianprops={'color':all_colors[4]})
        ax.set_ylim(0,1)
        # if site not in ['ABB','BD']:
        #     mb_meas = result_dict[c5][kp][site]['mb2024_meas']
        #     print(site,mb_meas,all_pareto[np.argmin(all_timeseries['mb'])])
        #     ax.set_xlim(2,4)
        #     ax.set_ylim(-3,0)
        #     ax.set_yticks([-4,-2,0])
        #     ax.tick_params(length=5)
        #     ax.fill_between([0,5], np.min(all_timeseries['mb']),  np.max(all_timeseries['mb']), alpha=0.8, color='gray', label='Pareto fronts')
        #     # ax.boxplot(all_timeseries['mb'], positions=[3], widths=[1], notch=True, 
        #     #                    boxprops=dict(facecolor='gray',alpha=0.8),patch_artist=True)
        #     ax.axhline(mb_meas,0,5, color='black',linestyle='--', linewidth=1.2) 
        #     ax.axhline(result_dict[best[0]][best[1]][site]['mb2024_mod'],0,5, color='black',linewidth=1.5) 

    for ax in axes_2:
        ax.set_xticks([])
        ax.set_xticklabels([])
        ax.tick_params(length=5)
        ax.minorticks_on()
        ax.tick_params(which='minor', length=2,direction='in')
    fig.suptitle('Site name',y=1.04)
    axes[0].set_ylabel('Surface height change (m)',fontsize=12)
    l1, = axes[-1].plot(np.nan,np.nan,color='black')
    l2, = axes[-1].plot(np.nan,np.nan,color='black',linestyle='--')
    l3 = axes[-1].fill_between([start,end], np.nan, np.nan, alpha=0.8, color='gray')
    leg = fig.legend([l2,l1,l3],['Observations','Best parameters','Pareto fronts'],ncols=3,fontsize=12,bbox_to_anchor=(0.85,0.37))
    leg.get_frame().set_facecolor('white')
    leg.get_frame().set_alpha(1)

    axes_2[0].set_ylabel('$R^2$',fontsize=12)

    if savefig:
        plt.savefig(base_fp + f'pareto_2024_dh.png',bbox_inches='tight',dpi=300)
    plt.show()

def plot_pareto_2024_mb(result_dict, all_pareto, best, savefig=False):
    elev = {'AB':1542,'B':1682,'D':1843,'T':1877}
    fig,ax = plt.subplots(figsize=(6,4),sharey=True)
    aaa = 0
    for s,site in enumerate(elev):
        color = all_colors[s]
        all_timeseries = {'mb':[]}
        for (c5,kp) in all_pareto:
            mb_mod = result_dict[c5][kp][site]['mb2024_mod']
            all_timeseries['mb'].append(mb_mod)
        min_series = np.min(np.array(all_timeseries['mb']))
        max_series = np.max(np.array(all_timeseries['mb']))
        mb_best =  result_dict[best[0]][best[1]][site]['mb2024_mod']
        mb_meas =  result_dict[best[0]][best[1]][site]['mb2024_meas']
        ax.bar(s*2 + aaa,mb_meas, color=color, edgecolor='black', linestyle='--',linewidth=1.2,width=1,hatch='///')
        yerr = [[mb_best - min_series],[max_series - mb_best]]
        ax.bar(1+s*2 + aaa,mb_best, color=color, edgecolor='black',linewidth=1.5,yerr=yerr,ecolor='k',
               error_kw={'elinewidth':1,'capsize':6,'capthick':2,'alpha':0.8},width=1)
        aaa += 1

        ax.tick_params(labelsize=10,length=5)
        ax.minorticks_on()
        ax.tick_params(which='minor', length=2)
        ax.set_xticks([0.5,3.5,6.5,9.5])
        ax.set_xlim([-1,11])
        ax.set_xticklabels([k+'\n'+str(elev[k])+' m a.s.l.' for k in elev],fontsize=12)
        ax.xaxis.set_minor_locator(plt.NullLocator())

    b1 = ax.bar(-5,0,hatch='///',color='none',edgecolor='black',linestyle='--',linewidth=1.2,label='Observations')
    b2 = ax.bar(-5,0,color='none',edgecolor='black',linewidth=1.5,label='Best parameters')
    b3 = mpl.lines.Line2D([0],[0],color='black',linewidth=1.5,linestyle='',marker='_',markeredgewidth=2,markersize=10,label='Pareto fronts')
    ax.legend(handles=[b1,b2,b3],fontsize=12)

    fig.supxlabel('Site name',y=-0.05)
    ax.set_ylabel('2024 summer mass balance (m w.e.)',fontsize=12)
    print(mpl.rcParams['font.family'])

    if savefig:
        plt.savefig(base_fp + f'pareto_2024_mb.png',bbox_inches='tight',dpi=300)
    plt.show()

def plot_best_snowpits(best, result_dict, years=range(2014,2019), 
                        plot_diff=True, savefig=False):
    with open('../MB_data/pits.pkl', 'rb') as file:
        site_profiles = pickle.load(file)
    colors = np.flip(['#F0E442','#E69F00','#009E73','#0072B2']) # 
    fig,axes = plt.subplots(1,6,figsize=(8,3),sharex=True,sharey=True,
                            gridspec_kw={'hspace':0.12,'wspace':0.15}, width_ratios=[1,1,1,1,1,2])
    axes = axes.flatten()
    date = run_info['long']['date']
    idx = run_info['long']['idx']
    best_setno = result_dict[best[0]][best[1]]['B']['set_no']
    best_runno = result_dict[best[0]][best[1]]['B']['run_no']

    for ss,site in enumerate(sitedict['long']):   
        ds,_,_ = getds(base_fp + f'{date}_{site}_{idx}/grid_{date}_set{best_setno}_run{best_runno}_0.nc')
        color = colors[ss]

        # Load data for the site
        profiles = site_profiles[site]

        # Loop through years
        for y,year in enumerate(years):
            ax = axes[y]

            # Some years don't have data: skip
            if year in profiles['sbd']:
                # Load data
                sbd = profiles['sbd'][year]
                dens_meas = profiles['density'][year]

                # Load dataset on the date the snowpit was taken
                sample_date = profiles['date'][year]
                dsyear = ds.sel(time=pd.to_datetime(f'{year}-{sample_date}'))

                # Add year to the axis
                coords = [-240,4.9] if plot_diff else [50,4.9]
                ax.text(coords[0],coords[1],str(year),fontsize=10)

                # Calculate layer density and determine snow indices
                ldz = dsyear.layerheight.values
                depth_mod = np.array([np.sum(ldz[:i+1])-(ldz[i]/2) for i in range(len(ldz))])
                dens_mod = dsyear['layerdensity'].values
                snow_idx = np.where(depth_mod < dsyear.snowdepth.values)[0]

                if plot_diff:
                    # Interpolate modeled density to the snowpit depths
                    dens_interp = np.interp(sbd,depth_mod,dens_mod)
                    diff = dens_interp - dens_meas

                    # Plot the difference
                    ax.plot(diff,profiles['sbd'][year],color=color)
                else:
                    # Plot the profiles
                    ax.plot(dens_meas,sbd,color=color,linestyle='--')
                    ax.plot(dens_mod[snow_idx],depth_mod[snow_idx],color=color)

                # Find the snow depth
                snowdepth_mod = depth_mod[snow_idx[-1]]
                snowdepth_pit = sbd[~np.isnan(sbd)][-1]

                # Plot a horizontal line for the snow depth
                # ax.axhline(snowdepth_mod,color=color)
                ax.axhline(snowdepth_pit,color=color,linestyle='--')            

                if plot_diff:
                    # Plot a bar of the snow depth
                    bar_width = 40
                    ax.bar(300-(ss+1)*bar_width,snowdepth_mod,width=bar_width,color=color,align='edge')
                    ax.axhline(snowdepth_pit,color=color,linestyle='--')

        # Dummy variable for legend
        axes[5].plot(np.nan,np.nan,label=f'Site {site}',color=color)

    # Dummy variables for legend
    if plot_diff:
        axes[5].plot(np.nan,np.nan,label='Difference\nfrom measured\ndensity',color='grey')
        axes[5].plot(np.nan,np.nan,color='black',linestyle='--',label='Measured\nsnow depth')
        axes[5].bar(np.nan,np.nan,color='gray',label='Modeled\nsnow depth')
    else:
        axes[5].plot(np.nan,np.nan,label='Modeled',color='grey')
        axes[5].plot(np.nan,np.nan,label='Measured',color='gray',linestyle='--')

    # Format axes
    for a,ax in enumerate(axes[:-1]):
        ax.invert_yaxis()
        ax.set_ylim(5,0)
        ax.set_yticks([0,2,4])
        ax.tick_params(length=5,labelsize=10)
        if plot_diff:
            ax.axvline(0,color='k',linewidth=0.5)
            ax.set_xlim(-300,300)
            ax.set_xticks([-200,0,200])
            fig.supxlabel('Difference in density (kg m$^{-3}$)\n(Modeled $-$ Measured)',fontsize=12,y=-0.12)
        else:
            ax.set_xlim(0,600)
            ax.set_xticks([0,250,500])
            fig.supxlabel('Density (kg m$^{-3}$)',fontsize=12,y=0.04)
        if a % 5 != 0:
            ax.tick_params('y',length=0)
        ax.axvline(0,linewidth=0.5,color='black')
    fig.supylabel('Depth below surface (m)',fontsize=12,x=0.05)
    axes[5].axis('off')
    axes[5].legend()
    # fig.suptitle(f'Gulkana end-of-winter snow density comparison',fontsize=13,y=1)
    if savefig:
        plt.savefig(base_fp+'best_density_pits.png',dpi=180,bbox_inches='tight')
    plt.show()

def plot_best_snowmass(best, result_dict, years=range(2015,2020), sites=['AU','B','D'],
                       savefig=False):
    with open('../MB_data/pits.pkl', 'rb') as file:
        site_profiles = pickle.load(file)
    fig,ax = plt.subplots(figsize=(6,3))
    colors = np.flip(['#F0E442','#E69F00','#009E73','#0072B2']) # 
    for ss,site in enumerate(sites):
        color = colors[ss+1]
        prof = site_profiles[site]
        snowmass_meas_list = []
        snowmass_mod_list = []
        date = run_info['long']['date']
        idx = run_info['long']['idx']
        best_setno = result_dict[best[0]][best[1]]['B']['set_no']
        best_runno = result_dict[best[0]][best[1]]['B']['run_no']

        ds,_,_ = getds(base_fp + f'{date}_{site}_{idx}/grid_{date}_set{best_setno}_run{best_runno}_0.nc')

        for year in years:
            year = int(year)
            if year in prof['sbd']:
                sample_heights = np.append(np.array([prof['sbd'][year][0]]),np.diff(np.array(prof['sbd'][year])))
                snowmass_meas = np.sum(prof['density'][year] * sample_heights) / 1000
                snowmass_meas_list.append(snowmass_meas)
            else:
                snowmass_meas_list.append(np.nan)
            
            # Load dataset on the date the snowpit was taken
            if year in prof['date']:
                sample_date = prof['date'][year]
            else:
                sample_date = '04-20'
            dsyear = ds.sel(time=pd.to_datetime(f'{year}-{sample_date}'))

            # Calculate layer density and determine snow indices
            ldz = dsyear.layerheight.values
            depth_mod = np.array([np.sum(ldz[:i+1])-(ldz[i]/2) for i in range(len(ldz))])
            snow_idx = np.where(depth_mod < dsyear.snowdepth.values)[0]

            snowmass_mod = np.sum(dsyear.layerheight.values[snow_idx] * dsyear.layerdensity.values[snow_idx]) / 1000
            snowmass_mod_list.append(snowmass_mod)
        ax.plot(years,snowmass_mod_list,color=color)
        ax.plot(years,snowmass_meas_list,color=color,linestyle='--')
    ax.tick_params(length=5,labelsize=10)
    ax.set_xlim(years[0],years[-1])
    ax.set_xticks(np.array(years,dtype=int))
    ax.set_ylabel('Mass of snow (m w.e.)')
    ax.set_title('Spring field date mass of snow (integrated snow pits)')
    if savefig:
        plt.savefig(base_fp+'best_snow_mass.png',dpi=180,bbox_inches='tight')
    plt.show()

def plot_best_albedo(best, result_dict, albedo_type='albedo', savefig=False):
    df = pd.read_csv('/trace/home/cvwilson/research/climate_data/AWS/Preprocessed/gulkana2024_bothalbedo.csv',index_col=0)
    df.index = pd.to_datetime(df.index) # - pd.Timedelta(hours=8)
    date = run_info['2024']['date']
    idx = run_info['2024']['idx']
    best_setno = result_dict[best[0]][best[1]]['B']['set_no']
    best_runno = result_dict[best[0]][best[1]]['B']['run_no']
    bds,s,e = getds(base_fp + f'{date}_B_{idx}/grid_{date}_set{best_setno}_run{best_runno}_0.nc')
    bds = bds.resample(time='d').mean()
    daily_albedo = []
    ice_albedo = []
    dates = pd.date_range('2024-04-20','2024-08-20',freq='d')
    dates_ice = np.concatenate([pd.date_range('2024-07-20','2024-08-10'),pd.date_range('2024-08-18','2024-08-22')])
    for date in dates:
        start = pd.to_datetime(str(date.date())+' 12:00')
        end = pd.to_datetime(str(date.date())+' 16:00')
        # print(df.loc[start:end,'albedo'].values)
        daily_albedo.append(np.mean(df.loc[start:end,albedo_type.replace('_','')]))
        if date in dates_ice:
            ice_albedo.append(np.mean(df.loc[start:end,albedo_type.replace('_','')]))
    y = np.arange(0,1.1,0.1)
    fig,ax = plt.subplots(figsize=(6.5,3))
    ax.fill_betweenx(y,[pd.to_datetime('2024-08-10')],[pd.to_datetime('2024-07-20')],color='grey',alpha=0.3,label='Ice exposed (time lapse camera)')
    ax.fill_betweenx(y,[pd.to_datetime('2024-08-18')],[pd.to_datetime('2024-08-22')],color='grey',alpha=0.3)
    if albedo_type == 'albedo':
        ax.scatter(pd.to_datetime('2024-06-19'),0.7123,marker='x',color='black',label='FieldSpec sample')
    else:
        ax.scatter(pd.to_datetime('2024-06-19'),0.7588,marker='x',color='black',label='FieldSpec sample')
    measured_label = 'Measured albedo' if albedo_type == 'albedo' else 'Measured visible albedo'
    modeled_label = 'Modeled albedo' if albedo_type == 'albedo' else 'Modeled visible albedo'
    ax.plot(dates,daily_albedo,label=measured_label,color='black',linestyle='--')
    ax.plot(bds.time,bds[albedo_type],label=modeled_label,color=plt.cm.Dark2(0))

    ax.set_xlim(dates[0],dates[-1])
    ax.set_ylim(0,1)
    ax.tick_params(which='major',labelsize=12,length=8,direction='inout')
    ax.minorticks_on()
    ax.tick_params(axis='x',which='minor',bottom=False)
    ax.tick_params(axis='y',which='minor',length=3,direction='in')
    ax.legend(fontsize=10)
    ylabel = 'Broadband albedo' if albedo_type == 'albedo' else 'Visible (400-750nm) albedo'
    ax.set_ylabel(ylabel,fontsize=13)
    ax.set_xticks(pd.date_range('2024-04-20',e,freq='MS'))
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%b %d'))
    ax.set_xticklabels(['May 1','Jun 1','Jul 1','Aug 1'])
    # ax.set_title(f'Mean ice albedo: {np.mean(ice_albedo):.3f}',fontsize=12)
    # fig.suptitle('Daily albedo (mean of 12:00 - 16:00) at Gulkana B, summer 2024',fontsize=13,y=1.03)
    if savefig:
        plt.savefig(base_fp+'best_albedo.png',dpi=300,bbox_inches='tight')
    plt.show()

def plot_best_by_site(error_list, result_dict):
    fig, axes = plt.subplots(1, len(error_list), figsize=(8,4), gridspec_kw={'hspace':0.5, 'wspace':0.5})
    axes = axes.flatten()
    bars = []
    for site in ['A','AU','B','D','mean']: # sitedict['long']):
        best_by = gsproc.get_best_normalized(error_list[:-1], result_dict, site=site, prints=False)
        c5,kp = best_by[-1]
        bars.append([])

        for error in error_list:
            bars[-1].append(result_dict[c5][kp]['mean'][error+'_MAE'])
    bars = np.array(bars).T
    colors = np.flip(['gray','#F0E442','#E69F00','#009E73','#0072B2']) #
    for a, ax in enumerate(axes):
        ax.bar(range(5), bars[a], color=colors)
        ax.set_xticks(range(5))
        ax.set_xticklabels(['A','AU','B','D','mean'], rotation=45)
        ax.set_title(error_list[a]+'_'+units[error_list[a]])
        ax.tick_params(axis='y', length=5)
    # fig.suptitle('MAE for each error metric calibrating on one site')
    plt.show()

def plot_heatmap_by_site(error_names, result_dict, metric='MAE', savefig=False):
    sites_all = ['A','AU','B','D','mean']
    weights = np.ones(len(error_names) - 1)*1/(len(error_names)-1)
    n_plots = len(error_names)
    n_rows = n_plots // 2
    n_rows = n_rows + 1 if n_plots % 2 != 0 else n_rows
    fig,axes = plt.subplots(n_rows, 2)
    axes = axes.flatten()
    for ax in axes:
        ax.set_xticks(np.arange(0.5,5.5))
        ax.set_yticks(np.arange(0.5,5.5))
        ax.set_xticklabels(sites_all)
        ax.set_yticklabels(sites_all)
        ax.tick_params(length=5)

    if 'AE' in metric:
        cmaps = ['Greens','Purples','Blues','Reds','Greys','Oranges']
    else:
        cmaps = diverging_cmaps
        cmaps[4] = 'Greys'

    errors_all = {}
    for error in error_names:
        errors_all[error] = np.empty((5,5))
    for s1,site1 in enumerate(sites_all):
        all_bootstrap_pareto = gsproc.get_pareto_fronts_bootstrap(1000, result_dict, error_names, site=site1)
        _, best = gsproc.get_frequency(all_bootstrap_pareto)
        c5, kp = best
        c5 = str(c5)
        kp = str(kp).replace('.0','')
        for e,error in enumerate(error_names):
            for s2,site2 in enumerate(sites_all):
                errors_all[error][s2,s1] = result_dict[c5][kp][site2][error+'_MAE']
    
    for e, error in enumerate(error_names):
        errors = errors_all[error]
        ax = axes[e]
        max_value = np.max(errors)
        min_value = np.min(errors)
        if metric == 'ME' and 'weighted' not in error:
            greater = max(np.abs(max_value), np.abs(min_value))
            max_value, min_value = (greater, -greater)
        grid = np.arange(6)
        c = ax.pcolormesh(grid, grid, errors, cmap=cmaps[e], 
                        vmin=min_value, vmax=max_value)
        cbar = plt.colorbar(c, ax=ax)
        cbar.ax.tick_params(length=5)
        axtitle = errorlabels[error.split('_')[0]]
        lp = 10 if 'weight' in error else 27
        ax.set_title(axtitle,fontsize=10,labelpad=lp)
        if e in [0,1]:
            ax.tick_params(labelbottom=False)
        if e in [1,3]:
            ax.tick_params(labelleft=False)
    
    fig.supxlabel('Site used for calibration',y=-0.01)
    fig.supylabel('Site MAE',x=0.03)
    # fig.suptitle(f'{methodlabels[metric]} for each error metric calibrating on one site')
    plt.subplots_adjust(hspace=0.6, wspace=0.2)
    if savefig:
        plt.savefig(base_fp+f'calibration_by_site_heatmap_{metric}.png',dpi=180,bbox_inches='tight')
    plt.show()

def plot_heatmap_by_site_weighted(error_names, result_dict, metric='MAE', savefig=False):
    sites_all = ['A','AU','B','D','mean']
    weights = np.ones(len(error_names) - 1)*1/(len(error_names)-1)
    n_plots = len(error_names)
    n_rows = n_plots // 2
    n_rows = n_rows + 1 if n_plots % 2 != 0 else n_rows
    fig = plt.figure(figsize=(8,n_rows*1.3)) # , gridspec_kw={'hspace':0.5, 'wspace':0.5}
    gs = mpl.gridspec.GridSpec(4,3, figure=fig, wspace=0.6)  # Increase wspace here
    ax1 = fig.add_subplot(gs[0:2, 0])  # top-left
    ax2 = fig.add_subplot(gs[0:2, 1])  # top-right
    ax3 = fig.add_subplot(gs[2:4, 0])  # bottom-left
    ax4 = fig.add_subplot(gs[2:4, 1])  # bottom-right
    ax5 = fig.add_subplot(gs[1:3, 2])  # centered vertically
    axes = [ax1,ax2,ax3,ax4,ax5]
    for ax in axes:
        ax.set_xticks(np.arange(0.5,5.5))
        ax.set_yticks(np.arange(0.5,5.5))
        ax.set_xticklabels(sites_all)
        ax.set_yticklabels(sites_all)
        ax.tick_params(length=5)

    if 'AE' in metric:
        cmaps = ['Greens','Purples','Blues','Reds','Greys','Oranges']
    else:
        cmaps = diverging_cmaps
        cmaps[4] = 'Greys'

    for e,error in enumerate(error_names):
        ax = axes[e]
        error += '_' + metric
        errors = np.empty((5,5))
        for s1,site1 in enumerate(sites_all): # sitedict['long']):
            best_by = gsproc.get_best_normalized(error_names[:-1], result_dict, site=site1, prints=False)
            c5,kp = best_by[-1]
            c5 = str(c5)
            kp = str(kp).replace('.0','')
            
            for s2,site2 in enumerate(sites_all):
                if 'weighted' not in error:
                    error_value = result_dict[c5][kp][site2][error]
                else:
                    errors_to_weight = []
                    for each in error_names[:-1]:
                        each += '_MAE'
                        errors_to_weight.append(result_dict[c5][kp][site2][each+'_norm'])
                    error_value = np.sum(errors_to_weight * np.array(weights))
                    if site1 == 'mean':
                        print(site2, error_value)
                errors[s2,s1] = error_value

            max_value = np.max(errors)
            min_value = np.min(errors)
            # if 'weighted' in error:
                # max_value /= 2
            if metric == 'ME' and 'weighted' not in error:
                greater = max(np.abs(max_value), np.abs(min_value))
                max_value, min_value = (greater, -greater)
        if e == 4:
            print(np.mean(errors[:4, 2]), np.mean(errors[:4, 1]))
        grid = np.arange(6)
        c = ax.pcolormesh(grid, grid, errors, cmap=cmaps[e], 
                        vmin=min_value, vmax=max_value)
        cbar = plt.colorbar(c, ax=ax)
        cbar.ax.tick_params(length=5)
        axtitle = errorlabels[error.split('_')[0]]
        # if 'snow' in axtitle:
        #     axtitle = axtitle.replace('snow','\nsnow')
        # if 'balance' in axtitle:
        #     axtitle = axtitle.replace('balance','\nbalance')
        lp = 15 if 'weight' in error else 27
        cbar.ax.set_ylabel(axtitle,rotation=270, labelpad=lp, fontsize=12)
        cbar.ax.yaxis.set_label_position('right')
        if e in [0,1]:
            ax.tick_params(labelbottom=False)
        if e in [1,3,4]:
            ax.tick_params(labelleft=False)
    
    for i in range(5):
        rect = mpl.patches.Rectangle((i,i), 1, 1, linewidth=1.5, edgecolor='k',linestyle='--', facecolor='none')
        ax.add_patch(rect)
    fig.supxlabel('Site used for calibration',y=-0.01)
    fig.supylabel('Site MAE',x=0.03)
    # fig.suptitle(f'{methodlabels[metric]} for each error metric calibrating on one site')
    plt.subplots_adjust(hspace=0.6, wspace=0.2)
    if savefig:
        plt.savefig(base_fp+f'calibration_by_site_heatmap_{metric}.png',dpi=180,bbox_inches='tight')
    plt.show()

def compare_calib_valid(error_list, all_calib, all_valid, savefig=False):
    n_plots = len(error_list)
    n_rows = n_plots // 2
    n_rows = n_rows + 1 if n_plots % 2 != 0 else n_rows
    fig,axes = plt.subplots(n_rows, 2, figsize=(6,n_rows*2),gridspec_kw={'hspace':0.6, 'wspace':0.3})
    axes = axes.flatten()
    for e, error in enumerate(error_list):
        ax = axes[e]
        calib_data = all_calib[:,e].flatten()
        valid_data = all_valid[:,e].flatten()
        _,bins = np.histogram(np.append(calib_data, valid_data))
        ax.hist(calib_data, bins=bins, histtype='step', color='limegreen', label='calibration', linewidth=2)
        ax.hist(valid_data, bins=bins, histtype='step', color='#0d7885', label='validation', linewidth=2)
        ax.set_title(errorlabels[error.split('_')[0]])
        ax.tick_params(length=5)
    axes[1].legend()
    title = 'Comparison of calibration and validation\nerror distribution for Pareto fronts'
    # fig.suptitle(title, y=1.03)
    fig.supylabel('Count')
    if savefig:
        plt.savefig(base_fp+'calib_valid_comparison.png',bbox_inches='tight', dpi=300)
    plt.show()

def plot_sensitivity(sens_dict,savefig=False):
    label_dict = {'kp':{'label':'Precipitation\nfactor (-)','high':3,'low':1},
                    'Boone_c5':{'label':'Densification\nparameter (-)','high':0.024,'low':0.014},
                    'lapserate':{'label':'Temperature lapse\nrate ( K km$^{-1}$)','high':'$-$7','low':'$-$5.5'},
                    # 'rfz_grainsize':{'label':'Refrozen\ngrain size ($\mu m$)','high':val,'low':val},
                    # 'roughness_rate':{'label':'Roughness\ndecay rate (mm/day)','high':val,'low':val},
                    'roughness_ice':{'label':'Ice\nroughness ( mm)','high':40,'low':10},
                    'albedo_ground':{'label':'Ground\nalbedo (-)','high':0.3,'low':0.1}, 
                    # 'diffuse_cloud_limit':'Cloudy\nthreshold (-)',
                    'ksp_BC':{'label':'Scavenging efficiency of BC (-)','high':1.2,'low':0.1},
                    'ksp_OC':{'label':'Scavenging efficiency of OC (-)','high':1.2,'low':0.1},
                    'ksp_dust':{'label':'Scavenging efficiency of dust (-)','high':0.1,'low':0.01},
                    'roughness_fresh_snow':{'label':'Fresh snow\nroughness ( mm)','high':1,'low':0.1},
                    'roughness_aged_snow':{'label':'Aged snow\nroughness ( mm)','high':20,'low':5}}
    base = sens_dict['base']['base']

    fig, axes = plt.subplots(2,1,figsize=(5,3),sharex=True)
    # colors = mpl.colormaps['Set3']
    colors = all_colors

    # Loop through vars
    all_names = []
    vv = -1
    for v, var in enumerate(sens_dict):
        if var in label_dict:
            vv += 1
            for c,compare in enumerate(['mb','density']):
                ax = axes[c] 
                point_50 = sens_dict[var]['-20'][compare] - base[compare]
                point_200 = sens_dict[var]['+20'][compare] - base[compare]
                x_50 = v - 0.2 -1
                x_200 = v + 0.2 - 1
                ax.bar(x_50, point_50, 0.4, align='center', color=colors[vv], hatch='///')
                ax.bar(x_200, point_200, 0.4, align='center', color=colors[vv])
                label = label_dict[var]['label'].replace('\n',' ')
                high = label_dict[var]['high']
                low = label_dict[var]['low']
                label = label.replace('(-)','()').replace('(',f'({low}$-${high}')
                if c == 0:
                    all_names.append(label)
                ax.axvline(x_200+0.3, color='k', linewidth=0.1)
                
                # ax.text(v-1, 0.27, str(high),ha='center',va='center')
                # ax.text(v-1, -0.27, str(low),ha='center',va='center')

    axes[0].set_ylim(-0.8,0.8)
    axes[1].set_ylim(-70,70)
    for ax in axes:
        ax.set_xticks(np.arange(len(sens_dict)-1)+0.3)
        ax.tick_params(axis='y',length=5)
        ax.axhline(0, linewidth=0.5, color='k')
        ax.set_xlim(-0.5,x_200+0.3)
    ax.set_xticklabels(all_names, rotation=45, ha='right')
    fig.supylabel('Deviation from default simulation',x=-0.1,fontsize=14)
    axes[0].set_ylabel('Mass balance\n(m w.e.)',fontsize=12)
    axes[1].set_ylabel('Mean snow\ndensity (kg m$^{-3}$)',fontsize=12)

    ax.bar(np.nan, np.nan, np.nan, color='lightgray',hatch='///',label='Lower end')
    ax.bar(np.nan, np.nan, np.nan, color='lightgray', label='Upper end')
    ax.legend(facecolor='white',frameon=True,bbox_to_anchor=(0.7,0.55))

    if savefig:
        plt.savefig(base_fp + 'sensitivity_analysis.png', dpi=200, bbox_inches='tight')
    plt.show()

def plot_bias_correction(mb_dict, savefig=False):
    mb_df = pd.read_csv(USGS_fp, index_col=0)
    fig, ax = plt.subplots(figsize=(4,3))
    colors = all_colors
    all = [[],[],[]]
    for s,site in enumerate(mb_dict):
        ba = mb_df.loc[mb_df['site_name'] == site].loc[2024]['ba']
        bw = mb_df.loc[mb_df['site_name'] == site].loc[2024]['bw']
        sacc = mb_df.loc[mb_df['site_name'] == site].loc[2024]['summer_accumulation']
        bs = ba - bw + sacc
        print(site, bs, 'og',mb_dict[site]['og'] - bs, 'bc',mb_dict[site]['bc'] - bs, 'aws', mb_dict[site]['aws'] - bs)

        ax.bar(s, mb_dict[site]['og'] - bs, 0.3, color=colors[s])
        ax.bar(s+0.3, mb_dict[site]['bc'] - bs, 0.3, color=colors[s], hatch='///')
        ax.bar(s+0.6, mb_dict[site]['aws'] - bs, 0.3, color=colors[s], hatch='....')
        all[0].append(mb_dict[site]['og'] - bs)
        all[1].append(mb_dict[site]['bc'] - bs)
        all[2].append(mb_dict[site]['aws'] - bs)
    ax.tick_params(axis='y',length=5)
    ax.set_xticks([0.3,1.3,2.3,3.3,4.3])
    ax.set_xticklabels(['AU','AB','B','D','T'],fontsize=12)
    ax.axhline(0, linewidth=0.5, color='k')
    ax.set_ylabel('Modeled $-$ Measured\nmass balance (m w.e.)',fontsize=12)
    ax.set_xlabel('Site name',fontsize=12)
    print('og',np.mean(np.abs(all[0])),'bc', np.mean(np.abs(all[1])),'aws', np.mean(np.abs(all[2])))

    ax.bar(np.nan, np.nan, np.nan, color='lightgray', label='Original MERRA-2')
    ax.bar(np.nan, np.nan, np.nan, color='lightgray', label='Bias-corrected MERRA-2',hatch='///')
    ax.bar(np.nan, np.nan, np.nan, color='lightgray', label='Weather station',hatch='....',)

    ax.legend()
    if savefig:
        plt.savefig(base_fp + 'bias_correction_comparison.png',dpi=200,bbox_inches='tight')
    plt.show()

def plot_fluxes(savefig=False):
    fp = gsproc.base_fp + 'bias_correct/'
    flux_labels = {'SWnet':'$SW_{net}$','LWnet':'$LW_{net}$',
                    'sensible':'$Q_s$','latent':'$Q_l$',
                    'rain':'$Q_r$','ground':'$Q_g$'}
    data_labels = {'og':'Original\nMERRA-2','bc':'Bias-corrected\nMERRA-2','aws':'Weather\nstation'}
    vars = ['SWnet','LWnet','sensible','latent','rain','ground']
    colors = all_colors
    fig, axes = plt.subplots(5,3, sharex=True, sharey=True,gridspec_kw={'hspace':0, 'wspace':0})
    lax = fig.add_axes([0.15,-0.06,0.8,0.05])
    for s,site in enumerate(['T','D','B','AB','AU']):
        for d,data in enumerate(['og','bc','aws']):
            ax = axes[s,d]
            ds = xr.open_dataset(fp + f'{site}_{data}_0.nc')
            ds['hour'] = ('time', pd.to_datetime(ds['time'].values).hour)
            ds = ds.set_coords('hour')
            ds['SWnet'] = ds['SWin'] + ds['SWout']
            ds['LWnet'] = ds['LWin'] + ds['LWout']
            ds_hourly = ds.groupby('hour').mean(dim='time')
            for v,var in enumerate(vars):
                ax.plot(ds_hourly['hour'], ds_hourly[var], color=colors[v])
                if s == 0 and d == 0:
                    lax.plot(np.nan, np.nan,label=flux_labels[var],color=colors[v])
            ds.close()
            ax.tick_params(length=5,which='major')
            ax.minorticks_on()
            ax.tick_params(length=2,which='minor')
            ax.set_xticks([0,8,16])
            ax.set_xlim(0,23)
            axes[0, d].set_title(data_labels[data],fontsize=12)
            
        axes[s, -1].set_ylabel(site,rotation=270,fontsize=12,labelpad=15)
        axes[s, -1].yaxis.set_label_position('right')
    fig.text(0.96, 0.48,'Site name',rotation=270,fontsize=12)
    fig.supxlabel('Hour of day')
    fig.supylabel('Flux magnitude (W m$^{-2}$)')

    lax.axis('off')
    lax.legend(ncols=len(vars))
    if savefig:
        plt.savefig(base_fp + 'fluxes.png',dpi=200,bbox_inches='tight')
    plt.show()

def find_precip_gradient():
    data = pd.read_csv('../MB_data/Gulkana/Input_Gulkana_Glaciological_Data.csv')
    grads = []
    colors = plt.get_cmap('Grays')
    norm = mpl.colors.Normalize(vmin=1990, vmax=2025)
    all_elev = []
    all_bw = []
    for year in np.unique(data['Year']):
        if year > 1999:
            year_data = data.loc[data['Year'] == year]
            year_elev = year_data['elevation'].values[~np.isnan(year_data['bw'].values)]
            year_bw = year_data['bw'].values[~np.isnan(year_data['bw'].values)]
            try:
                year_B_bw = year_data.loc[year_data['site_name'] == 'B']['bw'].values
                winter_abl_B =  year_data.loc[year_data['site_name'] == 'B']['winter_ablation'].values
                winter_abl_B = 0 if np.isnan(winter_abl_B) else 0
                year_acc_B = year_B_bw - winter_abl_B
            except:
                print('no site B for',year)
                continue
            winter_abl = year_data['winter_ablation'].values[~np.isnan(year_data['bw'].values)]
            winter_abl[np.isnan(winter_abl)] = 0
            year_acc = year_bw - winter_abl
            percentage_acc = year_acc / year_acc_B
            for i in range(len(percentage_acc)):
                all_bw.append(percentage_acc[i])
                all_elev.append(year_elev[i])
            # gradient = np.sum(year_elev * year_acc) / np.sum(year_elev**2)
            gradient ,b = np.polyfit(year_elev, percentage_acc , 1)
            if gradient > 0:
                grads.append(gradient)
                plt.scatter(year_elev, percentage_acc, color=colors(norm(year)))
                plt.plot(year_elev, year_elev * gradient+b, color=colors(norm(year)))
    
    gradient ,b = np.polyfit(all_elev, all_bw, 1)
    print(gradient)
    
    return np.nanmean(grads)