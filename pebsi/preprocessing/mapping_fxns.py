"""
This script contains functions to perform quantile mapping,
plot the results, and store the mapping data as called in
quantile_mapping.ipynb.

@author: clairevwilson
"""
import pandas as pd
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt
import matplotlib as mpl

all_colors = ['#63c4c7','#fcc02e','#4D559C','#60C252','#BF1F6A',
              '#F77808','#298282','#999999','#FF89B0','#427801']

def quantile_mapping(data_AWS,data_MERRA,
                     fn_store_quantiles=None):
    """
    Takes in AWS and MERRA-2 data and maps
    the MERRA-2 data to the AWS. If fn_store_quantiles
    is passed, stores the data to that fn.

    Parameters
    ----------
    data_AWS : np.array
        Array of AWS data
    data_MERRA : np.array
        Array of MERRA-2 data at the same timestamps
        as data_AWS
    fn_store_quantiles: str
        Filename to store the mappings to

    Returns
    =======
    data_MERRA_sorted : np.array
        Array of sorted MERRA-2 data
    quantile_map : np.array
        Mapping to interpolate MERRA-2 data onto
    """
    # Sort MERRA data and align AWS data accordingly
    sorted_indices = np.argsort(data_MERRA)
    data_MERRA_sorted = data_MERRA[sorted_indices]
    data_AWS_sorted = data_AWS[sorted_indices]

    # Compute empirical CDF
    reanalysis_cdf = (ss.rankdata(data_MERRA_sorted, method="average") - 1) / (len(data_MERRA_sorted) - 1)

    # Map MERRA quantiles to AWS quantiles
    quantile_map = np.interp(reanalysis_cdf, np.sort(reanalysis_cdf), np.sort(data_AWS_sorted))

    # Store data if executed with a var
    if fn_store_quantiles is not None:
        df = pd.DataFrame({'sorted': data_MERRA_sorted, 'mapping': quantile_map})
        df.to_csv(fn_store_quantiles)
        print('stored to', fn_store_quantiles)

    return data_MERRA_sorted, quantile_map

def plot_scatter(X_train, y_train, lims, fn_store_quantiles=None, plot_kde=False, plot_axes=None):
    """
    Creates a scatter plot with a 1:1 line 
    to view the difference between uncorrected
    and corrected MERRA-2 data.

    Parameters:
    X_train : np.array
        Array containing the AWS training data
    y_train : np.array 
        Array containing the MERRA-2 training data
    lims : tuple
        List or tuple of the plot y and x axis limits
    var : str
        Optional: the variable name being corrected
    plot_kde : Bool
        Optional: plot the density of points
        ! Takes much longer to execute this code
    plot_axes : plt.ax object
        Axis to plot on; otherwise a new ax is generated
    """
    if fn_store_quantiles is not None:
        df = pd.read_csv(fn_store_quantiles)
        sorted = df['sorted'].values
        mapping = df['mapping'].values
    else:
        # Train quantile map
        sorted, mapping = quantile_mapping(X_train, y_train)

    # Plot it
    if plot_axes is None:
        fig,(ax1,ax2) = plt.subplots(1,2,figsize=(6,2.75),gridspec_kw={'wspace':0.3,'hspace':0.3})
    else:
        (ax1,ax2) = plot_axes

    # Raw scatter
    if plot_kde:
        kde = ss.gaussian_kde([X_train, y_train])
        density = kde(np.vstack([X_train, y_train]))
        ax1.scatter(X_train, y_train, s=3, c=density, cmap=plt.cm.magma)
    else:
        ax1.scatter(X_train, y_train, s=2, c='gray')
    ax1.plot(lims,lims,'k--',label='1:1')
    ax1.legend()
    ax1.set_xlabel('AWS',fontsize=11)
    ax1.set_ylabel('MERRA-2',fontsize=11)
    ax1.set_title('Raw data')

    # Updated scatter
    updated = np.interp(y_train, sorted, mapping)
    if plot_kde:
        kde = ss.gaussian_kde([X_train, updated])
        density = kde(np.vstack([X_train, updated]))
        ax2.scatter(X_train, updated, s=3, c=density, cmap=plt.cm.magma)
    else:
        ax2.scatter(X_train, updated, s=2, c='gray')
    ax2.plot(lims,lims,'k--',label='1:1')
    ax2.set_xlabel('AWS',fontsize=11)
    ax2.set_title('Adjusted data')

    # All axes
    for ax in [ax1,ax2]:
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.tick_params(length=5)

    if plot_axes is None:
        return fig, (ax1,ax2)
    else:
        return ax1,ax2
    
def plot_quantile(X_train, y_train, y_all, timed, 
                  fn_store_quantiles=None, plot_axes=None, legend=True):
    """
    Creates a two panel plot showing A) the density distribution
    before and after correction of MERRA-2 data and the 
    distribution of AWS data as a histogram and B) a selected
    time range of data for all three datasets.

    Parameters:
    X_train : np.array
        Array containing the AWS training data
    y_train : np.array 
        Array containing the MERRA-2 training data
    y_all : np.array
        Array containing ALL MERRA-2 data for this variable
    timed : tuple
        List of clipped arrays containing:
        0: time (pd.date_range-type object)
        1: original MERRA-2 data clipped to time
        2: updated MERRA-2 data clipped to time
        3: AWS data clipped to time 
    var : str
        Optional: the variable name being corrected
    plot_axes : plt.ax object
        Axis to plot on; otherwise a new ax is generated
    legend : Bool
        Add legend to figure or not
    """
    if fn_store_quantiles is not None:
        df = pd.read_csv(fn_store_quantiles)
        sorted = df['sorted'].values
        mapping = df['mapping'].values
    else:
        # Train quantile map
        sorted, mapping = quantile_mapping(X_train, y_train)

    # Plot it
    if plot_axes is None:
        fig,(ax1,ax2) = plt.subplots(1,2,figsize=(6,2.75),gridspec_kw={'wspace':0.4})
    else:
        (ax1,ax2) = plot_axes

    # Colors
    c1,c2,c3,_,_, = ['#63c4c7','#fcc02e','#4D559C','#BF1F6A','#60C252']
    
    # Distributions
    updated_all = np.interp(y_all, sorted, mapping)
    _,hist_bins = np.histogram(X_train)
    ax1.hist(y_all,bins=hist_bins,histtype='step',orientation='horizontal',label='Original MERRA-2',linestyle='--',density=True,color=c1,linewidth=1.5)
    ax1.hist(updated_all,bins=hist_bins,histtype='step',orientation='horizontal',label='Corrected MERRA-2',density=True,color=c2,linewidth=1.5)
    ax1.hist(X_train,bins=hist_bins,histtype='step',orientation='horizontal',label='Weather station',density=True,color=c3,linewidth=1.5)
    ax1.set_ylabel('Probability density', fontsize=12)

    # Timeseries
    time, raw, adj, aws = timed
    ax2.plot(time, raw, linestyle='--',label='Original MERRA-2',color=c1)
    ax2.plot(time, adj, label='Corrected MERRA-2',color=c2)
    ax2.plot(time, aws,label='Weather station',color=c3)
    ax2.xaxis.set_major_formatter(mpl.dates.DateFormatter('%H:%M'))
    ax2.set_xticks(time[::12])
    ax2.set_xlim(time[0],time[-1])

    if legend:
        # Add fake axis
        ax3 = fig.add_axes([1.05,0.1,0.1,0.8])
        ax3.axis('off')
        ax3.plot(np.nan, np.nan,linestyle='--',label='Original MERRA-2',color=c1)
        ax3.plot(np.nan, np.nan,label='Corrected MERRA-2',color=c2)
        ax3.plot(np.nan, np.nan,label='Weather station',color=c3)
        ax3.legend(loc='center')

    # All axes
    for ax in [ax1,ax2]:
        ax.tick_params(length=5)

    if plot_axes is None:
        return fig, (ax1,ax2)
    else:
        return ax1,ax2
    
def select_random_48hr_window(df):
    # Ensure timestamps are sorted
    df = df.sort_index()
    df = df.reindex(pd.date_range(df.index[0], df.index[-1], freq='h'))

    window_size = 48

    # Slide through the DataFrame
    for i in range(len(df) - window_size + 1):
        window = df.iloc[i:i + window_size]
        if not window.isnull().any().any():
            return window.index