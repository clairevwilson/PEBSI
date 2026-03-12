import matplotlib.pyplot as plt 
import matplotlib as mpl
from scipy.stats import gaussian_kde
import pandas as pd
import numpy as np
import xarray as xr

# define filepaths
climate_fp = '/trace/group/rounce/cvwilson/climate_data/'
gfed_fp = climate_fp + 'UKESM/dr401_GFED/'
nofires_fp = climate_fp + 'UKESM/dw068_nofires/'
merra_fp = '~/research/climate_data/MERRA2/63.5_-145.6/'

# gulkana location
point_lat = 63.259091
point_lon = -145.428586

# deposition types
dep_list = ['wet','dry']
species_list = ['BC','OC']

plot_type = 'scatter'
resample = '1d'

# create 2x2 figure
fig, axes = plt.subplots(2, 2, figsize=(6, 6), 
                         gridspec_kw={'hspace':0.3, 'wspace':0.4})

# loop through BC/OC and wet/dry
for axrow, species in zip(axes, species_list):
    for ax, deptype in zip(axrow, dep_list):
        # open UK-ESM base simulation
        gfed_fn = gfed_fp + f'sum_{species.lower()}_{deptype}deposition_kgm-2s-1.nc'
        ds_gfed = xr.open_dataset(gfed_fn, decode_timedelta=False)

        # find variable name for this deposition type
        for v in list(ds_gfed.variables):
            if 'tendency' in v:
                vuse = v 

        # open UK-ESM no fires simulation
        nofires_fn = nofires_fp + f'sum_{species.lower()}_{deptype}deposition_kgm-2s-1.nc'
        ds_nofire = xr.open_dataset(nofires_fn, decode_timedelta=False)

        # get longitude in 0-360 system
        point_lon_360 = 360 + point_lon if point_lon < 0 else point_lon

        # select the point
        ds_gfed = ds_gfed[vuse].sel(latitude=point_lat, 
                                    longitude=point_lon_360,
                                    method='nearest')
        ds_nofire = ds_nofire[vuse].sel(latitude=point_lat, 
                                        longitude=point_lon_360,
                                        method='nearest')
        lat_gfed = ds_gfed.latitude.values
        lon_gfed = ds_gfed.longitude.values 
        assert abs(lat_gfed - point_lat) < 1 and abs(lon_gfed - point_lon_360) < 1, 'Wrong coord selected'
        
        # convert units
        ds_gfed = ds_gfed * (3600 * 24) # convert from kg m-2 s-1 to kg m-2 (DAILY data)
        ds_nofire = ds_nofire * (3600 * 24) # convert from kg m-2 s-1 to kg m-2 (DAILY data)

        # open MERRA-2 file
        dep_str_merra = 'WT' if deptype == 'wet' else 'DP'
        merra_fn = merra_fp + f'{species}{dep_str_merra}002_63.5_-145.6.nc'
        ds_merra2 = xr.open_dataarray(merra_fn)

        # convert units
        ds_merra2 = ds_merra2 * 3600 # convert from kg m-2 s-1 to kg m-2 (HOURLY data)

        # find timestamps in common
        start = max(ds_gfed.time.min().values,ds_merra2.time.min().values)
        end = min(ds_gfed.time.max().values,ds_merra2.time.max().values)

        # select common timestamps
        ds_gfed = ds_gfed.sel(time=slice(start, end))
        ds_nofire = ds_nofire.sel(time=slice(start, end))
        ds_merra2 = ds_merra2.sel(time=slice(start, end))

        # resample by summing
        ds_gfed = ds_gfed.resample(time=resample).sum()
        ds_nofire = ds_nofire.resample(time=resample).sum()
        ds_merra2 = ds_merra2.resample(time=resample).sum()
        print(f'base sum: ',ds_gfed.sum(dim='time').values, 'no fires sum',ds_nofire.sum(dim='time').values, 'MERRA sum',ds_merra2.sum(dim='time').values)

        # plot
        ax.tick_params(length=5)
        if plot_type == 'scatter':
            max_val = 0.00021 # max(ds_gfed.max().values, ds_nofire.max().values, ds_merra2.max().values)
            min_val = 1e-11 # max(min(ds_gfed.min().values, ds_nofire.min().values, ds_merra2.min().values), 1e-11)

            x = ds_merra2.values
            y = ds_gfed.values
            x = x[np.isfinite(x) & np.isfinite(y)]
            y = y[np.isfinite(x) & np.isfinite(y)]
            x_mean = np.mean(x)
            y_mean = np.mean(y)

            # Slope and intercept (OLS)
            slope = np.sum((x - x_mean)*(y - y_mean)) / np.sum((x - x_mean)**2)
            intercept = y_mean - slope * x_mean
            bias = np.mean(y - x)

            xy = np.vstack([ds_merra2.values, ds_gfed.values])
            z = gaussian_kde(xy)(xy)
            ax.scatter(ds_merra2.values, ds_gfed.values, c=z, cmap='magma', alpha=0.5)
            ax.plot([0, max_val], [0, max_val], 'k--', label='1:1')
            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)
            ax.set_xscale('log')
            ax.set_yscale('log')
            # ax.text(0.6, 0.02, f'Bias: {bias:.3f} kg m'+r'$^{-2}$',transform=ax.transAxes)
            # ax.text(0.6, 0.08, f'Slope: {slope:.3f}',transform=ax.transAxes)
        elif plot_type == 'timeseries':
            ax.plot(ds_gfed.time.values, ds_gfed.values, label='UK-ESM (GFED)')
            ax.plot(ds_nofire.time.values, ds_nofire.values, label='UK-ESM (no fires)')
            ax.plot(ds_merra2.time.values, ds_merra2.values, label='MERRA-2')

            ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mpl.dates.YearLocator(4))
        ax.set_title(f'{species} {deptype}')

axes[0,0].legend() # bbox_to_anchor=(0.65,0.8), loc='center',fontsize=9)
time_str = 'annual' if 'Y' in resample else 'daily'
fig.suptitle(time_str.capitalize() + ' deposition sums (kg m$^{-2}$)')
if plot_type == 'timeseries':
    plt.savefig(f'{time_str}_deposition_timeseries.png')
elif plot_type == 'scatter':
    fig.suptitle(time_str.capitalize() + ' deposition sums (kg m$^{-2}$)')
    fig.supxlabel('MERRA-2 deposition')
    fig.supylabel('UK-ESM deposition')
    plt.savefig(f'{time_str}_deposition_1to1.png')
plt.show()