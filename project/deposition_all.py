import matplotlib.pyplot as plt 
import matplotlib as mpl
from scipy.stats import gaussian_kde
import pandas as pd
import numpy as np
import xarray as xr
import time
import pickle
colors = ['#63c4c7','#fcc02e','#4D559C','#60C252','#BF1F6A']

# plotting options
plot_type = 'scatter'
resample = 'd'

# define filepaths
climate_fp = '/trace/group/rounce/cvwilson/climate_data/'
gfed_fp = climate_fp + 'UKESM/dr401_GFED/'
nofires_fp = climate_fp + 'UKESM/dw068_nofires/'
merra_base_fp = climate_fp + 'MERRA2/'
output_fp = climate_fp + '../figs/ukesm/'

# location
glacier = 'gulkana'
coord_dict = {'wolverine':'60.5_-148.7',
                'kahiltna':'63.0_-151.2',
                'kennicott':'61.5_-143.1',
                'taku':'58.5_-134.3',
                'lemon_creek':'58.5_-134.3',
                'gulkana':'63.5_-145.6'}

# deposition types
dep_list = ['wet','dry']
species_list = ['BC','OC']

# get deposition ratio maps
ratios_bc = xr.open_dataset(climate_fp + 'MERRA2/reg01_BC_regression_map.nc')
ratios_oc = xr.open_dataset(climate_fp + 'MERRA2/reg01_OC_regression_map.nc')

# loop through BC/OC and wet/dry
all_datasets = {'all_GFED':{'BCwet':[], 'BCdry':[], 'OCwet':[], 'OCdry':[]}, 
                'all_MERRA':{'BCwet':[], 'BCdry':[], 'OCwet':[], 'OCdry':[]}}
for glacier in coord_dict:
    start_time = time.time()

    coords = coord_dict[glacier]
    merra_coords = coord_dict[glacier]
    merra_fp = merra_base_fp + merra_coords + '/'

    site_df = pd.read_csv(f'../data/by_glacier/{glacier}/site_constants.csv', index_col=0)
    point_lat = site_df.loc['center', 'lat']
    point_lon = site_df.loc['center', 'lon']

    ratio_bc = ratios_bc['ratio'].sel(lat=point_lat, lon=point_lon, method='nearest').values
    ratio_oc = ratios_oc['ratio'].sel(lat=point_lat, lon=point_lon, method='nearest').values
    print(glacier.capitalize(), 'Ratios: BC', ratio_bc, 'OC', ratio_oc)

    for species in species_list:
        for deptype in dep_list:
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
            merra_fn = merra_fp + f'{species}{dep_str_merra}002_{merra_coords}.nc'
            ds_merra2 = xr.open_dataarray(merra_fn)

            # convert units
            ds_merra2 = ds_merra2 * 3600 # convert from kg m-2 s-1 to kg m-2 (HOURLY data)

            # apply ratio from hydrophilic --> both
            ratio = ratio_bc if species == 'BC' else ratio_oc
            if deptype == 'dry':
                ds_merra2 *= ratio

            # find timestamps in common
            start = max(ds_gfed.time.min().values,ds_merra2.time.min().values)
            end = min(ds_gfed.time.max().values,ds_merra2.time.max().values)

            # select common timestamps
            ds_gfed = ds_gfed.sel(time=slice(start, end))
            # ds_nofire = ds_nofire.sel(time=slice(start, end))
            ds_merra2 = ds_merra2.sel(time=slice(start, end))

            # resample by summing
            # ds_gfed = ds_gfed.resample(time=resample).sum()
            # ds_nofire = ds_nofire.resample(time=resample).sum()
            ds_merra2 = ds_merra2.resample(time=resample).sum()
            
            timer = time.time() - start_time
            print(f'Got', species, deptype, 'for', glacier, f'in {timer:.1f} seconds')
            for gfed, merra in zip(ds_gfed.values, ds_merra2.values):
                all_datasets['all_GFED'][species+deptype].append(gfed)
                all_datasets['all_MERRA'][species+deptype].append(merra)

with open('all_dep.pkl', 'wb') as file:
    pickle.dump(all_datasets, file)

# create 2x2 figure
fig, axes = plt.subplots(2, 2, figsize=(6, 6), 
                         gridspec_kw={'hspace':0.4, 'wspace':0.4})

for axrow, species in zip(axes, species_list):
    for ax, deptype in zip(axrow, dep_list):
        bins = np.logspace(-13, -4, num=20)
        assert len(all_datasets['all_MERRA'][species+deptype]) == len(all_datasets['all_GFED'][species+deptype])
        ax.hist(all_datasets['all_MERRA'][species+deptype], histtype='step', label='MERRA-2', bins=bins, color=colors[1])
        ax.hist(all_datasets['all_GFED'][species+deptype], histtype='step', label='GFED', bins=bins, color=colors[0])
        bias = np.mean(np.array(all_datasets['all_GFED'][species+deptype]) - np.array(all_datasets['all_MERRA'][species+deptype]))
        # ax.text(0.98, 0.98, f'Bias: {bias:.3e}', ha='right', va='top', transform=ax.transAxes)
        ax.set_title(species + ' '+ deptype + '\n' + f'Bias: {bias:.3e}')
        ax.set_xscale('log')
# axes[0, 0].legend()
fig.supxlabel('Deposition (kg m$^{-2}$)')
fig.supylabel('Count')
fig.suptitle('Daily deposition for six glaciers', y=1.01)
plt.savefig('deposition_all.png', bbox_inches='tight')
plt.show()