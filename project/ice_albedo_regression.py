"""
Annual ice albedo regression

This code brings together multiple datasets
to determine the mean ice albedo at a given
site.

1. Retrieve SAR snowlines for a given glacier
   using either Ascending or Descending scenes.
2. Retrieve the albedo datasets from Landsat 8-9
   and Sentinel 2 and concatenate into one cube.
3. For each albedo scene, mask the glacier based
   on the SAR snowline from that date.
4. Take the mean albedo for ice-exposed (below
   snowline) days at the site of interest.

@author: clairevwilson
"""

import xarray as xr
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 
import os 
import argparse 
import rasterio
from pyproj import Transformer

# define filepaths
home_fp = '~/research/'
base_fp = '/trace/group/rounce/cvwilson/rs/sar/'

# use Ascending, Descending or both (minimum) scenes to get snowline?
sar_direction_use = 'both' 
# use buffer around minimum snowline elevation? [m]
buffer = 0
# albedo above threshold is filtered out (assumed to be snow)
albedo_threshold = 0.6
# difference in snowline between SAR and albedo thresholds above this threshold is thrown out
snowline_threshold = 300
# plot maps of average albedo in any specific year?
plot_map_years = [2019, 2025]
# store average ice albedo in site_constants?
store_aice = True

site_dict = { # ABLATION AREA ONLY
    'kahiltna':['K53',], 'kennicott':['KC31','GTL'],
    'gulkana':['AU','B'], 'wolverine':['N','B'], # WOLVERINE
    'lemon_creek':['B']}
translate_rgi = {
                 'gulkana':{'6': '01.00570', '7':'01.05299'}, # GULKANA
                 'kahiltna':{'6':'01.22193','7':'01.04282'}, # KAHILTNA
                 'kennicott':{'6':'01.15645','7':'01.05740'}, # KENNICOTT
                 'wolverine':{'6':'01.09162','7':'01.11350'}, # WOLVERINE
                 'lemon_creek':{'6':'01.01104','7':'01.19406'}, # LEMON CREEK
                 'taku':{'6':'01.01390','7':'01.19709'}, # TAKU
                 }
coords_dict = {'wolverine':'60.5_-148.7',
                'kahiltna':'63.0_-151.2',
                'kennicott':'61.5_-143.1',
                'taku':'58.5_-134.3',
                'lemon_creek':'58.5_-134.3',
                'gulkana':'63.5_-145.6'}

for glacier in ['kahiltna']: # site_dict: # 
    # fill out file names
    glacier_fp = home_fp + 'PEBSI/data/by_glacier/'
    glacier_fn = glacier_fp + f'{glacier}/site_constants.csv'
    pathframe_fn = base_fp + 'Vertex_Path_Frame_info.csv'
    albedo_fp = base_fp + '../albedo/'
    dem_fp = base_fp + '../../dems/RGI1_DEM/RGI60-GLAC_NO_dem.tif'
    merra_fp = base_fp + '../../climate_data/MERRA2/COORDS/VAR_COORDS.nc'
    output_fp = base_fp + '../../figs/ice_albedo/'

    # find rgi7 glacier number
    rgi6id = translate_rgi[glacier]['6']
    rgi7id = translate_rgi[glacier]['7']
    folder = base_fp + rgi7id + '/'

    # ==================== 1. READ SAR DATA ====================
    # load dataframe containing path/row pairs
    df_pathframe = pd.read_csv(pathframe_fn, dtype=str)
    df_pathframe['Path'] = df_pathframe['Path'].apply(lambda x: f"{int(x):03d}")
    df_pathframe['Frame'] = df_pathframe['Frame'].apply(lambda x: f"{int(x):03d}")

    # initialize variable
    df_snowline = None

    # figure out the path of the ascending scene
    for fn in os.listdir(folder):
        # filter out extraneous files
        if 'snowline_elev_percentile' in fn and not 'ea' in fn and not 'eos' in fn:
            if sar_direction_use == 'both':
                df = pd.read_csv(folder + fn, parse_dates=True, index_col=0)

                if df_snowline is None:
                    df_snowline = df 
                else:
                    df_snowline = pd.concat([df_snowline, df])
            else:
                # find the path and frame number of this scene
                path = fn.split('ile_')[1][:3]
                frame = fn.split(path)[1].split('.csv')[0][1:]
                if len(frame) > 3:
                    frames = [str(f) for f in frame.split('_')]
                else:
                    frames = [str(frame)]

                # select the pathframe dataset at this path and frame(s)
                df_path = df_pathframe.loc[df_pathframe['Path'] == path]
                df_frame = df_path.loc[df_path['Frame'].isin(frames)]
                
                # determine direction of this path and frame(s)
                direction = df_frame['Direction'].values
                if len(direction) > 1 and len(np.unique(direction)) > 1:
                    assert 1==0, 'Frames have mismatched direction! Yikes. awwells@cmu.edu'
                else:
                    direction = direction[0]

                # if this is the correct direction, this is your dataset
                if direction == sar_direction_use:
                    fn_snow = fn
                    
                    # open the dataframe containing snowline elevations
                    df_snowline = pd.read_csv(folder + fn_snow, parse_dates=True, index_col=0)

    # take minimum of duplicate timestamps
    df_snowline['time'] = df_snowline.index
    df_snowline = df_snowline.groupby('time').min()

    # reindex dataframe so it contains all dates
    all_dates = pd.date_range(df_snowline.index[0], df_snowline.index[-1])
    df_snowline = df_snowline['snowline_elev_min_m'].reindex(all_dates).ffill()

    # =================== 2. READ ALBEDO DATA ===================
    # get names of each albedo datacube for S2, L8 and L9
    num = str(rgi7id[3:])
    use_list = ['s2','l8','l9']
    albedo_fns = [albedo_fp + f'{num}/{num}_{data}.nc' for data in use_list]

    # build dataset with all three datasets
    ds_all = None
    for albedo_fn, dtype in zip(albedo_fns, use_list):
        # open the dataset and get the proper CRS
        ds = xr.open_dataset(albedo_fn)
        crs = ds.spatial_ref.attrs['crs_wkt']
        epsg = crs.split('AUTHORITY["EPSG","')[-1].split('"]')[0]

        ds = ds['albedo'].rio.write_crs(epsg).reset_coords(drop=True).to_dataset()
        ds['dtype'] = ('time', np.array([dtype]*len(ds.time.values)).flatten())    
        if ds_all is None:
            ds_all = ds 
        else:
            ds_all = xr.concat([ds_all, ds], dim='time')

    # take mean of any duplicate timesteps
    ds_all = ds_all.squeeze('band')
    ds_albedo = ds_all.groupby('time').mean()

    # =================== 3. MASK OUT SNOW ===================
    # create mask based on elevation for each date that says below/above snowline
    dem_path = dem_fp.replace('GLAC_NO', rgi6id)
    da_dem = xr.open_dataarray(dem_path, engine='rasterio').squeeze()

    # ensure both datasets are on the same crs
    da_dem = da_dem.rio.reproject(f'EPSG:{epsg}')
    ds_albedo = ds_albedo.rio.write_crs(f'EPSG:{epsg}')

    # reproject to match the albedo grid
    da_dem_matched = da_dem.rio.reproject_match(ds_albedo)

    # create dataset of the snowline dataframe
    da_snowline = xr.DataArray(
        df_snowline.values, 
        coords={'time': df_snowline.index}, 
        dims=['time']
    )

    # create snow mask with buffer
    ice_mask = da_dem_matched <= da_snowline - buffer

    # also mask out NaNs
    if da_dem.rio.nodata is not None:
        valid_terrain = da_dem_matched != da_dem.rio.nodata
        ice_mask = ice_mask & valid_terrain

    # masked albedo dataset
    ds_masked = ds_albedo.where(ice_mask).sortby(['x','y'])

    # mask out bad SAR days where it does not capture snowline well
    snow_elevs = da_dem_matched.where(ds_masked['albedo'] >= albedo_threshold)
    data_elevs = da_dem_matched.where(ds_masked['albedo'].notnull())
    est_snowline = (snow_elevs
                    .fillna(data_elevs.max(dim=['x', 'y']))
                    .min(dim=['x', 'y'])
                    )
    valid_times = abs(est_snowline - da_snowline) <= snowline_threshold
    ds_masked = ds_masked.where(valid_times, drop=True)

    # group by year
    annual_spatial_albedo = (ds_masked['albedo']
                             .groupby('time.year')
                             .mean(dim=['time']))
    
    # ============== 4. GET LAP DEPOSITION DATA ==============
    # get annual black carbon deposition for each year
    coords = coords_dict[glacier]
    carbon_vn = 'BCDP002'
    bc_ds = xr.open_dataarray(merra_fp.replace('COORDS', coords).replace('VAR', carbon_vn))

    # select it to point albedo time
    start_year = annual_spatial_albedo.year.values[0]
    end_year = annual_spatial_albedo.year.values[-1]
    bc_ds = bc_ds.sel(time=slice(str(start_year - 1)+'-10-01', str(end_year)+'-01-01'))

    # get annual sum of black carbon deposition
    annual_deposition = bc_ds.resample(time='YS-OCT').sum() * 3600

    # FIGURE 1: MAPS
    list_figs = [plt.subplots(figsize=(5, 5)) for _ in plot_map_years]
    years_use = []
    for year, (fig, ax) in zip(plot_map_years, list_figs):
        # get closest year to the requested year that we have data for
        all_years = annual_spatial_albedo.year.values
        year_use = all_years[np.argmin(np.abs(all_years - year))]
        years_use.append(year_use)

        # crop out NaNs
        year_albedo = (annual_spatial_albedo.sel(year=year_use)
                        .where(lambda x: ~x.isnull(), drop=True))
        rect = mpl.patches.Rectangle(
                    (0, 0), 1, 1,
                    transform=ax.transAxes,
                    facecolor='none',
                    edgecolor='darkgray',
                    hatch='///',
                    linewidth=0
                )
        ax.add_patch(rect)
        year_albedo.plot(ax=ax, 
            cmap='Grays_r', 
            vmin=0.1, vmax=0.6,
            cbar_kwargs={'label': 'Mean Ice Albedo'}
        )

    # SITE LOOP
    for site in site_dict[glacier]:
        # =============== 5. EXTRACT SITE INFORMATION ===============
        site_df = pd.read_csv(glacier_fn, index_col='site')
        elevation = site_df.loc[site, 'elevation']
        point_lat = float(site_df.loc[site, 'lat'])
        point_lon = float(site_df.loc[site, 'lon'])

        # ============= 6. GET POINT AND SPATIAL ALBEDO =============
        # convert point lat/lon to x/y
        transformer = Transformer.from_crs('EPSG:4326', f'EPSG:{epsg}', always_xy=True)
        point_x, point_y = transformer.transform(point_lon, point_lat)

        # group by years and take the average
        annual_point_albedo = (ds_masked['albedo']
                            .sel(x=point_x, y=point_y, method='nearest')
                            .groupby('time.year').mean(dim='time'))

        # create dataframe from annual albedo
        df_annual_albedo = annual_point_albedo.to_dataframe(name='aice').reset_index()

        # get point time-averaged ice albedo
        avg_point_albedo = (ds_masked['albedo']
                            .sel(x=point_x, y=point_y, method='nearest')
                            .mean(dim='time')).values
        if store_aice:
            site_df.loc[site, 'a_ice'] = avg_point_albedo 
        
        # =================== 7. REGRESSION ====================
        # relate to ice albedo
        X = annual_deposition.values 
        y = annual_point_albedo.values 

        # mask any nans
        mask = ~np.isnan(X) & ~np.isnan(y)
        X = X[mask].flatten()
        y = y[mask].flatten()

        # get slope, intercept of line of fit
        slope, intercept = np.polyfit(X, y, 1)

        # get R2
        y_pred = slope * X + intercept
        res = y - y_pred 
        r2 = 1 - np.sum(res**2) / np.sum((y - np.mean(y))**2)

        # =================== 9. FIGURES ====================
        # add sites to maps
        for (fig, ax) in list_figs:
            ax.text(point_x, point_y, site, fontdict={'size':15, 'color':'red'})

        # FIGURE 2: REGRESSION
        # plot data
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(X, y, color='darkred')

        # plot regression line
        Xplot = np.array([ax.get_xlim()[0], ax.get_xlim()[-1]])
        yplot = slope * Xplot + intercept
        ax.plot(Xplot, yplot, color='k', linewidth=0.5)
        ax.set_xlim(Xplot)
        ax.text(0.98, 0.92, f'Slope: {slope:.2e}', ha='right', transform=ax.transAxes)
        ax.text(0.98, 0.85, f'Intercept: {intercept:.3f}', ha='right', transform=ax.transAxes)
        ax.text(0.98, 0.77, f'R$^2$: {r2:.3f}', ha='right', transform=ax.transAxes)
        ax.set_xlabel('Annual BC deposition (kg m$^{-2}$)       ')
        ax.set_ylabel('Annual mean ice albedo (-)')
        glacier_str = [f.capitalize() for f in glacier.split('_')]
        glacier_str = ' '.join(glacier_str)
        ax.set_title(f'{glacier_str} {site}')
        plt.savefig(output_fp + f'{glacier}_{site}_ice_regression.png', bbox_inches='tight', dpi=300)
        plt.close()
        print('Done with', glacier, site)
    
    for year, (fig, ax) in zip(years_use, list_figs):
        fig.savefig(output_fp + f'{glacier}_aice_{year}.png')
        plt.close(fig)