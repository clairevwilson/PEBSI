"""
Loads the ice albedo information on a 
per-glacier basis utilizing SAR snowlines
to determine dates of ice exposure.
"""

import xarray as xr
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 
import geopandas as gpd
import os 
import argparse 
import rasterio
from pyproj import Transformer
from data_handling import translate_rgi

glaciers = ['kahiltna','gulkana','kennicott','wolverine','lemon_creek','taku']

# define filepaths
base_fp = '/ocean/projects/ees260009p/cwilson4/data/sar/'
output_fp = '/ocean/projects/ees260009p/cwilson4/data/ice_albedo/'

# use Ascending, Descending or both (minimum) scenes to get snowline?
sar_direction_use = 'both' 
# use buffer around minimum snowline elevation? [m]
buffer = 0
# albedo above threshold is filtered out (assumed to be snow)
albedo_threshold = 0.65
# difference in snowline between SAR and albedo thresholds above this threshold is thrown out
snowline_threshold = 300
# store average ice albedo in site_constants?
store_aice = True

rgi = gpd.read_file(base_fp + '../../RGI/rgi60/01_rgi60_Alaska/01_rgi60_Alaska.shp')

for glacier in glaciers:
    # fill out file names
    pathframe_fn = base_fp + 'Vertex_Path_Frame_info.csv'
    albedo_fp = base_fp + '../albedo/'
    dem_fp = base_fp + '../dems/RGI1_DEM/RGI60-GLAC_NO_dem.tif'

    # find rgi7 glacier number
    rgi6id = translate_rgi[glacier]['6']
    rgi7id = translate_rgi[glacier]['7']
    folder = base_fp + rgi7id + '/'

    # grab shapefile for this glacier 
    glacier_outline = rgi.loc[rgi['RGIId'] == 'RGI60-' + rgi6id]

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
                    assert 1==0, 'Frames have mismatched direction! Yikes. Ask awwells@cmu.edu'
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
    albedo_fns = [albedo_fp + f'{num}/RGI2000-v7.0-G-01-{num}_{data}.nc' for data in use_list]

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

    # crop to RGI outline
    ds_all = ds_all.rio.write_crs(epsg)
    glacier_outline = glacier_outline.to_crs(epsg)
    ds_all = ds_all.rio.clip(glacier_outline.geometry.values, glacier_outline.crs)

    # filter out bad values 
    ds_all = ds_all.where((ds_all['albedo'] > 0.1) & (ds_all['albedo'] < 0.9))

    # change time to dates
    ds_all['time'] = pd.to_datetime(ds_all.time.dt.date)

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

    # take overall average albedo 
    average_spatial_albedo = ds_masked['albedo'].mean(dim=['time'])

    average_spatial_albedo.rio.to_raster(
        output_fp + f'{rgi6id}_albedo.tif',
        driver='GTiff',
        compress='deflate',
        tiled=True
    )