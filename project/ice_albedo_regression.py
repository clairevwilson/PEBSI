import xarray as xr
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 
import os 
import argparse 

home_fp = '~/research/'
base_fp = '/trace/group/rounce/cvwilson/rs/sar/'
glacier = 'taku'
site = 'MG1'

parser = argparse.ArgumentParser()
parser.add_argument('-g', '--glacier', default=glacier, type=str, help='Glacier name')
parser.add_argument('-s', '--site', default=site, type=str, help='Site name')
args = parser.parse_args()

translate_rgi = {
                 'gulkana':'01.05299', # GULKANA
                 'kahiltna':'01.04282', # KAHILTNA
                 'kennicott':'01.05740', # KENNICOTT
                 'wolverine':'01.11350', # WOLVERINE
                 'lemon_creek':'01.19406', # LEMON CREEK
                 'taku':'01.19709', # TAKU
                 }

# find site elevation 
glacier_fp = home_fp + 'PEBSI/data/by_glacier/'
site_df = pd.read_csv(glacier_fp + f'{args.glacier}/site_constants.csv', index_col='site')
elevation = site_df.loc[site, 'elevation']

# find rgi7 glacier number
rgi7id = translate_rgi[args.glacier]
folder = base_fp + rgi7id + '/'

# load dataframe containing path/row pairs
df_pathframe = pd.read_csv(base_fp + 'Vertex_Path_Frame_info.csv', dtype=str)
df_pathframe['Path'] = df_pathframe['Path'].apply(lambda x: f"{int(x):03d}")
df_pathframe['Frame'] = df_pathframe['Frame'].apply(lambda x: f"{int(x):03d}")

# figure out the path of the ascending scene
for fn in os.listdir(folder):
    # filter out extraneous files
    if 'snowline_elev_percentile' in fn and not 'ea' in fn and not 'eos' in fn:
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

        # only use this fn if the direction is Ascending
        if direction == 'Ascending':
            fn_snow = fn

# open the dataframe containing snowline elevations
df_snowline = pd.read_csv(folder + fn_snow, parse_dates=True, index_col=0)

# reindex dataframe so it contains all dates
all_dates = pd.date_range(df_snowline.index[0], df_snowline.index[-1])
df_snowline = df_snowline['snowline_elev_min_m'].reindex(all_dates).ffill()

# get data filepaths
albedo_fp = base_fp + '../albedo/'
num = str(rgi7id[3:])
use_list = ['s2','l8','l9']
albedo_fns = [albedo_fp + f'{num}/{num}_{data}.nc' for data in use_list]

# build dataset
ds_all = None
for albedo_fn, dtype in zip(albedo_fns, use_list):
    # open the dataset and get the proper CRS
    ds = xr.open_dataset(albedo_fn)
    crs = ds.spatial_ref.attrs['crs_wkt']
    epsg = crs.split('AUTHORITY["EPSG","')[-1].split('"]')[0]

    ds = ds['albedo'].rio.write_crs(crs).reset_coords(drop=True).to_dataset()
    ds['dtype'] = ('time', np.array([dtype]*len(ds.time.values)).flatten())    
    if ds_all is None:
        ds_all = ds 
    else:
        ds_all = xr.concat([ds_all, ds], dim='time')

# take mean of any duplicate timesteps
ds_all = ds_all.squeeze('band')
ds = ds_all.groupby("time").mean()

# create mask based on elevation for each date that says below/above snowline
# loop through years
# average albedo per elevation band per grid cell or per elevation band? 