import os
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from data_handling import *
from scipy.stats import gaussian_kde

translate_names = {}
for glacier, number in translate_rgi.items():
    translate_names[number['6']] = glacier

# filepaths
albedo_fp = '../../data/rs/albedo/updated/'
output_fp = '../../Output/gulkana_windfactor_precgrad_0/'
rgi_fp = '../../RGI/rgi60/01_rgi60_Alaska/01_rgi60_Alaska.shp'

# open model dataset for all sites
ds_output = xr.open_zarr(output_fp + 'output.zarr')
ds_output['time'] = ds_output.time - pd.Timedelta(hours=8)

# open RGI
rgi = gpd.read_file(rgi_fp)

all_x, all_y = [], []
for rgi_id in np.unique(ds_output['rgiid'].values):
    name = translate_names[rgi_id]
    
    # get datasets of model and measurement at this glacier
    albedo_glac = Albedo(name)
    glacier_outline = rgi[rgi['RGIId'] == 'RGI60-'+str(rgi_id)]

    idx_id = np.where(ds_output['rgiid'].values == rgi_id)[0]
    points = ds_output.point.values[idx_id]
    model_glac = ds_output.sel(point=points, layer=0)

    # select measurements at the modeled points
    proj = Transformer.from_crs('EPSG:4326', f'EPSG:{albedo_glac.epsg}', always_xy=True)
    x, y = proj.transform(model_glac.lon.values, model_glac.lat.values)
    measured_at_points = (albedo_glac.ds
                 .sel(x=xr.DataArray(x, dims='point'), y=xr.DataArray(y, dims='point'), method='nearest')
                 .drop_duplicates('time'))
    measured = (measured_at_points
                .assign_coords(lon_=('point', model_glac.lon.values), lat_=('point', model_glac.lat.values))
                .set_index(point=['lon_', 'lat_']))

    # select model at the measured times
    modeled = (model_glac
               .sel(time=measured.time, method='nearest')
               .drop_duplicates('time')
               .assign_coords(lon_=('point', model_glac.lon.values), lat_=('point', model_glac.lat.values))
               .set_index(point=['lon_', 'lat_']))

    # gather March means for model and measurement
    march_meas = measured['albedo'].where(measured.time.dt.month == 3, drop=True).dropna(dim='time')
    march_mean_meas = float(march_meas.mean()) if march_meas.sizes['time'] > 0 else float(measured['albedo'].mean())

    march_mod = modeled['albedo'].where(modeled.time.dt.month == 3, drop=True)
    march_mean_mod = float(march_mod.mean()) if march_mod.sizes['time'] > 0 else float(modeled['albedo'].mean())

    # calculate anomoly vs March mean
    meas_anom = measured['albedo'] - march_mean_meas
    mod_anom = modeled['albedo'] - march_mean_mod

    # clip to the same times and align coordinates exactly
    meas_anom = meas_anom.sel(time=mod_anom.time.values, method='nearest').assign_coords(time=mod_anom.time)

    all_x.append(meas_anom.values.flatten())
    all_y.append(mod_anom.values.flatten())

    # calculate error on the grid
    mae = np.abs(meas_anom - mod_anom).mean(dim='time').load()

    outline_proj = glacier_outline.to_crs(f'EPSG:{albedo_glac.epsg}')
    bounds = outline_proj.total_bounds
    dx, dy = bounds[2] - bounds[0], bounds[3] - bounds[1]
    if dx > dy:
        fig_w, fig_h = 8, 8 * dy / dx
    else:
        fig_w, fig_h = 8 * dx / dy, 8

    mae_2d = mae.unstack('point')  # dims: (lon_, lat_)
    lon_2d, lat_2d = np.meshgrid(mae_2d.lon_.values, mae_2d.lat_.values)
    x_2d, y_2d = proj.transform(lon_2d, lat_2d)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    pm = ax.pcolormesh(x_2d, y_2d, mae_2d.values.T, cmap='viridis', shading='nearest')
    fig.colorbar(pm, ax=ax)
    outline_proj.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)
    plt.tight_layout()
    plt.savefig(output_fp + f'{name}_albedo_mae.png', dpi=300)

fig, ax = plt.subplots()

all_x = np.array(all_x).flatten()
all_y = np.array(all_y).flatten()
no_nans = ~np.isnan(all_x) & ~np.isnan(all_y)
all_x = all_x[no_nans]
all_y = all_y[no_nans]

xy = np.vstack([all_x, all_y])
kde = gaussian_kde(xy)
z = kde(xy)

idx = z.argsort()
x, y, z = all_x[idx], all_y[idx], z[idx]

minn, maxx = (-0.8, 0)

ax.scatter(x, y, c=z, cmap='magma')
ax.plot([minn, maxx], [minn, maxx], 'k--')

bias = np.mean(all_y - all_x)
MAE = np.mean(np.abs((all_y - all_x)))
ax.text(0.02, 0.95, f'Bias: {bias:.3f}', transform=ax.transAxes)
ax.text(0.02, 0.90, f'MAE: {MAE:.3f}', transform=ax.transAxes)

ax.set_xlabel('Remotely sensed albedo anomoly')
ax.set_xlim(minn, maxx)

ax.set_ylabel('Modeled albedo anomoly')
ax.set_ylim(minn, maxx)

plt.savefig(output_fp + 'albedo_1to1.png', dpi=300)
plt.show()