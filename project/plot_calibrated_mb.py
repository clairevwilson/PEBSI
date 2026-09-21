from data_handling import MassBalance, translate_rgi, plt
import numpy as np
import pandas as pd
import xarray as xr
import os
import pickle
import dask

dask.config.set(scheduler='synchronous')

with open('sites.pkl', 'rb') as f:
    sites_by_rgiid = pickle.load(f)

rgi_map = {}
for glacier, numbers in translate_rgi.items():
    rgi_map[numbers['6']] = glacier

ROOT = '/ocean/projects/ees260009p/cwilson4/Output/AD_optimize_2/'
ds_all = xr.open_zarr(ROOT + 'best.zarr', consolidated=False)

site_meta = pd.read_csv('/jet/home/cwilson4/PEBSI/data/glacier_metadata.csv', dtype=str)
site_meta['lat'] = site_meta['lat'].astype(float)
site_meta['lon'] = site_meta['lon'].astype(float)

rgiid_all = ds_all['rgiid'].values
lats_all = ds_all['lat'].values
lons_all = ds_all['lon'].values

plot_fp = ROOT + 'mb_plots/'
os.makedirs(plot_fp, exist_ok=True)

for rgiid, sites in sites_by_rgiid.items():
    glacier_idx = np.where(rgiid_all == rgiid)[0]
    glacier_lats = lats_all[glacier_idx]
    glacier_lons = lons_all[glacier_idx]

    for site in sites:
        row = site_meta.loc[(site_meta['rgiid'] == rgiid) & (site_meta['site'] == site)]
        if row.empty:
            continue
        site_lat, site_lon = float(row['lat'].iloc[0]), float(row['lon'].iloc[0])

        # nearest point in degree space, lon scaled by cos(lat) since a
        # degree of longitude is shorter than a degree of latitude here
        coslat = np.cos(np.radians(site_lat))
        dist = (glacier_lats - site_lat)**2 + ((glacier_lons - site_lon) * coslat)**2
        idx = glacier_idx[np.argmin(dist)]

        mb = MassBalance(rgi_map[rgiid], site)
        ds = ds_all.isel(point=idx)[['mass_balance']]
        mb.get_model_mb(ds)

        fig, ax = mb.plot_mb(savefig=f'{plot_fp}{rgiid}_{site}.png')
        plt.close(fig)
