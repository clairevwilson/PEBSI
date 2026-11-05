import os
import pickle
import numpy as np
import pandas as pd
import xarray as xr

from project.data_handling import MassBalance, translate_rgi

ROOT = '/ocean/projects/ees260009p/cwilson4/Output/AD_result_1/'
YEAR = 2004

with open('project/sites.pkl', 'rb') as f:
    sites_by_rgiid = pickle.load(f)
rgi_to_name = {v['6']: k for k, v in translate_rgi.items()}
df_meta = pd.read_csv('data/glacier_metadata.csv', dtype={'rgiid': str})

ds_all = xr.open_zarr(ROOT + 'output.zarr', consolidated=False)
rgiid_all = ds_all['rgiid'].values
lat_all = ds_all['lat'].values
lon_all = ds_all['lon'].values

rows = []
for rgiid, site_list in sites_by_rgiid.items():
    glac_points = np.where(rgiid_all == rgiid)[0]
    if len(glac_points) == 0:
        continue
    glacier = rgi_to_name[rgiid]
    for site in site_list:
        meta = df_meta.loc[(df_meta['rgiid'] == rgiid) & (df_meta['site'] == site)]
        if meta.empty:
            continue
        lat, lon = meta['lat'].iloc[0], meta['lon'].iloc[0]
        dist = np.hypot(lat_all[glac_points] - lat,
                        (lon_all[glac_points] - lon) * np.cos(np.deg2rad(lat)))
        point = glac_points[np.argmin(dist)]

        mb = MassBalance(glacier, site)
        if mb.dataset != 'seasonal':
            continue
        mb.get_model_mb(ds_all.isel(point=point)[['mass_balance']].load())

        idx = list(mb.idx_summer)
        starts = pd.to_datetime(mb.period_starts[idx])
        meas, mod = mb.meas[idx], mb.mod[idx]
        in_year = starts.year == YEAR
        if not in_year.any():
            continue
        others = ~in_year & ~np.isnan(meas)
        rank_meas = (meas[others] < meas[in_year][0]).sum() + 1
        rank_mod = (mod[others] < mod[in_year][0]).sum() + 1
        rows.append({'glacier': glacier, 'site': site,
                     'meas_2004': meas[in_year][0], 'mod_2004': mod[in_year][0],
                     'meas_mean': meas[others].mean(), 'mod_mean': mod[others].mean(),
                     'meas_rank': f'{rank_meas}/{others.sum() + 1}',
                     'mod_rank': f'{rank_mod}/{others.sum() + 1}'})

df = pd.DataFrame(rows)
df['meas_anom'] = df['meas_2004'] - df['meas_mean']
df['mod_anom'] = df['mod_2004'] - df['mod_mean']
pd.set_option('display.width', 200)
print(f'Summer {YEAR} index-site balance (m w.e.); rank 1 = most negative summer on record')
print(df.round(2).to_string(index=False))
