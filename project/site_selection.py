"""
Site selection of USGS Benchmark Glacier sites with
>= 5 years of seasonal mass balance data.
"""
import pandas as pd
import pickle 
import numpy as np
import os
from data_handling import MassBalance, translate_rgi

benchmark_fp = '/ocean/projects/ees260009p/cwilson4/MB_data/'
benchmark_glaciers = [f.lower() for f in os.listdir(benchmark_fp) if not '.' in f]

all_sites = {} 
n_tot = 0

df_meta = pd.read_csv('../data/glacier_metadata.csv', dtype=str)
site_locs = pd.read_csv(benchmark_fp + 'Glacier_Mass_Balance_Data_Sites.csv')

add_rows = []

for glacier in benchmark_glaciers:
    if glacier in ['southcascade','sperry','point','stakes']:
        continue 
    if glacier == 'lemoncreek':
        glacier = 'lemon_creek'
    
    rgiid = translate_rgi[glacier]['6']
    all_sites[rgiid] = []

    name_parts = [f.capitalize() for f in glacier.split('_')]
    name_fmtd = ''
    for f in name_parts:
        name_fmtd += f

    data_fn = benchmark_fp + f'{name_fmtd}/Input_{name_fmtd}_Glaciological_Data.csv'
    assert os.path.exists(data_fn), f'benchmark data was not found for {glacier}'
    
    df = pd.read_csv(data_fn, parse_dates=True)
    df_glac = df_meta.loc[df_meta['rgiid'] == rgiid]
    
    for site in np.unique(df['site_name']):
        try:
            mb = MassBalance(glacier, site, min_n_winter = 3)
            assert len(mb.period_starts) >= 10
            assert mb.dataset == 'seasonal'

            all_sites[rgiid].append(site)
            n_tot += 1

            if site not in df_glac['site'].values:
                glac_loc = site_locs.loc[site_locs['Glacier'] == name_fmtd]
                site_lat = glac_loc.loc[glac_loc['site_name'] == site, 'latitude'].values[0]
                site_lon = glac_loc.loc[glac_loc['site_name'] == site, 'longitude'].values[0]
                elev = df.loc[df['site_name'] == site, 'elevation'].mean()

                add_rows.append({'rgiid': rgiid, 'site': site, 
                    'lat': site_lat, 'lon': site_lon,
                    'elevation': elev,
                })
                
            a_ice = df_glac.loc[df_glac['site'] == site, 'a_ice'].values[0]

        except:
            continue 

    if len(all_sites[rgiid]) > 0:
        print(f'{glacier:<20} {len(all_sites[rgiid])} valid sites {all_sites[rgiid]}')

new_df = pd.DataFrame(add_rows)
df_complete = pd.concat([df_meta, new_df], ignore_index=True).drop(columns=['Unnamed: 0'])
df_complete.to_csv('../data/glacier_metadata.csv')

print(f'Found {n_tot} valid sites . . .')

with open('sites.pkl', 'wb') as f:
    pickle.dump(all_sites, f)
print('Done: stored site dict to sites.pkl.')