import sys
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sys.path.append('project')
from data_handling import MassBalance, translate_rgi

model_fp = sys.argv[1] if len(sys.argv) > 1 else '/ocean/projects/ees260009p/cwilson4/Output/controlled/base_2/output.zarr'
ds_base = xr.open_zarr(model_fp)

# sites = {'gulkana': ['A', 'AU', 'B', 'D', 'T'],
#          'wolverine': ['N', 'B', 'AU', 'EC'],
#          'lemon_creek': ['A', 'B', 'C', 'D', 'E', 'F'],
#          'kennicott': ['GTH','GTL','KC']}
import pickle 
with open('project/sites.pkl', 'rb') as f:
    sites = pickle.load(f)
sites.pop('01.01390')
rgi_to_name = {v['6']: k for k, v in translate_rgi.items()}
sites = {rgi_to_name.get(k, k): v for k, v in sites.items()}

colors = ['#63c4c7','#fcc02e','#4D559C','#60C252','#BF1F6A']

df_meta = pd.read_csv('data/glacier_metadata.csv', dtype={'rgiid': str})
rgiid_all = ds_base['rgiid'].values
lat_all = ds_base['lat'].values
lon_all = ds_base['lon'].values
elev_all = ds_base['elev'].values

rows = []
for glacier, site_list in sites.items():
    rgiid = translate_rgi[glacier]['6']
    glac_points = np.where(rgiid_all == rgiid)[0]
    for site in site_list:
        meta = df_meta.loc[(df_meta['rgiid'] == rgiid) & (df_meta['site'] == site)].iloc[0]
        dist = np.hypot(lat_all[glac_points] - meta['lat'],
                        (lon_all[glac_points] - meta['lon']) * np.cos(np.deg2rad(meta['lat'])))
        point = glac_points[np.argmin(dist)]

        mb = MassBalance(glacier, site, use='annual')
        has_total_mass = 'total_mass' in ds_base
        ds_point = ds_base.isel(point=point)[['mass_balance', 'total_mass'] if has_total_mass else ['mass_balance']].load()
        mb.get_model_mb(ds_point)
        if np.all(np.isnan(mb.meas)):
            print(f'{glacier} {site}: no overlapping annual data')
            continue

        # periods after a point runs out of ice have no meaningful melt
        zero_mass = np.where(ds_point['total_mass'].values < 0.1)[0] if has_total_mass else []
        if len(zero_mass) > 0:
            has_ice = mb.period_ends < ds_point.time.values[zero_mass[0]]
            mb.mod = mb.mod[has_ice]
            mb.meas = mb.meas[has_ice]
            mb.period_ends = mb.period_ends[has_ice]
            if len(mb.meas) == 0:
                print(f'{glacier} {site}: melted out before first observation period')
                continue

        valid = np.isin(mb.df['fall_date'].pipe(pd.to_datetime, format='mixed').values, mb.period_ends)
        obs_elev = mb.df.loc[valid, 'elevation'].mean()
        rows.append({'glacier': glacier, 'site': site, 'point': point,
                     'dist_km': np.min(dist) * 111,
                     'obs_elev': obs_elev, 'mod_elev': elev_all[point],
                     'n_years': len(mb.meas),
                     'obs_ba': np.mean(mb.meas), 'mod_ba': np.mean(mb.mod),
                     'bias': np.mean(mb.mod - mb.meas)})

df = pd.DataFrame(rows)
pd.options.display.float_format = '{:.2f}'.format
print(df.to_string(index=False))

print('\nGradients (m w.e. per 100 m) and mean bias:')
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for (glacier, df_glac), color in zip(df.groupby('glacier', sort=False), colors):
    grad_obs = np.polyfit(df_glac['obs_elev'], df_glac['obs_ba'], 1)[0] * 100
    grad_mod = np.polyfit(df_glac['mod_elev'], df_glac['mod_ba'], 1)[0] * 100
    print(f'{glacier}: obs {grad_obs:.2f}, mod {grad_mod:.2f}, mean bias {df_glac["bias"].mean():.2f} m w.e.')

    axes[0].plot(df_glac['obs_ba'], df_glac['obs_elev'], 'o-', c=color, label=f'{glacier} obs')
    axes[0].plot(df_glac['mod_ba'], df_glac['mod_elev'], 's--', c=color, label=f'{glacier} mod')
    axes[1].plot(df_glac['bias'], df_glac['obs_elev'], 'o-', c=color, label=glacier)
    for _, row in df_glac.iterrows():
        axes[1].annotate(row['site'], (row['bias'], row['obs_elev']), fontsize=8,
                         xytext=(4, 2), textcoords='offset points')

axes[0].set_xlabel('Mean annual mass balance (m w.e.)')
axes[0].set_ylabel('Elevation (m)')
axes[0].legend(fontsize=7)
axes[1].axvline(0, c='k', lw=0.8)
axes[1].set_xlabel('Model - observed annual mass balance (m w.e.)')
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig('index_site_mb_bias.png', dpi=150)
