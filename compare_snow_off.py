import sys
import xarray as xr
import pandas as pd
import numpy as np
sys.path.append('project')
from data_handling import translate_rgi
from glacierwide_loss import SnowlineMelt

model_fp = sys.argv[1] if len(sys.argv) > 1 else '/ocean/projects/ees260009p/cwilson4/Output/AD_optimize_2/best.zarr'
ds_model = xr.open_zarr(model_fp)

sites = {'gulkana': ['A', 'AU', 'B', 'D', 'T'],
         'wolverine': ['N', 'B', 'AU', 'EC'],
         'lemon_creek': ['A', 'B', 'C', 'D', 'E', 'F']}

df_meta = pd.read_csv('data/glacier_metadata.csv', dtype={'rgiid': str})
rgiid_all = ds_model['rgiid'].values
lat_all = ds_model['lat'].values
lon_all = ds_model['lon'].values
elev_all = ds_model['elev'].values

def snow_off_dates(snow, time):
    """
    First date each melt season (Apr-Oct) that the site is snow-free
    on two consecutive acquisitions. NaT if snow never clears.
    Acquisitions with no retrieval (NaN) are skipped.
    """
    time = pd.to_datetime(time)
    snow = np.asarray(snow, dtype=float)
    dates = {}
    for year in np.unique(time.year):
        idx = np.where((time.year == year) & (time.month >= 4) & (time.month <= 10)
                       & ~np.isnan(snow))[0]
        if len(idx) < 2 or not snow[idx[0]]:
            continue
        dates[year] = pd.NaT
        for i, j in zip(idx[:-1], idx[1:]):
            if not snow[i] and not snow[j]:
                dates[year] = time[i]
                break
    return dates

rows = []
for glacier, site_list in sites.items():
    rgiid = translate_rgi[glacier]['6']
    glac_points = np.where(rgiid_all == rgiid)[0]
    points = []
    for site in site_list:
        meta = df_meta.loc[(df_meta['rgiid'] == rgiid) & (df_meta['site'] == site)].iloc[0]
        dist = np.hypot(lat_all[glac_points] - meta['lat'],
                        (lon_all[glac_points] - meta['lon']) * np.cos(np.deg2rad(meta['lat'])))
        points.append(glac_points[np.argmin(dist)])

    if 'surftype' in ds_model:
        ds_sites = ds_model.isel(point=points)[['surftype', 'total_water', 'lat', 'lon']].load()
    else:
        # no surface type stored: bare ice sits at each point's constant minimum albedo
        ds_sites = ds_model.isel(point=points)[['albedo', 'total_water', 'lat', 'lon']].load()
        ice_albedo = ds_sites['albedo'].min(dim='time')
        ds_sites['surftype'] = xr.where(ds_sites['albedo'] > ice_albedo + 0.02, 0, 2)
    snowline = SnowlineMelt(glacier)
    snowline.get_model_snow(ds_sites)

    for i, (site, point) in enumerate(zip(site_list, points)):
        mod_dates = snow_off_dates(snowline.mod_snow[:, i], snowline.time)
        meas_dates = snow_off_dates(snowline.meas_snow[:, i], snowline.time)
        for year in sorted(set(mod_dates) & set(meas_dates)):
            rows.append({'glacier': glacier, 'site': site, 'elev': elev_all[point],
                         'year': year, 'meas': meas_dates[year], 'mod': mod_dates[year]})

df = pd.DataFrame(rows)
df['diff_days'] = (df['mod'] - df['meas']).dt.days
df.to_csv('snow_off_comparison.csv', index=False)

summary = df.groupby(['glacier', 'site'], sort=False).agg(
    elev=('elev', 'first'),
    n_years=('year', 'count'),
    meas_doy=('meas', lambda x: x.dt.dayofyear.mean()),
    mod_doy=('mod', lambda x: x.dt.dayofyear.mean()),
    meas_never_clear=('meas', lambda x: x.isna().sum()),
    mod_never_clear=('mod', lambda x: x.isna().sum()),
    mean_diff_days=('diff_days', 'mean'),
    median_diff_days=('diff_days', 'median'))
pd.options.display.float_format = '{:.1f}'.format
print(summary.to_string())
