"""
Ice albedo tifs for every glacier in data/albedo_cubes (region/sensor/RGI7.nc).

Same method as generate_ice_albedo_tif.py (SAR snowline mask from
data/sar/sar_by_elevation, Wells et al. Zenodo v1.0.0), but with the COP30
DEM and each cube's own outline mask, and processed in row blocks to bound
memory. Glaciers without SAR snowlines are skipped.

Outputs are named by RGI6 ID when the RGI7 glacier matches exactly one
RGI6 outline (centroid-in-polygon, 1:1, area within 25%), else by RGI7 ID.
"""
import os
import glob
import argparse
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
from scipy.ndimage import binary_erosion
from multiprocessing import Pool

base_fp = '/home/claire/Research/'
cube_fp = base_fp + 'data/albedo_cubes/'
output_fp = base_fp + 'data/ice_albedo/'
rgi6_fn = base_fp + 'RGI/rgi60/01_rgi60_Alaska/01_rgi60_Alaska.shp'
rgi7_fn = base_fp + 'RGI/rgi70/RGI2000-v7.0-G-global-attributes.csv'
dem_fn = base_fp + 'data/dems/COP30/COP30_reg01.vrt'
sar_fp = base_fp + 'data/sar/sar_by_elevation/'

# same thresholds as generate_ice_albedo_tif.py
albedo_threshold = 0.55
final_albedo_threshold = 0.45
edge_buffer_px = 1
# dates where SAR snowline and highest ice pixel differ by more than this are thrown out
snowline_threshold = 300
sensors = ['s2', 'l8', 'l9']
# max bytes of albedo held per row block
block_bytes = 256e6


def match_rgi6():
    """RGI7 id -> output name (RGI6 '01.xxxxx' if clean 1:1 match, else RGI7 id)."""
    rgi7 = pd.read_csv(rgi7_fn, dtype={'o1region': str},
                       usecols=['rgi_id', 'o1region', 'area_km2', 'cenlon', 'cenlat'])
    rgi7 = rgi7[rgi7['o1region'] == '01']
    pts = gpd.GeoDataFrame(rgi7, geometry=gpd.points_from_xy(rgi7.cenlon, rgi7.cenlat), crs='EPSG:4326')
    rgi6 = gpd.read_file(rgi6_fn)[['RGIId', 'Area', 'geometry']].to_crs('EPSG:4326')
    joined = gpd.sjoin(pts, rgi6, how='left', predicate='within')
    joined = joined.drop_duplicates('rgi_id', keep=False)  # point in >1 RGI6 polygon
    n_hits = joined['RGIId'].value_counts()
    ok = (joined['RGIId'].notna()
          & joined['RGIId'].map(n_hits).eq(1)
          & ((joined['area_km2'] / joined['Area']).between(0.8, 1.25)))
    names = dict(zip(rgi7['rgi_id'], rgi7['rgi_id']))
    names.update(dict(zip(joined.loc[ok, 'rgi_id'], joined.loc[ok, 'RGIId'].str.replace('RGI60-', ''))))
    return names


def read_sar_snowline(folder):
    """Daily SAR snowline elevation (min over all tracks), as in generate_ice_albedo_tif.py."""
    dfs = [pd.read_csv(folder + fn, parse_dates=True, index_col=0) for fn in os.listdir(folder)
           if 'snowline_elev_percentile' in fn and not 'ea' in fn and not 'eos' in fn]
    df = pd.concat(dfs)
    df['time'] = df.index
    df = df.groupby('time').min()
    all_dates = pd.date_range(df.index[0], df.index[-1])
    return df['snowline_elev_min_m'].reindex(all_dates).ffill()


def process(args):
    fns, out_fn, sar_dir = args
    dss = [xr.open_dataset(fn) for fn in fns]
    ds0 = dss[0]
    for ds in dss[1:]:
        assert np.allclose(ds.x, ds0.x) and np.allclose(ds.y, ds0.y), f'grid mismatch {fns}'
    crs = ds0.spatial_ref.attrs['crs_wkt']
    x, y = ds0.x.values, ds0.y.values
    ny, nx = len(y), len(x)

    # dates across all sensors; duplicates are averaged (as in groupby('time').mean())
    dates = np.concatenate([ds.time.values.astype('datetime64[D]') for ds in dss])
    order = np.argsort(dates, kind='stable')
    _, starts = np.unique(dates[order], return_index=True)

    date_idx = np.repeat(np.arange(len(starts)), np.diff(np.append(starts, len(dates))))

    # rows per block, aligned to the file's y chunking
    ychunk = (ds0.albedo.encoding.get('chunksizes') or (1, 1, 1))[1]
    rows = max(1, int(block_bytes / (len(dates) * nx * 4)) // ychunk) * ychunk
    blocks = [slice(r0, r0 + rows) for r0 in range(0, ny, rows)]

    def read(sl):
        return np.concatenate([ds.albedo.isel(y=sl).values for ds in dss])[order]

    # DEM on the cube grid
    dx, dy = abs(x[1] - x[0]), abs(y[1] - y[0])
    dem = np.full((ny, nx), np.nan, dtype=np.float32)
    with rasterio.open(dem_fn) as src:
        reproject(rasterio.band(src, 1), dem, src_nodata=src.nodata, dst_nodata=np.nan,
                  dst_transform=from_origin(x.min() - dx / 2, y.max() + dy / 2, dx, dy),
                  dst_crs=crs, resampling=Resampling.average)
    if y[0] < y[-1]:  # cube stored south-up
        dem = dem[::-1]
    outline = ds0.shape_mask.values > 0

    # SAR snowline per albedo date (NaN outside the SAR record -> date dropped)
    sar = read_sar_snowline(sar_dir)
    snowline = sar.reindex(pd.to_datetime(dates[order][starts])).values

    def ice(sl, keep_dates):
        # per-scene ice albedo below the snowline, on the glacier
        a = read(sl)
        a[(a <= 0.1) | (a >= 0.9) | (a >= albedo_threshold) | ~np.isfinite(a)
          | ~(dem[sl] <= snowline[date_idx][:, None, None]) | ~outline[sl]
          | ~keep_dates[date_idx][:, None, None]] = np.nan
        return a

    # pass 1: highest ice pixel per date; drop dates far from the SAR snowline
    max_elev = np.full(len(starts), -np.inf)
    for sl in blocks:
        a = ice(sl, np.ones(len(starts), bool))
        has = np.logical_or.reduceat(np.isfinite(a), starts, axis=0)
        max_elev = np.maximum(max_elev, np.where(has, dem[sl], -np.inf).max(axis=(1, 2)))
        del a, has
    keep_dates = np.abs(max_elev - snowline) <= snowline_threshold

    # pass 2: median ice albedo over kept dates
    median = np.full((ny, nx), np.nan, dtype=np.float32)
    for sl in blocks:
        a = ice(sl, keep_dates)
        valid = np.isfinite(a)
        sums = np.add.reduceat(np.where(valid, a, 0), starts, axis=0)
        counts = np.add.reduceat(valid, starts, axis=0)
        del a, valid
        with np.errstate(invalid='ignore'):
            daily = sums / counts
        del sums, counts
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN pixels
            median[sl] = np.nanmedian(daily, axis=0)
        del daily

    # outline mask, eroded by edge_buffer_px to drop likely-retreated margins
    mask = binary_erosion(outline, structure=np.ones((3, 3)), iterations=edge_buffer_px)
    for ds in dss:
        ds.close()
    median[~mask | (median > final_albedo_threshold)] = np.nan

    # crop to outline extent (like rio.clip) and orient north-up
    if y[0] < y[-1]:
        median, y = median[::-1], y[::-1]
    rr, cc = np.where(mask)
    if len(rr) == 0:
        return out_fn, 0, np.nan
    median = median[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
    transform = from_origin(x[cc.min()] - dx / 2, y[rr.min()] + dy / 2, dx, dy)
    with rasterio.open(out_fn, 'w', driver='GTiff', height=median.shape[0], width=median.shape[1],
                       count=1, dtype='float32', crs=crs, transform=transform, nodata=np.nan,
                       compress='deflate', tiled=True) as dst:
        dst.write(median, 1)
    n = int(np.isfinite(median).sum())
    return out_fn, n, float(np.nanmean(median)) if n else np.nan


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_jobs', type=int, default=1)
    parser.add_argument('--limit', type=int, default=None, help='only first N glaciers (testing)')
    parser.add_argument('--largest_first', action='store_true')
    args = parser.parse_args()

    names = match_rgi6()
    tasks, no_sar = [], []
    for s2_fn in glob.glob(cube_fp + '*/s2/*.nc'):
        rgi7id = os.path.basename(s2_fn)[:-3]
        sar_dir = sar_fp + f'01.{rgi7id[-5:]}/'
        if not os.path.isdir(sar_dir):
            no_sar.append(rgi7id)
            continue
        fns = [s2_fn.replace('/s2/', f'/{s}/') for s in sensors]
        fns = [fn for fn in fns if os.path.exists(fn)]
        out_fn = output_fp + f'{names[rgi7id]}_albedo.tif'
        if not os.path.exists(out_fn):
            tasks.append((fns, out_fn, sar_dir, rgi7id, sum(os.path.getsize(f) for f in fns)))
    if args.largest_first:
        tasks.sort(key=lambda t: -t[4])
    tasks = tasks[:args.limit]
    print(f'{len(tasks)} glaciers to process; {len(no_sar)} skipped without SAR: {no_sar}', flush=True)

    log = []
    with Pool(args.n_jobs, maxtasksperchild=20) as pool:
        for i, (out_fn, n, mean) in enumerate(pool.imap_unordered(process, [t[:3] for t in tasks])):
            log.append((os.path.basename(out_fn), n, mean))
            if i % 50 == 0:
                print(f'{i + 1}/{len(tasks)} {os.path.basename(out_fn)} n_px={n} mean={mean:.3f}', flush=True)

    out_to_rgi7 = {os.path.basename(t[1]): t[3] for t in tasks}
    df = pd.DataFrame(log, columns=['file', 'n_ice_px', 'mean_albedo'])
    df.insert(0, 'rgi7_id', df['file'].map(out_to_rgi7))
    log_fn = output_fp + 'cube_ice_albedo_log.csv'
    df.to_csv(log_fn, mode='a', header=not os.path.exists(log_fn), index=False)
    print(f'Done. {(df.n_ice_px == 0).sum()} glaciers had no ice pixels.')
