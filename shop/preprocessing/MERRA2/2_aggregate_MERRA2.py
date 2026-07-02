"""
Step 2 of PEBSI MERRA-2 preprocessing: aggregates daily MERRA-2 files into per-variable
zarr stores (one per dataset in dataset_variables), then saves QC plots for each variable.
Files are batched by year and read in parallel via open_mfdataset for speed.
"""
import os
import glob
from collections import defaultdict
import numpy as np
import pandas as pd
import xarray as xr
import zarr
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============================== USER CONFIG ==============================
data_fp = '/run/media/claire/TOSHIBA EXT/MERRA2/'
fp_zarr_store = data_fp + 'zarr_store/'
fp_figs = data_fp + 'Figs/'

# name for the RGI region downloaded
roi = 'reg01'

# bounding box, copied from 1_download_MERRA2.py, used to trim files to a common grid
lat_min, lat_max = 50, 72
lon_min, lon_max = -180, -133.25
# ===========================================================================

dataset_variables = {
    'slv': ['T2M', 'U2M', 'V2M', 'QV2M', 'PS'],
    'adg': ['BCDP002', 'BCWT002', 'OCDP002', 'OCWT002', 'DUDP003', 'DUWT003'],
    'rad': ['SWGDN', 'LWGAB', 'CLDTOT'],
    'flx': ['PRECTOTCORR'],
}


def _file_time(f):
    assert 'Nx.' in f, 'File name is in an unexpected format: manually specify how to pull timestamp from the filename'
    return pd.to_datetime(f.split('Nx.')[-1][:8]) + pd.Timedelta(minutes=30)


def process_files(dataset, data_fp, filetype='.nc4'):
    variables = dataset_variables[dataset]
    daily_files = sorted(glob.glob(data_fp + '*' + filetype))

    # check which times already exist for each var
    times_var = {}
    for var in variables:
        fn_zarr_var = os.path.join(fp_zarr_store, var, f'{var}_{roi}.zarr')
        if os.path.exists(fn_zarr_var):
            times_var[var] = set(pd.to_datetime(xr.open_zarr(fn_zarr_var).time.values))
        else:
            times_var[var] = set()

    # only bother opening files that are missing from at least one var
    file_times = {f: _file_time(f) for f in daily_files}
    new_files = [f for f in daily_files if any(file_times[f] not in times_var[var] for var in variables)]
    if not new_files:
        print(f'{dataset}: nothing new to add')
        return

    # batch by year: one open_mfdataset + one to_zarr write per var per year, instead of
    # one to_zarr call per file (which gets dramatically slower as the store grows).
    # this also bounds how much work is lost if the run gets interrupted.
    files_by_year = defaultdict(list)
    for f in new_files:
        files_by_year[file_times[f].year].append(f)

    n_added = 0
    for year in tqdm(sorted(files_by_year), desc=f'Processing {dataset}', unit='year'):
        year_files = files_by_year[year]
        ds = xr.open_mfdataset(year_files, combine='nested', concat_dim='time', parallel=True, chunks={'time': 24})
        ds = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max)).load()
        n_added += len(year_files)

        for var in variables:
            da = ds[var]
            mask = np.array([t not in times_var[var] for t in pd.to_datetime(da.time.values)])
            if not mask.any():
                continue
            da = da.isel(time=mask)
            times_var[var].update(pd.to_datetime(da.time.values))

            fn_zarr_var = os.path.join(fp_zarr_store, var, f'{var}_{roi}.zarr')
            if os.path.exists(fn_zarr_var):
                da.to_dataset(name=var).to_zarr(fn_zarr_var, mode='a', append_dim='time', consolidated=False)
            else:
                da.to_zarr(fn_zarr_var, mode='w', consolidated=False)
        ds.close()

    print(f'Concatenated {n_added} files to {dataset}_{roi}')

    # resort and dedupe each var, since files can be appended out of order
    for var in dataset_variables[dataset]:
        fn_zarr_var = os.path.join(fp_zarr_store, var, f'{var}_{roi}.zarr')
        fn_zarr_unsorted = os.path.join(fp_zarr_store, var, f'{var}_{roi}_unsorted.zarr')

        # rename the existing unsorted zarr store to a temporary name
        ii = 0
        while os.path.exists(fn_zarr_unsorted.replace('unsorted', f'unsorted_{ii}')):
            ii += 1
        fn_zarr_unsorted = fn_zarr_unsorted.replace('unsorted', f'unsorted_{ii}')
        os.rename(fn_zarr_var, fn_zarr_unsorted)

        # sort and chunk the new dataset
        ds_sorted = xr.open_zarr(fn_zarr_unsorted).sortby('time')
        ds_sorted = ds_sorted.sel(time=~ds_sorted.time.to_index().duplicated())
        ds_sorted = ds_sorted.chunk({'time': 8760, 'lat': 20, 'lon': 16})
        for v in ds_sorted.variables:
            ds_sorted[v].encoding.clear()

        # store sorted zarr and consolidate metadata
        ds_sorted.to_zarr(fn_zarr_var, mode='w', consolidated=False)
        zarr.consolidate_metadata(zarr.DirectoryStore(fn_zarr_var))

    print(f'Resorted and reconsolidated all vars in {dataset}')

# quality control figure: missing data percentage
def plot_qc(var):
    ds = xr.open_zarr(os.path.join(fp_zarr_store, var, f'{var}_{roi}.zarr'))
    da = ds[var]
    n_time = da.sizes['time']

    plt.figure()
    (da.isnull().sum(dim='time') / n_time).plot(cmap='viridis', vmin=0, vmax=1)
    plt.title(f'Data missing percentage over RGI region {roi.split("reg")[-1]}')
    plt.savefig(fp_figs + f'{var}_missing_area.png', dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots()
    is_deposition = any(k in var for k in ('OC', 'BC', 'DU', 'PREC'))
    da_agg = da.sum(dim=['lat', 'lon']) if is_deposition else da.mean(dim=['lat', 'lon'])
    ax.plot(da_agg.time.values, da_agg.values)
    start = pd.to_datetime(da.time.values[0]).strftime('%d %b %Y')
    end = pd.to_datetime(da.time.values[-1]).strftime('%d %b %Y')
    ax.set_ylabel(var)
    ax.set_title(f'{var} timeseries for the Alaska region\n({start} to {end})')
    plt.savefig(fp_figs + f'{var}_full_timeseries.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    os.makedirs(fp_figs, exist_ok=True)

    for dataset in tqdm(dataset_variables, desc='Datasets', unit='dataset'):
        process_files(dataset, data_fp + dataset + '/')

    for dataset, variables in dataset_variables.items():
        for var in tqdm(variables, desc=f'QC plots ({dataset})', unit='var'):
            plot_qc(var)

    print('Done.')
