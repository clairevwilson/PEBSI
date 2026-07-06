"""
One-off fix script: converts any slv zarr stores written in zarr_format=3 back to
zarr_format=2, and rechunks any stores with the wrong chunk shape to (8760, 20, 16).
Run this before re-running 2_aggregate_MERRA2.py to create the missing QV2M store.
"""
import os
import zarr
import xarray as xr

fp_zarr_store = '/Volumes/TOSHIBA EXT/MERRA2/zarr_store/'
roi = 'reg01'

# only fixing slv; other datasets look fine
slv_vars = ['T2M', 'U2M', 'V2M', 'QV2M', 'PS']

TARGET_CHUNKS = (8760, 20, 16)
TARGET_FORMAT = 2


def fix_store(var):
    path = os.path.join(fp_zarr_store, var, f'{var}_{roi}.zarr')
    if not os.path.exists(path):
        print(f'{var}: store missing — skipping (re-run 2_aggregate_MERRA2.py to create it)')
        return

    z = zarr.open_group(path, mode='r')
    arr = z[var]
    current_chunks = tuple(arr.chunks)
    current_format = arr.metadata.zarr_format

    needs_fix = (current_chunks != TARGET_CHUNKS) or (current_format != TARGET_FORMAT)
    print(f'{var}: chunks={current_chunks}, zarr_format={current_format} -> ', end='')

    if not needs_fix:
        print('OK, no changes needed')
        zarr.consolidate_metadata(path)
        return

    # find a non-colliding backup name
    backup = os.path.join(fp_zarr_store, var, f'{var}_{roi}_backup_0.zarr')
    ii = 0
    while os.path.exists(backup):
        ii += 1
        backup = os.path.join(fp_zarr_store, var, f'{var}_{roi}_backup_{ii}.zarr')

    print(f'fixing (backup -> {os.path.basename(backup)})')
    os.rename(path, backup)

    ds = xr.open_zarr(backup, consolidated=False)
    ds = ds.chunk({'time': TARGET_CHUNKS[0], 'lat': TARGET_CHUNKS[1], 'lon': TARGET_CHUNKS[2]})
    for v in ds.variables:
        ds[v].encoding.clear()

    ds.to_zarr(path, mode='w', consolidated=True, zarr_format=TARGET_FORMAT,
               encoding={var: {'chunks': TARGET_CHUNKS}})
    print(f'{var}: done')


if __name__ == '__main__':
    for var in slv_vars:
        fix_store(var)
    print('\nAll done. Re-run 2_aggregate_MERRA2.py to create the missing QV2M store.')
