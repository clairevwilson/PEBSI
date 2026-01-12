"""
3_get_tile.py

If MERRA-2 data for an entire region was grabbed,
this script will cut out the single grid cell
for an individual glacier of interest and store 
the files in the format expected by PEBSI.

This function pulls from the Zarr store
developed in steps 1-2.

@author: clairevwilson
"""

# Imports
import xarray as xr 
import os
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('-roi', action='store', type=str, default='reg01',
                    help='Name of region of interest in zarr store to access')
parser.add_argument('-fp_out', action='store', type=str, default=None,
                    help='Path to the folder where data will be stored')
parser.add_argument('-fp_in', action='store', type=str, default='../MERRA-2/zarr_store/',
                    help='Path to the Zarr warehouse folder')
parser.add_argument('-lat', action='store', type=str, default='63.5',
                    help='Latitude of the grid cell to access')
parser.add_argument('-lon', action='store', type=str, default='-145.625',
                    help='Longitude of the grid cell to access')
parser.add_argument('-start', action='store', type=str, default=None,
                    help='Start timestamp to clip the data')
parser.add_argument('-end', action='store', type=str, default=None,
                    help='End timestamp to clip the data')
args = parser.parse_args()

def pull_grid_cell(fp_out, fp_zarr, cenlat, cenlon, 
                   roi, start=None, end=None):
    """
    Function which grabs all the variables
    for a given grid cell in the MERRA-2 data
    and moves them to a user-supplied fp_out 
    for easy transfer to a home computer or
    supercomputer.

    If cenlat/cenlon is not an exact match,
    the function will take the nearest cell.

    Parameters
    ----------
    fp_out : str
        Folder to save all the data
    cenlat / cenlon : float or str
        Center latitude and longitude 
        of the grid cell to access
    start / end : None or timestamp
        If supplied, the dataset will
        be clipped to these timestamps.
    """
    # Ensure output folder exists
    assert fp_out is not None, '! Input a folder in -fp_out'
    os.makedirs(fp_out, exist_ok=True)

    # Loop through all variable stores
    for var in os.listdir(fp_zarr):
        zarr_path = os.path.join(fp_zarr, var, f'{var}_{roi}.zarr')
        assert os.path.exists(zarr_path), f'! Missing zarr for {var}'

        # Open the Zarr store lazily
        ds = xr.open_zarr(zarr_path, consolidated=True)

        # Select nearest grid cell
        ds_cell = ds.sel(lat=float(cenlat), lon=float(cenlon), method='nearest')

        # Clip by time if requested
        if start is not None or end is not None:
            ds_cell = ds_cell.sel(time=slice(start, end))

        # Save to NetCDF file
        out_file = os.path.join(fp_out, f'{var}_{cenlat}_{cenlon}.nc')
        ds_cell.to_netcdf(out_file)

        print(f'Stored {var} as {out_file}')

if __name__ in '__main__':
    pull_grid_cell(
        fp_out = args.fp_out, fp_zarr = args.fp_in,
        cenlat = args.lat, cenlon = args.lon,
        start = args.start, end = args.end,
        roi = args.roi,
    )