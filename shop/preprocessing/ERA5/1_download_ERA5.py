"""
Step 1 of ERA5 preprocessing: downloads ERA5-Land hourly 10 m wind (u and v
components) from the Copernicus Climate Data Store for a small bounding box,
one NetCDF per year holding the chosen months.

ERA5-Land is on a 0.1 degree grid; the CDS crops it to the "area" box
server-side, so a box a couple of grid cells wide returns just the cells
around the site. Each year is one request (the CDS queue wait per request
dominates the run time), and several years are requested in parallel.

Before running:
- Uses the same CDS login as CARRA-2: ~/.cdsapirc with the CDS API url/key.
- Accept the ERA5-Land licence on the CDS dataset page
  (https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land,
  Download tab -> Terms of use), or requests fail with a licence error.

Files already on disk are skipped, so re-run the script to retry any that failed.
"""
import os
import cdsapi
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================== USER CONFIG ==============================
data_fp = '/home/claire/Research/climate_data/ERA5/'
site = 'gulkana'

# bounding box in degrees (negatives for west longitudes); the Gulkana 1480 AWS
# is at 63.261, -145.410, so this box holds the grid cells around it
lat_min, lat_max = 63.2, 63.3
lon_min, lon_max = -145.5, -145.3

start_year, end_year = 2000, 2005   # inclusive
months = [6, 7, 8]                  # summers only

variables = ['10m_u_component_of_wind', '10m_v_component_of_wind']

workers = 3                         # parallel requests (CDS limits active requests per user)
# ===========================================================================

dataset = 'reanalysis-era5-land'
months_tag = ''.join(f'{m:02d}' for m in months)
filename_template = f'ERA5-Land_wind_{site}_YEAR_m{months_tag}.nc'
os.makedirs(data_fp, exist_ok=True)

# check which years are missing
missing = []
for year in range(start_year, end_year + 1):
    fn = data_fp + filename_template.replace('YEAR', str(year))
    if not os.path.exists(fn):
        missing.append((year, fn))

if not missing:
    raise SystemExit('Got all files!')
print(f'Downloading {len(missing)} years', flush=True)


def download_one(year, fn):
    request = {
        'variable': variables,
        'year': str(year),
        'month': [f'{m:02d}' for m in months],
        'day': [f'{d:02d}' for d in range(1, 32)],  # CDS drops days a month doesn't have
        'time': [f'{h:02d}:00' for h in range(24)],
        'area': [lat_max, lon_min, lat_min, lon_max],  # N, W, S, E
        'data_format': 'netcdf',
        'download_format': 'unarchived',
    }
    try:
        cdsapi.Client().retrieve(dataset, request, fn)
        return f'Got {fn}'
    except Exception as e:
        if os.path.exists(fn):
            os.remove(fn)
        return f'Skipping {year} ({e})'


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(download_one, year, fn) for year, fn in missing]
        for future in as_completed(futures):
            print(future.result(), flush=True)

    print('Done. Re-run the script to check for and retry any files that failed.')
