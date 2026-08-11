"""
Step 1 of PEBSI MERRA-2 preprocessing: downloads MERRA-2 data for a bounding box
and date range, looping over all four datasets (slv, rad, flx, adg).

Before running:
- Download the global sample file (contains geopotential for every grid cell) from
  https://opendap.earthdata.nasa.gov/collections/C1276812819-GES_DISC/granules/M2C0NXASM.5.12.4%3AMERRA2_101.const_2d_asm_Nx.00000000.nc4.dap.nc4?dap4.ce=/PHIS;/time;/lat;/lon
  and save it as MERRA2_constants.nc in data_fp below.
- Set up a NASA EarthData login (https://urs.earthdata.nasa.gov/) and generate a
  bearer token from your profile page, then set it as the EARTHDATA_TOKEN
"""
import os
import re
import threading
import requests
import numpy as np
import pandas as pd
import xarray as xr
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
from urllib.parse import urlparse, unquote
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================== USER CONFIG ==============================
data_fp = '/run/media/claire/TOSHIBA EXT/MERRA2/'
fn_gp = data_fp + 'MERRA2_constants.nc'

# bounding box in degrees (negatives for west longitudes / south latitudes)
lat_min, lat_max = 50, 72           # ALASKA
lon_min, lon_max = -180, -126.5

start_time = '1980-01-01'           # defaults to 00:00 hrs
end_time = '2026-05-01'             # data downloaded up to and not including this date

workers = 8                         # parallel download threads

all_LAPs = False                    # True: adg download includes all LAP species/bins
                                    # False: only bc/oc wet+dry bin 2 and dust wet+dry bin 3

# token = os.environ.get('EARTHDATA_TOKEN')
token = 'eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6ImN2d2lsc29uIiwiZXhwIjoxNzg4MTI4Nzk0LCJpYXQiOjE3ODI5NDQ3OTQsImlzcyI6Imh0dHBzOi8vdXJzLmVhcnRoZGF0YS5uYXNhLmdvdiIsImlkZW50aXR5X3Byb3ZpZGVyIjoiZWRsX29wcyIsImFjciI6ImVkbCIsImFzc3VyYW5jZV9sZXZlbCI6M30.Foe0bx39FdK5LQctmgAK2fRli9ZTNI_i90fvzdBCbzx8nA8fJDLZpfvptnoK3ZuYOrvehE3pSNrunOKX-jz1lV5xpYjUaqgteWzx5JKM53J9T5uMPmE36TxMlktX8a2RrAbras_LJUQzqbQJFCa2rDZffbKWoG3wxnwKMkN8W9NQ4GzJ4b22Cx2isSSIkuQgLxX9NVB8SHPsh80LA6e4ig08j-5XFoz_Myeb6oTC0-cgqmt9-gAtpCxX7CkfrOUluY0QeDz_pelvF83q2EaeRUmOkSDe4Ya7kJdCR9JdYOGAuDP6nmA4KG_In-Qz4twziW9Qat9kL77CAH9tyC4SBw'
if not token:
    raise SystemExit('NASA Earthdata token required: set the EARTHDATA_TOKEN env var.\n'
                      'Generate one at https://urs.earthdata.nasa.gov/profile -> "Generate Token".')

# ===========================================================================

datasets = ['slv', 'rad', 'flx', 'adg']
filename_template = 'MERRA2_VERSION.tavg1_2d_DATASET_Nx.DATE.nc4.nc4'

vars_by_dataset = {'slv': ['T2M','U2M','V2M','PS','QV2M'],
                   'adg': ['OCDP002', 'OCWT002', 'BCDP002', 'BCWT002', 'DUDP003', 'DUWT003'],
                   'rad': ['SWGDN','LWGAB','CLDTOT'],
                   'flx': ['PRECTOTCORR']
}

if all_LAPs:
    vars_by_dataset['adg'] += ['BCDP001', 'OCDP001',
                 'DUDP001', 'DUWT001', 'DUDP002', 'DUWT002', 
                 'DUDP004', 'DUWT004', 'DUDP005', 'DUWT005']

url_templates = {}

for dataset in datasets:
    str_1 = 'https://goldsmr4.gesdisc.eosdis.nasa.gov/opendap/MERRA2/'
    str_2 = f'M2T1NX{dataset.upper()}.5.12.4'
    str_3 = f'/YEAR/MONTH/MERRA2_VERSION.tavg1_2d_{dataset}_Nx.DATE.nc4.nc4?'

    template = str_1 + str_2 + str_3 
    for var in vars_by_dataset[dataset]:
        template += f'{var}[0:23][Y1:Y2][X1:X2],'
    template += 'time,lat[Y1:Y2],lon[X1:X2]'

    url_templates[dataset] = template

def version(year, new_version=False):
    if year < 1992:
        v = '100'
    elif year <= 2000:
        v = '200'
    elif year <= 2010:
        v = '300'
    else:
        v = '400'
    if new_version:
        v = v[0:2] + '1'
    return v


def safe_filename(url):
    base = urlparse(url).path
    return os.path.basename(unquote(base))


# ------------------------------- build download list -------------------------------

# global sample file gives the lat/lon grid used to find the bounding box indices
ds_gp = xr.open_dataset(fn_gp).drop_dims('time')
lat_min_idx = np.where(ds_gp.lat.values >= lat_min)[0][0]
lat_max_idx = np.where(ds_gp.lat.values <= lat_max)[0][-1]
lon_min_idx = np.where(ds_gp.lon.values >= lon_min)[0][0]
lon_max_idx = np.where(ds_gp.lon.values <= lon_max)[0][-1]
X1, X2, Y1, Y2 = lon_min_idx, lon_max_idx, lat_min_idx, lat_max_idx
print(f'lat bounded by {lat_min_idx}:{lat_max_idx}, lon bounded by {lon_min_idx}:{lon_max_idx}')

downloads = []  # (url, dest_dir) pairs across all datasets
for dataset in datasets:
    dest_dir = os.path.join(data_fp, dataset) + '/'
    os.makedirs(dest_dir, exist_ok=True)

    # check which days are missing
    missing_days = []
    for date in pd.date_range(start_time, end_time):
        v = version(date.year)
        date_fmtd = date.strftime('%Y%m%d')
        date_fn = filename_template.replace('DATE', date_fmtd).replace('DATASET', dataset)
        # file can exist under two different version types (e.g., 400 and 401)
        if not os.path.exists(dest_dir + date_fn.replace('VERSION', v)):
            if not os.path.exists(dest_dir + date_fn.replace('VERSION', version(date.year, True))):
                missing_days.append(date_fmtd)

    print(f'[{dataset}] {len(missing_days)} missing files' if missing_days else f'[{dataset}] all files present')

    for date_fmtd in missing_days:
        url = url_templates[dataset]
        url = url.replace('YEAR', date_fmtd[:4]).replace('MONTH', date_fmtd[4:6]).replace('DATE', date_fmtd)
        url = url.replace('VERSION', version(int(date_fmtd[:4])))
        url = url.replace('X1', str(X1)).replace('X2', str(X2)).replace('Y1', str(Y1)).replace('Y2', str(Y2))
        downloads.append((url, dest_dir))

if not downloads:
    raise SystemExit('Got all files!')
print(f'Downloading {len(downloads)} files total across {len(datasets)} datasets')

# ----------------------------- download (crash-protected) -----------------------------

_local = threading.local()


def get_session():
    if not hasattr(_local, 'session'):
        s = requests.Session()
        s.headers.update({'Authorization': f'Bearer {token}'})
        retry = Retry(total=2, backoff_factor=1, status_forcelist=[500, 502, 503, 504],
                      allowed_methods=['GET'])
        s.mount('https://', HTTPAdapter(max_retries=retry))
        s.mount('http://', HTTPAdapter(max_retries=retry))
        _local.session = s
    return _local.session


def _version_candidates(url):
    yield url
    match = re.search(r'_(\d{3})', url)
    if match:
        old_version = int(match.group(1))
        yield url.replace(match.group(1), str(old_version + 1), 1)


def download_one(url, dest_dir):
    session = get_session()
    try:
        for candidate in _version_candidates(url):
            filename = os.path.join(dest_dir, safe_filename(candidate))
            with session.get(candidate, stream=True, timeout=(10, 120)) as response:
                if response.status_code == 200:
                    with open(filename, 'wb') as out_file:
                        for chunk in response.iter_content(chunk_size=1 << 20):
                            out_file.write(chunk)
                    return None
                elif response.status_code == 404:
                    continue
                else:
                    return f'{candidate} (HTTP {response.status_code})'
    except Exception as e:
        return f'{url} (error: {e})'
    return url  # signal skip


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_one, url, dest_dir): url for url, dest_dir in downloads}
        with tqdm(total=len(downloads), desc='Downloading MERRA-2 files', unit='file') as pbar:
            for future in as_completed(futures):
                skipped = future.result()
                if skipped:
                    tqdm.write(f'Skipping {skipped}, no valid version found.')
                pbar.update(1)

    print('Done. Re-run the script to check for and retry any files that failed.')
