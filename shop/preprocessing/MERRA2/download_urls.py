import os
import argparse
import re
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
from urllib.parse import urlparse, unquote
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

data_fp= '/Volumes/TOSHIBA EXT/MERRA2/'
parser = argparse.ArgumentParser()
parser.add_argument('-u', '--url_fn', action='store', nargs='+',
                    help='list of file names containing URLs to download')
parser.add_argument('-t', '--token', action='store',
                    help='NASA Earthdata Bearer token (from urs.earthdata.nasa.gov)')
parser.add_argument('-w', '--workers', type=int, default=4,
                    help='number of parallel download threads (default: 8)')
args = parser.parse_args()

args.token = "eyJ0eXAiOiJKV1QiLCJvcmlnaW4iOiJFYXJ0aGRhdGEgTG9naW4iLCJzaWciOiJlZGxqd3RwdWJrZXlfb3BzIiwiYWxnIjoiUlMyNTYifQ.eyJ0eXBlIjoiVXNlciIsInVpZCI6ImN2d2lsc29uIiwiZXhwIjoxNzg4MTE3OTIwLCJpYXQiOjE3ODI5MzM5MjAsImlzcyI6Imh0dHBzOi8vdXJzLmVhcnRoZGF0YS5uYXNhLmdvdiIsImlkZW50aXR5X3Byb3ZpZGVyIjoiZWRsX29wcyIsImFjciI6ImVkbCIsImFzc3VyYW5jZV9sZXZlbCI6M30.fITLhJ66Tvfcun6s-GTWvxe8regO30n1sYaN6q-Qla-IboM8KmJmOriaxYEWoT94j0oGmpMWUn9lX5u5COLmVQXZdVp9u5DT0uPdoPGu5aRKUoe_EibLt6DIDLN9KLZzDerM5eenaps5w4YwoZQtCI132agDvB-YngFmLE0TAhBjxe250hTGclMetT-Qhx7qGErbAPXMzaKOj0LF-8uaaI5U-Au0azafJuyFMsLFqy8X1UoWwfl5SPs9b1QXUQviQhrRmJ0Y278abQ50BWptq_biongv3uxPbS93N83eXGUZd17PxmyiXGNTh_LE0-1bjFk95SCG7Zoipp1X8CQeKQ"

token = args.token or os.environ.get('EARTHDATA_TOKEN')
if not token:
    raise SystemExit('NASA Earthdata token required: pass -t <token> or set EARTHDATA_TOKEN env var.\n'
                     'Generate one at https://urs.earthdata.nasa.gov/profile → "Generate Token".')

# Thread-local sessions so each thread has its own connection pool
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

url_files = args.url_fn

def safe_filename(url):
    base = urlparse(url).path
    return os.path.basename(unquote(base))

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

def _version_candidates(url):
    yield url
    match = re.search(r'_(\d{3})', url)
    if match:
        old_version = int(match.group(1))
        yield url.replace(match.group(1), str(old_version + 1), 1)

for fn_urls in url_files:
    dataset = fn_urls.split('MERRA2/')[-1][:3]
    dest_dir = data_fp + dataset

    with open(fn_urls, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_one, url, dest_dir): url for url in urls}
        with tqdm(total=len(urls), desc=f'Downloading files from {dataset}', unit='file') as pbar:
            for future in as_completed(futures):
                skipped = future.result()
                if skipped:
                    tqdm.write(f'Skipping {skipped}, no valid version found.')
                pbar.update(1)
