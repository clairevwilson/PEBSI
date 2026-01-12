import xarray as xr
import numpy as np
import pandas as pd
import os
import argparse
import re
import requests
from tqdm import tqdm
from urllib.parse import urlparse, unquote

data_fp= '../MERRA-2/'
parser = argparse.ArgumentParser()
parser.add_argument('-u', '--url_fn', action='store', nargs='+', 
                    help='list of file names containing URLs to download')
args = parser.parse_args()

# Start downloading session
session = requests.Session()
# session.auth = (args.username, args.password)

# REMAINING FILES TO GET:
# PRECTOTCORR 50 to 80; -180 to -170
url_prectotcorr = '../MERRA-2/flx/url_FIX.txt'
# SWGDN 70 to 80; -170 to -160
url_swgdn = '../MERRA-2/rad/url_FIX.txt'
# all adg 70 to 80; -140 to -130
url_adg = '../MERRA-2/adg/url_FIX.txt'
# QV2M all
url_qv2m = '../MERRA-2/slv/url_FIX.txt'
urls = args.url_fn

# Define function to get good filenames
def safe_filename(url):
    base = urlparse(url).path
    return os.path.basename(unquote(base))

# Define function to check version number
def try_url(url):
    response = session.get(url, stream=True)
    if response.status_code == 200:
        return url
    elif response.status_code == 404:
        match = re.search(r'_(\d{3})', url)
        if match:
            old_version = int(match.group(1))
            new_version = old_version + 1
            new_url = url.replace(str(old_version), str(new_version), 1)
            response2 = session.get(new_url, stream=True)
            if response2.status_code == 200:
                return new_url
    return None

# Loop through url txt files
for fn_urls in urls:
    dataset = fn_urls.split('MERRA-2/')[-1][:3]

    # open urls and create a list
    with open(fn_urls, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]

    # loop through urls with progress bar for downloads
    for url in tqdm(urls, desc=f'Downloading files from {dataset}', unit='file'):
        # get the file name to store to
        filename = os.path.join(data_fp + dataset, safe_filename(url))

        # ensure the url exists
        final_url = try_url(url)
        if final_url:
            # download
            response = requests.get(final_url, stream=True)
            # store
            with open(filename, 'wb') as out_file:
                out_file.write(response.content)
        else:
            tqdm.write(f'Skipping {url}, no valid version found.')