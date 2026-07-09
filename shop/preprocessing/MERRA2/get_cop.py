"""
Vibe-coded, but worked for me.

This file downloads all of the COP30 data in chunks
for a region of interest and then generates the
.vrt file which enables them to be accessed altogether.
"""

import os
import subprocess
import urllib.request
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
import pystac_client
import planetary_computer
from tqdm import tqdm

output_dir = "~/local/data/dems/COP30/"

# If your region crosses the antimeridian (180° line), split it into
# two bboxes — one in the Western Hemisphere and one in the Eastern —
# because STAC queries cannot span that boundary.
bboxes = {
    "Western Hemisphere": [-180.0, 51.0, -129.0, 72.0],
    "Eastern Hemisphere": [172.0, 51.0, 180.0, 55.0]
}

def download_tile(url, output_dir):
    """Downloads a signed STAC asset URL using its true native filename."""
    # Cleanly strip away the secure SAS token parameters (?st=...&sig=...)
    parsed_url = urlparse(url)
    clean_path = parsed_url.path  # e.g., '/items/Copernicus_DSM_COG_10_N60_00_W145_00_DEM.tif'

    # Extract the actual native GeoTIFF filename
    filename = os.path.basename(clean_path)
    dest_path = os.path.join(output_dir, filename)

    # Skip if file already exists (allows resuming if interrupted)
    if os.path.exists(dest_path):
        return

    try:
        urllib.request.urlretrieve(url, dest_path)
    except Exception as e:
        print(f"\n[Error] Failed to download {filename}: {e}")


def download_cop30(bboxes, vrt_path, output_dir=output_dir):
    """
    Download COP30 tiles for one or more bounding boxes and build a VRT index.

    Parameters
    ----------
    bboxes : dict[str, list] or list
        Either a single bbox [minx, miny, maxx, maxy], or a dict mapping
        zone names to bboxes (useful when a region straddles the 180° line,
        e.g. Alaska's Aleutians).
    vrt_path : str
        Output path for the .vrt file that indexes all downloaded tiles.
    output_dir : str
        Directory where GeoTIFF tiles are saved.
    """
    if isinstance(bboxes, list):
        bboxes = {'Region': bboxes}

    os.makedirs(output_dir, exist_ok=True)

    # 1. Initialize the STAC catalog client with automatic token signing
    print("Connecting to Planetary Computer STAC API...")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    download_urls = set()

    # 2. Query the STAC API for each zone
    print("Searching for land-intersecting COP30 tiles...")
    for zone_name, bbox in bboxes.items():
        search = catalog.search(
            collections=["cop-dem-glo-30"],
            bbox=bbox
        )
        items = list(search.get_items())
        print(f"  -> Found {len(items)} tiles in {zone_name}")

        for item in items:
            if "data" in item.assets:
                download_urls.add(item.assets["data"].href)

    total_tiles = len(download_urls)
    print(f"\nTotal unique tiles to download: {total_tiles}")

    # 3. Download files concurrently using a thread pool
    print("Starting download pool...")
    # Using 4 workers to stay safe under Microsoft's unsigned rate limits
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(tqdm(
            executor.map(lambda url: download_tile(url, output_dir), download_urls),
            total=total_tiles,
            desc="Downloading DEM Tiles"
        ))

    print(f"\nSuccess! All tiles saved to: {os.path.abspath(output_dir)}")

    # 4. Build a VRT from all downloaded TIFFs so the region can be
    #    accessed as a single virtual mosaic (no data is copied).
    tif_files = sorted(
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.endswith('.tif')
    )
    if not tif_files:
        print("No .tif files found — skipping VRT creation.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(vrt_path)), exist_ok=True)
    subprocess.run(
        ['gdalbuildvrt', vrt_path] + tif_files,
        check=True
    )
    print(f"VRT written to: {os.path.abspath(vrt_path)}")


if __name__ == "__main__":
    download_cop30(
        bboxes=bboxes,
        vrt_path=os.path.expanduser("~/local/data/dems/COP30/COP30_reg01.vrt"),
        output_dir=os.path.expanduser(output_dir),
    )
