import os
import urllib.request
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
import pystac_client
import planetary_computer
from tqdm import tqdm

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

def download_alaska_cop30(output_dir="~/local/data/dems/COP30/"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Initialize the STAC catalog client with automatic token signing
    print("Connecting to Planetary Computer STAC API...")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )
    
    # 2. Define Alaska's boundaries, split cleanly at the 180° line
    alaska_zones = {
        "Mainland & Eastern Aleutians (Western Hemisphere)": [-180.0, 51.0, -129.0, 72.0],
        "Western Aleutians (Eastern Hemisphere)": [172.0, 51.0, 180.0, 55.0]
    }
    
    download_urls = set()
    
    # 3. Query the STAC API for the global 30m collection
    print("Searching for land-intersecting COP30 tiles...")
    for zone_name, bbox in alaska_zones.items():
        search = catalog.search(
            collections=["cop-dem-glo-30"],
            bbox=bbox
        )
        items = list(search.get_items())
        print(f"  -> Found {len(items)} tiles in {zone_name}")
        
        for item in items:
            if "data" in item.assets:
                # 'data' points directly to the Cloud-Optimized GeoTIFF (.tif)
                download_urls.add(item.assets["data"].href)
                
    total_tiles = len(download_urls)
    print(f"\nTotal unique tiles to download: {total_tiles}")
    
    # 4. Download files concurrently using a thread pool
    print("Starting download pool...")
    # Using 4 workers to stay safe under Microsoft's unsigned rate limits
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(tqdm(
            executor.map(lambda url: download_tile(url, output_dir), download_urls), 
            total=total_tiles, 
            desc="Downloading DEM Tiles"
        ))
        
    print(f"\nSuccess! All tiles saved to: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    download_alaska_cop30()