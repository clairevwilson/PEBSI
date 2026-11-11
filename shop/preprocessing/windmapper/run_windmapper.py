"""
Runs Windmapper (patched WindNinja) on a bounding box around each RGI glacier, cropped
from a regional DEM mosaic, and writes one netCDF per glacier in the format PEBSI reads
(params.windmap_fn): spdup(direction, y, x) on a regular lat/lon grid.

Glaciers run in parallel. Each worker process is pinned to its own set of
CORES_PER_JOB cores; Windmapper sizes its internal pool and WindNinja's threads from
the CPU affinity, so this keeps the machine from being oversubscribed.

Run inside the `windmapper` conda env:
    conda activate windmapper
    python run_windmapper.py
"""
import os
import shutil
import subprocess
import multiprocessing as mp
from concurrent import futures
import numpy as np
import geopandas as gpd
import xarray as xr
from shapely.geometry import box
from osgeo import gdal
from tqdm import tqdm

gdal.UseExceptions()

# ============================== USER CONFIG ==============================
dem_fn = '/home/claire/Research/data/dems/COP30/COP30_reg01.vrt'
rgi_fn = '/home/claire/Research/RGI/rgi60/01_rgi60_Alaska/01_rgi60_Alaska.shp'
out_dir = '/home/claire/Research/data/windmapper_v2/'   # {gid}.nc written here
work_dir = '/home/claire/Research/wind_tools/runs/cop30_work/'  # per-glacier scratch, removed on success

wn_exe = '/home/claire/Research/wind_tools/windninja_install/bin/WindNinja_cli'
windmapper_py = '/home/claire/Research/wind_tools/Windmapper/windmapper.py'

# None = every glacier in rgi_fn with min_area_km2 <= Area < max_area_km2, otherwise a list like ['RGI60-01.00570', ...]
# (2-50 km2 batch done 2026-11-10; this setting is the >= 50 km2 batch)
rgi_ids = None
min_area_km2 = 50.0   # ignored when rgi_ids is set
max_area_km2 = np.inf # ignored when rgi_ids is set; larger glaciers need ~2-3 GB each in WindNinja

# Windmapper settings (match the existing data/windmapper files)
res_wind = 100        # WindNinja mesh resolution (m)
ncat = 16             # number of wind directions
targ_res = 50000      # averaging window for the speed-up (m)
dem_res = 30          # resolution of the UTM DEM handed to WindNinja (m)

buffer_deg = 0.15     # WindNinja domain extends this far beyond the glacier's lat/lon bounds
buffer_m = 3000       # output grid extends this far beyond the glacier's bounds (must fit inside the domain)

cores_per_job = 1
n_jobs = 4            # WindNinja takes ~1-3 GB per job (tiles cap it for big glaciers); size n_jobs to free RAM
                      # and run under a memory cap, OOM-killable before anything else, e.g.:
                      #   systemd-run --user --scope -p MemoryMax=14G -p MemorySwapMax=0 \
                      #       choom -n 1000 -- python run_windmapper.py
overwrite = False     # skip glaciers that already have an output file
# ===========================================================================

ATTRS = ['GLIMSId', 'Area', 'Zmin', 'Zmax', 'Aspect', 'Slope', 'Lmax']


def init_worker(slots):
    # pin this worker (and the Windmapper/WindNinja processes it launches) to its own cores
    os.sched_setaffinity(0, slots.get())


def prepare_dem(geom_ll, glac_dir):
    """Crop the regional DEM to a UTM box around the glacier. Returns (utm DEM path, lat/lon output bounds)."""
    glac = gpd.GeoSeries([geom_ll], crs='EPSG:4326')
    utm = glac.estimate_utm_crs()

    # output lat/lon box: glacier bounds + buffer, in metres
    xmin, ymin, xmax, ymax = glac.to_crs(utm).total_bounds
    out_box = gpd.GeoSeries([box(xmin - buffer_m, ymin - buffer_m, xmax + buffer_m, ymax + buffer_m)], crs=utm)
    out_bounds = out_box.to_crs('EPSG:4326').total_bounds

    # WindNinja domain: UTM envelope of the glacier's lat/lon bounds + buffer_deg
    lon0, lat0, lon1, lat1 = geom_ll.bounds
    domain = box(lon0 - buffer_deg, lat0 - buffer_deg, lon1 + buffer_deg, lat1 + buffer_deg)
    te = gpd.GeoSeries([domain], crs='EPSG:4326').to_crs(utm).total_bounds

    dem_utm = os.path.join(glac_dir, 'dem_utm.tif')
    ds = gdal.Warp(dem_utm, dem_fn, dstSRS=utm.to_wkt(), outputBounds=te, xRes=dem_res, yRes=dem_res,
                   resampleAlg='bilinear', outputType=gdal.GDT_Float32)
    band = ds.GetRasterBand(1)
    z = band.ReadAsArray()
    # COP30 ocean is 0; Windmapper masks values <= 0, so treat the sea as a flat 1 m surface
    band.WriteArray(np.maximum(z, 1.0))
    ds = None
    return dem_utm, out_bounds


def run_glacier(row):
    gid = row['RGIId'].replace('RGI60-', '')
    out_fn = os.path.join(out_dir, gid + '.nc')
    glac_dir = os.path.join(work_dir, gid)
    wm_out = os.path.join(glac_dir, 'output') + '/'
    shutil.rmtree(glac_dir, ignore_errors=True)  # leftovers from a failed run (gdal.Warp would write into an old DEM)
    os.makedirs(glac_dir)

    dem_utm, out_bounds = prepare_dem(row['geometry'], glac_dir)

    cfg = os.path.join(glac_dir, 'param.py')
    with open(cfg, 'w') as f:
        f.write(f"wn_exe = '{wn_exe}'\nres_wind = {res_wind}\nncat = {ncat}\n"
                f"use_existing_dem = True\ndem_filename = '{dem_utm}'\n"
                f"wind_average = 'grid'\ntarg_res = {targ_res}\nuser_output_dir = '{wm_out}'\n")
    with open(os.path.join(glac_dir, 'windmapper.log'), 'w') as log:
        subprocess.run(['python', windmapper_py, cfg], cwd=glac_dir, stdout=log, stderr=subprocess.STDOUT,
                       check=True)

    # reproject each direction's speed-up onto a common lat/lon grid
    directions = [int(d) for d in np.arange(0, 360., 360. / ncat)]
    spdup, res = [], None
    for d in directions:
        vrt = os.path.join(wm_out, f'ref-DEM-utm_{d}_spd_up_{targ_res}.vrt')
        kw = {} if res is None else {'xRes': res[0], 'yRes': res[1]}
        ds = gdal.Warp('', vrt, format='MEM', dstSRS='EPSG:4326', outputBounds=out_bounds,
                       resampleAlg='bilinear', dstNodata=np.nan, **kw)
        gt = ds.GetGeoTransform()
        res = (gt[1], -gt[5])
        spdup.append(ds.GetRasterBand(1).ReadAsArray())
    spdup = np.stack(spdup).astype(np.float32)
    assert np.isfinite(spdup).all(), f'{gid}: NaNs in output grid'

    ny, nx = spdup.shape[1:]
    lon = gt[0] + gt[1] * (np.arange(nx) + 0.5)
    lat = gt[3] + gt[5] * (np.arange(ny) + 0.5)

    ds_out = xr.Dataset(
        {'spdup': (('direction', 'y', 'x'), spdup, {'coordinates': 'lat lon'})},
        coords={'direction': ('direction', np.array(directions, dtype=np.int64)),
                'lat': ('y', lat.astype(np.float32)),
                'lon': ('x', lon.astype(np.float32))})
    ds_out.attrs = {'RGIId': row['RGIId'], **{a: str(row[a]) for a in ATTRS},
                    'res_wind_m': res_wind, 'targ_res_m': targ_res, 'ncat': ncat}
    ds_out.to_netcdf(out_fn)

    shutil.rmtree(glac_dir)
    return gid


def main():
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    rgi = gpd.read_file(rgi_fn).to_crs('EPSG:4326')
    if rgi_ids is not None:
        rgi = rgi[rgi['RGIId'].isin(rgi_ids)]
    else:
        rgi = rgi[(rgi['Area'] >= min_area_km2) & (rgi['Area'] < max_area_km2)]
    if not overwrite:
        done = rgi['RGIId'].str.replace('RGI60-', '').apply(lambda g: os.path.exists(os.path.join(out_dir, g + '.nc')))
        rgi = rgi[~done]
    # largest glaciers first so the long jobs don't straggle at the end
    rgi = rgi.sort_values('Area', ascending=False)
    rows = [r for _, r in rgi.iterrows()]
    print(f'Running Windmapper on {len(rows)} glaciers with {n_jobs} jobs x {cores_per_job} cores')

    ctx = mp.get_context('spawn')
    slots = ctx.Queue()
    for i in range(n_jobs):
        slots.put(set(range(i * cores_per_job, (i + 1) * cores_per_job)))

    failed = {}
    with futures.ProcessPoolExecutor(n_jobs, mp_context=ctx, initializer=init_worker, initargs=(slots,)) as ex:
        jobs = {ex.submit(run_glacier, r): r['RGIId'] for r in rows}
        for job in tqdm(futures.as_completed(jobs), total=len(jobs)):
            try:
                job.result()
            except Exception as e:
                failed[jobs[job]] = repr(e)

    if failed:
        print(f'{len(failed)} glaciers failed (scratch dirs + windmapper.log kept in {work_dir}):')
        for gid, err in failed.items():
            print(f'  {gid}: {err}')


if __name__ == '__main__':
    main()
