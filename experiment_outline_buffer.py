"""
Tests whether an elevation-gated buffer (mask pixels below Zmed that sit
near the RGI boundary) actually recovers real glacier retreat, using
USGS's own remeasured outlines for the three benchmark glaciers as
ground truth against the ~2000-era RGI outline the model currently uses.

For each glacier: rasterize RGI-minus-most-recent-USGS (the "should be
masked" area) onto the ice albedo grid, alongside each pixel's DEM
elevation and its distance to the RGI boundary. Then scores the
elevation < Zmed" + "N-pixel buffer" rule against that ground truth at
a few buffer widths, so the width can be picked by evidence rather than
guessed.

    python experiment_outline_buffer.py
"""
import geopandas as gpd
import numpy as np
import rioxarray as rxr
import shapely
from scipy.spatial import cKDTree

from host_paths import HOST_PATHS, host
from pebsi.io import mesh

GLACIERS = {
    'gulkana':     ('01.00570', 'Gulkana Glacier'),
    'wolverine':   ('01.09162', 'Wolverine Glacier'),
    'lemon_creek': ('01.01104', 'LemonCreek Glacier'),
}
BUFFER_WIDTHS_PX = [1, 2, 3]

p = HOST_PATHS[host]
usgs = gpd.read_file('/ocean/projects/ees260009p/cwilson4/data/outlines/USGS_Glacier_Boundaries_shp')
rgi_region_gdf = gpd.read_file(p['rgi_fp'] + '../01_rgi60_Alaska/01_rgi60_Alaska.shp')

for name, (gid, usgs_name) in GLACIERS.items():
    print('=' * 78)
    print(name.upper())
    print('=' * 78)

    rgi_poly, crs = mesh.glacier_polygon(rgi_region_gdf, gid)

    sub = usgs.loc[usgs['Glacier'] == usgs_name].sort_values('Year')
    latest = sub.iloc[-1]
    print(f'USGS outline: {usgs_name}, year {latest["Year"]}, '
          f'RGI area {rgi_poly.area/1e6:.2f} km2 vs USGS {latest["Area"]:.2f} km2')

    usgs_poly = gpd.GeoSeries([latest.geometry], crs=usgs.crs).to_crs(crs).iloc[0]
    usgs_poly = usgs_poly.buffer(0)

    retreated = rgi_poly.difference(usgs_poly)
    print(f'retreated area: {retreated.area/1e6:.3f} km2 '
          f'({100*retreated.area/rgi_poly.area:.1f}% of RGI area)')

    # DEM for elevation and Zmed
    g = rgi_region_gdf.loc[rgi_region_gdf['RGIId'] == 'RGI60-' + gid]
    dem = rxr.open_rasterio(p['cop30_vrt_path']).squeeze().drop_vars('band')
    b = g.to_crs(dem.rio.crs).total_bounds
    dem = dem.rio.clip_box(b[0] - .02, b[1] - .02, b[2] + .02, b[3] + .02)
    dem = dem.rio.reproject(crs)
    if dem.rio.nodata is not None:
        dem = dem.where((dem != dem.rio.nodata) & np.isfinite(dem))

    # ice albedo grid: this is the raster actually being filtered
    alb = rxr.open_rasterio(p['ice_albedo_fn'].format(gid=gid)).squeeze().drop_vars('band')
    alb = alb.rio.reproject(crs) if alb.rio.crs != crs else alb
    res = abs(alb.rio.resolution()[0])
    ax, ay = np.meshgrid(alb.x.values, alb.y.values)
    on_rgi = shapely.contains_xy(rgi_poly, ax, ay)

    # elevation per albedo pixel, nearest DEM sample
    dgx, dgy = np.meshgrid(dem.x.values, dem.y.values)
    dem_pts = np.column_stack([dgx.ravel(), dgy.ravel()])
    dem_vals = dem.values.ravel()
    valid_dem = np.isfinite(dem_vals)
    tree = cKDTree(dem_pts[valid_dem])
    alb_pts = np.column_stack([ax[on_rgi], ay[on_rgi]])
    nn = tree.query(alb_pts, workers=-1)[1]
    elev = dem_vals[valid_dem][nn]

    zmed = float(g['Zmed'].item())
    is_retreated = shapely.contains_xy(retreated, ax[on_rgi], ay[on_rgi])

    boundary = rgi_poly.exterior
    dist_to_edge = shapely.distance(
        shapely.points(ax[on_rgi], ay[on_rgi]), boundary)

    n = on_rgi.sum()
    print(f'{n} albedo pixels on-ice, resolution {res:.0f} m, Zmed={zmed:.0f} m')
    print(f'truth: {is_retreated.sum()} pixels ({100*is_retreated.mean():.1f}%) '
          f'actually retreated per USGS')
    below_zmed = elev < zmed
    print(f'of those, {is_retreated[below_zmed].sum()} ({100*is_retreated[below_zmed].mean() if below_zmed.any() else 0:.1f}% '
          f'of below-Zmed pixels) below Zmed vs '
          f'{is_retreated[~below_zmed].sum()} ({100*is_retreated[~below_zmed].mean() if (~below_zmed).any() else 0:.1f}% '
          f'of above-Zmed pixels) above Zmed')

    print()
    print(f'{"buffer (px)":>12} {"predicted":>10} {"precision":>10} {"recall":>8} {"F1":>6}')
    for w in BUFFER_WIDTHS_PX:
        pred = dist_to_edge <= w * res
        tp = (pred & is_retreated).sum()
        fp = (pred & ~is_retreated).sum()
        fn = (~pred & is_retreated).sum()
        precision = tp / max(pred.sum(), 1)
        recall = tp / max(is_retreated.sum(), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        print(f'{w:>12} {pred.sum():>10} {precision:>10.3f} {recall:>8.3f} {f1:>6.3f}')
    print()
