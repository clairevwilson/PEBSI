"""
Two tests of the claim that the sweep is converging on the DEM grid via
slope variance.

1. Does b actually curve with slope? Losing within-cell variance only
   biases the answer through nonlinearity, so if b is near-linear in
   slope the whole explanation collapses.

2. Is mass balance a clean function of the slope variance captured, and
   does the captured fraction reach 1 at h ~ the DEM resolution? Fits
   both and reports where they land.
"""
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as rxr
import shapely
import xarray as xr
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

from host_paths import HOST_PATHS, host
from pebsi.io import mesh
from pebsi.io.terrain import Terrain
from project.glacierwide_loss import translate_rgi

RESULTS = 'project/point_density_results/'
OUTDIR = '/ocean/projects/ees260009p/cwilson4/Output/point_density_test/'
YEARS = 3.0
FINEST = {'gulkana': 80.0, 'kennicott': 250.0}

p = HOST_PATHS[host]


def terrain(glacier):
    gid = translate_rgi[glacier]['6']
    region = gid.split('.')[0]
    names = [f.split('.')[0] for f in os.listdir(p['rgi_fp'])
             if f.startswith(region)]
    gdf = gpd.read_file(f"{p['rgi_fp']}../{names[0]}/{names[0]}.shp")
    poly, crs = mesh.glacier_polygon(gdf, gid)
    g = gdf.loc[gdf['RGIId'] == 'RGI60-' + gid]

    dem = rxr.open_rasterio(p['cop30_vrt_path']).squeeze().drop_vars('band')
    b = g.to_crs(dem.rio.crs).total_bounds
    dem = dem.rio.clip_box(b[0] - .02, b[1] - .02, b[2] + .02, b[3] + .02)
    dem = dem.rio.reproject(crs)
    if dem.rio.nodata is not None:
        dem = dem.where((dem != dem.rio.nodata) & np.isfinite(dem))
    rx, ry = dem.rio.resolution()
    dyg, dxg = np.gradient(dem.values, ry, rx)
    slope = np.rad2deg(np.arctan(np.sqrt(dxg ** 2 + dyg ** 2)))
    gx, gy = np.meshgrid(dem.x.values, dem.y.values)
    ok = shapely.contains_xy(poly, gx, gy) & np.isfinite(slope)
    return (poly, np.column_stack([gx[ok], gy[ok]]), slope[ok],
            dem.values[ok], abs(rx))


def cm(pts, px, v):
    m, _ = Terrain.cell_mean(pts, px, v)
    e = ~np.isfinite(m)
    if e.any():
        m[e] = v[cKDTree(px).query(pts[e], workers=-1)[1]]
    return m


def saturating(h, v_inf, L, q):
    """Captured variance rises toward v_inf as the cell approaches the pixel."""
    return v_inf / (1.0 + (h / L) ** q)


for glacier in ('gulkana', 'kennicott'):
    poly, px, pslope, pelev, dem_res = terrain(glacier)
    df = pd.read_csv(os.path.join(RESULTS, f'{glacier}_mesh_convergence.csv'))
    df = df.sort_values('actual_n_points').reset_index(drop=True)

    print('=' * 74)
    print(f'{glacier.upper()}   DEM {dem_res:.1f} m,  pixel slope var '
          f'{pslope.var():.2f}')
    print('=' * 74)

    # ---- 1. curvature of b with respect to slope, at the finest mesh ----
    h_fine = FINEST[glacier]
    xs, ys, w, _ = mesh.mesh_polygon(poly, h_fine)
    pts = np.column_stack([xs, ys])
    s_fine = cm(pts, px, pslope)
    z_fine = cm(pts, px, pelev)

    run = sorted(__import__('glob').glob(
        os.path.join(OUTDIR, f'{glacier}_h{h_fine}_wf2.5_*')))
    rds = xr.open_zarr(os.path.join(run[-1], 'output.zarr'))
    b_fine = rds.mass_balance.sum(dim='time').compute().values
    rds.close()
    assert len(b_fine) == len(w)

    # take out the elevation signal first, then look at what slope does
    A = np.column_stack([z_fine, np.ones(len(z_fine))])
    resid = b_fine - A @ np.linalg.lstsq(A * np.sqrt(w)[:, None],
                                         b_fine * np.sqrt(w), rcond=None)[0]
    lin = np.polyfit(s_fine, resid, 1, w=np.sqrt(w))
    quad = np.polyfit(s_fine, resid, 2, w=np.sqrt(w))
    print(f'b vs slope (elevation removed):')
    print(f'  linear   db/dslope = {lin[0]:+.4f} m w.e. per deg')
    print(f'  quadratic curvature = {2 * quad[0]:+.5f} m w.e. per deg^2')
    var_gap = pslope.var() - np.average(
        (s_fine - np.average(s_fine, weights=w)) ** 2, weights=w)
    print(f'  variance still missing at h={h_fine:.0f}: {var_gap:.2f} deg^2')
    print(f'  implied Jensen term 0.5*curv*missing_var = '
          f'{0.5 * 2 * quad[0] * var_gap / YEARS:+.4f} m w.e./yr')

    # ---- 2. captured variance and mass balance against it ----
    rows = []
    for _, r in df.iterrows():
        h = float(r['point_spacing'])
        xs, ys, w2, _ = mesh.mesh_polygon(poly, h)
        pts2 = np.column_stack([xs, ys])
        sv = cm(pts2, px, pslope)
        m = np.average(sv, weights=w2)
        var = np.average((sv - m) ** 2, weights=w2)
        rows.append((h, r['actual_n_points'], var / pslope.var(),
                     r['mass_balance'] / YEARS))
    sw = pd.DataFrame(rows, columns=['h', 'N', 'frac', 'mb'])

    c = np.polyfit(sw['frac'], sw['mb'], 1)
    pred = np.polyval(c, sw['frac'])
    r2 = 1 - ((sw['mb'] - pred) ** 2).sum() / (
        (sw['mb'] - sw['mb'].mean()) ** 2).sum()
    print()
    print(f'mass balance vs captured variance fraction:')
    print(f'  linear R^2 = {r2:.4f}')
    print(f'  extrapolated to 100% captured: {np.polyval(c, 1.0):+.4f} m w.e./yr')
    print(f'  value at finest run:           {sw["mb"].iloc[-1]:+.4f} m w.e./yr')

    popt, _ = curve_fit(saturating, sw['h'], sw['frac'],
                        p0=[1.0, 200.0, 1.0], maxfev=20000)
    print()
    print(f'captured-variance fit  v_inf={popt[0]:.3f}, L={popt[1]:.0f} m, '
          f'q={popt[2]:.2f}')
    for target in (0.95, 0.99):
        # invert: h = L*((v_inf/target)-1)^(1/q)
        ratio = popt[0] / target - 1.0
        h_t = popt[1] * ratio ** (1 / popt[2]) if ratio > 0 else np.nan
        print(f'  {target:.0%} captured at h = {h_t:>6.1f} m '
              f'({h_t / dem_res:.1f}x DEM)')
    print()
    print(f'{"h":>7} {"N":>7} {"captured":>10} {"MB/yr":>10} {"fit":>10}')
    for _, r in sw.iterrows():
        print(f'{r["h"]:>7.0f} {int(r["N"]):>7} {r["frac"]:>9.3f} '
              f'{r["mb"]:>10.4f} {np.polyval(c, r["frac"]):>10.4f}')
    print()
