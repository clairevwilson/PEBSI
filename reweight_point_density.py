"""
Recomputes glacier-wide mass balance for every saved point_density_convergence.py
run using the corrected Voronoi-cell area weighting (mesh.voronoi_weights),
straight from each run's already-saved output.zarr (lat, lon, mass_balance) --
no re-simulation needed, since the fix only changes how weight is derived from
point locations, not the physics.

Scans OUTDIR for every saved {glacier}_n{n}[_wf{wind_factor}]_0 directory,
for every glacier in translate_rgi, and writes one CSV per (glacier,
wind_factor) combination found: wind_factor=3 (default/baseline) goes to
project/point_density_results/{glacier}_point_density.csv, any other
wind_factor to project/point_density_results/{glacier}_point_density_wf{wf}.csv.

Usage:
    python reweight_point_density.py

@author: clairevwilson
"""
import os
import re

import geopandas as gpd
import pandas as pd
import xarray as xr
from pyproj import Transformer
from shapely.geometry import Point

from AD_optimize import HOST_PATHS, host
from pebsi.io import mesh
from project.glacierwide_loss import translate_rgi

OUTDIR = os.path.normpath(os.path.join(HOST_PATHS[host]['output_fp'], '..', 'point_density_test'))
RESULTS_DIR = 'project/point_density_results/'
RGI_FP = HOST_PATHS[host]['rgi_fp']

GLACIER_ALT = '|'.join(sorted(translate_rgi.keys(), key=len, reverse=True))
DIR_RE = re.compile(rf'^(?P<glacier>{GLACIER_ALT})_n(?P<n>\d+)(?:_wf(?P<wf>[\d.]+))?_0$')

# runs made before --wind-factor existed used baseline['wind_factor'] (3.0)
# and have no _wf suffix in their directory name
DEFAULT_WIND_FACTOR = 3.0


def load_rgi_polygon(rgi_id):
    region_name = [f.split('.')[0] for f in os.listdir(RGI_FP) if f.startswith('01')][0]
    gdf = gpd.read_file(os.path.join(RGI_FP, f'../{region_name}/{region_name}.shp'))
    row = gdf.loc[gdf['RGIId'] == 'RGI60-' + rgi_id]
    metric_crs = mesh.get_metric_crs(row)
    return row.to_crs(metric_crs).union_all(), metric_crs, row['Area'].item()


def reweight_one(out_dir, polygon_metric, metric_crs):
    ds = xr.open_zarr(os.path.join(out_dir, 'output.zarr'))
    lat, lon = ds.lat.values, ds.lon.values
    mb_per_point = ds.mass_balance.sum(dim='time').compute().values
    ds.close()

    transformer = Transformer.from_crs('EPSG:4326', metric_crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    points = [Point(xi, yi) for xi, yi in zip(x, y)]
    weights = mesh.voronoi_weights(points, polygon_metric)

    return len(lat), float(mb_per_point.mean()), float((mb_per_point * weights).sum())


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for glacier in translate_rgi.keys():
        rgi_id = translate_rgi[glacier]['6']
        polygon_metric, metric_crs, area_km2 = load_rgi_polygon(rgi_id)

        by_wf = {}  # wind_factor -> list of row dicts
        for entry in sorted(os.listdir(OUTDIR)):
            m = DIR_RE.match(entry)
            if not m or m.group('glacier') != glacier:
                continue
            n = int(m.group('n'))
            wf = float(m.group('wf')) if m.group('wf') else DEFAULT_WIND_FACTOR

            actual_n, mb_unweighted, mb_weighted = reweight_one(
                os.path.join(OUTDIR, entry), polygon_metric, metric_crs)
            print(f'{glacier} wf={wf} n={n:>5} actual={actual_n:>5} '
                  f'unweighted={mb_unweighted:+.4f} weighted={mb_weighted:+.4f}')

            by_wf.setdefault(wf, []).append({
                'glacier': glacier,
                'area_km2': area_km2,
                'wind_factor': wf,
                'requested_n_points': n,
                'actual_n_points': actual_n,
                'mass_balance': mb_weighted,
                'mass_balance_unweighted': mb_unweighted,
            })

        for wf, rows in by_wf.items():
            df = pd.DataFrame(rows).sort_values('requested_n_points').reset_index(drop=True)
            reference_mb = df.iloc[-1]['mass_balance']
            df['mb_diff_from_densest'] = df['mass_balance'] - reference_mb

            suffix = '' if wf == DEFAULT_WIND_FACTOR else f'_wf{wf:g}'
            out_csv = os.path.join(RESULTS_DIR, f'{glacier}_point_density{suffix}.csv')
            df.to_csv(out_csv, index=False)
            print(f'Saved {out_csv}')


if __name__ == '__main__':
    main()
