"""
Meshes every glacier a region-wide run would simulate, without running
the model, to see how many points the survey for that region costs.

Glaciers are picked the same way config.py picks them when no rgi_ids
are given: every glacier in the region larger than min_area.

Usage:
    python region_mesh_preview.py
    python region_mesh_preview.py --region 1 --spacing 700 --min-area 2

@author: clairevwilson
"""
import argparse
import os
import time
from multiprocessing import Pool

import geopandas as gpd
import pandas as pd

from project.parameters import HOST_PATHS, host
from pebsi import defaults
from pebsi.io import mesh

OUTDIR = 'project/point_density_results/'

# point_density_rule.SURVEY_SPACING, the spacing at which sigma is converged;
# not imported because that module loads JAX, which cannot be forked
SURVEY_SPACING = 700.0

RGI_GDF = None


def load_region(region, rgi_fp, min_area, max_area=None):
    """Reads the region's attributes and outlines, cropped to the area range."""
    region = str(region).zfill(2)
    csv_fn = [f for f in os.listdir(rgi_fp) if f.startswith(region) and f.endswith('csv')][0]
    name = csv_fn.split('.')[0]
    df = pd.read_csv(os.path.join(rgi_fp, csv_fn))
    df = df.loc[df['Area'] > min_area]
    if max_area is not None:
        df = df.loc[df['Area'] <= max_area]
    gdf = gpd.read_file(os.path.join(rgi_fp, f'../{name}/{name}.shp'))
    gdf = gdf.loc[gdf['RGIId'].isin(df['RGIId'])]
    return df, gdf


def mesh_one(args):
    """Meshes one glacier and returns its diagnostics."""
    gid, spacing = args
    polygon, _ = mesh.glacier_polygon(RGI_GDF, gid)
    _, _, _, diagnostics = mesh.mesh_polygon(polygon, spacing)
    return dict(rgiid=gid, area_km2=round(polygon.area / 1e6, 3),
                n_points=diagnostics['n_points'],
                area_ratio=round(diagnostics['area_ratio'], 4))


def main():
    global RGI_GDF
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--region', type=int, default=defaults.rgi_region)
    parser.add_argument('--spacing', type=float, default=SURVEY_SPACING,
                        help=f'element edge length [m] (default {SURVEY_SPACING:.0f})')
    parser.add_argument('--min-area', type=float, default=defaults.min_area,
                        help=f'smallest glacier to include [km2] (default {defaults.min_area})')
    parser.add_argument('--max-area', type=float, default=None,
                        help='largest glacier to include [km2] (default no limit)')
    parser.add_argument('--limit', type=int, default=None,
                        help='mesh only the first N glaciers, for a quick test')
    args = parser.parse_args()

    df, RGI_GDF = load_region(args.region, HOST_PATHS[host]['rgi_fp'],
                               args.min_area, args.max_area)
    gids = [f.split('-')[-1] for f in df['RGIId']][:args.limit]
    print(f'Meshing {len(gids)} glaciers over {args.min_area:g} km2 '
          f'at h={args.spacing:.0f} m on {os.cpu_count()} CPUs')

    start = time.time()
    n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    with Pool(n_workers) as pool:
        rows = pool.map(mesh_one, [(gid, args.spacing) for gid in gids], chunksize=4)
    table = pd.DataFrame(rows).sort_values('n_points', ascending=False)
    print(f'Done in {(time.time() - start) / 60:.1f} min')

    os.makedirs(OUTDIR, exist_ok=True)
    area_tag = f'_a{args.min_area:g}-{args.max_area:g}' if args.max_area else f'_a{args.min_area:g}+'
    out_fn = os.path.join(OUTDIR, f'region{args.region:02d}{area_tag}_mesh_h{args.spacing:.0f}.csv')
    table.to_csv(out_fn, index=False)

    print()
    print(f'Total: {table["n_points"].sum()} points across {len(table)} glaciers '
          f'({table["area_km2"].sum():.0f} km2)')
    print(f'Single-point glaciers: {(table["n_points"] == 1).sum()}')
    print(table['n_points'].describe().to_string())
    print()
    print('Largest:')
    print(table.head(15).to_string(index=False))
    print(f'\nWrote {out_fn}')


if __name__ == '__main__':
    main()
