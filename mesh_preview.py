"""
Previews the patch-conforming mesh for a glacier without running PEBSI.

Reports the point count and mesh quality at each element edge length, so a
spacing sweep can be sized before any simulation time is spent on it, and
optionally saves a plot of the mesh itself.

Usage:
    python mesh_preview.py gulkana
    python mesh_preview.py kennicott --spacing 800 400 200
    python mesh_preview.py gulkana --spacing 300 --plot gulkana_mesh.png

@author: clairevwilson
"""
import argparse
import os

import geopandas as gpd
import numpy as np
import pandas as pd

from AD_optimize import HOST_PATHS, host
from pebsi.io import mesh
from point_density_convergence import DEFAULT_SPACINGS
from project.glacierwide_loss import translate_rgi


def load_outline(rgi_id, rgi_fp):
    """Reads one glacier's outline out of the regional RGI shapefile."""
    region = rgi_id.split('.')[0]
    all_rgi = os.listdir(rgi_fp)
    region_name = [f.split('.')[0] for f in all_rgi if f.startswith(region)]
    assert len(region_name) == 1, f'Did not find RGI region {region} data'
    shapefile_fn = f'../{region_name[0]}/{region_name[0]}.shp'
    gdf = gpd.read_file(os.path.join(rgi_fp, shapefile_fn))
    return mesh.glacier_polygon(gdf, rgi_id)


def plot_mesh(polygon, h, out_fn):
    """Draws the elements, the outline and the point weights."""
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    nodes, simplices = mesh.mesh_nodes(polygon, h)
    xs, ys, weights, _ = mesh.mesh_polygon(polygon, h)

    fig, ax = plt.subplots(figsize=(10, 10))
    for poly in mesh._as_polygons(polygon):
        for ring in [poly.exterior, *poly.interiors]:
            coords = np.asarray(ring.coords)
            ax.plot(coords[:, 0], coords[:, 1], color='k', lw=1.2, zorder=3)

    tri = Triangulation(nodes[:, 0], nodes[:, 1], simplices)
    ax.triplot(tri, color='0.7', lw=0.3, zorder=1)
    scat = ax.scatter(xs, ys, c=weights * len(weights), s=6, cmap='viridis', zorder=2)
    fig.colorbar(scat, ax=ax, label='weight relative to a uniform 1/N')

    ax.set_aspect('equal')
    ax.set_title(f'h = {h:.0f} m, N = {len(xs)}')
    ax.set_xlabel('easting [m]')
    ax.set_ylabel('northing [m]')
    fig.savefig(out_fn, dpi=150, bbox_inches='tight')
    print(f'Saved mesh plot to {out_fn}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('glacier', choices=sorted(translate_rgi.keys()))
    parser.add_argument('--spacing', type=float, nargs='+', default=None)
    parser.add_argument('--plot', type=str, default=None,
                        help='filename to save a plot of the mesh (first spacing only)')
    args = parser.parse_args()

    rgi_id = translate_rgi[args.glacier]['6']
    polygon, _ = load_outline(rgi_id, HOST_PATHS[host]['rgi_fp'])

    spacings = args.spacing if args.spacing is not None else DEFAULT_SPACINGS
    spacings = sorted(set(spacings), reverse=True)

    n_holes = sum(len(p.interiors) for p in mesh._as_polygons(polygon))
    print(f'{args.glacier}: {polygon.area / 1e6:.1f} km2, {n_holes} interior rings')
    print()

    rows = []
    for h in spacings:
        xs, _, _, diag = mesh.mesh_polygon(polygon, h)
        rows.append({
            'spacing_m': h,
            'n_points': diag['n_points'],
            'area_ratio': diag['area_ratio'],
            'mean_min_angle': diag['mean_min_angle'],
        })
    print(pd.DataFrame(rows).to_string(index=False,
                                       float_format=lambda v: f'{v:.4g}'))

    if args.plot is not None:
        plot_mesh(polygon, spacings[0], args.plot)


if __name__ == '__main__':
    main()
