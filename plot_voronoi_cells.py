"""
Draws a glacier's model points as their Voronoi cells, filled by the area
each one represents -- the weight PEBSI actually gives that point.

Change the resolution with --spacing. With no --spacing it uses every h
already simulated for that glacier (read from the mesh convergence CSV),
so the panels line up with runs whose mass balance you can look at.

Usage:
    python plot_voronoi_cells.py gulkana --spacing 300
    python plot_voronoi_cells.py gulkana --spacing 800 450 300 200
    python plot_voronoi_cells.py gulkana                  # every h already run
    python plot_voronoi_cells.py gulkana --lattice 500    # the old clipped lattice

@author: clairevwilson
"""
import argparse
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely.geometry as geom
from matplotlib.colors import TwoSlopeNorm

from host_paths import HOST_PATHS, host
from pebsi.io import mesh
from project.glacierwide_loss import translate_rgi

RESULTS_DIR = 'project/point_density_results/'


def load_outline(rgi_id):
    """Reads one glacier's outline out of the regional RGI shapefile."""
    rgi_fp = HOST_PATHS[host]['rgi_fp']
    region = rgi_id.split('.')[0]
    region_name = [f.split('.')[0] for f in os.listdir(rgi_fp) if f.startswith(region)]
    assert len(region_name) == 1, f'Did not find RGI region {region} data'
    gdf = gpd.read_file(os.path.join(rgi_fp, f'../{region_name[0]}/{region_name[0]}.shp'))
    return mesh.glacier_polygon(gdf, rgi_id)


def spacings_already_run(glacier):
    """Element sizes that have a simulated mass balance in the results CSV."""
    fn = os.path.join(RESULTS_DIR, f'{glacier}_mesh_convergence.csv')
    if not os.path.exists(fn):
        return []
    return sorted(pd.read_csv(fn)['point_spacing'].unique(), reverse=True)


def points_for(polygon, crs, spacing, lattice):
    """
    Places points either with the patch-conforming mesh or with the old
    clipped lattice, and returns them with their Voronoi cells.
    """
    if lattice:
        lons, lats, _ = mesh.grid_polygon(polygon, int(spacing), crs, 0.05)
        pts_ll = gpd.GeoDataFrame(
            geometry=[geom.Point(a, b) for a, b in zip(lons, lats)],
            crs='EPSG:4326').to_crs(crs)
        points = list(pts_ll.geometry)
        label = f'lattice, n={int(spacing)}'
    else:
        xs, ys, _, _ = mesh.mesh_polygon(polygon, spacing)
        points = [geom.Point(x, y) for x, y in zip(xs, ys)]
        label = f'h = {spacing:.0f} m'

    cells = mesh.voronoi_cells(points, polygon)
    return points, cells, label


def draw_cells(ax, polygon, points, cells, values, cmap, norm, title):
    """
    Fills each Voronoi cell by a per-point value, over the glacier
    outline. Shared with plot_mb_cells.py so the two figures are laid
    out identically and only the colour means something different.

    Parameters
    ==========
    ax : matplotlib Axes
        Axes to draw into
    polygon : shapely geometry
        Glacier outline in metric coordinates
    points : list of shapely Points
        Model points, in the same CRS
    cells : sequence of shapely geometries
        Each point's Voronoi cell, clipped to the outline
    values : 1D array
        Value to colour each cell by
    cmap : str
        Matplotlib colormap name
    norm : matplotlib Normalize
        Mapping from value to colour
    title : str
        Panel title
    """
    gdf = gpd.GeoDataFrame({'value': values}, geometry=list(cells))
    gdf.plot(ax=ax, column='value', cmap=cmap, norm=norm,
             edgecolor='0.35', linewidth=0.25)

    for poly in mesh._as_polygons(polygon):
        for ring in [poly.exterior, *poly.interiors]:
            xy = np.asarray(ring.coords)
            ax.plot(xy[:, 0], xy[:, 1], color='k', lw=1.3, zorder=3)

    ax.scatter([p.x for p in points], [p.y for p in points],
               s=1.5, color='k', alpha=0.55, zorder=4)

    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.set_title(title, fontsize=10)


def draw(ax, polygon, points, cells, label, vmax):
    """Fills each cell by its weight relative to a uniform 1/N."""
    areas = np.array([c.area for c in cells])
    relative = areas / areas.sum() * len(areas)
    title = (f'{label}\nN = {len(points)}, '
             f'max {relative.max():.1f}x uniform')
    draw_cells(ax, polygon, points, cells, relative, 'RdBu_r',
               TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=vmax), title)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('glacier', choices=sorted(translate_rgi.keys()))
    parser.add_argument('--spacing', type=float, nargs='+', default=None,
                        help='element edge length(s) in m; default is every h already run')
    parser.add_argument('--lattice', type=int, nargs='+', default=None,
                        help='instead draw the old clipped lattice at these point counts')
    parser.add_argument('--vmax', type=float, default=2.5,
                        help='top of the colour scale, in multiples of a uniform 1/N')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    polygon, crs = load_outline(translate_rgi[args.glacier]['6'])

    lattice = args.lattice is not None
    values = args.lattice if lattice else args.spacing
    if values is None:
        values = spacings_already_run(args.glacier)
        assert values, (f'No simulated resolutions for {args.glacier}; '
                        'pass --spacing explicitly')

    # two columns keeps the figure roughly square instead of a long strip
    ncol = min(len(values), 2)
    nrow = int(np.ceil(len(values) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(8, 6), squeeze=False)

    for ax, value in zip(axes.ravel(), values):
        points, cells, label = points_for(polygon, crs, value, lattice)
        draw(ax, polygon, points, cells, label, args.vmax)
        print(f'{label}: N={len(points)}')
    for ax in axes.ravel()[len(values):]:
        ax.set_axis_off()

    mappable = plt.cm.ScalarMappable(cmap='RdBu_r',
                                     norm=TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=args.vmax))
    fig.colorbar(mappable, ax=axes, shrink=0.6,
                 label='cell area relative to a uniform 1/N  (white = its fair share)')

    fig.suptitle(f'{args.glacier}: model points and the area each represents', y=0.99)
    out = args.out or f'{args.glacier}_voronoi_cells.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
