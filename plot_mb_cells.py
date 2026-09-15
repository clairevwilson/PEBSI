"""
Draws a glacier's model points as their Voronoi cells, filled by the
annual mass balance rate each point actually simulated.

Laid out exactly like plot_voronoi_cells.py -- same cells, same outline,
same panels -- so the two can be read side by side: one shows how much
area a point speaks for, this one shows what it says. Values come from
the simulated output.zarr, not from a re-run.

Change the resolution with --spacing. With no --spacing it draws every h
that has output on disk for that glacier.

Usage:
    python plot_mb_cells.py gulkana --spacing 300
    python plot_mb_cells.py gulkana --spacing 800 450 300 200
    python plot_mb_cells.py gulkana

@author: clairevwilson
"""
import argparse
import glob
import os
import re

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import shapely.geometry as geom
import xarray as xr
from matplotlib.colors import TwoSlopeNorm

from host_paths import HOST_PATHS, host
from pebsi.io import mesh
from plot_voronoi_cells import draw_cells, load_outline
from project.glacierwide_loss import translate_rgi

OUTDIR = os.path.normpath(os.path.join(HOST_PATHS[host]['output_fp'],
                                       '..', 'point_density_test'))
HOURS_PER_YEAR = 365.25 * 24


def run_dirs(glacier):
    """
    Maps each simulated element size to its output directory.

    Directories are named {glacier}_h{spacing}_wf{wind}_{i}. Anything
    carrying extra settings in the tag belongs to a different
    experiment, so it is skipped rather than guessed at.
    """
    runs = {}
    for d in sorted(glob.glob(os.path.join(OUTDIR, f'{glacier}_h*'))):
        zarrs = sorted(glob.glob(os.path.join(d, 'output.zarr')))
        if not zarrs:
            continue
        tag = os.path.basename(d).split('_h')[1].split('_')[0]
        if not re.fullmatch(r'[0-9.]+', tag):
            continue
        # later directories for the same spacing are later reruns
        runs[float(tag)] = zarrs[-1]
    return runs


def load_run(zarr_fn, crs):
    """
    Reads one run's points and annual mass balance rate.

    Returns the points already reprojected into the glacier's metric
    CRS, so they can be turned straight into Voronoi cells.
    """
    ds = xr.open_zarr(zarr_fn)
    total = ds.mass_balance.sum(dim='time').compute().values
    years = ds.sizes['time'] / HOURS_PER_YEAR
    rate = total / years

    pts = gpd.GeoDataFrame(
        geometry=[geom.Point(a, b) for a, b in zip(ds.lon.values, ds.lat.values)],
        crs='EPSG:4326').to_crs(crs)
    weight = ds.weight.values if 'weight' in ds else None
    ds.close()
    return list(pts.geometry), rate, weight


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('glacier', choices=sorted(translate_rgi.keys()))
    parser.add_argument('--spacing', type=float, nargs='+', default=None,
                        help='element edge length(s) in m; default is every h on disk')
    parser.add_argument('--vlim', type=float, default=None,
                        help='symmetric colour limit [m w.e. per year]; default from the data')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    polygon, crs = load_outline(translate_rgi[args.glacier]['6'])

    runs = run_dirs(args.glacier)
    assert runs, f'No simulated output found for {args.glacier} under {OUTDIR}'

    values = args.spacing if args.spacing is not None else sorted(runs, reverse=True)
    missing = [v for v in values if v not in runs]
    assert not missing, (f'No output on disk for h={missing}; '
                         f'available: {sorted(runs, reverse=True)}')

    # build every panel first, so one colour scale covers them all
    panels = []
    for spacing in values:
        points, rate, weight = load_run(runs[spacing], crs)
        cells = mesh.voronoi_cells(points, polygon)
        panels.append((spacing, points, cells, rate, weight))
        print(f'h = {spacing:.0f} m: N={len(points)}, '
              f'rate {rate.min():+.2f} to {rate.max():+.2f} m w.e. a-1')

    vlim = args.vlim or max(np.abs(p[3]).max() for p in panels)
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)

    # two columns keeps the figure roughly square instead of a long strip
    ncol = min(len(panels), 2)
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(8, 6), squeeze=False)

    for ax, (spacing, points, cells, rate, weight) in zip(axes.ravel(), panels):
        glacierwide = float((weight * rate).sum()) if weight is not None else float(rate.mean())
        title = (f'h = {spacing:.0f} m; '
                 f'N = {len(points)}\nGlacier-wide {glacierwide:+.3f} m w.e. a$^{{-1}}$')
        draw_cells(ax, polygon, points, cells, rate, 'RdBu', norm, title)
    for ax in axes.ravel()[len(panels):]:
        ax.set_axis_off()

    mappable = plt.cm.ScalarMappable(cmap='RdBu', norm=norm)
    fig.colorbar(mappable, ax=axes, shrink=0.6,
                 label='mass balance rate [m w.e. a$^{-1}$]')

    fig.suptitle(f'{args.glacier}: simulated mass balance at each model point', y=0.99)
    out = args.out or f'{args.glacier}_mb_cells.png'
    fig.savefig(out, dpi=300, bbox_inches='tight')
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
