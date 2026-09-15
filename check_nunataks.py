"""
defeature drops interior rings smaller than min_feature_frac*h^2, so the
domain being meshed changes with h. This lists, for each glacier, how
many nunataks survive at each spacing in the sweep, and the spacing at
which the last one is admitted.
"""
import geopandas as gpd
import numpy as np
import shapely.geometry as geom

from host_paths import HOST_PATHS, host
from pebsi.io import mesh
from project.glacierwide_loss import translate_rgi

FRAC = 1 / 9
SWEEPS = {
    'gulkana':   [1200, 1000, 800, 650, 550, 450, 350, 250, 175, 125, 90, 80],
    'kennicott': [2000, 1600, 1200, 1000, 800, 600, 450, 375, 300, 250],
}

p = HOST_PATHS[host]
for glacier, sweep in SWEEPS.items():
    gid = translate_rgi[glacier]['6']
    region = gid.split('.')[0]
    names = [f.split('.')[0] for f in __import__('os').listdir(p['rgi_fp'])
             if f.startswith(region)]
    gdf = gpd.read_file(f"{p['rgi_fp']}../{names[0]}/{names[0]}.shp")
    poly, _ = mesh.glacier_polygon(gdf, gid)

    rings = []
    for g in ([poly] if poly.geom_type == 'Polygon' else poly.geoms):
        rings += [geom.Polygon(r).area for r in g.interiors]
    rings = np.array(sorted(rings))

    print('=' * 66)
    print(f'{glacier.upper()}  {poly.area / 1e6:.1f} km2, {len(rings)} nunataks')
    print('=' * 66)
    if len(rings):
        print(f'nunatak areas: min {rings.min():.0f} m2, median '
              f'{np.median(rings):.0f} m2, max {rings.max() / 1e6:.2f} km2')
        print(f'total nunatak area {rings.sum() / 1e6:.2f} km2')
        # h at which each ring is admitted: FRAC*h^2 <= area
        h_admit = np.sqrt(rings / FRAC)
        print(f'h to admit the smallest nunatak: {h_admit.min():.0f} m')
    print()
    print(f'{"h":>7} {"nunataks kept":>15} {"ice area meshed km2":>21}')
    for h in sweep:
        kept = (rings >= FRAC * h * h).sum()
        filled = rings[rings < FRAC * h * h].sum()
        print(f'{h:>7} {kept:>15} {(poly.area + filled) / 1e6:>21.2f}')
    print()
