"""
Estimates the bias from using one uniform h=1000 m across all five
glaciers instead of each one's validated minimum h, and the resulting
point count.

N at h=1000 is exact -- computed directly from the mesh geometry, no
simulation needed. MB at h=1000 is linearly interpolated between the
two bracketing points already in each glacier's sweep, since none of
them were run at exactly h=1000 except Kennicott; it's an estimate,
not a measured value, and is flagged when extrapolated past the
sweep's range rather than interpolated within it.

    python estimate_h1000_bias.py
"""
import os

import numpy as np
import pandas as pd
import geopandas as gpd

from host_paths import HOST_PATHS, host
from pebsi.io import mesh
from project.glacierwide_loss import translate_rgi

TARGET_H = 1000.0
YEARS = 3.0
RESULTS = 'project/point_density_results/'
SUFFIX_PRIORITY = ['_final', '_albedofix', '_costheta', '']

# from the validated convergence analysis earlier this session
CONVERGED = {
    'gulkana': -1.166,
    'lemon_creek': 0.712,
    'kahiltna': -0.3216,
    'wolverine': 1.447,
    'kennicott': 0.775,
}


def load(glacier):
    for s in SUFFIX_PRIORITY:
        fn = os.path.join(RESULTS, f'{glacier}_mesh_convergence{s}.csv')
        if os.path.exists(fn):
            return pd.read_csv(fn), s
    return None, None


p = HOST_PATHS[host]
rgi_gdf = gpd.read_file(p['rgi_fp'] + '../01_rgi60_Alaska/01_rgi60_Alaska.shp')

total_n = 0
print(f'{"glacier":>12} {"N@h1000":>9} {"MB@h1000":>10} {"converged":>10} '
      f'{"bias":>8} {"how":>12} {"data":>10}')
for glacier in CONVERGED:
    gid = translate_rgi[glacier]['6']
    poly, crs = mesh.glacier_polygon(rgi_gdf, gid)
    xs, ys, w, diag = mesh.mesh_polygon(poly, TARGET_H)
    n_at_1000 = len(xs)
    total_n += n_at_1000

    df, suffix = load(glacier)
    df = df.sort_values('point_spacing', ascending=False)
    h = df['point_spacing'].values
    mb = df['mass_balance'].values / YEARS

    if TARGET_H in h:
        mb_1000 = mb[h == TARGET_H][0]
        how = 'exact run'
    else:
        above = h[h > TARGET_H]
        below = h[h < TARGET_H]
        if len(above) and len(below):
            h_lo, h_hi = below.max(), above.min()
            mb_lo, mb_hi = mb[h == h_lo][0], mb[h == h_hi][0]
            frac = (TARGET_H - h_lo) / (h_hi - h_lo)
            mb_1000 = mb_lo + frac * (mb_hi - mb_lo)
            how = f'interp {h_lo:.0f}-{h_hi:.0f}'
        else:
            mb_1000 = np.nan
            how = 'OUT OF RANGE'

    bias = mb_1000 - CONVERGED[glacier]
    print(f'{glacier:>12} {n_at_1000:>9} {mb_1000:>10.4f} {CONVERGED[glacier]:>10.4f} '
          f'{bias:>+8.4f} {how:>12} {suffix:>10}')

print()
print(f'total N at h=1000 across all 5 glaciers: {total_n}')
