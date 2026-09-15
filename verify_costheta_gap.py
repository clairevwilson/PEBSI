"""
Direct, from-scratch confirmation that the cos_theta averaging-order fix
closes the h=1200-h=300 mesh convergence gap, using only actual mass
balance values pulled straight from the four saved output.zarr runs --
no cos_theta values, no fitted sensitivity, no unit conversion anywhere.

    Delta_current    = MB(h=1200, on-the-fly)  - MB(h=300, on-the-fly)
    Delta_precomputed = MB(h=1200, precomputed) - MB(h=300, precomputed)

Both are pure m w.e./yr differences. If gap-closing survives this direct
form, the earlier 91.3% result was not an artifact of converting through
two different implied cos_theta sensitivities.

    python verify_costheta_gap.py
"""
import geopandas as gpd
import numpy as np
import xarray as xr

from host_paths import HOST_PATHS, host
from pebsi.io import mesh
from project.glacierwide_loss import translate_rgi

GLACIER = 'gulkana'
OUTDIR = '/ocean/projects/ees260009p/cwilson4/Output/point_density_test/'
YEARS = 3.0

RUNS = {
    (1200.0, 'current'):     'gulkana_h1200_ct0_wf2.5_0',
    (1200.0, 'precomputed'): 'gulkana_h1200_ct1_wf2.5_0',
    (300.0,  'current'):     'gulkana_h300_ct0_wf2.5_0',
    (300.0,  'precomputed'): 'gulkana_h300_ct1_wf2.5_0',
}

p = HOST_PATHS[host]
gid = translate_rgi[GLACIER]['6']
region = gid.split('.')[0]
import os
names = [f.split('.')[0] for f in os.listdir(p['rgi_fp']) if f.startswith(region)]
gdf = gpd.read_file(f"{p['rgi_fp']}../{names[0]}/{names[0]}.shp")
polygon, crs = mesh.glacier_polygon(gdf, gid)

# mesh weights reconstructed independently of the experiment script,
# from the polygon alone -- do not depend on option_precomputed_costheta
weights_by_h = {}
for h in (1200.0, 300.0):
    xs, ys, w, _ = mesh.mesh_polygon(polygon, h)
    weights_by_h[h] = w
    print(f'h={h:.0f}: reconstructed {len(w)} points, weight sum={w.sum():.6f}')

print()
mb = {}
for (h, order), dirname in RUNS.items():
    ds = xr.open_zarr(os.path.join(OUTDIR, dirname, 'output.zarr'))
    mb_per_point = ds.mass_balance.sum(dim='time').compute().values
    ds.close()
    w = weights_by_h[h]
    assert len(mb_per_point) == len(w), (
        f'{dirname}: {len(mb_per_point)} points in output, '
        f'{len(w)} in reconstructed mesh -- mismatch')
    mb[(h, order)] = float((mb_per_point * w).sum()) / YEARS
    print(f'{dirname:>32}  N={len(w):>4}  MB={mb[(h, order)]:+.6f} m w.e./yr')

print()
print('=' * 70)
print('MATCHED-UNITS DIFFERENTIAL TEST -- MB values only, no conversion')
print('=' * 70)
d_current = mb[(1200.0, 'current')] - mb[(300.0, 'current')]
d_precomp = mb[(1200.0, 'precomputed')] - mb[(300.0, 'precomputed')]
print(f'Delta_current     = MB(1200,current)     - MB(300,current)     '
      f'= {mb[(1200.0,"current")]:+.6f} - ({mb[(300.0,"current")]:+.6f}) '
      f'= {d_current:+.6f}')
print(f'Delta_precomputed = MB(1200,precomputed) - MB(300,precomputed) '
      f'= {mb[(1200.0,"precomputed")]:+.6f} - ({mb[(300.0,"precomputed")]:+.6f}) '
      f'= {d_precomp:+.6f}')
print()
print(f'gap closed: {100 * (1 - abs(d_precomp) / abs(d_current)):.1f}%')
print(f'residual to explain was 0.0244 m w.e./yr; '
      f'gap shrank by {abs(d_current) - abs(d_precomp):.4f}')
