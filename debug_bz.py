"""
Checks whether the modelled balance profile b(z) is the same relation at
every mesh resolution, or whether refining the mesh moves the relation
itself.

The sweep's drift has to be one or the other:

  sampling   the same b(z) at every h, sampled differently by the mesh
  relation   b(z) itself shifts, so a point at a given elevation returns
             a different balance depending on how coarse the mesh is

Part 1 overlays b(z) on common elevation bands across resolutions.
Part 2 splits the drift by applying the finest run's b(z) to each coarse
run's own (elevation, weight) pairs: that reproduces the coarse mass
balance only if the drift is sampling.

Reads elev, weight and mass_balance straight out of the saved runs.

    python debug_bz.py

@author: clairevwilson
"""
import glob
import os

import numpy as np
import xarray as xr

OUTDIR = '/ocean/projects/ees260009p/cwilson4/Output/point_density_test/'
SPACINGS = [1200.0, 700.0, 400.0, 200.0, 80.0]
FINEST = 80.0
BAND = 100.0


def load(h):
    """Pulls per-point elevation, area weight and total balance."""
    run = sorted(glob.glob(os.path.join(OUTDIR, f'gulkana_h{h}_wf2.5_*')))
    assert run, f'no saved run for h={h}'
    ds = xr.open_zarr(os.path.join(run[-1], 'output.zarr'))
    z = ds['elev'].values
    w = ds['weight'].values
    b = ds.mass_balance.sum(dim='time').compute().values
    ds.close()
    return z, w, b


runs = {}
for h in SPACINGS:
    print(f'loading h={h:.0f}...', flush=True)
    runs[h] = load(h)

print()
print('=' * 78)
print('RUN SUMMARY')
print('=' * 78)
print(f'{"h":>7} {"N":>6} {"sum wt":>9} {"wtd mean z":>12} {"wtd MB":>10} {"unwtd MB":>10}')
for h in SPACINGS:
    z, w, b = runs[h]
    print(f'{h:>7.0f} {len(z):>6} {w.sum():>9.5f} {np.average(z, weights=w):>12.2f} '
          f'{(b * w).sum():>10.4f} {b.mean():>10.4f}')

print()
print('=' * 78)
print('1. b(z) ON COMMON ELEVATION BANDS')
print('=' * 78)

zf, wf, bf = runs[FINEST]
lo = np.floor(zf.min() / BAND) * BAND
hi = np.ceil(zf.max() / BAND) * BAND
edges = np.arange(lo, hi + BAND, BAND)

header = f'{"band":>13}' + ''.join(f'{f"h={h:.0f}":>10}' for h in SPACINGS)
print(header)
print(f'{"":>13}' + ''.join(f'{"":>10}' for _ in SPACINGS))

profiles = {h: [] for h in SPACINGS}
centres = []
for e0, e1 in zip(edges[:-1], edges[1:]):
    row = f'{f"{e0:.0f}-{e1:.0f}":>13}'
    vals = []
    for h in SPACINGS:
        z, w, b = runs[h]
        m = (z >= e0) & (z < e1)
        if m.sum() == 0:
            vals.append(np.nan)
            row += f'{"-":>10}'
        else:
            v = np.average(b[m], weights=w[m])
            vals.append(v)
            row += f'{v:>10.3f}'
    centres.append((e0 + e1) / 2)
    for h, v in zip(SPACINGS, vals):
        profiles[h].append(v)
    print(row)

print()
print('difference from the finest run, same band:')
print(f'{"band":>13}' + ''.join(f'{f"h={h:.0f}":>10}' for h in SPACINGS[:-1]))
for i, (e0, e1) in enumerate(zip(edges[:-1], edges[1:])):
    row = f'{f"{e0:.0f}-{e1:.0f}":>13}'
    base = profiles[FINEST][i]
    for h in SPACINGS[:-1]:
        v = profiles[h][i]
        row += '        -' + ' ' if np.isnan(v) or np.isnan(base) else f'{v - base:>+10.3f}'
    print(row)

print()
print('A column that is systematically negative means the coarse mesh melts')
print('harder at that elevation -- the relation moved, not the sampling.')

print()
print('=' * 78)
print('2. SAMPLING vs RELATION')
print('=' * 78)

# b(z) from the finest run, area-weighted within each band
fine_c, fine_b = [], []
for i, c in enumerate(centres):
    if not np.isnan(profiles[FINEST][i]):
        fine_c.append(c)
        fine_b.append(profiles[FINEST][i])
fine_c, fine_b = np.array(fine_c), np.array(fine_b)

print(f'{"h":>7} {"actual MB":>11} {"predicted":>11} {"unexplained":>13}')
for h in SPACINGS:
    z, w, b = runs[h]
    actual = (b * w).sum()
    predicted = (np.interp(z, fine_c, fine_b) * w).sum()
    print(f'{h:>7.0f} {actual:>11.4f} {predicted:>11.4f} {actual - predicted:>+13.4f}')

print()
print('predicted applies the finest run\'s b(z) to each mesh\'s own hypsometry,')
print('so it carries only the sampling. If predicted is flat while actual')
print('drifts, the drift is in the relation and the unexplained column is it.')
