"""
Scores every (kp, wind_factor) combination in the loss_surface run and
saves the four glacier-wide log-losses to an .npz, summed across every
calibration glacier -- the same convention AD_optimize's own total uses.

The run tiled the combined mesh (every glacier at once) once per
combination, so combination c occupies points [c*n_base:(c+1)*n_base];
each block still holds every glacier's points together, told apart by
the 'rgiid' field the output carries. Each glacier's slice is scored
against its own observations the way AD_optimize scores it --
March-referenced albedo deltas, and mass balance compared only over the
period the Hugonnet data actually covers, area-weighted by ds.weight
rather than a flat mean across points -- and the per-glacier losses are
summed into that combination's total.

Slow (it reads ~2 GB of daily output per glacier), so it is kept apart
from plot_loss_surface.py -- run this once, then iterate on the figure.

@author: clairevwilson
"""
import os
import sys
from collections import defaultdict

import argparse
import glob

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from project.glacierwide_loss import Albedo, SnowlineMelt, MassBalance
from project.parameters import translate_rgi

# the most recent loss_surface.py run, used when --run-dir is not given
RUN_DIR = '/ocean/projects/ees260009p/cwilson4/Output/loss_surface_0'


def resolve_run_dir(run_dir):
    """
    A loss_surface.py run keeps everything it produced in one directory:
    the output .zarr, the grid .npz naming the combinations and glaciers,
    and (once this script has run) the losses. Pulls the three paths out
    of one, rather than making the caller repeat the directory.

    The grid .npz carries the run's --tag in its name, so it is matched by
    pattern instead of assumed.
    """
    grids = sorted(glob.glob(os.path.join(run_dir, 'loss_surface_grid*.npz')))
    assert grids, (f'no loss_surface_grid*.npz in {run_dir} -- an interrupted '
                    'run never gets one, since it is written only after the '
                    'simulation finishes')
    assert len(grids) == 1, f'several grid .npz files in {run_dir}: {grids}'
    return dict(grid_fn=grids[0],
                out_zarr=os.path.join(run_dir, 'output.zarr'),
                loss_fn=os.path.join(run_dir, 'loss_surface_losses.npz'))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', default=RUN_DIR,
                   help='loss_surface.py run directory holding output.zarr and '
                        'the grid .npz (default %(default)s)')
    p.add_argument('--out-zarr', default=None,
                   help='override the .zarr taken from --run-dir')
    p.add_argument('--grid-fn', default=None,
                   help='override the grid .npz taken from --run-dir; its '
                        'glacier list drives which glaciers are scored')
    p.add_argument('--loss-fn', default=None,
                   help='override where the per-metric loss surfaces are written')
    args = p.parse_args()

    resolved = resolve_run_dir(args.run_dir)
    for k, v in resolved.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    return args

ALBEDO_SIGMA = 0.0777

METRICS = ('albedo', 'mb', 'snow', 'melt')

# only these are touched anywhere in the loop below; mass_balance/weight
# are handled separately (see mb_weighted below) since their observation
# period runs 2000-2020, far wider than albedo/snow's post-2017 window
NEEDED_VARS = ['lon', 'lat', 'time', 'rgiid', 'albedo', 'surftype', 'total_water']


def main():
    args = parse_args()
    grid = np.load(args.grid_fn, allow_pickle=True)
    kp_values = grid['kp']
    wf_values = grid['wind_factor']
    n_base = int(grid['n_base'])
    n_combos = int(grid['n_combos'])
    glaciers = [str(g) for g in grid['glaciers']]
    start, end = str(grid['start']), '2025-03-20' # str(grid['end'])
    rgi_ids = {g: translate_rgi[g]['6'] for g in glaciers}

    print(f'grid  {args.grid_fn}', flush=True)
    print(f'zarr  {args.out_zarr}', flush=True)
    print(f'loss  {args.loss_fn}', flush=True)
    print(f'{len(glaciers)} glaciers x {n_combos} combos, {n_base} points '
          f'per combo, {start} to {end}', flush=True)

    ds = xr.open_zarr(args.out_zarr, consolidated=False)

    # measured sides are identical for every combination (all replicas sit
    # at the same coordinates), so the observation loaders are built once
    # per glacier
    obs = {}
    for g in glaciers:
        albedo = Albedo(g, use='s2')
        snowmelt = SnowlineMelt(g)
        mb = MassBalance(g, dates=(start, end))
        print(f'  {g:<12} MB measured {mb.meas:+.3f} +/- {mb.sigma:.3f} '
              f'm w.e. over {mb.matched_start.date()} to '
              f'{mb.matched_end.date()}', flush=True)
        obs[g] = dict(albedo=albedo, snowmelt=snowmelt, mb=mb)

    # every combo covers the same calendar span, so the whole time x point
    # block only needs to be pulled from the zarr once, trimmed to the
    # variables used and the window the observations actually fall in --
    # loading it fresh per combo re-decompresses the same full-width chunks
    # (point isn't chunked below the full 191600 width) n_combos times over
    tmins, tmaxs = [], []
    for g in glaciers:
        o = obs[g]
        tmins += [o['albedo'].ds_meas.time.values.min(),
                  o['snowmelt'].ds_meas_snow.time.values.min(),
                  o['snowmelt'].ds_meas_melt.time.values.min()]
        tmaxs += [o['albedo'].ds_meas.time.values.max(),
                  o['snowmelt'].ds_meas_snow.time.values.max(),
                  o['snowmelt'].ds_meas_melt.time.values.max()]
    pad = np.timedelta64(5, 'D')
    tmin, tmax = min(tmins) - pad, max(tmaxs) + pad

    print(f'Loading {NEEDED_VARS} for {tmin} to {tmax}...', flush=True)
    ds_all = ds[NEEDED_VARS].sel(time=slice(tmin, tmax)).load()
    print('Loaded.', flush=True)

    # mass_balance spans 2000-2020 at full point width -- loading it raw
    # would be the dominant memory cost, but get_model_mb only ever needs
    # it summed over each glacier's matched_start:matched_end, so reduce it
    # lazily via dask (streaming over the on-disk time chunks) and only
    # materialize the resulting (point,) vector, grouped by glaciers that
    # share the same matched period so the underlying chunks aren't reread
    # once per glacier
    periods = defaultdict(list)
    for g in glaciers:
        mb = obs[g]['mb']
        periods[(mb.matched_start, mb.matched_end)].append(g)

    rgiid_all = ds['rgiid'].values
    weight_all = ds['weight'].values
    mb_weighted = np.full(ds.sizes['point'], np.nan)
    for (mstart, mend), gs in periods.items():
        print(f'Reducing mass_balance for {gs} over {mstart} to {mend}...', flush=True)
        mask = np.isin(rgiid_all, [rgi_ids[g] for g in gs])
        mb_sum = (ds['mass_balance']
                  .sel(time=slice(mstart, mend))
                  .isel(point=mask)
                  .sum(dim='time')
                  .compute().values)
        mb_weighted[mask] = mb_sum * weight_all[mask]
    print('MB reduced.', flush=True)

    losses = {k: np.full(n_combos, np.nan) for k in METRICS}
    losses['total'] = np.full(n_combos, np.nan)
    per_glacier = {g: {k: np.full(n_combos, np.nan) for k in METRICS}
                   for g in glaciers}

    for c in range(n_combos):
        combo_slice = slice(c * n_base, (c + 1) * n_base)
        block = ds_all.isel(point=combo_slice)
        rgiid = block.rgiid.values
        combo_mb_weighted = mb_weighted[combo_slice]

        totals = {k: 0.0 for k in METRICS}
        for g in glaciers:
            gmask = rgiid == rgi_ids[g]
            sub = block.isel(point=gmask)
            o = obs[g]

            o['albedo'].get_model_albedo(sub)
            o['albedo'].get_deltas(method='march_mean')
            a_loss = o['albedo'].log_loss(sigma=ALBEDO_SIGMA)

            o['snowmelt'].get_model_snow(sub)
            s_loss, m_loss = o['snowmelt'].bernoulli_loss()

            # area-weighted by ds.weight rather than a flat mean across
            # points, and truncated to the period the observation covers
            # (pre-reduced above, since mass_balance's own 2000-2020 span
            # is far wider than what albedo/snow need)
            o['mb'].mod = float(combo_mb_weighted[gmask].sum())
            mb_loss = o['mb'].log_loss(mod=o['mb'].mod)

            vals = dict(albedo=a_loss, snow=s_loss, melt=m_loss, mb=mb_loss)
            for k, v in vals.items():
                per_glacier[g][k][c] = float(v)
                totals[k] += float(v)

        for k in METRICS:
            losses[k][c] = totals[k]
        losses['total'][c] = sum(totals.values())

        if c % 10 == 0 or c == n_combos - 1:
            print(f'  [{c + 1}/{n_combos}] kp={grid["kp_of_point"][c * n_base]:.1f} '
                  f'wf={grid["wind_factor_of_point"][c * n_base]:.1f} '
                  f'total={losses["total"][c]:.3f}', flush=True)

    # combos were built kp-major, so index c = i_kp * n_wf + j_wf
    shape = (len(kp_values), len(wf_values))
    grids = {k: v.reshape(shape) for k, v in losses.items()}
    per_glacier_grids = {f'{g}_{k}': v.reshape(shape)
                         for g, kv in per_glacier.items() for k, v in kv.items()}

    np.savez(args.loss_fn, kp=kp_values, wind_factor=wf_values,
             glaciers=np.array(glaciers), start=np.array(start),
             end=np.array(end), n_base=n_base, **grids, **per_glacier_grids)
    print(f'\nWrote {args.loss_fn}', flush=True)

    best = np.unravel_index(np.nanargmin(grids['total']), shape)
    print(f'Best total {grids["total"][best]:.3f} at '
          f'kp={kp_values[best[0]]:.1f}, wind_factor={wf_values[best[1]]:.1f}',
          flush=True)


if __name__ == '__main__':
    main()
