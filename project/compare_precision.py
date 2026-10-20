"""
Compares the float64 and float32 runs of the controlled experiment's base
case (batch_jobs/float32_test.sh): glacier-wide annual and cumulative mass
balance, per-point totals of the mass balance terms, the hourly surface
fields, and the final layer state from each run's checkpoint.
"""
import numpy as np
import xarray as xr

from project.parameters import GLACIERS, translate_rgi

OUTPUT_FP = '/ocean/projects/ees260009p/cwilson4/Output/'
RUN_64 = OUTPUT_FP + 'precision_float64/base_0/'
RUN_32 = OUTPUT_FP + 'precision_float32/base_0/'

# per-point totals over the run
TOTAL_VARS = ['mass_balance', 'melt', 'accumulation', 'refreeze', 'runoff', 'rainfall']

# hourly fields compared step by step
HOURLY_VARS = ['albedo', 'surftemp', 'top_water', 'sensible_heat', 'latent_heat',
               'ground_heat', 'error']

# final-state layer fields from checkpoint.npz
LAYER_VARS = ['ltemp', 'ldensity', 'lice', 'lwater', 'lheight', 'lgrainsize']

# near-surface water [kg m-2] above which the surface counts as wet,
# matching WET_WATER_THRESHOLD = 0.002 m w.e. in AD_optimize.py
WET_THRESHOLD = 2.0


def open_run(fp):
    ds = xr.open_zarr(fp + 'output.zarr', consolidated=False)
    return ds.chunk({'time': 8760, 'point': -1})


def glacierwide(da, rgiid, weight, gid):
    mask = rgiid == gid
    w = weight[mask] / weight[mask].sum()
    return (da.isel(point=np.where(mask)[0]) * xr.DataArray(w, dims='point')).sum('point')


def main():
    ds64 = open_run(RUN_64)
    ds32 = open_run(RUN_32)
    rgiid = np.asarray(ds64['rgiid'].values, dtype=str)
    weight = np.asarray(ds64['weight'].values, dtype=np.float64)

    print('========== glacier-wide mass balance [m w.e.] ==========')
    mb64 = ds64['mass_balance'].astype(np.float64)
    mb32 = ds32['mass_balance'].astype(np.float64)
    annual64 = mb64.resample(time='YS-OCT').sum().compute()
    annual32 = mb32.resample(time='YS-OCT').sum().compute()
    for name in GLACIERS:
        gid = translate_rgi[name]['6']
        a64 = glacierwide(annual64, rgiid, weight, gid).values
        a32 = glacierwide(annual32, rgiid, weight, gid).values
        diff = a32 - a64
        print(f'{name:<12} cumulative float64 {a64.sum():+9.4f}  float32 {a32.sum():+9.4f}  '
              f'diff {diff.sum():+.2e}   annual diff max |{np.abs(diff).max():.2e}|  '
              f'rms {np.sqrt(np.mean(diff ** 2)):.2e}')

    print('\n========== per-point totals over the run ==========')
    for var in TOTAL_VARS:
        t64 = ds64[var].astype(np.float64).sum('time').compute().values
        t32 = ds32[var].astype(np.float64).sum('time').compute().values
        diff = t32 - t64
        rel = np.abs(diff).max() / max(np.abs(t64).max(), 1e-30)
        print(f'{var:<14} mean |float64| {np.abs(t64).mean():.4e}  '
              f'max |diff| {np.abs(diff).max():.3e}  rms diff {np.sqrt(np.mean(diff ** 2)):.3e}  '
              f'max |diff| / max |total| {rel:.2e}')

    print('\n========== hourly fields ==========')
    for var in HOURLY_VARS:
        d = (ds32[var].astype(np.float64) - ds64[var].astype(np.float64))
        max_abs = float(np.abs(d).max().compute())
        rms = float(np.sqrt((d ** 2).mean()).compute())
        mean64 = float(np.abs(ds64[var]).mean().compute())
        print(f'{var:<14} mean |float64| {mean64:.4e}  max |diff| {max_abs:.3e}  rms diff {rms:.3e}')

    wet64 = (ds64['top_water'] > WET_THRESHOLD)
    wet32 = (ds32['top_water'] > WET_THRESHOLD)
    flipped = float((wet64 != wet32).mean().compute())
    print(f'wet/dry state differs in {100 * flipped:.3f}% of point-hours')

    print('\n========== final layer state (checkpoint) ==========')
    c64 = np.load(RUN_64 + 'checkpoint.npz')
    c32 = np.load(RUN_32 + 'checkpoint.npz')
    for var in LAYER_VARS:
        a64 = c64[var].astype(np.float64)
        a32 = c32[var].astype(np.float64)
        diff = a32 - a64
        print(f'{var:<12} mean |float64| {np.abs(a64).mean():.4e}  '
              f'max |diff| {np.abs(diff).max():.3e}  rms diff {np.sqrt(np.mean(diff ** 2)):.3e}')
    cme64 = c64['cum_mass_error'].astype(np.float64)
    cme32 = c32['cum_mass_error'].astype(np.float64)
    print(f'cum_mass_error max |float64| {np.abs(cme64).max():.3e}  max |float32| {np.abs(cme32).max():.3e}')
    same_type = np.mean(c64['ltype'] == c32['ltype'])
    print(f'layer type identical in {100 * same_type:.2f}% of point-layers')


if __name__ == '__main__':
    main()
