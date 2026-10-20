"""
Compares the forward runs of batch_jobs/impurity_depth_compare.sh: the
emulator fed impurity concentrations averaged over the top 5 cm against the
top layer's own. Reports glacier-wide mass balance and summer albedo per
glacier, and the surface dust at Gulkana D on the day of the field samples.
"""
import numpy as np
import pandas as pd
import xarray as xr

from project.parameters import GLACIERS, translate_rgi

OUTPUT_FP = '/ocean/projects/ees260009p/cwilson4/Output/'
RUNS = {'window5cm': OUTPUT_FP + 'impurity_window5cm/base_0/output.zarr',
        'toplayer': OUTPUT_FP + 'impurity_toplayer/base_0/output.zarr'}

# Gulkana D and the two top-3 cm surface samples taken there [ppm]
SITE_LAT, SITE_LON = 63.285115, -145.385018
SAMPLE_DATE = '2024-08-20'
SAMPLES_PPM = (160, 262)
SAMPLE_DEPTHS = (0.03, 0.05)

SUMMER_MONTHS = [6, 7, 8]


def glacierwide(da, rgiid, weight, gid):
    mask = rgiid == gid
    w = xr.DataArray(weight[mask] / weight[mask].sum(), dims='point')
    return (da.isel(point=np.where(mask)[0]) * w).sum('point')


def depth_mean(ds, var, depth):
    """Mass-weighted mean of a per-layer concentration over the top depth [m]."""
    h = ds['layerheight'].fillna(0.0)
    top = h.cumsum('layer') - h
    frac = ((depth - top) / h.where(h > 0, 1.0)).clip(0.0, 1.0)
    mass = h * ds['layerdensity'].fillna(0.0) * frac
    return (ds[var].fillna(0.0) * mass).sum('layer') / mass.sum('layer')


def main():
    ds = {k: xr.open_zarr(fn, consolidated=False) for k, fn in RUNS.items()}
    ref = ds['window5cm']
    rgiid = np.asarray(ref['rgiid'].values, dtype=str)
    weight = np.asarray(ref['weight'].values, dtype=np.float64)

    print('========== glacier-wide mass balance [m w.e. a-1], hydrological years ==========')
    annual = {k: d['mass_balance'].resample(time='YS-OCT').sum().compute() for k, d in ds.items()}
    print(f'{"glacier":<12} {"window5cm":>10} {"toplayer":>10} {"diff":>8}   worst year diff')
    for name in GLACIERS:
        gid = translate_rgi[name]['6']
        a = {k: glacierwide(v, rgiid, weight, gid).values for k, v in annual.items()}
        diff = a['window5cm'] - a['toplayer']
        print(f'{name:<12} {a["window5cm"].mean():>10.3f} {a["toplayer"].mean():>10.3f} '
              f'{diff.mean():>+8.3f}   {np.abs(diff).max():.3f}')

    print('\n========== summer (JJA) albedo, glacier-wide mean ==========')
    print(f'{"glacier":<12} {"window5cm":>10} {"toplayer":>10} {"diff":>8}')
    for name in GLACIERS:
        gid = translate_rgi[name]['6']
        m = {}
        for k, d in ds.items():
            alb = d['albedo'].sel(time=d['time.month'].isin(SUMMER_MONTHS))
            m[k] = float(glacierwide(alb.mean('time'), rgiid, weight, gid).compute())
        print(f'{name:<12} {m["window5cm"]:>10.3f} {m["toplayer"]:>10.3f} {m["window5cm"] - m["toplayer"]:>+8.3f}')

    gul = np.where(rgiid == translate_rgi['gulkana']['6'])[0]
    lat = np.asarray(ref['lat'].values)[gul]
    lon = np.asarray(ref['lon'].values)[gul]
    p = gul[np.argmin((lat - SITE_LAT) ** 2 + ((lon - SITE_LON) * np.cos(np.deg2rad(SITE_LAT))) ** 2)]
    print(f'\n========== Gulkana D, nearest point {p} '
          f'({float(ref["lat"][p]):.4f}, {float(ref["lon"][p]):.4f}, {float(ref["elev"][p]):.0f} m) ==========')
    print(f'measured top-3 cm dust on {SAMPLE_DATE}: {SAMPLES_PPM[0]} and {SAMPLES_PPM[1]} ppm')
    for k, d in ds.items():
        day = d.isel(point=p).sel(time=SAMPLE_DATE).compute()
        dust = '  '.join(f'top {int(z * 100)} cm {float(depth_mean(day, "layerdust", z)):7.1f} ppm'
                         for z in SAMPLE_DEPTHS)
        print(f'  {k:<10} {dust}  top layer {float(day["layerheight"].isel(layer=0)) * 100:.1f} cm  '
              f'albedo {float(day["albedo"]):.3f}')

    # August 2024 at the site, to see the run-up to the sample date
    print(f'\nAugust 2024 top-3 cm dust at Gulkana D [ppm], every 5 days:')
    for k, d in ds.items():
        aug = d.isel(point=p).sel(time=slice('2024-08-01', '2024-08-31')).compute()
        vals = depth_mean(aug, 'layerdust', 0.03).values[::5]
        print(f'  {k:<10} ' + ' '.join(f'{v:7.1f}' for v in vals))


if __name__ == '__main__':
    main()
