"""
Synthetic wildfires inserted into the summer 2006 UK-ESM no-fires deposition at Kennicott,
one small set of deposition files per run for PEBSI's ukesm_fp.

The average Kennicott fire is the fire-attributable deposition (GFED minus no-fires) of BC and
OC, wet and dry, averaged over the days extreme_weather_analysis/detect_events.py flags as wildfires at Kennicott. Each
run adds intensity x that daily deposition for `duration` days from `start`. Start, duration and
intensity come from a Latin hypercube, with intensity sampled on a log scale; run 0 has no event.

The design table gives each event's total fire-attributable BC+OC deposition as PEBSI sees it,
after its UK-ESM to MERRA-2 ratio, in mg m-2.
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import qmc

ROOT = '/ocean/projects/ees260009p/cwilson4/climate_data/'
ukesm_fp = ROOT + 'UKESM/{run}/sum_{sp}_{t}deposition_kgm-2s-1.nc'
out_fp = ROOT + 'UKESM/kennicott_fire{run:03d}/sum_{sp}_{t}deposition_kgm-2s-1.nc'
ratio_fp = ROOT + 'ukesm_merra2_reg01_{sp}{t}.nc'
EVENTS_FN = ROOT + 'extreme_events_daily_ukesm.nc'
OUT_DIR = '/ocean/projects/ees260009p/cwilson4/Output/synthetic_wildfires/'
DESIGN_FN = os.path.join(OUT_DIR, 'synthetic_wildfire_design.csv')
ANOMALY_FN = os.path.join(OUT_DIR, 'synthetic_wildfire_anomaly.csv')

UKESM_VN = 'tendency_of_atmosphere_mass_content_of_{sp}_dry_aerosol_particles_due_to_{t}_deposition'
UKESM_SPECIES = {'bc': 'elemental_carbon', 'oc': 'particulate_organic_matter'}
COMPONENTS = [(sp, t) for sp in ['bc', 'oc'] for t in ['dry', 'wet']]
BASE_RUN = 'dr401_GFED'
NOFIRE_RUN = 'dw068_nofires'

# MERRA-2 cell holding Kennicott; the UK-ESM cell nearest it is the one PEBSI reads
LAT, LON = 61.5, -143.125
# PEBSI reads the grid spacing from neighbouring cells, so each file keeps a 3x3 block
HALF_BLOCK = 1
# daily values stamped at 12:00 UTC; covers the PEBSI run, 2005-04-01 to 2006-10-01
WINDOW = slice('2005-01-01', '2006-12-31')

N_RUNS = 100
START_RANGE = ('2006-05-01', '2006-08-31')
DURATION_RANGE = (3, 20)  # days, inclusive; 3 is the wildfire minimum in detect_events.py
INTENSITY_RANGE = (0.5, 10)  # multiple of the average Kennicott fire, sampled log-uniformly
SEED = 0


def open_component(run, sp, t):
    """UK-ESM deposition for one run and component, as a DataArray."""
    vn = UKESM_VN.format(sp=UKESM_SPECIES[sp], t=t)
    return xr.open_dataset(ukesm_fp.format(run=run, sp=sp, t=t), decode_timedelta=False)[vn]


def block_indices(da):
    """Latitude/longitude index slices of the 3x3 block centred on the cell nearest Kennicott."""
    i = int(np.abs(da.latitude.values - LAT).argmin())
    j = int(np.abs(da.longitude.values - LON % 360).argmin())
    return (slice(i - HALF_BLOCK, i + HALF_BLOCK + 1), slice(j - HALF_BLOCK, j + HALF_BLOCK + 1))


def sample_design():
    """Run 0 without an event, then N_RUNS Latin hypercube samples of start, duration, intensity."""
    u = qmc.LatinHypercube(d=3, seed=SEED).random(N_RUNS)
    start0, start1 = pd.Timestamp(START_RANGE[0]), pd.Timestamp(START_RANGE[1])
    n_starts = (start1 - start0).days + 1
    n_durations = DURATION_RANGE[1] - DURATION_RANGE[0] + 1
    low, high = INTENSITY_RANGE
    design = pd.DataFrame({
        'start': start0 + pd.to_timedelta(np.floor(u[:, 0] * n_starts), unit='D'),
        'duration': DURATION_RANGE[0] + np.floor(u[:, 1] * n_durations).astype(int),
        'intensity': low * (high / low) ** u[:, 2],
    })
    no_event = pd.DataFrame({'start': [pd.NaT], 'duration': [0], 'intensity': [0.0]})
    design = pd.concat([no_event, design], ignore_index=True)
    design.index.name = 'run'
    return design


def main(design, run_fp, design_fn):
    """Writes the deposition of every run in design to run_fp and the design table to design_fn."""
    fire_days = xr.open_dataset(EVENTS_FN)['wildfire'].sel(y=LAT, x=LON, method='nearest').to_series()
    fire_days = fire_days.index[fire_days.values.astype(bool)]
    print(f"{len(fire_days)} wildfire days at Kennicott")

    base, anomaly, ratio, attrs = {}, {}, {}, {}
    for sp, t in COMPONENTS:
        gfed = open_component(BASE_RUN, sp, t)
        nofire = open_component(NOFIRE_RUN, sp, t)
        lat_block, lon_block = block_indices(nofire)
        gfed = gfed.isel(latitude=lat_block, longitude=lon_block).load()
        nofire = nofire.isel(latitude=lat_block, longitude=lon_block).load()
        centre = dict(latitude=HALF_BLOCK, longitude=HALF_BLOCK)

        # the events file labels each UK-ESM day by its midnight
        diff = (gfed - nofire).isel(centre).to_series()
        diff.index = diff.index - pd.Timedelta(hours=12)
        anomaly[(sp, t)] = diff.loc[fire_days].mean()

        base[(sp, t)] = nofire.sel(time=WINDOW)
        attrs[(sp, t)] = nofire.attrs
        lat, lon = float(nofire.latitude[HALF_BLOCK]), float(nofire.longitude[HALF_BLOCK])
        ratio[(sp, t)] = float(xr.open_dataarray(ratio_fp.format(sp=sp, t=t))
                               .sel(lat=LAT, lon=LON, method='nearest'))
    print(f"UK-ESM cell {lat}, {lon}")

    anomaly_table = pd.DataFrame({
        'ukesm_kgm-2s-1': {f'{sp}{t}': anomaly[(sp, t)] for sp, t in COMPONENTS},
        'ratio': {f'{sp}{t}': ratio[(sp, t)] for sp, t in COMPONENTS},
    })
    # daily fire-attributable deposition as PEBSI sees it
    anomaly_table['mgm-2d-1'] = anomaly_table['ukesm_kgm-2s-1'] * anomaly_table['ratio'] * 86400 * 1e6
    anomaly_table.to_csv(ANOMALY_FN)
    print(anomaly_table.to_string())

    times = base[COMPONENTS[0]].time.values
    days = pd.DatetimeIndex(times) - pd.Timedelta(hours=12)

    totals = []
    for run, (start, duration, intensity) in design.iterrows():
        # run 0's start is NaT, so no day falls in its event
        event = (days >= start) & (days < start + pd.Timedelta(days=duration))
        total = 0
        for sp, t in COMPONENTS:
            da = base[(sp, t)].copy()
            added = intensity * anomaly[(sp, t)] * event
            da.values = np.maximum(da.values + added[:, None, None], 0)
            centre_added = (da - base[(sp, t)]).isel(latitude=HALF_BLOCK, longitude=HALF_BLOCK)
            total += float(centre_added.sum()) * ratio[(sp, t)] * 86400 * 1e6

            da.attrs = attrs[(sp, t)]
            fn = run_fp.format(run=run, sp=sp, t=t)
            os.makedirs(os.path.dirname(fn), exist_ok=True)
            da.to_dataset().to_netcdf(fn)
        totals.append(total)
        print(f"  run {run:03d}: start {start.date() if duration > 0 else '-'}, {duration} d, "
              f"intensity {intensity:.2f}, {total:.1f} mg m-2", flush=True)

    design['total_deposition'] = totals
    design.to_csv(design_fn)
    print(f"Saved {design_fn}")


if __name__ == '__main__':
    main(sample_design(), out_fp, DESIGN_FN)
