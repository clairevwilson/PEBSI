"""
Synthetic heatwaves inserted into the summer 2006 no-heatwave forcing at the MERRA-2 cell
holding Kennicott, one small forcing set per run for PEBSI's secondary climate folder.

The heatwave is the reverse of extreme_weather_analysis/prepare_no_heatwaves.py: on the hours that script replaced,
the removed anomaly is original minus no-heatwaves forcing. Its mean by local hour of day,
over every Kennicott heatwave hour, is the average Kennicott heatwave. Each run adds
intensity x that anomaly to T2M, SWGDN, LWGAB and QV2M for `duration` local days from
`start`, caps SWGDN at the clear-sky maximum, sets precipitation to zero over those days and
recomputes RH2M. Start, duration and intensity come from a Latin hypercube; run 0 has no event.

The design table also gives each event's intensity as defined in extreme_weather_analysis/detect_events.py: daily max
T2M above the 90th percentile day-of-year threshold, and the K day total over the days that
pass the 5-day heatwave criterion.
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import qmc
from scipy.ndimage import binary_erosion, binary_dilation

ROOT = '/ocean/projects/ees260009p/cwilson4/climate_data/MERRA2/'
base_fp = ROOT + 'reg01/{v}_reg01.zarr'
nohw_fp = ROOT + 'reg01_no_heatwaves/{v}_reg01_no_heatwaves.zarr'
out_fp = ROOT + 'reg01_kennicott_hw{run:03d}/{v}_reg01_kennicott_hw{run:03d}.zarr'
OUT_DIR = '/ocean/projects/ees260009p/cwilson4/Output/synthetic_heatwaves/'
DESIGN_FN = os.path.join(OUT_DIR, 'synthetic_heatwave_design.csv')
ANOMALY_FN = os.path.join(OUT_DIR, 'synthetic_heatwave_anomaly.csv')

# every Kennicott point falls in this cell
LAT, LON = 61.5, -143.125
UTC_OFFSET = pd.Timedelta(hours=8)
# covers the PEBSI run, 2005-04-01 to 2006-10-01 UTC
WINDOW = slice('2005-01-01', '2006-12-31')
YEAR = 2006

N_RUNS = 100
START_RANGE = ('2006-05-01', '2006-08-31')
DURATION_RANGE = (5, 20)  # days, inclusive
INTENSITY_RANGE = (0.5, 2)  # multiple of the average Kennicott heatwave
SEED = 0
HW_MIN_DAYS = 5

PERTURB_VARS = ['T2M', 'SWGDN', 'LWGAB', 'QV2M']
OUT_VARS = ['T2M', 'SWGDN', 'LWGAB', 'RH2M', 'PRECTOTCORR']


def load_cell(fp, var):
    """Hourly series at the Kennicott cell, indexed in local time."""
    da = xr.open_zarr(fp.format(v=var))[var].sel(lat=LAT, lon=LON, method='nearest')
    s = da.to_series().astype(np.float64)
    s.index = s.index - UTC_OFFSET
    return s


def compute_doy_threshold(daily_max, percentile=90, half_window=7):
    """(366,) percentile of daily values within +/- half_window days of each day of year."""
    doys = daily_max.index.dayofyear.values
    thresh = np.empty(366)
    for d in range(1, 367):
        dist = np.minimum(np.abs(doys - d), 365 - np.abs(doys - d))
        thresh[d - 1] = np.nanpercentile(daily_max.values[dist <= half_window], percentile)
    return thresh


def relative_humidity(temp, qv, ps):
    """RH [%] with the formula PEBSI uses to build RH2M from MERRA-2."""
    temp_c = temp - 273.15
    esat = 0.61094 * np.exp(17.625 * temp_c / (temp_c + 243.04)) * 1000
    ws = 0.622 * esat / (ps - esat)
    w = qv / (1 - qv)
    return w / ws * 100


def sample_design():
    """Run 0 without an event, then N_RUNS Latin hypercube samples of start, duration, intensity."""
    u = qmc.LatinHypercube(d=3, seed=SEED).random(N_RUNS)
    start0, start1 = pd.Timestamp(START_RANGE[0]), pd.Timestamp(START_RANGE[1])
    n_starts = (start1 - start0).days + 1
    n_durations = DURATION_RANGE[1] - DURATION_RANGE[0] + 1
    design = pd.DataFrame({
        'start': start0 + pd.to_timedelta(np.floor(u[:, 0] * n_starts), unit='D'),
        'duration': DURATION_RANGE[0] + np.floor(u[:, 1] * n_durations).astype(int),
        'intensity': INTENSITY_RANGE[0] + u[:, 2] * (INTENSITY_RANGE[1] - INTENSITY_RANGE[0]),
    })
    no_event = pd.DataFrame({'start': [pd.NaT], 'duration': [0], 'intensity': [0.0]})
    design = pd.concat([no_event, design], ignore_index=True)
    design.index.name = 'run'
    return design


def heatwave_days(exceeds):
    """Days in runs of at least HW_MIN_DAYS consecutive exceedances."""
    struct = np.ones(HW_MIN_DAYS, dtype=bool)
    return binary_dilation(binary_erosion(exceeds, structure=struct), structure=struct) & exceeds


def write_run(run, forcing, attrs, run_fp):
    """Writes each output variable as a (time, 1, 1) zarr in UTC, at run_fp formatted with run and var."""
    times = (forcing.index + UTC_OFFSET).values
    for var in OUT_VARS:
        da = xr.DataArray(forcing[var].values.astype(np.float32)[:, None, None],
                          dims=('time', 'lat', 'lon'),
                          coords={'time': times, 'lat': [LAT], 'lon': [LON]},
                          attrs=attrs[var], name=var)
        fn = run_fp.format(run=run, v=var)
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        da.to_dataset().to_zarr(fn, mode='w', zarr_format=2)


def main(design, run_fp, design_fn):
    """Writes the forcing of every run in design to run_fp and the design table to design_fn."""
    print("Loading Kennicott cell...")
    orig = pd.DataFrame({v: load_cell(base_fp, v) for v in PERTURB_VARS + ['PS', 'PRECTOTCORR']})
    nohw = pd.DataFrame({v: load_cell(nohw_fp, v) for v in PERTURB_VARS})
    orig = orig.loc[nohw.index]
    attrs = {v: xr.open_zarr(base_fp.format(v=v))[v].attrs for v in OUT_VARS}

    # average Kennicott heatwave by local hour: what prepare_no_heatwaves.py removed
    removed = orig[PERTURB_VARS] - nohw
    is_hw = removed['T2M'] != 0
    anomaly = removed[is_hw].groupby(removed[is_hw].index.hour).mean()
    anomaly.index.name = 'local_hour'
    anomaly.to_csv(ANOMALY_FN)
    n_hw_days = is_hw.groupby(is_hw.index.normalize()).any().sum()
    print(f"Average heatwave from {n_hw_days} days; daily mean anomaly:")
    print(anomaly.mean().to_string())

    # clear-sky cap: highest SWGDN in the record at that local hour within +/- 7 days of year
    sw = orig['SWGDN']
    sw_cap = sw.groupby([sw.index.dayofyear, sw.index.hour]).max().unstack()
    sw_cap = sw_cap.rolling(15, center=True, min_periods=1).max()

    # heatwave threshold as in prepare_no_heatwaves.py: daily max over May-Sep local days
    temp_daily = orig['T2M'].resample('1D').max()
    temp_thresh = compute_doy_threshold(temp_daily[temp_daily.index.month.isin(range(5, 10))])

    base = nohw.loc[WINDOW].copy()
    base['PS'] = orig.loc[WINDOW, 'PS']
    base['PRECTOTCORR'] = orig.loc[WINDOW, 'PRECTOTCORR']
    hours = base.index.hour
    season = pd.date_range(f'{YEAR}-05-01', f'{YEAR}-09-30', freq='D')
    season_thresh = temp_thresh[season.dayofyear - 1]

    stats = []
    for run, (start, duration, intensity) in design.iterrows():
        forcing = base.copy()
        if duration > 0:
            event = (base.index >= start) & (base.index < start + pd.Timedelta(days=duration))
            for var in PERTURB_VARS:
                forcing.loc[event, var] += intensity * anomaly.loc[hours[event], var].values
            cap = sw_cap.values[base.index.dayofyear[event] - 1, hours[event]]
            forcing.loc[event, 'SWGDN'] = np.clip(forcing.loc[event, 'SWGDN'], 0, cap)
            forcing.loc[event, 'PRECTOTCORR'] = 0
        forcing['RH2M'] = relative_humidity(forcing['T2M'], forcing['QV2M'], forcing['PS'])
        write_run(run, forcing, attrs, run_fp)

        # event intensity in the detect_events.py sense, over the 2006 season
        exceedance = forcing['T2M'].resample('1D').max().loc[season].values - season_thresh
        hw = heatwave_days(exceedance > 0)
        # run 0's start is NaT, so no day falls in its event
        in_event = (season >= start) & (season < start + pd.Timedelta(days=duration))
        stats.append({
            'mean_exceedance': exceedance[in_event].mean() if in_event.any() else np.nan,
            'heatwave_days': int(hw.sum()),
            'heatwave_Kday': float(exceedance[hw].sum()),
        })
        print(f"  run {run:03d}: start {start.date() if duration > 0 else '-'}, "
              f"{duration} d, intensity {intensity:.2f}, "
              f"{stats[-1]['heatwave_days']} heatwave days", flush=True)

    design = design.join(pd.DataFrame(stats, index=design.index))
    design.to_csv(design_fn)
    print(f"Saved {design_fn}")


if __name__ == '__main__':
    main(sample_design(), out_fp, DESIGN_FN)
