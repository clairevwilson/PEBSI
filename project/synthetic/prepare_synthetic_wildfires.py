"""
Average wildfire of every MERRA-2 cell in CELLS, for PEBSI to add on the fly as a synthetic
wildfire (wildfire_* in pebsi/defaults.py).

`python prepare_synthetic_wildfires.py anomaly` saves it from UK-ESM deposition (GFED minus
no-fires) to ANOMALY_NC.
`python prepare_synthetic_wildfires.py anomaly merra2` does the same for MERRA-2 deposition
(MERRA2_ANOMALY_NC), from the wildfires extreme_weather_analysis/prepare_no_wildfires.py removed.
"""
import sys
import socket
import numpy as np
import pandas as pd
import xarray as xr

if 'forge' in socket.gethostname():
    ROOT = '/home/claire/Research/climate_data/'
else:
    ROOT = '/ocean/projects/ees260009p/cwilson4/climate_data/'
ANOMALY_NC = ROOT + 'UKESM/wildfire_anomaly.nc'
MERRA2_ANOMALY_NC = ROOT + 'MERRA2/reg01_no_wildfires/wildfire_anomaly.nc'
MERRA2_EVENTS_FN = ROOT + 'extreme_events_daily_merra2_wildfire_1day.nc'
MERRA2_FP = ROOT + 'MERRA2/{d}/{v}_{d}.zarr'
MERRA2_VARS = {'bcdry': 'BCDP002', 'bcwet': 'BCWT002', 'ocdry': 'OCDP002', 'ocwet': 'OCWT002'}
# MERRA-2 cells of gulkana, lemon_creek, wolverine, kennicott and kahiltna (two cells)
CELLS = [(63.5, -145.625), (58.5, -134.375), (60.5, -148.75), (61.5, -143.125),
         (62.5, -151.25), (63.0, -151.25)]
ukesm_fp = ROOT + 'UKESM/{run}/sum_{sp}_{t}deposition_kgm-2s-1.nc'
EVENTS_FN = ROOT + 'extreme_events_daily_ukesm.nc'

UKESM_VN = 'tendency_of_atmosphere_mass_content_of_{sp}_dry_aerosol_particles_due_to_{t}_deposition'
UKESM_SPECIES = {'bc': 'elemental_carbon', 'oc': 'particulate_organic_matter'}
COMPONENTS = [(sp, t) for sp in ['bc', 'oc'] for t in ['dry', 'wet']]
BASE_RUN = 'dr401_GFED'
NOFIRE_RUN = 'dw068_nofires'


def open_component(run, sp, t):
    """UK-ESM deposition for one run and component, as a DataArray."""
    vn = UKESM_VN.format(sp=UKESM_SPECIES[sp], t=t)
    return xr.open_dataset(ukesm_fp.format(run=run, sp=sp, t=t), decode_timedelta=False)[vn]


def save_anomaly():
    """
    Average fire of each cell in CELLS: fire-attributable UK-ESM deposition (GFED minus no-fires)
    of each component at the UK-ESM cell nearest it, averaged over the cell's wildfire days.
    """
    events = xr.open_dataset(EVENTS_FN)['wildfire']
    lats = sorted({c[0] for c in CELLS})
    lons = sorted({c[1] for c in CELLS})
    ds = xr.Dataset(coords={'lat': lats, 'lon': lons})
    for sp, t in COMPONENTS:
        diff = open_component(BASE_RUN, sp, t) - open_component(NOFIRE_RUN, sp, t)
        values = np.full((len(lats), len(lons)), np.nan)
        for la, lo in CELLS:
            fire_days = events.sel(y=la, x=lo, method='nearest').to_series()
            fire_days = fire_days.index[fire_days.values.astype(bool)]
            cell = diff.sel(latitude=la, longitude=lo % 360, method='nearest').to_series()
            # the events file labels each UK-ESM day by its midnight
            cell.index = cell.index - pd.Timedelta(hours=12)
            values[lats.index(la), lons.index(lo)] = cell.loc[fire_days].mean()
        ds[f'{sp}{t}'] = (('lat', 'lon'), values)
        ds[f'{sp}{t}'].attrs = {'units': 'kg m-2 s-1'}
    ds.attrs['description'] = (f'Average daily fire-attributable deposition ({BASE_RUN} minus '
                               f'{NOFIRE_RUN}) over each cell\'s wildfire days, at the nearest UK-ESM cell')
    ds.to_netcdf(ANOMALY_NC)
    print(ds)
    print(f"Saved {ANOMALY_NC}")


def save_merra2_anomaly():
    """
    Average fire of each cell in CELLS for MERRA-2 deposition: fire-attributable deposition (reg01
    minus reg01_no_wildfires) of each component, as daily means of the raw representative bin
    (before PEBSI's ratio to total deposition), averaged over the cell's wildfire days.
    """
    events = xr.open_dataset(MERRA2_EVENTS_FN)['wildfire']
    lats = sorted({c[0] for c in CELLS})
    lons = sorted({c[1] for c in CELLS})
    ds = xr.Dataset(coords={'lat': lats, 'lon': lons})
    for name, vn in MERRA2_VARS.items():
        values = np.full((len(lats), len(lons)), np.nan)
        for la, lo in CELLS:
            fire_days = events.sel(y=la, x=lo).to_series()
            fire_days = fire_days.index[fire_days.values.astype(bool)]
            diff = 0
            for d in ['reg01', 'reg01_no_wildfires']:
                s = xr.open_zarr(MERRA2_FP.format(d=d, v=vn), consolidated=False)[vn].sel(lat=la, lon=lo).to_series()
                diff = s if d == 'reg01' else diff - s
            # the events file labels UTC days by their midnight
            daily = diff.groupby(diff.index.normalize()).mean()
            values[lats.index(la), lons.index(lo)] = daily.loc[fire_days].mean()
        ds[name] = (('lat', 'lon'), values)
        ds[name].attrs = {'units': 'kg m-2 s-1'}
    ds.attrs['description'] = ('Average daily fire-attributable MERRA-2 deposition (reg01 minus reg01_no_wildfires, '
                               'representative bin) over each cell\'s wildfire days')
    ds.to_netcdf(MERRA2_ANOMALY_NC)
    print(ds)
    print(f"Saved {MERRA2_ANOMALY_NC}")


if __name__ == '__main__':
    if sys.argv[1:] == ['anomaly']:
        save_anomaly()
    elif sys.argv[1:] == ['anomaly', 'merra2']:
        save_merra2_anomaly()
