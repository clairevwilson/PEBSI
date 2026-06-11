"""
Prepares a netCDF file containing the data
needed for quantile data on the regional scale
for multiple variables.

"""
import os 
import pandas as pd
import numpy as np
import xarray as xr
from netCDF4 import Dataset as ncDataset

# ========= WEATHER STATION INFO =========
master_dict = {
    'kahiltna': {
        1: {'lat': 62.94225, 'lon':-151.29205, 'elev':2380, 'time_col':'UTC_time', 'timezone':0, 'type': 'benchmark',
            'variables': {'wind': 'm s-1', 'temp':'C'}},
    },
    'gulkana': {
        1: {'lat':63.28180, 'lon':-145.42631, 'elev':1725, 'time_col':'UTC_time', 'timezone':0, 'type': 'benchmark',
            'variables': {'SWin': 'W m-2','LWin': 'W m-2','temp':'C','rh':'%'}},
        2: {'lat':63.285514, 'lon':-145.410336, 'elev':1682, 'time_col':'TIMESTAMP', 'timezone':-8, 'type': '-', 'variables': {'wind': 'm s-1'},
            'fn': '../../../climate_data/AWS/Processed/gulkana/gulkana2024.csv'},
        # 3: {'lat':63.26137, 'lon':-145.41021, 'elev':1480, 'time_col':'UTC_time', 'timezone':0, 'variables': {'temp': 'C','wind': 'm s-1'}}, 
    },
    'wolverine': {
        1: {'lat':60.38192, 'lon':-148.93966, 'elev':990, 'time_col':'UTC_time', 'timezone':0, 'type': 'benchmark',
            'variables': {'temp': 'C','wind': 'm s-1','rh': '%','sp': 'hPa'}},
        2: {'lat':60.39486, 'lon':-148.94524, 'elev':1420, 'time_col':'local_time', 'timezone':-8, 'type': 'benchmark',
            'variables': {'SWin': 'W m-2'}},
    },
    'lemon_creek':{
        1: {'lat':58.36756, 'lon':-134.36627, 'elev':1280, 'time_col':'local_time', 'timezone':-8, 'type': 'benchmark',
            'variables': {'temp': 'C','rh': '%','sp': 'hPa'}},
    }
}

benchmark_fp = '../../../climate_data/AWS/Benchmark/{g}/'
benchmark_fn = '{g}{e}_hourly_LVL2_{y}.csv'
benchmark_var_names = {'temp': 'site_temp_USGS', 
                       'SWin': 'ShortwaveIn',
                       'LWin': 'LongwaveIn',
                       'rh': 'RelHum', 
                       'wind': 'WindSpeed', 
                       'sp': 'Barom'}

merra2_fn = '/Volumes/TOSHIBA EXT/MERRA2/{v}/{v}_reg01.zarr'
merra2_var_names = {'temp': 'T2M','rh': 'RH2M',
                    'SWin': 'SWGDN', 'LWin': 'LWGAB',
                    'wind': 'U2M', 'sp': 'PS',}

general_var_names = {'temp': 'temp', 'wind': 'uwind'}

expected_units = {'temp':'C', 'wind': 'm s-1', 'SWin': 'J m-2', 'LWin': 'J m-2', 'rh': '%', 'sp': 'Pa'}

# Define unit conversion for units of MERRA-2 and AWS data
def MERRA2_unit_conversion(data, units_in, units_out):
    if units_in == 'K':
        return data - 273.15
    elif units_in == 'W m-2':
        return data * 3600
    elif units_in != units_out:
        print(f'Warning! Units of MERRA-2 data may not be consistent. Input was {units_in}')
    return data
    
def AWS_unit_conversion(data, units_in, units_out):
    if units_in == 'W m-2':
        return data * 3600
    elif units_in == 'hPa':
        return data * 100
    elif units_in != units_out:
        print(f'Warning! Units of AWS data may not be consistent. Input was {units_in}')
    return data

storage = []
quantiles = np.linspace(0, 1, 1000) # break data into 1000 quantiles 

for glacier, stations in master_dict.items():
    for _, data in stations.items():
        lat = data['lat']
        lon = data['lon']
        elevation = data['elev']
        time_col = data['time_col']
        timezone = data['timezone']
        station_type = data['type']
        variables = data['variables']

        # load and concatenate all the files
        if station_type == 'benchmark':
            df = None
            fp = benchmark_fp.format(g=glacier.replace('_',''))
            if 'LVL2' not in os.listdir(fp):
                fp = fp + str(elevation) + '/LVL2/'
            else:
                fp += 'LVL2/'
            for fn in os.listdir(fp):
                if 'hourly' in fn and str(elevation) in fn:
                    _df = pd.read_csv(fp + fn)
                    if df is None:
                        df = _df.copy()
                    else:
                        df = pd.concat([df, _df])
        else:
            fn_data = data['fn']
            assert os.path.exists(fn_data), f'Data not found at {fn_data}'
            df = pd.read_csv(fn_data)

        # set index as datetimes in UTC
        df.index = pd.to_datetime(df[time_col], format='mixed') - pd.Timedelta(hours=timezone)
        df = df.sort_index()

        for var, aws_units in variables.items():
            var_merra = merra2_var_names[var]
            if station_type == 'benchmark':
                var_aws = benchmark_var_names[var]
            else:
                var_aws = general_var_names[var]

            data_aws = df[var_aws]
            data_merra = xr.open_zarr(merra2_fn.format(v=var_merra), consolidated=False)
            data_merra = data_merra.sel(lat = lat, lon = lon, method='nearest')[var_merra]

            if var == 'wind':
                data_v2m = xr.open_zarr(merra2_fn.format(v='V2M'))
                data_v2m = data_v2m.sel(lat = lat, lon = lon, method='nearest')['V2M']
                data_merra.values = np.sqrt(data_v2m.values **2 + data_merra.values **2)
            
            # crop data to common dates
            first_date = max(pd.to_datetime(data_aws.index[0]), 
                             pd.to_datetime(data_merra.time.values[0]))
            last_date = min(pd.to_datetime(data_aws.index[-1]),
                            pd.to_datetime(data_merra.time.values[-1]))
            data_aws = data_aws.loc[slice(first_date, last_date)]
            data_merra = data_merra.sel(time=slice(first_date, last_date))

            # set the MERRA-2 timestamps a half hour back (they come as 12:30, 1:30, etc.)
            data_merra['time'] = data_merra['time'] - pd.Timedelta(minutes=30)

            # find common timestamps
            common = df.index.intersection(data_merra['time'].to_index())
            values_merra = data_merra.sel(time=common).values
            values_aws = data_aws.loc[common].values

            # clip out nans
            values_merra = values_merra[np.isfinite(values_aws)]
            values_aws = values_aws[np.isfinite(values_aws)]

            # handle units
            values_merra = MERRA2_unit_conversion(values_merra, data_merra.attrs['units'], expected_units[var])
            values_aws = AWS_unit_conversion(values_aws, aws_units, expected_units[var])

            # sort the data and store it in the dict
            quantiles_merra = np.quantile(values_merra, quantiles)
            quantiles_aws = np.quantile(values_aws, quantiles)

            storage.append({
                'glacier': glacier, 'elev': elevation, 'var': var,
                'aws_cdf': quantiles_aws, 
                'merra_cdf': quantiles_merra, 
                'lat': lat, 'lon': lon
            })

# write to netCDF
ds_dict = {}
for var in set(r['var'] for r in storage):
    var_record = [r for r in storage if r['var'] == var]
    stations = [f"{r['glacier']}_{r['elev']}" for r in var_record]
    ds = xr.Dataset(
        {
            'aws_cdf':   (('station', 'quantile'), np.array([r['aws_cdf']   for r in var_record])),
            'merra_cdf': (('station', 'quantile'), np.array([r['merra_cdf'] for r in var_record])),
        },
        coords={
            'station': stations,
            'quantile': quantiles,
            'lat': ('station', [r['lat'] for r in var_record]),
            'lon': ('station', [r['lon'] for r in var_record]),
            'elevation': ('station', [r['elev'] for r in var_record])
        },
    )
    ds_dict[var] = ds

# save each variable as a group
for var, ds in ds_dict.items():
    ds.to_netcdf('quantile_cdfs.nc', group=var, mode='w' if var == list(ds_dict)[0] else 'a')