"""
Prepares a netCDF file containing the data
needed for quantile data on the regional scale
for multiple variables.

"""
import os 
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

plot_quantiles = True

# ========= WEATHER STATION INFO =========
master_dict = {
    # WESTERN ALASKA
    'kahiltna': {
        1: {'lat': 62.94225, 'lon':-151.29205, 'elev':2380, 'time_col':'UTC_time', 'timezone':0, 'type': 'benchmark',
            'variables': {'wind': 'm s-1', 'temp':'C'}},
    },
    # EASTERN ALASKA
    'gulkana': {
        1: {'lat':63.28180, 'lon':-145.42631, 'elev':1725, 'time_col':'UTC_time', 'timezone':0, 'type': 'benchmark',
            'variables': {'SWin': 'W m-2','LWin': 'W m-2','temp':'C','rh':'%'}},
        2: {'lat':63.285514, 'lon':-145.410336, 'elev':1682, 'time_col':'TIMESTAMP', 'timezone':-8, 'type': '-', 'variables': {'wind': 'm s-1'},
            'fn': '../../../climate_data/AWS/Processed/gulkana/gulkana2024.csv'},
        # 3: {'lat':63.26137, 'lon':-145.41021, 'elev':1480, 'time_col':'UTC_time', 'timezone':0, 'variables': {'temp': 'C','wind': 'm s-1'}}, 
    },
    # KENAI
    'wolverine': {
        1: {'lat':60.38192, 'lon':-148.93966, 'elev':990, 'time_col':'UTC_time', 'timezone':0, 'type': 'benchmark',
            'variables': {'temp': 'C','wind': 'm s-1','rh': '%','sp': 'kPa'}},
        2: {'lat':60.39486, 'lon':-148.94524, 'elev':1420, 'time_col':'local_time', 'timezone':-8, 'type': 'benchmark',
            'variables': {'SWin': 'W m-2'}},
    },
    # COASTAL
    'lemon_creek':{
        1: {'lat':58.36756, 'lon':-134.36627, 'elev':1280, 'time_col':'local_time', 'timezone':-8, 'type': 'benchmark',
            'variables': {'temp': 'C','rh': '%','sp': 'kPa'}},
    },
    # ST ELIAS
    'kaskawulsh':{
        1: {'lat': 60.7421, 'lon': -139.1659, 'elev': 1800, 'time_col':'Unnamed: 0', 'timezone':0, 'type': '-', 
            'variables': {'temp': 'C', 'wind': 'm s-1', 'SWin': 'W m-2', 'LWin': 'W m-2', 'rh': '%', 'sp': 'mbar'},
            'fn': '../../../climate_data/AWS/Raw/kaskawulsh/preprocessed_2019.csv'},
    },
    'lowell':{
        1: {'lat': 60.30271, 'lon': -138.57565, 'elev': 1419, 'time_col':'Unnamed: 0', 'timezone':0, 'type': '', 
            'variables': {'temp': 'C', 'rh': '%'},
            'fn': '../../../climate_data/AWS/Raw/lowell/preprocessed_all.csv'}
    },
    # WRANGELLS
    'gates':{
        1: {'lat': 61.6029, 'lon': -143.0132, 'elev': 1237, 'time_col':'Unnamed: 0', 'timezone':0, 'type': '', 
            'variables': {'temp': 'C', 'wind': 'm s-1', 'rh': '%', 'SWin': 'W m-2'},
            'fn': '../../../climate_data/AWS/Raw/gates/preprocessed_all.csv'}
    },
    # BROOKS (McCALL)
    # TALKEETNA (? - don't think there's nay here, but there ARE high elevation stations))

    'dummy':{
        1: {'lat': 0, 'lon': 0, 'elev': 0, 'time_col':'', 'timezone':0, 'type': '', 'variables': {'temp': 'C'}}
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

merra2_fn = '/Volumes/TOSHIBA EXT/climate_data/MERRA2/reg01/{v}_reg01.zarr'
merra2_var_names = {'temp': 'T2M','rh': 'RH2M',
                    'SWin': 'SWGDN', 'LWin': 'LWGAB',
                    'wind': 'U2M', 'sp': 'PS',}

general_var_names = {'temp': 'Temperature', 'sp': 'Pressure', 'wind': 'uwind'}

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
    elif units_in in ['hPa', 'mbar']:
        return data * 100
    elif units_in == 'kPa':
        return data * 1000
    elif units_in != units_out:
        print(f'Warning! Units of AWS data may not be consistent. Input was {units_in}')
    return data

storage = []
quantiles = np.linspace(0, 1, 1000) # break data into 1000 quantiles 

for glacier, stations in master_dict.items():
    if glacier == 'dummy': continue
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
            elif var in df.columns:
                var_aws = var 
            elif var.capitalize() in df.columns:
                var_aws = var.capitalize()
            elif var.upper() in df.columns:
                var_aws = var.upper()
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
            not_na = np.isfinite(values_aws) & np.isfinite(values_merra)
            values_merra = values_merra[not_na]
            values_aws = values_aws[not_na]

            # handle units
            values_merra = MERRA2_unit_conversion(values_merra, data_merra.attrs['units'], expected_units[var])
            values_aws = AWS_unit_conversion(values_aws, aws_units, expected_units[var])

            # handle SW data differently: remove zeros 
            if var == 'SWin':
                # mask anything that is effectively zero
                nonzero_mask = ((values_aws > 100) & (values_merra > 100))
                values_aws = values_aws[nonzero_mask]
                values_merra = values_merra[nonzero_mask]

            # sort the data and store it in the dict
            quantiles_merra = np.quantile(values_merra, quantiles)
            quantiles_aws = np.quantile(values_aws, quantiles)

            if plot_quantiles:
                plt.figure(figsize=(3, 3))
                min_val = np.min([values_merra, values_aws])
                max_val = np.max([values_merra, values_aws])

                plt.hist(quantiles_merra, bins=np.linspace(min_val, max_val, 50),
                         histtype='step', edgecolor='r', label='MERRA-2')
                plt.hist(quantiles_aws, bins=np.linspace(min_val, max_val, 50),
                         histtype='step', edgecolor='k', label='AWS')
                plt.legend()

                plt.title(f"{glacier.replace('_', ' ').capitalize()} {var}")

                os.makedirs('../../../figs/qm/', exist_ok=True)
                plt.savefig(f'../../../figs/qm/{var}_{glacier}_hist.png', dpi=300, bbox_inches='tight')
                plt.close()

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
    ds.to_netcdf('../../data/quantile_cdfs.nc', group=var, mode='w' if var == list(ds_dict)[0] else 'a')