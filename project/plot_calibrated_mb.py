from data_handling import MassBalance, translate_rgi, plt
import xarray as xr
import os
import pickle
import yaml
import dask

dask.config.set(scheduler='synchronous')

with open('calibrated_grid.pkl', 'rb') as f:
    calibrated = pickle.load(f)

rgi_map = {}
for glacier, numbers in translate_rgi.items():
    rgi_map[numbers['6']] = glacier

output_fp = '/ocean/projects/ees260009p/cwilson4/Output/gridsearch_3/'
ds_all = xr.open_zarr(output_fp + 'output.zarr')

# this store isn't chunked by point (chunks span all points), so a fresh
# .sel(point=idx) per site would re-decompress a whole time-window x
# all-points chunk each time; load once instead
ds_all = ds_all[['melt', 'accumulation', 'refreeze']].load()

with open(output_fp + 'config_gridsearch.yaml') as f:
    config = yaml.safe_load(f)

idx_by_site = {}
for i in range(len(config['sites'])):
    key = (config['rgi_ids'][i], config['sites'][i])
    idx_by_site.setdefault(key, []).append(i)

plot_fp = output_fp + 'mb_plots/'
os.makedirs(plot_fp, exist_ok=True)

for rgiid, sites in calibrated.items():
    for site, params in sites.items():
        idx = next(i for i in idx_by_site[(rgiid, site)]
                   if config['kp'][i] == params['kp'] and config['wind_factor'][i] == params['kw'])

        mb = MassBalance(rgi_map[rgiid], site)
        ds = ds_all.sel(point=idx)[['melt', 'accumulation', 'refreeze']]
        mb.get_model_mb(ds)

        fig, ax = mb.plot_mb(savefig=f'{plot_fp}{rgiid}_{site}.png')
        plt.close(fig)
