from data_handling import MassBalance, translate_rgi
import xarray as xr
import pandas as pd
import numpy as np
import os
import pickle
import yaml
import time
import dask
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
dask.config.set(scheduler='synchronous')

with open('sites.pkl', 'rb') as f:
    site_dict = pickle.load(f)

rgi_map = {}
for glacier, numbers in translate_rgi.items():
    rgi_map[numbers['6']] = glacier

output_fp = '/ocean/projects/ees260009p/cwilson4/Output/gridsearch_dense_2/'
ds_all = xr.open_zarr(output_fp + 'output.zarr')

# --- TEMP: this store isn't chunked by point (chunks span all points), so
# every .sel(point=idx) below decompresses a whole time-window x all-points
# chunk. Load the needed variables into memory once, up front, instead of
# repeating that full-chunk read per point. Delete this block once reading
# from output.zarr generated with the new point-chunked rechunk_final().
# print('loading dataset into memory (unchunked-by-point workaround)...', flush=True)
ds_all = ds_all[['melt', 'accumulation', 'refreeze', 'albedo']].load()
print('done loading', flush=True)
# --- END TEMP

model_t0 = ds_all.time.values[0]
model_t1 = ds_all.time.values[-1]

with open(output_fp + 'config_gridsearch.yaml') as f:
    config = yaml.safe_load(f)

# precompute site/rgiid -> matching config indices once, instead of
# rescanning the full config list for every (rgiid, site) pair
idx_by_site = {}
for i in range(len(config['sites'])):
    key = (config['rgi_ids'][i], config['sites'][i])
    idx_by_site.setdefault(key, []).append(i)


def process_site(rgiid, site):
    t0 = time.perf_counter()
    mb = MassBalance(rgi_map[rgiid], site)
    init_s = time.perf_counter() - t0

    if mb.dataset != 'seasonal':
        # print(f'  skipping {rgiid} {site}: no seasonal data (dataset={mb.dataset})', flush=True)
        return None

    # get_model_mb() needs >=2 benchmark periods overlapping the model's
    # simulated time range to compute anything; below that it silently
    # sets mod/meas=[nan] without setting idx_summer/idx_winter, so check
    # here instead of crashing inside mae()
    n_valid = np.sum((mb.period_starts >= model_t0) & (mb.period_ends <= model_t1))
    if n_valid < 2:
        print(f'  skipping {rgiid} {site}: only {n_valid} benchmark periods overlap '
              f'model time range [{model_t0}, {model_t1}]', flush=True)
        return None

    idx_site = idx_by_site.get((rgiid, site), [])

    winters = []
    summers = []
    io_s = 0.0
    compute_s = 0.0

    for n, idx in enumerate(idx_site):
        t1 = time.perf_counter()
        ds = ds_all.sel(point=idx) # [['melt', 'accumulation', 'refreeze']]
        io_s += time.perf_counter() - t1

        t2 = time.perf_counter()
        mb.get_model_mb(ds)
        summer, winter = mb.mae(True)
        compute_s += time.perf_counter() - t2

        summers.append(summer)
        winters.append(winter)

        # print(f'    {rgiid} {site}: {n + 1}/{len(idx_site)} points done '
        #       f'(t={time.perf_counter() - t_start:.1f}s)', flush=True)

    winter = np.array(winters)
    summer = np.array(summers)
    kp = np.array([config['kp'][idx] for idx in idx_site])
    kw = np.array([config['wind_factor'][idx] for idx in idx_site])

    # kp controls precipitation/accumulation, so pick it off winter MAE;
    # then, holding that kp fixed, pick wind_factor off summer MAE
    best_kp = kp[np.argmin(winter)]
    kp_mask = kp == best_kp
    best_idx = np.where(kp_mask)[0][np.argmin(summer[kp_mask])]

    best_summer = summer[best_idx]
    best_winter = winter[best_idx]
    best_kw = kw[best_idx]

    best_point = idx_site[best_idx]
    best_ds = ds_all.sel(point=best_point) # [['melt', 'accumulation', 'refreeze']]
    mb = MassBalance(rgi_map[rgiid], site)
    mb.get_model_mb(best_ds)

    return rgiid, site, best_point, best_winter, best_summer, best_kp, best_kw, init_s, io_s, compute_s, mb


print('  RGIId  | Site | Winter MAE | Summer MAE | kp  | kw')

tasks = [(rgiid, site) for rgiid, sites in site_dict.items() for site in sites]

t_start = time.perf_counter()

plot_dir = output_fp + 'best_mb_plots/'
os.makedirs(plot_dir, exist_ok=True)

out_dict = {}
with ThreadPoolExecutor(max_workers=min(32, len(tasks) or 1)) as executor:
    all_best_idx = []
    futures = [executor.submit(process_site, rgiid, site) for rgiid, site in tasks]
    for future in as_completed(futures):
        result = future.result()
        if result is None:
            continue
        rgiid, site, best_point, best_winter, best_summer, best_kp, best_kw, init_s, io_s, compute_s, mb = result
        elapsed = time.perf_counter() - t_start

        if rgiid not in out_dict:
            out_dict[rgiid] = {}

        summer_errors = mb.mod[mb.idx_summer] - mb.meas[mb.idx_summer]
        winter_errors = mb.mod[mb.idx_winter] - mb.meas[mb.idx_winter]
        if np.all(summer_errors < 0) or np.all(summer_errors > 0) or np.all(winter_errors < 0) or np.all(winter_errors > 0):
            print(f'Throwing out {rgiid}/{site} because of bias mass balance')
            pass
        else:
            out_dict[rgiid][site] = {'winter_MAE': best_winter, 'summer_MAE': best_summer,
                                'kp': best_kp, 'kw': best_kw}
            
            fig, ax = mb.plot_mb(savefig=plot_dir + f'{rgiid}_{site}.png')
            plt.close(fig)
            print(f'{rgiid:<12}{site:<9}{best_winter:<12.2f}{best_summer:<11.2f}{best_kp:<6}{best_kw:<8}')

            all_best_idx.append(best_point)

ds_best = ds_all.sel(point=all_best_idx)
ds_best.to_zarr(output_fp + 'best_params.zarr', mode='w')

with open('calibrated_grid.pkl', 'wb') as f:
    pickle.dump(out_dict, f)