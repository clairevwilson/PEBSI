"""
One-at-a-time sensitivity test for the five Gulkana sites.

Each parameter is perturbed to a low and high value while every
other parameter is held at baseline, all stacked into one
multi-point config (same pattern as gridsearch_gulkana.py). This
relies on Boone_c1-5, grainsize_rfz, and snow_threshold_low/high
being in pebsi/config.py's dynamic_fields list, and on the
per-point broadcasting fixes in massbalance.py/layers.py/
energybalance.py for those parameters.

constant_irrwater is True, so Sr (not Sr_dense/Sr_light) sets the
irreducible water content and is included below. The albedo
constants are left out: ice albedo comes from the preprocessed
tif and snow/firn albedo comes out of the SNICAR emulator, so
those constants aren't on the active code path.

@author: clairevwilson
"""
import yaml
import os

out_config_fn = 'config_sensitivity_gulkana.yaml'

rgi_id = '01.00570'
sites = ['A', 'AU', 'B', 'D', 'T']

configs = {}

# PHYSICS
configs['option_ice_albedo_tif'] = True
configs['option_windmaps'] = True
configs['option_accel_grains'] = True
configs['option_flat_plates'] = True
configs['option_dynamics'] = False
configs['constant_freshgrainsize'] = 54.5
configs['constant_irrwater'] = True
configs['precgrad'] = 0.000100

# CONFIGURATION
configs['start_date'] = '2014-03-01 00:00'
configs['end_date'] = '2025-08-20 00:00'
configs['debug'] = True
configs['store_data'] = True
configs['progress_bar'] = False
configs['store_vars'] = ['MB', 'EB', 'layerwater', 'layerheight', 'layertype']
configs['bias_vars'] = ['temp']

# FILEPATHS
configs['climate_fp'] = '/ocean/projects/ees260009p/cwilson4/climate_data/'
configs['rgi_fp'] = '/ocean/projects/ees260009p/cwilson4/RGI/rgi60/00_rgi60_attribs/'
configs['output_fp'] = '/ocean/projects/ees260009p/cwilson4/Output/sensitivity_gulkana/'
configs['cop30_vrt_path'] = '/ocean/projects/ees260009p/cwilson4/data/dems/COP30/COP30_reg01.vrt'
configs['shading_fp'] = '/ocean/projects/ees260009p/cwilson4/data/shading/'
configs['ice_albedo_fn'] = '/ocean/projects/ees260009p/cwilson4/data/ice_albedo/{gid}_albedo.tif'
configs['thickness_fn'] = '/ocean/projects/ees260009p/cwilson4/data/ice_thickness/RGI60-01/RGI60-{gid}_thickness.tif'
configs['windmap_fn'] = '/ocean/projects/ees260009p/cwilson4/data/windmapper/{gid}.nc'

# baseline value for every perturbed parameter
baseline = {
    'dust_factor': 10,
    'wind_factor': 3,
    'kp': 2.5,
    'lapse_rate': -6.5,
    'roughness_fresh_snow': 0.24,
    'roughness_aged_snow': 10,
    'roughness_firn': 4,
    'roughness_ice': 20,
    'roughness_aging_rate': 0.5,
    'Sr': 0.05,
    'Boone_c1': 2.7e-6,
    'Boone_c2': 0.042,
    'Boone_c3': 0.046,
    'Boone_c4': 0.081,
    'Boone_c5': 0.016,
    'grainsize_rfz': 1500,
}

# low/high bound for each perturbed parameter
bounds = {
    'dust_factor': (1, 20),
    'wind_factor': (1, 5),
    'kp': (1, 4),
    'lapse_rate': (-9.0, -4.0),
    'roughness_fresh_snow': (0.1, 1.0),
    'roughness_aged_snow': (2, 20),
    'roughness_firn': (1, 10),
    'roughness_ice': (5, 50),
    'roughness_aging_rate': (0.1, 2.0),
    'Sr': (0.02, 0.10),
    'Boone_c1': (1.35e-6, 5.4e-6),
    'Boone_c2': (0.021, 0.084),
    'Boone_c3': (0.023, 0.092),
    'Boone_c4': (0.0405, 0.162),
    'Boone_c5': (0.008, 0.032),
    'grainsize_rfz': (750, 3000),
}

# snow_threshold_low/high move together, window width fixed at 2 C,
# only the center shifts; the center bounds reuse the default low/high
snow_threshold_width = 2.0
snow_threshold_center_baseline = 1.2
snow_threshold_center_bounds = (0.2, 2.2)


def thresholds_from_center(center):
    half = snow_threshold_width / 2
    return center - half, center + half


runs = [dict(baseline)]
run_tags = ['baseline']
centers = [snow_threshold_center_baseline]

for param, (low, high) in bounds.items():
    for level, value in [('low', low), ('high', high)]:
        run = dict(baseline)
        run[param] = value
        runs.append(run)
        run_tags.append(f'{param}_{level}')
        centers.append(snow_threshold_center_baseline)

for level, center in [('low', snow_threshold_center_bounds[0]), ('high', snow_threshold_center_bounds[1])]:
    runs.append(dict(baseline))
    run_tags.append(f'snow_threshold_{level}')
    centers.append(center)

gids, site_list, tags = [], [], []
param_lists = {p: [] for p in baseline}
snow_threshold_low_list, snow_threshold_high_list = [], []

for site in sites:
    for run, tag, center in zip(runs, run_tags, centers):
        gids.append(rgi_id)
        site_list.append(site)
        tags.append(tag)
        for p in baseline:
            param_lists[p].append(run[p])
        low_t, high_t = thresholds_from_center(center)
        snow_threshold_low_list.append(low_t)
        snow_threshold_high_list.append(high_t)

configs['rgi_ids'] = gids
configs['sites'] = site_list
configs['n_points'] = len(site_list)
configs.update(param_lists)
configs['snow_threshold_low'] = snow_threshold_low_list
configs['snow_threshold_high'] = snow_threshold_high_list

with open(out_config_fn, 'w') as f:
    yaml.dump(configs, f, sort_keys=False)

print(f'Wrote {out_config_fn} with {len(site_list)} points '
      f'({len(sites)} sites x {len(runs)} one-at-a-time perturbations)')

if __name__ == '__main__':
    import simulation as sim

    args = sim.get_args()
    args.config_fn = out_config_fn

    model = sim.PEBSI(args)
    model.run()

    os.remove(out_config_fn)
