"""
Pregenerates and caches scattered-point site metadata for Bayesian
calibration, reusing PEBSI's own point-scattering (Terrain,
method_distribute='scatter') so calibration points sit on the same
adaptive grid a normal domain-style run would use, but keyed as named
'sites' so bayes_calibrate.py can batch many (kp, wind_factor)
replicates of the same point set into one multi-point config via the
'sites' distribution method.

@author: clairevwilson
"""
import os
import yaml
import pandas as pd

import simulation as sim
from pebsi.io.terrain import Terrain

CACHE_DIR = 'project/bayes_site_cache/'
METADATA_DIR = 'project/bayes_site_metadata/'


def get_scattered_sites(glacier_name, rgi_id, n_points):
    """
    Returns (site_names, metadata_fn) for n_points scattered across
    the glacier, generating and caching them on first use.
    """
    from project.bayes_calibrate import BASE_CONFIG  # deferred: bayes_calibrate imports this module

    cache_fn = os.path.join(CACHE_DIR, f'{glacier_name}_{n_points}.csv')
    metadata_fn = os.path.join(METADATA_DIR, f'{glacier_name}_{n_points}_metadata.csv')

    if os.path.exists(cache_fn):
        return list(pd.read_csv(cache_fn, dtype={'rgiid': str, 'site': str})['site']), metadata_fn

    tmp_config_fn = f'_scatter_{glacier_name}.yaml'
    configs = dict(BASE_CONFIG)
    # no simulation actually runs here, just need a valid (whole-day, for
    # output_freq='daily') date range to pass config validation
    configs['start_date'] = '2024-04-20 00:00'
    configs['end_date'] = '2024-04-20 23:00'
    configs['rgi_ids'] = [rgi_id]
    configs['n_points'] = n_points
    configs['output_fp'] = f'/ocean/projects/ees260009p/cwilson4/Output/_scatter_tmp/{glacier_name}/'
    with open(tmp_config_fn, 'w') as f:
        yaml.dump(configs, f, sort_keys=False)

    args = sim.get_args()
    args.config_fn = tmp_config_fn
    model = sim.PEBSI(args)

    terrain = Terrain(model.params)
    terrain.run_dem_functions()
    os.remove(tmp_config_fn)

    site_names = [f'pt{i:06d}' for i in range(terrain.N_POINTS)]
    df = pd.DataFrame({
        'rgiid': rgi_id,
        'site': site_names,
        'lat': terrain.lat_n,
        'lon': terrain.lon_n,
        'elevation': terrain.elev_n,
        'slope': terrain.slope_n,
        'aspect': terrain.aspect_n,
    })

    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)
    df.to_csv(cache_fn, index=False)
    df.to_csv(metadata_fn, index=False)
    return site_names, metadata_fn


def get_multiglacier_scattered_sites(glaciers, rgi_ids, n_points):
    """
    Like get_scattered_sites, but distributes n_points across several
    glaciers at once, area-weighted, using PEBSI's own multi-glacier
    'scatter' distribution (Terrain.scatter_points area-weights by RGI
    polygon area when given several distinct rgi_ids). Site names are
    only unique WITHIN each glacier's own rows (the 'sites' distribution
    method looks sites up by (rgiid, site) pair, not by site alone), so
    reused pt###### names across glaciers are fine.

    Returns (site_names, site_rgi_ids, site_glacier_names, metadata_fn),
    all length n_points (approximately -- scatter_points' adaptive grid
    search doesn't hit n_points exactly), aligned point-by-point.
    """
    from project.bayes_calibrate import BASE_CONFIG

    tag = '_'.join(glaciers)
    cache_fn = os.path.join(CACHE_DIR, f'multi_{tag}_{n_points}.csv')
    metadata_fn = os.path.join(METADATA_DIR, f'multi_{tag}_{n_points}_metadata.csv')

    if os.path.exists(cache_fn):
        df = pd.read_csv(cache_fn, dtype={'rgiid': str, 'site': str, 'glacier_name': str})
        return list(df['site']), list(df['rgiid']), list(df['glacier_name']), metadata_fn

    tmp_config_fn = f'_scatter_multi_{tag}.yaml'
    configs = dict(BASE_CONFIG)
    configs['start_date'] = '2024-04-20 00:00'
    configs['end_date'] = '2024-04-20 23:00'
    configs['rgi_ids'] = list(rgi_ids)
    configs['n_points'] = n_points
    configs['output_fp'] = f'/ocean/projects/ees260009p/cwilson4/Output/_scatter_tmp/multi_{tag}/'
    with open(tmp_config_fn, 'w') as f:
        yaml.dump(configs, f, sort_keys=False)

    args = sim.get_args()
    args.config_fn = tmp_config_fn
    model = sim.PEBSI(args)

    terrain = Terrain(model.params)
    terrain.run_dem_functions()
    os.remove(tmp_config_fn)

    rgi_to_name = dict(zip(rgi_ids, glaciers))

    # name sites sequentially within each glacier's own subset, not
    # globally, since (rgiid, site) together is what has to be unique
    site_names = []
    counters = {gid: 0 for gid in rgi_ids}
    for gid in terrain.rgiid_n:
        site_names.append(f'pt{counters[gid]:06d}')
        counters[gid] += 1

    df = pd.DataFrame({
        'rgiid': terrain.rgiid_n,
        'site': site_names,
        'glacier_name': [rgi_to_name[gid] for gid in terrain.rgiid_n],
        'lat': terrain.lat_n,
        'lon': terrain.lon_n,
        'elevation': terrain.elev_n,
        'slope': terrain.slope_n,
        'aspect': terrain.aspect_n,
    })

    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)
    df.to_csv(cache_fn, index=False)
    # metadata_fn only needs the columns Terrain's 'sites' lookup reads
    df[['rgiid', 'site', 'lat', 'lon', 'elevation', 'slope', 'aspect']].to_csv(metadata_fn, index=False)
    return site_names, list(df['rgiid']), list(df['glacier_name']), metadata_fn
