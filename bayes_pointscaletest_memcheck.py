"""
Feasibility check for bayes_pointscaletest.py: reports the exact device
memory PEBSI's compiled main() step-loop will need for 300,000 points at
a 1 year temporal chunk, without executing a single simulation step.

How: builds the real inputs (spatial preprocessing + first chunk of
forcings) exactly as PEBSI.run() would, then asks XLA's ahead-of-time
compiler for its buffer-assignment sizes (memory_analysis()) instead of
calling the jitted main(). Compilation is forced onto the CPU backend
(no GPU needed/touched -- safe to run on a login node), since
buffer-assignment sizes are governed by array shapes/dtypes and are
backend-independent to within normal fusion/layout variance.

@author: clairevwilson
"""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('PYTHONUNBUFFERED', '1')

import sys
import time
import yaml

import simulation as sim
from project.bayes_calibrate import BASE_CONFIG, baseline, align_end_date_for_daily_output
from project.scatter_sites import get_scattered_sites
from project.glacierwide_loss import translate_rgi

GLACIER = 'gulkana'
N_POINTS = 300_000

class _StopAfterCompile(Exception):
    pass

import psutil
_t0 = time.time()
_proc = psutil.Process()
def stage(label):
    rss_gb = _proc.memory_info().rss / 1024**3
    print(f'[{time.time() - _t0:7.1f}s] [RSS {rss_gb:6.2f} GiB] {label}', flush=True)

def main():
    stage('generating scattered sites...')
    rgi_id = translate_rgi[GLACIER]['6']
    site_names, metadata_fn = get_scattered_sites(GLACIER, rgi_id, N_POINTS)
    stage(f'got {len(site_names):,} sites')

    start_date = '2019-01-01 00:00'
    end_date = align_end_date_for_daily_output(start_date, '2021-01-01 00:00')

    configs = dict(BASE_CONFIG)
    configs['n_spinup_years'] = 0
    configs['temporal_chunk_years'] = 1
    configs['start_date'] = start_date
    configs['end_date'] = end_date
    configs['metadata_fn'] = metadata_fn
    configs['output_fp'] = '/ocean/projects/ees260009p/cwilson4/Output/bayes_pointscaletest_memcheck/'
    configs['sites'] = site_names
    configs['rgi_ids'] = [rgi_id] * len(site_names)
    configs['n_points'] = len(site_names)
    configs['kp'] = baseline['kp']
    configs['wind_factor'] = baseline['wind_factor']

    out_config_fn = '_bayes_pointscaletest_memcheck.yaml'
    with open(out_config_fn, 'w') as f:
        yaml.dump(configs, f, sort_keys=False)

    # Intercept the jitted main() step-loop: compile for the real chunk
    # shapes, record memory_analysis(), then abort before executing.
    orig_main = sim.main
    report = {}

    def profiling_main(initial_state, all_forcings, point_attrs, static_args, dynamic_args):
        stage('forcings packed; compiling main() for this chunk (no execution)...')
        lowered = orig_main.lower(
            initial_state, all_forcings, point_attrs, static_args, dynamic_args
        )
        compiled = lowered.compile()
        report['memory_analysis'] = compiled.memory_analysis()
        report['n_points'] = initial_state.albedo.shape[0]
        report['n_layers'] = initial_state.lice.shape[1]
        report['forcing_shape'] = {
            f: getattr(all_forcings, f).shape
            for f in ('temp', 'wind')
        }
        raise _StopAfterCompile()

    sim.main = profiling_main

    args = sim.get_args()
    args.config_fn = out_config_fn
    pebsi = sim.PEBSI(args)

    # instrument initialize()'s substeps so slow stages are visible live
    orig_prep_spatial = pebsi.prepare_spatial_inputs
    orig_prep_state = pebsi.prepare_initial_state
    orig_pack_states = pebsi.pack_states

    def timed_prep_spatial():
        # inlined copy of PEBSI.prepare_spatial_inputs with a stage() print
        # between each substep, to find exactly which one is slow at 300k points
        stage('prepare_spatial_inputs() starting')
        params = pebsi.params
        pebsi.dates = sim.pd.date_range(params.start_date, params.end_date, freq='h')

        terrain = sim.Terrain(params)
        stage('  Terrain() constructed')

        terrain.run_dem_functions()
        stage('  terrain.run_dem_functions() done')

        if params.option_ice_albedo_tif:
            albedo_ice_n = terrain.get_ice_albedo()
            pebsi.params.albedo_ice = albedo_ice_n
            pebsi.config.dynamic_args = pebsi.config.dynamic_args._replace(
                albedo_ice=sim.jnp.array(albedo_ice_n, dtype=sim.jnp.float64)
            )
            stage('  terrain.get_ice_albedo() done')

        if params.option_windmaps:
            terrain.get_wind_fields()
            stage('  terrain.get_wind_fields() done')

        terrain.validate_terrain_data()
        stage('  terrain.validate_terrain_data() done')

        terrain.load_shading()
        stage('  terrain.load_shading() done')

        if params.option_dynamics:
            terrain.get_initial_ice_thickness()
            stage('  terrain.get_initial_ice_thickness() done')

        _cl = sim.Climate.__new__(sim.Climate)
        _cl.params = pebsi.params
        _cl.terrain = terrain
        _cl.get_vardict()
        stage('  Climate.get_vardict() done')
        _cl.get_unique_cells()
        stage('  Climate.get_unique_cells() done')

        pebsi._cl = _cl
        pebsi.terrain = terrain
        stage('prepare_spatial_inputs() done')

    def timed_prep_state():
        stage('prepare_initial_state() starting (layer initialization)...')
        orig_prep_state()
        stage('prepare_initial_state() done')

    def timed_pack_states():
        stage('pack_states() starting...')
        result = orig_pack_states()
        stage('pack_states() done')
        return result

    orig_pack_forcings = pebsi.pack_forcings

    def timed_pack_forcings(*a, **kw):
        stage('pack_forcings() starting (climate data load)...')
        result = orig_pack_forcings(*a, **kw)
        stage('pack_forcings() done')
        return result

    pebsi.prepare_spatial_inputs = timed_prep_spatial
    pebsi.prepare_initial_state = timed_prep_state
    pebsi.pack_states = timed_pack_states
    pebsi.pack_forcings = timed_pack_forcings

    try:
        pebsi.run()
    except _StopAfterCompile:
        pass
    finally:
        sim.main = orig_main
        os.remove(out_config_fn)

    ma = report['memory_analysis']
    gib = 1024 ** 3
    live = ma.argument_size_in_bytes + ma.output_size_in_bytes + ma.temp_size_in_bytes - ma.alias_size_in_bytes

    print()
    print(f"n_points        = {report['n_points']:,}")
    print(f"n_layers        = {report['n_layers']}")
    print(f"forcing shapes  = {report['forcing_shape']}  (N_TIME, N_UNIQUE)")
    print()
    print(f"argument_size   = {ma.argument_size_in_bytes / gib:.2f} GiB  (input state + forcings)")
    print(f"output_size     = {ma.output_size_in_bytes / gib:.2f} GiB  (updated state + output records)")
    print(f"temp_size       = {ma.temp_size_in_bytes / gib:.2f} GiB  (scratch/intermediate buffers)")
    print(f"alias_size      = {ma.alias_size_in_bytes / gib:.2f} GiB  (buffers reused in place)")
    print(f"---------------------------------------------")
    print(f"estimated peak  = {live / gib:.2f} GiB  (CPU-backend estimate; expect it to differ some on GPU due to fusion/layout, but not by an order of magnitude)")

if __name__ == '__main__':
    main()
