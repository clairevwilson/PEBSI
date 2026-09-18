"""
Maps the loss surface over a (kp, wind_factor) grid in one forward run,
across every calibration glacier at once.

PEBSI is point-parallel and kp/wind_factor are already per-point arrays,
so a whole parameter sweep is just more points: every glacier's adaptive
mesh is tiled once per grid combination, each replica carrying its own
parameter pair, and the entire surface falls out of a single forward
simulation.

Physics settings come from project.parameters.BASE_CONFIG so the runs
match the optimizer's (25 layers, ice albedo tifs, windmaps); only the
grid, the window and the stored variables live here.

Daily output is written for every replica. Points are tiled, parameters
are repeated, so combination c occupies points [c*n_base:(c+1)*n_base],
where n_base is the combined point count across ALL glaciers -- every
block holds a full copy of every glacier's mesh, distinguishable by the
'rgiid' field the output already carries. Any stored variable reshapes
to (n_combos, n_base, ...) to recover the per-combination points. The
companion .npz records that mapping.

@author: clairevwilson
"""
import argparse
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import simulation as sim
from pebsi.io.terrain import Terrain
from project.parameters import host, HOST_PATHS, GLACIERS, BASE_CONFIG, translate_rgi

# defaults; overridable with --start-date/--end-date/--kp-values/
# --wind-factor-values. A daily-output start must be 00:00 and end 23:00
# (pebsi/config.py requires the inclusive hourly count to land on a
# multiple of 24, which any 00:00-to-23:00 pair always satisfies).
START_DATE = '2015-01-01 00:00'
END_DATE = '2020-04-01 23:00'

KP_VALUES = np.arange(0.5, 5.01, 0.5)
WIND_FACTOR_VALUES = np.arange(0.5, 5.01, 0.5)

# enough for all four glacier-wide metrics: mass balance, albedo, melt
# extent (total_water) and snow extent (surftype)
STORE_VARS = ['mass_balance', 'albedo', 'total_water', 'surftype']

out_config_fn = os.path.join(HOST_PATHS[host]['output_fp'], 'config_loss_surface.yaml')
grid_fn = os.path.join(HOST_PATHS[host]['output_fp'], 'loss_surface_grid.npz')


def build_config(start_date, end_date, kp=None, wind_factor=None):
    config = dict(BASE_CONFIG)
    config.update(
        rgi_ids=[translate_rgi[g]['6'] for g in GLACIERS],
        method_distribute='adaptive',
        start_date=start_date,
        end_date=end_date,
        output_freq='daily',
        store_vars=STORE_VARS,
        store_data=True,
        debug=False,
        progress_bar=False,
        **HOST_PATHS[host],
    )
    if kp is not None:
        config['kp'] = [float(v) for v in kp]
        config['wind_factor'] = [float(v) for v in wind_factor]
    with open(out_config_fn, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    return config


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--start-date', default=START_DATE,
                   help=f'must be 00:00 (default {START_DATE!r})')
    p.add_argument('--end-date', default=END_DATE,
                   help=f'must be 23:00 (default {END_DATE!r})')
    p.add_argument('--kp-values', type=float, nargs='+', default=None,
                   help='kp grid values (default 0.5 to 5.0 step 0.5)')
    p.add_argument('--wind-factor-values', type=float, nargs='+', default=None,
                   help='wind_factor grid values (default 0.5 to 5.0 step 0.5)')
    return p.parse_args()


def main():
    args = parse_args()
    kp_values = (np.asarray(args.kp_values) if args.kp_values is not None
                else KP_VALUES)
    wf_values = (np.asarray(args.wind_factor_values)
                if args.wind_factor_values is not None else WIND_FACTOR_VALUES)
    combos = [(kp, wf) for kp in kp_values for wf in wf_values]
    n_combos = len(combos)

    # the adaptive mesh decides how many points each glacier gets, and the
    # parameter arrays have to be sized to match, so count them first. This
    # is every glacier's mesh together -- one call, since method_distribute
    # builds all requested rgi_ids into a single point set.
    build_config(args.start_date, args.end_date)
    sim_args = sim.get_args(parse=False).parse_args([])
    sim_args.config_fn = out_config_fn
    terrain = Terrain(sim.PEBSI(sim_args).params)
    n_base = terrain.N_POINTS

    per_glacier = {g: int((terrain.rgiid_n == translate_rgi[g]['6']).sum())
                   for g in GLACIERS}
    print(f'{n_base} adaptive points across {len(GLACIERS)} glaciers x '
          f'{n_combos} combos = {n_base * n_combos} model points', flush=True)
    for g, n in per_glacier.items():
        print(f'  {g:<12} {n}', flush=True)
    print(f'kp           {kp_values}', flush=True)
    print(f'wind_factor  {wf_values}', flush=True)
    print(f'{args.start_date} to {args.end_date}', flush=True)

    # tile the combined mesh once per combination. Every replica sits at the
    # same coordinates as its original, so the DEM, shading, ice albedo and
    # wind lookups that run after this all land on the tiled points on
    # their own, and rgiid_n tiles along with them so each combo's block
    # still tells every point which glacier it belongs to. weight_n has to
    # be tiled explicitly here (not left to those later steps) since it is
    # set once inside get_points() and nothing downstream recomputes it.
    original_init = Terrain.__init__

    def tiled_init(self, params):
        original_init(self, params)
        self.lat_n = np.tile(self.lat_n, n_combos)
        self.lon_n = np.tile(self.lon_n, n_combos)
        self.rgiid_n = np.tile(self.rgiid_n, n_combos)
        self.weight_n = np.tile(self.weight_n, n_combos)
        self.N_POINTS = len(self.lat_n)

    Terrain.__init__ = tiled_init

    # points are tiled and parameters repeated, so combination c lands on
    # points [c*n_base:(c+1)*n_base] and every point in that block -- every
    # glacier alike -- carries that combination's pair
    kp = np.repeat([c[0] for c in combos], n_base)
    wind_factor = np.repeat([c[1] for c in combos], n_base)

    np.savez(grid_fn, kp=kp_values, wind_factor=wf_values,
             kp_of_point=kp, wind_factor_of_point=wind_factor,
             n_base=n_base, n_combos=n_combos,
             glaciers=np.array(GLACIERS), start=np.array(args.start_date),
             end=np.array(args.end_date))
    print(f'Wrote {grid_fn}', flush=True)

    build_config(args.start_date, args.end_date, kp=kp, wind_factor=wind_factor)
    sim_args = sim.get_args(parse=False).parse_args([])
    sim_args.config_fn = out_config_fn

    model = sim.PEBSI(sim_args)
    model.run()

    os.remove(out_config_fn)


if __name__ == '__main__':
    main()
