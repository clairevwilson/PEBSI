"""
Maps the loss surface over a (kp, wind_factor) grid in one forward run.

PEBSI is point-parallel and kp/wind_factor are already per-point arrays,
so a whole parameter sweep is just more points: the glacier's adaptive
mesh is tiled once per grid combination, each replica carrying its own
parameter pair, and the entire surface falls out of a single forward
simulation. Same trick gridsearch_gulkana.py uses across sites, applied
to the adaptive mesh instead.

Physics settings come from AD_optimize.BASE_CONFIG so the runs match the
optimizer's (25 layers, ice albedo tifs, windmaps); only the grid, the
window and the stored variables live here.

Daily output is written for every replica. Points are tiled, parameters
are repeated, so combination c occupies points [c*n_base:(c+1)*n_base]
and any stored variable reshapes to (n_combos, n_base, ...) to recover
the per-combination glacier. The companion .npz records that mapping.

@author: clairevwilson
"""
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import simulation as sim
from pebsi.io.terrain import Terrain
import AD_optimize as ad

GLACIER = 'gulkana'
START_DATE = '2015-01-01 00:00'
END_DATE = '2025-01-01 23:00'

KP_VALUES = np.arange(0.5, 5.01, 0.5)
WIND_FACTOR_VALUES = np.arange(0.5, 5.01, 0.5)

# enough for all four glacier-wide metrics: mass balance, albedo, melt
# extent (total_water) and snow extent (surftype)
STORE_VARS = ['mass_balance', 'albedo', 'total_water', 'surftype']

out_config_fn = os.path.join(ad.HOST_PATHS[ad.host]['output_fp'], 'config_loss_surface.yaml')
grid_fn = os.path.join(ad.HOST_PATHS[ad.host]['output_fp'], 'loss_surface_grid.npz')


def build_config(kp=None, wind_factor=None):
    config = dict(ad.BASE_CONFIG)
    config.update(
        rgi_ids=[ad.translate_rgi[GLACIER]['6']],
        method_distribute='adaptive',
        start_date=START_DATE,
        end_date=END_DATE,
        output_freq='daily',
        store_vars=STORE_VARS,
        store_data=True,
        debug=False,
        progress_bar=False,
        **ad.HOST_PATHS[ad.host],
    )
    if kp is not None:
        config['kp'] = [float(v) for v in kp]
        config['wind_factor'] = [float(v) for v in wind_factor]
    with open(out_config_fn, 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    return config


def main():
    combos = [(kp, wf) for kp in KP_VALUES for wf in WIND_FACTOR_VALUES]
    n_combos = len(combos)

    # the adaptive mesh decides how many points this glacier gets, and the
    # parameter arrays have to be sized to match, so count them first
    build_config()
    args = sim.get_args(parse=False).parse_args([])
    args.config_fn = out_config_fn
    n_base = Terrain(sim.PEBSI(args).params).N_POINTS

    print(f'{GLACIER}: {n_base} adaptive points x {n_combos} combos '
          f'= {n_base * n_combos} model points', flush=True)
    print(f'kp           {KP_VALUES}', flush=True)
    print(f'wind_factor  {WIND_FACTOR_VALUES}', flush=True)

    # tile the mesh once per combination. Every replica sits at the same
    # coordinates as its original, so the DEM, shading, ice albedo and wind
    # lookups that run after this all land on the tiled points on their own.
    original_init = Terrain.__init__

    def tiled_init(self, params):
        original_init(self, params)
        self.lat_n = np.tile(self.lat_n, n_combos)
        self.lon_n = np.tile(self.lon_n, n_combos)
        self.rgiid_n = np.tile(self.rgiid_n, n_combos)
        self.N_POINTS = len(self.lat_n)

    Terrain.__init__ = tiled_init

    # points are tiled and parameters repeated, so combination c lands on
    # points [c*n_base:(c+1)*n_base] and every point in that block carries
    # that combination's pair
    kp = np.repeat([c[0] for c in combos], n_base)
    wind_factor = np.repeat([c[1] for c in combos], n_base)

    np.savez(grid_fn, kp=KP_VALUES, wind_factor=WIND_FACTOR_VALUES,
             kp_of_point=kp, wind_factor_of_point=wind_factor,
             n_base=n_base, n_combos=n_combos,
             glacier=np.array(GLACIER), start=np.array(START_DATE),
             end=np.array(END_DATE))
    print(f'Wrote {grid_fn}', flush=True)

    build_config(kp=kp, wind_factor=wind_factor)
    args = sim.get_args(parse=False).parse_args([])
    args.config_fn = out_config_fn

    model = sim.PEBSI(args)
    model.run()

    os.remove(out_config_fn)


if __name__ == '__main__':
    main()
