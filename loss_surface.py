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
import tempfile
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import simulation as sim
from pebsi.io.terrain import Terrain
from project.parameters import host, HOST_PATHS, GLACIERS, BASE_CONFIG, translate_rgi, wind_factor_aws

# defaults; overridable with --start-date/--end-date/--kp-values/
# --wind-factor-values. A daily-output start must be 00:00 and end 23:00
# (pebsi/config.py requires the inclusive hourly count to land on a
# multiple of 24, which any 00:00-to-23:00 pair always satisfies).
START_DATE = '2000-01-01 00:00'
END_DATE = '2025-08-31 23:00'

KP_VALUES = np.arange(0.5, 5.01, 0.5)
WIND_FACTOR_VALUES = np.arange(0.5, 5.01, 0.5)

# enough for all four glacier-wide metrics: mass balance, albedo, melt
# extent (top_water) and snow extent (surftype)
STORE_VARS = ['mass_balance', 'albedo', 'top_water', 'surftype']

def run_filenames(tag):
    """
    The transient config path, and the tag the grid .npz is named with
    once the run's output directory exists. The config is handed to PEBSI
    to build from and deleted at the end; PEBSI copies its own resolved
    copy into the output directory regardless.
    """
    suffix = f'_{tag}' if tag else ''
    config_fn = os.path.join(tempfile.gettempdir(),
                             f'config_loss_surface{suffix}.yaml')
    return config_fn, suffix


def build_config(start_date, end_date, glaciers, preclimit, out_config_fn,
                 output_fp, kp=None, wind_factor=None, lapse_rate=None, og_kappa=None,
                 overrides=None):
    config = dict(BASE_CONFIG)
    config.update(
        rgi_ids=[translate_rgi[g]['6'] for g in glaciers],
        option_preclimit=preclimit,
        dust_factor=500,
        option_ice_thickness_tif=False,
        initial_ice_depth=500,
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
    # after the splat, so this run's directory wins over the host default
    config['output_fp'] = output_fp
    config.update(overrides or {})
    if kp is not None:
        config['kp'] = [float(v) for v in kp]
        config['wind_factor'] = [float(v) for v in wind_factor]
    if lapse_rate is not None:
        config['lapse_rate'] = [float(v) for v in lapse_rate]
    if og_kappa is not None:
        config['og_kappa'] = [float(v) for v in og_kappa]
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
    p.add_argument('--lapse-rate-values', type=float, nargs='+', default=None,
                   help='lapse_rate grid values [K km-1]; adds a third grid '
                        'axis after wind_factor (default: none, config value)')
    p.add_argument('--og-kappa-values', type=float, nargs='+', default=None,
                   help='og_kappa grid values (katabatic constant, used with '
                        '--set method_turbulent=OG or OG_hybrid); adds a grid '
                        'axis after lapse_rate (default: none, config value)')
    p.add_argument('--kp-by-glacier', default=None, metavar='RESULT_YAML',
                   help='take each glacier\'s kp from an AD_optimize-style result '
                        'yaml (glaciers, kp), so the kp grid values act as '
                        'multipliers of it (e.g. --kp-values 1.0 keeps it fixed)')
    p.add_argument('--glaciers', nargs='+', default=None,
                   help=f'glaciers to map (default all of {GLACIERS})')
    p.add_argument('--preclimit', action='store_true',
                   help='taper precipitation over the uppermost 25%% of the '
                        'elevation range on glaciers spanning >1000 m')
    p.add_argument('--output-fp', default=None,
                   help='directory this run writes to; the .zarr lands in '
                        '<output-fp>_N and the grid .npz beside it '
                        f'(default {HOST_PATHS[host]["output_fp"]!r})')
    p.add_argument('--wind-factor-by-glacier', action='store_true',
                   help='scale each glacier by its AWS-derived wind factor '
                        '(project.parameters.wind_factor_aws), so the '
                        'wind_factor grid values act as multipliers of it')
    p.add_argument('--set', nargs='+', default=[], metavar='KEY=VALUE',
                   help='config overrides, values parsed as YAML '
                        '(e.g. roughness_ice=20 bias_vars=[])')
    p.add_argument('--resume-from', default=None, metavar='RUN_DIR',
                   help='continue an interrupted run from RUN_DIR/checkpoint.npz, '
                        'appending to its output.zarr; pass the same grid arguments '
                        'as the original run so the tiled mesh is rebuilt identically')
    p.add_argument('--tag', default='',
                   help='suffix for the grid/config filenames, so a rerun '
                        "does not overwrite an earlier run's grid .npz")
    return p.parse_args()


def main():
    args = parse_args()
    kp_values = (np.asarray(args.kp_values) if args.kp_values is not None
                else KP_VALUES)
    wf_values = (np.asarray(args.wind_factor_values)
                if args.wind_factor_values is not None else WIND_FACTOR_VALUES)
    lr_values = (np.asarray(args.lapse_rate_values)
                if args.lapse_rate_values is not None else None)
    og_values = (np.asarray(args.og_kappa_values)
                if args.og_kappa_values is not None else None)
    combos = [(kp, wf, lr, ok) for kp in kp_values for wf in wf_values
              for lr in (lr_values if lr_values is not None else [None])
              for ok in (og_values if og_values is not None else [None])]
    n_combos = len(combos)

    glaciers = args.glaciers if args.glaciers is not None else GLACIERS
    unknown = [g for g in glaciers if g not in translate_rgi]
    assert not unknown, f'unknown glacier(s): {unknown}; known: {sorted(translate_rgi)}'
    output_fp = args.output_fp or HOST_PATHS[host]['output_fp']
    out_config_fn, suffix = run_filenames(args.tag)
    overrides = {k: yaml.safe_load(v) for k, v in (kv.split('=', 1) for kv in args.set)}

    # the adaptive mesh decides how many points each glacier gets, and the
    # parameter arrays have to be sized to match, so count them first. This
    # is every glacier's mesh together -- one call, since method_distribute
    # builds all requested rgi_ids into a single point set.
    build_config(args.start_date, args.end_date, glaciers, args.preclimit,
                 out_config_fn, output_fp, overrides=overrides)
    sim_args = sim.get_args(parse=False).parse_args([])
    sim_args.config_fn = out_config_fn
    terrain = Terrain(sim.PEBSI(sim_args).params)
    n_base = terrain.N_POINTS

    per_glacier = {g: int((terrain.rgiid_n == translate_rgi[g]['6']).sum())
                   for g in glaciers}
    print(f'{n_base} adaptive points across {len(glaciers)} glaciers x '
          f'{n_combos} combos = {n_base * n_combos} model points', flush=True)
    for g, n in per_glacier.items():
        print(f'  {g:<12} {n}', flush=True)
    print(f'kp           {kp_values}', flush=True)
    print(f'wind_factor  {wf_values}', flush=True)
    print(f'lapse_rate   {lr_values}', flush=True)
    print(f'og_kappa     {og_values}', flush=True)
    print(f'kp by glacier  {args.kp_by_glacier}', flush=True)
    print(f'{args.start_date} to {args.end_date}', flush=True)
    print(f'option_preclimit  {args.preclimit}', flush=True)
    print(f'wind factor by glacier  {args.wind_factor_by_glacier}', flush=True)
    print(f'overrides         {overrides}', flush=True)
    print(f'output_fp         {output_fp}', flush=True)

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

    # get_precip_taper runs later than this (it needs the elevations the DEM
    # step fills in afterwards), so it sees the tiled mesh, and it groups
    # points by glacier ID -- which every replica of a glacier shares. Left
    # alone it would treat all n_combos copies as one glacier: every
    # replica's accumulation area would anchor to a single point sitting in
    # one arbitrary combination's block, and one glacier-wide maximum taken
    # across every kp would set the floor for all of them. Build it on one
    # block instead and repeat that, shifting each block onto its own points
    # and its own glacier segments.
    original_taper = Terrain.get_precip_taper

    def tiled_taper(self):
        full_elev, full_rgiid, full_n = self.elev_n, self.rgiid_n, self.N_POINTS
        self.elev_n = full_elev[:n_base]
        self.rgiid_n = full_rgiid[:n_base]
        self.N_POINTS = n_base
        original_taper(self)
        self.elev_n, self.rgiid_n, self.N_POINTS = full_elev, full_rgiid, full_n

        n_glaciers = len(self.glacier_point_counts)
        point_offset = np.repeat(np.arange(n_combos) * n_base, n_base)
        glacier_offset = np.repeat(np.arange(n_combos) * n_glaciers, n_base)

        self.precip_anchor_idx_n = (np.tile(self.precip_anchor_idx_n, n_combos)
                                    + point_offset).astype(np.int32)
        self.glacier_idx_n = (np.tile(self.glacier_idx_n, n_combos)
                              + glacier_offset).astype(np.int32)
        self.precip_taper_n = np.tile(self.precip_taper_n, n_combos)
        self.precip_upper25_n = np.tile(self.precip_upper25_n, n_combos)
        self.glacier_point_counts = np.tile(self.glacier_point_counts, n_combos)

    Terrain.get_precip_taper = tiled_taper

    # points are tiled and parameters repeated, so combination c lands on
    # points [c*n_base:(c+1)*n_base] and every point in that block -- every
    # glacier alike -- carries that combination's pair
    kp = np.repeat([c[0] for c in combos], n_base)
    wind_factor = np.repeat([c[1] for c in combos], n_base)
    if args.wind_factor_by_glacier:
        base_wf = np.ones(n_base)
        for g in glaciers:
            base_wf[terrain.rgiid_n[:n_base] == translate_rgi[g]['6']] = wind_factor_aws[g]
        wind_factor = wind_factor * np.tile(base_wf, n_combos)

    if args.kp_by_glacier:
        with open(args.kp_by_glacier) as f:
            result = yaml.safe_load(f)
        kp_of = dict(zip(result['glaciers'], result['kp']))
        base_kp = np.ones(n_base)
        for g in glaciers:
            base_kp[terrain.rgiid_n[:n_base] == translate_rgi[g]['6']] = kp_of[g]
        kp = kp * np.tile(base_kp, n_combos)

    lapse_rate = (np.repeat([c[2] for c in combos], n_base)
                  if lr_values is not None else None)
    og_kappa = (np.repeat([c[3] for c in combos], n_base)
                if og_values is not None else None)

    build_config(args.start_date, args.end_date, glaciers, args.preclimit,
                 out_config_fn, output_fp, kp=kp, wind_factor=wind_factor,
                 lapse_rate=lapse_rate, og_kappa=og_kappa, overrides=overrides)
    sim_args = sim.get_args(parse=False).parse_args([])
    sim_args.config_fn = out_config_fn
    if args.resume_from:
        sim_args.resume_from = args.resume_from

    model = sim.PEBSI(sim_args)
    model.run()

    # written afterwards so it lands in the run's own output directory,
    # which Output only names once the run is under way
    grid_fn = os.path.join(model.output_fp, f'loss_surface_grid{suffix}.npz')
    lapse_kw = {} if lr_values is None else dict(lapse_rate=lr_values, lapse_rate_of_point=lapse_rate)
    if og_values is not None:
        lapse_kw.update(og_kappa=og_values, og_kappa_of_point=og_kappa)
    np.savez(grid_fn, kp=kp_values, wind_factor=wf_values,
             kp_of_point=kp, wind_factor_of_point=wind_factor, **lapse_kw,
             n_base=n_base, n_combos=n_combos,
             glaciers=np.array(glaciers), start=np.array(args.start_date),
             end=np.array(args.end_date))
    print(f'Wrote {grid_fn}', flush=True)

    os.remove(out_config_fn)


if __name__ == '__main__':
    main()
