"""
Forward run at the per-glacier (kp, wind_factor, precgrad) AD_optimize.py found,
with the optimizer's own configuration, storing hourly MB, EB and climate.
og_kappa (OG_hybrid) comes from the result, or from KAPPA_YAML when it was
held fixed during the optimization (then it lives only in the --init_from file).
Usage: python run_AD_result.py [RESULT_YAML] [KAPPA_YAML]
"""
import os
from datetime import datetime
import sys

import numpy as np
import yaml

import AD_optimize as ad
import simulation as sim
from pebsi.io.terrain import Terrain

METHOD_STABILITY = 'Braithwaite'

RESULT_FN = sys.argv[1] if len(sys.argv) > 1 else ad.HOST_PATHS[ad.host]['output_fp'] + 'AD_optimize_result.yaml'
KAPPA_FN = sys.argv[2] if len(sys.argv) > 2 else None
TAG = 'AD_result_' + datetime.now().strftime('%m_%d')
OUTPUT_FP = ad.HOST_PATHS[ad.host]['output_fp'].replace('AD_optimize', TAG)


def config_with(**overrides):
    return ad.build_config(dict(store_data=True, store_vars=['MB', 'EB', 'climate'],
                                output_freq='hourly', debug=True, output_fp=OUTPUT_FP,
                                method_stability=METHOD_STABILITY,
                                **overrides), tag=TAG)


def model_for(config_fn):
    args = sim.get_args(parse=False).parse_args([])
    args.config_fn = config_fn
    return sim.PEBSI(args)


def main():
    with open(RESULT_FN) as f:
        result = yaml.safe_load(f)
    if 'og_kappa' not in result and KAPPA_FN:
        with open(KAPPA_FN) as f:
            fixed = yaml.safe_load(f)
        kappa_of = dict(zip(fixed['glaciers'], fixed['og_kappa']))
        result['og_kappa'] = [kappa_of[g] for g in result['glaciers']]
    per_glacier = {name: dict(zip(result['glaciers'], result[name]))
                   for name in ('kp', 'wind_factor', 'precgrad', 'og_kappa') if name in result}

    # the adaptive mesh decides each glacier's point count, so build it first
    # to size the per-point parameter arrays
    rgiid_n = np.asarray(Terrain(model_for(config_with()).params).rgiid_n, dtype=str)
    glacier_of = {ad.translate_rgi[g]['6']: g for g in result['glaciers']}
    overrides = {name: [float(values[glacier_of[gid]]) for gid in rgiid_n]
                 for name, values in per_glacier.items()}
    if 'og_kappa' in overrides:
        overrides['method_turbulent'] = 'OG_hybrid'
    print(f'{RESULT_FN} -> {OUTPUT_FP}', {k: v for k, v in per_glacier.items()}, flush=True)

    model_for(config_with(**overrides)).run()


if __name__ == '__main__':
    main()
