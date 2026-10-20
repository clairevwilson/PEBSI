"""
Forward run at the per-glacier (kp, wind_factor) AD_optimize.py found,
with the optimizer's own configuration, storing hourly MB, EB and climate.
"""
import numpy as np
import yaml

import AD_optimize as ad
import simulation as sim
from pebsi.io.terrain import Terrain

RESULT_FN = ad.HOST_PATHS[ad.host]['output_fp'] + 'AD_optimize_result.yaml'
OUTPUT_FP = '/ocean/projects/ees260009p/cwilson4/Output/AD_result/'


def config_with(**overrides):
    return ad.build_config(dict(store_data=True, store_vars=['MB', 'EB', 'climate'],
                                output_freq='hourly', debug=True, output_fp=OUTPUT_FP,
                                **overrides), tag='AD_result')


def model_for(config_fn):
    args = sim.get_args(parse=False).parse_args([])
    args.config_fn = config_fn
    return sim.PEBSI(args)


def main():
    with open(RESULT_FN) as f:
        result = yaml.safe_load(f)
    best = {ad.translate_rgi[g]['6']: (kp, wf) for g, kp, wf
            in zip(result['glaciers'], result['kp'], result['wind_factor'])}

    # the adaptive mesh decides each glacier's point count, so build it first
    # to size the per-point parameter arrays
    rgiid_n = np.asarray(Terrain(model_for(config_with()).params).rgiid_n, dtype=str)
    kp = [best[gid][0] for gid in rgiid_n]
    wind_factor = [best[gid][1] for gid in rgiid_n]

    model_for(config_with(kp=kp, wind_factor=wind_factor)).run()


if __name__ == '__main__':
    main()
