"""
Prints forward-pass state summary around hour 13466 at wf=0.8125 to
identify what discrete event triggers the NaN gradient at that hour.
"""
import os
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
os.environ.setdefault('PEBSI_NORMAL_RUN_2015_2020', '1')
import jax
jax.config.update('jax_debug_nans', False)

import numpy as np
import jax.numpy as jnp

import jax_optimize as jo
from pebsi.main import main as pebsi_main
from decompose_gradient_by_field import build_single_site_model, SITE_INDEX, KP_VAL

WF_NAN = float(os.environ.get('PEBSI_NAN_WF', '0.8125'))
SNAP_HOUR = int(os.environ.get('PEBSI_SNAP_HOUR', '10962'))
EVENT_HOUR = int(os.environ.get('PEBSI_EVENT_HOUR', '13466'))
WINDOW = int(os.environ.get('PEBSI_WINDOW', '5'))

if __name__ == '__main__':
    model = build_single_site_model(SITE_INDEX)
    params = model.config.params
    static_args = model.config.static_args
    dynamic_args = model.config.dynamic_args

    fixed_dargs = dynamic_args._replace(
        wind_factor=jnp.array([WF_NAN], dtype=jnp.float32),
        kp=jnp.array([KP_VAL], dtype=jnp.float32),
    )

    print(f"Fast-forwarding to hour {SNAP_HOUR}...", flush=True)
    forcings_pre = model.pack_forcings(params, model.dates[:SNAP_HOUR], 0)
    snap, _ = pebsi_main(model.initial_state, forcings_pre, model.point_attrs, static_args, fixed_dargs)

    lo = max(SNAP_HOUR, EVENT_HOUR - WINDOW)
    hi = EVENT_HOUR + WINDOW + 1
    print(f"Stepping hour by hour from {lo} to {hi-1}...\n", flush=True)

    # Step forward to lo from snap
    forcings_mid = model.pack_forcings(params, model.dates[SNAP_HOUR:lo], SNAP_HOUR)
    state, _ = pebsi_main(snap, forcings_mid, model.point_attrs, static_args, fixed_dargs)

    print(f"{'hour':>6} {'lice_sum':>12} {'lwater_sum':>12} {'ltemp_min':>12} "
          f"{'lheight_sum':>12} {'n_layers_ice':>14} {'lice_min_nz':>14}", flush=True)

    for h in range(lo, hi):
        s = state
        lice = np.array(s.lice[0])  # first (only) point
        lwater = np.array(s.lwater[0])
        ltemp = np.array(s.ltemp[0])
        lheight = np.array(s.lheight[0])
        ice_mask = np.array(s.ice_mask[0])

        n_ice = int(np.sum(ice_mask))
        lice_nz = lice[lice > 0]
        lice_min_nz = float(lice_nz.min()) if len(lice_nz) > 0 else 0.0

        print(f"{h:>6} {float(np.sum(lice)):>12.4f} {float(np.sum(lwater)):>12.6f} "
              f"{float(np.min(ltemp)):>12.4f} {float(np.sum(lheight)):>12.4f} "
              f"{n_ice:>14} {lice_min_nz:>14.6f}", flush=True)

        # Advance by one hour
        forcings_1h = model.pack_forcings(params, model.dates[h:h+1], h)
        state, _ = pebsi_main(s, forcings_1h, model.point_attrs, static_args, fixed_dargs)
