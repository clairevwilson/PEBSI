"""
Print forcing values (especially precipitation/snowfall) at specific hours
to check if gradient-jump hours coincide with precipitation events.
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

HOURS = list(range(13310, 13370)) + list(range(13415, 13430))

if __name__ == '__main__':
    model = build_single_site_model(SITE_INDEX)
    params = model.config.params

    dargs = model.config.dynamic_args._replace(
        wind_factor=jnp.array([0.8125], dtype=jnp.float32),
        kp=jnp.array([KP_VAL], dtype=jnp.float32),
    )

    # pack forcings for the full range and inspect
    lo = min(HOURS)
    hi = max(HOURS) + 2
    forcings = model.pack_forcings(params, model.dates[lo:hi], lo)

    # print available forcing fields
    if hasattr(forcings, '_fields'):
        fields = forcings._fields
    elif hasattr(forcings, '__dict__'):
        fields = list(vars(forcings).keys())
    else:
        fields = [f for f in dir(forcings) if not f.startswith('_')]

    print("Forcing fields:", fields)

    # find precip-related fields
    precip_fields = [f for f in fields if any(k in f.lower() for k in
                     ['precip', 'snow', 'rain', 'solid', 'pcp', 'sf', 'pr'])]
    print("Precip-related:", precip_fields)

    if not precip_fields:
        # just print all scalar-ish fields for first few hours
        print("\nAll forcing values at hours 13340-13350:")
        for f in fields:
            arr = np.array(getattr(forcings, f))
            if arr.ndim <= 2:
                idx_lo = 13340 - lo
                idx_hi = 13350 - lo
                print(f"  {f}: {arr.flat[idx_lo:idx_hi]}")
    else:
        print(f"\n{'hour':>6}", end='')
        for f in precip_fields:
            print(f"  {f:>14}", end='')
        print()
        for h in HOURS:
            idx = h - lo
            print(f"{h:>6}", end='')
            for f in precip_fields:
                arr = np.array(getattr(forcings, f)).flatten()
                val = arr[idx] if idx < len(arr) else float('nan')
                print(f"  {val:>14.6f}", end='')
            print()
