"""Print key forcing fields for a specified hour range."""
import os, warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
os.environ.setdefault('PEBSI_NORMAL_RUN_2015_2020', '1')
import jax_optimize as jo
import numpy as np
from decompose_gradient_by_field import build_single_site_model, SITE_INDEX, KP_VAL

LO = int(os.environ.get('PEBSI_LO', '13388'))
HI = int(os.environ.get('PEBSI_HI', '13400'))

model = build_single_site_model(SITE_INDEX)
params = model.config.params
forcings = model.pack_forcings(params, model.dates[LO:HI+1], LO)

fields = ['temp', 'rh', 'tp', 'wind', 'shortwave_in', 'local_hour']
print(f"{'hour':>6}", end='')
for f in fields:
    print(f"  {f:>14}", end='')
print()

for h in range(LO, HI+1):
    i = h - LO
    print(f"{h:>6}", end='')
    for f in fields:
        arr = np.array(getattr(forcings, f)).flatten()
        val = arr[i] if i < len(arr) else float('nan')
        print(f"  {val:>14.6g}", end='')
    print()
