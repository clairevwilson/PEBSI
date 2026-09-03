"""Backward accumulation probe for gulkana/AU to find critical window."""
import os, warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
os.environ.setdefault('PEBSI_NORMAL_RUN_2015_2020', '1')
import jax
jax.config.update('jax_debug_nans', False)
jax.config.update('jax_enable_x64', True)
import time, yaml, numpy as np, jax.numpy as jnp
import jax_optimize as jo
from pebsi.main import main as pebsi_main

WF = float(os.environ.get('PEBSI_NAN_WF', '0.8125'))
KP = 1.1442
SITE_INDEX = 12
STEP = 500
END_HOUR = 31985

site_dict = jo.load_reduced_site_dict(jo.REDUCED_SITES_CONFIG)
site_order = jo.flatten_site_order(site_dict)
glacier, site = site_order[SITE_INDEX]
single = {glacier: [site]}
fp = jo.build_generated_config(single, jo.host, start_date=jo.DEBUG_START_DATE, end_date=jo.DEBUG_END_DATE, temporal_chunk_years=1)
with open(fp) as f: cfg = yaml.safe_load(f)
cfg['constant_snowfall_density'] = False
with open(fp, 'w') as f: yaml.dump(cfg, f, sort_keys=False)
model = jo.init_pebsi(fp)
params = model.config.params
static_args = model.config.static_args
dynamic_args = model.config.dynamic_args

dargs_f32 = dynamic_args._replace(
    wind_factor=jnp.array([WF], dtype=jnp.float32),
    kp=jnp.array([KP], dtype=jnp.float32),
)

probe_hours = list(range(0, END_HOUR, STEP)) + [END_HOUR - 1]
probe_hours = sorted(set(probe_hours))
print(f"gulkana/AU backward accumulation  WF={WF}  STEP={STEP}", flush=True)
print(f"Pre-computing {len(probe_hours)} snapshots...", flush=True)

snapshots = {}
state = model.initial_state
prev = 0
t0 = time.time()
for h in probe_hours:
    if h > prev:
        f = model.pack_forcings(params, model.dates[prev:h], prev)
        state, _ = pebsi_main(state, f, model.point_attrs, static_args, dargs_f32)
        state = jax.lax.stop_gradient(state)
    snapshots[h] = state
    prev = h
print(f"  done in {time.time()-t0:.1f}s\n", flush=True)

def window_grad(wf_val, snap, forcings):
    snap_f64 = jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float64) if jnp.issubdtype(x.dtype, jnp.floating) else x, snap)
    dargs = dynamic_args._replace(
        wind_factor=jnp.array([wf_val], dtype=jnp.float64),
        kp=jnp.array([KP], dtype=jnp.float64),
    )
    final, _ = pebsi_main(snap_f64, forcings, model.point_attrs, static_args, dargs)
    return jnp.sum(final.lice.astype(jnp.float64))

grad_fn = jax.jit(jax.grad(window_grad))

print(f"{'start_hour':>12} {'window_len':>10} {'grad':>16} {'|grad|':>14}", flush=True)
print('-' * 58, flush=True)
for h in reversed(probe_hours):
    snap = snapshots[h]
    forcings = model.pack_forcings(params, model.dates[h:END_HOUR], h)
    t0 = time.time()
    try:
        g = float(grad_fn(jnp.float64(WF), snap, forcings))
    except: g = float('nan')
    print(f"{h:>12} {END_HOUR-h:>10} {g:>+16.4e} {abs(g):>14.4e}  ({time.time()-t0:.0f}s)", flush=True)
