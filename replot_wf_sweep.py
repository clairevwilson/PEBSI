"""
Replots wf_sweep.csv with a physical y-axis: total ice mass in m w.e.
The raw CSV stores ||lice||^2 (sum of squared per-layer masses). We need to
re-run the forward pass at each dense wf to get sum(lice) instead — but
each is only ~3s, and we have 46 points, so ~2.5 min total.
"""
import os
os.environ.setdefault('PEBSI_NORMAL_RUN_2015_2020', '1')
import jax
jax.config.update('jax_debug_nans', False)

import time
import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import jax_optimize as jo
from pebsi.main import main as pebsi_main
from decompose_gradient_by_field import build_single_site_model, SITE_INDEX, KP_VAL, END_HOUR

DENSITY_WATER = 1000.0  # kg/m³, for m w.e. conversion

WF_LO = float(os.environ.get('PEBSI_SWEEP_LO', '0.70'))
WF_HI = float(os.environ.get('PEBSI_SWEEP_HI', '0.88'))
N_DENSE = int(os.environ.get('PEBSI_SWEEP_N_DENSE', '46'))
N_GRAD = int(os.environ.get('PEBSI_SWEEP_N_GRAD', '9'))

if __name__ == '__main__':
    model = build_single_site_model(SITE_INDEX)
    params = model.config.params
    static_args = model.config.static_args
    dynamic_args = model.config.dynamic_args
    forcings = model.pack_forcings(params, model.dates[:END_HOUR], 0)

    def total_ice_mwe(wf_val):
        dargs = dynamic_args._replace(
            wind_factor=jnp.array([wf_val], dtype=jnp.float32),
            kp=jnp.array([KP_VAL], dtype=jnp.float32),
        )
        final_state, _ = pebsi_main(model.initial_state, forcings, model.point_attrs, static_args, dargs)
        return jnp.sum(final_state.lice) / DENSITY_WATER

    fwd_jit = jax.jit(total_ice_mwe)
    grad_jit = jax.jit(jax.value_and_grad(total_ice_mwe))

    dense_wf = np.linspace(WF_LO, WF_HI, N_DENSE)
    grad_wf = np.linspace(WF_LO, WF_HI, N_GRAD)

    print("Computing dense forward sweep...", flush=True)
    dense_ice = []
    for i, w in enumerate(dense_wf):
        t0 = time.time()
        v = float(fwd_jit(w))
        dense_ice.append(v)
        print(f"  {i+1}/{N_DENSE} wf={w:.4f} ice={v:.4f} m w.e. ({time.time()-t0:.1f}s)", flush=True)

    print("Computing AD gradients for physical loss...", flush=True)
    grad_wf_pts = np.linspace(WF_LO, WF_HI, N_GRAD)
    grad_g_pts = []
    for i, w in enumerate(grad_wf_pts):
        t0 = time.time()
        try:
            _, g = grad_jit(w)
            g = float(g)
        except FloatingPointError:
            g = float('nan')
        grad_g_pts.append(g)
        print(f"  {i+1}/{N_GRAD} wf={w:.4f} dL/dwf={g:.4e} ({time.time()-t0:.1f}s)", flush=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 4), sharex=True)

    ax1.plot(dense_wf, dense_ice, '.-', lw=0.8, ms=4, color='steelblue')
    ax1.set_ylabel('Total ice mass (m w.e.)')
    ax1.set_title(f'K53 ice mass vs wind_factor, kp={KP_VAL}')

    secant = np.gradient(np.array(dense_ice), dense_wf)
    ax2.plot(dense_wf, secant, '.-', lw=0.8, ms=4, color='steelblue',
             label=f'secant slope')

    g_arr = np.array(grad_g_pts, dtype=float)
    finite_mask = np.isfinite(g_arr)
    ax2.scatter(grad_wf_pts[finite_mask], g_arr[finite_mask],
                color='red', marker='x', s=80, zorder=5, label='AD gradient')

    ax2.set_yscale('symlog', linthresh=0.1)
    ax2.set_yticks([10**-20, 10**-10, 10**-5, 10**0, 10**5, 10**10, 10**20])
    ax2.axhline(0, color='k', lw=0.5)
    ax2.set_xlabel('wind_factor')
    ax2.set_ylabel('d(ice m w.e.)/dwf')
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig('wf_sweep_physical.png', dpi=150)
    print("wrote wf_sweep_physical.png", flush=True)
