"""
Sweeps wind_factor and records L(wf) = ||lice(END_HOUR)||^2 (forward-only,
dense grid) plus AD dL/dwf (coarse grid, full backward each). Writes
wf_sweep.csv and wf_sweep.png: if the roughness diagnosis is right, the AD
gradient oscillates 100x around the secant slope visible in L(wf).
Same env vars as decompose_gradient_by_field.py.
"""
import os
os.environ.setdefault('PEBSI_NORMAL_RUN_2015_2020', '1')

import time
import numpy as np
import jax
import jax.numpy as jnp

import jax_optimize as jo
from pebsi.main import main as pebsi_main
from decompose_gradient_by_field import build_single_site_model, SITE_INDEX, WF_VAL, KP_VAL, END_HOUR

# jax_optimize turns on debug_nans; here a non-finite gradient is a data
# point to record, not a crash
jax.config.update('jax_debug_nans', False)

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

    def loss(wf_val):
        dargs = dynamic_args._replace(
            wind_factor=jnp.array([wf_val], dtype=jnp.float32) * jnp.ones(1, dtype=jnp.float32),
            kp=jnp.array([KP_VAL], dtype=jnp.float32),
        )
        final_state, _ = pebsi_main(model.initial_state, forcings, model.point_attrs, static_args, dargs)
        return jnp.sum(jnp.square(final_state.lice))

    loss_jit = jax.jit(loss)
    grad_jit = jax.jit(jax.value_and_grad(loss))

    dense_wf = np.linspace(WF_LO, WF_HI, N_DENSE)
    grad_wf = np.linspace(WF_LO, WF_HI, N_GRAD)

    dense_L = []
    for i, w in enumerate(dense_wf):
        t0 = time.time()
        L = float(loss_jit(w))
        dense_L.append(L)
        print(f"dense {i+1}/{N_DENSE} wf={w:.5f} L={L:.8e} ({time.time()-t0:.1f}s)", flush=True)

    grad_L, grad_g = [], []
    for i, w in enumerate(grad_wf):
        t0 = time.time()
        try:
            L, g = grad_jit(w)
            L, g = float(L), float(g)
        except FloatingPointError:
            L, g = float('nan'), float('nan')
        grad_L.append(L)
        grad_g.append(g)
        print(f"grad {i+1}/{N_GRAD} wf={w:.5f} L={L:.8e} dL/dwf={g:.6e} ({time.time()-t0:.1f}s)", flush=True)

    import csv
    with open('wf_sweep.csv', 'w', newline='') as f:
        wcsv = csv.writer(f)
        wcsv.writerow(['kind', 'wf', 'L', 'dLdwf'])
        for w, L in zip(dense_wf, dense_L):
            wcsv.writerow(['dense', w, L, ''])
        for w, L, g in zip(grad_wf, grad_L, grad_g):
            wcsv.writerow(['grad', w, L, g])

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    ax1.plot(dense_wf, dense_L, '.-', lw=0.8, ms=4)
    ax1.set_ylabel(r'$L = \|\mathrm{lice}\|^2$')
    ax1.set_title(f'K53 loss landscape and AD gradient, hours 1-{END_HOUR}, kp={KP_VAL}')

    secant = np.gradient(np.array(dense_L), dense_wf)
    ax2.plot(dense_wf, secant, '.-', lw=0.8, ms=4, label=f'secant slope of dense grid (h={dense_wf[1]-dense_wf[0]:.3f})')
    ax2.plot(grad_wf, grad_g, 'rx', ms=10, label='AD dL/dwf')
    ax2.set_yscale('symlog', linthresh=1e7)
    ax2.axhline(0, color='k', lw=0.5)
    ax2.set_xlabel('wind_factor')
    ax2.set_ylabel('dL/dwf')
    ax2.legend()
    fig.tight_layout()
    fig.savefig('wf_sweep.png', dpi=150)
    print("wrote wf_sweep.csv, wf_sweep.png", flush=True)
