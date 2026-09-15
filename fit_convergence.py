"""
Fits the mass balance sweep itself to find where it levels out.

Discretization error goes as MB(h) = MB_inf + A*h^p, so fitting that
gives both the converged value and the order of convergence p. An
exponential in N is fitted alongside for comparison. The asymptotic form
only holds as h shrinks, so each fit is repeated on the finer half to
show whether the answer is stable.
"""
import os

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

RESULTS = 'project/point_density_results/'
YEARS = 3.0
TOL = 0.01   # m w.e./yr


def power(h, mb_inf, a, p):
    return mb_inf + a * h ** p


def expo(n, mb_inf, a, tau):
    return mb_inf - a * np.exp(-n / tau)


for glacier in ('gulkana', 'kennicott'):
    df = pd.read_csv(os.path.join(RESULTS, f'{glacier}_mesh_convergence.csv'))
    df = df.sort_values('point_spacing', ascending=False).reset_index(drop=True)
    h = df['point_spacing'].values.astype(float)
    n = df['actual_n_points'].values.astype(float)
    mb = df['mass_balance'].values / YEARS

    print('=' * 72)
    print(f'{glacier.upper()}   {len(h)} resolutions, h from {h.max():.0f} '
          f'to {h.min():.0f} m')
    print('=' * 72)

    for label, mask in (('all resolutions', np.ones(len(h), bool)),
                        ('finer half', h <= np.median(h))):
        hh, nn, yy = h[mask], n[mask], mb[mask]
        sign = np.sign(yy[-1] - yy[0]) or 1.0
        try:
            pw, _ = curve_fit(power, hh, yy,
                              p0=[yy[-1], -sign * abs(yy[-1] - yy[0]), 0.8],
                              maxfev=60000)
            rp = yy - power(hh, *pw)
            r2p = 1 - (rp ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
        except Exception as e:
            pw, r2p = (np.nan,) * 3, np.nan

        try:
            ex, _ = curve_fit(expo, nn, yy,
                              p0=[yy[-1], yy[-1] - yy[0], nn.mean()],
                              maxfev=60000)
            re = yy - expo(nn, *ex)
            r2e = 1 - (re ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
        except Exception:
            ex, r2e = (np.nan,) * 3, np.nan

        print(f'\n  --- {label} ({mask.sum()} points) ---')
        print(f'  power  MB = {pw[0]:+.4f} {pw[1]:+.4g}*h^{pw[2]:.3f}   '
              f'R2={r2p:.4f}')
        print(f'         converged = {pw[0]:+.4f} m w.e./yr, order p = {pw[2]:.2f}')
        if np.isfinite(pw[0]) and np.isfinite(pw[2]) and pw[2] > 0:
            h_tol = (TOL / abs(pw[1])) ** (1 / pw[2])
            print(f'         within {TOL} of converged at h = {h_tol:.0f} m')
        print(f'  expon  MB = {ex[0]:+.4f} - {ex[1]:+.4g}*exp(-N/{ex[2]:.0f})  '
              f'R2={r2e:.4f}')
        print(f'         converged = {ex[0]:+.4f} m w.e./yr, '
              f'e-folding N = {ex[2]:.0f}')

    print()
    print(f'  finest run in sweep: {mb[-1]:+.4f} m w.e./yr at N={int(n[-1])}')
    print()
