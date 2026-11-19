"""
Compares the temperature distributions stored in the quantile mapping
dataset across stations, after normalizing each to a common reference
elevation with a constant lapse rate.

(a) AWS temperature distributions at the reference elevation
(b) MERRA-2 temperature distributions (same timestamps as each AWS),
    lapsed from the training cell elevation to the reference elevation
(c) Lapse-adjusted QM correction (AWS - MERRA-2) at each quantile, i.e.
    the bias left after accounting for station vs. cell elevation
"""
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

PEBSI_FP = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QM_FN = os.path.join(PEBSI_FP, 'data', 'quantile_cdfs.nc')
MERRA2_Z_FN = '/ocean/projects/ees260009p/cwilson4/climate_data/MERRA2/MERRA2_constants.nc'
FIG_FN = '/ocean/projects/ees260009p/cwilson4/Output/qm_temp_distributions.png'

LAPSE = -0.0065     # K m-1
REF_ELEV = 1500     # m

colors = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6']

ds = xr.open_dataset(QM_FN, group='temp')
quantiles = ds['quantile'].values
station_elev = ds.elevation.values

with xr.open_dataarray(MERRA2_Z_FN) as z:
    cell_elev = z.sel(lat=xr.DataArray(ds.lat.values, dims='station'),
                      lon=xr.DataArray(ds.lon.values, dims='station'),
                      method='nearest').isel(time=0).values / 9.80665

aws = ds.aws_cdf.values + LAPSE * (REF_ELEV - station_elev[:, None])
merra = ds.merra_cdf.values + LAPSE * (REF_ELEV - cell_elev[:, None])

# order stations by median lapse-adjusted AWS temperature, coldest first
order = np.argsort(np.median(aws, axis=1))

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={'width_ratios': [1, 1, 1]})
x = np.linspace(min(aws.min(), merra.min()) - 2, max(aws.max(), merra.max()) + 2, 400)

for c, s in zip(colors, order):
    name = str(ds.station.values[s]).rsplit('_', 1)[0].replace('_', ' ').title()
    label = f'{name} ({station_elev[s]} m; cell {cell_elev[s]:.0f} m)'
    axes[0].plot(x, gaussian_kde(aws[s])(x), color=c, lw=1.8, label=label)
    axes[1].plot(x, gaussian_kde(merra[s])(x), color=c, lw=1.8)
    axes[2].plot(quantiles, aws[s] - merra[s], color=c, lw=1.8)

ymax = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])
for ax, title in zip(axes[:2], ['(a) AWS', '(b) MERRA-2 (same timestamps)']):
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, ymax)
    ax.set_xlabel(f'Temperature at {REF_ELEV} m (°C)')
    ax.set_title(title, loc='left')
axes[0].set_ylabel('Density')
axes[1].set_yticklabels([])

axes[2].axhline(0, color='#52514e', lw=0.8)
axes[2].set_xlim(0, 1)
axes[2].set_xlabel('Quantile')
axes[2].set_ylabel('AWS − MERRA-2, lapse-adjusted (°C)')
axes[2].set_title('(c) QM correction beyond lapse rate', loc='left')

for ax in axes:
    ax.grid(color='#e5e4e0', lw=0.6)
    ax.set_axisbelow(True)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)

fig.legend(loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.13))
fig.suptitle(f'Quantile mapping temperature distributions, normalized with {LAPSE * 1000:g} K km$^{{-1}}$', y=1.0)
fig.tight_layout()
fig.savefig(FIG_FN, dpi=200, bbox_inches='tight')
print(f'saved to {FIG_FN}')
