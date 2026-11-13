"""
Warming-only mass loss from the synthetic heatwaves, and a simple model that predicts it.

Each synthetic event also sets precipitation to zero. The snowfall that would have fallen
during the event in the no-event run is rebuilt from the MERRA-2 forcing with PEBSI's own
steps (temperature quantile mapping to Gates station, lapse rate, kp, precipitation gradient,
high-elevation taper, 0.2-2.2 C snow/rain ramp) and added back to the effect, leaving the
mass loss from warming alone. Kennicott's MERRA-2 cell had no detected heatwaves in 2006,
so the original 2006 forcing is the no-event forcing.

The warming-only loss is modelled as
    loss = k * intensity * sum over event days of albedo(day)
where albedo is the glacier-wide midday (10-16 local) albedo of the no-event run on that day.
Skill is judged by leave-one-out prediction.
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

ROOT = '/home/claire/Research/'
OUT_DIR = ROOT + 'Output/synthetic_heatwaves/'
RESULTS_FN = os.path.join(OUT_DIR, 'synthetic_heatwave_results.csv')
ANOMALY_FN = os.path.join(OUT_DIR, 'synthetic_heatwave_anomaly.csv')
BASE_RUN = os.path.join(OUT_DIR, 'run_000_0', 'output.zarr')
OUT_FN = os.path.join(OUT_DIR, 'synthetic_heatwave_warming_effect.csv')
FIG_FN = os.path.join(OUT_DIR, 'synthetic_heatwave_sensitivity.png')
MERRA_FP = ROOT + 'climate_data/MERRA2/reg01/{v}_reg01.zarr'
QM_FN = ROOT + 'PEBSI/data/quantile_cdfs.nc'

LAT, LON = 61.5, -143.125
UTC_OFFSET = pd.Timedelta(hours=8)
WINDOW = slice('2006-04-25', '2006-10-01')
# from run_000_0/config.yaml and pebsi/defaults.py
KP = 1.6164887854490415
PRECGRAD = 1e-4             # m-1
ZMED = 1826.                # Kennicott RGI median elevation [m]
LAPSE = -6.5e-3             # K m-1
STATION_ELEV = 1237.        # Gates station, trained on this MERRA-2 cell
SNOW_LOW, SNOW_HIGH = 0.2, 2.2
PRECLIMIT_FRAC, PRECLIMIT_FLOOR = 0.75, 0.875


def cell_series(var):
    da = xr.open_zarr(MERRA_FP.format(v=var))[var].sel(lat=LAT, lon=LON, method='nearest')
    return da.sel(time=WINDOW).to_series().astype(float)


def glacier_snowfall(temp_c, precip_m, elev, weight):
    """Glacier-wide hourly snowfall [m w.e.] as PEBSI builds it from cell temp and precip."""
    q = xr.open_dataset(QM_FN, group='temp')
    s = list(q.station.values).index('gates_1237')
    qs, merra_cdf, aws_cdf = q['quantile'].values, q.merra_cdf.values[s], q.aws_cdf.values[s]
    quant = np.interp(temp_c, merra_cdf, qs)
    temp = temp_c + np.interp(quant, qs, aws_cdf) - np.interp(quant, qs, merra_cdf)
    temp = temp[:, None] + LAPSE * (elev[None, :] - STATION_ELEV)

    tp = precip_m[:, None] * KP * (1 + PRECGRAD * (elev[None, :] - ZMED))
    threshold = elev.min() + PRECLIMIT_FRAC * (elev.max() - elev.min())
    upper = elev >= threshold
    height_75 = elev[upper].min()
    taper = np.exp(-(elev - height_75) / (elev[upper].max() - height_75))
    anchor = np.where(upper)[0][np.argmin(elev[upper])]
    tp = np.where(upper, tp[:, [anchor]] * taper, tp)
    tp = np.where(upper, np.maximum(tp, PRECLIMIT_FLOOR * tp.max(1, keepdims=True)), tp)

    frac_rain = np.clip((temp - SNOW_LOW) / (SNOW_HIGH - SNOW_LOW), 0, 1)
    return (tp * (1 - frac_rain)) @ weight


# ---- removed snowfall and warming-only effect
base = xr.open_zarr(BASE_RUN, consolidated=False)
elev, weight = base['elev'].values, base['weight'].values
temp = cell_series('T2M') - 273.15
precip = cell_series('PRECTOTCORR') / 1000 * 3600
snow = pd.Series(glacier_snowfall(temp.values, precip.values, elev, weight), index=temp.index)
snow.index = snow.index - UTC_OFFSET

results = pd.read_csv(RESULTS_FN, parse_dates=['start'])
end = results['start'] + pd.to_timedelta(results['duration'], unit='D')
results['removed_snowfall'] = [snow[(snow.index >= s) & (snow.index < e)].sum() if n > 0 else 0.
                               for s, e, n in zip(results['start'], end, results['duration'])]
results['warming_effect'] = results['effect'] + results['removed_snowfall']
results.to_csv(OUT_FN, index=False)
print(f"Saved {OUT_FN}")

# ---- glacier-wide midday albedo of the no-event run, by local day
albedo = (base['albedo'].sel(time=WINDOW).values @ weight)
albedo = pd.Series(albedo, index=pd.to_datetime(base['time'].sel(time=WINDOW).values) - UTC_OFFSET)
albedo = albedo[(albedo.index.hour >= 10) & (albedo.index.hour <= 16)].resample('1D').mean()

events = results.query('run > 0').reset_index(drop=True)
duration = events['duration'].values.astype(float)
intensity = events['intensity'].values
loss = -events['warming_effect'].values
sum_albedo = np.array([albedo[s:s + pd.Timedelta(days=n - 1)].sum()
                       for s, n in zip(events['start'], events['duration'])])
mean_albedo = sum_albedo / duration


def predict(p, m):
    return p[0] * intensity[m] * sum_albedo[m]


def fit(m):
    return least_squares(lambda p: predict(p, m) - loss[m], [0.01]).x


every = np.ones(len(loss), bool)
p = fit(every)
loo = np.array([predict(fit(np.arange(len(loss)) != i), np.arange(len(loss)) == i)[0]
                for i in range(len(loss))])
loo_r2 = 1 - ((loss - loo) ** 2).sum() / ((loss - loss.mean()) ** 2).sum()
loo_rmse = np.sqrt(((loss - loo) ** 2).mean())
k = p[0]
print(f"removed snowfall: mean {events['removed_snowfall'].mean():.3f}, "
      f"max {events['removed_snowfall'].max():.3f} m w.e.")
print(f"k = {k:.4f}; LOO R2 {loo_r2:.3f}, LOO RMSE {loo_rmse:.4f}")

# ---- figure
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.5, 3.4))

per_day = loss / (duration * intensity)
ax0.scatter(mean_albedo, per_day, s=14, color='C0')
xs = np.linspace(mean_albedo.min(), mean_albedo.max(), 2)
ax0.plot(xs, k * xs, color='k', lw=1, label=f'{k:.4f} × albedo')
ax0.legend(fontsize=9, frameon=False)
ax0.set_xlabel('Mean glacier albedo during event,\nno-event run', fontsize=9)
ax0.set_ylabel('Warming mass loss per event day\n÷ intensity (m w.e.)', fontsize=9)
ax0.set_title('a) Loss per heatwave day', fontsize=9, loc='left')

lim = np.ceil(max(loss.max(), loo.max()) * 20) / 20
ax1.plot([-lim, 0], [-lim, 0], color='0.5', lw=1)
ax1.scatter(-loss, -loo, s=14, color='C0')
ax1.set_xlim(-lim, 0)
ax1.set_ylim(-lim, 0)
ax1.set_aspect('equal')
ax1.set_xlabel('PEBSI warming-only change\nin mass balance (m w.e.)', fontsize=9)
ax1.set_ylabel('Predicted, leave-one-out (m w.e.)', fontsize=9)
ax1.set_title('b) Prediction skill', fontsize=9, loc='left')
ax1.text(0.05, 0.95, f'R² = {loo_r2:.2f}\nRMSE = {loo_rmse:.3f} m w.e.', transform=ax1.transAxes,
         va='top', fontsize=9)
for ax in (ax0, ax1):
    ax.tick_params(labelsize=9)

plt.tight_layout()
plt.savefig(FIG_FN, dpi=300)
print(f"Saved {FIG_FN}")
