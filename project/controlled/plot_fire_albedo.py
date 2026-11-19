"""
Fire effect on glacier albedo in the MERRA-2 deposition controlled runs (Base minus No fires), against
each May-Sep's fire-attributable BC deposition, 2000-2025.

The hourly output has no snow depth, so a point-day counts as snow-covered when its ~local-noon
(21 UTC) albedo is above the point's bare-surface albedo: the ice tif or firn albedo, a constant
value the point returns to on many days (points that never reach one are always snow-covered).
Per glacier-season:
  d_alb        change in the effective albedo, 1 - sum(SWnet) / sum(SWin) over points and days
  d_alb_snow   the part from days snow-covered in both runs (darker snow)
  d_alb_area   the part from days snow-covered without fires but bare with them (snow lost early)
  d_min_sca    change in the season's minimum snow-covered area fraction
  d_min_snow   change in the season's minimum glacier-mean snow albedo (snow-covered points, noon)
The figure (in figs/) shows d_alb; the CSV holds every column.
Daily per-point inputs (DAILY_FN: May-Sep daily SWin and SWnet sums and 21 UTC albedo) were extracted
from the hourly output.zarr of each run.
"""
import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'synthetic'))
os.environ['EVENT'] = 'wildfire'
from plot_event_grid import MERRA2_FP, MERRA2_VARS, merra2_ratio, GLACIERS, translate_rgi

OUT = '/home/claire/Research/Output/controlled/'
DAILY_FN = OUT + '{}_0/per_point_daily.nc'
CSV_FN = OUT + 'fire_albedo.csv'
FIG_FN = OUT + 'figs/fire_albedo.png'
# the shared glacier colors (shop/plotting/plot_summer_anomaly_2004.py) and one marker per glacier
GLACIER_COLORS = {'wolverine': '#63c4c7', 'kennicott': '#fcc02e', 'kahiltna': '#4D559C',
                  'lemon_creek': '#60C252', 'gulkana': '#BF1F6A'}
MARKERS = dict(zip(GLACIERS, ['o', 's', '^', 'D', 'v']))


def fire_bc(lat, lon):
    """May-Sep fire-attributable BC deposition [mg m-2] per year at the MERRA-2 cell nearest (lat, lon),
    as PEBSI sees it."""
    lat, lon = round(lat / 0.5) * 0.5, round(lon / 0.625) * 0.625
    total = 0
    for name in ['bcdry', 'bcwet']:
        vn = MERRA2_VARS[name]
        s = [xr.open_zarr(MERRA2_FP.format(d=d, v=vn), consolidated=False)[vn].sel(lat=lat, lon=lon).to_series()
             for d in ['reg01', 'reg01_no_wildfires']]
        total = total + (s[0] - s[1]) * float(merra2_ratio(name, lat, lon)) * 3600 * 1e6
    total = total[total.index.month.isin(range(5, 10))].clip(lower=0)
    return total.groupby(total.index.year).sum()


def snow_mask(alb_b, alb_n):
    """Snow-covered point-days of both runs: noon albedo above the point's recurring bare value."""
    both = np.concatenate([alb_b, alb_n])
    bare = np.nanmin(both, axis=0)
    recurs = (np.abs(both - bare) < 1e-4).sum(axis=0) >= 5
    bare = np.where(recurs, bare, -1)
    return alb_b > bare + 0.005, alb_n > bare + 0.005


def main():
    b, n = (xr.open_dataset(DAILY_FN.format(r)).load() for r in ['base_merra2', 'no_fires_merra2'])
    # incoming shortwave differs slightly between runs (0.07% on average), so changes are weighted by Base's
    swin = b['shortwave_in'].values
    a_b = 1 - b['shortwave_net'].values / np.where(swin > 0, swin, np.nan)
    swin_n = n['shortwave_in'].values
    a_n = 1 - n['shortwave_net'].values / np.where(swin_n > 0, swin_n, np.nan)
    snow_b, snow_n = snow_mask(b['albedo'].values, n['albedo'].values)
    years = b.time.dt.year.values
    rgiid, weight = b['rgiid'].values.astype(str), b['weight'].values

    zarr = xr.open_zarr(OUT + 'base_merra2_0/output.zarr', consolidated=False)
    plat, plon = zarr['lat'].values, zarr['lon'].values
    rows = []
    for g in GLACIERS:
        m = rgiid == translate_rgi[g]['6']
        w = weight[m]
        bc = fire_bc(float(np.average(plat[m], weights=w)), float(np.average(plon[m], weights=w)))
        for y in range(2000, 2026):
            t = years == y
            sw = swin[t][:, m] * w
            d = np.nan_to_num((a_b[t][:, m] - a_n[t][:, m]) * sw)
            sb, sn = snow_b[t][:, m], snow_n[t][:, m]
            sca_b, sca_n = (sb * w).sum(1) / w.sum(), (sn * w).sum(1) / w.sum()
            alb_b, alb_n = b['albedo'].values[t][:, m], n['albedo'].values[t][:, m]
            snow_alb = [((a * s * w).sum(1) / (s * w).sum(1)) for a, s in [(alb_b, sb), (alb_n, sn)]]
            rows.append(dict(glacier=g, year=y, fire_bc=bc.get(y, 0.0),
                             d_alb=d.sum() / sw.sum(),
                             d_alb_snow=d[sb & sn].sum() / sw.sum(),
                             d_alb_area=d[~sb & sn].sum() / sw.sum(),
                             d_min_sca=sca_b.min() - sca_n.min(),
                             d_min_snow=np.nanmin(snow_alb[0]) - np.nanmin(snow_alb[1]),
                             min_sca=sca_n.min()))
    df = pd.DataFrame(rows)
    df.to_csv(CSV_FN, index=False)
    print(f'Saved {CSV_FN}')
    cols = ['d_alb', 'd_alb_snow', 'd_alb_area', 'd_min_sca', 'd_min_snow']
    print(df.groupby('glacier').apply(lambda d: d[cols].corrwith(d['fire_bc'])).round(2))
    print(df.sort_values('fire_bc', ascending=False).head(12).round(4).to_string())

    fig, ax = plt.subplots(figsize=(4, 4))
    for g in GLACIERS:
        d = df[df['glacier'] == g]
        ax.scatter(d['fire_bc'], d['d_alb'], s=16, marker=MARKERS[g], color=GLACIER_COLORS[g],
                   label=g.replace('_', ' ').title())
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlim(right=62)  # room for the legend clear of the points
    ax.set_xlabel('Season fire BC deposition (mg m$^{-2}$)', fontsize=9)
    ax.set_ylabel('Change in effective albedo from fires', fontsize=9)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1, 0.88), handletextpad=0.1,
              borderaxespad=0.2)
    plt.tight_layout()
    plt.savefig(FIG_FN, dpi=300)
    print(f'Saved {FIG_FN}')


if __name__ == '__main__':
    main()
