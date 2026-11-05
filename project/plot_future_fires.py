"""
Compares the base controlled run against the two SSP245 future fire
deposition runs (MPI, GFDL): water-year minimum snow albedo and
water-year mass balance for each glacier, as future minus base.

Snow vs ice: no surface type is stored, but ice albedo is a fixed
per-point value that does not depend on deposition, while snow albedo
does. A timestep is treated as ice when all three runs have the
same albedo there; those timesteps are dropped before taking the
minimum. Each glacier's value is the area-weighted mean of its
points' minima.

Mass balance comes from per_glacier.nc written by
shop/postprocessing/aggregate_per_glacier.py.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

base_fp = '/ocean/projects/ees260009p/cwilson4/Output/controlled/'
runs = {'Base': 'base_0', 'SSP245 MPI': 'ssp245_mpi_0', 'SSP245 GFDL': 'ssp245_gfdl_0'}
colors = {'SSP245 MPI': '#2a78d6', 'SSP245 GFDL': '#eb6834'}

# the SSP245 deposition ends 2014-11-30, so those runs stop 2014-09-01;
# WY1998-2013 are the complete water years common to all three runs
end_date = '2014-08-31 23:00'
first_wy, last_wy = 1998, 2013


def fmt_name(glacier):
    return ' '.join(f.capitalize() for f in glacier.split('_'))


def water_year(da, how):
    annual = getattr(da.resample(time='YS-OCT'), how)()
    annual = annual.assign_coords(time=annual.time.dt.year + 1).rename(time='water_year')
    return annual.sel(water_year=slice(first_wy, last_wy))


# annual mass balance per glacier
glac = {sim: xr.open_dataset(base_fp + f'{run}/per_glacier.nc') for sim, run in runs.items()}
mb = {sim: water_year(ds['mass_balance'].sel(time=slice(None, end_date)), 'sum').load()
      for sim, ds in glac.items()}
glaciers = glac['Base'].glacier.values
gid_to_name = dict(zip(glac['Base'].rgiid.values, glaciers))

# annual minimum snow albedo at each point, then weighted per glacier
zarrs = {sim: xr.open_zarr(base_fp + f'{run}/output.zarr', consolidated=False)
         for sim, run in runs.items()}
albedo = {sim: ds['albedo'].sel(time=slice(None, end_date)) for sim, ds in zarrs.items()}
ice = (albedo['Base'] == albedo['SSP245 MPI']) & (albedo['Base'] == albedo['SSP245 GFDL'])
snow_min = {sim: water_year(a.where(~ice), 'min').load() for sim, a in albedo.items()}

points = zarrs['Base']
weight = xr.DataArray(points['weight'].values, dims='point')
point_glacier = np.array([gid_to_name[g] for g in points['rgiid'].values.astype(str)])


def glacier_mean(da):
    out = []
    for g in glaciers:
        mask = point_glacier == g
        d, w = da.isel(point=mask), weight.isel(point=mask)
        out.append((d * w).sum('point') / w.where(d.notnull()).sum('point'))
    return xr.concat(out, dim=xr.DataArray(glaciers, dims='glacier', name='glacier'))


alb = {sim: glacier_mean(da) for sim, da in snow_min.items()}

no_snow = snow_min['Base'].isnull().mean('point')
print(f'share of points with no snow timesteps left (by WY): '
      f'max {float(no_snow.max()):.3f}', flush=True)

# summary table: base means and mean future minus base over the common years
rows = []
for g in glaciers:
    row = {'glacier': fmt_name(g),
           'base min snow albedo': float(alb['Base'].sel(glacier=g).mean()),
           'base MB (m w.e.)': float(mb['Base'].sel(glacier=g).mean())}
    for sim in colors:
        row[f'{sim} d albedo'] = float((alb[sim] - alb['Base']).sel(glacier=g).mean())
        row[f'{sim} d MB (m w.e.)'] = float((mb[sim] - mb['Base']).sel(glacier=g).mean())
    rows.append(row)
table = pd.DataFrame(rows).set_index('glacier')
print(table.round(4).to_string(), flush=True)
table.to_csv(base_fp + 'future_fires_summary.csv')

# future minus base: albedo on top, mass balance below
fig, axes = plt.subplots(2, len(glaciers), figsize=(11, 4.5), layout='constrained',
                         sharex=True, sharey='row')
for j, g in enumerate(glaciers):
    for row, (data, label) in enumerate([(alb, 'Δ min snow albedo'),
                                         (mb, 'Δ annual MB (m w.e.)')]):
        ax = axes[row, j]
        ax.axhline(0, color='0.6', linewidth=0.8)
        for sim, c in colors.items():
            diff = (data[sim] - data['Base']).sel(glacier=g)
            ax.plot(diff.water_year, diff, color=c, linewidth=2, marker='o',
                    markersize=4, label=sim)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', color='0.9', linewidth=0.6)
        if j == 0:
            ax.set_ylabel(label)
        if row == 0:
            ax.set_title(fmt_name(g), fontsize=10)
        else:
            ax.tick_params(axis='x', labelrotation=45)
fig.supxlabel('Water year')
axes[0, 0].legend(frameon=False, fontsize=8)
fig.suptitle('Future fire deposition minus base (WY1998–2013)', fontsize=11)

out_fn = base_fp + 'future_fires_albedo_mb.png'
fig.savefig(out_fn, dpi=200)
print(f'Saved {out_fn}', flush=True)
