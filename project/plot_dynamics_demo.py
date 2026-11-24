"""
Maps the mean flux divergence, surface mass balance and net mass balance
over the last n_years of run_dynamics_demo.py on the simulation's Voronoi
cells, so GLIDE has had the earlier years to adjust to its starting geometry.
Flux divergence is the dynamic contribution to net mass balance
(negative in the accumulation area, positive in the ablation area),
so net mass balance = SMB + flux divergence.
"""
import yaml
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from pyproj import CRS, Transformer

output_fp = '/ocean/projects/ees260009p/cwilson4/Output/dynamics_demo_2/'
rgi_fp = '/ocean/projects/ees260009p/cwilson4/RGI/rgi60/01_rgi60_Alaska/01_rgi60_Alaska.shp'
rgi_id = '01.00570'
n_years = 5

# ===================== LOAD =====================
data = np.load(f'{output_fp}dynamics_demo.npz')
dt = data['dt'][-n_years:, None]
smb = (data['smb'][-n_years:] / dt).mean(axis=0)
flux_div = (data['dyn'][-n_years:] / dt).mean(axis=0)
net = smb + flux_div
lats = data['lat']
lons = data['lon']
with open(f'{output_fp}config.yaml') as f:
    end_year = int(str(yaml.safe_load(f)['end_date'])[:4])

# ===================== VORONOI CELLS =====================
clon = float(np.mean(lons))
clat = float(np.mean(lats))
metric_crs = CRS(f'+proj=laea +lat_0={clat:.2f} +lon_0={clon:.2f} +datum=WGS84 +units=m')
to_metric = Transformer.from_crs('EPSG:4326', metric_crs, always_xy=True)
xs, ys = to_metric.transform(lons, lats)

# two lon/lat ticks per axis; meridians and parallels are effectively straight at this scale
lon_ticks = [-145.47, -145.37]
lat_ticks = [63.26, 63.30]
x_ticks, _ = to_metric.transform(lon_ticks, [clat] * 2)
_, y_ticks = to_metric.transform([clon] * 2, lat_ticks)

gdf = gpd.read_file(rgi_fp)
glacier = gdf[gdf['RGIId'] == 'RGI60-' + rgi_id].to_crs(metric_crs)
outline = glacier.geometry.union_all()
pts = gpd.GeoDataFrame({'flux_div': flux_div, 'smb': smb, 'net': net},
                       geometry=gpd.points_from_xy(xs, ys), crs=metric_crs)
cells = gpd.GeoDataFrame(geometry=pts.geometry.voronoi_polygons(extend_to=outline), crs=metric_crs)
cells = gpd.sjoin(cells, pts, predicate='contains').drop(columns='index_right')
cells['geometry'] = cells.geometry.intersection(outline)

# ===================== PLOT =====================
panels = [('flux_div', 'Flux divergence'),
          ('smb', 'Climatic mass balance'),
          ('net', 'Net mass balance')]
lim = np.nanpercentile(np.abs(np.concatenate([flux_div, smb, net])), 98)
norm = TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim)

plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10})
fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.52), sharex=True, sharey=True)
fig.subplots_adjust(left=0.1, right=0.86, top=0.75, bottom=0.129, wspace=0.04)
glacier_wide = np.average(cells['smb'], weights=cells.area)
fig.suptitle(f'Gulkana Glacier, {end_year - n_years}–{end_year}\n'
             f'(glacier-wide mass balance: {glacier_wide:.2f} m w.e. yr$^{{-1}}$)')
for ax, (var, title) in zip(axes, panels):
    cells.plot(column=var, ax=ax, cmap='RdBu', norm=norm, edgecolor='0.55', linewidth=0.15)
    glacier.plot(ax=ax, facecolor='none', edgecolor='0.15', linewidth=0.6)
    glacier_mean = np.average(cells[var], weights=cells.area)
    ax.set_title(title) # , loc='left')
    # ax.text(0, 1.0, f'mean {glacier_mean:+.2f}', transform=ax.transAxes,
    #         ha='left', va='top', color='0.35')
    ax.set_aspect('equal')
    ax.set_xticks(x_ticks, [f'{-v:.2f}°W' for v in lon_ticks])
    ax.set_yticks(y_ticks, [f'{v:.2f}°N' for v in lat_ticks])

# panels share one scale, so a single bar covers all three
axes[0].add_artist(AnchoredSizeBar(axes[0].transData, 2000, '2 km', loc='lower right',
                                   pad=0.4, borderpad=0.6, sep=3, frameon=False,
                                   size_vertical=60))

# equal aspect shrinks the panels, so size the colorbar to where they actually draw
axes[-1].apply_aspect()
pos = axes[-1].get_position()
cax = fig.add_axes([pos.x1 + 0.015, pos.y0, 0.015, pos.height])
cbar = fig.colorbar(axes[0].collections[0], cax=cax, orientation='vertical', extend='both')
cbar.set_label('Mass balance\n(m w.e. yr$^{-1}$)')
cbar.ax.tick_params(length=2)

plt.savefig(f'{output_fp}dynamics_demo_map.png', dpi=300, bbox_inches='tight')
print(f'saved to {output_fp}dynamics_demo_map.png')
