"""
Maps the mean flux divergence, surface mass balance and net mass balance
over the last n_years of run_dynamics_demo.py on the simulation's Voronoi
cells, so GLIDE has had the earlier years to adjust to its starting geometry.
Flux divergence is the dynamic contribution to net mass balance
(negative in the accumulation area, positive in the ablation area),
so net mass balance = SMB + flux divergence.
"""
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
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

# ===================== VORONOI CELLS =====================
clon = float(np.mean(lons))
clat = float(np.mean(lats))
metric_crs = CRS(f'+proj=laea +lat_0={clat:.2f} +lon_0={clon:.2f} +datum=WGS84 +units=m')
to_metric = Transformer.from_crs('EPSG:4326', metric_crs, always_xy=True)
xs, ys = to_metric.transform(lons, lats)

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
          ('smb', 'Surface mass balance'),
          ('net', 'Net mass balance')]
lim = np.nanpercentile(np.abs(np.concatenate([flux_div, smb, net])), 98)
norm = TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim)

plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10})
fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.2, wspace=0.04)
for ax, (var, title) in zip(axes, panels):
    cells.plot(column=var, ax=ax, cmap='RdBu', norm=norm, edgecolor='0.55', linewidth=0.15)
    glacier.plot(ax=ax, facecolor='none', edgecolor='0.15', linewidth=0.6)
    glacier_mean = np.average(cells[var], weights=cells.area)
    ax.set_title(title) # , loc='left')
    # ax.text(0, 1.0, f'mean {glacier_mean:+.2f}', transform=ax.transAxes,
    #         ha='left', va='top', color='0.35')
    ax.set_aspect('equal')

cax = fig.add_axes([0.37, 0.02, 0.26, 0.03])
cbar = fig.colorbar(axes[0].collections[0], cax=cax, orientation='horizontal', extend='both')
cbar.set_label('Mass balance (m w.e. yr$^{-1}$)')
cbar.ax.tick_params(length=2)

plt.savefig(f'{output_fp}dynamics_demo_map.png', dpi=300, bbox_inches='tight')
print(f'saved to {output_fp}dynamics_demo_map.png')
