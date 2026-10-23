"""
Plot the outlines of the study glaciers side by side at the same scale.
"""
import geopandas as gpd
import matplotlib.pyplot as plt
from pyproj import CRS

rgi_fp = '/ocean/projects/ees260009p/cwilson4/RGI/rgi60/01_rgi60_Alaska/01_rgi60_Alaska.shp'
fig_fp = '/ocean/projects/ees260009p/cwilson4/Output/glacier_outlines.png'
glaciers = {'01.00570':'Gulkana',
            '01.09162':'Wolverine',
            '01.01104':'Lemon Creek',
            '01.15645':'Kennicott',
            '01.22193':'Kahiltna'}
pad = 1000 # padding around each outline [m]
scalebar = 10000 # [m]

# ===================== LOAD OUTLINES =====================
gdf = gpd.read_file(rgi_fp)
outlines = []
for rgi_id in glaciers:
    glacier = gdf[gdf['RGIId'] == 'RGI60-' + rgi_id]
    # local LAEA centered on each glacier so all are in meters with minimal distortion
    clon = float(glacier['CenLon'].iloc[0])
    clat = float(glacier['CenLat'].iloc[0])
    metric_crs = CRS(f'+proj=laea +lat_0={clat:.2f} +lon_0={clon:.2f} +datum=WGS84 +units=m')
    outlines.append(glacier.to_crs(metric_crs))

# ===================== PLOT =====================
# axis widths are proportional to glacier x extent and all share the tallest y extent,
# so with equal aspect every panel has the same meters per inch
bounds = [g.total_bounds for g in outlines]
widths = [b[2] - b[0] + 2*pad for b in bounds]
height = max(b[3] - b[1] for b in bounds) + 2*pad

fig, axes = plt.subplots(1, len(outlines), figsize=(12, 6), width_ratios=widths)
for ax, glacier, b, rgi_id in zip(axes, outlines, bounds, glaciers):
    glacier.plot(ax=ax, facecolor='lightsteelblue', edgecolor='k', linewidth=0.6)
    cy = (b[1] + b[3]) / 2
    ax.set_xlim(b[0] - pad, b[2] + pad)
    ax.set_ylim(cy - height/2, cy + height/2)
    ax.set_aspect('equal', anchor='C')
    ax.axis('off')
    area = glacier['Area'].iloc[0]
    ax.set_title(f'{glaciers[rgi_id]}\n{area:.0f} km$^2$', fontsize=10)

# scale bar in the empty lower left of the Kennicott panel
ax = axes[3]
x0 = ax.get_xlim()[0] + pad
y0 = ax.get_ylim()[0] + pad
ax.plot([x0, x0 + scalebar], [y0, y0], color='k', linewidth=2)
ax.text(x0 + scalebar/2, y0 + pad/2, f'{scalebar/1000:.0f} km', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(fig_fp, dpi=200)
print(f'saved to {fig_fp}')
