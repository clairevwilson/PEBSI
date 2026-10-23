"""
Plots the five calibration glaciers on a simple outline map of Alaska,
each placed at its RGI centroid.
"""
import os
import sys

import geopandas as gpd

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))
from project.parameters import translate_rgi, GLACIERS

rgi_fp = '/ocean/projects/ees260009p/cwilson4/RGI/rgi60/01_rgi60_Alaska/01_rgi60_Alaska.shp'
fig_fn = '/ocean/projects/ees260009p/cwilson4/Output/glacier_map.png'

rgi = gpd.read_file(rgi_fp, ignore_geometry=True)
rgi = rgi.set_index('RGIId')

proj = ccrs.AlbersEqualArea(central_longitude=-150, central_latitude=60,
                            standard_parallels=(55, 65))
fig, ax = plt.subplots(figsize=(6, 5), subplot_kw={'projection': proj})
ax.set_extent([-170, -129, 54, 71.5], crs=ccrs.PlateCarree())
ax.add_feature(cfeature.COASTLINE.with_scale('50m'), lw=0.6)
ax.add_feature(cfeature.BORDERS.with_scale('50m'), lw=0.6)

# labels that would run into the coast or the frame go on the left
label_left = ['wolverine', 'lemon_creek']

for glacier in GLACIERS:
    row = rgi.loc[f'RGI60-{translate_rgi[glacier]["6"]}']
    ax.plot(row.CenLon, row.CenLat, marker='o', ms=6, color='C3',
            mec='k', mew=0.5, transform=ccrs.PlateCarree())
    left = glacier in label_left
    ax.text(row.CenLon + (-0.8 if left else 0.8), row.CenLat,
            glacier.replace('_', ' ').title(), fontsize=9, va='center',
            ha='right' if left else 'left', transform=ccrs.PlateCarree(),
            bbox=dict(fc='white', ec='none', alpha=0.8, pad=1))

fig.savefig(fig_fn, dpi=200, bbox_inches='tight')
print(f'Wrote {fig_fn}')
