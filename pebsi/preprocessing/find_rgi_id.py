import pandas as pd
import geopandas as gpd
import argparse
import os
from shapely.geometry import Point
import matplotlib.pyplot as plt
import matplotlib as mpl

dcolors = ['#63c4c7','#fcc02e','#4D559C','#60C252','#BF1F6A',
              '#F77808','#298282','#999999','#FF89B0','#427801']

parser = argparse.ArgumentParser()
parser.add_argument('-reg','--region',action='store',
                    default='00',type=str,
                    help='RGI region the glacier is in')
parser.add_argument('-t', '--text', action='store', 
                    default='',type=str,
                    help='Text to search for in glacier names')
parser.add_argument('-lat','--latitude',action='store',
                    default=0, type=float,
                    help='Latitude near glacier')
parser.add_argument('-lon','--longitude',action='store',
                    default=0, type=float,
                    help='Longitude near glacier')
parser.add_argument('-p','--plot',action='store_true',
                    default=False, help='Plot shapefiles?')
parser.add_argument('-sr','--search_radius',action='store',
                    default=10, type=float,
                    help='Radius to search around the provided lat/lon [km]')
parser.add_argument('--plot_fn',action='store',
                    default='test.png',type=str,
                    help='Filename relative to this folder for the plot')
parser.add_argument('-hr','--help_regions',action='store_true',
                    help='Print all the region names for help identifying your region')
args = parser.parse_args()

assert args.region != '00', 'Define what region to check with -reg (guess if you have no idea)'
if len(args.region) < 2:
    args.region = '0' + args.region

# Open attributes file
RGI_fp = '../../../RGI/rgi60/00_rgi60_attribs/'
for fn in os.listdir(RGI_fp):
    if args.help_regions:
        print(fn.split('.')[0])
    if args.region in fn and 'csv' in fn:
        f = fn.split('.')[0]
        RGI_fn = f + '.csv'
        shapefile_fp = '../../../RGI/rgi60/' + f + '/'
        shapefile_fn = f + '.shp'
RGI_df = pd.read_csv(RGI_fp + RGI_fn, index_col=0)

if len(args.text) > 0:
    print(f'Searching region {args.region} attributes for {args.text}')
    match = RGI_df[RGI_df['Name'].str.contains(args.text, case=False, na=False)]
    n_matches = len(match.index)
    if n_matches > 0:
        print(f'Found {n_matches} entries:')
        print(match)
    else:
        print(f'Found 0 matches in region {args.region}')

if args.latitude or args.longitude:
    print(f'Searching region {args.region} for glaciers within {args.search_radius} km of {args.latitude}, {args.longitude}')

    RGI_gdf = gpd.GeoDataFrame(
        RGI_df,
        geometry=gpd.points_from_xy(RGI_df['CenLon'], RGI_df['CenLat']),
        crs="EPSG:4326"  # WGS84 (latitude/longitude)
    )

    # Convert to a projected CRS (meters) for accurate distance calculations
    RGI_gdf = RGI_gdf.to_crs(epsg=3857)

    # Reference point as geometry
    ref_point = gpd.GeoSeries([Point(args.longitude, args.latitude)], crs="EPSG:4326").to_crs(epsg=3857).iloc[0]

    # Calculate distances in meters
    RGI_gdf['distance_m'] = RGI_gdf.geometry.distance(ref_point)
    # print(RGI_gdf)

    # Filter for entries within the distance (in m)
    RGI_df['distance_m'] = RGI_gdf['distance_m']
    result = RGI_df[RGI_gdf['distance_m'] <= args.search_radius * 1000].sort_values(by='distance_m')
    n_results = len(result.index)
    if n_results > 5:
        print(f'Found {n_results} glaciers, listing the five nearest:')
        print(result.head(5))
    elif n_results > 0:
        print(f'Found {n_results} glacier(s):')
        print(result)
    else:
        print(f'Found 0 glaciers within {args.search_radius} km, quitting...')
        quit()

if args.plot:
    shapefile = gpd.read_file(shapefile_fp + shapefile_fn)
    shapefile.index = [x.split('.')[-1] for x in shapefile['RGIId']]
    list_IDs = []
    sorted_IDs = []
    if len(args.text) > 0:
        list_IDs += [x.split('.')[-1]  for x in match.index]
    if args.latitude and args.longitude:
        sorted_IDs += [x.split('.')[-1]  for x in result.index]
    
    # Make plot
    fig, ax = plt.subplots()
    ax.grid(True, zorder=0)
    

    if len(list_IDs) > 0:
        cmap = plt.get_cmap('Set3')
        if len(list_IDs) > 10:
            print('Too many results to include them all in the plot legend: search more specific')
            shape = shapefile.loc[list_IDs[:10]]
            shape.plot('RGIId', ax=ax, zorder=2, legend=True,cmap=cmap)
    
            shape = shapefile.loc[list_IDs[10:]]
            shape.plot(ax=ax, zorder=3,color='gray')
        else:
            shape = shapefile.loc[list_IDs]
            shape.plot('RGIId', ax=ax, zorder=3, legend=True, cmap=cmap)
    
    if len(sorted_IDs) > 0:
        cmap = plt.get_cmap('magma')
        first_five = shapefile.loc[sorted_IDs[:5]]
        first_five['rank'] = [0,1,2,3,4]
        first_five.plot('RGIId', ax=ax, cmap=cmap, legend=True,zorder=2)
        for i, id in enumerate(sorted_IDs):
            norm = mpl.colors.Normalize(vmin=0, vmax=5)
            shape = shapefile.loc[[id]]
            shape.plot(ax=ax, color=cmap(norm(i)),zorder=3)

        ref_point = gpd.GeoSeries([Point(args.longitude, args.latitude)], crs="EPSG:4326")
        ref_point.plot(ax=ax, color='red', marker='+', markersize=200,zorder=5)  
        cbar_ax = fig.add_axes([0.89, 0.10, 0.02, 0.77])
        bins = [0,1,2,3,4,5]
        cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                    cax=cbar_ax, ticks=bins,
                    orientation='vertical',
                    spacing='proportional')
        cb.ax.tick_params(labelsize=10,length=5)
        cb.ax.set_ylabel('Ranked distance from point (0 = closest)')
    ax.set_xlabel('Longitude ($^{\circ}$)')
    ax.set_ylabel('Latitude ($^{\circ}$)')
    plt.savefig(args.plot_fn,bbox_inches='tight')
    print(f'Saved figure to {args.plot_fn}')