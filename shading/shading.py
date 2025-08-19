"""
Created on Tue Mar 19 11:32:50 2024

Shading model for PEBSI

Requirements: - DEM which contains glacier and surrounding ridges
              - Coordinates for point to perform calculations

1. Input site coordinates and time zone
2. Load DEM and calculate slope/aspect
3. Determine horizon angles
        Optional: plot horizon search
4. Calculate sky-view factor
5. Calculate direct clear-sky slope-corrected irradiance and
   shading for each hour of the year
6. Store shade .csv
        Optional: plot results

@author: clairevwilson
"""
# Built-in libraries
import os
import argparse
# External libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import rioxarray as rxr
import xarray as xr
import pandas as pd
import geopandas as gpd
import suncalc
from pyproj import Transformer
from numpy import pi, cos, sin, arctan

# =================== INPUTS ===================
# this section is unused if running this from run_simulation_eb
site_by = 'latlon'                  # method to choose lat/lon ('id' or 'latlon')
site = 'AWS'                        # name of site for indexing .csv OR specify lat/lon
lat,lon = [0.0178817,-78.0040317]   # [60.8155986,-139.1236350] # site latitude,longitude 
timezone = pd.Timedelta(hours=-5)   # time zone of location
glacier_name = 'cayambe'            # name of glacier for labeling
glac_no = '01.16195'                # RGI glacier ID

# storage options
plot = ['result','search','horizon']           # list from ['result','search','horizon']
store = ['result','result_plot','search_plot','horizon_plot']   # list from ['result','result_plot','search_plot','horizon_plot']
result_vars = ['dirirrslope','shaded']      # variables to include in plot

# model options 
get_shade = True    # run shade model?
get_direct = False  # run slope-corrected irradiance model? (unused)
assert get_shade or get_direct, 'Why are you running this?'

# model parameters
time_freq = '30min'     # timestep in offset alias notation
angle_step = 5          # step to calculate horizon angle (degrees)
search_length = 5000    # distance to search from center point (m)
sub_dt = 10             # timestep to calculate solar corrections (minutes)
buffer = 20             # min num of gridcells away from which horizon can be found

# ================ INPUT FILEPATHS =================
# input filepath
dem_fp = f'../../Data/dems/{glacier_name}_dem.tif'   # DEM containing glacier + surroundings # gulkana/Gulkana_DEM_20m
# RGI for glacier shapefile
rgi_fp = f'../../RGI/rgi60/'
# output filepath
data_fp = os.getcwd().split('PyGEM-EB')[0]
fp_base = data_fp + 'PyGEM-EB/shading/'
fp_out = fp_base + f'../data/by_glacier/'
# optional: site constants file
site_fp = fp_out + 'GLACIER/site_constants.csv'

# =================== CONSTANTS ===================
I0 = 1368       # solar constant in W m-2
P0 = 101325     # sea-level pressure in Pa
PSI = 0.75      # vertical atmospheric clear-sky transmissivity
MEAN_RAD = 1    # mean earth-sun radius in AU

# =================== VISUALIZATION ===================
varprops = {'dirirr':{'label':'direct flat-surface irradiance [W m-2]','cmap':'plasma'},
            'dirirrslope':{'label':'direct slope-corrected irradiance [W m-2]','cmap':'plasma'},
            'shaded':{'label':'shading [black = shaded]','cmap':'binary'},
            'sun_elev':{'label':'sun elevation angle [$\circ$]','cmap':'Spectral_r'},
            'horizon_elev':{'label':'horizon elevation angle [[$\circ$]','cmap':'YlOrRd'},
            'sun_az':{'label':'sun azimuth angle [$\circ$]','cmap':'twilight'}}

class Shading():
    """
    Shading model which finds the horizon angle at every
    azimuth and determines whether a given point is in 
    the sun or shade for every hour (or user-defined dt)
    of the year.
    """
    def __init__(self,args=None):
        """
        Initializes shading model by opening the DEM and 
        calculating slope and aspect.
        If this model is executed frpm PyGEM-EB, args is
        filled in run_simulation_eb.py with the needed inputs.
        """
        # parse command line args if they were not input
        if not args:
            args = self.parse_args()

        # define site filepath and make the folder if it doesn't exist
        args.site_fp = site_fp.replace('GLACIER',args.glac_name)
        if not os.path.exists(fp_out + args.glac_name):
            os.mkdir(fp_out + args.glac_name)

        # open or create the site constants file
        if os.path.exists(args.site_fp):
            # get site lat and lon    
            site_df = pd.read_csv(args.site_fp,index_col=0)
            self.site_df = site_df
            if args.site_by == 'id':
                args.lat = site_df.loc[args.site,'lat']    # latitude of point of interest
                args.lon = site_df.loc[args.site,'lon']    # longitude of point of interest
        else:
            self.site_df = pd.DataFrame({'lat':lat,'lon':lon},index=[args.site])
            self.site_df.index.name = 'site'

        # save the args
        self.args = args

        # check if the DEM exists; if not, print out the bounding box to manually retrieve one
        self.get_shapefile()
        if not os.path.exists(args.dem_fp):
            minx, miny, maxx, maxy = self.shapefile.total_bounds
            dem_not_found = 'DEM was not found: download for the box around:'
            bounding_box = f'      latitude: {miny:.6f} to {maxy:.6f}    longitude: {minx:.6f} to {maxx:.6f}'
            move_to = f'                and move to {args.dem_fp}'
            assert os.path.exists(args.dem_fp), f'{dem_not_found}\n{bounding_box}\n{move_to}'

        # Load the DEM
        self.load_dem()

        # Define output filepaths and make any directories
        gn = args.glac_name
        if not os.path.exists(fp_base + 'plots/'):
            os.mkdir(fp_base + 'plots/')
        if not os.path.exists(fp_base + f'plots/{gn}'):
            os.mkdir(fp_base + f'plots/{gn}')
        if not os.path.exists(fp_out + f'{gn}/shade/'):
            os.mkdir(fp_out + f'{gn}/shade/')
        self.shade_fp = fp_out + f'{gn}/shade/{gn}{args.site}_shade.csv'
        self.irr_fp = fp_out + f'{gn}/shade/{gn}{args.site}_irr.csv'
        self.out_image_fp = fp_base + f'plots/{gn}/{gn}{args.site}.png'
        self.out_horizon_fp = fp_base + f'plots/{gn}/{gn}{args.site}_horizon.png'
        self.out_search_fp = fp_base + f'plots/{gn}/{gn}{args.site}_angles.png'
        return

    def parse_args(self):
        # =================== PARSE ARGS ===================
        parser = argparse.ArgumentParser(description='pygem-eb shading model')
        parser.add_argument('-latitude','--lat',action='store',default=lat)
        parser.add_argument('-longitude','--lon',action='store',default=lon)
        parser.add_argument('-glac_no',action='store',default=glac_no)
        parser.add_argument('-dem_fp',action='store',default=dem_fp)
        parser.add_argument('-site',action='store',default=site)
        parser.add_argument('-timezone',action='store',default=timezone)
        parser.add_argument('-glac_name',action='store',default=glacier_name)
        parser.add_argument('-site_by',action='store',default=site_by)
        parser.add_argument('-plot',action='store',default=plot)
        parser.add_argument('-store',action='store',default=store)
        args = parser.parse_args()
        args.glac_name += args.site
        return args
    
    def main(self):
        """
        Executes functions to find the horizon, 
        irradiance and shade along with plots.
        """
        args = self.args

        # find the horizon in every direction
        self.find_horizon()
        # horizon angle search
        if 'search' in args.plot:
            self.plot_search()
        # horizon angle vs azimuth
        if 'horizon' in args.plot:
            self.plot_horizon()

        # get hourly dataframe of solar irradiance
        df = self.irradiance(time_freq)
        # pcolormesh plot of results
        if 'result' in args.plot:
            self.plot_result(df)

        # store results
        if 'result' in args.store:
            if get_shade:
                df['shaded'].astype(bool).to_csv(self.shade_fp,
                                                header=f'skyview={self.sky_view}')
            if get_direct:
                df['dirirrslope'].astype(float).to_csv(self.irr_fp,
                                                header=f'skyview={self.sky_view}')
        return

    # =================== FUNCTIONS ===================
    def r_sun(self,time):
        """Gets earth-to-sun radius in AU"""
        doy = time.day_of_year
        radius = 1 - 0.01672*cos(0.9856*(doy-4))
        return radius

    def pressure(self,elev):
        """Adjusts air pressure by elevation"""
        P = np.exp(-0.0001184*elev)*P0
        return P

    def zenith(self,time):
        """Calculates solar zenith angle for time, lat and lon"""
        time_UTC = time - self.args.timezone
        lon = self.args.lon
        lat = self.args.lat
        altitude_angle = suncalc.get_position(time_UTC,lon,lat)['altitude']
        zenith = pi/2 - altitude_angle if altitude_angle > 0 else np.nan
        return zenith

    def declination(self,time):
        """Calculates solar declination"""
        doy = time.day_of_year
        delta = -23.4*cos(360*(doy+10)/365) * pi/180
        return delta

    def hour_angle(self,time):
        hour_angle = 15*(12-time.hour)
        return hour_angle

    def select_coordinates(self,angle,length):
        """Creates a line of points from the starting cell
        to select grid cells in a given direction (angle in 
        deg 0-360 where 0 is North)"""
        # get starting coordinates
        start_x = self.xx
        start_y = self.yy
        step_size = self.x_res

        # convert angle to radians and make 0 north
        rad = angle * pi/180 + pi/2

        # get change in x and y for each step
        dx = - step_size * cos(rad) # negative so it travels clockwise
        dy = step_size * sin(rad)

        # define end
        n_steps = np.ceil(length / step_size).astype(int)
        end_x = start_x + dx*n_steps
        end_y = start_y + dy*n_steps
        
        # create lines
        xs = np.linspace(start_x,end_x,n_steps)
        ys = np.linspace(start_y,end_y,n_steps)
        if xs.shape > ys.shape:
            ys = np.ones(n_steps) * start_y
        elif ys.shape > xs.shape:
            xs = np.ones(n_steps) * start_x
        
        return xs,ys
    
    def get_shapefile(self):
        # find the shapefile in the RGI from the glac_no
        id = self.args.glac_no
        region = id.split('.')[0]
        # search RGI folders for the correct shapefile
        for folder in os.listdir(fp_base + rgi_fp):
            if folder[:2] == region:
                for f in os.listdir(fp_base + rgi_fp + folder):
                    if f[-3:] == 'shp':
                        fn = folder + '/' + f
        # open and index regional shapefile to the glacier
        all_gdf = gpd.read_file(fp_base + rgi_fp + fn).set_crs(epsg=4326)
        shapefile = all_gdf.loc[all_gdf['RGIId'] == 'RGI60-'+id]
        self.shapefile = shapefile

    def get_utm_epsg(self, lat, lon):
        if not -80.0 <= lat <= 84.0:
            raise ValueError('UTM zones are only defined between 84°N and 80°S')
        zone_number = int((lon + 180) / 6) + 1
        if lat >= 0:
            epsg_code = 32600 + zone_number  # Northern hemisphere
        else:
            epsg_code = 32700 + zone_number  # Southern hemisphere
        return epsg_code
    
    def load_dem(self):
        """
        Loads the DEM and calculates aspect and slope from it.
        """
        # open files
        dem_fp = self.args.dem_fp
        dem = rxr.open_rasterio(dem_fp,masked=True).isel(band=0)
        
        # convert DEM to appropriate UTM coordinate system
        epsg = self.get_utm_epsg(self.args.lat, self.args.lon)
        dem = dem.rio.reproject(f'EPSG:{epsg}',resolution=30)

        # determine resolution in meters
        self.x_res = dem.rio.resolution()[0]
        self.y_res = dem.rio.resolution()[1]

        # ensure consistent coordinates for the shapefile
        self.shapefile = self.shapefile.to_crs(dem.rio.crs)

        # filter below sea level
        dem = dem.where(dem > 0)
        # filter extremes
        dem = dem.where(dem < 6000)

        # calculate gradient from DEM
        dx,dy = np.gradient(dem,self.y_res,self.x_res)
        # calculate slope and aspect from gradient
        slope = np.arctan(np.sqrt(dx**2 + dy**2))
        aspect = np.arctan2(-dy,-dx)
        # adjust so 0 is North
        aspect = (aspect + 2*np.pi) % (2*np.pi) 
        # store in a DataArray
        slope = xr.DataArray(slope, dims=['y', 'x'], coords={'y': dem.y, 'x': dem.x})
        aspect = xr.DataArray(aspect, dims=['y', 'x'], coords={'y': dem.y, 'x': dem.x})

        # get min/max elevation for plotting
        self.min_elev = int(np.round(np.nanmin(dem.values)/100,0)*100)
        self.max_elev = int(np.round(np.nanmax(dem.values)/100,0)*100)

        # get UTM coordinates from lat/lon
        transformer = Transformer.from_crs('EPSG:4326', dem.rio.crs, always_xy=True)
        xx, yy = transformer.transform(self.args.lon, self.args.lat)
        # check point is in bounds
        bounds = np.array(dem.rio.bounds())
        x_in = xx >= bounds[0] and xx <= bounds[2]
        y_in = yy >= bounds[1] and yy <= bounds[3]
        assert x_in and y_in,'point out of raster bounds'
        self.xx = xx
        self.yy = yy

        # get elevation of point from grid
        self.point_elev = dem.sel(x=xx, y=yy, method='nearest').values

        # get slope and aspect at point of interest
        point_aspect = aspect.sel(x=xx, y=yy, method='nearest').values
        point_slope = slope.sel(x=xx, y=yy, method='nearest').values
        print(f'~ Point stats at {self.args.glac_name} {self.args.site}:')
        print(f'        elevation: {self.point_elev:.0f} m a.s.l.')
        print(f'        aspect: {point_aspect*180/pi:.1f} o')
        print(f'        slope: {point_slope*180/pi:.2f} o')

        # out
        self.dem = dem
        self.slope = slope
        self.aspect = aspect
        self.point_aspect = point_aspect
        self.point_slope = point_slope
        return

    def find_horizon_point(self,elev,xs,ys):
        """Finds the horizon along a line of elevation
        values paired to x,y coordinates
        - elev: array of elevations in a single direction
        - xs, ys: coordinates corresponding to elev"""
        # calculate distance from origin and height relative to origin
        distances = np.sqrt((xs-self.xx)**2+(ys-self.yy)**2)
        distances[np.where(distances < 1e-3)[0]] = 1e-6
        heights = elev - self.point_elev
        heights[np.where(heights < 0)[0]] = 0

        # identify maximum horizon elevation angle
        elev_angles = arctan(heights/distances)
        idx = np.argmax(elev_angles[buffer:]) + buffer

        # index out information about the horizon point
        horizon_angle = elev_angles[idx]
        horizon_x = xs[idx]
        horizon_y = ys[idx]
        return horizon_angle,horizon_x,horizon_y,elev_angles

    def find_horizon(self):
        args = self.args
        # plot DEM as background
        if 'search' in args.plot:
            fig,ax = plt.subplots(figsize=(6,6))
            self.dem.plot(ax=ax,cmap='viridis')
            self.shapefile.plot(ax=ax,color='none',edgecolor='black',linewidth=1.5)
            plt.axis('equal')

        # loop through angles
        self.angles = np.arange(0,360,angle_step)
        horizons = {}
        for ang in self.angles:
            # set up dict to store
            horizons[ang] = {'horizon_elev':[],'hz_x':[],'hz_y':[]}
            
            # get line in the direction of choice
            xs, ys = self.select_coordinates(ang,search_length)
            
            # select elevation gridcells along the line
            x_select = xr.DataArray(xs,dims=['location'])
            y_select = xr.DataArray(ys,dims=['location'])
            elev = self.dem.sel(x=x_select,y=y_select,method='nearest').values

            # filter out nans
            xs = xs[~np.isnan(elev)]
            ys = ys[~np.isnan(elev)]
            elev = elev[~np.isnan(elev)]
            
            # find the horizon
            hz_ang,hz_x,hz_y,all_angles = self.find_horizon_point(elev,xs,ys)
            horizons[ang]['horizon_elev'] = hz_ang
            horizons[ang]['hz_x'] = hz_x
            horizons[ang]['hz_y'] = hz_y

            # visualize elevations 
            if 'search' in args.plot:
                norm = mpl.colors.Normalize(vmin=self.min_elev,vmax=self.max_elev)
                cmap = plt.cm.viridis
                scalar_map = mpl.cm.ScalarMappable(norm=norm,cmap=cmap)
                colors = scalar_map.to_rgba(elev)
                plt.scatter(xs,ys,color=colors,s=1,marker='.',alpha=0.7)
                plt.scatter(hz_x,hz_y,color='red',marker='x',s=50)

        # plot shapefile over everything
        if 'search' in args.plot:
            self.shapefile.plot(ax=ax,color='none',edgecolor='black',linewidth=1.5)

        # calculate sky-view factor
        horizon_elev = np.array([horizons[ang]['horizon_elev'] for ang in self.angles])
        angle_step_rad = angle_step * pi/180
        sky_view = np.sum(cos(horizon_elev)**2 * angle_step_rad) / (2*pi)
        
        # out
        self.sky_view = sky_view
        self.horizon_elev = horizon_elev
        return

    def irradiance(self,time_freq='30min'):
        """
        Calculates potential clear-sky slope-corrected irradiance in W m-2
        (Icorr) at the input time frequency. This is the maximum amount of 
        direct sunlight the surface can receive. The effect of aspect and
        shading are included (Icorr = 0 when the point is in the shade.)
        """
        args = self.args

        # loop through hours of the year and store data
        store_vars = ['dirirr','dirirrslope','shaded','corr_factor','sun_elev','horizon_elev','sun_az']
        year_hours = pd.date_range('2024-01-01 00:00','2024-12-31 23:55',freq=time_freq)
        timestep_min = (year_hours[1] - year_hours[0]).total_seconds()/60
        df = pd.DataFrame(data = np.ones((len(year_hours),len(store_vars))),
                        columns=store_vars,index=year_hours)
        for time in year_hours:
            # loop to get sub-hourly values and average
            sub_vars = ['shaded','Icorr','I','corr_factor','zenith']
            period_dict = {}
            for var in sub_vars:
                period_dict[var] = np.array([])
            for minutes in np.arange(0,timestep_min,sub_dt):
                # calculate time-dependent variables
                time_UTC = time - args.timezone + pd.Timedelta(minutes = minutes)
                P = self.pressure(self.point_elev)
                r = self.r_sun(time + pd.Timedelta(minutes = minutes))
                Z = self.zenith(time + pd.Timedelta(minutes = minutes))
                d = self.declination(time + pd.Timedelta(minutes = minutes))
                h = self.hour_angle(time + pd.Timedelta(minutes = minutes))
                period_dict['zenith'] = np.append(period_dict['zenith'],Z)

                # calculate direct clear-sky irradiance (not slope corrected)
                I = I0 * (MEAN_RAD/r)**2 * PSI**(P/P0/np.cos(Z)) * np.cos(Z)
                period_dict['I'] = np.append(period_dict['I'],I)

                # get sun elevation and azimuth angle
                sunpos = suncalc.get_position(time_UTC,args.lon,args.lat)
                sun_elev = sunpos['altitude']       # solar elevation angle
                # suncalc gives azimuth with 0 = South, we want 0 = North
                sun_az = sunpos['azimuth'] + pi     # solar azimuth angle

                # get nearest angle of horizon calculations to the sun azimuth
                idx = np.argmin(np.abs(self.angles*pi/180 - sun_az))
                
                # check if the sun elevation angle is below the horizon angle
                shaded = 1 if sun_elev < self.horizon_elev[idx] else 0
                period_dict['shaded'] = np.append(period_dict['shaded'],shaded)

                # incident angle calculation
                cosTHETA = cos(self.point_slope)*cos(Z) + sin(self.point_slope)*sin(Z)*cos(sun_az - self.point_aspect)
                Icorr = I * min(cosTHETA/cos(Z),5) * (shaded-1)*-1
                period_dict['corr_factor'] = np.append(period_dict['corr_factor'],min(cosTHETA/cos(Z),5))
                period_dict['Icorr'] = np.append(period_dict['Icorr'],Icorr)

            # extract sub-hourly-timestep arrays
            I = period_dict['I'][~np.isnan(period_dict['I'])]
            Icorr = period_dict['Icorr']
            cosZ = cos(period_dict['zenith'])
            corrf = period_dict['corr_factor']
            dt = np.ones(len(Icorr)) * sub_dt

            # find hourly means (avoid nans)
            if ~np.any(np.isnan(Icorr)):
                mean_I = np.sum(Icorr*cosZ*corrf*dt) / np.sum(dt)
                if np.sum(Icorr*cosZ) > 0:
                    mean_corr_factor = np.sum(Icorr*cosZ*corrf) / np.sum(Icorr*cosZ)
                else:
                    mean_corr_factor = 5
            else:
                mean_I = 0
                mean_corr_factor = 0
            median_shaded = int(np.median(period_dict['shaded']))
            
            # store data
            df.loc[time,'shaded'] = median_shaded
            df.loc[time,'dirirrslope'] = mean_I
            df.loc[time,'corr_factor'] = mean_corr_factor
            if len(I) > 0:
                df.loc[time,'dirirr'] = np.mean(I)
            else:
                df.loc[time,'dirirr'] = np.nan

            # unnecessary, just for plotting
            df.loc[time,'sun_az'] = sun_az * 180/pi
            df.loc[time,'horizon_elev'] = self.horizon_elev[idx] * 180/pi
            df.loc[time,'sun_elev'] = sun_elev * 180/pi
            if median_shaded:
                df.loc[time,'horizon_elev'] = np.nan
        
        return df

    def plot_search(self):
        plt.scatter(self.xx,self.yy,marker='*',color='orange')
        plt.title(f'{self.args.glac_name} Horizon Search \n Sky-view Factor = {self.sky_view:.3f}')
        plt.ylabel('Northing')
        plt.xlabel('Easting')
        if 'search_plot' in self.args.store:
            plt.savefig(self.out_search_fp)
        else:
            plt.show()

    def plot_result(self,df):
        args = self.args
        # initialize plot
        nrows = 2 if len(result_vars) > 2 else 1
        ncols = len(result_vars) if len(result_vars) <= 2 else int(len(result_vars)/2)
        fig,axes = plt.subplots(nrows,ncols,figsize=(ncols*4,nrows*3),layout='constrained')
        axes = axes.flatten()
        fig.suptitle(f'Annual plots for {args.glac_name} \n Sky-view factor: {self.sky_view:.3f}')
        
        # get days and hours of the year
        days = np.arange(366)
        hours = np.arange(0,24)
        
        # loop through variables to plot
        for i,var in enumerate(result_vars):
            # gather plot information
            label = varprops[var]['label']
            cmap = varprops[var]['cmap']
            ax = axes[i]

            # reshape and plot data
            df_plot = df.resample('h').mean()
            vardata = df_plot[var].to_numpy().reshape((len(days),len(hours)))
            pc = ax.pcolormesh(days,hours,vardata.T, cmap=cmap)

            # add colorbar ('shaded' is binary)
            if var != 'shaded':
                clb = fig.colorbar(pc,ax=ax,aspect=10,pad=0.02)
                clb.set_label(var,loc='top')

            # add labels
            ax.set_ylabel('Hour of day')
            ax.set_xlabel('Day of year')
            ax.set_title(label)
        
        # store or show plot
        if 'result_plot' in args.store:
            plt.savefig(self.out_image_fp,dpi=150)
        else: 
            plt.show()

    def plot_horizon(self):
        fig,ax = plt.subplots()
        args = self.args
        ax.fill_between(self.angles,self.horizon_elev*180/pi,color='black',alpha=0.6)
        ax.plot(self.angles,self.horizon_elev*180/pi,color='black')
        ax.set_xlabel('Azimuth angle ($\circ$)')
        ax.set_ylabel('Horizon angle ($\circ$)')
        fig.suptitle(args.glac_name+' shading by azimuth angle (0$^{\circ}$N)',fontsize=14)
        if 'horizon_plot' in args.store:
            plt.savefig(self.out_horizon_fp)
        else:
            plt.show()

    def store_site_info(self):
        self.site_df.loc[self.args.site,'lat'] = self.args.lat
        self.site_df.loc[self.args.site,'lon'] = self.args.lon
        if 'elevation' in self.site_df.columns:
            if np.isnan(self.site_df.loc[self.args.site,'elevation']):
                self.site_df.loc[self.args.site,'elevation'] = int(self.point_elev)
                string = '~ Saved sky view, slope, aspect and elevation '
            else:
                string = '~ Saved sky view, slope, and aspect '
        else:
            self.site_df.loc[self.args.site,'elevation'] = int(self.point_elev)
            string = '~ Saved sky view, slope, aspect and elevation '
        self.site_df.loc[self.args.site,'sky_view'] = self.sky_view
        self.site_df.loc[self.args.site,'slope'] = self.point_slope*180/pi
        self.site_df.loc[self.args.site,'aspect'] = self.point_aspect*180/pi
        self.site_df.to_csv(self.args.site_fp)
        print(string + f'to {self.args.glac_name}/site_constants.csv ~')

# RUN MODEL
if __name__ == '__main__':
    model = Shading()
    model.main()
    model.store_site_info()