import xarray as xr
import rioxarray as rxr
import pandas as pd
import geopandas as gpd
import numpy as np
import os, sys
import time
from pyproj import CRS
from pysolar.solar import get_altitude, get_azimuth
import shapely.geometry as geom
from dataclasses import dataclass
from shading.gpu_shading import Shading

@dataclass
class Terrain:
    """
    Container for GPU arrays scaled to the number of points.
    """
    def __init__(self, params):
        self.params = params
        self.shade_fn = os.path.join(params.shading_data_fp, '{id}_shadows.nc')

        # start timer for loading spatial inputs
        self.start_time = time.time()

        # create spatial points and load the DEM info for points
        self.get_points()
        return

    def get_points(self):
        """
        Takes a list of RGI IDs, extracts their geometric centerlines, 
        and samples them at perfectly equal intervals.
        
        Returns:
        --------
        glacier_lats, glacier_lons : 2D lists/arrays of shape (len(rgi_ids), num_points_per_glacier)
        """
        # load RGI data into memory for this region
        self.get_rgi_data()

        if self.params.method_distribute == 'scatter':
            lats, lons, glaciers = self.scatter_points()
        else:
            lats, lons, glaciers = [], [], []
            df = pd.read_csv('data/by_glacier/gulkana/site_constants.csv', index_col=0)
            for site in ['AU','B','D']:
                lats.append(df.loc[site, 'lat'])
                lons.append(df.loc[site, 'lon'])
                glaciers.append('01.00570')

            df = pd.read_csv('data/by_glacier/wolverine/site_constants.csv', index_col=0)
            for site in ['N','B','EC']:
                lats.append(df.loc[site, 'lat'])
                lons.append(df.loc[site, 'lon'])
                glaciers.append('01.09162')

        # store to self
        self.lat_n = np.array(lats)
        self.lon_n = np.array(lons)
        self.rgiid_n = np.array(glaciers, dtype=str)
        self.rgiid_unique = np.unique(self.rgiid_n)
        self.N_POINTS = len(self.lat_n)
        return
    
    def get_rgi_data(self):
        # find the regional shapefile
        region = str(self.params.rgi_region).zfill(2)
        all_rgi = os.listdir(self.params.rgi_fp)
        region_name = [f.split('.')[0] for f in all_rgi if f.startswith(region)]
        assert len(region_name) == 1, f'Did not find RGI region {region} data'
        shapefile_fn = f'../{region_name[0]}/{region_name[0]}.shp'

        # read in the shapefile
        df = pd.read_csv(os.path.join(self.params.rgi_fp, region_name[0]+'.csv'))
        gdf = gpd.read_file(os.path.join(self.params.rgi_fp, shapefile_fn))
        if gdf.crs.is_geographic:
            gdf = gdf.to_crs(epsg=32632) # Force metric project tracking

        # store to self 
        self.rgi_df = df 
        self.rgi_gdf = gdf
        return

    def scatter_points(self, tolerance=0.05):
        """
        Samples approximately N points evenly distributed inside a polygon shapefile
        using an adaptive grid spacing search.
        """
        # find the total number of parallel processes available 
        N_PARALLEL = self.params.n_points

        # first find the number of points for each glacier
        ids_fmtd = ['RGI60-'+id for id in self.rgiid_unique]
        rgi_df = self.rgi_df.loc[self.rgi_df['RGIId'].isin(ids_fmtd)]
        total_area = rgi_df['Area'].sum()
        rgi_df['exact_points'] = (rgi_df['Area'] / total_area) * N_PARALLEL
        rgi_df['points'] = rgi_df['exact_points'].round().astype(int)

        # load the geodataframe
        rgi_gdf = self.rgi_gdf

        # fix discrepancy to get exactly N_PARALLEL points
        current_sum = rgi_df['points'].sum()
        remainder = int(N_PARALLEL - current_sum)

        if remainder != 0:
            # find the indices of the largest rounding fractions to adjust
            rgi_df['residual'] = rgi_df['exact_points'] - rgi_df['points']

            if remainder > 0:
                # need more points: add them to the ones that were rounded down the most
                idx = rgi_df['residual'].nlargest(remainder).index
                rgi_df.loc[idx, 'points'] += 1
            elif remainder < 0:
                # have too many points: subtract from the ones that were rounded up the most
                idx = rgi_df['residual'].nsmallest(abs(remainder)).index
                rgi_df.loc[idx, 'points'] -= 1

        # loop through RGI IDs
        lats, lons, glaciers = [], [], []
        for id in self.rgiid_unique:
            target_n = rgi_df.loc[rgi_df['RGIId'] == 'RGI60-'+id, 'points'].item()
            current_glacier = rgi_gdf.loc[rgi_gdf['RGIId'] == 'RGI60-'+id]
            
            polygon = current_glacier.unary_union
            xmin, ymin, xmax, ymax = polygon.bounds
            area = polygon.area
            
            # Initial analytical guess for even grid spacing: sqrt(Area / N)
            spacing = np.sqrt(area / target_n)
            
            # optimization loop to fine-tune spacing to hit your exact target N count
            for _ in range(15):
                x_coords = np.arange(xmin, xmax, spacing)
                y_coords = np.arange(ymin, ymax, spacing)
                
                # create coordinate meshgrid matrix
                xv, yv = np.meshgrid(x_coords, y_coords)
                candidate_points = [geom.Point(x, y) for x, y in zip(xv.ravel(), yv.ravel())]
                
                # vectorized boundary clipping mask: Keep points strictly inside the polygon
                points_inside = [p for p in candidate_points if polygon.contains(p)]
                current_count = len(points_inside)
                
                # Check if we are within acceptable tolerance of our target N count
                if abs(current_count - target_n) / target_n <= tolerance:
                    break
                    
                # Adjust grid step density dynamically based on overshoot/undershoot
                spacing *= np.sqrt(current_count / target_n)

            # Package coordinates out cleanly
            points_gdf = gpd.GeoDataFrame(geometry=points_inside, crs=rgi_gdf.crs)
            points_latlon = points_gdf.to_crs(epsg=4326)
            xs = points_latlon.geometry.x.tolist()
            ys = points_latlon.geometry.y.tolist()
            
            # append lats and lons to the global list
            for lon, lat in zip(xs, ys):
                lons.append(lon)
                lats.append(lat)
                glaciers.append(id)
            
        return lats, lons, glaciers
    
    def load_dem_info(self, dem, lats_in, lons_in):
        """
        Loads the DEM to get slope, aspect, 
        and elevation of each point.

        Parameters
        ==========
        dem : DataArray
            DEM for a region
        lats_in, lons_in : 1D arrays
            Latitude and longitude of points within
            this DEM
        """
        dem = dem['elevation']

        # mask nodata
        nodata = dem.rio.nodata 
        if nodata is not None:
            dem = dem.where((dem != nodata) & np.isfinite(dem))

        # get the resolution of the dataset in m
        x_res, y_res = dem.rio.resolution()
        x_res, y_res = abs(x_res), abs(y_res)

        # calculate slope and aspect from gradient
        dy, dx = np.gradient(np.squeeze(dem.values), y_res, x_res)
        slope_vals = np.arctan(np.sqrt(dx**2 + dy**2))
        aspect_vals = np.arctan2(-dy, -dx)

        # put data into DataArrays for clean indexing
        slope = xr.DataArray(slope_vals, coords=dem.coords, dims=dem.dims)
        aspect = xr.DataArray(aspect_vals, coords=dem.coords, dims=dem.dims)
        lat_xr = xr.DataArray(lats_in , dims='points')
        lon_xr = xr.DataArray(lons_in, dims='points')

        # reproject 2D datasets into lat/lon coordinates
        dem = dem.rio.reproject('EPSG:4326')
        slope = slope.rio.reproject('EPSG:4326')
        aspect = aspect.rio.reproject('EPSG:4326')

        # extract spatial attributes at lat and lon points
        elev_n = dem.sel(y=lat_xr, x=lon_xr, method='nearest').values 
        slope_n = slope.sel(y=lat_xr, x=lon_xr, method='nearest').values
        aspect_n = aspect.sel(y=lat_xr, x=lon_xr, method='nearest').values 

        # estimate local timezone from longitude
        tz_n = np.round(lons_in / 15)

        return elev_n, slope_n, aspect_n, tz_n
    
    def get_median_elevation(self):
        """
        Finds the median glacier elevation for every 
        individual point, dynamically loading across 
        multiple regional CSV files.
        """
        # figure out what unique IDs there are
        unique_ids = np.unique(self.rgiid_n)
        unique_ids_fmtd = ['RGI60-'+id for id in unique_ids]
        ids = self.rgiid_n
        
        # store mapping by glacier ID: median elvation
        glacier_med_elev = {}
        
        # crop the RGI df to the IDs 
        df = self.rgi_df
        df_filtered = df[df['RGIId'].isin(unique_ids_fmtd)]
        
        # store these values in master lookup
        for _, row in df_filtered.iterrows():
            glacier_med_elev[row['RGIId'].split('-')[-1]] = float(row['Zmed'])

        # stretch glacier-scale values to points
        stretched_elevations = []
        for gid in ids:
            median_elev = glacier_med_elev[gid]
            stretched_elevations.append(median_elev)
            
        # store median elevation
        self.median_elev_n = np.array(stretched_elevations)
        return
    
    def yield_dem_chunks(self, block_size_deg=0.5, buffer_meters=10_000):
        """
        Generator that yields sliced, buffered sub-DEM datasets 
        and their corresponding glacier sub-DataFrames for gathering 
        DEM information and running shading.
        """
        rgi_gdf = self.rgi_gdf
        region = str(self.params.rgi_region).zfill(2)
        vrt_path = self.params.cop30_vrt_path.format(r=region)
        assert os.path.exists(vrt_path), f'Missing DEM data expected at {vrt_path}'
        
        # load the master DEM using the VRT file produced in preprocessing
        master_dem = (rxr.open_rasterio(vrt_path)
                    .squeeze().drop_vars('band'))
        self.dem_crs = dem_crs = master_dem.rio.crs
        
        # reproject RGI dataset to match the DEM
        if rgi_gdf.crs != dem_crs:
            rgi_gdf = rgi_gdf.to_crs(dem_crs)

        if dem_crs.is_geographic:
            buffer = buffer_meters / 111000.0
        else:
            buffer = buffer_meters

        # generate a coarse bounding box grid over the region
        minx, miny, maxx, maxy = rgi_gdf.total_bounds
        x_edges = np.arange(minx, maxx + block_size_deg, block_size_deg)
        y_edges = np.arange(miny, maxy + block_size_deg, block_size_deg)

        for i in range(len(x_edges) - 1):
            for j in range(len(y_edges) - 1):
                # Filter for glaciers whose centroid falls within this block
                glaciers_in_block = rgi_gdf[
                    rgi_gdf['CenLon'].between(x_edges[i], x_edges[i+1]) &
                    rgi_gdf['CenLat'].between(y_edges[j], y_edges[j+1]) &
                    rgi_gdf['RGIId'].isin(['RGI60-' + i for i in self.rgiid_unique])
                ]
                
                if glaciers_in_block.empty:
                    continue

                # buffer the area surrounding the glacier for shading model
                total_bounds = glaciers_in_block.total_bounds
                buffered_bounds = (
                    total_bounds[0] - buffer,
                    total_bounds[1] - buffer,
                    total_bounds[2] + buffer,
                    total_bounds[3] + buffer
                )

                # get metric projection
                self.dem_crs = metric_crs = self._get_metric_crs(glaciers_in_block)

                # slice and creproject the cropped chunk
                sub_dem = master_dem.rio.clip_box(*buffered_bounds)
                sub_dem = sub_dem.rio.reproject(metric_crs)
                glaciers_in_block = glaciers_in_block.to_crs(metric_crs)

                # push to local RAM/GPU context
                sub_dem_ds = sub_dem.compute().to_dataset(name="elevation")
                
                # Pass the chunk data up to the execution loop
                yield sub_dem_ds, glaciers_in_block

    def yield_single_dem(self):
        """
        Single-glacier mode: load a single DEM directly and yields it
        as a single chunk with all glaciers.
        """
        glacier_id = self.params.rgi_ids[0]
        dem_fn = self.params.dem_fn.format(g=glacier_id)
        assert os.path.exists(dem_fn), f'Missing DEM at {dem_fn}'

        # load the DEM
        master_dem = (rxr.open_rasterio(dem_fn)
                      .squeeze().drop_vars('band'))
        
        # crop the RGI geodataframe to the actual glaciers here
        rgi_gdf = self.rgi_gdf
        rgi_gdf = rgi_gdf[
            rgi_gdf['RGIId'].isin(['RGI60-' + i for i in self.rgiid_unique])
        ]
        
        # convert the DEM to a good equal-area projection
        metric_crs = self._get_metric_crs(rgi_gdf)
        master_dem = master_dem.rio.reproject(metric_crs)

        # make sure coordinate systems are consistent
        self.dem_crs = metric_crs 
        if rgi_gdf.crs != metric_crs:
            rgi_gdf = rgi_gdf.to_crs(metric_crs) 

        sub_dem_ds = master_dem.compute().to_dataset(name = 'elevation')
        yield sub_dem_ds, rgi_gdf

    def _get_dem_chunks(self, block_size_deg, buffer_meters):
        """
        Routes to the correct yield function whether there is
        a single DEM to grab or a region to parse.
        """
        if self.params.dem_fn is not None:
            return self.yield_single_dem()
        else:
            return self.yield_dem_chunks(block_size_deg, buffer_meters)
        
    def _get_metric_crs(self, gdf):
        """
        Derives a region-appropriate equal-area projection 
        using the glacier centroid."""
        centroid = gdf.to_crs(epsg=4326).union_all().centroid
        return CRS(
            f"+proj=laea +lat_0={centroid.y:.2f} +lon_0={centroid.x:.2f} "
            f"+datum=WGS84 +units=m +no_defs"
        )

    def run_dem_functions(self, block_size_deg=0.5, buffer_meters=10_000):
        """
        Processes DEM-dependent variables by chunking
        a large COP30 DEM into pieces.
        1. Loads DEM information of elevation, slope, aspect
        2. Runs shading model for any glaciers that have not
           already been preprocessed.

        Parameters
        ==========
        block_size_deg : float
            Size of the chunks to break the DEM into
        buffer_meters : float
            Buffer to add around glacier bounds when 
            cropping DEM to capture surrounding peaks
        """
        # make sure there is storage space for shading output
        output_fp = self.params.shading_data_fp
        os.makedirs(output_fp, exist_ok=True)

        # storage for 1D arrays
        compiled_inputs = {
            'elev_n': np.full(self.N_POINTS, np.nan),
            'slope_n': np.full(self.N_POINTS, np.nan),
            'aspect_n': np.full(self.N_POINTS, np.nan),
            'tz_n': np.full(self.N_POINTS, -1, dtype=np.int32),
        }

        # loop through DEM chunks and corresponding glacier subsets
        for sub_dem_ds, glaciers_in_block in self._get_dem_chunks(block_size_deg, buffer_meters):
            
            # gather all the DEM info for the points in this block
            block_ids = [g.split('-')[-1] for g in glaciers_in_block['RGIId']]
            block_mask = np.isin(self.rgiid_n, np.array(block_ids))

            # convert lat and lon to numpy arrays 
            lat_n = self.lat_n.get() if hasattr(self.lat_n, 'get') else self.lat_n
            lon_n = self.lon_n.get() if hasattr(self.lon_n, 'get') else self.lon_n

            # pass the points in this block to the DEM info loader
            lats_in = lat_n[block_mask]
            lons_in = lon_n[block_mask]
            elev_b, slope_b, aspect_b, tz_b = self.load_dem_info(sub_dem_ds, lats_in, lons_in)

            # store the outputs
            compiled_inputs['elev_n'][block_mask] = elev_b
            compiled_inputs['slope_n'][block_mask] = slope_b
            compiled_inputs['aspect_n'][block_mask] = aspect_b
            compiled_inputs['tz_n'][block_mask] = tz_b

            # check if any of these glaciers don't have a shading file
            missing_glaciers = []
            for _, glacier in glaciers_in_block.iterrows():
                id = glacier['RGIId'].split('-')[-1]
                shade_fn = self.shade_fn.format(id=id) 
                if not os.path.exists(shade_fn):
                    missing_glaciers.append(glacier)

            # if there are no missing glaciers, skip shading
            if len(missing_glaciers) < 1:
                continue

            print(f'Computing GPU shadows for {len(missing_glaciers)} missing glaciers in current chunk...')
            
            # calculate regional center point and bounds
            centroid_geo = glaciers_in_block.to_crs(epsg=4326).union_all().centroid
            minx, miny, maxx, maxy = glaciers_in_block.total_bounds

            # crop the DEM to the actual glaciers needed, plus a buffer for surrounding peaks
            y_slice = (slice(maxy + buffer_meters, miny - buffer_meters) 
                       if sub_dem_ds['y'][0] > sub_dem_ds['y'][-1] 
                       else slice(miny - buffer_meters, maxy + buffer_meters))
            x_slice = slice(minx - buffer_meters, maxx + buffer_meters)
            cropped_dem_ds = sub_dem_ds.sel(y=y_slice, x=x_slice)
            
            # Fire up the shading compute engine
            shading_model = Shading(cropped_dem_ds, step_size=1.0)
            shading_model.latitude = centroid_geo.y
            shading_model.longitude = centroid_geo.x
            
            datetimes_utc = pd.date_range('2000-01-01 00:00', '2000-12-31 23:00', freq='h', tz='UTC')
            masks_gpu, sun_az, sun_zen, svf = shading_model.compute_shadow_masks(datetimes_utc)
            
            mask_3d_cpu = masks_gpu.get() if hasattr(masks_gpu, 'get') else masks_gpu

            datetimes_clean = datetimes_utc.tz_localize(None)
            subregion_masks = xr.Dataset(
                {'shadow_mask': (['time','y','x'], mask_3d_cpu.astype(bool)),
                 'solar_azimuth': (['time'], sun_az),
                 'solar_zenith': (['time'], sun_zen),
                 'sky_view_factor': (['y','x'], svf)},
                coords={'time': datetimes_clean, 'y': cropped_dem_ds['y'], 'x': cropped_dem_ds['x']}
            )

            # crop shadows cleanly to individual glacier box
            for glacier in missing_glaciers:
                glacier_id = glacier['RGIId'].split('-')[-1]
                g_bounds = glacier.geometry.bounds
                
                glacier_masks = subregion_masks.sel(
                    x=slice(g_bounds[0], g_bounds[2]),
                    y=slice(g_bounds[3], g_bounds[1])
                )

                # grab the CUDA arrays as numpy
                glacier_masks = glacier_masks.as_numpy()
                
                fn_out = self.shade_fn.format(id=id) 
                glacier_masks.to_netcdf(fn_out)
                print('stored', glacier_id)

            # clear items from memory
            del subregion_masks
            if 'glacier_masks' in locals(): del glacier_masks
            del masks_gpu
            if 'mask_3d_cpu' in locals(): del mask_3d_cpu

            import gc
            gc.collect()

            try:
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except:
                pass

        # store compiled inputs to self
        self.elev_n = compiled_inputs['elev_n']
        self.slope_n = compiled_inputs['slope_n']
        self.aspect_n = compiled_inputs['aspect_n']
        self.tz_n = compiled_inputs['tz_n']
        return
    
    def load_shading(self, dates):
        # define storage for shading masks
        N_POINTS = self.N_POINTS 
        N_TIME = len(dates)
        masks = np.full((N_POINTS, N_TIME), 2, dtype=np.int8)
        azimuth = np.full((N_POINTS, N_TIME), np.pi)
        zenith = np.zeros((N_POINTS, N_TIME))
        sky_view_factor = np.ones(N_POINTS)

        # find unique RGI IDs in the list of all points
        for id in self.rgiid_unique:
            idx = np.where(self.rgiid_unique == id)[0]

            # get xr DataArrays for the slicing variables
            target_lat = xr.DataArray(self.lat_n[idx] , dims='points')
            target_lon = xr.DataArray(self.lon_n[idx] , dims='points')

            # load shading file
            fn = self.shade_fn.format(id=id)
            ds = xr.open_dataset(fn)
            ds_doy = ds.time.dt.dayofyear.values
            ds_hour = ds.time.dt.hour.values

            # map times in dates to the index of that doy/hour in ds
            lookup = {(doy, hour): i for i, (doy, hour) in enumerate(zip(ds_doy, ds_hour))}
            target_indices = [lookup[(d, h)] for d, h in zip(dates.dayofyear, dates.hour)]
            target_time_idx = xr.DataArray(target_indices, dims='time')
            
            # select data for the lats, lons, and times
            selected = (ds
                .sel(y=target_lat, x=target_lon, method='nearest')
                .isel(time=target_time_idx)
                .transpose('points','time'))

            masks[idx, :] = selected['shadow_mask'].values
            azimuth[idx, :] = selected['solar_azimuth'].values
            zenith[idx, :] = selected['solar_zenith'].values
            sky_view_factor[idx] = selected['sky_view_factor'].values

        self.sky_view_factor = sky_view_factor
        self.solar_zenith = zenith 
        self.solar_azimuth = azimuth
        self.shadow_mask = masks.astype(bool)
        return

    def validate_terrain_data(self):
        """
        Validates the spatial inputs do not
        contain nans and are the correct length
        """
        names = ['lat_n','lon_n','elev_n','slope_n','aspect_n']
        for name in names:
            input_array = getattr(self, name)
            missing = len(np.where(np.isnan(input_array))[0])
            missing_str = f'Missing {name[:-2]} data for {missing} points'
            assert ~np.any(np.isnan(input_array)), missing_str

            wrong_len = f'Wrong length in {name}\nShould be {self.N_POINTS}; is {len(input_array)}'
            assert len(input_array) == self.N_POINTS, wrong_len

        self.get_median_elevation()

        # make sure passed parameter arrays are the correct length
        for k, v in vars(self.params).items():
            if isinstance(v, np.ndarray) and len(v) > 1:
                assert len(v) == self.N_POINTS, \
                    f"Parameter '{k}' has length {len(v)} but expected 1 or {self.N_POINTS}"


        elapsed = time.time()-self.start_time
        if elapsed < 60:
            unit = 's'
        elif elapsed < 3600:
            elapsed /= 60 
            unit = 'min'
        else:
            elapsed /= 3600
            unit = 'hr'
        if self.params.debug:
            print(f"~ Terrain processing of {self.N_POINTS} points complete in {elapsed:.1f} {unit}")
        return