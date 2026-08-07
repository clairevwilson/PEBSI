"""
Terrain class for PEBSI

Contains functions that handle the spatial
distribution of points, including loading
their DEM information (elevation, slope)
and executing the shading model.
"""
# Internal libraries
import os
import time 
import psutil
# External libraries
import xarray as xr
import rioxarray as rxr
import pandas as pd
import geopandas as gpd
import numpy as np
from pyproj import CRS, Transformer
import shapely.geometry as geom
from rasterio.enums import Resampling
# Local libraries
from pebsi.shading.shading import Shading

class Terrain:
    """
    Container for GPU arrays for points in the simulation.
    """
    def __init__(self, params):
        """
        Load the points and start timer for Terrain functions.
        """
        self.params = params
        shade_fp = params.sample_shading_fp if params.rgi_region == 0 else params.shading_fp
        self.shade_fn = os.path.join(shade_fp, params.shading_fn)

        # start timer for loading spatial inputs
        self.start_time = time.time()

        # create spatial points and load the DEM info for points
        self.get_points()
        return

    def get_points(self):
        """
        Parses method_distribute to break the model domain
        into points. Stores each point latitude, longitude,
        and glacier ID to self.
        """
        # load RGI data into memory for this region
        self.get_rgi_data()

        # test glacier: skip distribution logic entirely, use a hardcoded point
        if self.params.rgi_region == 0:
            gid = self.params.rgi_ids[0]
            self.lat_n = np.array([60.0])
            self.lon_n = np.array([-150.0])
            self.rgiid_n = np.array([gid], dtype=str)
            self.rgiid_unique = np.array([gid], dtype=str)
            self.N_POINTS = 1
            self.elev_n = self.slope_n = self.aspect_n = None
            return

        if self.params.method_distribute == 'scatter':
            lats, lons, glaciers = self.scatter_points()

            self.elev_n = None
            self.slope_n = None 
            self.aspect_n = None 

        elif self.params.method_distribute == 'sites':
            ns = len(self.params.sites)
            ng = len(self.params.rgi_ids)
            assert ns == ng, f'N sites (is {ns}) must equal N rgi_ids (is {ng})'
            lats, lons, glaciers = [], [], []
            elevs, slopes, aspects = [], [], []

            metadata_fn = self.params.metadata_fn
            metadata = pd.read_csv(metadata_fn, dtype=str, index_col='rgiid')
            for gid, site in zip(self.params.rgi_ids, self.params.sites):
                assert gid in metadata.index, \
                    f'To index by site, glacier ID must be associated with name in glacier_metadata ({gid})'

                gid_df = metadata.loc[gid]
                gid_df = gid_df.set_index('site')

                lats.append(float(gid_df.loc[site, 'lat']))
                lons.append(float(gid_df.loc[site, 'lon']))
                elevs.append(float(gid_df.loc[site, 'elevation']))
                slopes.append(float(gid_df.loc[site, 'slope']))
                aspects.append(float(gid_df.loc[site, 'aspect']))
                glaciers.append(gid)

            self.elev_n = np.array(elevs)
            self.slope_n = np.array(slopes)
            self.aspect_n = np.array(aspects)

        # store to self
        self.lat_n = np.array(lats)
        self.lon_n = np.array(lons)
        self.rgiid_n = np.array(glaciers, dtype=str)
        self.rgiid_unique = np.unique(self.rgiid_n)
        self.N_POINTS = len(self.lat_n)
        return
    
    def get_rgi_data(self):
        """
        Loads the shapefile for the region of interest.
        """
        # region 00 is the test glacier — no real RGI data needed
        if self.params.rgi_region == 0:
            self.rgi_df = pd.DataFrame()
            self.rgi_gdf = None
            return

        # find the regional shapefile
        region = str(self.params.rgi_region).zfill(2)
        all_rgi = os.listdir(self.params.rgi_fp)
        region_name = [f.split('.')[0] for f in all_rgi if f.startswith(region)]
        assert len(region_name) == 1, f'Did not find RGI region {region} data'
        shapefile_fn = f'../{region_name[0]}/{region_name[0]}.shp'

        # read in the shapefile
        df = pd.read_csv(os.path.join(self.params.rgi_fp, region_name[0]+'.csv'))
        gdf = gpd.read_file(os.path.join(self.params.rgi_fp, shapefile_fn))

        # store to self
        self.rgi_df = df
        self.rgi_gdf = gdf
        return

    def scatter_points(self, tolerance=0.05):
        """
        Samples approximately n_points, evenly distributed 
        inside a polygon shapefile using an adaptive grid 
        spacing search. Glaciers are naturally weighted by
        their area (bigger = more points).

        Parameters
        ==========
        tolerance : float
            Acceptable deviation between actual points 
            generated and n_points
        """

        # find the total number of parallel processes available 
        N_PARALLEL = self.params.n_points

        # first find the number of points for each glacier
        unique_ids = np.unique(self.params.rgi_ids)
        ids_fmtd = ['RGI60-'+id for id in unique_ids]
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
        for gid in unique_ids:
            target_n = rgi_df.loc[rgi_df['RGIId'] == 'RGI60-'+gid, 'points'].item()
            current_glacier = rgi_gdf.loc[rgi_gdf['RGIId'] == 'RGI60-'+gid]
            
            polygon = current_glacier.unary_union
            xmin, ymin, xmax, ymax = polygon.bounds
            area = polygon.area
            
            # initial analytical guess for even grid spacing: sqrt(Area / N)
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
                
                # check if we are within acceptable tolerance of our target N count
                if abs(current_count - target_n) / target_n <= tolerance:
                    break
                    
                # adjust grid step density dynamically based on overshoot/undershoot
                spacing *= np.sqrt(current_count / target_n)

            # package coordinates as lists
            points_gdf = gpd.GeoDataFrame(geometry=points_inside, crs=rgi_gdf.crs)
            points_latlon = points_gdf.to_crs(epsg=4326)
            xs = points_latlon.geometry.x.tolist()
            ys = points_latlon.geometry.y.tolist()
            
            # append lats and lons to the global list
            for lon, lat in zip(xs, ys):
                lons.append(lon)
                lats.append(lat)
                glaciers.append(gid)
            
        return lats, lons, glaciers
    
    def load_dem_info(self, dem, lats_in, lons_in):
        """
        Loads the DEM to get slope, aspect, and elevation 
        of each point.

        Parameters
        ==========
        dem : DataArray
            DEM for a region
        lats_in, lons_in : 1D arrays
            Latitude and longitude of points within this DEM
        """
        dem = dem['elevation']

        # mask nodata
        nodata = dem.rio.nodata 
        if nodata is not None:
            dem = dem.where((dem != nodata) & np.isfinite(dem))

        # get the resolution of the dataset in m
        x_res, y_res = dem.rio.resolution()

        # calculate gradient and get the slope
        dx, dy = np.gradient(dem, y_res, x_res)
        slope_vals = np.arctan(np.sqrt(dx**2 + dy**2))
        slope_vals = np.rad2deg(slope_vals)

        aspect_vals = np.arctan2(-dy, -dx)
        aspect_vals = (aspect_vals + 2*np.pi) % (2*np.pi)
        aspect_vals = np.rad2deg(aspect_vals) % 360

        # put data into DataArrays for clean indexing
        slope = xr.DataArray(slope_vals, coords=dem.coords, dims=dem.dims)
        aspect = xr.DataArray(aspect_vals, coords=dem.coords, dims=dem.dims)
        lat_xr = xr.DataArray(lats_in, dims='points')
        lon_xr = xr.DataArray(lons_in, dims='points')

        # reproject 2D datasets into lat/lon coordinates
        dem = dem.rio.reproject('EPSG:4326', resampling=Resampling.bilinear)
        slope = slope.rio.reproject('EPSG:4326', resampling=Resampling.bilinear)
        aspect = aspect.rio.reproject('EPSG:4326')

        # extract spatial attributes at lat and lon points
        elev_n = dem.sel(y=lat_xr, x=lon_xr, method='nearest').values 
        slope_n = slope.sel(y=lat_xr, x=lon_xr, method='nearest').values
        aspect_n = aspect.sel(y=lat_xr, x=lon_xr, method='nearest').values 

        return elev_n, slope_n, aspect_n
    
    def get_median_elevation(self):
        """
        Finds the median glacier elevation for every  individual 
        point, dynamically loading across multiple regional 
        CSV files.
        """
        # test glacier (region 00): use the hardcoded elevation from run_dem_functions
        if self.params.rgi_region == 0:
            self.median_elev_n = self.elev_n.copy()
            return

        # figure out what unique IDs there are
        unique_ids = np.unique(self.rgiid_n)
        unique_ids_fmtd = ['RGI60-'+i for i in unique_ids]
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
        Generator that yields sliced, buffered sub-DEM 
        datasets and their corresponding glacier sub-DataFrames 
        for gathering DEM information and running shading.
        """
        rgi_gdf = self.rgi_gdf
        region = str(self.params.rgi_region).zfill(2)
        vrt_path = self.params.cop30_vrt_path.format(r=region)
        assert os.path.exists(vrt_path), f'Missing DEM data expected at {vrt_path}'
        
        # load the master DEM using the VRT file produced in preprocessing
        master_dem = (rxr.open_rasterio(vrt_path)
                    .squeeze().drop_vars('band'))
        self.dem_crs = dem_crs = master_dem.rio.crs
        rgi_gdf = rgi_gdf.to_crs(dem_crs)
        
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

        for xi in range(len(x_edges) - 1):
            for yj in range(len(y_edges) - 1):
                # Filter for glaciers whose centroid falls within this block
                glaciers_in_block = rgi_gdf[
                    rgi_gdf['CenLon'].between(x_edges[xi], x_edges[xi+1]) &
                    rgi_gdf['CenLat'].between(y_edges[yj], y_edges[yj+1]) &
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

                # slice and reproject the cropped chunk
                sub_dem = master_dem.rio.clip_box(*buffered_bounds)
                sub_dem = sub_dem.rio.reproject(metric_crs)
                glaciers_in_block = glaciers_in_block.to_crs(metric_crs)

                # push to local RAM/GPU context
                sub_dem_ds = sub_dem.compute().to_dataset(name="elevation")
                
                # Pass the chunk data up to the execution loop
                yield sub_dem_ds, glaciers_in_block

    def yield_single_dem(self):
        """
        Single-glacier mode: load a single DEM directly 
        and yields it as a single chunk.
        """
        gid = self.params.rgi_ids[0]
        dem_fn = self.params.dem_fn.format(g=gid)
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
        master_dem = master_dem.rio.reproject(
            metric_crs, resampling=Resampling.bilinear
        )

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
        using the glacier centroid.

        Parameters
        ==========
        gdf : gpd.GeoDataFrame
            RGI dataframe clipped to the glacier(s) of interest
        """
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
        # test glacier (region 00): no DEM needed — use sample fixed values
        if self.params.rgi_region == 0:
            self.elev_n = np.array([1500.0])
            self.slope_n = np.array([10.0])
            self.aspect_n = np.array([180.0])
            return

        # make sure there is storage space for shading output
        output_fp = self.params.shading_fp
        os.makedirs(output_fp, exist_ok=True)

        # storage for 1D arrays
        compiled_inputs = {
            'elev_n': np.full(self.N_POINTS, np.nan),
            'slope_n': np.full(self.N_POINTS, np.nan),
            'aspect_n': np.full(self.N_POINTS, np.nan),
        }

        dem_vars = np.array([self.elev_n, self.slope_n, self.aspect_n], dtype=float)
        shade_exists = [os.path.exists(self.shade_fn.format(gid=gid)) for gid in self.rgiid_unique]
        if np.all(~np.isnan(dem_vars)) and np.all(shade_exists):
            # skip loading DEM if all the data is already confirmed
            return

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
            elev_b, slope_b, aspect_b = self.load_dem_info(sub_dem_ds, lats_in, lons_in)

            # store the outputs
            compiled_inputs['elev_n'][block_mask] = elev_b
            compiled_inputs['slope_n'][block_mask] = slope_b
            compiled_inputs['aspect_n'][block_mask] = aspect_b

            # check if any of these glaciers don't have a shading file
            missing_glaciers = []
            for _, glacier in glaciers_in_block.iterrows():
                gid = glacier['RGIId'].split('-')[-1]
                shade_fn = self.shade_fn.format(gid=gid) 
                if not os.path.exists(shade_fn):
                    missing_glaciers.append(gid)

            # if there are no missing glaciers, skip shading
            if len(missing_glaciers) < 1:
                continue

            print(f'Computing GPU shadows for {len(missing_glaciers)} missing glacier(s) in current chunk...')
            
            # calculate regional center point and bounds
            centroid_geo = glaciers_in_block.to_crs(epsg=4326).union_all().centroid
            minx, miny, maxx, maxy = glaciers_in_block.total_bounds

            # crop the DEM to the actual glaciers needed, plus a buffer for surrounding peaks
            y_slice = (slice(maxy + buffer_meters, miny - buffer_meters) 
                       if sub_dem_ds['y'][0] > sub_dem_ds['y'][-1] 
                       else slice(miny - buffer_meters, maxy + buffer_meters))
            x_slice = slice(minx - buffer_meters, maxx + buffer_meters)
            cropped_dem_ds = sub_dem_ds.sel(y=y_slice, x=x_slice)
            
            # initialize shading engine
            shading_model = Shading(cropped_dem_ds, step_size=1.0)
            shading_model.latitude = centroid_geo.y
            shading_model.longitude = centroid_geo.x
            
            datetimes_utc = pd.date_range('2000-01-01 00:00', '2000-12-31 23:00', freq='h', tz='UTC')
            masks_gpu, sun_az_rad, sun_zen_rad, svf = shading_model.compute_shadow_masks(datetimes_utc)
            
            # check size of dataset to avoid crashing RAM
            nt, nx, ny = masks_gpu.shape
            array_bytes = nt * ny * nx # int8 = 1 byte each cell

            # leave ~4 GB of RAM free
            available_ram = psutil.virtual_memory().available
            threshold_bytes = available_ram - 4e9
            if array_bytes > threshold_bytes:
                factor = np.ceil(np.sqrt(array_bytes / threshold_bytes)).astype(int)
                masks_gpu = masks_gpu[:, ::factor, ::factor]
                svf = svf[::factor, ::factor]
                y_coords = cropped_dem_ds['y'].values[::factor]
                x_coords = cropped_dem_ds['x'].values[::factor]
                print(f'Downsampled shadow mask by {factor} to fit in RAM')
            else:
                y_coords = cropped_dem_ds['y'].values
                x_coords = cropped_dem_ds['x'].values
            
            mask_3d_cpu = masks_gpu.get() if hasattr(masks_gpu, 'get') else masks_gpu

            # store shadow mask with UTC datetimes
            datetimes_clean = datetimes_utc.tz_localize(None)
            subregion_masks = xr.Dataset(
                {'shadow_mask': (['time','y','x'], mask_3d_cpu.astype(bool), {'units':'0=shade, 1=sun'}),
                 'solar_azimuth': (['time'], sun_az_rad, {'units': 'radians'}),
                 'solar_zenith': (['time'], sun_zen_rad, {'units': 'radians'}),
                 'sky_view_factor': (['y','x'], svf, {'units':'-'})},
                coords={'time': datetimes_clean, 'y': y_coords, 'x':x_coords}
            ).rio.write_crs(cropped_dem_ds.rio.crs)

            # crop shadows cleanly to individual glacier box
            for gid in missing_glaciers:
                g_bounds = glacier.geometry.bounds
                
                glacier_masks = subregion_masks.sel(
                    x=slice(g_bounds[0], g_bounds[2]),
                    y=slice(g_bounds[3], g_bounds[1])
                )

                # grab the CUDA arrays as numpy
                glacier_masks = glacier_masks.as_numpy()
                
                fn_out = self.shade_fn.format(gid=gid) 
                glacier_masks.chunk({'time': 8784, 'y': 8, 'x': 8}).to_zarr(fn_out, zarr_format=2)
                print(f'stored {gid} to {fn_out}')

            # clear items from memory
            del subregion_masks
            if 'glacier_masks' in locals(): del glacier_masks
            del masks_gpu, svf
            if 'mask_3d_cpu' in locals(): del mask_3d_cpu

            import gc
            gc.collect()

            try:
                import cupy as cp
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except:
                pass

        # store compiled inputs to self, if they were not already specified
        for var in ['elev_n','aspect_n','slope_n']:
            existing = getattr(self, var, None)
            new = compiled_inputs[var] 

            if existing is None:
                setattr(self, var, new)
            else:
                idx_nan = np.isnan(existing)
                existing[idx_nan] = new[idx_nan]
                setattr(self, var, existing)

        return
    
    def load_shading(self):
        """
        Loads the full shading mask for the points in the
        simulation from the preprocessed shading .zarr.
        Stores all (dayofyear, hour) entries from the file so
        pack_forcings can index by DOY+hour for any calendar year.
        """
        N_POINTS = self.N_POINTS
        shading_lookup = {}
        N_TIME = None
        masks = azimuth = zenith = None
        sky_view_factor = np.ones(N_POINTS)

        for gid in np.unique(self.rgiid_n):
            gid_idx = np.where(self.rgiid_n == gid)[0]

            fn = self.shade_fn.format(gid=gid)
            ds = xr.open_zarr(fn)
            ds = ds.rio.set_spatial_dims(x_dim='x', y_dim='y')
            ds = ds.rio.write_crs(ds['spatial_ref'].attrs['crs_wkt'])

            ds_doy = ds.time.dt.dayofyear.values
            ds_hour = ds.time.dt.hour.values

            # build lookup from the file's own time axis — covers all DOY+hours in the data
            if not shading_lookup:
                N_TIME = len(ds.time)
                shading_lookup = {(int(doy), int(hour)): i
                                  for i, (doy, hour) in enumerate(zip(ds_doy, ds_hour))}
                masks = np.full((N_POINTS, N_TIME), 2, dtype=np.int8)
                azimuth = np.full((N_POINTS, N_TIME), np.pi)
                zenith = np.zeros((N_POINTS, N_TIME))

            transformer = Transformer.from_crs("EPSG:4326", ds.rio.crs, always_xy=True)
            x_pts, y_pts = transformer.transform(self.lon_n[gid_idx], self.lat_n[gid_idx])
            target_x = xr.DataArray(x_pts, dims='points')
            target_y = xr.DataArray(y_pts, dims='points')

            selected = (ds
                .sel(y=target_y, x=target_x, method='nearest')
                .transpose('points', 'time'))

            masks[gid_idx, :] = selected['shadow_mask'].values
            azimuth[gid_idx, :] = selected['solar_azimuth'].values
            zenith[gid_idx, :] = selected['solar_zenith'].values
            sky_view_factor[gid_idx] = selected['sky_view_factor'].values

            ds.close()

        self.sky_view_factor = sky_view_factor
        self.solar_zenith = zenith
        self.solar_azimuth = azimuth
        self.shadow_mask = masks.astype(bool)
        self.shading_lookup = shading_lookup
        return

    def get_ice_albedo(self):
        """
        Samples each point's ice albedo from a per-glacier
        ice albedo GeoTIFF (params.ice_albedo_fn).

        Only called when params.option_ice_albedo_tif is True.
        """
        N_POINTS = self.N_POINTS
        ice_albedo_n = np.zeros(N_POINTS)

        for gid in np.unique(self.rgiid_n):
            gid_idx = np.where(self.rgiid_n == gid)[0]

            fn = self.params.ice_albedo_fn.format(gid=gid)
            da = rxr.open_rasterio(fn).squeeze().drop_vars('band')

            transformer = Transformer.from_crs("EPSG:4326", da.rio.crs, always_xy=True)
            x_pts, y_pts = transformer.transform(self.lon_n[gid_idx], self.lat_n[gid_idx])
            target_x = xr.DataArray(x_pts, dims='points')
            target_y = xr.DataArray(y_pts, dims='points')

            selected = da.sel(x=target_x, y=target_y, method='nearest')
            ice_albedo_n[gid_idx] = np.nan_to_num(selected.values, nan=self.params.albedo_ice)

            da.close()

        if self.params.debug:
            print('~ Loaded ice albedo from preprocessed tifs')

        return ice_albedo_n

    def get_initial_ice_thickness(self):
        """
        Samples each point's initial ice thickness from a per-glacier
        ice-thickness GeoTIFF (params.thickness_fn) and derives a fixed 
        bed elevation (elev_n - thickness_n).

        Only called when params.option_dynamics is True.
        """
        N_POINTS = self.N_POINTS
        thickness_n = np.zeros(N_POINTS)

        for gid in np.unique(self.rgiid_n):
            gid_idx = np.where(self.rgiid_n == gid)[0]

            fn = self.params.thickness_fn.format(gid=gid)
            da = rxr.open_rasterio(fn).squeeze().drop_vars('band')

            transformer = Transformer.from_crs("EPSG:4326", da.rio.crs, always_xy=True)
            x_pts, y_pts = transformer.transform(self.lon_n[gid_idx], self.lat_n[gid_idx])
            target_x = xr.DataArray(x_pts, dims='points')
            target_y = xr.DataArray(y_pts, dims='points')

            selected = da.sel(x=target_x, y=target_y, method='nearest')
            thickness_n[gid_idx] = np.nan_to_num(selected.values, nan=0.0)

            da.close()

        # fixed for the whole run: initial ice volume and bedrock topography
        self.thickness_n = np.maximum(thickness_n, 0.0)
        self.bed_n = self.elev_n - self.thickness_n
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

            wrong_len = (f'Wrong length in {name}\n'
                         f'Should be {self.N_POINTS}; is {len(input_array)}')
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