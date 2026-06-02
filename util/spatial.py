import xarray as xr
import rioxarray as rxr
import pandas as pd
import geopandas as gpd
from scipy.ndimage import distance_transform_edt
from rasterio.mask import mask as riomask
import numpy as np
import geopandas as gpd
import os, sys
import shapely.geometry as geom
from shapely.ops import voronoi_diagram, linemerge
from dataclasses import dataclass
from typing import Any

from shading.gpu_shading import Shading

try:
    import cupy as cp 
    xp = cp
except:
    xp = np

@dataclass
class SpatialData:
    """
    Container for GPU arrays scaled to the number of points.
    """
    args: Any 

    # Spatial coordinates (len n)
    lon_n: Any = None
    lat_n: Any = None
    rgiid_n: Any = None
    
    # Spatial attributes (len n)
    elev_n: Any = None
    slope_n: Any = None
    aspect_n: Any = None
    tz_n: Any = None

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

        if self.args.method_distribute == 'scatter':
            lats, lons, glaciers = self.scatter_points()
        else:
            lats, lons, glaciers = [], [], []
            df = pd.read_csv('data/by_glacier/gulkana/site_constants.csv', index_col=0)
            for site in ['AU','B','Z']:
                lats.append(df.loc[site, 'lat'])
                lons.append(df.loc[site, 'lon'])
                glaciers.append('01.00570')

        # store as xp arrays
        self.lat_n = xp.array(lats)
        self.lon_n = xp.array(lons)
        self.rgiid_n = np.array(glaciers, dtype=str)
        self.n = len(self.lat_n)
        return
    
    def get_rgi_data(self):
        # find the regional shapefile
        region = str(self.args.rgi_region).zfill(2)
        all_rgi = os.listdir(self.args.rgi_fp)
        region_name = [f.split('.')[0] for f in all_rgi if f.startswith(region)]
        assert len(region_name) == 1, f'Did not find RGI region {region} data'
        shapefile_fn = f'../{region_name[0]}/{region_name[0]}.shp'

        # read in the shapefile
        df = pd.read_csv(os.path.join(self.args.rgi_fp, region_name[0]+'.csv'))
        gdf = gpd.read_file(os.path.join(self.args.rgi_fp, shapefile_fn))
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
        N_PARALLEL = 1000

        # first find the number of points for each glacier
        ids_fmtd = ['RGI60-'+id for id in self.args.rgi_ids]
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
        for id in self.args.rgi_ids:
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
                candidate_points = [Point(x, y) for x, y in zip(xv.ravel(), yv.ravel())]
                
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
    
        # THIS DOES NOT WORK YET
        # elevation_step = self.args.bin_step
        # all_regions = os.listdir(self.args.rgi_fp)
        # glac_no = self.args.glac_no

        # raw_points = []
        # max_points = 0

        # # loop through RGI regions included in glac_no
        # for region in xp.unique([f[:2] for f in glac_no]):
        #     region_name = [f for f in all_regions if f.startswith(region)][0][:-4]
        #     shapefile_fn = f'../{region_name}/{region_name}.shp'

        #     # load the region shapefile
        #     gdf = gpd.read_file(self.args.rgi_fp + shapefile_fn)

        #     # filter by the IDs in glac_no
        #     glac_no_fmtd = ['RGI60-'+f for f in glac_no]
        #     selected_glaciers = gdf[gdf['RGIId'].isin(glac_no_fmtd)]

        #     for idx, row in selected_glaciers.iterrows():
        #         glacier_poly = row['geometry']
                
        #         # get voronoi diagram (centerline approximation)
        #         boundary_points = glacier_poly.boundary.interpolate(
        #             xp.linspace(0, glacier_poly.boundary.length, 100)
        #         )
        #         voronoi = voronoi_diagram(MultiLineString([LineString(boundary_points)]))
                
        #         # filter the Voronoi lines to only keep the ones completely INSIDE the glacier
        #         centerline_segments = [line for line in voronoi.geoms if line.within(glacier_poly)]
                
        #         try:
        #             # merge the loose segments into a single continuous LineString
        #             merged_line = linemerge(centerline_segments)
        #             if isinstance(merged_line, MultiLineString):
        #                 # if it branched, grab the longest continuous line (the main branch)
        #                 merged_line = max(merged_line.geoms, key=lambda l: l.length)
        #         except Exception:
        #             # fallback: if skeletonization fails on a weird geometry, use a simplified straight axis
        #             minx, miny, maxx, maxy = glacier_poly.bounds
        #             merged_line = LineString([(minx, miny), (maxx, maxy)])

        #         # sample heavily first to get the elevation profile along the centerline
        #         sample_dists = xp.linspace(0, merged_line.length, 200)
        #         sample_pts = [merged_line.interpolate(d) for d in sample_dists]
        #         sample_lons = [p.x for p in sample_pts if not p.is_empty]
        #         sample_lats = [p.y for p in sample_pts if not p.is_empty]
                
        #         # get elevations for the sample centerline points
        #         sample_elevs = self.dem.sel(x=sample_lons, y=sample_lats, method="nearest").values
                
        #         # determine min and max elevations for this specific glacier
        #         min_el, max_el = xp.nanmin(sample_elevs), xp.nanmax(sample_elevs)
                
        #         # define target elevations based on bin_step
        #         target_elevs = xp.arange(xp.ceil(min_el / elevation_step) * elevation_step, max_el, elevation_step)
                
        #         # map target elevations back to spatial distances along the centerline line
        #         target_dists = xp.interp(target_elevs, sample_elevs, sample_dists)
        #         sampled_points = [merged_line.interpolate(d) for d in target_dists]
                
        #         # extract coordinates for this glacier
        #         glac_lons = [p.x for p in sampled_points]
        #         glac_lats = [p.y for p in sampled_points]
                
        #         # update max_points and collect the raw data
        #         if len(glac_lats) > max_points:
        #             max_points = len(glac_lats)
                
        #         raw_points.append((glac_lats, glac_lons))

        # num_points = len(raw_points)
        # lats = xp.full((num_points, max_points), xp.nan)
        # lons = xp.full((num_points, max_points), xp.nan)
        # mask = xp.zeros((num_points, max_points), dtype=bool)

        # for idx, (lats, lons) in enumerate(raw_points):
        #     n = len(lats)
        #     if n == 0: continue
        #     lats[idx, :n] = lats
        #     lons[idx, :n] = lons
        #     mask[idx, :n] = True
    
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
        # get the resolution of the dataset in m
        x_res, y_res = dem.rio.resolution()
        x_res, y_res = abs(x_res), abs(y_res)

        # calculate slope and aspect from gradient
        dx, dy = xp.gradient(xp.array(dem.values), y_res, x_res)
        slope_vals = xp.arctan(xp.sqrt(dx**2 + dy**2))
        aspect_vals = xp.arctan2(-dy, -dx)

        # handle xp/np arrays so they can be put in a DataArray
        if hasattr(slope_vals, 'get'):
            slope_vals = slope_vals.get()
            aspect_vals = aspect_vals.get()
        else:
            slope_vals = np.asarray(slope_vals)
            aspect_vals = np.asarray(aspect_vals)

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
        sel_elev = dem.sel(y=lat_xr, x=lon_xr, method='nearest').values 
        sel_slope = slope.sel(y=lat_xr, x=lon_xr, method='nearest').values
        sel_aspect = aspect.sel(y=lat_xr, x=lon_xr, method='nearest').values 

        # convert to xp arrays 
        elev_n = xp.array(sel_elev)
        slope_n = xp.array(sel_slope)
        aspect_n = xp.array(sel_aspect)

        # estimate local timezone from longitude
        tz_n = xp.round(self.lon_n / 15)

        return elev_n, slope_n, aspect_n, tz_n
    
    def validate_spatial_data(self):
        """
        Validates the spatial inputs do not
        contain nans and are the correct length
        """
        names = ['lat_n','lon_n','elev_n','slope_n','aspect_n']
        for name in names:
            input_array = getattr(self, name)
            missing = len(xp.where(xp.isnan(input_array))[0])
            missing_str = f'Missing {name[:-2]} data for {missing} points'
            assert ~xp.any(xp.isnan(input_array)), missing_str

            wrong_len = f'Wrong length in {name}\nShould be {self.n}; is {len(input_array)}'
            assert len(input_array) == self.n, wrong_len

        self.get_median_elevation()

        if self.args.debug:
            print('~ Inputs validated')
        return
    
    def get_median_elevation(self):
        """
        Finds the median glacier elevation for every 
        individual point, dynamically loading across 
        multiple regional CSV files.
        """
        # figure out what unique IDs there are
        unique_ids = xp.unique(self.rgiid_n)
        ids = self.rgiid_n
        
        # store mapping by glacier ID: mediane elvation
        glacier_med_elev = {}
        
        # crop the RGI df to the IDs 
        df = self.rgi_df
        df_filtered = df[df['RGIId'].isin(unique_ids)]
        
        # store these values in master lookup
        for _, row in df_filtered.iterrows():
            glacier_med_elev[row['RGIId'].split('-')[-1]] = float(row['Zmed'])

        # stretch glacier-scale values to points
        stretched_elevations = []
        for gid in ids:
            median_elev = glacier_med_elev[gid]
            stretched_elevations.append(median_elev)
            
        # store median elevation
        self.median_elev_n = xp.array(stretched_elevations, dtype=xp.float64)
        return
    
    def yield_dem_chunks(self, block_size_deg=4.0, buffer_meters=20000):
        """
        Generator that yields sliced, buffered 
        sub-DEM datasets and their corresponding 
        glacier sub-DataFrames for gathering 
        DEM information and running shading.
        """
        rgi_gdf = self.rgi_gdf
        region = str(self.args.rgi_region).zfill(2)
        vrt_path = self.args.vrt_path.format(r=region)
        
        # load the master DEM using the VRT file produced in preprocessing
        master_dem = rxr.open_rasterio(vrt_path, chunks=True).squeeze().drop_vars("band")
        dem_crs = master_dem.rio.crs
        
        # reproject RGI dataset to match the DEM
        if rgi_gdf.crs != dem_crs:
            rgi_gdf = rgi_gdf.to_crs(dem_crs)

        # generate a coarse bounding box grid over the region
        minx, miny, maxx, maxy = rgi_gdf.total_bounds
        x_edges = np.arange(minx, maxx + block_size_deg, block_size_deg)
        y_edges = np.arange(miny, maxy + block_size_deg, block_size_deg)

        for i in range(len(x_edges) - 1):
            for j in range(len(y_edges) - 1):
                block_box = geom.box(x_edges[i], y_edges[j], x_edges[i+1], y_edges[j+1])
                
                # Filter for glaciers whose centroid falls within this block
                glaciers_in_block = rgi_gdf[
                    rgi_gdf.centroid.within(block_box) & 
                    rgi_gdf['RGIId'].isin(['RGI60-' + id for id in self.args.rgi_ids])
                ]
                
                if glaciers_in_block.empty:
                    continue

                # buffer the area surrounding the glacier for shading model
                total_bounds = glaciers_in_block.total_bounds
                buffered_bounds = (
                    total_bounds[0] - buffer_meters,
                    total_bounds[1] - buffer_meters,
                    total_bounds[2] + buffer_meters,
                    total_bounds[3] + buffer_meters
                )

                # slice and push to local RAM/GPU context
                sub_dem = master_dem.rio.clip_box(*buffered_bounds)
                sub_dem_ds = sub_dem.compute().to_dataset(name="elevation")
                
                # Pass the chunk data up to the execution loop
                yield sub_dem_ds, glaciers_in_block

    def run_shading(self, block_size_deg=4.0, buffer_meters=20000):
        """
        Processes shading for all glaciers in the region using the DEM splitting 
        generator, calculating shadows only if the output file doesn't already exist.
        """
        # make sure there is storage space for shading output
        output_fp = self.args.shading_data_fp
        os.makedirs(output_fp, exist_ok=True)

        # loop through DEM chunks and corresponding glacier subsets
        for sub_dem_ds, glaciers_in_block in self.yield_dem_chunks(block_size_deg, buffer_meters):
            
            # --- PHASE 1: Hook your other loaders in here ---
            # You can pass sub_dem_ds and glaciers_in_block to load_dem() 
            # to pull slope/aspect metrics cleanly using the exact same chunk matrix!
            # self.load_dem(sub_dem_ds, glaciers_in_block)

            # --- PHASE 2: Conditional Shading Process ---
            # Check if ANY glaciers in this chunk are missing their shading profiles
            missing_glaciers = []
            for _, glacier in glaciers_in_block.iterrows():
                out_file = os.path.join(output_fp, f"{glacier['RGIId']}_shadows.nc")
                if not os.path.exists(out_file):
                    missing_glaciers.append(glacier)

            # If every glacier in this chunk is already cached on disk, skip the GPU completely!
            if not missing_glaciers:
                continue

            print(f"Computing GPU shadows for {len(missing_glaciers)} missing glaciers in current chunk...")
            
            # Calculate regional center point coordinates
            centroid_geo = glaciers_in_block.to_crs(epsg=4326).union_all().centroid
            
            # Fire up the shading compute engine
            shading_model = Shading(sub_dem_ds, x_coord="x", y_coord="y", step_size=1.0)
            shading_model.latitude = centroid_geo.y
            shading_model.longitude = centroid_geo.x
            
            datetimes_utc = pd.date_range('2000-01-01', '2000-12-31', freq='h', tz='UTC')
            mask_dict = shading_model.compute_shadow_masks(datetimes_utc)
            
            mask_arrays = [m.get() if hasattr(m, 'get') else m for m in mask_dict.values()]
            mask_3d = np.stack(mask_arrays, axis=0).astype(np.float16)

            subregion_masks = xr.Dataset(
                {"shadow_mask": (["time", "y", "x"], mask_3d)},
                coords={"time": datetimes_utc, "y": sub_dem_ds["y"], "x": sub_dem_ds["x"]}
            )

            # crop shadows cleanly to individual glacier box
            for glacier in missing_glaciers:
                glacier_id = glacier['RGIId']
                g_bounds = glacier.geometry.bounds
                
                glacier_masks = subregion_masks.sel(
                    x=slice(g_bounds[0], g_bounds[2]),
                    y=slice(g_bounds[3], g_bounds[1])
                )
                
                out_file = os.path.join(output_fp, f"{glacier_id}_shadows.nc")
                glacier_masks.to_netcdf(out_file)

    def run_shading(self, block_size_deg=4.0, buffer_meters=20000):
        """
        Processes shading for all glaciers in the region 
        by breaking COP30 DEM into chunks, running the
        shading model on each chunk, and then cropping out
        individual RGI IDs to store as separate .nc files.
        """
        if len(self.args.rgi_ids) > 1:
            print(f'Running shading model for {len(self.args.rgi_ids)} glaciers...')

        # open the master DEM using the VRT file produced in preprocessing
        rgi_gdf = self.rgi_gdf
        region = str(self.args.rgi_region).zfill(2)
        vrt_path = self.args.format(r=region)
        master_dem = rxr.open_rasterio(vrt_path, chunks=True).squeeze().drop_vars("band")
        dem_crs = master_dem.rio.crs
        
        # ensure RGI matches the DEM project coordinates
        if rgi_gdf.crs != dem_crs:
            rgi_gdf = rgi_gdf.to_crs(dem_crs)

        # make sure directory exists for shading output
        output_fp = self.args.shading_data_fp
        os.makedirs(output_fp, exist_ok=True)

        # generate a coarse bounding box grid over Alaska
        minx, miny, maxx, maxy = rgi_gdf.total_bounds
        x_edges = np.arange(minx, maxx + block_size_deg, block_size_deg)
        y_edges = np.arange(miny, maxy + block_size_deg, block_size_deg)

        # loop through the macro block grid
        for i in range(len(x_edges)-1):
            for j in range(len(y_edges)-1):
                print(f'Processing block {i+1}/{len(x_edges)-1}, {j+1}/{len(y_edges)-1}...')
                block_box = geom.box(x_edges[i], y_edges[j], x_edges[i+1], y_edges[j+1])
                
                # find glaciers whose centroid fall within this block
                glaciers_in_block = rgi_gdf[
                    rgi_gdf.centroid.within(block_box) & 
                    rgi_gdf['RGIId'].isin(['RGI60-'+id for id in self.args.rgi_ids])
                ]
                
                if glaciers_in_block.empty:
                    continue

                # get total spatial envelope of ALL glaciers in this block combined
                total_bounds = glaciers_in_block.total_bounds
                
                # Apply the shadow peak buffer (e.g., 20km) around the entire envelope
                buffered_bounds = (
                    total_bounds[0] - buffer_meters,
                    total_bounds[1] - buffer_meters,
                    total_bounds[2] + buffer_meters,
                    total_bounds[3] + buffer_meters
                )

                # read just this chunk out of the Master Virtual Raster
                try:
                    sub_dem = master_dem.rio.clip_box(*buffered_bounds)
                    # load fully into RAM/GPU memory now that it's a small, safe chunk
                    sub_dem_ds = sub_dem.compute().to_dataset(name="elevation")
                except Exception as e:
                    print(f"Skipping empty or out-of-bounds DEM block: {e}")
                    continue

                # calculate subregion center for the UTC solar calculation
                centroid_geo = glaciers_in_block.to_crs(epsg=4326).union_all().centroid
                
                # initialize GPU shading model for this block
                shading_model = Shading(sub_dem_ds, x_coord="x", y_coord="y", step_size=1.0)
                shading_model.latitude = centroid_geo.y
                shading_model.longitude = centroid_geo.x
                
                # generate shadow masks for all times within a year (use arbitrary leap year)
                datetimes_utc = pd.date_range('2000-01-01', '2000-12-31', freq='h', tz='UTC')
                mask_dict = shading_model.compute_shadow_masks(datetimes_utc)
                
                # unpack GPU arrays to standard xarray datasets
                mask_arrays = [m.get() if hasattr(m, 'get') else m for m in mask_dict.values()]
                mask_3d = np.stack(mask_arrays, axis=0).astype(np.float16)

                subregion_masks = xr.Dataset(
                    {"shadow_mask": (["time", "y", "x"], mask_3d)},
                    coords={"time": datetimes_utc, "y": sub_dem_ds["y"], "x": sub_dem_ds["x"]}
                )

                # loop through individual glaciers in this block
                for _, glacier in glaciers_in_block.iterrows():
                    glacier_id = glacier['RGIId']
                    g_bounds = glacier.geometry.bounds
                    
                    # crop cleanly to individual glacier box
                    glacier_masks = subregion_masks.sel(
                        x=slice(g_bounds[0], g_bounds[2]),
                        y=slice(g_bounds[3], g_bounds[1])
                    )
                    
                    # save .nc for this glacier
                    out_file = os.path.join(output_fp, f"{glacier_id}_shadows.nc")
                    glacier_masks.to_netcdf(out_file)
