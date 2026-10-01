import os
import re
import socket
import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pyproj import Transformer
import rioxarray
from rioxarray.merge import merge_arrays
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

if 'trace' in socket.gethostname():
    base_fp = '/trace/group/rounce/cvwilson/'
    home_fp = '/trace/home/cvwilson/research/'
elif 'bridges' in socket.gethostname():
    base_fp = '/ocean/projects/ees260009p/cwilson4/'
    home_fp = '/jet/home/cwilson4/'
elif 'lantern' in socket.gethostname():
    base_fp = '/Users/cvw/local/'
    home_fp = '/Users/cvw/local/'
elif 'campfire' in socket.gethostname():
    base_fp = '/home/claire/Local/Output/'
    home_fp = '/home/claire/Local/'
else:
    base_fp = 'C:/Users/cvw30/Research/Output/ddf/'
    home_fp = 'C:/Users/cvw30/Research/'

# moved to project.parameters (shared with every script that builds a
# config, not just loss scoring); re-exported here so existing imports of
# translate_rgi/translate_names from this module keep working
from project.parameters import translate_rgi, translate_names

# albedo observation uncertainty, derived from the data
ALBEDO_SIGMA = 0.0777

class Albedo():
    def __init__(self, name, use='s2'):
        """
        Grabs the distributed timeseries of remotely sensed albedo
        to compare against a glacier-wide model run.

        Parameters
        name : str
            Glacier name
        use : str
            Dataset to use from ['l8','l9','s2','all']
        """
        # store input attributes
        self.name = name
        self.use = use

        # define dictionary for labeling datasets
        self.dataset_dict = {'l8':'Landsat 8', 'l9':'Landsat 9', 's2':'Sentinel-2',
                             'mean':'mean of Landsat/Sentinel scenes'}

        # get RGI7 glacier ID number
        glac_no = translate_rgi[name]['7']
        self.glac_no = glac_no

        # open dataframes
        self.albedo_fp = base_fp + 'data/albedo/'

        # grab the distributed data
        self.get_glacier_albedo()

        # put time into numpy format
        self.time = np.array(pd.to_datetime(self.time))
        return

    def get_glacier_albedo(self):
        num = str(self.glac_no[3:])

        if self.use == 'all':
            use_list = ['s2','l8','l9']
        else:
            if type(self.use) != list:
                use_list = [self.use]
            else:
                use_list = self.use

        # get filename for this glac_no
        albedo_fns = [self.albedo_fp + f'{num}/RGI2000-v7.0-G-01-{num}_{data}.nc' for data in use_list]

        # build dataset
        self.time = []
        self.dtype = []
        self.ds_meas = None
        for albedo_fn, dtype in zip(albedo_fns, use_list):
            # open the dataset and get the proper CRS
            ds = xr.open_dataset(albedo_fn)
            self.crs = ds.spatial_ref.attrs['crs_wkt']

            # convert coordinates
            ds = ds['albedo'].rio.write_crs(self.crs).reset_coords(drop=True).to_dataset()
            ds['dtype'] = ('time', np.array([dtype]*len(ds.time.values)).flatten())
            if self.ds_meas is None:
                self.ds_meas = ds
            else:
                self.ds_meas = xr.concat([self.ds_meas, ds], dim='time')

        self.time = np.array(self.time)
        self.dtype = np.array(self.dtype)

        self.ds_meas = self.ds_meas.squeeze('band')

        # get rid of duplicates by averaging 
        ds_mean = self.ds_meas.groupby("time").mean()
        s = self.ds_meas["dtype"].to_series()
        dupes = s.index.duplicated(keep=False)
        s.loc[dupes] = "mean"
        dtype_da = xr.DataArray(
            s.groupby(level="time").first().values.astype(object),
            dims=["time"],
            coords={"time": s.groupby(level="time").first().index},
        )
        ds_mean["dtype"] = dtype_da
        self.ds_meas = ds_mean.sortby('time')
        self.ds_meas.attrs['crs'] = self.crs
        return

    def get_model_albedo(self, model_ds, months_range=list(range(3, 10))):
        """
        Selects the remotely sensed albedo nearest to each point in a
        glacier-wide model run, and matches the model to those times.

        Parameters
        model_ds : xr.Dataset
            Glacier-wide model output with a 'point' dimension and
            'lon'/'lat' coordinates giving each simulation point's location.
        """
        proj = Transformer.from_crs('EPSG:4326', self.crs, always_xy=True)
        x, y = proj.transform(model_ds.lon.values, model_ds.lat.values)

        measured = (self.ds_meas
                    .sel(
                        x=xr.DataArray(x, dims='point'),
                        y=xr.DataArray(y, dims='point'),
                        method='nearest')
                    .drop_duplicates('time'))

        valid_times = measured.time.where(measured.time.dt.month.isin(months_range), drop=True)
        measured = measured.sel(time=valid_times)

        if measured.time.size == 0:
            print('! No valid measurements')
            return

        meas_period = f'{measured.time.values[0]} : {measured.time.values[-1]}'
        mod_period = f'{model_ds.time.values[0]} : {model_ds.time.values[-1]}'
        assert valid_times[0] <= model_ds.time.values[-1], f'Measurements start after model period\nModel: {mod_period}\nMeasured: {meas_period}'
        assert valid_times[-1] >= model_ds.time.values[0], f'Measurements end before model period\nModel: {mod_period}\nMeasured: {meas_period}'

        modeled = model_ds.sel(time=measured.time, method='nearest')

        self.mod = modeled['albedo'].values
        self.meas = measured['albedo'].values
        self.ds_meas_matched = measured
        self.model_ds = model_ds
        self.format = 'values'
        return

    def get_deltas(self, method='march_mean'):
        """
        choose method from 'max', 'march_mean'
        """
        dates = pd.to_datetime(self.ds_meas_matched.time.values)
        years = np.unique(dates.year)

        mod = self.mod.copy()
        meas = self.meas.copy()

        if method == 'max':
            for year in years:
                idx = np.where(dates.year == year)[0]
                mod[idx] -= np.nanmax(mod[idx], axis=0)
                meas[idx] -= np.nanmax(meas[idx], axis=0)

        elif method == 'march_mean':
            all_meas_march = []
            all_mod_march = []
            for year in years:
                march_idx = np.where((dates.year == year) & (dates.month == 3))[0]
                if len(march_idx) > 0:
                    march_meas = np.nanmean(meas[march_idx], axis=0)
                else:
                    march_meas = np.full(meas.shape[1], np.nan)

                march_mod = self.model_ds.sel(time=slice(f'{year}-03-01', f'{year}-04-01'))['albedo'].mean(dim='time').values

                all_meas_march.append(march_meas)
                all_mod_march.append(march_mod)

            all_meas_march = np.array(all_meas_march)
            all_mod_march = np.array(all_mod_march)

            mean_meas = np.nanmean(all_meas_march, axis=0)
            nan_meas = np.isnan(all_meas_march)
            all_meas_march[nan_meas] = np.broadcast_to(mean_meas, all_meas_march.shape)[nan_meas]

            mean_mod = np.nanmean(all_mod_march, axis=0)
            nan_mod = np.isnan(all_mod_march)
            all_mod_march[nan_mod] = np.broadcast_to(mean_mod, all_mod_march.shape)[nan_mod]

            for i, year in enumerate(years):
                idx = np.where(dates.year == year)[0]
                mod[idx] -= all_mod_march[i]
                meas[idx] -= all_meas_march[i]

        self.mod = mod
        self.meas = meas
        self.format = 'deltas'
        return

    def mae(self, mod=None, meas=None):
        mod = self.mod if mod is None else mod
        meas = self.meas if meas is None else meas
        return np.nanmean(np.abs(mod - meas))

    def log_loss(self, mod=None, meas=None, sigma=ALBEDO_SIGMA):
        """
        Gaussian negative log-likelihood of the albedo residuals
        [nats], assuming observations are mod ~ N(meas, sigma^2).
        Puts albedo error on the same log-likelihood scale as the
        Bernoulli log-loss used for snowline/melt, so the two can
        be summed into one calibration objective. sigma should
        reflect the albedo product's actual measurement/retrieval
        uncertainty, not be tuned as a free weight.
        """
        mod = self.mod if mod is None else mod
        meas = self.meas if meas is None else meas
        residual = mod - meas
        return np.nanmean(0.5 * np.log(2 * np.pi * sigma**2) + residual**2 / (2 * sigma**2))

    def bias(self, mod=None, meas=None):
        mod = self.mod if mod is None else mod
        meas = self.meas if meas is None else meas
        return np.nanmean(mod - meas)

    def rmse(self, mod=None, meas=None):
        mod = self.mod if mod is None else mod
        meas = self.meas if meas is None else meas
        return np.sqrt(np.nanmean(np.square(mod - meas)))

    def plot_1to1(self):
        mod = np.array(self.mod).flatten()
        meas = np.array(self.meas).flatten()

        no_nan = (~np.isnan(mod)) & (~np.isnan(meas))
        mod, meas = mod[no_nan], meas[no_nan]

        fig, ax = plt.subplots()
        
        xy = np.vstack([mod, meas])
        kde = gaussian_kde(xy)
        z = kde(xy)

        idx = z.argsort()
        x, y, z = meas[idx], mod[idx], z[idx]

        minn, maxx = (-0.8, 0) if self.format == 'deltas' else (0.1, 0.9)

        ax.scatter(x, y, c=z, cmap='magma')
        ax.plot([minn, maxx], [minn, maxx], 'k--')

        bias = np.mean(mod - meas)
        MAE = np.mean(np.abs((mod - meas)))
        ax.text(0.02, 0.95, f'Bias: {bias:.3f}', transform=ax.transAxes)
        ax.text(0.02, 0.90, f'MAE: {MAE:.3f}', transform=ax.transAxes)

        ax.set_xlabel('Remotely sensed albedo differences')
        ax.set_xlim(minn, maxx)

        ax.set_ylabel('Modeled albedo differences')
        ax.set_ylim(minn, maxx)
        ax.tick_params(length=5)

        # plt.savefig('../figs/new_albedo_agreement.png', dpi=300)
        plt.show()

class Wet():
    def __init__(self, name, direction=None, water_threshold=0.03):
        """
        Loads the SAR wet extent preprocessed 2D dataset.

        Parameters
        ----------
        name : glacier common name
        direction : ascending, descending, or None for both
        water_threshold : model total water [m w.e.] above which
                          the surface is considered wet (seen by SAR)
        """
        self.name = name
        self.direction = direction
        self.water_threshold = water_threshold

        # find rgi7 glacier number
        rgi7id = translate_rgi[name]['7']
        folder = base_fp + 'data/sar/sar_cubes/'
        fn = folder + f'{rgi7id}_wet_extent.nc'

        # load dataset
        ds = xr.open_dataset(fn)
        if direction is not None:
            ds = ds.sel(time=ds['pass_direction'] == direction)

        self.crs = ds.attrs['projection']
        self.dem = ds['dem']
        self.glacier_mask = ds['glacier_mask']

        # per-pixel wet mask is already gridded, use it directly;
        # kept as float (1/0) so missing retrievals stay NaN
        self.ds_meas_wet = ds['is_wet'].astype(float)
        return

    def get_model_wet(self, model_ds):
        # total_water is in kg m-2, threshold is in m w.e.
        mod_wet = model_ds['total_water'].resample(time='1d').min() > self.water_threshold * 1000

        # select the measured boolean grid at each simulation point
        proj = Transformer.from_crs('EPSG:4326', self.crs, always_xy=True)
        x, y = proj.transform(model_ds.lon.values, model_ds.lat.values)
        meas_wet = self.ds_meas_wet.sel(
            x=xr.DataArray(x, dims='point'), y=xr.DataArray(y, dims='point'), method='nearest')

        # keep only measurement times inside the model's time range
        start = max(mod_wet.time.values[0], meas_wet.time.values[0])
        end = min(mod_wet.time.values[-1], meas_wet.time.values[-1])
        meas_wet = meas_wet.sel(time=slice(start, end))

        # measurement times are sparse and irregular, so match each one to
        # its nearest model timestep instead of slicing on a shared range
        tolerance = pd.Timedelta('12h')
        mod_wet_matched = mod_wet.reindex(time=meas_wet.time, method='nearest', tolerance=tolerance)

        # drop measurement times with no model timestep close enough to match
        valid = mod_wet_matched.notnull().all(dim='point')

        self.mod_wet = mod_wet_matched.sel(time=valid).values.astype(bool)
        self.meas_wet = meas_wet.sel(time=valid).values
        self.time = meas_wet.sel(time=valid).time.values

        assert self.mod_wet.shape == self.meas_wet.shape
        return

    def bernoulli_loss(self, mod_wet=None, meas_wet=None, eps=1e-3):
        mod_wet = self.mod_wet if mod_wet is None else mod_wet
        meas_wet = self.meas_wet if meas_wet is None else meas_wet

        p = np.clip(np.asarray(mod_wet, dtype=float), eps, 1 - eps)
        y = np.asarray(meas_wet, dtype=float)
        return -np.nanmean(y * np.log(p) + (1 - y) * np.log(1 - p))

def _load_rgi_outline(glac_no):
    """
    Loads one glacier's outline polygon from the regional RGI shapefile,
    in the shapefile's native (lat/lon) CRS. Shared by DEM (which then
    reprojects it to match a sample raster) and MassBalance's tif loader
    (which reprojects it independently per tile, since tiles can sit in
    different local UTM zones).
    """
    region = glac_no[:2]
    rgi_fp = base_fp + 'RGI/rgi60/'
    for fn in os.listdir(rgi_fp):
        if region in fn and 'Zone' not in fn:
            reg_name = fn
    shp_reg = gpd.read_file(rgi_fp + f'{reg_name}/{reg_name}.shp')
    outline = shp_reg[shp_reg['RGIId'] == f'RGI60-{glac_no}']
    assert len(outline), f'no RGI outline found for RGI60-{glac_no}'
    return outline


class MassBalance():
    def __init__(self, name, dates=None, source='csv', density=850):
        """
        Glacier-wide mass balance from Hugonnet et al. (2021), expressed
        as an average annual rate [m w.e. yr-1] so the model is compared
        as a rate rather than a net balance over the period.

        Parameters
        name : str
            Glacier name
        dates : tuple of strings
            Start and end of the model run. For source='csv' the run must
            cover the observation period (2000-01-01 to 2020-01-01); for
            source='tif' they select the {start}_{end} raster directory.
        source : str
            'csv' (default): per-glacier 20-year rates supplied by Hugonnet
            (df_pergla_global_20yr-filled.csv). mb_mwea_err already combines
            the dh/dt, density (850 +/- 60 kg m-3) and outline uncertainties.

            'tif': a period-specific dhdt raster pair (e.g. the 2015-2020
            epoch product), converted to m w.e. yr-1 with `density`. No
            valid glacier-wide uncertainty yet, so sigma is NaN.
        density : float
            Ice density for the tif dh/dt conversion [kg m-3]
        """
        self.name = name
        self.density = density
        self.density_water = 1000
        self.source = source

        if dates is not None:
            self.start = pd.to_datetime(dates[0])
            self.end = pd.to_datetime(dates[1])
        else:
            self.start = pd.to_datetime('2000-01-01')
            self.end = pd.to_datetime('2020-01-01')

        if source == 'csv':
            self._load_from_csv()
        elif source == 'tif':
            self._load_from_tif()
        else:
            raise ValueError(f"source must be 'csv' or 'tif', got {source!r}")

        self.n_years = (self.matched_end - self.matched_start).days / 365.25
        return

    def _load_from_csv(self):
        rgiid = f'RGI60-{translate_rgi[self.name]["6"]}'
        data_fp = base_fp + 'data/mass_balance/df_pergla_global_20yr-filled.csv'
        df = pd.read_csv(data_fp, usecols=['rgiid', 't1', 't2', 'mb_mwea', 'mb_mwea_err'])
        row = df[df['rgiid'] == rgiid]
        assert len(row) == 1, f'expected one row for {rgiid} in {data_fp}, found {len(row)}'
        row = row.iloc[0]

        self.matched_start = pd.to_datetime(row['t1'])
        self.matched_end = pd.to_datetime(row['t2'])
        assert self.start <= self.matched_start and self.end >= self.matched_end, (
            f'run {self.start.date()} to {self.end.date()} does not cover the '
            f'observation period {self.matched_start.date()} to {self.matched_end.date()}')

        self.meas = row['mb_mwea']
        self.sigma = row['mb_mwea_err']
        return

    def _load_from_tif(self):
        """
        Period-specific dh/dt [m/yr] from a raster. The dhdt_err raster is
        a per-pixel local uncertainty, not the uncertainty of the
        glacier-wide mean (averaging it naively overstated sigma by 17-31x
        in testing), so sigma is left as NaN until that is derived properly.
        """
        glac_no = translate_rgi[self.name]['6']
        outline = _load_rgi_outline(glac_no)

        rate, matched = self._clip_period_raster('dhdt', outline)
        self.matched_start, self.matched_end = matched

        self.meas = rate * self.density / self.density_water
        self.sigma = np.nan

        warnings.warn(
            "\n   source='tif' gives self.meas but NOT self.sigma (NaN). "
            'See _load_from_tif docstring.',
            UserWarning)
        return

    def _clip_period_raster(self, kind, outline):
        """
        Finds every {kind} ('dhdt' or 'dhdt_err') tile under this period's
        directory whose bounding box overlaps `outline`, mosaics them if
        the glacier straddles more than one tile (reprojecting onto the
        first tile's CRS -- tiles can sit in different local UTM zones
        near a zone boundary), clips to the outline, and returns
        (glacier-mean value, (period_start, period_end)) -- the period
        parsed from the tile filename(s), asserted identical across every
        tile used.
        """
        period_dir = os.path.join(base_fp, 'data/mass_balance',
                                   f'{self.start.year}_{self.end.year}', kind)
        if not os.path.isdir(period_dir):
            # fallback for a period whose tiles were never split into their
            # own {start}_{end}/{kind}/ subfolder
            period_dir = os.path.join(base_fp, f'data/mass_balance/{kind}')

        candidates = []
        for fn in sorted(os.listdir(period_dir)):
            if not fn.endswith('.tif'):
                continue
            fp = os.path.join(period_dir, fn)
            da = rioxarray.open_rasterio(fp).squeeze()
            outline_here = outline.to_crs(da.rio.crs)
            minx, miny, maxx, maxy = outline_here.total_bounds
            bx0, by0, bx1, by1 = da.rio.bounds()
            if maxx < bx0 or minx > bx1 or maxy < by0 or miny > by1:
                continue  # tile's bbox doesn't overlap the glacier's bbox
            candidates.append((fn, da))

        assert candidates, (f'no {kind} tile under {period_dir} overlaps '
                             f'{self.name}')

        if len(candidates) == 1:
            fn, da = candidates[0]
            mosaic = da
        else:
            # glacier straddles a tile boundary -- reproject the rest onto
            # the first tile's CRS and mosaic before clipping
            target_crs = candidates[0][1].rio.crs
            rasters = [da if i == 0 else da.rio.reproject(target_crs)
                       for i, (fn, da) in enumerate(candidates)]
            mosaic = merge_arrays(rasters)

        outline_here = outline.to_crs(mosaic.rio.crs)
        clipped = mosaic.rio.clip(outline_here.geometry, outline_here.crs,
                                  drop=True, all_touched=True)

        nodata = clipped.rio.nodata
        vals = clipped.values
        valid = np.isfinite(vals)
        if nodata is not None:
            valid &= (vals != nodata)
        assert valid.any(), (f'{kind} tile(s) for {self.name} have no valid '
                              'pixels inside the outline')
        mean_val = float(vals[valid].mean())

        periods = set()
        for fn, da in candidates:
            m = re.search(r'(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})', fn)
            assert m, f'could not parse a period from tile filename {fn}'
            periods.add((pd.to_datetime(m.group(1)), pd.to_datetime(m.group(2))))
        assert len(periods) == 1, (f'{kind} tiles under {period_dir} do not '
                                    f'all share the same period: {periods}')

        return mean_val, periods.pop()

    def get_model_mb(self, ds):
        """
        Average annual glacier-wide mass balance [m w.e. yr-1] from a
        model output dataset, assuming the output contains unique points
        on one glacier weighted by ds.weight. The model is summed over
        matched_start to matched_end (the observation period) and divided
        by its length, so a longer run is truncated to the observation.
        """
        model_start = pd.to_datetime(ds.time.values[0]).date()
        model_end = pd.to_datetime(ds.time.values[-1]).date()

        model_start_in_range = ds.time.values[0] <= pd.to_datetime(self.matched_start)
        model_end_in_range = ds.time.values[-1] >= pd.to_datetime(self.matched_end)
        assert model_start_in_range and model_end_in_range, \
            f'Model run does not cover observation period (spans {model_start} to {model_end})'

        ds_obs = ds.sel(time=slice(self.matched_start, self.matched_end))
        mb_weighted = ds_obs.mass_balance.sum(dim='time') * ds_obs.weight
        self.mod = mb_weighted.sum().values / self.n_years
        return

    def log_loss(self, mod=None, meas=None, sigma=None):
        """
        Gaussian negative log-likelihood of the mass balance residual
        [nats], assuming mod ~ N(meas, sigma^2), all in m w.e. yr-1.
        sigma is the reported measurement uncertainty, not a tuned weight.
        """
        mod = self.mod if mod is None else mod
        meas = self.meas if meas is None else meas
        sigma = self.sigma if sigma is None else sigma
        residual = mod - meas
        return 0.5 * np.log(2 * np.pi * sigma**2) + residual**2 / (2 * sigma**2)

class DEM():
    def __init__(self, name, epsg='default'):
        # store input attributes
        self.name = name

        # get RGI7 glacier ID number
        glac_no = translate_rgi[name]['6']
        self.glac_no = glac_no

        # get DEM filename
        dem_fp = base_fp + 'data/dems/RGI1_DEM/'
        dem_fn = dem_fp + f'RGI60-{self.glac_no}_dem.tif'

        # open DEM
        dem = xr.open_dataarray(dem_fn, engine='rasterio').squeeze()

        # ensure both datasets are on the same crs
        if epsg != 'default':
            dem = dem.rio.reproject(f'EPSG:{epsg}')
        self.dem = dem

        # open shapefile
        self.shp = _load_rgi_outline(glac_no)

        # get min and max elevation
        self.min_elev = float(dem.min().values)
        self.max_elev = float(dem.max().values)
        return

    def reproject_to(self, da_sample):
        # reproject to match the grid of the da_sample
        self.shp = self.shp.to_crs(da_sample.crs)
        self.dem = self.dem.rio.reproject_match(da_sample)
        return