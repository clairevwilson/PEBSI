"""
pebsi_glide_connector

Couples PEBSI (point-based surface mass + energy balance) to GLIDE
(gridded ice dynamics) so that a PEBSI simulation can evolve
glacier geometry.

PEBSI tracks a scattered point cloud (N_POINTS,) as 1D arrays of
arbitrary lat/lon locations, while GLIDE tracks a structured
raster (ny, nx) grid. This connector translates spatial mapping
such that GLIDE is run periodically in the PEBSI simulation on
a raster interpolated from the PEBSI simulation points.

GLIDE simulations are run on an RGI O2 subregion basis.

Open TODOs (need a decision before this runs end-to-end)
----------------------------------------------------------
1. `GlideCoupler._build_glide_model()` -- GLIDE needs a bed-topography
   raster and an initial ice-thickness raster covering the bounding box
   of every glacier in an O2 sub-region, in a shared metric CRS. PEBSI's
   Terrain class already loads a surface DEM per glacier (see
   util/terrain.py: yield_single_dem / yield_dem_chunks, self.dem_crs,
   Terrain._get_metric_crs) but nowhere loads ice thickness, and nothing
   currently mosaics DEM tiles across a whole O2 region. Either:
     (a) point this at a prepared per-region thickness raster (e.g.
         consensus ice thickness, OGGM), or
     (b) derive one, following the pattern in
         glide/examples/wrangell/preprocessing/make_dem.py and
         make_bedradar.py.
   Until this is filled in, _build_glide_model raises NotImplementedError.
"""

import numpy as np
import cupy as cp
import jax.numpy as jnp
from scipy.interpolate import griddata, RegularGridInterpolator
from pyproj import Transformer

from glide.model import IceDynamics

RHO_ICE = 917.0     # kg m-3
RHO_WATER = 1000.0  # kg m-3


# --------------------------------------------------------------------- #
#                     coordinate / regridding helpers
# --------------------------------------------------------------------- #

def latlon_to_glacier_crs(lat, lon, crs):
    """Project PEBSI point lat/lon (EPSG:4326) into a glacier's metric CRS."""
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = transformer.transform(np.asarray(lon), np.asarray(lat))
    return np.asarray(x), np.asarray(y)


def points_to_grid(x_pts, y_pts, values, x_grid, y_grid,
                    method="linear", fill_method="nearest"):
    """
    Regrid scattered PEBSI point values onto GLIDE's regular (ny, nx) grid.

    PEBSI points are irregularly scattered and don't cover the full 
    glacier extent uniformly, so griddata is used to interpolate onto
    a GLIDE-type grid. Cells outside the scattered points' are back-filled 
    with a nearest-neighbor pass.

    Parameters
    ----------
    x_pts, y_pts : (N_POINTS,) arrays, meters, in the glacier's metric CRS
    values : (N_POINTS,) array, quantity to regrid
    x_grid, y_grid : 1-D cell-center coordinate arrays for the target grid
        (as stored on GlideCoupler.grids[region])

    Returns
    -------
    (len(y_grid), len(x_grid)) array
    """
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_vals = griddata((x_pts, y_pts), values, (xx, yy), method=method)

    nan_mask = np.isnan(grid_vals)
    if np.any(nan_mask):
        fill_vals = griddata((x_pts, y_pts), values, (xx, yy), method=fill_method)
        grid_vals[nan_mask] = fill_vals[nan_mask]

    return grid_vals


def grid_to_points(x_grid, y_grid, grid_vals, x_pts, y_pts):
    """
    Sample an updated GLIDE grid field (e.g. new surface elevation) back
    onto PEBSI's scattered points via bilinear interpolation.
    """
    y_grid = np.asarray(y_grid)
    grid_vals = np.asarray(grid_vals)
    if y_grid[0] > y_grid[-1]:
        y_grid = y_grid[::-1]
        grid_vals = grid_vals[::-1, :]

    interp = RegularGridInterpolator(
        (y_grid, x_grid), grid_vals, bounds_error=False, fill_value=None
    )
    return interp(np.column_stack([y_pts, x_pts]))


def mwe_to_ice_thickness(mb_mwe):
    """
    Converts net surface mass balance (m water equivalent)
    to ice-equivalent thickness (m).
    """
    return mb_mwe * RHO_WATER / RHO_ICE


# --------------------------------------------------------------------- #
#                              GlideCoupler
# --------------------------------------------------------------------- #

class GlideCoupler:
    """
    Owns one GLIDE IceDynamics model per RGI O2 sub-region (a cluster of
    spatially-grouped glaciers) and drives annual coupling steps between
    PEBSI and GLIDE.
    """

    def __init__(self, terrain, params, n_levels=5):
        self.terrain = terrain
        self.params = params
        self.n_levels = n_levels

        self._o2_of_gid = self._build_o2_lookup(terrain)

        # group this run's glaciers by O2 sub-region
        self.region_glaciers = {}  # o2_region -> list of rgiids
        for gid in terrain.rgiid_unique:
            region = self._o2_of_gid[gid]
            self.region_glaciers.setdefault(region, []).append(gid)

        self.models = {}  # o2_region -> glide.model.IceDynamics
        self.grids = {}   # o2_region -> {'x': ndarray, 'y': ndarray, 'crs': pyproj.CRS}

        for region, gids_in_region in self.region_glaciers.items():
            model, grid_info = self._build_glide_model(region, gids_in_region)
            self.models[region] = model
            self.grids[region] = grid_info

    # ----------------------------- setup ------------------------------ #

    def _build_o2_lookup(self, terrain):
        """
        Maps each RGI ID present in this run to its O2Region, using the
        attribute table PEBSI already loads in Terrain.get_rgi_data()
        (util/terrain.py -> self.rgi_df, read from
        00_rgi60_attribs/{region}.csv, which includes an O2Region column).

        terrain.rgiid_n entries are the bare RGI ID (e.g. "01.00570", no
        "RGI60-" prefix) -- see Terrain.scatter_points / get_median_elevation
        for the same RGIId <-> rgiid_n format convention.
        """
        df = terrain.rgi_df
        short_id = df["RGIId"].str.split("-").str[-1]
        lookup = dict(zip(short_id, df["O2Region"]))

        missing = set(terrain.rgiid_unique) - set(lookup)
        if missing:
            raise KeyError(f"No O2Region found in rgi_df for RGI IDs: {sorted(missing)}")
        return lookup

    def _build_glide_model(self, region, gids_in_region):
        """
        One-time construction of a shared GLIDE grid + IceDynamics model
        covering every glacier in O2 sub-region `region`. See module
        TODO #1: this needs a bed-topography raster and an initial
        ice-thickness raster spanning the bounding box of all glaciers in
        `gids_in_region`, in one shared metric CRS, before it can build a
        real Multigrid.

        Expected shape once implemented
        --------------------------------
        - union the RGI polygons for `gids_in_region` (terrain.rgi_gdf) to
          get the region's bounding box; pick a shared metric CRS for it
          (e.g. Terrain._get_metric_crs on that subset)
        - pick dx / n_levels such that ny, nx are each divisible by
          2 ** (n_levels - 1) (GLIDE's multigrid requirement)
        - model = IceDynamics(n_levels=self.n_levels, ny=ny, nx=nx, dx=dx,
                               x0=x0, y0=y0, crs=crs)
        - model.mg.state.H.set(initial_thickness); H_prev.set(same)
        - model.mg.geometry.bed.set(bed)
        - model.mg.geometry.depth.set(np.maximum(-bed, 0))
        - model.mg.rheology.B.set(...); .n.set(...); .eps_reg.set(...)
        - model.mg.sliding.beta.set(...); .m.set(...)
        - model.mg.forcing.smb.set(np.zeros((ny, nx)))  # overwritten each year
        - cells outside every glacier's footprint should end up with
          H = 0 (ice-free), same as ice-free bedrock/ocean cells in the
          Greenland/Antarctica examples

        Returns
        -------
        model : glide.model.IceDynamics
        grid_info : dict with 'x', 'y' (1-D cell-center coordinate arrays,
            meters, pulled from model.mg.levels[0].grid) and 'crs'
            (pyproj.CRS), used to regrid this region's PEBSI points.
        """
        raise NotImplementedError(
            f"_build_glide_model needs a bed-topography + initial "
            f"ice-thickness source for O2 region {region} "
            f"(glaciers: {gids_in_region}; see module docstring TODO #1)."
        )

    # ------------------------- per-year coupling ----------------------- #

    def couple_annual_step(self, terrain, point_attrs, annual_mb_mwe, year, dt_years=1.0):
        """
        Advance every O2 region's GLIDE model by one year using this
        year's PEBSI-computed surface mass balance, then push the
        updated geometry back onto PEBSI's points.

        Parameters
        ----------
        terrain : util.terrain.Terrain
            Current PEBSI terrain object (point coordinates + elevation).
        point_attrs : pebsi.state.PointAttributes
            Current PEBSI point attributes; `elevation` is replaced.
        annual_mb_mwe : (N_POINTS,) array
            Net surface mass balance per point for the year just run, in
            m w.e. (accumulation + refreeze - melt, summed over the year).
            Aggregating this from StepOutputs across chunks is the
            caller's (simulation.py's) responsibility.
        year : int
            Calendar year just completed; passed to GLIDE as `t`, in years.
            Must be used consistently across calls.
        dt_years : float
            Coupling interval in years (default 1.0, per the annual
            coupling decision this connector was designed around).

        Returns
        -------
        terrain : util.terrain.Terrain
            Same object, `elev_n` updated in place.
        point_attrs : pebsi.state.PointAttributes
            New NamedTuple with updated `elevation`.
        """
        new_elev_n = np.array(terrain.elev_n, copy=True)

        for region, gids_in_region in self.region_glaciers.items():
            region_idx = np.where(np.isin(terrain.rgiid_n, gids_in_region))[0]
            grid = self.grids[region]

            x_pts, y_pts = latlon_to_glacier_crs(
                terrain.lat_n[region_idx], terrain.lon_n[region_idx], grid["crs"]
            )

            smb_grid = self._smb_to_grid(region, x_pts, y_pts, annual_mb_mwe[region_idx])
            self._push_forcing(region, smb_grid)
            self._advance(region, t=float(year), dt=dt_years)

            new_surface = self._extract_surface_elevation(region)
            new_elev_n[region_idx] = grid_to_points(
                grid["x"], grid["y"], new_surface, x_pts, y_pts
            )

        terrain.elev_n = new_elev_n
        point_attrs = point_attrs._replace(elevation=jnp.array(new_elev_n))
        return terrain, point_attrs

    # ------------------------- GLIDE step pieces ------------------------ #

    def _smb_to_grid(self, region, x_pts, y_pts, mb_mwe_pts):
        """Regrid this year's point SMB (m w.e.) onto the region's GLIDE grid, as m ice yr^-1."""
        grid = self.grids[region]
        mb_ice_pts = mwe_to_ice_thickness(mb_mwe_pts)
        return points_to_grid(x_pts, y_pts, mb_ice_pts, grid["x"], grid["y"])

    def _push_forcing(self, region, smb_grid):
        self.models[region].mg.forcing.smb.set(smb_grid)

    def _advance(self, region, t, dt):
        """Run one GLIDE step covering `dt` years."""
        self.models[region].forward(cp.float32(t), cp.float32(dt))

    def _extract_surface_elevation(self, region):
        """Return updated surface elevation (bed + H) on this region's grid, as NumPy."""
        level = self.models[region].mg.levels[0]
        H = cp.asnumpy(level.state.H.data)
        bed = cp.asnumpy(level.geometry.bed.data)
        return bed + H
