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

Bed and initial ice thickness are taken straight from the PEBSI
terrain object, which loads bed elevation from the ice thickness 
and DEM inputs. All of GLIDE's physical parameters are also taken
from the PEBSI config.

Since GLIDE gives a total mass change per point, but PEBSI has already
accounted for SMB mass change, this is removed from the GLIDE output
to provide the dynamics-only change in mass.
"""

import numpy as np
import cupy as cp
import jax.numpy as jnp
from scipy.interpolate import griddata, RegularGridInterpolator
from pyproj import Transformer
import sys, os
sys.path.append(os.path.join(os.getcwd(), '../glide/'))
from glide.model import IceDynamics
from pebsi.physics.layers import apply_dynamics_mass_change

import jax

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


def mwe_to_ice_thickness(mb_mwe, density_ice, density_water):
    """
    Converts net surface mass balance (m water equivalent)
    to ice-equivalent thickness (m).
    """
    return mb_mwe * density_water / density_ice


# --------------------------------------------------------------------- #
#                              GlideCoupler
# --------------------------------------------------------------------- #

class GlideCoupler:
    """
    Owns one GLIDE IceDynamics model per RGI O2 sub-region (a cluster of
    spatially-grouped glaciers) and drives annual coupling steps between
    PEBSI and GLIDE.

    Parameters
    ----------
    terrain : pebsi.io.terrain.Terrain
        Must already have bed_n/thickness_n (see module docstring),
        i.e. built with params.option_dynamics True.
    """

    def __init__(self, terrain, params):
        self.terrain = terrain
        self.params = params
        self.n_levels = params.dynamics_n_levels
        self.dx = params.dynamics_dx
        self.margin = params.dynamics_margin
        self.rheology_B = params.dynamics_rheology_B
        self.rheology_n = params.dynamics_rheology_n
        self.rheology_eps_reg = params.dynamics_rheology_eps_reg
        self.sliding_beta = params.dynamics_sliding_beta
        self.sliding_m = params.dynamics_sliding_m

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
        (pebsi/io/terrain.py -> self.rgi_df, read from
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
        covering every glacier in O2 sub-region `region`. Bed and initial
        ice thickness come from terrain.bed_n / terrain.thickness_n (see
        module docstring), regridded here.

        Returns
        -------
        model : glide.model.IceDynamics
        grid_info : dict with 'x', 'y' (1-D cell-center coordinate arrays,
            meters, pulled from model.mg.levels[0].grid) and 'crs'
            (pyproj.CRS), used to regrid this region's PEBSI points.
        """
        terrain = self.terrain
        dx = self.dx

        # union bounding box + a shared metric CRS for this region
        ids_fmtd = ['RGI60-' + gid for gid in gids_in_region]
        region_gdf = terrain.rgi_gdf[terrain.rgi_gdf['RGIId'].isin(ids_fmtd)]
        crs = terrain._get_metric_crs(region_gdf)
        region_gdf = region_gdf.to_crs(crs)

        minx, miny, maxx, maxy = region_gdf.total_bounds
        minx -= self.margin
        miny -= self.margin
        maxx += self.margin
        maxy += self.margin

        # round grid dims up to a multiple of 2 ** (n_levels - 1)
        pad = 2 ** (self.n_levels - 1)
        nx = max(1, int(np.ceil((maxx - minx) / dx)))
        ny = max(1, int(np.ceil((maxy - miny) / dx)))
        nx += (-nx) % pad
        ny += (-ny) % pad

        # x0/y0 are the first cell-center coords (glide.grid.Grid: x
        # increases from x0, y *decreases* from y0 -- top row first)
        x0 = cp.float32(minx + dx / 2)
        y0 = cp.float32(maxy - dx / 2)

        model = IceDynamics(n_levels=self.n_levels, ny=ny, nx=nx, dx=cp.float32(dx),
                             x0=x0, y0=y0, crs=crs)

        # pull the grid's actual coordinates back out so everything downstream
        # uses exactly what GLIDE built, rather than a separately-computed copy
        x = cp.asnumpy(model.mg.levels[0].x_cell)
        y = cp.asnumpy(model.mg.levels[0].y_cell)

        region_idx = np.where(np.isin(terrain.rgiid_n, gids_in_region))[0]
        x_pts, y_pts = latlon_to_glacier_crs(
            terrain.lat_n[region_idx], terrain.lon_n[region_idx], crs
        )

        bed = points_to_grid(x_pts, y_pts, terrain.bed_n[region_idx], x, y)
        thickness = points_to_grid(x_pts, y_pts, terrain.thickness_n[region_idx], x, y)
        thickness = np.maximum(thickness, 0.0)

        model.mg.state.H.set(thickness)
        model.mg.state.H_prev.set(thickness)
        model.mg.geometry.bed.set(bed)
        model.mg.geometry.depth.set(np.maximum(-bed, 0))

        model.mg.rheology.B.set(cp.full((ny, nx), self.rheology_B, dtype=cp.float32))
        model.mg.rheology.n.set(self.rheology_n)
        model.mg.rheology.eps_reg.set(self.rheology_eps_reg)

        model.mg.sliding.beta.set(cp.full((ny, nx), self.sliding_beta, dtype=cp.float32))
        model.mg.sliding.m.set(self.sliding_m)

        model.mg.forcing.smb.set(np.zeros((ny, nx), dtype=np.float32))

        grid_info = {'x': x, 'y': y, 'crs': crs}
        return model, grid_info

    # -------------------------- period coupling -------------------------- #

    def couple_step(self, terrain, point_attrs, state, period_mb_mwe, t_years, dt_years):
        """
        Advance every O2 region's GLIDE model by `dt_years` using PEBSI's
        surface mass balance accumulated over that period, then push the
        updated geometry and ice-dynamics mass changes back onto PEBSI.

        The coupling period is independent of temporal_chunks and may
        span multiple chunks or vice versa -- see simulation.py's main
        loop, which accumulates period_mb_mwe/dt_years across chunks
        until dynamics_period_years worth of time has passed.

        Parameters
        ----------
        terrain : pebsi.io.terrain.Terrain
            Current PEBSI terrain object (point coordinates + elevation).
        point_attrs : pebsi.state.PointAttributes
            Current PEBSI point attributes; `elevation` is replaced.
        state : pebsi.state.GlacierState
            Current PEBSI glacier state
        period_mb_mwe : (N_POINTS,) array
            Net surface mass balance per point accumulated over the
            whole `dt_years`-long coupling period, in m w.e.
            (accumulation + refreeze - melt, summed over that period).
            Aggregating this from StepOutputs across chunks is the
            caller's (simulation.py's) responsibility.
        t_years : float
            Cumulative simulated time elapsed so far; passed to GLIDE as
            `t`. Must be used consistently across calls (i.e. always
            incremented by the `dt_years` actually used each call).
        dt_years : float
            Length of the period `period_mb_mwe` covers, in years (>= 1;
            see pebsi.defaults.dynamics_period_years).

        Returns
        -------
        terrain : pebsi.io.terrain.Terrain
            Same object, `elev_n` updated in place.
        point_attrs : pebsi.state.PointAttributes
            New NamedTuple with updated `elevation`.
        state : pebsi.state.GlacierState
            New NamedTuple with updated basal_reservoir and ice mass.
        """
        N_POINTS = terrain.N_POINTS
        new_elev_n = np.array(terrain.elev_n, copy=True)

        for region, gids_in_region in self.region_glaciers.items():
            region_idx = np.where(np.isin(terrain.rgiid_n, gids_in_region))[0]
            grid = self.grids[region]

            x_pts, y_pts = latlon_to_glacier_crs(
                terrain.lat_n[region_idx], terrain.lon_n[region_idx], grid["crs"]
            )

            # ice thickness before this step's dynamics
            H_old_grid = self._extract_ice_thickness(region)
            H_old_pts = grid_to_points(grid["x"], grid["y"], H_old_grid, x_pts, y_pts)

            # push this period's SMB as an annual RATE (GLIDE scales it by dt
            # internally) and advance -- period_mb_mwe is a TOTAL over
            # dt_years, so it must be divided down to a rate first
            mb_rate_mwe = period_mb_mwe[region_idx] / dt_years  # m w.e. yr^-1
            smb_ice_rate = mwe_to_ice_thickness(
                mb_rate_mwe, self.params.density_ice, self.params.density_water
            )  # m ice yr^-1
            smb_grid = points_to_grid(x_pts, y_pts, smb_ice_rate, grid["x"], grid["y"])
            self._push_forcing(region, smb_grid)
            self._advance(region, t=float(t_years), dt=dt_years)

            # ice thickness / surface after this step's dynamics
            H_new_grid = self._extract_ice_thickness(region)
            H_new_pts = grid_to_points(grid["x"], grid["y"], H_new_grid, x_pts, y_pts)

            new_surface = self._extract_surface_elevation(region, H_grid=H_new_grid)
            new_elev_n[region_idx] = grid_to_points(
                grid["x"], grid["y"], new_surface, x_pts, y_pts
            )

            # dynamics-only mass change: total change minus what SMB alone
            # already explains over the whole period (PEBSI's own layers
            # already applied the SMB part -- see module docstring)
            dH_total = H_new_pts - H_old_pts
            dH_smb = smb_ice_rate * dt_years
            dH_dynamics = dH_total - dH_smb
            dmass_pts = dH_dynamics * self.params.density_ice  # kg m-2

            mask_full = np.zeros(N_POINTS, dtype=bool)
            mask_full[region_idx] = True
            dmass_full = np.zeros(N_POINTS)
            dmass_full[region_idx] = dmass_pts

            state = apply_dynamics_mass_change(
                state, jnp.array(mask_full), jnp.array(dmass_full), self.params
            )
            jax.debug.print('dmass: {}', jnp.array(dmass_full))

        terrain.elev_n = new_elev_n
        point_attrs = point_attrs._replace(elevation=jnp.array(new_elev_n))
        return terrain, point_attrs, state

    # ------------------------- GLIDE step pieces ------------------------ #

    def _push_forcing(self, region, smb_grid):
        self.models[region].mg.forcing.smb.set(smb_grid)

    def _advance(self, region, t, dt):
        """Run one GLIDE step covering `dt` years."""
        self.models[region].forward(cp.float32(t), cp.float32(dt))

    def _extract_ice_thickness(self, region):
        """Return this region's current ice thickness (H), as NumPy."""
        return cp.asnumpy(self.models[region].mg.levels[0].state.H.data)

    def _extract_surface_elevation(self, region, H_grid=None):
        """Return updated surface elevation (bed + H) on this region's grid, as NumPy."""
        if H_grid is None:
            H_grid = self._extract_ice_thickness(region)
        bed = cp.asnumpy(self.models[region].mg.levels[0].geometry.bed.data)
        return bed + H_grid
