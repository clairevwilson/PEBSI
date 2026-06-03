import calendar
import datetime
import logging
from pathlib import Path

import numpy as np
import pytz
import xarray as xr
from pysolar.solar import get_altitude, get_azimuth
from tqdm import tqdm
try:
    import cupy as cp
    xp = cp
    use_gpu = True
except:
    xp = np
    use_gpu = False

class Shading:
    """
    Solar shading calculator using CUDA ray tracing (HIGHLY RECOMMENDED)
    or numpy (very slow). Needs a DEM as input from which slope, aspect,
    sky-view factor, and terrain shading are calculated.

    Also contains functions to apply shading and slope effects to a 
    shortwave dataset.
    """

    def __init__(self, dem, step_size=1.0, kernel_path=None):
        """
        Parameters
        ----------
        dem : xr.Dataset
            Dataset with an 'elevation' variable on a (y, x) grid.
        x_coord, y_coord : str
            Name of x and y coordinate in DEM
        step_size : float
            Step size in grid cells for ray tracing
        kernel_path : str
            Path to the azimuth_trace CUDA kernel source (.cu). 
            Ignored when running on CPU. Defaults to the bundled 
            kernel at shading/cuda/azimuth_trace.cu.
        """
        self.gpu = use_gpu
        
        self.step_size = step_size
        self.z = xp.array(dem.elevation.values, dtype=xp.float32)
        self.ny, self.nx = self.z.shape

        dx, dy = dem.rio.resolution()
        self.grid_resolution = (dy, dx)

        dZdy, dZdx = xp.gradient(self.z, dy, dx)
        self.dZdx = dZdx
        self.dZdy = -dZdy

        # CUDA kernel setup (GPU only)
        self.kernel = None
        if self.gpu:
            import cupy as cp
            if kernel_path is None:
                module_dir = Path(__file__).parent
                kernel_path = module_dir / "cuda" / "azimuth_trace.cu"
            else:
                kernel_path = Path(kernel_path)

            with open(kernel_path, "r") as f:
                kernel_code = f.read()
            kernels = cp.RawModule(code=kernel_code)
            self.kernel = kernels.get_function("azimuth_trace")
            self.block_size = (16, 16)
            self.grid_size = (self.nx // 16 + 1, self.ny // 16 + 1)
            self.step_size = cp.float32(step_size)

    def run_shadow_kernel(self, azimuth_deg):
        """
        Maximum terrain zenith angle along each ray for a given solar azimuth.
        On GPU, runs the CUDA azimuth_trace kernel. 
        On CPU, walks rays using numpy (slow).

        Returns
        -------
        max_zenith : xp.ndarray, shape (ny, nx)
            Maximum horizon elevation angle seen from
            each pixel in the given azimuth direction.
        """
        if self.gpu:
            import cupy as cp
            max_zenith = cp.zeros(self.z.shape, dtype=cp.float32)
            max_j = cp.zeros(self.z.shape, dtype=cp.uint32)
            max_i = cp.zeros(self.z.shape, dtype=cp.uint32)

            j_basis = cp.float32(np.sin(np.deg2rad(azimuth_deg)))
            i_basis = -cp.float32(np.cos(np.deg2rad(azimuth_deg)))

            self.nx = cp.int32(self.nx)
            self.ny = cp.int32(self.ny)

            self.kernel(
                self.grid_size,
                self.block_size,
                (max_zenith, max_j, max_i, self.z, j_basis, i_basis,
                 self.step_size, self.nx, self.ny),
            )
            return max_zenith
        
        else:
            # CPU fallback: simple scan along azimuth rays
            az_rad = np.deg2rad(azimuth_deg)
            dj = float(np.sin(az_rad))   # column step
            di = -float(np.cos(az_rad))  # row step (north = row 0)

            z_np = np.asarray(self.z)
            max_zenith = np.zeros((self.ny, self.nx), dtype=np.float32)

            for row in range(self.ny):
                for col in range(self.nx):
                    elev0 = z_np[row, col]
                    best = 0.0
                    dist = self.step_size
                    r, c = row + di * dist, col + dj * dist
                    while 0 <= int(r) < self.ny and 0 <= int(c) < self.nx:
                        elev = z_np[int(r), int(c)]
                        rise_run = (elev - elev0) / (dist * self.grid_resolution)
                        if rise_run > best:
                            best = rise_run
                        dist += self.step_size
                        r, c = row + di * dist, col + dj * dist
                    max_zenith[row, col] = best

            return xp.array(max_zenith, dtype=xp.float32)

    def horizon_zenith_angle(self, azimuth_deg):
        """Terrain horizon zenith angle in degrees. Shape (ny, nx)."""
        max_zenith = self.run_shadow_kernel(azimuth_deg)
        return xp.rad2deg(xp.arctan(max_zenith))

    def solar_position(self, dt):
        """Solar (altitude_deg, azimuth_deg) for a local datetime."""
        altitude = get_altitude(self.latitude, self.longitude, dt)
        azimuth = get_azimuth(self.latitude, self.longitude, dt)
        return altitude, azimuth

    def shadow_mask(self, altitude_deg, azimuth_deg):
        """
        Compute soft shadow mask for a single solar position.

        Parameters
        ==========
        altitude_deg : float
            Solar elevation angle above the horizon (degrees).
        azimuth_deg : float
            Solar azimuth angle (degrees, clockwise from north).

        Returns
        -------
        xp.ndarray, shape (ny, nx), values in [0, 1]
            0 = fully shadowed, 1 = fully sunlit.
        """
        zenith_deg = self.horizon_zenith_angle(azimuth_deg)
        z_i = altitude_deg - zenith_deg
        return 1.0 / (1.0 + xp.exp(-z_i / 0.1))

    def compute_shadow_masks(self, datetimes):
        """
        Compute shadow masks for a sequence of datetimes.

        This is the intended first step before feeding MERRA-2 radiation data.
        Results are keyed by datetime so they can be looked up efficiently in
        the second step.

        Parameters
        ==========
        datetimes : list of datetime.datetime
            Sequence of timezone-aware datetimes to evaluate. Typically every
            hour (or every MERRA-2 timestep) over your period of interest.

        Returns
        -------
        masks : dict
            xp.ndarray of shape (ny, nx) to index by datetime
        """
        ny, nx = self.z.shape
        total_steps = len(datetimes)
        
        # preallocate a single 3D array for all masks
        masks_cpu = np.zeros((total_steps, ny, nx), dtype=np.int8)

        for idx, dt in enumerate(tqdm(datetimes, desc="shadow masks", unit="step")):
            altitude, azimuth = self.solar_position(dt)
            if altitude <= 0:
                continue # sun below horizon, mask remains zero
            else:
                mask_gpu = self.shadow_mask(altitude, azimuth).astype(xp.int8)
                masks_cpu[idx] = mask_gpu.get() if hasattr(mask_gpu, 'get') else np.asarray(mask_gpu)

                # clear the gpu_mask from VRAM
                del mask_gpu
        return masks_cpu
    
    def skyviewfactor(self, num_azimuths = 16):
        """
        Calculates sky-view factor (integration of horizon angles)
        for each pixel.

        Parameters
        ==========
        num_azimuths : int
            Number of azimuth directions to sample (more = more accurate).

        Returns
        -------
        svf : xp.ndarray
            Sky-view factor on the grid (ny, nx)
        """
        svf = xp.zeros(self.z.shape, dtype=xp.float32)
        azimuths = np.linspace(0, 360, num_azimuths, endpoint=False)

        for az in tqdm(azimuths, desc="sky view factor", unit="az"):
            horizon_rad = xp.deg2rad(self.horizon_zenith_deg(az))
            svf += xp.cos(horizon_rad) ** 2

        return svf / num_azimuths

    def apply_merra2_radiation(self, shadow_masks, shortwave_data):
        """
        Apply shadow, slope, and sky-view factor corrections to
        MERRA-2 shortwave radiation data.

        Parameters
        ==========
        shadow_masks : dict
            Output of compute_shadow_masks(): maps datetime to shadow mask
            array of shape (ny, nx).
        shortwave_data : xr.DataArray
            MERRA-2 incident shortwave radiation dataset containing time
            dimension and spatial coordinates for the simulation points.

        Returns
        -------
        adjusted_shortwave: xp.ndarray
            Per-pixel radiation timeseries.
        """
        raise NotImplementedError(
            "apply_merra2_radiation() is a placeholder. "
            "Implement your slope/aspect correction and MERRA-2 scaling here."
        )