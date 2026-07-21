from pathlib import Path
from pyproj import Transformer

import numpy as np
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
    """

    def __init__(self, dem, step_size=1.0, kernel_path=None):
        """
        Parameters
        ==========
        dem : xr.Dataset
            Dataset with an 'elevation' variable on a (y, x) grid.
        x_coord, y_coord : str
            Name of x and y coordinate in DEM
        step_size : float
            Step size in grid cells for ray tracing
        kernel_path : str
            Path to the azimuth_trace CUDA kernel source (.cu). 
            Ignored when running on CPU. Defaults to the bundled 
            kernel at pebsi/shading/cuda/azimuth_trace.cu.
        """
        self.gpu = use_gpu

        # get center latitude and longitude of the DEM passed
        centroid = dem.rio.transform() * (dem.rio.width / 2, dem.rio.height / 2)
        transformer = Transformer.from_crs(dem.rio.crs, "EPSG:4326", always_xy=True)
        self.center_longitude, self.center_latitude = transformer.transform(*centroid)
        
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

            dj = cp.float32(np.sin(np.deg2rad(azimuth_deg)))
            di = -cp.float32(np.cos(np.deg2rad(azimuth_deg)))

            steps_j = int(self.nx / (abs(float(dj)) + 1e-6))
            steps_i = int(self.ny / (abs(float(di)) + 1e-6))
            max_steps = cp.int32(min(steps_j, steps_i))

            grid_res = self.grid_resolution
            self.grid_res_m = np.sqrt((di * grid_res[0])**2 + (dj * grid_res[1])**2)

            nx = cp.int32(self.nx)
            ny = cp.int32(self.ny)

            self.kernel(
                self.grid_size,
                self.block_size,
                (max_zenith, max_j, max_i, self.z, dj, di,
                 self.step_size, max_steps, nx, ny),
            )
            return max_zenith
        
        else:
            # CPU fallback: simple scan along azimuth rays
            az_rad = np.deg2rad(azimuth_deg)
            dj = float(np.sin(az_rad))   # column step
            di = -float(np.cos(az_rad))  # row step (north = row 0)

            grid_res = self.grid_resolution
            res = np.sqrt((di * grid_res[0])**2 + (dj * grid_res[1])**2)

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
                        rise_run = (elev - elev0) / (dist * res)
                        if rise_run > best:
                            best = rise_run
                        dist += self.step_size
                        r, c = row + di * dist, col + dj * dist
                    max_zenith[row, col] = best

            return xp.array(max_zenith, dtype=xp.float32)

    def horizon_elev_deg(self, azimuth_deg):
        """Terrain horizon elevation angle in degrees. Shape (ny, nx)."""
        max_elev = self.run_shadow_kernel(azimuth_deg)
        return xp.rad2deg(xp.arctan(max_elev / self.grid_res_m))

    def solar_position(self, dt):
        """Solar (altitude_deg, azimuth_deg) for a local datetime."""
        altitude = get_altitude(self.center_latitude, self.center_longitude, dt)
        azimuth = get_azimuth(self.center_latitude, self.center_longitude, dt)
        return altitude, azimuth

    def shadow_mask(self, altitude_deg, azimuth_deg):
        """
        Compute hard shadow mask for a single solar position.

        Parameters
        ==========
        altitude_deg : float
            Solar elevation angle above the horizon (degrees).
        azimuth_deg : float
            Solar azimuth angle (degrees, clockwise from north).

        Returns
        -------
        xp.ndarray, shape (ny, nx) dtype int
            0 = shadowed, 1 = sunlit.
        """
        elev_deg = self.horizon_elev_deg(azimuth_deg)
        return (altitude_deg > elev_deg).astype(xp.int8)

    def compute_shadow_masks(self, datetimes):
        """
        Compute shadow masks for a sequence of datetimes,
        which should be a leap year.

        Parameters
        ==========
        datetimes : list of datetime.datetime
            Sequence of timezone-aware datetimes in
            UTC to evaluate.

        Returns
        -------
        masks : xp.ndarray
            Shadow mask of shape (nt, ny, nx)
        sun_azimuth, sun_zenith : xp.ndarray
            Solar position for each hour in datetimes
            for the centerpoint of the grid
        sky_view : xp.ndarray
            Sky-view factor on the grid (ny, nx)
        """
        ny, nx = self.z.shape
        total_steps = len(datetimes)
        
        # preallocate a single 3D array for all masks
        masks_cpu = np.zeros((total_steps, ny, nx), dtype=np.int8)

        # allocate 1D arrays for the solar position
        sun_zenith = np.zeros(total_steps, dtype=np.float32)
        sun_azimuth = np.zeros(total_steps, dtype=np.float32)

        for idx, dt in enumerate(tqdm(datetimes, desc="shadow masks", unit="step")):
            altitude, azimuth = self.solar_position(dt)
            sun_zenith[idx] = np.radians(90.0 - altitude)
            sun_azimuth[idx] = np.radians(azimuth)


            if altitude <= 0:
                continue # sun below horizon, mask remains zero
            else:
                mask_gpu = self.shadow_mask(altitude, azimuth)
                masks_cpu[idx] = mask_gpu.get() if hasattr(mask_gpu, 'get') else np.asarray(mask_gpu)

                # clear the gpu_mask from VRAM
                del mask_gpu

        # calculate time-invariant sky-view factor 
        sky_view = self.skyviewfactor()
        return masks_cpu, sun_azimuth, sun_zenith, sky_view
    
    def skyviewfactor(self, num_azimuths = 64):
        """
        Calculates sky-view factor for each pixel.
        svf = cos^2(horizon angle / 

        Parameters
        ==========
        num_azimuths : int
            Number of azimuth directions to sample 
            (more azimuths = more accurate).

        Returns
        -------
        svf : xp.ndarray
            Sky-view factor on the grid (ny, nx)
        """
        sum_sin2 = xp.zeros(self.z.shape, dtype=xp.float32)
        azimuths = np.linspace(0, 360, num_azimuths, endpoint=False)
        azimuths += np.random.uniform(0, 360 / num_azimuths)

        for az in azimuths:
            horizon_elev_rad =  xp.deg2rad(self.horizon_elev_deg(az))
            sum_sin2 += xp.sin(horizon_elev_rad)**2

        # final expression: normalize by n_azimuths
        return 1 - sum_sin2 / num_azimuths