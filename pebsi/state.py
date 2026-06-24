"""
State classes for PEBSI

Contains the following state trackers:
  - GlacierState: updated in the main time
    loop; contains all glacier attributes
  - ClimateState: contains temporal information
    and climatic forcings 
  - PointAttributes: contains time-invariant
    spatial information
  - StepOutputs: updated in the main time 
    loop; contains all output attributes
"""
# External libraries
from typing import NamedTuple
import jax.numpy as jnp

class GlacierState(NamedTuple):
    """
    An immutable, JAX-compatible snapshot of the physical state of every 
    point and layer in the simulation.

    Array dimensions:
        - 1D properties: (N_POINTS,)
        - 2D properties: (N_POINTS, N_LAYERS)
    """

    # ============================ Point Attributes ==========================
    # =============================== (N_POINTS,) ============================

    albedo: jnp.ndarray         # Albedo [-]
    albedo_surr: jnp.ndarray    # Albedo of off-glacier surroundings [-]
    surftemp: jnp.ndarray       # Surface temperature [C]
    roughness: jnp.ndarray      # Surface roughness [m]
    last_snow: jnp.ndarray      # Time index of last snowfall [-]

    # ------------------------------- Trackers -------------------------------
    annual_firn_converted: jnp.ndarray  # True when snow is converted to firn
    annual_min_albedo: jnp.ndarray      # Minimum albedo of each year for firn
    annual_max_snow: jnp.ndarray        # Maximum mass of snow reset annually
    days_since_snowfall: jnp.ndarray    # Days since last snowfall event
    delayed_snow: jnp.ndarray           # Snow that fell but wasn't yet added
    cum_mass_error: jnp.ndarray         # Mass error accumulator
    basal_reservoir: jnp.ndarray        # Reservoir for accumulated ice mass that
                                        # is pushed out of the layer domain
    past_snow: jnp.ndarray              # (N_POINTS, N* = (new_snow_days * 24))
                                        # Snow fallen in the past N* timesteps

    # ============================ Layer Attributes ==========================
    # ========================== (N_POINTS, N_LAYERS) ========================
    
    lheight: jnp.ndarray        # Layer height [m]
    ldepth: jnp.ndarray         # Depth of layer midpoint [m]
    snow_mask: jnp.ndarray      # Mask of snow layers
    firn_mask: jnp.ndarray      # Mask of firn layers
    ice_mask: jnp.ndarray       # Mask of ice layers

    # ------------------------- Intensive Properties -------------------------
    ldensity: jnp.ndarray       # Layer density [kg m-3]
    ltemp: jnp.ndarray          # Layer temperature [C]
    ltype: jnp.ndarray          # Layer type (0=snow, 1=firn, 2=ice)
    lage: jnp.ndarray           # Layer age [days]
    lgrainsize: jnp.ndarray     # Layer grain size [um]

    # ------------------------- Extensive Properties -------------------------
    lice: jnp.ndarray           # Solid ice content, incl. refreeze [kg m-2]
    lwater: jnp.ndarray         # Liquid water content [kg m-2]
    lrefreeze: jnp.ndarray      # Refreeze content [kg m-2]
    ldrefreeze: jnp.ndarray     # Refreeze generated in this timestep [kg m-2]
    lBC: jnp.ndarray            # Black carbon content [kg m-2]
    lOC: jnp.ndarray            # Organic carbon content [kg m-2]
    ldust: jnp.ndarray          # Dust content [kg m-2]

class ClimateState(NamedTuple):
    """
    An immutable, JAX-compatible snapshot of the climate state for every
    point in the simulation.

    Array dimensions: (N_TIME, N_POINTS)
        - 1D properties: (N_TIME,)
        - 2D properties: (N_TIME, N_POINTS)
    """

    # =============================== Time (1D) ===============================

    time_idx: jnp.ndarray       # Scalar integer tracking current time index
    year: jnp.ndarray           # Calendar year (e.g. 2026)
    month: jnp.ndarray          # Calendar month (1 to 12)
    day: jnp.ndarray            # Calendar day (1 to 31)
    hour: jnp.ndarray           # Calendar hour (0 to 23), UTC
    doy: jnp.ndarray            # Calendar day of year (0 to 366)

    local_hour: jnp.ndarray     # (2D) Calendar hour in local time (0 to 23)
    
    # ========================= Climate variables (2D) ========================

    tempC: jnp.ndarray          # 2 meter air temperature [C]
    tempK: jnp.ndarray          # 2 meter air temperature [K]
    tp: jnp.ndarray             # Total precipitation [m w.e.]
    prec: jnp.ndarray           # Precipitation rate [m w.e. s-1]
    wind: jnp.ndarray           # Wind speed [m s-1]
    winddir: jnp.ndarray        # Wind direction [deg]
    sp: jnp.ndarray             # Surface pressure [Pa]
    rh: jnp.ndarray             # 2 meter relative humidity [%]
    tcc: jnp.ndarray            # Total cloud cover [-]

    # ---------------------------- Radiation terms ----------------------------
    shortwave_in: jnp.ndarray   # Incoming shortwave radiation [J m-2]
    longwave_in: jnp.ndarray    # Incoming longwave radiation [J m-2]
    shadow_mask: jnp.ndarray    # Boolean shadow mask [-]
    solar_azimuth: jnp.ndarray  # Solar azimuth angle [rad]
    solar_zenith: jnp.ndarray   # Solar zenith angle [rad]

    # ---------------------------- Deposition terms ---------------------------
    bcdry: jnp.ndarray          # Dry black carbon deposition [kg m-2 s-1]
    bcwet: jnp.ndarray          # Wet black carbon deposition [kg m-2 s-1]
    ocdry: jnp.ndarray          # Dry organic carbon deposition [kg m-2 s-1]
    ocwet: jnp.ndarray          # Wet organic carbon deposition [kg m-2 s-1]
    dustdry: jnp.ndarray        # Dry dust deposition [kg m-2 s-1]
    dustwet: jnp.ndarray        # Wet dust deposition [kg m-2 s-1]

class PointAttributes(NamedTuple):
    """
    Time-invariant spatial attributes for each point
    
    Array dimensions: (N_POINTS,)
    """
    latitude: jnp.ndarray           # Point latitude [deg]
    longitude: jnp.ndarray          # Point longitude [deg, -180 to 180]
    elevation: jnp.ndarray          # Point elevation [m a.s.l.]
    slope: jnp.ndarray              # Point slope [deg]
    aspect: jnp.ndarray             # Point aspect [deg, 0=N]
    sky_view_factor: jnp.ndarray    # Sky-view factor [-]

class StepOutputs(NamedTuple):
    """
    Energy balance and mass fluxes for a single timestep at every point.

    Array dimensions:
        - 1D properties: (N_POINTS,)
        - 2D properties: (N_POINTS, N_LAYERS)
    """

    # =========================== Energy balance ==========================

    melt_energy: jnp.ndarray        # Energy available for melt [W m-2]
    shortwave_in: jnp.ndarray       # Incoming shortwave radiation [W m-2]
    shortwave_ref: jnp.ndarray      # Reflected shortwave radiation [W m-2]
    longwave_in: jnp.ndarray        # Incoming longwave radiation [W m-2]
    longwave_out: jnp.ndarray       # Emitted longwave radiation [W m-2]
    sensible_heat: jnp.ndarray      # Sensible heat flux [W m-2]
    latent_heat: jnp.ndarray        # Latent heat flux [W m-2]
    rain_heat: jnp.ndarray          # Rain heat flux [W m-2]
    ground_heat: jnp.ndarray        # Ground heat flux [W m-2]
    albedo: jnp.ndarray             # Broadband albedo [-]
    surftemp: jnp.ndarray           # Surface temperature

    # =========================== Mass balance ===========================

    error: jnp.ndarray              # Mass conservation error [m w.e.]
    melt: jnp.ndarray               # Melt [m w.e.]
    refreeze: jnp.ndarray           # Refreeze [m w.e.]
    accumulation: jnp.ndarray       # Accumulation [m w.e.]
    runoff: jnp.ndarray             # Runoff [m w.e.]
    rainfall: jnp.ndarray           # Rainfall [m w.e.]
    sublimation: jnp.ndarray        # Sublimation (ice->vapor) [m w.e.]
    deposition: jnp.ndarray         # Deposition (vapor->ice) [m w.e.]
    evaporation: jnp.ndarray        # Evaporation (water->vapor) [m w.e.]
    condensation: jnp.ndarray       # Condensation (vapor->water) [m w.e.]
    cumrefreeze: jnp.ndarray        # Refrozen mass in layers [m w.e.]

    # ============================= Climate ==============================
    airtemp: jnp.ndarray            # Air temperature [C]
    rh: jnp.ndarray                 # Relative humidity [%]
    wind: jnp.ndarray               # Wind speed [m s-1]
    winddir: jnp.ndarray            # Wind direction [deg]
    tp: jnp.ndarray                 # Precipitation [m]
    sp: jnp.ndarray                 # Surface pressure [Pa]

    # ============================== Layers ==============================
    layerheight: jnp.ndarray        # Layer height [m]
    layertemp: jnp.ndarray          # Layer temperature [C]
    layerwater: jnp.ndarray         # Layer water [kg m-2]
    layerdensity: jnp.ndarray       # Layer density [kg m-3]
    layerrefreeze: jnp.ndarray      # Layer refrozen mass [kg m-2]
    layergrainsize: jnp.ndarray     # Layer grainsize [um]
    layertype: jnp.ndarray          # Layer type [-]
    layerage: jnp.ndarray           # Layer age [days]
    layerBC: jnp.ndarray            # Layer BC concentration [ppb]
    layerOC: jnp.ndarray            # Layer OC concentration [ppm]
    layerdust: jnp.ndarray          # Layer dust concentration [ppm]