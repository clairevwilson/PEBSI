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
    
    # --- Grid Tracking ---
    step_idx: jnp.ndarray       # Scalar integer tracking current time index

    # ============================ Point Attributes ==========================
    # =============================== (N_POINTS,) ============================

    albedo: jnp.ndarray         # Albedo [-]
    albedo_surr: jnp.ndarray    # Albedo of off-glacier surroundings [-]
    surftemp: jnp.ndarray       # Surface temperature [C]

    # ------------------------------- Trackers -------------------------------
    annual_firn_converted: jnp.ndarray      # True when snow is converted to firn
    annual_min_albedo: jnp.ndarray          # Minimum albedo reset annually
    annual_max_snow: jnp.ndarray            # Maximum mass of snow reset annually
    days_since_snowfall: jnp.ndarray        # Days since last snowfall event
    delayed_snow: jnp.ndarray               # Snow that fell but wasn't yet added

    # ============================ Layer Attributes ==========================
    # ========================== (N_POINTS, N_LAYERS) ========================
    
    lheight: jnp.ndarray        # Layer height [m]
    ldepth: jnp.ndarray         # Depth of layer midpoint [m]

    # ------------------------- Intensive Properties -------------------------
    ldensity: jnp.ndarray       # Layer density [kg m-3]
    ltemp: jnp.ndarray          # Layer temperature [C]
    ltype: jnp.ndarray          # Layer type (0=snow, 1=firn, 2=ice)
    lage: jnp.ndarray           # Layer age [days]
    lgrainsize: jnp.ndarray     # Layer grain size [um]

    # ------------------------- Extensive Properties -------------------------
    lice: jnp.ndarray           # Solid ice content, including refreeze [kg m-2]
    lwater: jnp.ndarray         # Liquid water content [kg m-2]
    lrefreeze: jnp.ndarray      # Refreeze content [kg m-2]
    dlrefreeze: jnp.ndarray     # Refreeze generated in current timestep [kg m-2]
    lBC: jnp.ndarray            # Black carbon content [kg m-2]
    lOC: jnp.ndarray            # Organic carbon content [kg m-2]
    ldust: jnp.ndarray          # Dust content [kg m-2]

class ClimateState(NamedTuple):
    """
    An immutable, JAX-compatible snapshot of the climate state for every
    point in the simulation.

    Array dimensions: (N_POINTS, N_TIME)
    """
    
    # =========================== Climate variables ==========================

    tempC: jnp.ndarray          # 2 meter air temperature [C]
    tempK: jnp.ndarray          # 2 meter air temperature [K]
    tp: jnp.ndarray             # Total precipitation [m w.e.]
    prec: jnp.ndarray           # Precipitation rate [m w.e. s-1]
    wind: jnp.ndarray           # Wind speed [m s-1]
    winddir: jnp.ndarray        # Wind direction [deg]
    sp: jnp.ndarray             # Surface pressure [Pa]
    rh: jnp.ndarray             # 2 meter relative humidity [%]
    tcc: jnp.ndarray            # Total cloud cover [-]

    # --------------------------- Radiation terms ---------------------------
    shortwave_in: jnp.ndarray   # Incoming shortwave radiation [J m-2]
    longwave_in: jnp.ndarray    # Incoming longwave radiation [J m-2]
    shadow_mask: jnp.ndarray    # Boolean shadow mask [-]

    # --------------------------- Deposition terms --------------------------
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
    elevation: jnp.ndarray
    slope: jnp.ndarray
    aspect: jnp.ndarray
    timezone: jnp.ndarray

class StepOutputs(NamedTuple):
    """
    Energy balance and mass fluxes for a single timestep at every point.

    Array dimensions: (N_POINTS,)
    """

    # =========================== Energy balance ==========================

    net_radiation: jnp.ndarray      # Net radiation [W m-2]
    net_shortwave: jnp.ndarray      # Net shortwave radiation [W m-2]
    net_longwave: jnp.ndarray       # Net longwave radiation [W m-2]
    shortwave_down: jnp.ndarray     # Incoming shortwave radiation [W m-2]
    shortwave_up: jnp.ndarray       # Reflected shortwave radiation [W m-2]
    longwave_down: jnp.ndarray      # Incoming longwave radiation [W m-2]
    longwave_up: jnp.ndarray        # Emitted longwave radiation [W m-2]
    sensible_heat: jnp.ndarray      # Sensible heat flux [W m-2]
    latent_heat: jnp.ndarray        # Latent heat flux [W m-2]
    ground_heat: jnp.ndarray        # Ground heat flux [W m-2]
    melt_energy: jnp.ndarray        # Energy available for melt [W m-2]

    # =========================== Mass balance ===========================

    melt: jnp.ndarray               # Melt [m w.e.]
    refreeze: jnp.ndarray           # Refreeze [m w.e.]
    accumulation: jnp.ndarray       # Accumulation [m w.e.]
    sublimation: jnp.ndarray        # Sublimation (ice-->vapor) [m w.e.]
    deposition: jnp.ndarray         # Deposition (vapor-->ice) [m w.e.]
    evaporation: jnp.ndarray        # Evaporation (water-->vapor) [m w.e.]
    condensation: jnp.ndarray       # Condensation (vapor-->water) [m w.e.]