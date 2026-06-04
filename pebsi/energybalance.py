"""
Energy balance class for PEBSI

Loads climate variables for each timestep
and calculates the surface energy balance
from individual heat fluxes.

@author: clairevwilson
"""
# External libraries
import jax
import jax.numpy as jnp

class EnergyBalanceDriver():
    def __init__(self, params, static_args):
        """
        Stores parameters and physical constants
        for accessing within mass balance functions.
        """
        self.prms = params 
        self.args = static_args

    def solve_energy_balance(self, state, forcings, point_attrs):
        """
        Finds the exact surftemp where compute_fluxes 
        equals 0. Runs inside the spatial vmap.
        """
        # 1. check if the melt energy is positive with 0 surface temperature
        t_melt = 0.0
        y_melt, _ = self.compute_fluxes(t_melt, state, forcings, point_attrs)

        # 2. solve for the root of the energy balance equation
        # initialize two starting guesses
        t0 = state.surftemp     # previous surface temperature
        t1 = forcings.tempC     # current air temperature

        # evaluate the initial energy balance residuals for both guesses
        y0, _ = self.compute_fluxes(t0, state, forcings, point_attrs)
        y1, _ = self.compute_fluxes(t1, state, forcings, point_attrs)

        # define the internal loop step that JAX will execute
        def secant_step(carry, _):
            # unpack the current state of the root finder
            t_prev, t_curr, y_prev, y_curr = carry
            
            # calculate the denominator with divide-by-zero safeguard
            denom = jnp.where(jnp.abs(y_curr - y_prev) < 1e-4, 1e-4, y_curr - y_prev)
            
            # standard Secant formula to calculate the next temperature guess
            t_next = t_curr - y_curr * (t_curr - t_prev) / denom
            
            # evaluate the new energy balance residual with the updated temperature
            y_next, _ = self.compute_fluxes(t_next, state, forcings, point_attrs)
            
            # pass the updated values forward to the next iteration
            return (t_curr, t_next, y_curr, y_next), None

        # run a strict, fixed-length loop of 10 steps using jax.lax.scan
        initial_carry = (t0, t1, y0, y1)
        final_carry, _ = jax.lax.scan(secant_step, initial_carry, xs=None, length=8)
        surftemp_cooling = final_carry[1]

        # 3. extract final results for melting or cooling case
        is_melting = y_melt > 0.0
        surftemp_final = jnp.where(is_melting, t_melt, surftemp_cooling)

        # extract our final, optimized results
        _, fluxes = self.compute_fluxes(surftemp_final, state, forcings, point_attrs)
        
        updated_state = state._replace(surftemp=surftemp_final)
        return updated_state, fluxes
    
    def compute_fluxes(self, surftemp_guess, state, forcings, point_attrs):
        """
        Calculates net energy balance [W m-2] for a 
        guessed surface temperature (for use in secant
        root-finder.)
        """
        guessed_state = state._replace(surftemp=surftemp_guess)

        # SHORTWAVE RADIATION
        SWin, SWref = self.get_SW(guessed_state, forcings, point_attrs)
        SWnet = SWin + SWref

        # handle penetrating shortwave
        if self.args.option_SWpen:
            frac_abs_surf = jnp.where(
                guessed_state.ltype[:, 0] == 0,
                self.args.frac_absrad_snow,
                self.args.frac_absrad_ice
            )
        else:
            frac_absrad = 1
        SWnet_surf = SWnet * frac_abs_surf 
        SWnet_pen = SWnet * (1 - frac_abs_surf)
                    
        # LONGWAVE RADIATION
        LWin, LWout = self.get_LW(guessed_state, forcings)
        LWnet = LWin + LWout

        # RAIN FLUX (Qp)
        Qp = self.get_rain(guessed_state, forcings)

        # GROUND FLUX (Qg)
        Qg = self.get_ground(guessed_state)

        # TURBULENT FLUXES (Qs and Ql)
        Qs, Ql = self.get_turbulent(guessed_state, forcings, point_attrs)

        # OUTPUTS
        Qm = SWnet_surf + LWnet + Qp + Qs + Ql + Qg
        fluxes = {
            'SWin': SWin, 'SWref': SWref,
            'LWin': LWin, 'LWout': LWout,
            'sensible_heat': Qs, 'latent_heat': Ql,
            'rain_heat': Qp, 'ground_heat': Qg,
            'melt_heat': Qm, 'SWnet_surface': SWnet_surf,
            'SWnet_penetrating': SWnet_pen
        }
        return Qm, fluxes
    
    def get_SW(self, state, forcings, point_attrs):
        """
        Calculates incoming and outgoing shortwave heat
        flux accounting for:
        - Albedo
        - Shading
        - Topographic effects (diffuse radiation; slope)
        """
        # CONSTANTS
        sky_view = point_attrs.sky_view_factor
        slope = point_attrs.slope * jnp.pi/180
        aspect = point_attrs.aspect * jnp.pi/180
        dt = self.args.dt

        # albedo input
        albedo = state.albedo
        albedo_surr = state.albedo_surr

        # get solar position and shading
        sun_az = forcings.solar_azimuth
        sun_zen = forcings.solar_zenith
        shade = forcings.shadow_mask

        # calculate slope correction
        cos_theta = (jnp.cos(sun_zen)*jnp.cos(slope) + 
                    jnp.sin(sun_zen)*jnp.sin(slope)*jnp.cos(sun_az - aspect))
        slope_correction = jnp.clip(cos_theta / jnp.cos(sun_zen), 0, 5)
        
        # SWin needs to be corrected for shade
        # get sky (diffuse+direct) and terrain (diffuse) SWin
        SWin_sky = forcings.shortwave_in / dt
        SWin_terrain = SWin_sky * (1-sky_view) * albedo_surr

        # split sky into direct and diffuse
        f_diff = self.diffuse_fraction(SWin_sky, sun_zen, forcings.doy)
        SWin_direct = SWin_sky * (1-f_diff)
        SWin_diffuse = SWin_sky * f_diff * sky_view

        # determine overall incoming flux
        SWin = jnp.where(
            shade,
            SWin_terrain + SWin_diffuse,
            SWin_terrain + SWin_diffuse + SWin_direct * slope_correction
        )

        # get reflected radiation
        SWref = SWin * albedo * -1
        return SWin, SWref

    def get_LW(self, state, forcings):
        """
        Calculates incoming and outgoing longwave heat
        flux. If not input in climate data, scheme follows 
        Klok and Oerlemans (2002) for calculating net 
        longwave radiation from the air temperature
        and cloud cover.
        
        Parameters
        ==========
        surftemp : float
            Surface temperature [C]
        """
        args = self.args

        # CONSTANTS
        SIGMA_SB = args.sigma_SB
        CTOK = args.celsius_to_kelvin
        EPS = args.surf_emissivity

        # unpack surface temperature 
        surftemp = state.surftemp

        # calculate LWout from surftemp
        surftempK = surftemp + CTOK
        LWout = -1 * EPS * SIGMA_SB * surftempK**4
        
        # pull LWin straight from data
        LWin = forcings.longwave_in / args.dt
            
        return LWin, LWout
    
    def get_rain(self, state, forcings):
        """
        Calculates amount of energy supplied by
        precipitation that falls as rain.
        
        Parameters
        ==========
        surftemp : float
            Surface temperature [C]
        """
      
        # CONSTANTS
        SNOW_THRESHOLD_LOW = self.args.snow_threshold_low
        SNOW_THRESHOLD_HIGH = self.args.snow_threshold_high
        DENSITY_WATER = self.args.density_water
        CP_WATER = self.args.Cp_water

        # unpack climate variables
        airtemp = forcings.tempC
        surftemp = state.surftemp 
        precip_rate = forcings.prec

        # define rain vs snow scaling
        rain_scale = jnp.linspace(0,1,20)
        temp_scale = jnp.linspace(SNOW_THRESHOLD_LOW,SNOW_THRESHOLD_HIGH,20)
        
        # get fraction of precip that is rain
        frac_rain = jnp.interp(forcings.tempC, temp_scale, rain_scale)

        Qp = (airtemp - surftemp)*precip_rate*frac_rain*DENSITY_WATER*CP_WATER
        return Qp
    
    def get_ground(self, state):
        """
        Calculates amount of energy supplied to the surface
        by heat conduction from the temperate ice.
        
        Parameters
        ==========
        surftemp : float
            Surface temperature [C]
        """

        # CONSTANTS
        K_ICE = self.args.k_ice
        TEMP_TEMP = self.args.temp_temp 
        TEMP_DEPTH = self.args.temp_depth
        
        # calculate ground flux from surface temperature
        Qg = -1 * K_ICE * (state.surftemp - TEMP_TEMP) / TEMP_DEPTH
        return Qg
    
    def get_turbulent(self, state, forcings, point_attrs):
        """
        Calculates turbulent (sensible and latent heat)
        fluxes based on Monin-Obukhov Similarity Theory 
        or Bulk Richardson number.

        Parameters
        ==========
        surftemp : float
            Surface temperature [C]
        roughness : float
            Surface roughness [m]
        """
        args = self.args 

        # CONSTANTS
        KARMAN = args.karman
        GRAVITY = args.gravity
        R_GAS = args.R_gas
        MM_AIR = args.molarmass_air
        CP_AIR = args.Cp_air
        WIND_REF_Z = args.wind_ref_height

        # spatial attributes
        slope = point_attrs.slope
        surftemp = state.surftemp
        z0 = state.roughness  # Roughness length for momentum
        z0t = z0/100          # Roughness length for heat
        z0q = z0/10           # Roughness length for moisture

        # adjust wind speed to reference height
        z = 2 # reference height in m
        if WIND_REF_Z != 2:
            wind_2m = forcings.wind * jnp.log(2/z0) / jnp.log(WIND_REF_Z/z0)
        else:
            wind_2m = forcings.wind

        # transform humidity into mixing ratio (q) 
        Ewz = self.sat_vapor_pressure(forcings.tempC)  # saturation vapor pressure at 2m
        Ew0 = self.sat_vapor_pressure(surftemp)  # saturation vapor pressure at the surface
        
        qz = (forcings.rh / 100) * 0.622 * (Ewz / (forcings.sp - Ewz))
        q0 = 1.0 * 0.622 * (Ew0 / (forcings.sp - Ew0))

        # get air density from PV=nRT
        density_air = forcings.sp / R_GAS / forcings.tempK * MM_AIR

        # latent heat term depends on direction of heat exchange
        is_sublimating = (surftemp == 0.0) & ((qz - q0) > 0.0)
        Lv = jnp.where(is_sublimating, args.Lv_evap, args.Lv_sub)

        # calculate richardson number
        safe_wind_sq = jnp.where(wind_2m == 0.0, 1e-5, wind_2m ** 2)
        RICHARDSON = (GRAVITY / forcings.tempK) * (forcings.tempC - surftemp) \
                                                    * (z - z0) / safe_wind_sq
        # override Richardson to 0 if wind was actually 0
        RICHARDSON = jnp.where(wind_2m == 0.0, 0.0, RICHARDSON)

        # calculate stability coefficients
        csT = KARMAN ** 2 / (jnp.log(z / z0) * jnp.log(z / z0t))
        csQ = KARMAN ** 2 / (jnp.log(z / z0) * jnp.log(z / z0q))

        # Psi stability factor (Beljaars & Holtslag)
        psi = jnp.where(
            RICHARDSON <= 0.0,
            (1.0 - 15.0 * RICHARDSON) ** 0.5,  # Unstable branch
            jnp.exp(-5.0 * RICHARDSON)         # Stable branch
        )
        
        # final flux calculation
        Qs = density_air * CP_AIR * csT * psi * wind_2m * (forcings.tempC - surftemp) * jnp.cos(slope)
        Ql = density_air * Lv * csQ * psi * wind_2m * (qz - q0) * jnp.cos(slope)

        return Qs, Ql
    
    def sat_vapor_pressure(self, airtemp, method='ARM'):
        """
        Calculates vapor pressure [Pa] 
        from air temperature 

        Parameters
        ==========
        airtemp : float
            Air temperature [C]
        """
        # CONSTANTS
        CTOK = self.args.celsius_to_kelvin

        # calculate saturation vapor pressure in kPa
        if method in ['ARM']:
            P = 0.61094*jnp.exp(17.625*airtemp/(airtemp+243.04)) # kPa
        elif method in ['Sonntag']:
            # follows COSIPY
            airtempK = airtemp + CTOK
            P = jnp.where(
                airtempK > CTOK,
                0.6112*jnp.exp(17.67*(airtemp-CTOK)/(airtemp-29.66)), # over water
                0.6112*jnp.exp(22.46*(airtemp-CTOK)/(airtemp-0.55)) # over ice
            )

        # return vapor pressure in Pa
        return P*1000

    def diffuse_fraction(self,rad_glob,solar_zenith,doy):
        """
        Determines the fraction shortwave radiation 
        that is diffuse using an empirical formulation 
        from the clearness index, which is the ratio of 
        horizontal global radiation to potential 
        (extraterrestrial) radiation.

        Based on Wohlfahrt (2016) Appendix C 
        (10.1016/j.agrformet.2016.05.012)

        Parameters
        ==========
        rad_glob : float
            Horizontal global (all-sky) radiation [W m-2]
        solar_zenith : float
            Solar zenith angle [rad]
        """
        # CONSTANTS
        SOLAR_CONSTANT = self.args.solar_constant
        P1 = 0.1001
        P2 = 4.7930
        P3 = 9.4758
        P4 = 0.2465

        # calculate potential (extraterrestrial) shortwave radiation
        doy_term = jnp.cos(2*jnp.pi*doy/365.25)
        zen_term = jnp.cos(solar_zenith)
        rad_pot = SOLAR_CONSTANT * (1 + 0.033 * doy_term) * zen_term

        # determine clearness index
        CI = rad_glob / rad_pot
        CI = jnp.clip(CI, 0, 1)

        # empirical relationship for diffuse fraction
        diffuse_fraction = jnp.exp(-jnp.exp(P1-(P2-P3*CI)))*(1-P4)+P4
        return diffuse_fraction