"""
Energy balance class for PEBSI

Loads climate variables for each timestep
and calculates the surface energy balance
from individual heat fluxes.

@author: clairevwilson
"""
# External libraries
import pandas as pd
import numpy as np
import suncalc
# Internal libraries
from util.config import ConfigError

class energyBalance():
    """
    Energy balance scheme that calculates the surface 
    energy balance and penetrating shortwave radiation. 
    This class is updated within main() every timestep, 
    so it stores the climate data and surface fluxes 
    for a single timestep.
    """ 
    def __init__(self,massbal,timestamp):
        """
        Loads in the climate data at a given timestep 
        to use in the surface energy balance.

        Parameters
        ==========
        climateds : xr.Dataset
            Climate dataset containing meteorological
            inputs (temperature, wind speed, etc.)
        timestamp : pd.Datetime
            Timestamp to index the climate dataset.
        args : command-line arguments
        """
        # pull other classes from mass balance class
        climate = massbal.climate
        args = massbal.args
        layers = massbal.layers
        surface = massbal.surface

        # CONSTANTS
        SPH = args.seconds_per_hour
        CTOK = args.celsius_to_kelvin

        # unpack climate variables
        time_idx = climate.get_idx(timestamp)
        # climateds_now = climate.cds.isel(time=time_idx)
        self.tempC = climate.data['temp'][time_idx]
        self.tp = climate.data['tp'][time_idx]
        self.sp = climate.data['sp'][time_idx]
        self.rh = climate.data['rh'][time_idx]
        self.wind = climate.data['wind'][time_idx]
        self.tcc = climate.data['tcc'][time_idx]
        self.SWin_ds = climate.data['SWin'][time_idx]
        self.SWout_ds = climate.data['SWout'][time_idx]
        self.albedo_ds = climate.data['albedo'][time_idx]
        self.LWin_ds = climate.data['LWin'][time_idx]
        self.LWout_ds = climate.data['LWout'][time_idx]
        self.NR_ds = climate.data['NR'][time_idx]
        self.bcdry = climate.data['bcdry'][time_idx]
        self.bcwet = climate.data['bcwet'][time_idx]
        self.ocdry = climate.data['ocdry'][time_idx]
        self.ocwet = climate.data['ocwet'][time_idx]
        self.dustdry = climate.data['dustdry'][time_idx]
        self.dustwet = climate.data['dustwet'][time_idx]

        # time
        self.timestamp = timestamp
        self.dt = args.dt
        self.iters = 0

        # store previous timestep incoming shortwave
        if time_idx != 0:
            self.last_SWin_ds = climate.data['SWin'][time_idx - 1]
        else:
            self.last_SWin_ds = self.SWin_ds

        # main variables
        self.climateds = climate.cds
        self.args = args

        # define additional useful values
        self.tempK = self.tempC + CTOK
        self.prec =  self.tp / SPH     # tp is hourly total precip, prec is the rate in m/s
        self.rh = 100 if self.rh > 100 else self.rh
        self.get_roughness(surface.days_since_snowfall,layers)

        # apply factors
        self.wind *= float(args.wind_factor)
        if timestamp.day_of_year > args.snow_free_doy:
            # self.dustwet *= args.dust_factor 
            self.dustdry *= args.dust_factor

        # radiation terms
        self.measured_SWin = 'SWin' in climate.measured_vars
        self.nanLWin = True if np.isnan(self.LWin_ds) else False
        self.nanSWout = True if np.isnan(self.SWout_ds) else False
        self.nanLWout = True if np.isnan(self.LWout_ds) else False
        self.nanNR = True if np.isnan(self.NR_ds) else False
        self.nanalbedo = True if np.isnan(self.albedo_ds) else False
        return

    def surface_EB(self,surftemp,surface):
        """
        Calculates the surface heat fluxes for the 
        current timestep.

        Parameters
        ==========
        surftemp : float
            Temperature of the surface snow [C]
        surface : float
            Class object from pebsi.surface
        mode : str, default: 'sum'
            Options: 'sum', or 'optim'
            Return heat flux sum or absolute value of sum
            ('optim' is for BFGS optimization)

        Returns
        -------
        Qm : float OR np.ndarray
            Returns the sum of heat fluxes
        """
        # SHORTWAVE RADIATION  (Snet)
        SWin,SWout = self.get_SW(surface)

        # Handle penetrating shortwave separately
        if self.args.option_SWpen:
            if surface.stype in ['snow']:
                FRAC_ABSRAD = self.args.frac_absrad_snow
            else:
                FRAC_ABSRAD = self.args.frac_absrad_ice
        else:
            FRAC_ABSRAD = 1
        self.SWnet_surf = (SWin + SWout) * FRAC_ABSRAD
        self.SWnet_penetrating = (SWin + SWout) * (1 - FRAC_ABSRAD)

        # Store with surface fraction applied
        self.SWin = SWin
        self.SWout = SWout[0] if '__iter__' in dir(SWout) else SWout
        self.SWin_surf = self.SWin * FRAC_ABSRAD 
        self.SWout_surf = self.SWout * FRAC_ABSRAD
                    
        # LONGWAVE RADIATION (Lnet)
        LWin,LWout = self.get_LW(surftemp)
        self.LWnet = LWin + LWout
        self.LWin = LWin
        self.LWout = LWout[0] if '__iter__' in dir(LWout) else LWout

        # NET RADIATION
        if self.nanNR:
            NR = self.SWnet_surf + self.LWnet
            self.NR = NR
        else:
            NR = self.NR_ds / self.dt - self.SWnet_penetrating
            self.NR = self.NR_ds / self.dt - self.SWnet_penetrating

        # RAIN FLUX (Qp)
        Qp = self.get_rain(surftemp)
        self.rain = Qp[0] if '__iter__' in dir(Qp) else Qp

        # GROUND FLUX (Qg)
        Qg = self.get_ground(surftemp)
        self.ground = Qg[0] if '__iter__' in dir(Qg) else Qg

        # TURBULENT FLUXES (Qs and Ql)
        Qs, Ql = self.get_turbulent(surftemp)
        self.sens = Qs[0] if '__iter__' in dir(Qs) else Qs
        self.lat = Ql[0] if '__iter__' in dir(Ql) else Ql

        # OUTPUTS
        Qm = NR + Qp + Qs + Ql + Qg

        # keep track of iterations
        self.iters += 1

        return Qm
    
    def get_SW(self,surface):
        """
        Calculates incoming and outgoing shortwave heat
        flux accounting for:
        - Slope factor for direct radiation
        - Fraction of sky diffuse radiation
        - Shading
        - Terrain-reflected diffuse radiation
        
        Parameters
        ==========
        surface
            Class object from pebsi.surface
        """
        args = self.args

        # CONSTANTS
        SKY_VIEW = args.sky_view
        LAT = args.lat
        LON = args.lon
        SLOPE = args.slope * np.pi/180
        ASPECT = args.aspect * np.pi/180

        # albedo inputs
        albedo = surface.albedo
        spectral_weights = surface.spectral_weights
        if np.abs(1-np.sum(spectral_weights)) > 1e-5:
            ConfigError('surface.spectral_weights dont sum to 1: SNICAR issue')

        # get solar position
        time_UTC = self.timestamp - args.timezone
        sunpos = suncalc.get_position(time_UTC,LON,LAT)
        # suncalc gives azimuth with 0 = South, we want 0 = North
        SUN_AZ = sunpos['azimuth'] + np.pi     # solar azimuth angle
        SUN_ZEN = np.pi/2 - sunpos['altitude'] # solar zenith angle

        # calculate slope correction
        cos_theta = (np.cos(SUN_ZEN)*np.cos(SLOPE) + 
                    np.sin(SUN_ZEN)*np.sin(SLOPE)*np.cos(SUN_AZ - ASPECT))
        slope_correction = min(cos_theta / np.cos(SUN_ZEN), 5)
        slope_correction = max(slope_correction,0)
        
        # SWin needs to be corrected for shade
        if self.measured_SWin:
            # if point elev != AWS elev
            # is AWS in the sun?
            # if so: is the point in the sun?
                # if so: just calcualte diffuse
                # if not: neglect SWin, just diffuse
            # if not: is the point in the sun?
                # if so: COMPLICATED
                # if not: SWin AWS = SWin point
            SWin = self.SWin_ds/self.dt * slope_correction
            self.SWin_sky = np.nan
            self.SWin_terr = np.nan
        else:
            # get sky (diffuse+direct) and terrain (diffuse) SWin
            SWin_sky = self.SWin_ds/self.dt
            SWin_terrain = SWin_sky*(1-SKY_VIEW)*surface.albedo_surr

            # split sky into direct and diffuse
            f_diff = self.diffuse_fraction(SWin_sky, SUN_ZEN)
            SWin_direct = SWin_sky * (1-f_diff)
            SWin_diffuse = SWin_sky * f_diff * SKY_VIEW

            # correct for shade
            time_2024 = self.timestamp.replace(year=2024)
            self.shade = bool(surface.shading_df.loc[time_2024,'shaded'])

            # determine overall SWin flux
            if self.shade:
                SWin = SWin_terrain + SWin_diffuse
            else:
                SWin = SWin_terrain + SWin_diffuse + SWin_direct * slope_correction

            # store sky and terrain portions
            self.SWin_sky = SWin_diffuse if self.shade else SWin_sky
            self.SWin_terr = SWin_terrain

        # get reflected radiation
        if self.nanSWout and self.nanalbedo:
            albedo = albedo[0] if len(spectral_weights) < 2 else albedo
            SWout = -np.sum(SWin*spectral_weights*albedo)
        elif not self.nanalbedo:
            albedo = self.albedo_ds
            surface.bba = albedo
            SWout = -SWin*albedo
        else:
            SWout = -self.SWout_ds/self.dt
            # store albedo
            if -SWout < SWin and SWin > 0:
                surface.bba = max(0, min(1, -SWout / SWin))
        return SWin,SWout

    def get_LW(self,surftemp):
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

        if self.nanLWout:
            # calculate LWout from surftemp
            surftempK = surftemp + CTOK
            LWout = -EPS*SIGMA_SB*surftempK**4
        else:
            # take LWout from data
            LWout = -self.LWout_ds/self.dt
        
        if self.nanLWin and self.nanNR:
            # WARNING: THIS IS UNTESTED
            # calculate LWin from air temperature
            ezt = self.sat_vapor_pressure(self.tempC) * self.rh   # vapor pressure in hPa
            Ecs = .23 + .433*(ezt/self.tempK)**(1/8)  # clear-sky emissivity
            Ecl = 0.984               # cloud emissivity, Klok and Oerlemans, 2002
            Esky = Ecs*(1-self.tcc**2)+Ecl*self.tcc**2    # sky emissivity
            LWin = SIGMA_SB*(Esky*self.tempK**4)
        elif not self.nanLWin:
            # take LWin from data
            LWin = self.LWin_ds/self.dt
        elif not self.nanNR:
            # take LWout from net radiation data
            LWin = self.NR_ds/self.dt - LWout - self.SWin - self.SWout
            
        return LWin,LWout
    
    def get_rain(self,surftemp):
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

        # define rain vs snow scaling
        rain_scale = np.linspace(0,1,20)
        temp_scale = np.linspace(SNOW_THRESHOLD_LOW,SNOW_THRESHOLD_HIGH,20)
        
        # get fraction of precip that is rain
        if self.tempC < SNOW_THRESHOLD_LOW:
            frac_rain = 0
        elif SNOW_THRESHOLD_LOW < self.tempC < SNOW_THRESHOLD_HIGH:
            frac_rain = np.interp(self.tempC,temp_scale,rain_scale)
        else:
            frac_rain = 1

        Qp = (self.tempC-surftemp)*self.prec*frac_rain*DENSITY_WATER*CP_WATER
        return Qp
    
    def get_ground(self,surftemp):
        """
        Calculates amount of energy supplied to the surface
        by heat conduction from the temperate ice.
        
        Parameters
        ==========
        surftemp : float
            Surface temperature [C]
        """
        args = self.args 

        # CONSTANTS
        K_ICE = args.k_ice
        
        # calculate ground flux from surface temperature
        if args.method_ground in ['MolgHardy']:
            Qg = -K_ICE * (surftemp - args.temp_temp) / args.temp_depth
        else:
            ConfigError('Choose ground method from [MolgHardy]')
        return Qg
    
    def get_turbulent(self,surftemp):
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
        SLOPE = self.args.slope * np.pi/180

        # ROUGHNESS LENGTHS
        z0 = self.roughness  # Roughness length for momentum
        z0t = z0/100         # Roughness length for heat
        z0q = z0/10          # Roughness length for moisture

        # adjust wind speed to reference height
        z = 2 # reference height in m
        if args.wind_ref_height != 2:
            wind_2m *= np.log(2/z0) / np.log(WIND_REF_Z/z0)
        else:
            wind_2m = self.wind

        # transform humidity into mixing ratio (q) 
        Ewz = self.sat_vapor_pressure(self.tempC)  # saturation vapor pressure at 2m
        Ew0 = self.sat_vapor_pressure(surftemp)    # saturation vapor pressure at the surface
        qz = (self.rh/100)*0.622*(Ewz/(self.sp-Ewz))
        q0 = 1.0*0.622*(Ew0/(self.sp-Ew0))

        # get air density from PV=nRT
        density_air = self.sp/R_GAS/self.tempK*MM_AIR

        # latent heat term depends on direction of heat exchange
        if surftemp == 0. and (qz-q0) > 0:
            Lv = args.Lv_evap
        else:
            Lv = args.Lv_sub 

        # initiate loop
        loop = True
        counter = 0
        L = 0
        Qs_last = np.inf
        if args.method_turbulent in ['MO-similarity']:
            while loop:
                # calculate stability terms
                fric_vel = KARMAN*wind_2m / (np.log(z/z0)-self.PhiM(z,L))
                cD = KARMAN**2/np.square(np.log(z/z0) - self.PhiM(z,L) - self.PhiM(z0,L))
                csT = KARMAN*np.sqrt(cD) / (np.log(z/z0t) - self.PhiT(z,L) - self.PhiT(z0,L))
                csQ = KARMAN*np.sqrt(cD) / (np.log(z/z0q) - self.PhiT(z,L) - self.PhiT(z0,L))
                
                # calculate fluxes
                Qs = density_air*CP_AIR*csT*wind_2m*(self.tempC - surftemp)*np.cos(SLOPE)
                Ql = density_air*Lv*csQ*wind_2m*(qz-q0)*np.cos(SLOPE)

                # recalculate L
                if np.abs(Qs) < 1e-5:
                    Qs = 1e-5 # prevent overflow errors
                L = fric_vel**3*(self.tempK)*density_air*CP_AIR/(KARMAN*GRAVITY*Qs)
                L = max(L,0.3) # DEBAM uses this limit to prevent over-stabilization

                # check convergence
                counter += 1
                diff = np.abs(Qs_last-Qs)
                if counter > 10 or diff < 1e-1:
                    loop = False

                Qs_last = Qs
        elif args.method_turbulent in ['BulkRichardson']:
            # calculate Richardson number
            if wind_2m != 0:
                RICHARDSON = GRAVITY/self.tempK*(self.tempC-surftemp)*(z-z0)/wind_2m**2
            else:
                RICHARDSON = 0

            # calculate stability coefficients
            csT = KARMAN**2/(np.log(z/z0) * np.log(z/z0t))
            csQ = KARMAN**2/(np.log(z/z0) * np.log(z/z0q))

            if args.method_stability in ['cutoff']:
                if RICHARDSON <= 0.01:
                    psi = 1
                elif 0.01 < RICHARDSON <= 0.2:
                    psi = np.square(1-5*RICHARDSON)
                else:
                    psi = 0
            elif args.method_stability in ['BeljaarsHoltslag']:
                # Beljaars and Holtslag
                if RICHARDSON <= 0:
                    psi = (1.0 - 15.0 * RICHARDSON)**0.5 # unstable
                else:
                    psi = np.exp(-5.0 * RICHARDSON) # stable
            else:
                ConfigError('Choose stability correction from [BeljaarsHoltslag, cutoff]')
            
            # calculate fluxes
            Qs = density_air*CP_AIR*csT*psi*wind_2m*(self.tempC - surftemp)*np.cos(SLOPE)
            Ql = density_air*Lv*csQ*psi*wind_2m*(qz-q0)*np.cos(SLOPE)

        else:
            ConfigError('Choose turbulent method from [MO-similarity, BulkRichardson]')
        
        return Qs, Ql
    
    def get_dry_deposition(self, layers):
        """
        Adds dry deposition of light-absorbing particles
        to the surface layer.

        Parameters
        ==========
        layers
            Class object from pebsi.layers
        """
        # switch runs have no LAPs
        if self.args.switch_LAPs == 0:
            self.bcdry = 0
            self.ocdry = 0
            self.dustdry = 0

        # ice layers are not affected by LAPs
        if layers.ltype[0] != 'ice':
            layers.lBC[0] += self.bcdry * self.dt
            layers.lOC[0] += self.ocdry * self.dt
            layers.ldust[0] += self.dustdry * self.dt
        return
    
    def get_roughness(self,days_since_snowfall,layers):
        """
        Function to determine the roughness length of the
        surface. This assumes the roughness of snow
        linearly degrades with time in 60 days from that 
        of fresh snow to firn.

        Parameters
        ==========
        days_since_snowfall : int
            Number of days since fresh snow occurred
        layers
            Class object from pebsi.layers
        """
        # CONSTANTS
        ROUGHNESS_FRESH_SNOW = self.args.roughness_fresh_snow
        ROUGHNESS_AGED_SNOW = self.args.roughness_aged_snow
        ROUGHNESS_FIRN = self.args.roughness_firn
        ROUGHNESS_ICE = self.args.roughness_ice
        AGING_RATE = self.args.roughness_aging_rate

        # determine roughness from surface type
        layertype = layers.ltype
        if layertype[0] in ['snow']:
            sigma = min(ROUGHNESS_FRESH_SNOW + AGING_RATE * days_since_snowfall, ROUGHNESS_AGED_SNOW)
        elif layertype[0] in ['firn']:
            sigma = ROUGHNESS_FIRN
        elif layertype[0] in ['ice']:
            sigma = ROUGHNESS_ICE

        # return roughness in m
        self.roughness = sigma / 1000
        return 
    
    def sat_vapor_pressure(self,airtemp,method='ARM'):
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
            P = 0.61094*np.exp(17.625*airtemp/(airtemp+243.04)) # kPa
        elif method in ['Sonntag']:
            # follows COSIPY
            airtemp += CTOK
            if airtemp > CTOK: # over water
                P = 0.6112*np.exp(17.67*(airtemp-CTOK)/(airtemp-29.66))
            else: # over ice
                P = 0.6112*np.exp(22.46*(airtemp-CTOK)/(airtemp-0.55))

        # return vapor pressure in Pa
        return P*1000

    def diffuse_fraction(self,rad_glob,solar_zenith):
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
        doy = self.timestamp.day_of_year
        rad_pot = SOLAR_CONSTANT*(1+0.033*np.cos(2*np.pi*doy/365.25))*np.cos(solar_zenith)

        # exit if it's night
        if np.cos(solar_zenith) <= 0 or rad_pot < 1:
            return 0.0

        # determine clearness index
        CI = rad_glob / rad_pot
        CI = np.clip(CI, 0, 1)

        # empirical relationship for diffuse fraction
        diffuse_fraction = np.exp(-np.exp(P1-(P2-P3*CI)))*(1-P4)+P4
        return diffuse_fraction

    def stable_PhiM(self,z,L):
        """
        Calculates stability correction factor
        for the stable case.

        Parameters
        ==========
        z : float
            Reference height [m]
        L : float
            Obhukhov length [m]
        """
        zeta = z/L
        if zeta > 1:
            phim = -4*(1+np.log(zeta)) - zeta
        elif zeta > 0:
            phim = -5*zeta
        else:
            phim = 0
        return phim

    def PhiM(self,z,L):
        """
        Determines piecewise calculation of universal
        function for momentum for the Monin-Obhukhov 
        turbulent flux method

        Parameters
        ==========
        z : float
            Reference height [m]
        L : float
            Obhukhov length [m]
        """
        if L < 0:
            X = np.power((1-16*z/L),0.25)
            phim = 2*np.log((1+X)/2) + np.log((1+X**2)/2) - 2*np.arctan(X) + np.pi/2
        elif L > 0: # stable
            phim = self.stable_PhiM(z, L)
        else:
            phim = 0.0
        return phim

    def PhiT(self,z,L):
        """
        Determines piecewise calculation of universal
        function for heat for the Monin-Obhukhov 
        turbulent flux method

        Parameters
        ==========
        z : float
            Reference height [m]
        L : float
            Obhukhov length [m]
        """
        if L < 0:
            X = np.power((1-19.3*z/L),0.25)
            phit = 2*np.log((1+X**2)/2)
        elif L > 0: # stable
            phit = self.stable_PhiM(z, L)
        else:
            phit = 0.0
        return phit