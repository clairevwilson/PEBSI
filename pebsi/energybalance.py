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
# Local libraries
import pebsi.input as prms

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

        # unpack climate variables
        climateds_now = climate.cds.sel(time=timestamp)
        self.tempC = climateds_now['temp'].values
        self.tp = climateds_now['tp'].values
        self.sp = climateds_now['sp'].values
        self.rh = climateds_now['rh'].values
        self.wind = climateds_now['wind'].values
        self.tcc = climateds_now['tcc'].values
        self.SWin_ds = climateds_now['SWin'].values
        self.SWout_ds = climateds_now['SWout'].values
        self.albedo_ds = climateds_now['albedo'].values
        self.LWin_ds = climateds_now['LWin'].values
        self.LWout_ds = climateds_now['LWout'].values
        self.NR_ds = climateds_now['NR'].values
        self.bcdry = climateds_now['bcdry'].values
        self.bcwet = climateds_now['bcwet'].values
        self.ocdry = climateds_now['ocdry'].values
        self.ocwet = climateds_now['ocwet'].values
        self.dustdry = climateds_now['dustdry'].values
        self.dustwet = climateds_now['dustwet'].values

        # time
        self.timestamp = timestamp
        self.dt = prms.dt

        # store previous timestep incoming shortwave
        if timestamp != pd.to_datetime(climate.cds.time.values[0]):
            last_stamp = timestamp - pd.Timedelta(seconds=self.dt)
            self.last_SWin_ds = climate.cds.sel(time=last_stamp)['SWin'].values
        else:
            self.last_SWin_ds = self.SWin_ds

        # main variables
        self.climateds = climate.cds
        self.args = args

        # define additional useful values
        self.tempK = self.tempC + 273.15
        self.prec =  self.tp / 3600     # tp is hourly total precip, prec is the rate in m/s
        self.rh = 100 if self.rh > 100 else self.rh
        self.get_roughness(surface.days_since_snowfall,layers)

        # adjust wind speed
        self.wind *= float(prms.wind_factor)

        # radiation terms
        self.measured_SWin = 'SWin' in climate.measured_vars
        self.nanLWin = True if np.isnan(self.LWin_ds) else False
        self.nanSWout = True if np.isnan(self.SWout_ds) else False
        self.nanLWout = True if np.isnan(self.LWout_ds) else False
        self.nanNR = True if np.isnan(self.NR_ds) else False
        self.nanalbedo = True if np.isnan(self.albedo_ds) else False
        return

    def surface_EB(self,surftemp,surface,mode='sum'):
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
        self.SWin = SWin
        self.SWout = SWout[0] if '__iter__' in dir(SWout) else SWout

        # Handle penetrating shortwave separately
        if prms.option_SWpen:
            if surface.stype in ['snow']:
                FRAC_ABSRAD = prms.frac_absrad_snow
            else:
                FRAC_ABSRAD = prms.frac_absrad_ice
        else:
            FRAC_ABSRAD = 1
        self.SWnet_surf = (SWin + SWout) * FRAC_ABSRAD
        self.SWnet_penetrating = (SWin + SWout) * (1 - FRAC_ABSRAD)
                    
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
            NR = self.NR_ds / self.dt
            self.NR = self.NR_ds / self.dt

        # RAIN FLUX (Qp)
        Qp = self.get_rain(surftemp)
        self.rain = Qp[0] if '__iter__' in dir(Qp) else Qp

        # GROUND FLUX (Qg)
        Qg = self.get_ground(surftemp)
        self.ground = Qg[0] if '__iter__' in dir(Qg) else Qg

        # TURBULENT FLUXES (Qs and Ql)
        Qs, Ql = self.get_turbulent(surftemp)
        self.sens = Qs[0] if '__iter__' in dir(Qs) else Qs
        self.lat = Ql[0] if '__iter__' in dir(Qs) else Ql

        # OUTPUTS
        Qm = NR + Qp + Qs + Ql + Qg

        if mode in ['sum']:
            return Qm
        elif mode in ['optim']:
            return np.abs(Qm)
        else:
            assert 1==0, 'argument \'mode\' in function surfaceEB should be sum or optim'
    
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
        # CONSTANTS
        SKY_VIEW = self.args.sky_view
        LAT = self.args.lat
        LON = self.args.lon
        SLOPE = self.args.slope * np.pi/180
        ASPECT = self.args.aspect * np.pi/180

        # albedo inputs
        albedo = surface.albedo
        spectral_weights = surface.spectral_weights
        assert np.abs(1-np.sum(spectral_weights)) < 1e-5, 'Solar weights dont sum to 1'

        # get solar position
        time_UTC = self.timestamp - self.args.timezone
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
            SWin_diffuse = SWin_sky * f_diff

            # correct direct radiation for slope
            SWin_direct *= slope_correction

            # correct for shade
            time_str = str(self.timestamp).replace(str(self.timestamp.year),'2024')
            time_2024 = pd.to_datetime(time_str)
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
                surface.bba = -SWout / SWin
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
        if self.nanLWout:
            # calculate LWout frmo surftemp
            surftempK = surftemp+273.15
            LWout = -prms.sigma_SB*surftempK**4
        else:
            # take LWout from data
            LWout = -self.LWout_ds/self.dt
        
        if self.nanLWin and self.nanNR:
            # calculate LWin from air temperature
            ezt = self.vapor_pressure(self.tempC)    # vapor pressure in hPa
            Ecs = .23+ .433*(ezt/self.tempK)**(1/8)  # clear-sky emissivity
            Ecl = 0.984               # cloud emissivity, Klok and Oerlemans, 2002
            Esky = Ecs*(1-self.tcc**2)+Ecl*self.tcc**2    # sky emissivity
            LWin = prms.sigma_SB*(Esky*self.tempK**4)
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
        SNOW_THRESHOLD_LOW = prms.snow_threshold_low
        SNOW_THRESHOLD_HIGH = prms.snow_threshold_high
        DENSITY_WATER = prms.density_water
        CP_WATER = prms.Cp_water

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
        # calculate ground flux from surface temperature
        K_ICE = prms.k_ice
        if prms.method_ground in ['MolgHardy']:
            Qg = -K_ICE * (surftemp - prms.temp_temp) / prms.temp_depth
        else:
            assert 1==0, 'Ground flux method not accepted; choose from [\'MolgHardy\']'
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
        # CONSTANTS
        KARMAN = prms.karman
        GRAVITY = prms.gravity
        R_GAS = prms.R_gas
        MM_AIR = prms.molarmass_air
        CP_AIR = prms.Cp_air
        WIND_REF_Z = prms.wind_ref_height
        SLOPE = self.args.slope * np.pi/180

        # ROUGHNESS LENGTHS
        z0 = self.roughness  # Roughness length for momentum
        z0t = z0/100         # Roughness length for heat
        z0q = z0/10          # Roughness length for moisture

        # adjust wind speed to reference height
        z = 2 # reference height in m
        if prms.wind_ref_height != 2:
            wind_2m *= np.log(2/z0) / np.log(WIND_REF_Z/z0)
        else:
            wind_2m = self.wind

        # transform humidity into mixing ratio (q) 
        Ewz = self.vapor_pressure(self.tempC)  # vapor pressure at 2m
        Ew0 = self.vapor_pressure(surftemp)    # vapor pressure at the surface
        qz = (self.rh/100)*0.622*(Ewz/(self.sp-Ewz))
        q0 = 1.0*0.622*(Ew0/(self.sp-Ew0))

        # get air density from PV=nRT
        density_air = self.sp/R_GAS/self.tempK*MM_AIR

        # latent heat term depends on direction of heat exchange
        if surftemp == 0. and (qz-q0) > 0:
            Lv = prms.Lv_evap
        else:
            Lv = prms.Lv_sub 

        # initiate loop
        loop = True
        counter = 0
        L = 0
        Qs_last = np.inf
        if prms.method_turbulent in ['MO-similarity']:
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
                    if diff > 1:
                        print('Turbulent fluxes didnt converge; Qs still changing by',diff)

                Qs_last = Qs
        elif prms.method_turbulent in ['BulkRichardson']:
            # calculate Richardson number
            if wind_2m != 0:
                RICHARDSON = GRAVITY/self.tempK*(self.tempC-surftemp)*(z-z0)/wind_2m**2
            else:
                RICHARDSON = 0

            # calculate stability coefficients
            csT = KARMAN**2/(np.log(z/z0) * np.log(z/z0t))
            csQ = KARMAN**2/(np.log(z/z0) * np.log(z/z0q))
            if RICHARDSON <= 0.01:
                psi = 1
            elif 0.01 < RICHARDSON <= 0.2:
                psi = np.square(1-5*RICHARDSON)
            else:
                psi = 0
            
            # calculate fluxes
            Qs = density_air*CP_AIR*csT*psi*wind_2m*(self.tempC - surftemp)*np.cos(SLOPE)
            Ql = density_air*Lv*csQ*psi*wind_2m*(qz-q0)*np.cos(SLOPE)
        else:
            assert 1==0, 'Choose turbulent method from MO-similarity or BulkRichardson'
        
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
        # CONSTANTS
        BC_RATIO = prms.ratio_BC2_BCtot
        OC_RATIO = prms.ratio_OC2_OCtot
        DUST_RATIO = prms.ratio_DU3_DUtot
        
        # switch runs have no LAPs
        if prms.switch_LAPs == 0:
            self.bcdry = 0
            self.ocdry = 0
            self.dustdry = 0

        # ice layers are not affected by LAPs
        if layers.ltype[0] != 'ice':
            layers.lBC[0] += self.bcdry * self.dt * BC_RATIO
            layers.lOC[0] += self.ocdry * self.dt * OC_RATIO
            layers.ldust[0] += self.dustdry * self.dt * DUST_RATIO
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
        ROUGHNESS_FRESH_SNOW = prms.roughness_fresh_snow
        ROUGHNESS_AGED_SNOW = prms.roughness_aged_snow
        ROUGHNESS_FIRN = prms.roughness_firn
        ROUGHNESS_ICE = prms.roughness_ice
        AGING_RATE = prms.roughness_aging_rate

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
    
    def vapor_pressure(self,airtemp,method='ARM'):
        """
        Calculates vapor pressure [Pa] 
        from air temperature 

        Parameters
        ==========
        airtemp : float
            Air temperature [C]
        """
        if method in ['ARM']:
            P = 0.61094*np.exp(17.625*airtemp/(airtemp+243.04)) # kPa
        elif method in ['Sonntag']:
            # follows COSIPY
            airtemp += 273.15
            if airtemp > 273.15: # over water
                P = 0.6112*np.exp(17.67*(airtemp-273.15)/(airtemp-29.66))
            else: # over ice
                P = 0.6112*np.exp(22.46*(airtemp-273.15)/(airtemp-0.55))
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
        SOLAR_CONSTANT = prms.solar_constant
        P1 = 0.1001
        P2 = 4.7930
        P3 = 9.4758
        P4 = 0.2465

        # calculate potential (extraterrestrial) shortwave radiation
        doy = self.timestamp.day_of_year
        rad_pot = SOLAR_CONSTANT*(1+0.033*np.cos(2*np.pi*doy/366))*np.cos(solar_zenith)

        # determine clearness index
        CI = rad_glob / rad_pot

        # empirical relationship for diffuse fraction
        if CI > 50:
            diffuse_fraction = P4
        else:
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