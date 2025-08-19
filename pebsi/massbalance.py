"""
Mass balance and output class for PEBSI

Contains main() function which executes
all energy and mass balance calculations
in an hourly time loop.

@author: clairevwilson
"""
# Built-in libraries
import os, sys
# External libraries
import numpy as np
import pandas as pd
import xarray as xr
# Local libraries
import pebsi.input as prms
from pebsi.energybalance import energyBalance
from pebsi.layers import Layers
from pebsi.surface import Surface
np.set_printoptions(suppress=True)

class massBalance():
    """
    Mass balance scheme which contains the main()
    model function with the core time loop. This
    class handles all mass balance equations and 
    terms.
    """
    def __init__(self,args,climate):
        """
        Initializes the layers and surface classes 
        and model time for the mass balance scheme.

        Parameters
        ==========
        args : command-line arguments
        climate
            Class object frmo pebsi.climate
        """
        # set up model time
        self.dt = prms.dt
        self.days_since_snowfall = 0
        self.time_list = climate.dates
        self.firn_converted = False

        # initialize climate, layers and surface classes
        self.args = args
        self.climate = climate
        self.layers = Layers(climate,args)
        self.surface = Surface(self.layers,self.time_list,args,climate)

        # initialize output class
        self.output = Output(self.time_list,args)

        # initialize mass balance check variables
        self.previous_mass = np.sum(self.layers.lice + self.layers.lwater)
        self.lice_before = np.sum(self.layers.lice)
        self.lwater_before = np.sum(self.layers.lwater)
        return
    
    def main(self):
        """
        Core function which executes the time loop
        for all mass balance and energy balance 
        calculations.
        """
        # get classes
        layers = self.layers
        surface = self.surface

        # CONSTANTS
        DENSITY_WATER = prms.density_water

        # ===== ENTER TIME LOOP =====
        for time in self.time_list:
            # >>> INITIALIZE TIMESTEP <<<
            self.time = time

            # initialize the energy balance to unpack climate data for this timestep
            enbal = energyBalance(self,time)
            self.enbal = enbal

            # get rain and snowfall amounts [kg m-2]
            rainfall,snowfall = self.get_precip()

            # >>> ADD SNOW AND LAPs <<<
            # add fresh snow to layers
            snowfall = self.add_snow(snowfall)

            # add dry deposited BC, OC and dust to layers
            enbal.get_dry_deposition(layers)

            # >>> UPDATE DAILY PROPERTIES <<<
            if time.hour == 0:
                # surrounding albedo, surface type, days since snowfall
                surface.daily_updates(layers,time)
                self.days_since_snowfall = surface.days_since_snowfall

                # grain size
                if self.args.switch_melt == 2 and layers.nlayers > 2:
                    self.get_grain_size()

            if time.hour in prms.albedo_TOD:
                # update albedo
                surface.get_albedo(layers,time)

            # >>> SURFACE ENERGY BALANCE <<<
            # simultaneously solve surface energy balance and surface temperature
            surface.get_surftemp(enbal,layers)

            # >>> MELTING <<<
            # calculate subsurface heating from penetrating SW
            subsurf_layermelt = self.subsurface_heating()
    
            # combine surface and subsurface melt into one array
            layermelt = self.melting(subsurf_layermelt)

            # sum melt for output
            melt = np.sum(layermelt)
            # fully melted layers were removed from layermelt so add that mass
            if self.melted_layers != 0:
                melt += np.sum(self.melted_layers.mass)

            # >>> PERCOLATION <<<
            # route meltwater and LAPs through snow/firn
            runoff = self.percolation(layermelt,rainfall)
            
            # >>> TEMPERATURE PROFILE <<<
            # recalculate the temperature profile considering conduction
            self.thermal_conduction()

            # >>> PHASE CHANGES <<<
            # calculate mass gain or loss from phase changes
            self.phase_changes()

            # >>> REFREEZING & DENSIFICATION <<<
            # calculate refreeze
            refreeze = self.refreezing()

            # run densification (daily)
            if time.hour == 0:
                self.densification()
            
            # >>> CHECK LAYERS <<<
            # check and update layer sizes
            layers.check_layer_sizes()

            # if towards the end of summer, check if old snow should become firn
            doy = time.day_of_year
            date_in_range = doy >= prms.start_end_summer and doy <= prms.start_end_summer + 60
            if date_in_range and time.hour == 0 and not self.firn_converted:
                self.end_of_summer()

            # if start of calendar year, reset firn tracker
            if time.day_of_year == 1 and time.hour == 0:
                self.firn_converted = False

            # check mass conserves
            mass_in = snowfall + rainfall + self.condensation + self.deposition
            mass_out = runoff + self.evaporation + self.sublimation
            self.check_mass_conservation(mass_in, mass_out)
          
            # >>> STORE OUTPUT <<<
            # convert units of mass balance terms
            self.runoff = runoff / DENSITY_WATER
            self.melt = melt / DENSITY_WATER
            self.refreeze = refreeze / DENSITY_WATER
            self.accum = snowfall / DENSITY_WATER

            # store timestep data
            self.output.store_timestep(self,enbal,surface,layers,time)   

            # >>> END TIMESTEP <<<
            # debugging: print current state and monthly melt at the start of each month
            if time.is_month_start and time.hour == 0 and self.args.debug:
                self.current_state_prints()

            # check if we still have a glacier before next timestep
            self.check_glacier_exists()

        # ===== COMPLETED SIMULATION: STORE DATA =====
        if self.args.store_data:
            self.output.store_data()

        # optionally store spectral albedo
        if prms.store_bands:
            surface.albedo_df.to_csv(prms.albedo_out_fp.replace('.csv',f'_{self.args.elev}.csv'))
        
        # delete temporary files
        self.delete_temp_files()
        return
    
    def get_precip(self):
        """
        Determines whether rain or snowfall occurred 
        and outputs amounts.

        Returns:
        --------
        rain, snow : float
            Specific mass of liquid and solid 
            precipitation [kg m-2]
        """
        # get classes
        enbal = self.enbal

        # CONSTANTS
        SNOW_THRESHOLD_LOW = prms.snow_threshold_low
        SNOW_THRESHOLD_HIGH = prms.snow_threshold_high
        DENSITY_WATER = prms.density_water

        # define rain vs snow scaling 
        rain_scale = np.linspace(0,1,20)
        temp_scale = np.linspace(SNOW_THRESHOLD_LOW,SNOW_THRESHOLD_HIGH,20)

        if enbal.tempC <= SNOW_THRESHOLD_LOW: 
            # precip falls as snow
            rain = 0
            snow = enbal.tp*DENSITY_WATER
        elif SNOW_THRESHOLD_LOW < enbal.tempC < SNOW_THRESHOLD_HIGH:
            # mix of rain and snow
            fraction_rain = np.interp(enbal.tempC,temp_scale,rain_scale)
            rain = enbal.tp*fraction_rain*DENSITY_WATER
            snow = enbal.tp*(1-fraction_rain)*DENSITY_WATER
        else:
            # precip falls as rain
            rain = enbal.tp*DENSITY_WATER
            snow = 0
        
        return rain,snow  # kg m-2
    
    def add_snow(self,snowfall):
        """
        Adds snowfall to the layers. If the existing top 
        layer has a large enough difference in density 
        (eg. firn or ice), the fresh snow is a new layer,
        otherwise it is merged with the top snow layer.
        
        Parameters
        ==========
        snowfall : float
            Fresh snow mass [kg m-2]

        Returns
        =======
        snowfall : float
            Actual snow mass that was added [kg m-2]
        """
        # get classes
        layers = self.layers
        enbal = self.enbal

        # add delayed snow to snowfall
        snowfall += layers.delayed_snow
        if snowfall == 0:
            # skip this function if no snow
            return 0
        
        # define initial mass for conservation check
        initial_mass = np.sum(layers.lice + layers.lwater)

        # check switches
        if self.args.switch_snow == 0:
            # snow falls with the same properties as the current top layer
            new_density = layers.ldensity[0]
            new_height = snowfall/new_density
            new_grainsize = layers.lgrainsize[0]
            new_BC = layers.lBC[0]/layers.lheight[0]*new_height
            new_OC = layers.lOC[0]/layers.lheight[0]*new_height
            new_dust = layers.ldust[0]/layers.lheight[0]*new_height
            new_age = layers.lage[0]
        elif self.args.switch_snow == 1:
            # check if using constant density for new snow
            if prms.constant_snowfall_density:
                new_density = prms.constant_snowfall_density
            else:
                # CROCUS formulation of density (Vionnet et al. 2012)
                new_density = max(109+6*(enbal.tempC-0.)+26*enbal.wind**0.5,50)
            
            # check if using constant grain size for new snow
            if prms.constant_freshgrainsize:
                new_grainsize = prms.constant_freshgrainsize
            else:
                # CLM formulation of grain size (CLM5.0 Documentation)
                airtemp = enbal.tempC
                new_grainsize = np.piecewise(airtemp,
                                    [airtemp<=-30,-30<airtemp<0,airtemp>=0],
                                    [54.5,54.5+5*(airtemp+30),204.5])

            # height and age of new layer
            new_height = snowfall/new_density
            new_age = self.time
            if type(layers.lage[0]) == int:
                print('new age', layers.lage[0])

            # wet deposition occurs in snowfall
            new_BC = enbal.bcwet * enbal.dt
            new_OC = enbal.ocwet * enbal.dt
            new_dust = enbal.dustwet * enbal.dt
            
            # update new snow timestamp
            self.surface.snow_timestamp = self.time

        # check switch for LAPs
        if prms.switch_LAPs != 1:
            new_BC = 0
            new_OC = 0
            new_dust = 0
            
        # conditions: if any are TRUE, create a new layer
        new_layer_cond = np.array([layers.ltype[0] in 'ice',
                            layers.ltype[0] in 'firn',
                            layers.ldensity[0] > new_density*3])
        # unless there is a small surface layer, then combine new snow
        small_surf_layer = layers.lheight[0] < 1e-3

        # check conditions
        if np.any(new_layer_cond) and not small_surf_layer:
            # check if there is enough snow to create a new layer (1 mm cutoff)
            if new_height < 1e-3:
                # delay small amounts of snowfall: avoids computational issues
                layers.delayed_snow = snowfall
                return 0 # no snow was added so return 0
            else:
                # create a dataframe of all the new properties
                new_layer = pd.DataFrame([enbal.tempC,0,snowfall/new_density,'snow',snowfall,
                                      new_grainsize,new_BC,new_OC,new_dust,new_age],
                                     index=['T','w','h','t','m','g','BC','OC','dust','time'])
                # add the layer and reset delayed_snow to 0
                layers.add_layers(new_layer)
                layers.delayed_snow = 0
                if type(layers.lage[0]) == int:
                    print('htfffere', layers.lage[0], new_age)
        else:
            # get new layers mass
            new_layermass = layers.lice[0] + snowfall

            # take mass-weighted average surface layer and new snow
            layers.ldensity[0] = (layers.ldensity[0]*layers.lice[0] + new_density*snowfall)/(new_layermass)
            layers.ltemp[0] = (layers.ltemp[0]*layers.lice[0] + enbal.tempC*snowfall)/(new_layermass)
            layers.lgrainsize[0] = (layers.lgrainsize[0]*layers.lice[0] + new_grainsize*snowfall)/(new_layermass)

            # mass-weighted age
            decimal_time = layers.to_decimal_year([self.time, layers.lage[0]])
            layer_mass = np.array([snowfall, layers.lice[0]])
            mean_time = np.sum(decimal_time*layer_mass)/np.sum(layer_mass)
            layers.lage[0] = layers.from_decimal_year(mean_time)

            # add mass to layers
            layers.lice[0] = new_layermass
            layers.lheight[0] += snowfall/new_density
            
            # sum LAPs
            layers.lBC[0] = layers.lBC[0] + new_BC
            layers.lOC[0] = layers.lOC[0] + new_OC
            layers.ldust[0] = layers.ldust[0] + new_dust
            
            # check if layer got too big and needs to be split
            if layers.lheight[0] > (prms.dz_toplayer * 2):
                layers.split_layer(0)
            
            # reset delayed snow
            layers.delayed_snow = 0

        # update layer depth from new layer heights
        layers.update_layer_props()

        # CHECK MASS CONSERVATION
        change = np.sum(layers.lice + layers.lwater) - initial_mass
        assert np.abs(change - snowfall) < prms.mb_threshold

        # return actual snowfall that was added, including any delayed_snow
        return snowfall
    
    def get_grain_size(self):
        """
        Updates grain size according to wet and dry
        metamorphism, refreeze, and addition of fresh
        snow.
        """
        # get classes
        enbal = self.enbal
        layers = self.layers
        surface = self.surface

        # CONSTANTS
        WET_C = prms.wet_snow_C
        PI = np.pi
        RFZ_GRAINSIZE = prms.rfz_grainsize
        FIRN_GRAINSIZE = prms.firn_grainsize
        ICE_GRAINSIZE = prms.ice_grainsize
        dt = prms.daily_dt

        # get temperatures
        airtemp = enbal.tempC
        surftemp = surface.stemp

        if prms.constant_freshgrainsize:
            FRESH_GRAINSIZE = prms.constant_freshgrainsize
        else:
            FRESH_GRAINSIZE = np.piecewise(airtemp,[airtemp<=-30,-30<airtemp<0,airtemp>=0],
                                       [54.5,54.5+5*(airtemp+30),204.5])
            
        # only update grain size if we have snow layers
        if len(layers.snow_idx) > 0:
            idx = layers.snow_idx
            n = len(idx)
            
            # get fractions of refreeze and old snow
            m_refreeze = layers.drefreeze[idx]
            m_snow = layers.lice[idx] - m_refreeze
            f_snow = m_snow / layers.lice[idx]
            f_rfz = m_refreeze / layers.lice[idx]
            f_liq = layers.lwater[idx] / (layers.lwater[idx] + layers.lice[idx])

            # define values for lookup table
            dz = layers.lheight.copy()[idx]
            T = layers.ltemp.copy()[idx] + 273.15
            p = layers.ldensity.copy()[idx]
            grainsize = layers.lgrainsize.copy()[idx]
            surftempK = surftemp + 273.15

            # dry metamorphism
            if prms.constant_drdry:
                drdry = np.ones(len(idx))*prms.constant_drdry * dt # um
                drdry[np.where(grainsize>RFZ_GRAINSIZE)[0]] = 0
            else:
                # calculate temperature gradient (central in space)
                dTdz = np.zeros_like(T)
                if len(idx) > 2:
                    dTdz[0] = (surftempK - (T[0]*dz[0]+T[1]*dz[1]) / (dz[0]+dz[1]))/dz[0]
                    dTdz[1:-1] = ((T[:-2]*dz[:-2] + T[1:-1]*dz[1:-1]) / (dz[:-2] + dz[1:-1]) -
                            (T[1:-1]*dz[1:-1] + T[2:]*dz[2:]) / (dz[1:-1] + dz[2:])) / dz[1:-1]
                    dTdz[-1] = dTdz[-2] # bottom temp gradient isn't used, set to next layer up
                elif len(idx) == 2: # use top ice layer for temp gradient
                    T_2layer = np.array([surftempK,T[0],T[1],layers.ltemp[2]+273.15])
                    depth_2layer = np.array([0,layers.ldepth[0],layers.ldepth[1],layers.ldepth[2]])
                    dTdz = (T_2layer[0:2] - T_2layer[2:]) / (depth_2layer[0:2] - depth_2layer[2:])
                else: # single layer
                    # layers.ltemp is in C so use surftemp, not surftempK
                    dTdz = (surftemp - layers.ltemp[1]) / layers.ldepth[1]
                    dTdz = np.array([dTdz])

                # take absolute value of gradient (direction does not matter)
                dTdz = np.abs(dTdz)

                # force values to be within lookup table ranges
                p[np.where(p < 50)[0]] = 50
                p[np.where(p > 400)[0]] = 400
                dTdz[np.where(dTdz > 300)[0]] = 300
                T[np.where(T < 223.15)[0]] = 223.15
                T[np.where(T > 273.15)[0]] = 273.15

                # interpolate lookup table at the values of T,dTdz,p
                ds = prms.grainsize_ds.copy(deep=True)
                ds = ds.interp(TVals=T.astype(float),
                            DTDZVals=dTdz.astype(float),
                            DENSVals=p.astype(float))
                
                # extract values
                diag = np.zeros((n,n,n),dtype=bool)
                for i in range(n):
                    diag[i,i,i] = True
                tau = ds.taumat.to_numpy()[diag].astype(float)
                kap = ds.kapmat.to_numpy()[diag].astype(float)
                dr0 = ds.dr0mat.to_numpy()[diag].astype(float)

                # calculate dry grain growth
                drdrydt = []
                for r,t,k,g in zip(dr0,tau,kap,grainsize):
                    # condition to avoid 0 denominator / negative with fractional power
                    if t + g <= FRESH_GRAINSIZE:
                        drdrydt.append(r*np.power(t/(t + 1e-6),1/k)/dt)
                    else:
                        drdrydt.append(r*np.power(t/(t + g - FRESH_GRAINSIZE),1/k)/dt)
                # get growth for this timestep
                drdry = np.array(drdrydt) * dt

            # wet metamorphism
            drwetdt = WET_C*f_liq**3/(4*PI*(grainsize/1e6)**2)
            drwet = drwetdt * dt * 1e6 # transform to um from m

            # get change in grain size due to aging
            aged_grainsize = grainsize + drdry + drwet
                      
            # sum contributions of snow and refreeze
            grainsize = aged_grainsize*f_snow + RFZ_GRAINSIZE*f_rfz

            # enforce maximum grainsize
            grainsize[grainsize > FIRN_GRAINSIZE] = FIRN_GRAINSIZE

            # update grainsize in layers
            layers.lgrainsize[idx] = grainsize
            layers.lgrainsize[layers.firn_idx] = FIRN_GRAINSIZE 
            layers.lgrainsize[layers.ice_idx] = ICE_GRAINSIZE

        elif len(layers.firn_idx) > 0: # no snow, but there is firn
            layers.lgrainsize[layers.firn_idx] = FIRN_GRAINSIZE
            layers.lgrainsize[layers.ice_idx] = ICE_GRAINSIZE
        else: # no snow or firn, just ice
            layers.lgrainsize[layers.ice_idx] = ICE_GRAINSIZE

        return 

    def subsurface_heating(self):
        """
        Calculates melt in subsurface layers (excluding
        layer 0) due to penetrating shortwave radiation.

        Returns
        -------
        layermelt : np.ndarray
            Subsurface melt for each layer [kg m-2]
        """
        # get classes
        layers = self.layers
        enbal = self.enbal

        # check if this function can be skipped
        if layers.nlayers == 1: # only one layer: no subsurface to heat
            return [0.] # surface melt is filled in melting()
        if enbal.SWnet_penetrating < 1e-6: # no penetrating radiation
            return np.zeros(layers.nlayers)
        
        # CONSTANTS
        HEAT_CAPACITY_ICE = prms.Cp_ice
        LH_RF = prms.Lh_rf

        # LAYERS IN
        ld = layers.ldepth.copy()
        lT = layers.ltemp.copy()
        lm = layers.lice.copy()

        # determine extinction coefficient from surface layer type
        if layers.ltype[0] == 'snow':
            EXTINCT_COEF = prms.extinct_coef_snow
        else:
            EXTINCT_COEF = prms.extinct_coef_ice

        # absorbed shortwave for each layer
        SWnet_pen = enbal.SWnet_penetrating
        layerSW = SWnet_pen*np.exp(-EXTINCT_COEF*ld)
        layerSW[layerSW < 1e-6] = 0 # cut off tiny amounts of energy
        layerSW[0] = 0 # surface layer handled separately

        # recalculate layer temperatures, excluding the top layer (calculated separately)
        lT[1:] += layerSW[1:]*self.dt/(lm[1:]*HEAT_CAPACITY_ICE)

        # calculate melt from temperatures above 0
        layermelt = np.zeros(layers.nlayers)
        leftover_melt = 0

        for layer, temp in enumerate(lT):
            # convert leftover melt to energy [J m-2]
            leftover_energy = leftover_melt * LH_RF

            if temp > 0.:
                # melting: calculate melt energy from layer temperature
                sensible_energy = temp * lm[layer] * HEAT_CAPACITY_ICE
                total_energy = sensible_energy + leftover_energy
                melt = total_energy / LH_RF
                # set layer temp to the melting point
                lT[layer] = 0.
            else:
                # calculate energy needed to warm the layer to melting point
                required_energy = abs(temp) * lm[layer] * HEAT_CAPACITY_ICE

                if leftover_energy >= required_energy:
                    # use leftover energy to warm to melting point and melt
                    lT[layer] = 0.
                    leftover_energy -= required_energy
                    melt = leftover_energy / LH_RF
                else:
                    # not enough energy to warm to melting point; warm partially
                    lT[layer] += leftover_energy / (lm[layer] * HEAT_CAPACITY_ICE)
                    melt = 0

            # cap melt at available layer mass
            if melt > lm[layer]:
                layermelt[layer] = lm[layer]
                leftover_melt = melt - lm[layer]
            else:
                layermelt[layer] = melt
                leftover_melt = 0

        # force surface layer melt to be 0 (calculated in melting)
        layermelt[0] = 0

        # LAYERS OUT
        layers.ltemp = lT
        return layermelt

    def melting(self,subsurf_melt):
        """
        For cases when layers are melting. Can melt 
        multiple surface layers at once if Qm is 
        sufficiently high. Otherwise, adds the surface
        layer melt to the array containing subsurface 
        melt to return the total layer melt. 
        
        This function DOES NOT remove melted mass from 
        layers. That is done in percolation().

        Parameters
        ==========
        subsurf_melt : np.ndarray
            Subsurface melt for each layer [kg m-2]
        
        Returns
        -------
        layermelt : np.ndarray
            Melt for each layer [kg m-2]
        """
        # get classes
        layers = self.layers

        # CONSTANTS
        LH_RF = prms.Lh_rf

        # LAYERS IN
        lm = layers.lice.copy()
        layermelt = subsurf_melt.copy()       # mass of melt due to penetrating SW [kg m-2]
        initial_mass = np.sum(layers.lice + layers.lwater)

        # calculate surface melt
        surface_melt = max(0,self.surface.Qm*self.dt/LH_RF)     # mass of melt due to SEB [kg m-2]

        # check if melt by surface energy balance completely melts surface layer
        if surface_melt > lm[0]: 
            # distribute surface melt into next layers down
            layer = 0
            while surface_melt > 0 and layer < len(layermelt):
                capacity = lm[layer] - layermelt[layer]  # how much more this layer can take
                melt_added = min(surface_melt, capacity)
                layermelt[layer] += melt_added
                surface_melt -= melt_added
                layer += 1
        else:
            # only surface layer is melting or surface melt is 0
            layermelt[0] = surface_melt

        # check how many layers fully melted
        fully_melted = []
        if np.any(lm - layermelt <= 0):
            melted_subsurf = np.where(lm - layermelt <= 0)[0]
            for i in melted_subsurf:
                if i not in fully_melted:
                    fully_melted.append(i)
            fully_melted = np.array(fully_melted, dtype=int)
        
        # create melted layers class 
        self.melted_layers = MeltedLayers(layers, fully_melted)

        # remove layers that were completely melted 
        removed = 0 # accounts for indexes of layers changing with loop
        for layer in fully_melted:
            layers.remove_layer(layer-removed)
            removed += 1

        # remove fully melted layers from layermelt
        mask = np.ones(len(layermelt))
        mask[fully_melted] = False
        layermelt = layermelt[np.array(mask,dtype=bool)]

        # CHECK MASS CONSERVATION
        change = np.sum(layers.lice + layers.lwater) - initial_mass
        if len(fully_melted) > 0: # account for melted layers
            change += np.sum(self.melted_layers.mass)
        assert np.abs(change) < prms.mb_threshold, f'melting failed mass conservation in {self.output.out_fn}'

        return layermelt
        
    def percolation(self,layermelt,rainfall=0):
        """
        Updates the liquid water content in each layer
        with downward percolation and removes melted
        mass from layer dry mass.

        Parameters
        ==========
        layermelt: np.ndarray
            Array containing melt amount for each layer
        rainfall : float
            Additional liquid water input from 
            rainfall [kg m-2]

        Returns
        -------
        runoff : float
            Runoff of liquid water lost to system [kg m-2]
        """
        # get classes
        layers = self.layers

        # CONSTANTS
        DENSITY_WATER = prms.density_water
        DENSITY_ICE = prms.density_ice
        FRAC_IRREDUC = prms.Sr
        dt = self.dt

        # get index of percolating (snow/firn) layers
        snow_firn_idx = np.concatenate([layers.snow_idx,layers.firn_idx])
        # check if there is an ice layer within the snow/firn
        if len(snow_firn_idx) > 0 and layers.ice_idx[0] < snow_firn_idx[-1]:
            if layers.ice_idx[0] == 0 and len(layers.snow_idx) == 0: 
                # surface ice layer with firn underneath: all water runs off
                snow_firn_idx = []
            else:
                # impermeable layer caused by densification/refreeze
                snow_firn_idx = snow_firn_idx[:layers.ice_idx[0]]

        # initialize variables
        initial_mass = np.sum(layers.lice + layers.lwater)
        rain_bool = rainfall > 0
        runoff = 0  # any flow that leaves the point laterally

        # get incoming water flux
        if len(self.melted_layers.mass) > 0:
            # sum of rainfall and mass of fully melted layers
            water_in = rainfall + np.sum(self.melted_layers.mass)
        else:
            # no melted layers, incoming water is just rain
            water_in = rainfall

        if len(snow_firn_idx) > 0:
            # LAYERS IN
            lm = layers.lice.copy()[snow_firn_idx]
            lw = layers.lwater.copy()[snow_firn_idx]
            lh = layers.lheight.copy()[snow_firn_idx]
            layermelt_sf = layermelt[snow_firn_idx]

            # calculate volumetric fractions (theta)
            # before moving melt because liquid water can exceed layer capacity
            vol_f_liq = lw / (lh*DENSITY_WATER)
            vol_f_ice = lm / (lh*DENSITY_ICE)
            porosity = 1 - vol_f_ice
            vol_f_liq[vol_f_liq > porosity] = porosity[vol_f_liq > porosity]

            # remove / move snow melt to layer water
            lm -= layermelt_sf
            lh -= layermelt_sf / layers.ldensity[snow_firn_idx]
            lw += layermelt_sf

            # reduce layer refreeze (refreeze melts first)
            layers.lrefreeze[snow_firn_idx] -= layermelt_sf
            layers.lrefreeze[layers.lrefreeze < 0] = 0

            # initialize flow into the top layer
            q_out = water_in / dt # q is a rate so need dt
            q_in_store = []
            q_out_store = []
            for layer in snow_firn_idx:
                # set flow in equal to flow out of the previous layer
                q_in = q_out

                # calculate flow out of layer i
                q_out = DENSITY_WATER*lh[layer]/dt * (
                        vol_f_liq[layer]-FRAC_IRREDUC*porosity[layer])
                
                # check limits on flow out (q_out)
                # first check underlying layer holding capacity
                if layer < len(porosity) - 1 and vol_f_liq[layer] <= 0.3:
                    next = layer+1
                    lim = DENSITY_WATER*lh[next]/dt * (1-vol_f_ice[next]-vol_f_liq[next])
                else: # no limit on bottom layer
                    lim = np.inf
                # cannot have more flow out than flow in + existing water
                lim = min(lim,q_in + lw[layer])
                q_out = min(q_out,lim)
                # cannot be negative
                q_out = max(0,q_out)

                # layer mass balance
                lw[layer] += (q_in - q_out)*dt
                q_in_store.append(q_in)
                q_out_store.append(q_out)

                # layer cannot contain more water than there is pore space
                layer_porosity = max(1 - lm[layer] / (lh[layer]*DENSITY_ICE),0)
                water_lim = lh[layer]*layer_porosity*DENSITY_WATER
                if lw[layer] > water_lim: # excess runs off
                    runoff += lw[layer] - water_lim
                    lw[layer] = water_lim

            # LAYERS OUT
            layers.lheight[snow_firn_idx] = lh
            layers.lwater[snow_firn_idx] = lw
            layers.lice[snow_firn_idx] = lm
            runoff += q_out*dt + np.sum(layermelt[layers.ice_idx])

            # remove melted ice mass (only snow/firn mass was handled above)
            for layer in layers.ice_idx:
                layers.lice[layer] -= layermelt[layer]
                layers.lheight[layer] -= layermelt[layer] / layers.ldensity[layer]

            # move LAPs 
            if self.args.switch_LAPs == 1:
                self.move_LAPs(np.array(q_out_store),rain_bool,snow_firn_idx)
        else:
            # no percolation, but need to move melt to runoff
            layers.lice -= layermelt
            layers.lheight -= layermelt / layers.ldensity
            runoff += water_in + np.sum(layermelt)

        # make sure layers didn't get too small from removing melt
        layers.check_layer_sizes()

        # CHECK MASS CONSERVATION
        ins = water_in
        outs = runoff
        change = np.sum(layers.lice + layers.lwater) - initial_mass
        assert np.abs(change - (ins-outs)) < prms.mb_threshold, f'percolation failed mass conservation in {self.output.out_fn}'
        return runoff
        
    def move_LAPs(self,q_out,rain_bool,snow_firn_idx):
        """
        Moves LAPs vertically through the snow and firn
        layers according to water flow from percolation.

        Parameters
        ==========
        q_out : np.ndarray
            Water flowrate out of each layer [kg m-2 s-1]
        rain_bool : Bool
            Raining or not?
        snow_firn_idx : np.ndarray
            Indices of snow and firn layers
        """
        # get classes
        layers = self.layers
        enbal = self.enbal

        # CONSTANTS
        PARTITION_COEF_BC = prms.ksp_BC
        PARTITION_COEF_OC = prms.ksp_OC
        PARTITION_COEF_DUST = prms.ksp_dust
        dt = prms.dt

        # LAYERS IN
        lw = layers.lwater[snow_firn_idx]
        lm = layers.lice[snow_firn_idx]

        # layer mass of each species in kg m-2
        mBC = layers.lBC[snow_firn_idx]
        mOC = layers.lOC[snow_firn_idx]
        mdust = layers.ldust[snow_firn_idx]

        # get wet deposition into top layer if it's raining
        if rain_bool and prms.switch_LAPs == 1: # Switch runs have no BC
            mBC[0] += enbal.bcwet * dt
            mOC[0] += enbal.ocwet * dt
            mdust[0] += enbal.dustwet * prms.ratio_DU3_DUtot * dt

        # layer mass mixing ratio in kg kg-1
        cBC = mBC / (lw + lm)
        cOC = mOC / (lw + lm)
        cdust = mdust / (lw + lm)

        # add LAPs from fully melted layers
        if self.melted_layers != 0:
            m_BC_in_top = np.array(np.sum(self.melted_layers.BC) / dt)
            m_OC_in_top = np.array(np.sum(self.melted_layers.OC) / dt)
            m_dust_in_top = np.array(np.sum(self.melted_layers.dust) / dt)
        else:
            m_BC_in_top = np.array([0],dtype=float) 
            m_OC_in_top = np.array([0],dtype=float) 
            m_dust_in_top = np.array([0],dtype=float)

        # partition in aqueous phase for incoming flux
        m_BC_in_top *= PARTITION_COEF_BC
        m_OC_in_top *= PARTITION_COEF_OC
        m_dust_in_top *= PARTITION_COEF_DUST

        # inward fluxes = outward fluxes from previous layer
        m_BC_in = PARTITION_COEF_BC*q_out[:-1]*cBC[:-1]
        m_OC_in = PARTITION_COEF_OC*q_out[:-1]*cOC[:-1]
        m_dust_in = PARTITION_COEF_DUST*q_out[:-1]*cdust[:-1]
        m_BC_in = np.append(m_BC_in_top,m_BC_in)
        m_OC_in = np.append(m_OC_in_top,m_OC_in)
        m_dust_in = np.append(m_dust_in_top,m_dust_in)

        # outward fluxes are simply (flow out)*(concentration of the layer)
        m_BC_out = PARTITION_COEF_BC*q_out*cBC
        m_OC_out = PARTITION_COEF_OC*q_out*cOC
        m_dust_out = PARTITION_COEF_DUST*q_out*cdust

        # mass balance on each constituent
        dmBC = (m_BC_in - m_BC_out)*dt
        dmOC = (m_OC_in - m_OC_out)*dt
        dmdust = (m_dust_in - m_dust_out)*dt
        mBC += dmBC.astype(float)
        mOC += dmOC.astype(float)
        mdust += dmdust.astype(float)

        # LAYERS OUT
        layers.lBC[snow_firn_idx] = mBC
        layers.lOC[snow_firn_idx] = mOC
        layers.ldust[snow_firn_idx] = mdust
        return
    
    def refreezing(self):
        """
        Calculates refreeze in layers due to temperatures 
        below freezing with liquid water content.

        Returns:
        --------
        refreeze : float
            Total amount of refreeze [kg m-2]
        """
        # get classes
        layers = self.layers

        # CONSTANTS
        HEAT_CAPACITY_ICE = prms.Cp_ice
        DENSITY_ICE = prms.density_ice
        LH_RF = prms.Lh_rf

        # LAYERS IN
        snow_firn_idx = np.concatenate([layers.snow_idx,layers.firn_idx])
        lT = layers.ltemp.copy()[snow_firn_idx]
        lw = layers.lwater.copy()[snow_firn_idx]
        lm = layers.lice.copy()[snow_firn_idx]
        lh = layers.lheight.copy()[snow_firn_idx]

        # define initial mass for conservation check
        initial_mass = np.sum(layers.lice + layers.lwater)

        # initialize refreeze at  0
        refreeze = np.zeros(len(snow_firn_idx))

        # loop through layers
        for layer, T in enumerate(lT):
            if T < 0. and lw[layer] > 0:
                # calculate potential for refreeze [J m-2]
                E_cold = np.abs(T)*lm[layer]*HEAT_CAPACITY_ICE  # cold content available 
                E_water = lw[layer]*LH_RF  # amount of water to freeze
                E_pore = (DENSITY_ICE*lh[layer]-lm[layer])*LH_RF # pore space available
                
                # calculate amount of refreeze in kg m-2
                dm_ref = np.min([abs(E_cold),abs(E_water),abs(E_pore)])/LH_RF

                # add refreeze to array in kg m-2
                refreeze[layer] = dm_ref

                # add refreeze to layer ice mass
                lm[layer] += dm_ref
                # update layer temperature from latent heat
                lT[layer] = min(0,-(E_cold-dm_ref*LH_RF)/(HEAT_CAPACITY_ICE*lm[layer]))

                # update water content
                lw[layer] = max(0,lw[layer]-dm_ref)
        
        # update refreeze with new refreeze content
        layers.drefreeze[snow_firn_idx] = refreeze      # change in refreeze this timestep
        layers.lrefreeze[snow_firn_idx] += refreeze     # total layer refrozen mass

        # LAYERS OUT
        layers.ltemp[snow_firn_idx] = lT
        layers.lwater[snow_firn_idx] = lw
        layers.lice[snow_firn_idx] = lm
        layers.update_layer_props()

        # CHECK MASS CONSERVATION
        change = np.sum(layers.lice + layers.lwater) - initial_mass
        assert np.abs(change) < prms.mb_threshold, f'refreezing failed mass conservation in {self.output.out_fn}'
        return np.sum(refreeze)
    
    def densification(self):
        """
        Calculates densification of layers due to 
        compression from overlying mass.
        """
        # get classes
        layers = self.layers

        # CONSTANTS
        GRAVITY = prms.gravity
        R = prms.R_gas
        VISCOSITY_SNOW = prms.viscosity_snow
        rho = prms.constant_snowfall_density
        DENSITY_FRESH_SNOW = rho if rho else 50
        DENSITY_ICE = prms.density_ice
        dt = prms.daily_dt

        # LAYERS IN
        snowfirn_idx = np.append(layers.snow_idx,layers.firn_idx)
        lp = layers.ldensity.copy()
        lT = layers.ltemp.copy()
        lm = layers.lice.copy()
        lw = layers.lwater.copy()

        # define initial mass for conservation check
        initial_mass = np.sum(layers.lice + layers.lwater)

        # Boone / Anderson (1976) method (COSIPY)
        if prms.method_densification in ['Boone']:
            # EMPIRICAL PARAMETERS
            c1 = 2.7e-6     # s-1 (2.7e-6)
            c2 = 0.042      # K-1 (0.042)
            c3 = 0.046      # m3 kg-1 (0.046)
            c4 = 0.081      # K-1 (0.081)
            c5 = float(self.args.Boone_c5)

            for layer in snowfirn_idx:
                weight_above = GRAVITY*np.sum(lm[:layer]+lw[:layer])
                viscosity = VISCOSITY_SNOW*np.exp(c4*(0.-lT[layer])+c5*lp[layer])

                # get change in density
                mass_term = weight_above/viscosity
                temp_term = -c2*(0.-lT[layer])
                dens_term = -c3*max(0,lp[layer]-DENSITY_FRESH_SNOW)
                dRho = (mass_term+c1*np.exp(temp_term+dens_term))*lp[layer]*dt
                lp[layer] += dRho

        # Herron Langway (1980) method
        elif prms.method_densification in ['HerronLangway']:
            # yearly accumulation is the maximum layer snow mass in mm w.e. yr-1
            a = layers.max_snow / (dt*365) # kg m-2 = mm w.e.
            k = np.zeros_like(lp)
            b = np.zeros_like(lp)
            for layer,density in enumerate(lp[snowfirn_idx]):
                lTK = lT[layer] + 273.15
                if density < 550:
                    b[layer] = 1
                    k[layer] = 11*np.exp(-10160/(R*lTK))
                else:
                    b[layer] = 0.5
                    k[layer] = 575*np.exp(-21400/(R*lTK))
            dRho = k*a**b*(DENSITY_ICE - lp)/DENSITY_ICE*dt
            lp += dRho

        # Kojima (1967) method (JULES)
        elif prms.method_densification in ['Kojima']:
            NU_0 = 1e7      # Pa s
            RHO_0 = 50      # kg m-3
            k_S = 4000      # K
            T_m = 0. + 273.15
            for layer in snowfirn_idx:
                weight_above = GRAVITY*np.sum(lm[:layer]+lw[:layer])

                # get change in density
                T_K = lT[layer] + 273.15
                exp_term = np.exp(k_S/T_m - k_S/T_K - lp[layer]/RHO_0)
                dRho = lp[layer]*weight_above/NU_0*exp_term
                lp[layer] += dRho

        # LAYERS OUT
        layers.ldensity = lp
        layers.lheight = lm / lp
        layers.update_layer_props('depth')

        # check if new firn or ice layers were created
        layers.update_layer_types()

        # CHECK MASS CONSERVATION
        change = np.sum(layers.lice + layers.lwater) - initial_mass
        assert np.abs(change) < prms.mb_threshold, f'densification failed mass conservation in {self.output.out_fn}'
        return
    
    def phase_changes(self):
        """
        Calculates mass lost or gained from latent heat
        exchange (sublimation, deposition, evaporation,
        or condensation).
        """
        # get classes
        layers = self.layers
        surface = self.surface
        enbal = self.enbal

        # CONSTANTS
        LV_SUB = prms.Lv_sub
        LV_VAP = prms.Lv_evap

        # get initial mass for conservation check
        initial_mass = np.sum(layers.lice + layers.lwater)

        # get mass fluxes from latent heat
        if surface.stemp < 0.:
            # SUBLIMATION / DEPOSITION
            dm = enbal.lat*self.dt/(LV_SUB) # kg m-2
            # yes solid-vapor fluxes
            sublimation = -1*min(dm,0)
            deposition = max(dm,0)
            # no liquid-vapor fluxes
            evaporation = 0
            condensation = 0

            # check if dm causes negativity
            if layers.lice[0] + dm < 0: 
                # remove mass from next layer
                remaining_dm = -(np.abs(dm) - layers.lice)
                layers.lice[1] += remaining_dm
                layers.lheight[1] += remaining_dm / layers.ldensity[1]
                layers.lice[0] = 0
                layers.remove_layer(0)
            else:
                # add mass to layer if it doesn't cause negativity
                layers.lice[0] += dm
        else:
            # EVAPORATION / CONDENSATION
            dm = enbal.lat*self.dt/(LV_VAP) # kg m-2
            # no solid-vapor fluxes
            sublimation = 0
            deposition = 0
            # yes liquid-vapor fluxes
            evaporation = -1*min(dm,0)
            condensation = max(dm,0)

            # check if dm causes negativity
            if layers.lwater[0] + dm < 0: 
                layer = 0
                while np.abs(dm) > 0 and layer < layers.nlayers:
                    # calculate the maximum water loss possible for the current layer
                    change = min(np.abs(dm), layers.lwater[layer])
                    layers.lwater[layer] -= change
                    
                    # reduce the absolute magnitude of dm
                    if dm < 0:
                        dm += change  # increase dm towards 0 when negative
                    else:
                        dm -= change  # decrease dm towards 0 when positive
                    layer += 1
                
                if layer == layers.nlayers:
                    # no water is left to handle the remaining dm
                    evaporation += dm  # add the leftover dm to evaporation
            else:
                # add water to layer if it doesn't cause negativity
                layers.lwater[0] += dm

        # set vapor fluxes to self
        self.sublimation = sublimation
        self.deposition = deposition
        self.evaporation = evaporation
        self.condensation = condensation
        self.vapor_solid = sublimation if sublimation != 0 else deposition
        self.vapor_liquid = evaporation if evaporation != 0 else condensation
        
        # CHECK MASS CONSERVATION
        ins = deposition + condensation
        outs = sublimation + evaporation
        change = np.sum(layers.lice + layers.lwater) - initial_mass
        assert np.abs(change - (ins-outs)) < prms.mb_threshold, f'phase change failed mass conservation in {self.output.out_fn}'
        return
      
    def thermal_conduction(self):
        """
        Resolves the temperature profile with vertical
        heat conduction following the Forward-in-Time-
        Central-in-Space (FTCS) scheme

        Parameters
        ==========
        layers
            Class object from pebsi.layers
        surftemp : float
            Surface temperature [C]
        """        
        # get classes
        layers = self.layers
        surftemp = self.surface.stemp

        # CONSTANTS
        CP_ICE = prms.Cp_ice
        DENSITY_ICE = prms.density_ice
        DENSITY_WATER = prms.density_water
        TEMP_TEMP = prms.temp_temp
        TEMP_DEPTH = prms.temp_depth
        K_ICE = prms.k_ice
        K_WATER = prms.k_water
        K_AIR = prms.k_air
        MAX_DT = prms.max_temp_change

        # do not need this function if glacier is completely ripe
        if np.sum(layers.ltemp) == 0.:
            return
        
        # check layer sizes for numeric stability
        layers.check_layer_sizes()

        # determine layers that are below temperate ice depth
        if layers.ice_idx[0] > 0:
            # if there is snow/firn, adjust to be relative to the ice surface
            TEMP_DEPTH += layers.ldepth[layers.ice_idx[0] - 1]
        temperate_idx = np.where(layers.ldepth > TEMP_DEPTH)[0]
        if len(temperate_idx) < 1:
            temperate_idx = [layers.nlayers - 1]
        diffusing_idx = np.arange(temperate_idx[0])
        layers.ltemp[temperate_idx] = TEMP_TEMP

        # LAYERS IN
        nl = len(diffusing_idx)
        lh = layers.lheight[diffusing_idx]
        lp = layers.ldensity[diffusing_idx]
        lT_prev = layers.ltemp[diffusing_idx]
        lm = layers.lice[diffusing_idx]
        lw = layers.lwater[diffusing_idx]
        lT = layers.ltemp[diffusing_idx]

        # get snow/firn layer conductivity 
        ice_idx = layers.ice_idx
        if prms.method_conductivity in ['Sauter']:
            f_ice = (lm/DENSITY_ICE) / lh
            f_liq = (lw/DENSITY_WATER) / lh
            f_air = 1 - f_ice - f_liq
            f_air[f_air < 0] = 0
            lcond = f_ice*K_ICE + f_liq*K_WATER + f_air*K_AIR
        elif prms.method_conductivity in ['VanDusen']:
            lcond = 0.21e-01 + 0.42e-03*lp + 0.22e-08*lp**3
        elif prms.method_conductivity in ['Douville']:
            lcond = 2.2*np.power(lp/DENSITY_ICE,1.88)
        elif prms.method_conductivity in ['Jansson']:
            lcond = 0.02093 + 0.7953e-3*lp + 1.512e-12*lp**4
        elif prms.method_conductivity in ['OstinAndersson']:
            lcond = -8.71e-3 + 0.439e-3*lp + 1.05e-6*lp**2
        # get ice conductivity (constant)
        diffusing_ice_idx = list(set(ice_idx)&set(diffusing_idx))
        if len(diffusing_ice_idx) > 0:
            lcond[diffusing_ice_idx] = K_ICE

        # get timestep for heat equation
        dt_heat = self.dt / prms.n_heat_steps

        # check number of layers
        if nl > 2:
            # loop through thermal conduction equation
            for _ in range(prms.n_heat_steps):
                # heights of imaginary average bins between layers
                up_lh = np.array([np.mean(lh[i:i+2]) for i in range(nl-2)])  # upper layer 
                dn_lh = np.array([np.mean(lh[i+1:i+3]) for i in range(nl-2)])  # lower layer

                # conductivity
                up_kcond = np.array([np.mean(lcond[i:i+2]*lh[i:i+2]) for i in range(nl-2)]) / up_lh
                dn_kcond = np.array([np.mean(lcond[i+1:i+3]*lh[i+1:i+3]) for i in range(nl-2)]) / dn_lh

                # density
                up_dens = np.array([np.mean(lp[i:i+2]) for i in range(nl-2)]) / up_lh
                dn_dens = np.array([np.mean(lp[i+1:i+3]) for i in range(nl-2)]) / dn_lh

                # top layer uses surftemp boundary condition
                surf_heat_0 = up_kcond[0]*2/(up_dens[0]*lh[0])*(surftemp-lT_prev[0])
                subsurf_heat_0 = dn_kcond[0]/(up_dens[0]*up_lh[0])*(lT_prev[0]-lT_prev[1])
                lT[0] = lT_prev[0] + (surf_heat_0 - subsurf_heat_0)*dt_heat/(CP_ICE*lh[0])

                # if top layer of snow is very thin on top of ice, it can break this calculation
                if lT[0] > 0 or lT[0] < -50: 
                    lT[0] = np.mean([surftemp,lT_prev[1]])

                # middle layers solve heat equation
                surf_heat = up_kcond/(up_dens*up_lh)*(lT_prev[:-2]-lT_prev[1:-1])
                subsurf_heat = dn_kcond/(dn_dens*dn_lh)*(lT_prev[1:-1]-lT_prev[2:])
                dT = (surf_heat - subsurf_heat)*dt_heat/(CP_ICE*lh[1:-1])

                # limit the temperature change 
                exceeded_max = np.where(np.abs(dT) > MAX_DT / prms.n_heat_steps)[0]
                dT[exceeded_max] = MAX_DT / prms.n_heat_steps
                lT[1:-1] = lT_prev[1:-1] + dT

                # update 'previous' temperature in loop
                lT_prev = lT.copy()

        # cases for less than 3 layers do not need to be iterated
        elif nl > 1:
            lT = np.array([surftemp/2,0])
        else:
            lT = np.array([0])

        # LAYERS OUT
        layers.ltemp[diffusing_idx] = lT
        return 
    
    def end_of_summer(self):
        """
        Checks prognostically if enough snow will fall
        in the upcoming days to constitute the start
        of the accumulation season. If so, snow layers
        are transformed to firn and cumulative refreeze
        is reset to 0.
        """
        # get classes
        layers = self.layers

        # exit function if there is no snow
        if len(layers.snow_idx) == 0:
            return
        
        # CONSTANTS
        NDAYS = prms.new_snow_days
        SNOW_THRESHOLD = prms.new_snow_threshold
        T_LOW = prms.snow_threshold_low
        T_HIGH = prms.snow_threshold_high
        FIRN_AGE = prms.firn_age

        # only merge firn if there is old snow
        dates_snow = pd.to_datetime(layers.lage[layers.snow_idx])
        snow_age_series = pd.Series(self.time - dates_snow)
        snow_age = np.array(snow_age_series.dt.days)
        
        if np.any(snow_age >= FIRN_AGE):
            # define rain vs snow scaling 
            rain_scale = np.linspace(1,0,20)
            temp_scale = np.linspace(T_LOW,T_HIGH,20)

            # index the temperature and precipitation of the upcoming period
            end_time = min(self.time_list[-1],self.time+pd.Timedelta(days=NDAYS))
            check_dates = pd.date_range(self.time,end_time,freq='h')
            check_temp = self.climate.cds.sel(time=check_dates)['temp'].values
            check_tp = self.climate.cds.sel(time=check_dates)['tp'].values

            # create array to mask tp to snow amounts
            mask = np.interp(check_temp,temp_scale,rain_scale)
            upcoming_snow = np.sum(check_tp*mask)
           
            # check if we are getting enough snow to surpass the threshold
            if upcoming_snow < SNOW_THRESHOLD:
                # not getting enough snow: exit function
                return
            else:
                # getting new snow: set the timestamp
                firn_merged_time = self.time

            # check which layers are old enough to merge
            merge_layers = np.where(snow_age >= FIRN_AGE)[0]

            # set age of layers to be the oldest layer
            layers.lage[merge_layers] = layers.lage[merge_layers[-1]]
            
            for _ in range(merge_layers[0], merge_layers[-1]):
                layers.merge_layers(merge_layers[0])
            layers.ltype[merge_layers[0]] = 'firn'
            if self.args.debug:
                print('Converted firn on',firn_merged_time)
            layers.update_layer_props([])   # only update firn_idx
            self.firn_converted = True

            # reset cumulative refreeze
            layers.lrefreeze *= 0
            return

    def current_state_prints(self):
        """
        Prints some useful information to keep track 
        of a model run.

        Parameters
        ==========
        timestamp : pd.Datetime
            Current timestep
        airtemp : float
            Air temperature [C]
        """
        # get classes
        timestamp = self.time
        airtemp = self.enbal.tempC

        # gather variables to print out
        layers = self.layers
        surftemp = self.surface.stemp
        albedo = self.surface.bba
        melte = np.mean(self.output.meltenergy_output[-720:])
        melt = np.sum(self.output.melt_output[-720:])
        accum = np.sum(self.output.accum_output[-720:])
        ended_month = (timestamp - pd.Timedelta(days=1)).month_name()
        year = timestamp.year if ended_month != 'December' else timestamp.year - 1

        layers.update_layer_props()
        snowdepth = np.sum(layers.lheight[layers.snow_idx])
        firndepth = np.sum(layers.lheight[layers.firn_idx])
        icedepth = np.sum(layers.lheight[layers.ice_idx])

        # begin prints
        print(f'MONTH COMPLETED: {ended_month} {year} with +{accum:.2f} and -{melt:.2f} m w.e.')
        print(f'Currently {airtemp:.2f} C with {melte:.0f} W m-2 melt energy')
        print(f'----------surface albedo: {albedo:.3f} -----------')
        print(f'-----------surface temp: {surftemp:.2f} C-----------')
        if len(layers.snow_idx) > 0:
            print(f'|       snow depth: {snowdepth:.2f} m      {len(layers.snow_idx)} layers      |')
        if len(layers.firn_idx) > 0:
            print(f'|       firn depth: {firndepth:.2f} m      {len(layers.firn_idx)} layers      |')
        print(f'|       ice depth: {icedepth:.2f} m      {len(layers.ice_idx)} layers      |')
        for l in range(min(2,layers.nlayers)):
            print(f'--------------------layer {l}---------------------')
            print(f'     T = {layers.ltemp[l]:.1f} C                 h = {layers.lheight[l]:.3f} m ')
            print(f'                 p = {layers.ldensity[l]:.0f} kg/m3')
            print(f'Water Mass : {layers.lwater[l]:.2f} kg/m2   Dry Mass : {layers.lice[l]:.2f} kg/m2')
        print('================================================')
        return
    
    def check_mass_conservation(self,mass_in,mass_out):
        """
        Checks mass was conserved within the last timestep
        
        Parameters
        ==========
        mass_in : float
            Sum of mass in (precipitation) (kg m-2)
        mass_out : float
            Sum of mass out (runoff) (kg m-2)
        """
        # difference in mass since the last timestep
        current_mass = np.sum(self.layers.lice + self.layers.lwater)
        diff = current_mass - self.previous_mass
        in_out = mass_in - mass_out
        if np.abs(diff - in_out) >= prms.mb_threshold and self.args.debug:
            # debugging print steps in case mass conservation is failed
            print(self.time,'discrepancy of',np.abs(diff - in_out) - prms.mb_threshold,self.output.out_fn)
            print('in',mass_in,'out',mass_out,'currently',current_mass,'was',self.previous_mass)
            print('ice before',self.lice_before,'ice after',np.sum(self.layers.lice))
            print('w before',self.lwater_before,'w after',np.sum(self.layers.lwater))
            print('melt',self.melt,'rfz',self.refreeze,'accum',self.accum)
        assert np.abs(diff - in_out) < prms.mb_threshold, f'Timestep {self.time} failed mass conservation in {self.output.out_fn}'
        
        # new initial mass
        self.previous_mass = current_mass
        self.lice_before = np.sum(self.layers.lice)
        self.lwater_before = np.sum(self.layers.lwater)
        return
    
    def check_glacier_exists(self):
        """
        Checks there is still a glacier. If not, ends 
        the run and saves the output.
        """
        # load layer height
        layerheight = np.sum(self.layers.lheight)
        if layerheight < prms.min_glacier_depth:
            # new end date
            start = self.time_list[0]
            end = self.time
            new_time = pd.date_range(start,end,freq='h')
            self.output.n_timesteps = len(new_time)

            # load the output
            with xr.open_dataset(self.output.out_fn) as dataset:
                ds = dataset.load()
                # chop it to the new end date
                ds = ds.sel(time=new_time)
            # store output
            ds.to_netcdf(self.output.out_fn)

            # save the data
            if self.args.store_data:
                self.output.store_data()
            print('Glacier fully melted in',self.args.out)
            
            # end the run
            self.exit(failed=False)
        return
    
    def delete_temp_files(self):
        """
        Deletes any temporary files that were created
        for parallel runs.
        """
        # delete inputs file
        if os.path.exists(self.surface.snicar_fn):
            if self.surface.snicar_fn.split('/')[-1] != 'inputs.yaml':
                os.remove(self.surface.snicar_fn)

        # delete ice spectrum file
        if os.path.exists(self.surface.ice_spectrum_fp):
            os.remove(self.surface.ice_spectrum_fp)
        return

    def exit(self,failed=True):
        """
        Exit function. Default usage sends an error 
        message if debug is on. Otherwise, exits the run.

        Parameters
        ==========
        failed : Bool
            If True, prints some layer properties for
            debugging. Else, ends the run with no prints.
        """
        self.delete_temp_files()
        if self.args.debug and failed:
            print('Failed in mass balance')
            print('Current layers',self.ltype)
            print('Layer temp:',self.ltemp)
            print('Layer density:',self.ldensity)
        sys.exit()    

class Output():
    """
    Output class which stores the data during the
    simulation and saves it to a netcdf file upon
    run completion.
    """
    def __init__(self,time,args):
        """
        Creates netcdf file where the model output 
        will be saved.

        Parameters
        ==========
        time : list-like
            List of times used in the simulation
        args : command-line args
        """
        # get unique filename
        self.out_fn = prms.output_filepath + args.out
        i = 0
        while os.path.exists(self.out_fn+f'{i}.nc'):
            i += 1
        self.out_fn += str(i)
        self.out_fn += '.nc'

        # info needed to create the output file
        self.n_timesteps = len(time)
        zeros = np.zeros([self.n_timesteps,prms.max_nlayers])

        # create variable name dict
        vn_dict = {'EB':['SWin','SWout','LWin','LWout','rain','ground',
                         'sensible','latent','meltenergy','albedo'],
                   'MB':['melt','refreeze','runoff','accum','cumrefreeze','dh',
                         'vaporsolid','vaporliquid'],
                   'temp':['airtemp','surftemp'],
                   'layers':['layertemp','layerdensity','layerwater','layerheight',
                             'layerage','layertype','layergrainsize','layerrefreeze',
                             'layerBC','layerOC','layerdust'],
                    'SW':['vis_albedo','SWin_sky','SWin_terr']}
        
        # create file to store outputs
        all_variables = xr.Dataset(data_vars = dict(
                SWin = (['time'],zeros[:,0],{'units':'W m-2'}),
                SWout = (['time'],zeros[:,0],{'units':'W m-2'}),
                SWin_sky = (['time'],zeros[:,0],{'units':'W m-2'}),
                SWin_terr = (['time'],zeros[:,0],{'units':'W m-2'}),
                LWin = (['time'],zeros[:,0],{'units':'W m-2'}),
                LWout = (['time'],zeros[:,0],{'units':'W m-2'}),
                rain = (['time'],zeros[:,0],{'units':'W m-2'}),
                ground = (['time'],zeros[:,0],{'units':'W m-2'}),
                sensible = (['time'],zeros[:,0],{'units':'W m-2'}),
                latent = (['time'],zeros[:,0],{'units':'W m-2'}),
                meltenergy = (['time'],zeros[:,0],{'units':'W m-2'}),
                albedo = (['time'],zeros[:,0],{'units':'0-1'}),
                vis_albedo = (['time'],zeros[:,0],{'units':'0-1'}),
                melt = (['time'],zeros[:,0],{'units':'m w.e.'}),
                refreeze = (['time'],zeros[:,0],{'units':'m w.e.'}),
                cumrefreeze = (['time'],zeros[:,0],{'units':'m w.e.'}),
                runoff = (['time'],zeros[:,0],{'units':'m w.e.'}),
                accum = (['time'],zeros[:,0],{'units':'m w.e.'}),
                vaporliquid = (['time'],zeros[:,0],{'units':'m w.e.'}),
                vaporsolid = (['time'],zeros[:,0],{'units':'m w.e.'}),
                airtemp = (['time'],zeros[:,0],{'units':'C'}),
                surftemp = (['time'],zeros[:,0],{'units':'C'}),
                layertemp = (['time','layer'],zeros,{'units':'C'}),
                layerwater = (['time','layer'],zeros,{'units':'kg m-2'}),
                layerrefreeze = (['time','layer'],zeros,{'units':'kg m-2'}),
                layerheight = (['time','layer'],zeros,{'units':'m'}),
                layerdensity = (['time','layer'],zeros,{'units':'kg m-3'}),
                layerBC = (['time','layer'],zeros,{'units':'ppb'}),
                layerOC = (['time','layer'],zeros,{'units':'ppb'}),
                layerdust = (['time','layer'],zeros,{'units':'ppm'}),
                layergrainsize = (['time','layer'],zeros,{'units':'um'}),
                layerage = (['time','layer'],zeros),
                layertype = (['time','layer'],zeros),
                dh = (['time'],zeros[:,0],{'units':'m'})
                ),
                coords=dict(
                    time=(['time'],time),
                    layer=(['layer'],np.arange(prms.max_nlayers))
                    ))
        # select variables from the specified input
        vars_list = vn_dict[prms.store_vars[0]]
        for var in prms.store_vars[1:]:
            assert var in vn_dict, 'Choose store_vars from [MB,EB,temp,layers,SW]'
            vars_list.extend(vn_dict[var])
        self.vars_list = vars_list
        
        # create the netcdf file to store output
        if args.store_data:
            assert os.path.exists(prms.output_filepath), f'Create output folder at {prms.output_filepath}'
            all_variables[self.vars_list].to_netcdf(self.out_fn)

        # ENERGY BALANCE OUTPUTS
        self.SWin_output = []       # incoming shortwave [W m-2]
        self.SWout_output = []      # outgoing shortwave [W m-2]
        self.LWin_output = []       # incoming longwave [W m-2]
        self.LWout_output = []      # outgoing longwave [W m-2]
        self.rain_output = []       # rain energy [W m-2]
        self.ground_output = []     # ground energy [W m-2]
        self.sensible_output = []   # sensible energy [W m-2]
        self.latent_output = []     # latent energy [W m-2]
        self.meltenergy_output = [] # melt energy [W m-2]
        self.albedo_output = []     # surface broadband albedo [-]

        # TEMPERATURE OUTPUTS
        self.airtemp_output = []    # downscaled air temperature [C]
        self.surftemp_output = []   # surface temperature [C]

        # MASS BALANCE OUTPUTS
        self.melt_output = []           # melt by timestep [m w.e.]
        self.refreeze_output = []       # refreeze by timestep [m w.e.]
        self.cumrefreeze_output = []    # cumulative refreeze by timestep [m w.e.]
        self.accum_output = []          # accumulation by timestep [m w.e.]
        self.runoff_output = []         # runoff by timestep [m w.e.]
        self.dh_output = []             # surface height change by timestep [m]
        self.vaporliquid_output = []    # liquid-vapor mass flux [m w.e.]  
        self.vaporsolid_output = []     # solid-vapor mass flux [m w.e.]  

        # DETAILED SHORTWAVE OUTPUTS
        self.SWin_sky_output = []   # incoming sky shortwave [W m-2]
        self.SWin_terr_output = []  # incoming terrain shortwave [W m-2]
        self.vis_albedo_output = [] # surface visible albedo [-]

        # LAYER OUTPUTS
        self.layertemp_output = dict()      # layer temperature [C]
        self.layerwater_output = dict()     # layer water content [kg m-2]
        self.layerdensity_output = dict()   # layer density [kg m-3]
        self.layerheight_output = dict()    # layer height [m]
        self.layerBC_output = dict()        # layer black carbon content [ppb]
        self.layerOC_output = dict()        # layer organic carbon content [ppb]
        self.layerdust_output = dict()      # layer dust content [ppm]
        self.layergrainsize_output = dict() # layer grain size [um]
        self.layerrefreeze_output = dict()  # layer refreeze [kg m-2]
        self.layerage_output = dict()       # layer age [datetime]
        self.layertype_output = dict()      # layer type [-]
        self.last_height = prms.initial_ice_depth+prms.initial_firn_depth+prms.initial_snow_depth
        return
    
    def store_timestep(self,massbal,enbal,surface,layers,step):
        """
        Appends the current values to each output list.

        Parameters
        ==========
        massbal
            Class object from pebsi.massbalance
        enbal
            Class object from pebsi.energybalance
        surface
            Class object from pebsi.surface
        layers
            Class object from pebsi.layers
        step : pd.Datetime
            Current timestamp
        """
        step = str(step)
        self.SWin_output.append(float(enbal.SWin))
        self.SWout_output.append(float(enbal.SWout))
        self.LWin_output.append(float(enbal.LWin))
        self.LWout_output.append(float(enbal.LWout))
        self.rain_output.append(float(enbal.rain))
        self.ground_output.append(float(enbal.ground))
        self.sensible_output.append(float(enbal.sens))
        self.latent_output.append(float(enbal.lat))
        self.meltenergy_output.append(float(surface.Qm))
        self.albedo_output.append(float(surface.bba))
        
        self.surftemp_output.append(float(surface.stemp))
        self.airtemp_output.append(float(enbal.tempC))

        self.melt_output.append(float(massbal.melt))
        self.refreeze_output.append(float(massbal.refreeze))
        self.cumrefreeze_output.append(float(np.sum(layers.lrefreeze))/prms.density_water)
        self.runoff_output.append(float(massbal.runoff))
        self.accum_output.append(float(massbal.accum))
        self.dh_output.append(np.sum(layers.lheight)-self.last_height)
        self.last_height = np.sum(layers.lheight)
        self.vaporliquid_output.append(massbal.vapor_liquid/prms.density_water)
        self.vaporsolid_output.append(massbal.vapor_solid/prms.density_water)

        self.layertemp_output[step] = layers.ltemp.copy()
        self.layerwater_output[step] = layers.lwater.copy()
        self.layerheight_output[step] = layers.lheight.copy()
        self.layerdensity_output[step] = layers.ldensity.copy()
        self.layerBC_output[step] = layers.lBC / layers.lheight * 1e6
        self.layerOC_output[step] = layers.lOC / layers.lheight * 1e6
        self.layerdust_output[step] = layers.ldust / layers.lheight * 1e3
        self.layergrainsize_output[step] = layers.lgrainsize.copy()
        self.layerrefreeze_output[step] = layers.lrefreeze.copy()
        self.layerage_output[step] = layers.lage.copy()
        mapping = {'snow': 0, 'firn': 1, 'ice': 2}
        self.layertype_output[step] = [mapping[l] for l in layers.ltype]

        self.vis_albedo_output.append(float(surface.vis_a))
        self.SWin_sky_output.append(float(enbal.SWin_sky))
        self.SWin_terr_output.append(float(enbal.SWin_terr))

    def store_data(self):
        """
        Saves all data in the netcdf file.
        """
        # load output dataset
        with xr.open_dataset(self.out_fn) as dataset:
            ds = dataset.load()
            # store variables
            if 'EB' in prms.store_vars:
                ds['SWin'].values = self.SWin_output
                ds['SWout'].values = self.SWout_output
                ds['LWin'].values = self.LWin_output
                ds['LWout'].values = self.LWout_output
                ds['rain'].values = self.rain_output
                ds['ground'].values = self.ground_output
                ds['sensible'].values = self.sensible_output
                ds['latent'].values = self.latent_output
                ds['meltenergy'].values = self.meltenergy_output
                ds['albedo'].values = self.albedo_output
            if 'MB' in prms.store_vars:
                ds['melt'].values = self.melt_output
                ds['refreeze'].values = self.refreeze_output
                ds['runoff'].values = self.runoff_output
                ds['accum'].values = self.accum_output
                ds['dh'].values = self.dh_output
                ds['cumrefreeze'].values = self.cumrefreeze_output
                ds['vaporliquid'].values = self.vaporliquid_output
                ds['vaporsolid'].values = self.vaporsolid_output
            if 'temp' in prms.store_vars:
                ds['airtemp'].values = self.airtemp_output
                ds['surftemp'].values = self.surftemp_output
            if 'SW' in prms.store_vars:
                ds['SWin_sky'].values = self.SWin_sky_output
                ds['SWin_terr'].values = self.SWin_terr_output
                ds['vis_albedo'].values = self.vis_albedo_output
            if 'layers' in prms.store_vars:
                layertemp_output = pd.DataFrame.from_dict(self.layertemp_output,orient='index')
                layerdensity_output = pd.DataFrame.from_dict(self.layerdensity_output,orient='index')
                layerheight_output = pd.DataFrame.from_dict(self.layerheight_output,orient='index')
                layerwater_output = pd.DataFrame.from_dict(self.layerwater_output,orient='index')
                layerBC_output = pd.DataFrame.from_dict(self.layerBC_output,orient='index')
                layerOC_output = pd.DataFrame.from_dict(self.layerOC_output,orient='index')
                layerdust_output = pd.DataFrame.from_dict(self.layerdust_output,orient='index')
                layergrainsize_output = pd.DataFrame.from_dict(self.layergrainsize_output,orient='index')
                layerrefreeze_output = pd.DataFrame.from_dict(self.layerrefreeze_output,orient='index')
                layerage_output = pd.DataFrame.from_dict(self.layerage_output,orient='index')
                layertype_output = pd.DataFrame.from_dict(self.layertype_output,orient='index')
                
                if len(layertemp_output.columns) < prms.max_nlayers:
                    n_columns = len(layertemp_output.columns)
                    for i in range(n_columns,prms.max_nlayers):
                        nans = np.zeros(self.n_timesteps)*np.nan
                        layertemp_output[str(i)] = nans
                        layerdensity_output[str(i)] = nans
                        layerheight_output[str(i)] = nans
                        layerwater_output[str(i)] = nans
                        layerBC_output[str(i)] = nans
                        layerOC_output[str(i)] = nans
                        layerdust_output[str(i)] = nans
                        layergrainsize_output[str(i)] = nans
                        layerrefreeze_output[str(i)] = nans
                        layerage_output[str(i)] = nans
                        layertype_output[str(i)] = nans
                else:
                    n = len(layertemp_output.columns)
                    print(f'Need to increase max_nlayers: currently have {n} layers')
                    self.exit()

                ds['layertemp'].values = layertemp_output
                ds['layerheight'].values = layerheight_output
                ds['layerdensity'].values = layerdensity_output
                ds['layerwater'].values = layerwater_output
                ds['layerBC'].values = layerBC_output
                ds['layerOC'].values = layerOC_output
                ds['layerdust'].values = layerdust_output
                ds['layergrainsize'].values = layergrainsize_output
                ds['layerrefreeze'].values = layerrefreeze_output
                ds['layerage'].values = layerage_output
                ds['layertype'].values = layertype_output

        # save NetCDF
        ds.to_netcdf(self.out_fn)

        return ds
    
    def add_vars(self):
        """
        Calculates additional variables from other
        existing variables in the output dataset.
        - Net shortwave radiation flux SWnet [W m-2]
        - Net longwave radiation flux LWnet [W m-2]
        - Net radiation NetRad [W m-2]
        - Net mass balance MB [m w.e.]
        """
        if 'SWin' in self.vars_list:
            with xr.open_dataset(self.out_fn) as dataset:
                ds = dataset.load()

                # add summed radiation terms
                SWnet = ds['SWin'] + ds['SWout']
                LWnet = ds['LWin'] + ds['LWout']
                NetRad = SWnet + LWnet
                ds['SWnet'] = (['time'],SWnet.values,{'units':'W m-2'})
                ds['LWnet'] = (['time'],LWnet.values,{'units':'W m-2'})
                ds['NetRad'] = (['time'],NetRad.values,{'units':'W m-2'})

                # add summed mass balance term
                MB = ds['accum'] + ds['refreeze'] - ds['melt']
                ds['MB'] = (['time'],MB.values,{'units':'m w.e.'})

            # save NetCDF 
            ds.to_netcdf(self.out_fn)
        return
    
    def add_basic_attrs(self,args,time_elapsed,climate):
        """
        Adds informational attributes to the output dataset.
        - glacier name, site, and elevation
        - length of the simulation (time_elapsed)
        - simulation dates (run_start and run_end)
        - list of variables from AWS/reanalysis
        - AWS and reanalysis dataset names
        - model run date
        - machine that ran the simulation (machine) 
        
        Parameters
        ==========
        args : command line arguments
        time_elapsed : float
            Run time for the whole simulation
        climate
            Class object from pebsi.climate
        """
        time_elapsed = f'{time_elapsed:.1f} s'
        elev = str(args.elev)+' m a.s.l.'

        # get information on variable sources
        which_re = prms.reanalysis
        re_str = ''
        if args.use_AWS:
            measured = climate.measured_vars
            AWS_name = args.glac_name
            AWS_elev = climate.AWS_elev
            which_AWS = f'{AWS_name} {AWS_elev}'
            AWS_str = ', '.join(measured)
            re_vars = [e for e in climate.all_vars if e not in measured]
            if 'vwind' in re_vars and not 'uwind' in re_vars:
                re_vars.remove('vwind')
            if 'uwind' in re_vars and not 'vwind' in re_vars:
                re_vars.remove('uwind')
            re_str += ', '.join(re_vars)
        else:
            re_str += 'all'
            AWS_str = 'none'
            which_AWS = 'none'
        
        # store new attributes
        with xr.open_dataset(self.out_fn) as dataset:
            ds = dataset.load()
            ds = ds.assign_attrs(glacier=args.glac_name,
                                 elevation=elev,
                                 site=str(args.site),
                                 from_AWS=AWS_str,
                                 which_AWS=which_AWS,
                                 from_reanalysis=re_str,
                                 which_re=which_re,
                                 run_start=str(args.startdate),
                                 run_end=str(args.enddate),
                                 model_run_date=str(pd.Timestamp.today()),
                                 time_elapsed=time_elapsed,
                                 run_by=prms.machine)
            if args.n_simultaneous_processes > 1:
                ds = ds.assign_attrs(task_id=str(args.task_id))

        # save NetCDF
        ds.to_netcdf(self.out_fn)
        print(f'~ Success: saved to {self.out_fn}')
        return
    
    def add_attrs(self,new_attrs):
        """
        Adds new attributes as a dict to the output dataset.

        Parameters
        ==========
        new_attrs : dict
            Attributes to store
        """
        with xr.open_dataset(self.out_fn) as dataset:
            ds = dataset.load()
            if not new_attrs:
                return ds
            ds = ds.assign_attrs(new_attrs)
        ds.to_netcdf(self.out_fn)
        return
    
    def get_output(self):
        """
        Returns the output dataset.
        """
        return xr.open_dataset(self.out_fn)
    
class MeltedLayers():
    def __init__(self, layers, fully_melted):
        self.water = layers.lwater[fully_melted]
        self.ice = layers.lice[fully_melted]
        self.mass = self.water + self.ice 
        self.BC = layers.lBC[fully_melted]
        self.OC = layers.lOC[fully_melted]
        self.dust = layers.ldust[fully_melted]