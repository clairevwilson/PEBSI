"""
Mass balance class for PEBSI

Contains main() function which executes
all energy and mass balance calculations
in an hourly time loop.

@author: clairevwilson
"""
# Built-in libraries
from tqdm import tqdm
import time as pytime
from types import SimpleNamespace
# External libraries
import jax
import jax.numpy as jnp
import pandas as pd
import xarray as xr
# Local libraries
import util.layers as layers

class MassBalanceDriver:
    def __init__(self, params, static_args):
        """
        Stores parameters and physical constants
        for accessing within mass balance functions.
        """
        self.prms = params 
        self.args = static_args

    # -------------------- ADDING NEW MASS --------------------
    def add_new_mass(self, state, forcings):
        # divide incoming precip to rain and snow
        rainfall, snowfall = self.get_precip_amounts(forcings)

        # add new snow to layers
        snowfall, state = self.add_accumulation(
            snowfall, state, forcings
        )

        # add dry deposition of light-absorbing particles
        state = self.add_dry_deposition(
            state, forcings
        )
        return rainfall, snowfall, state

    def get_precip_amounts(self, forcings):
        """
        Determines whether rain or snowfall occurred 
        and outputs amounts.

        Returns:
        --------
        rain, snow : float
            Specific mass of liquid and solid 
            precipitation [kg m-2]
        """
        # CONSTANTS
        SNOW_THRESHOLD_LOW = self.args.snow_threshold_low
        SNOW_THRESHOLD_HIGH = self.args.snow_threshold_high
        DENSITY_WATER = self.args.density_water

        # define rain vs snow scaling 
        rain_scale = jnp.linspace(0,1,20)
        temp_scale = jnp.linspace(SNOW_THRESHOLD_LOW,SNOW_THRESHOLD_HIGH,20)

        # calculate fraction of rain
        fraction_rain = jnp.interp(forcings.tempC, temp_scale, rain_scale)
        rain = forcings.tp * fraction_rain * DENSITY_WATER 
        snow = forcings.tp*(1-fraction_rain)*DENSITY_WATER

        # make sure amounts to the boundaries correctly
        rain = jnp.where(forcings.tempC <= SNOW_THRESHOLD_LOW, 0, rain)
        snow = jnp.where(forcings.tempC > SNOW_THRESHOLD_HIGH, 0, snow)
        
        return rain,snow # kg m-2

    def add_accumulation(self, snowfall, state, forcings):
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
        # get args
        args = self.args

        # grab forcings objects we need
        time_idx = forcings.time_idx
        wind = forcings.wind 
        tempC = forcings.tempC
        bcwet = forcings.bcwet
        ocwet = forcings.ocwet 
        dustwet = forcings.dustwet

        # add delayed snow to snowfall
        total_snowfall = snowfall + state.delayed_snow
        
        # define initial mass for conservation check
        initial_mass = jnp.sum(state.lice + state.lwater, axis=1)

        # check if using constant density for new snow
        if args.constant_snowfall_density:
            new_density = args.constant_snowfall_density
        else:
            # CROCUS formulation of density (Vionnet et al. 2012)
            new_density = jnp.maximum(109+6*(tempC-0.)+26*wind**0.5,50)
        
        # check if using constant grain size for new snow
        if args.constant_freshgrainsize:
            new_grainsize = args.constant_freshgrainsize
        else:
            # CLM formulation of grain size (CLM5.0 Documentation)
            airtemp = tempC
            new_grainsize = jnp.select(
                [airtemp <= -30, airtemp < 0],
                [54.5, 54.5 + 5 * (airtemp + 30)],
                default=204.5
            )

        # height and age of new layer
        new_height = total_snowfall/new_density
        new_age = jnp.zeros_like(total_snowfall)

        # wet deposition occurs in snowfall
        new_BC = bcwet * args.dt
        new_OC = ocwet * args.dt
        new_dust = dustwet * args.dt

        # pack properties of new layer into a namespace
        new_layer = {
            'ltemp': tempC,
            'lheight': snowfall / new_density,
            'ltype': jnp.full_like(tempC, 0),
            'lice': snowfall,
            'ldensity': new_density,
            'lgrainsize': new_grainsize,
            'lBC': new_BC,
            'lOC': new_OC,
            'ldust': new_dust,
            'lage': new_age,
            'lwater': jnp.full_like(tempC, 0),
            'lrefreeze': jnp.full_like(tempC, 0),
            'ldrefreeze': jnp.full_like(tempC, 0)
        }
        
        # define conditions for making a new layer for accumulation
        surf_not_snow = (state.ltype[:, 0] > 0)
        density_threshold = (state.ldensity[:, 0] > (new_density * 3))
        new_layer_cond = surf_not_snow | density_threshold

        # check for small surface snow layer (merge new snow with it no matter what)
        small_surf_layer = (state.lheight[:, 0] < 1e-3) & (state.ltype[:, 0] == 0)

        # define masks for possible cases
        create_new_mask = new_layer_cond & (~small_surf_layer) & (new_height >= 1e-3)
        delay_mask = new_layer_cond & (~small_surf_layer) & (new_height < 1e-3)
        merge_new_mask = (~new_layer_cond) | small_surf_layer

        action_taken_mask = create_new_mask | merge_new_mask

        # handle cases 1 & 3: create a new layer and merge snowfall with existing layer
        state = layers.add_top_layer(state, create_new_mask, new_layer)
        state = layers.merge_new_layer(state, merge_new_mask, new_layer, args)

        # handle case 2: delaying this snowfall to the next timestep
        updated_delayed_snow = jnp.where(delay_mask, snowfall, state.delayed_snow)
        updated_delayed_snow = jnp.where(action_taken_mask, 0.0, updated_delayed_snow)

        # update surface snow timestemp
        updated_last_snow = jnp.where(action_taken_mask, time_idx, state.last_snow)

        state = state._replace(
            delayed_snow = updated_delayed_snow,
            last_snow = updated_last_snow
        )
        
        # update layer depth from new layer heights
        state = layers.update_layer_props(state, args.density_ice)

        # accumulate mass error
        change = jnp.sum(state.lice + state.lwater, axis=1) - initial_mass
        mass_error = change - jnp.sum(snowfall)
        new_error_total = (state.cum_mass_error + mass_error)
        state = state._replace(cum_mass_error = new_error_total)        

        # return actual snowfall that was added, including any delayed_snow
        return snowfall, state

    def add_dry_deposition(self, state, forcings):
        """
        Adds dry deposition of light-absorbing particles
        to the surface layer.

        Parameters
        ==========
        layers
            Class object from pebsi.layers
        """
        # CONSTANTS
        dt = self.args.dt

        # define mask for snow/firn and top layer
        snow_firn_mask = state.ltype < 2
        layers_idx = jnp.arange(state.lice.shape[1])
        mask = snow_firn_mask & (layers_idx == 0)[None, :]

        # define previous amounts
        prev_BC = state.lBC
        prev_OC = state.lOC
        prev_dust = state.ldust

        # storage for updated properties
        new_properties = {}

        # add LAPs to top layer where it is snow or firn
        new_properties['lBC'] = jnp.where(
            mask, prev_BC + forcings.bcdry[:, None] * dt, prev_BC
        )

        new_properties['lOC'] = jnp.where(
            mask, prev_OC + forcings.ocdry[:, None] * dt, prev_OC
        )

        new_properties['ldust'] = jnp.where(
            mask, prev_dust + forcings.dustdry[:, None] * dt, prev_dust
        )

        return state._replace(**new_properties)
    
    # -------------------- DAILY / SUB-DAILY UPDATES --------------------
    def run_daily_routines(self, state, forcings, point_attrs):
        """
        Checks if we are running sup-hourly updates and
        executes get_daily_updates or skip_daily_updates.
        """
        albedo_TOD = self.args.albedo_TOD

        # parse time index for daily functions
        is_day_start = forcings.hour == 0
        is_albedo_step = jnp.any(forcings.hour == jnp.array(albedo_TOD))

        # either run or skip daily routines depending on hour of day
        state = jax.lax.cond(
            is_day_start | is_albedo_step,
            lambda s: self.get_daily_updates(s, forcings, point_attrs),
            lambda s: self.skip_daily_updates(s),
            state
        )
        return state
    
    def get_daily_updates(self, state, forcings, point_attrs):
        """
        Executes daily update functions (albedo, surrounding 
        albedo, days since snowfall).
        """
        albedo_TOD = self.args.albedo_TOD
        is_day_start = forcings.hour == 0
        is_albedo_step = jnp.any(forcings.hour == jnp.array(albedo_TOD))

        # === albedo ===
        # new_albedo = surface.get_albedo(state, forcings, point_attrs)

        # === surrounding albedo ===
        ALBEDO_GROUND = self.args.albedo_ground
        ALBEDO_SNOW = self.args.albedo_fresh_snow

        # update max snow if snow mass is greater than running max
        current_snow = jnp.sum(state.lice * state.snow_mask, axis=1)
        new_annual_max_snow = jnp.maximum(current_snow, state.annual_max_snow)
        
        # get fraction of max annual snow at each point and scale albedo
        snow_fraction = jnp.clip(current_snow / new_annual_max_snow, 0.0, 1.0)
        new_albedo_surr = ALBEDO_GROUND + (ALBEDO_SNOW - ALBEDO_GROUND) * snow_fraction

        # === days since snowfall ===
        time_idx = jnp.full(state.lice.shape[0], forcings.time_idx)
        hours_since_snowfall = time_idx - state.days_since_snowfall
        new_days_since_snowfall = jnp.round(hours_since_snowfall / 24).astype(jnp.int32)

        # only update the properties requested based on hour of day
        # updated_albedo = jnp.where(is_albedo_step, new_albedo, state.albedo)
        new_albedo_surr = jnp.where(
            is_day_start, new_albedo_surr, state.albedo_surr
        )
        new_annual_max_snow = jnp.where(
            is_day_start, new_annual_max_snow, state.annual_max_snow
        )
        new_days_since_snowfall = jnp.where(
            is_day_start, new_days_since_snowfall, state.days_since_snowfall
        )
        
        # return the modified state snapshot
        return state._replace(
            # albedo=updated_albedo,
            days_since_snowfall=new_days_since_snowfall,
            annual_max_snow=new_annual_max_snow,
            albedo_surr=new_albedo_surr
        )

    def skip_daily_updates(self, state):
        return state

    # -------------------- VERTICAL EXCHANGES --------------------
    def vertical_processes(self, state, forcings, point_attrs, fluxes):
        # subsurface heating and melting
        state, melt_array, mass_to_route = self.heating_melting(state, fluxes)

        # percolate meltwater and route LAPs
        for var, data in mass_to_route.items():
            fluxes[var] = jnp.sum(data, axis=1)
        state, runoff, fluxes = self.percolation(state, fluxes)
        state = self.route_particles(state, forcings, fluxes)

        return state 
    
    def heating_melting(self, state, fluxes):
        args = self.args
        layers_idx = jnp.arange(state.lice.shape[1])[None, :]

        # CONSTANTS
        EXTINCT_SNOW = args.extinct_coef_snow 
        EXTINCT_ICE = args.extinct_coef_ice 
        CP_ICE = args.Cp_ice
        CP_WATER = args.Cp_water
        LH_RF = args.Lh_rf
        dt = args.dt

        # load the amount of heat added to each layer
        Q_abs_surface = fluxes['melt_heat'][:, None]
        Q_penetrating = fluxes['SWnet_penetrating'][:, None]
        extinct_coefs = jnp.where(state.ice_mask, EXTINCT_ICE, EXTINCT_SNOW)

        f_top = jnp.exp(-1 * extinct_coefs * (state.ldepth - state.lheight / 2))
        f_bottom = jnp.exp(-1 * extinct_coefs * (state.ldepth + state.lheight / 2))
        Q_abs_layer = Q_penetrating * (f_top - f_bottom)

        layer_heat = jnp.where(layers_idx > 0, Q_abs_layer, Q_abs_surface)

        # recalculate layer temperatures using effective heat capacity
        lmass = state.lice + state.lwater
        cp_eff = (state.lice * CP_ICE + state.lwater * CP_WATER) / lmass

        # ---------- CASCADE FUNCTION ----------
        # transpose all arrays to (N_LAYERS, N_POINTS) for lax.scan
        scan_heat = jnp.transpose(layer_heat)
        scan_lice = jnp.transpose(state.lice)
        scan_lwater = jnp.transpose(state.lwater)
        scan_ltemp = jnp.transpose(state.ltemp)
        scan_lmass = jnp.transpose(lmass)
        scan_cp_eff = jnp.transpose(cp_eff)

        # pack all the inputs into a tuple
        layer_inputs = (scan_heat, scan_lice, scan_lwater, 
                        scan_ltemp, scan_lmass, scan_cp_eff)

        # define the sequential step (one layer at a time)
        def _melt_energy_cascade(carry, inputs):
            surplus_energy = carry
            lheat, lice, lwater, ltemp, lmass, lcp = inputs

            # energy flux into this layer, including surplus from above [J m-2]
            total_heat_in = (lheat + surplus_energy) * dt

            # energy needed to warm this layer to 0.
            energy_to_zero = -1 * ltemp * lmass * lcp
            
            # check if we have more energy tha needed to warm to 0.
            warmed_past_zero = total_heat_in > energy_to_zero
            
            # calculate temperature from all heat, regardless of how much
            partial_warm_temp = ltemp + (total_heat_in / (lmass * lcp))
            # clip ptemperature of points that were warmed past melting point
            intermediate_temp = jnp.where(warmed_past_zero, 0.0, partial_warm_temp)
            
            # leftover energy after melting to 0. converted to mass
            melt_energy_available = jnp.maximum(0.0, total_heat_in - energy_to_zero)
            potential_melt = melt_energy_available / LH_RF
            
            # actual melt is capped by how much solid ice physically exists in this layer
            actual_melt = jnp.minimum(potential_melt, lice)
            
            # calculate unspent melt energy that must cascade lower [W m-2]
            carry = jnp.maximum(
                0., (melt_energy_available - (actual_melt * args.Lh_rf)) / dt
            )

            # update layer states based on calculations
            updated_lice = lice - actual_melt
            updated_lwater = lwater + actual_melt
            updated_ltemp = intermediate_temp  # Will be 0.0 if any melting occurred

            # Return the carry for the next iteration, and save outputs for this layer
            return carry, (updated_lice, updated_lwater, updated_ltemp, actual_melt)

        # no additional energy enters the first layer
        initial_carry = jnp.zeros(state.lice.shape[0]) 
        
        # execute the lax.scan for cascading melt calculations
        _, (out_lice, out_lwater, out_ltemp, layermelt) = jax.lax.scan(
            _melt_energy_cascade, initial_carry, layer_inputs
        )

        # transpose updated quantities back to original (N_POINTS, N_LAYERS) shape
        properties = state._asdict()
        properties['lice'] = jnp.transpose(out_lice)
        properties['lwater'] = jnp.transpose(out_lwater)
        properties['ltemp'] = jnp.transpose(out_ltemp)
        
        # store updated properties
        state = state._replace(**properties)
        
        # layermelt is now actual melt amounts in (N_POINTS, N_LAYERS) shape
        layermelt = jnp.transpose(layermelt)

        # store the mass in the layers that are about to be deleted
        fully_melted_mask = properties['lice'] <= 0.001
        mass_to_route = {}
        mass_to_route['meltwater'] = jnp.where(fully_melted_mask, properties['lwater'], 0)
        mass_to_route['BC'] = jnp.where(fully_melted_mask, properties['lBC'], 0)
        mass_to_route['OC'] = jnp.where(fully_melted_mask, properties['lOC'], 0)
        mass_to_route['dust'] = jnp.where(fully_melted_mask, properties['ldust'], 0)

        # collapse grid to purge melted layers
        for _ in range(3):
            # looping more than once is rarely needed
            # only if multiple layers fully melted in the same point
            fully_melted_mask = properties['lice'] <= 0.001
            melt_point_mask = jnp.any(fully_melted_mask, axis=1)
            melt_layer_idx = jnp.argmax(fully_melted_mask.astype(jnp.int32), axis=1)

            # collapse one layer per point where mask if True
            state = layers.remove_layer(
                state, melt_point_mask, melt_layer_idx, args
            )

        return state, layermelt, mass_to_route
        
    def percolation(self, state, fluxes):
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
        properties = state._asdict()
        args = self.args

        # CONSTANTS
        DENSITY_WATER = args.density_water
        DENSITY_ICE = args.density_ice

        # transpose arrays for layer cascade
        scan_lice = jnp.transpose(properties['lice'])
        scan_lwater = jnp.transpose(properties['lwater'])
        scan_lheight = jnp.transpose(properties['lheight'])
        scan_ldensity = jnp.transpose(properties['ldensity'])
        scan_ice_mask = jnp.transpose(properties['ice_mask'])

        # calculate porosity
        vol_f_ice = scan_lice / (scan_lheight * DENSITY_ICE)
        porosity = jnp.maximum(0.0, 1.0 - vol_f_ice)

        # calculate irreducible water content
        if args.constant_irrwater:
            frac_irreduc = jnp.full_like(porosity, args.Sr)
        else:
            frac_irreduc = jnp.where(scan_ldensity > 500.0, args.Sr_dense, args.Sr_light)
        water_irreduc_capacity = porosity * scan_lheight * DENSITY_WATER * frac_irreduc

        # pack all the inputs
        layer_inputs = (scan_lwater, water_irreduc_capacity, scan_ice_mask)

        # define cascade function
        def _percolation_cascade(carry, inputs):
            # carry tracks: (current water flux in, cumulative runoff)
            q_in, current_runoff = carry
            lwater, capacity, is_barrier = inputs

            # if this is an ice layer, everything here and below runs off
            q_in = jnp.where(is_barrier, 0.0, q_in)
            q_in_blocked = jnp.where(is_barrier, q_in, 0.0)

            # remaining room for water before hitting irreducible water content
            available_room = jnp.maximum(0.0, capacity - lwater)
            q_out = jnp.maximum(0.0, q_in - available_room)

            # update layer liquid water content
            updated_lwater = lwater + (q_in - q_out)

            # accumulate runoff from both barriers and any water escaping the bottom layer
            next_runoff = current_runoff + q_in_blocked

            return (q_out, next_runoff), (updated_lwater, q_out)
        
        # water into top layer includes rainfall and melted layer mass
        water_in = fluxes['rainfall'] + fluxes['meltwater']
        initial_carry = (water_in, jnp.zeros_like(water_in))

        # execute the lax.scan for cascading melt calculations
        (bottom_q, runoff), (updated_lwater, q_outs) = jax.lax.scan(
            _percolation_cascade, initial_carry, layer_inputs
        )

        # sum any last leaks of water out the bottom with runoff
        total_runoff = runoff + bottom_q

        # update layer water
        state = state._replace(
            lwater=jnp.transpose(updated_lwater)
        )

        # store arrays of actual flow into and out of each layer
        fluxes['q_out'] = jnp.transpose(q_outs)

        return state, total_runoff, fluxes
        
    def route_particles(self, state, forcings, fluxes):
        """
        Moves LAPs vertically through the snow and firn
        layers according to water flow from percolation.

        Parameters
        ==========
        q_out : np.ndarray
            Water flow out of each layer [kg m-2]
        rain_bool : Bool
            Raining or not?
        snow_firn_idx : np.ndarray
            Indices of snow and firn layers
        """
        args = self.args
        properties = state._asdict()
        layers_idx = jnp.arange(state.lice.shape[1])[None, :]

        # CONSTANTS
        PARTITION_COEF_BC = args.ksp_BC
        PARTITION_COEF_OC = args.ksp_OC
        PARTITION_COEF_DUST = args.ksp_dust
        dt = args.dt

        # transpose arrays for layer cascade
        scan_lwater = jnp.transpose(properties['lwater'])
        scan_lice = jnp.transpose(properties['lice'])
        scan_mBC = jnp.transpose(properties['lBC'])
        scan_mOC = jnp.transpose(properties['lOC'])
        scan_mdust = jnp.transpose(properties['ldust'])
        scan_q_out = jnp.transpose(fluxes['q_out'])

        # pre-calculate mixing ratios [kg kg-1]
        scan_lmass = scan_lwater + scan_lice
        safe_mass = jnp.where(scan_lmass > 0.0, scan_lmass, 1.0)

        # inject wet-deposited particles into water flowing in
        BC_wet_flux = forcings.bcwet * dt 
        OC_wet_flux = forcings.ocwet * dt 
        dust_wet_flux = forcings.dustwet * dt

        # pack all the inputs
        layer_inputs = (safe_mass, scan_mBC, scan_mOC, scan_mdust, 
                        scan_q_out, (layers_idx[0, :] == 0))
        
        # define cascade function
        def _particle_cascade(carry, inputs):
            BCin, OCin, dustin = carry
            mass, mBC, mOC, mdust, q_out, is_surface = inputs 

            # add wet deposition if this is the top layer
            BCin = jnp.where(is_surface, BCin + BC_wet_flux, BCin)
            OCin = jnp.where(is_surface, OCin + OC_wet_flux, OCin)
            dustin = jnp.where(is_surface, dustin + dust_wet_flux, dustin)

            # instantly mix new particles into this layer
            cBC = jnp.where(mass > 0.0, mBC / mass, 0.0)
            cOC = jnp.where(mass > 0.0, mOC / mass, 0.0)
            cdust = jnp.where(mass > 0.0, mdust / mass, 0.0)

            # compute partition leaving the layer
            BCout_pot = PARTITION_COEF_BC * q_out * cBC 
            OCout_pot = PARTITION_COEF_OC * q_out * cOC 
            dustout_pot = PARTITION_COEF_DUST * q_out * cdust 

            # cap mass at amount previously in the layer
            BCout = jnp.minimum(BCout_pot, mBC)
            OCout = jnp.minimum(OCout_pot, mOC)
            dustout = jnp.minimum(dustout_pot, mdust)

            # update mass of particles 
            updated_mBC = mBC + (BCin - BCout)
            updated_mOC = mOC + (OCin - OCout)
            updated_mdust = mdust + (dustin - dustout)

            # carry forward mass leaving this layer
            next_carry = (BCout, OCout, dustout)
            outputs = (updated_mBC, updated_mOC, updated_mdust)

            return next_carry, outputs
        
        # initialize carry with the flow in from fully melted layers
        initial_carry = (fluxes['BC'], fluxes['OC'], fluxes['dust'])

        _, (out_mBC, out_mOC, out_mdust) = jax.lax.scan(
            _particle_cascade, initial_carry, layer_inputs
        )

        properties['lBC'] = jnp.transpose(out_mBC)
        properties['lOC'] = jnp.transpose(out_mOC)
        properties['ldust'] = jnp.transpose(out_mdust)
        
        state = state._replace(**properties)
        return state

    # def refreezing(self):
    #     """
    #     Calculates refreeze in layers due to temperatures 
    #     below freezing with liquid water content.

    #     Returns:
    #     --------
    #     refreeze : float
    #         Total amount of refreeze [kg m-2]
    #     """
    #     # get classes
    #     layers = self.layers
    #     args = self.args

    #     # CONSTANTS
    #     HEAT_CAPACITY_ICE = args.Cp_ice
    #     DENSITY_ICE = args.density_ice
    #     LH_RF = args.Lh_rf

    #     # LAYERS IN
    #     snow_firn_idx = np.concatenate([layers.snow_idx,layers.firn_idx])
    #     lT = layers.ltemp.copy()[snow_firn_idx]
    #     lw = layers.lwater.copy()[snow_firn_idx]
    #     lm = layers.lice.copy()[snow_firn_idx]
    #     lh = layers.lheight.copy()[snow_firn_idx]
    #     lmw = lm + lw

    #     # skip if no snow or firn
    #     if len(snow_firn_idx) < 1:
    #         return 0

    #     # define initial mass for conservation check
    #     initial_mass = np.sum(layers.lice + layers.lwater)

    #     # initialize refreeze at  0
    #     refreeze = np.zeros(len(snow_firn_idx))

    #     # loop through layers
    #     for layer, T in enumerate(lT):
    #         if T < 0. and lw[layer] > 0:
    #             # calculate potential for refreeze [J m-2]
    #             E_cold = np.abs(T)*lm[layer]*HEAT_CAPACITY_ICE  # cold content available 
    #             E_water = lw[layer]*LH_RF  # amount of water to freeze
    #             E_pore = (DENSITY_ICE*lh[layer]-lm[layer])*LH_RF # pore space available
                
    #             # calculate amount of refreeze in kg m-2
    #             dm_ref = np.min([abs(E_cold),abs(E_water),abs(E_pore)])/LH_RF

    #             # add refreeze to array in kg m-2
    #             refreeze[layer] = dm_ref

    #             # add refreeze to layer ice mass
    #             lm[layer] += dm_ref
    #             # update layer temperature from latent heat (cannot exceed 0)
    #             T_new = lT[layer] + dm_ref*LH_RF/(HEAT_CAPACITY_ICE*lm[layer])
    #             lT[layer] = min(0,T_new)

    #             # update water content
    #             lw[layer] = max(0,lw[layer]-dm_ref)
        
    #     # update refreeze with new refreeze content
    #     layers.drefreeze[snow_firn_idx] = refreeze      # change in refreeze this timestep
    #     layers.lrefreeze[snow_firn_idx] += refreeze     # total layer refrozen mass

    #     # LAYERS OUT
    #     layers.ltemp[snow_firn_idx] = lT
    #     layers.lwater[snow_firn_idx] = lw
    #     layers.lice[snow_firn_idx] = lm
    #     layers.update_layer_props()

    #     # CHECK MASS CONSERVATION
    #     change = np.sum(layers.lice + layers.lwater) - initial_mass
    #     assert np.abs(change) < args.mb_threshold, f'refreezing failed mass conservation in {self.output.out_fn}'
    #     return np.sum(refreeze)

    def evolve_grain_size(self, state, forcings):
        """
        Updates grain size according to wet and dry
        metamorphism, refreeze, and addition of fresh
        snow.
        """
        args = self.args

        # CONSTANTS
        WET_C = args.wet_grain_C
        PI = jnp.pi
        RFZ_GRAINSIZE = args.grainsize_rfz
        FIRN_GRAINSIZE = args.grainsize_firn
        ICE_GRAINSIZE = args.grainsize_ice
        CTOK = args.celsius_to_kelvin
        dt = args.dt

        # get temperatures
        airtemp = forcings.tempC
        surftemp = state.surftemp

        # find fresh snow grainsize
        if args.constant_freshgrainsize:
            FRESH_GRAINSIZE = args.constant_freshgrainsize
        else:
            FRESH_GRAINSIZE = jnp.select(
                [airtemp <= -30, airtemp < 0],
                [54.5, 54.5 + 5 * (airtemp + 30)],
                default=204.5
            )[:, jnp.newaxis]

        # define snow, firn and ice masks 
        snow_mask = state.snow_mask 
        ice_mask = state.ice_mask
            
        # grab layer masses everywhere
        m_total = state.lice
        m_refreeze = state.ldrefreeze       # differential refreeze: added this step
        m_snow = state.lice - m_refreeze   # "old snow" (includes old refreeze)
        
        # define mass fractions of old snow and refreeze
        f_snow = m_snow / m_total
        f_rfz = m_refreeze / m_total
        
        # calculate liquid water fraction everywhere
        mw_total = state.lwater + state.lice
        f_liq = state.lwater / mw_total    # fraction of total mass inc. liquid water

        # grab arrays needed for dry grain metamorphosis lookup
        dz = state.lheight
        T = state.ltemp + CTOK
        p = state.ldensity
        grainsize = state.lgrainsize
        
        # calculate surface temperature in K
        surftempK = surftemp + CTOK

        # DRY METAMORPHISM
        if args.constant_drdry:
            # apply constant drdry growth rate except where grainsize is too large
            drdry = jnp.where(
                grainsize < RFZ_GRAINSIZE,
                jnp.full_like(grainsize, args.constant_drdry * dt),
                jnp.zeros_like(grainsize)
            )
        else:
            # calculate dTdz in 2D
            dTdz = jnp.zeros_like(T)

            # top layer gradient utilizes surface temperature
            top_layer_val = (surftempK - (T[:, 0] * dz[:, 0] + T[:, 1] * dz[:, 1]) \
                            / (dz[:, 0] + dz[:, 1])) / dz[:, 0]
            dTdz = dTdz.at[:, 0].set(top_layer_val)

            # interior layers using vectorized slice formulation
            t_upper = (T[:, :-2] * dz[:, :-2] + T[:, 1:-1] * dz[:, 1:-1]) \
                            / (dz[:, :-2] + dz[:, 1:-1])
            t_lower = (T[:, 1:-1] * dz[:, 1:-1] + T[:, 2:] * dz[:, 2:]) \
                            / (dz[:, 1:-1] + dz[:, 2:])
            interior_vals = (t_upper - t_lower) / dz[:, 1:-1]
            dTdz = dTdz.at[:, 1:-1].set(interior_vals)

            # bottom layer gets assigned the same dTdz as the layer above
            dTdz = dTdz.at[:, -1].set(dTdz[:, -2])

            # take absolute value (direction does not matter)
            dTdz = jnp.abs(dTdz)

            # clip data to lookup table limits
            p = jnp.clip(p, 50.0, 400.0)
            dTdz = jnp.clip(dTdz, 0.0, 300.0)
            T = jnp.clip(T, 223.15, 273.15)

            # flatten matrices to feed grid interpolators in one parallel operation
            input_matrix = jnp.column_stack((T.ravel(), dTdz.ravel(), p.ravel()))

            tau = args.interp_tau(input_matrix).reshape(state.lice.shape)
            kap = args.interp_kap(input_matrix).reshape(state.lice.shape)
            dr0 = args.interp_dr0(input_matrix).reshape(state.lice.shape)

            # calculate denominator in drdry equation
            avoid_div_zero_mask = (tau + grainsize) <= FRESH_GRAINSIZE
            denominator = jnp.where(
                avoid_div_zero_mask, 
                tau + 1e-6, # avoid 0 denominator
                tau + grainsize - FRESH_GRAINSIZE
            )
            
            # determine actual dry grain growth rate from parameters
            drdrydt = dr0 * jnp.power(tau / denominator, 1.0 / kap) / dt
            drdry = drdrydt * dt

        # WET METAMORPHISM
        grainsize_m = grainsize / 1e6
        drwetdt = WET_C * (f_liq ** 3) / (4.0 * PI * (grainsize_m ** 2))
        drwet = drwetdt * dt * 1e6
        
        # accelerate grain growth?
        if args.option_accel_grains:
            F = jnp.exp(0.01 * layers.ldensity)
            drwet *= F

        # apply metamorphosis and refreezing 
        aged_grainsize = grainsize + drdry + drwet
        updated_grainsize = aged_grainsize * f_snow + RFZ_GRAINSIZE * f_rfz
        updated_snow_grainsize = jnp.clip(updated_grainsize, None, RFZ_GRAINSIZE)

        # store the updated snow grainsize
        all_updated_grainsize = jnp.where(
            snow_mask, updated_snow_grainsize, FIRN_GRAINSIZE
        )
        all_updated_grainsize = jnp.where(
            ice_mask, ICE_GRAINSIZE, all_updated_grainsize,
        )

        return all_updated_grainsize
    
    def get_roughness(self, state):
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
        layertype = state.ltype
        roughness = jnp.minimum(
            ROUGHNESS_FRESH_SNOW + AGING_RATE * state.days_since_snowfall, 
            ROUGHNESS_AGED_SNOW
        )

        # overwrite firn and ice values
        roughness = jnp.where(layertype[:, 0] == 1, ROUGHNESS_FIRN, roughness)
        roughness = jnp.where(layertype[:, 0] == 2, ROUGHNESS_ICE, roughness)

        # return roughness in m
        return roughness / 1000

    # def densification(self):
    #     """
    #     Calculates densification of layers due to 
    #     compression from overlying mass.
    #     """
    #     # get classes
    #     layers = self.layers
    #     args = self.args

    #     # CONSTANTS
    #     GRAVITY = args.gravity
    #     R = args.R_gas
    #     VISCOSITY_SNOW = args.viscosity_snow
    #     rho = args.constant_snowfall_density
    #     DENSITY_FRESH_SNOW = rho if rho else 50
    #     DENSITY_ICE = args.density_ice
    #     DENSITY_WATER = args.density_water
    #     CTOK = args.celsius_to_kelvin
    #     dt = args.daily_dt

    #     # LAYERS IN
    #     snowfirn_idx = np.append(layers.snow_idx,layers.firn_idx)
    #     lp = layers.ldensity.copy()
    #     lT = layers.ltemp.copy()
    #     lm = layers.lice.copy()
    #     lw = layers.lwater.copy()

    #     # define initial mass for conservation check
    #     initial_mass = np.sum(layers.lice + layers.lwater)

    #     # Boone / Anderson (1976) method (COSIPY)
    #     if args.method_densification in ['Boone']:
    #         # EMPIRICAL PARAMETERS
    #         c1 = args.Boone_c1
    #         c2 = args.Boone_c2
    #         c3 = args.Boone_c3
    #         c4 = args.Boone_c4
    #         c5 = args.Boone_c5

    #         for layer in snowfirn_idx:
    #             weight_above = GRAVITY*np.sum(lm[:layer]+lw[:layer])
    #             viscosity = VISCOSITY_SNOW*np.exp(c4*(0.-lT[layer])+c5*lp[layer])

    #             # get change in density
    #             mass_term = weight_above/viscosity
    #             temp_term = -c2*(0.-lT[layer])
    #             dens_term = -c3*max(0,lp[layer]-DENSITY_FRESH_SNOW)
    #             dRho = (mass_term+c1*np.exp(temp_term+dens_term))*lp[layer]*dt
    #             lp[layer] += dRho

    #     # Herron Langway (1980) method
    #     elif args.method_densification in ['HerronLangway']:
    #         # yearly accumulation is the maximum layer snow mass in mm w.e. yr-1
    #         a = layers.max_snow / (dt*365) # kg m-2 = mm w.e.
    #         k = np.zeros_like(lp)
    #         b = np.zeros_like(lp)
    #         for layer,density in enumerate(lp[snowfirn_idx]):
    #             lTK = lT[layer] + CTOK
    #             if density < 550:
    #                 b[layer] = 1
    #                 k[layer] = 11*np.exp(-10160/(R*lTK))
    #             else:
    #                 b[layer] = 0.5
    #                 k[layer] = 575*np.exp(-21400/(R*lTK))
    #         dRho = k*a**b*(DENSITY_ICE - lp)/DENSITY_ICE*dt
    #         lp += dRho

    #     # Kojima (1967) method (JULES)
    #     elif args.method_densification in ['Kojima']:
    #         NU_0 = 1e7      # Pa s
    #         RHO_0 = 50      # kg m-3
    #         k_S = 4000      # K
    #         T_m = 0. + CTOK
    #         for layer in snowfirn_idx:
    #             weight_above = GRAVITY*np.sum(lm[:layer]+lw[:layer])

    #             # get change in density
    #             T_K = lT[layer] + CTOK
    #             exp_term = np.exp(k_S/T_m - k_S/T_K - lp[layer]/RHO_0)
    #             dRho = lp[layer]*weight_above/NU_0*exp_term
    #             lp[layer] += dRho

    #     # check if any water was squeezed out by densification
    #     squeezed_out = 0
    #     for layer in snowfirn_idx:
    #         # irreducible water content depends on density
    #         # if lp[layer] > 500:
    #         #     FRAC_IRREDUC = args.Sr_dense
    #         # else:
    #         #     FRAC_IRREDUC = args.Sr_light
    #         FRAC_IRREDUC = args.Sr
    #         porosity = 1 - lp[layer] / DENSITY_ICE
    #         lh = lm[layer] / lp[layer]
    #         water_irreduc = porosity * lh * DENSITY_WATER * FRAC_IRREDUC
    #         if lw[layer] > water_irreduc:
    #             squeezed_out += lw[layer] - water_irreduc
    #             lw[layer] = water_irreduc

    #     # LAYERS OUT
    #     layers.ldensity = lp
    #     layers.lheight = lm / lp
    #     layers.lwater = lw
    #     layers.update_layer_props('depth')

    #     # check if new firn or ice layers were created
    #     layers.update_layer_types()

    #     # CHECK MASS CONSERVATION
    #     change = np.sum(layers.lice + layers.lwater) - initial_mass + squeezed_out
    #     assert np.abs(change) < args.mb_threshold, f'densification failed mass conservation in {self.output.out_fn}'
    #     return squeezed_out

    # def phase_changes(self):
    #     """
    #     Calculates mass lost or gained from latent heat
    #     exchange (sublimation, deposition, evaporation,
    #     or condensation).
    #     """
    #     # get classes
    #     layers = self.layers
    #     surface = self.surface
    #     args = self.args

    #     # CONSTANTS
    #     LV_SUB = args.Lv_sub
    #     LV_VAP = args.Lv_evap

    #     # get initial mass for conservation check
    #     initial_mass = np.sum(layers.lice + layers.lwater)

    #     # get latent heat from enbal
    #     latent = self.enbal.lat

    #     # get mass fluxes from latent heat
    #     if surface.stemp < 0.:
    #         # SUBLIMATION / DEPOSITION
    #         dm = latent*self.dt/(LV_SUB) # kg m-2
    #         # yes solid-vapor fluxes
    #         sublimation = -1*min(dm,0)
    #         deposition = max(dm,0)
    #         # no liquid-vapor fluxes
    #         evaporation = 0
    #         condensation = 0

    #         # check if dm causes negativity
    #         if layers.lice[0] + dm < 0: 
    #             layer = 0
    #             while np.abs(dm) > 0 and layer < layers.nlayers:
    #                 # calculate the maximum mass loss possible for the current layer
    #                 change = min(np.abs(dm), layers.lice[layer])
    #                 layers.lice[layer] -= change
    #                 layers.lheight[layer] -= change / layers.ldensity[layer]
                    
    #                 # reduce the absolute magnitude of dm
    #                 if dm < 0:
    #                     dm += change  # increase dm towards 0 when negative
    #                 else:
    #                     dm -= change  # decrease dm towards 0 when positive

    #                 # remove or advance layer
    #                 if layers.lice[layer] == 0:
    #                     # layer fully sublimated: move liquid water to next layer and remove
    #                     if layers.lwater[layer] > 0:
    #                         layers.lwater[layer+1] += layers.lwater[layer]
    #                     layers.remove_layer(0)
    #                 else:
    #                     # no layer was removed: advance layer
    #                     layer += 1
                
    #         else:
    #             # add water to layer if it doesn't cause negativity
    #             layers.lice[0] += dm
    #     else:
    #         # EVAPORATION / CONDENSATION
    #         dm = latent*self.dt/(LV_VAP) # kg m-2
    #         # no solid-vapor fluxes
    #         sublimation = 0
    #         deposition = 0
    #         # yes liquid-vapor fluxes
    #         evaporation = -1*min(dm,0)
    #         condensation = max(dm,0)

    #         # check if dm causes negativity
    #         if layers.lwater[0] + dm < 0: 
    #             # reset evaporation to 0 and accumulate actual mass lost
    #             evaporation = 0
    #             dm_to_process = np.abs(dm)
                
    #             layer = 0
    #             while dm_to_process > args.mb_threshold and layer < layers.nlayers:
    #                 change = min(dm_to_process, layers.lwater[layer])
    #                 evaporation += change
    #                 layers.lwater[layer] -= change
    #                 dm_to_process -= change
    #                 layer += 1
                    
    #         else:
    #             # add water to layer if it doesn't cause negativity
    #             layers.lwater[0] += dm

    #     # check we didn't add liquid water to ice layer
    #     runoff = 0
    #     if layers.ltype[0] == 'ice':
    #         for layer in layers.ice_idx:
    #             runoff += layers.lwater[layer]
    #             layers.lwater[layer] = 0

    #     # set vapor fluxes to self
    #     self.sublimation = sublimation
    #     self.deposition = deposition
    #     self.evaporation = evaporation
    #     self.condensation = condensation
    #     self.vapor_solid = sublimation if sublimation != 0 else deposition
    #     self.vapor_liquid = evaporation if evaporation != 0 else condensation
        
    #     # CHECK MASS CONSERVATION
    #     ins = deposition + condensation
    #     outs = sublimation + evaporation + runoff
    #     change = np.sum(layers.lice + layers.lwater) - initial_mass
    #     if np.abs(change - (ins-outs)) >= args.mb_threshold:
    #         print(self.time, 'change', change, 'ins', ins, 'outs', outs)
    #     assert np.abs(change - (ins-outs)) < args.mb_threshold, f'phase change failed mass conservation in {self.output.out_fn}'
    #     return runoff
        
    # def thermal_conduction(self):
    #     """
    #     Resolves the temperature profile with vertical
    #     heat conduction following the Forward-in-Time-
    #     Central-in-Space (FTCS) scheme

    #     Parameters
    #     ==========
    #     layers
    #         Class object from pebsi.layers
    #     surftemp : float
    #         Surface temperature [C]
    #     """        
    #     # get classes
    #     layers = self.layers
    #     surftemp = self.surface.stemp
    #     args = self.args

    #     # CONSTANTS
    #     CP_ICE = args.Cp_ice
    #     DENSITY_ICE = args.density_ice
    #     DENSITY_WATER = args.density_water
    #     TEMP_TEMP = args.temp_temp
    #     TEMP_DEPTH = args.temp_depth
    #     K_ICE = args.k_ice
    #     K_WATER = args.k_water
    #     K_AIR = args.k_air
    #     MAX_DT = args.max_temp_change

    #     # do not need this function if glacier is completely ripe
    #     if np.sum(layers.ltemp) == 0.:
    #         return
        
    #     # check layer sizes for numeric stability
    #     layers.check_layer_sizes()

    #     # determine layers that are below temperate ice depth
    #     if layers.ice_idx[0] > 0:
    #         # if there is snow/firn, adjust to be relative to the ice surface
    #         TEMP_DEPTH += layers.ldepth[layers.ice_idx[0] - 1]
    #     temperate_idx = np.where(layers.ldepth > TEMP_DEPTH)[0]
    #     if len(temperate_idx) < 1:
    #         temperate_idx = [layers.nlayers - 1]
    #     diffusing_idx = np.arange(temperate_idx[0])
    #     layers.ltemp[temperate_idx] = TEMP_TEMP

    #     # LAYERS IN
    #     nl = len(diffusing_idx)
    #     lh = layers.lheight[diffusing_idx]
    #     lp = layers.ldensity[diffusing_idx]
    #     lT_prev = layers.ltemp[diffusing_idx]
    #     lm = layers.lice[diffusing_idx]
    #     lw = layers.lwater[diffusing_idx]
    #     lT = layers.ltemp[diffusing_idx]

    #     # get snow/firn layer conductivity 
    #     ice_idx = layers.ice_idx
    #     if args.method_conductivity in ['Sauter']:
    #         f_ice = (lm/DENSITY_ICE) / lh
    #         f_liq = (lw/DENSITY_WATER) / lh
    #         f_air = 1 - f_ice - f_liq
    #         f_air[f_air < 0] = 0
    #         lcond = f_ice*K_ICE + f_liq*K_WATER + f_air*K_AIR
    #     elif args.method_conductivity in ['VanDusen']:
    #         lcond = 0.21e-01 + 0.42e-03*lp + 0.22e-08*lp**3
    #     elif args.method_conductivity in ['Douville']:
    #         lcond = 2.2*np.power(lp/DENSITY_ICE,1.88)
    #     elif args.method_conductivity in ['Jansson']:
    #         lcond = 0.02093 + 0.7953e-3*lp + 1.512e-12*lp**4
    #     elif args.method_conductivity in ['OstinAndersson']:
    #         lcond = -8.71e-3 + 0.439e-3*lp + 1.05e-6*lp**2
    #     # get ice conductivity (constant)
    #     diffusing_ice_idx = list(set(ice_idx)&set(diffusing_idx))
    #     if len(diffusing_ice_idx) > 0:
    #         lcond[diffusing_ice_idx] = K_ICE

    #     # get timestep for heat equation
    #     dt_heat = self.dt / args.n_heat_steps

    #     # check number of layers
    #     if nl > 2:
    #         # distances between centers of layers
    #         dz = 0.5 * (lh[:-1] + lh[1:]) 

    #         # thermal conductivity at the interfaces
    #         k_inter = 0.5 * (lcond[:-1] + lcond[1:])

    #         # loop through timesteps
    #         for _ in range(args.n_heat_steps):
    #             # flux from surface into layer 0 [W m-2]
    #             flux_surf = lcond[0] * (surftemp - lT_prev[0]) / (0.5 * lh[0])
                
    #             # flux between layers [W m-2]
    #             flux_inter = k_inter * (lT_prev[:-1] - lT_prev[1:]) / dz

    #             # temperature change of top layer
    #             dT_0 = (flux_surf - flux_inter[0]) * dt_heat / (CP_ICE * lp[0] * lh[0])
    #             lT[0] = lT_prev[0] + dT_0
                
    #             # temperature change of other layers
    #             dT_mid = (flux_inter[:-1] - flux_inter[1:]) * dt_heat / (CP_ICE * lp[1:-1] * lh[1:-1])
                
    #             # cap temperature change to a limit
    #             dT_limit = MAX_DT / args.n_heat_steps
    #             dT_mid = np.clip(dT_mid, -dT_limit, dT_limit)
    #             lT[1:-1] = lT_prev[1:-1] + dT_mid

    #             # safety check for top layer stability
    #             if lT[0] > 0 or lT[0] < -50:
    #                 lT[0] = np.mean([surftemp, lT_prev[1]])

    #             lT_prev = lT.copy()

    #     # cases for less than 3 layers do not need to be iterated
    #     elif nl > 1:
    #         lT = np.array([surftemp/2,0])
    #     else:
    #         lT = np.array([0])

    #     # LAYERS OUT
    #     layers.ltemp[diffusing_idx] = lT
    #     return 

    # def end_of_summer(self):
    #     """
    #     Checks prognostically if enough snow will fall
    #     in the upcoming days to constitute the start
    #     of the accumulation season. If so, snow layers
    #     are transformed to firn and cumulative refreeze
    #     is reset to 0.
    #     """
    #     # get classes
    #     layers = self.layers
    #     surface = self.surface
    #     args = self.args

    #     # exit function if there is no snow
    #     if len(layers.snow_idx) == 0:
    #         return
        
    #     # CONSTANTS
    #     NDAYS = args.new_snow_days
    #     SNOW_THRESHOLD = args.new_snow_threshold
    #     T_LOW = args.snow_threshold_low
    #     T_HIGH = args.snow_threshold_high
    #     FIRN_AGE = args.firn_age

    #     # only merge firn if there is old snow
    #     dates_snow = pd.to_datetime(layers.lage[layers.snow_idx])
    #     snow_age_series = pd.Series(self.time - dates_snow)
    #     snow_age = np.array(snow_age_series.dt.days)
        
    #     if np.any(snow_age >= FIRN_AGE):
    #         # define rain vs snow scaling 
    #         rain_scale = np.linspace(1,0,20)
    #         temp_scale = np.linspace(T_LOW,T_HIGH,20)

    #         # index the temperature and precipitation of the upcoming period
    #         end_time = min(self.time_list[-1],self.time+pd.Timedelta(days=NDAYS))
    #         check_dates = pd.date_range(self.time,end_time,freq='h')
    #         check_temp = self.climate.cds.sel(time=check_dates)['temp'].values
    #         check_tp = self.climate.cds.sel(time=check_dates)['tp'].values

    #         # create array to mask tp to snow amounts
    #         mask = np.interp(check_temp,temp_scale,rain_scale)
    #         upcoming_snow = np.sum(check_tp*mask)
            
    #         # check if we are getting enough snow to surpass the threshold
    #         if upcoming_snow < SNOW_THRESHOLD:
    #             # not getting enough snow: exit function
    #             return
    #         else:
    #             # getting new snow: set the timestamp
    #             firn_merged_time = self.time

    #         # MERGING SNOW LAYERS INTO FIRN!
    #         # first, store the past summer surface
    #         year = self.time.year 
    #         self.layers.firn_albedos[year] = surface.min_annual_albedo

    #         # check which layers are old enough to merge
    #         merge_layers = np.where(snow_age >= FIRN_AGE)[0]

    #         # set age of layers to be the oldest layer
    #         layers.lage[merge_layers] = layers.lage[merge_layers[-1]]
            
    #         # loop through layers and merge
    #         for _ in range(merge_layers[0], merge_layers[-1]):
    #             layers.merge_layers(merge_layers[0])

    #         # make sure the layer type for the new firn layer is 'firn'
    #         layers.ltype[merge_layers[0]] = 'firn'

    #         # debugging print statement
    #         if self.args.debug:
    #             print('Converted firn on',firn_merged_time)

    #         # update firn_idx and firn_converted
    #         layers.update_layer_props([])
    #         self.firn_converted = True

    #         # reset cumulative refreeze and annual albedo
    #         layers.lrefreeze *= 0
    #         surface.min_annual_albedo = 1
    #         return

    # def current_state_prints(self):
    #     """
    #     Prints some useful information to keep track 
    #     of a model run.

    #     Parameters
    #     ==========
    #     timestamp : pd.Datetime
    #         Current timestep
    #     airtemp : float
    #         Air temperature [C]
    #     """
    #     # get classes
    #     timestamp = self.time
    #     airtemp = self.enbal.tempC

    #     # gather variables to print out
    #     layers = self.layers
    #     surftemp = self.surface.stemp
    #     albedo = self.surface.bba
    #     melte = np.mean(self.output.meltenergy_output[-720:])
    #     melt = np.sum(self.output.melt_output[-720:])
    #     accum = np.sum(self.output.accum_output[-720:])
    #     ended_month = (timestamp - pd.Timedelta(days=1)).month_name()
    #     year = timestamp.year if ended_month != 'December' else timestamp.year - 1

    #     layers.update_layer_props()
    #     snowdepth = np.sum(layers.lheight[layers.snow_idx])
    #     firndepth = np.sum(layers.lheight[layers.firn_idx])
    #     icedepth = np.sum(layers.lheight[layers.ice_idx])

    #     # begin prints
    #     self.timer.printout()
    #     print(f'MONTH COMPLETED: {ended_month} {year} with +{accum:.2f} and -{melt:.2f} m w.e.')
    #     print(f'Currently {airtemp:.2f} C with {melte:.0f} W m-2 melt energy')
    #     print(f'----------surface albedo: {albedo:.3f} -----------')
    #     print(f'-----------surface temp: {surftemp:.2f} C-----------')
    #     if len(layers.snow_idx) > 0:
    #         print(f'|       snow depth: {snowdepth:.2f} m      {len(layers.snow_idx)} layers      |')
    #     if len(layers.firn_idx) > 0:
    #         print(f'|       firn depth: {firndepth:.2f} m      {len(layers.firn_idx)} layers      |')
    #     print(f'|       ice depth: {icedepth:.2f} m      {len(layers.ice_idx)} layers      |')
    #     for l in range(min(2,layers.nlayers)):
    #         print(f'--------------------layer {l}---------------------')
    #         print(f'     T = {layers.ltemp[l]:.1f} C                 h = {layers.lheight[l]:.3f} m ')
    #         print(f'                 p = {layers.ldensity[l]:.0f} kg/m3')
    #         print(f'Water Mass : {layers.lwater[l]:.2f} kg/m2   Dry Mass : {layers.lice[l]:.2f} kg/m2')
    #     print('================================================')
    #     return

    # def check_mass_conservation(self,mass_in,mass_out):
    #     """
    #     Checks mass was conserved within the last timestep
        
    #     Parameters
    #     ==========
    #     mass_in : float
    #         Sum of mass in (precipitation) (kg m-2)
    #     mass_out : float
    #         Sum of mass out (runoff) (kg m-2)
    #     """
    #     args = self.args

    #     # difference in mass since the last timestep
    #     current_mass = np.sum(self.layers.lice + self.layers.lwater)
    #     diff = current_mass - self.previous_mass
    #     in_out = mass_in - mass_out

    #     # debugging print steps in case mass conservation is failed
    #     if np.abs(diff - in_out) >= args.mb_threshold and self.args.debug:
    #         print(self.time,'discrepancy of',np.abs(diff - in_out) - args.mb_threshold,self.output.out_fn)
    #         print('in',mass_in,'out',mass_out,'currently',current_mass,'was',self.previous_mass)
    #         print('ice before',self.lice_before,'ice after',np.sum(self.layers.lice))
    #         print('w before',self.lwater_before,'w after',np.sum(self.layers.lwater))
    #         print('melt',self.melt,'rfz',self.refreeze,'accum',self.accum)
    #     assert np.abs(diff - in_out) < args.mb_threshold, f'Timestep {self.time} failed mass conservation in {self.output.out_fn}'
        
    #     # new initial mass
    #     self.previous_mass = current_mass
    #     self.lice_before = np.sum(self.layers.lice)
    #     self.lwater_before = np.sum(self.layers.lwater)
    #     return

    # def check_glacier_exists(self):
    #     """
    #     Checks there is still a glacier. If not, ends 
    #     the run and saves the output.
    #     """
    #     # load layer height
    #     total_height = np.sum(self.layers.lheight)
    #     if total_height < self.args.min_glacier_depth:
    #         # new end date
    #         start = self.time_list[0]
    #         end = self.time
    #         new_time = pd.date_range(start,end,freq='h')
    #         self.output.n_timesteps = len(new_time)

    #         # load the output
    #         with xr.open_dataset(self.output.out_fn) as dataset:
    #             ds = dataset.load()
    #             # chop it to the new end date
    #             ds = ds.sel(time=new_time)
    #         # store output
    #         ds.to_netcdf(self.output.out_fn)

    #         # save the data
    #         if self.args.store_data:
    #             self.output.store_data()
    #         print(f'Glacier fully melted on {self.time} in {self.args.output_fn}')
            
    #         return True # no glacier remaining
    #     else:
    #         return False # still glacier

    # def store_simulation(self):
    #     """
    #     Stores data model output and
    #     deletes temporary files used in 
    #     the simulation.
    #     """
    #     # store main simulation
    #     if self.args.store_data:
    #         if self.args.debug:
    #             print('~ Success! Storing data . . . ~')
    #         self.output.store_data()

    #     # optionally store spectral albedo
    #     if self.args.store_bands:
    #         self.surface.albedo_df.to_csv(self.args.albedo_out_fn.format(s=self.args.site))
    #     return

    # def iterable(self, iterable, **kwargs):
    #     return tqdm(iterable, **kwargs) if self.args.progress_bar else iterable

    # class MeltedLayers():
    # def __init__(self, layers, fully_melted):
    #     self.water = layers.lwater[fully_melted]
    #     self.ice = layers.lice[fully_melted]
    #     self.mass = self.water + self.ice 
    #     self.BC = layers.lBC[fully_melted]
    #     self.OC = layers.lOC[fully_melted]
    #     self.dust = layers.ldust[fully_melted]

    # class ProgressTimer:
    # """
    # Keeps track of time elapsed and 
    # estimates time remaining based on
    # the number of timesteps.
    # """
    # def __init__(self, total_steps):
    #     self.total_steps = total_steps
    #     self.start = pytime.perf_counter()
    #     self.elapsed = 0
    #     self.remaining = float("inf")
    #     self.step = -1

    # def update(self):
    #     """
    #     Steps counter and estimates remaining time.
    #     """
    #     now = pytime.perf_counter()
    #     elapsed = now - self.start
    #     self.step += 1

    #     frac = self.step / self.total_steps
    #     est_total = elapsed / frac if frac > 0 else float("inf")
    #     remaining = est_total - elapsed

    #     self.remaining = remaining 
    #     self.elapsed = elapsed

    # def printout(self):
    #     percent_done = self.step / self.total_steps * 100
    #     blocks_total = 48
    #     n_blocks_filled = int(percent_done / 100 * blocks_total)
    #     n_blocks_empty = blocks_total - n_blocks_filled
    #     print(''.join(['█']*n_blocks_filled) + ''.join(['-']*n_blocks_empty))
    #     print(
    #         f"{percent_done:.0f}%  "
    #         f"[ Elapsed: {self.elapsed/60:.2f} min | Remaining: {self.remaining/60:.2f} min ]"
    #     )