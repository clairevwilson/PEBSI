"""
Mass balance class for PEBSI

Contains functions to calculate mass
balance processes such as melting, 
refreezing,and percolation.
"""
# External libraries
import jax
import jax.numpy as jnp
# Local libraries
import util.layers as layers
import pebsi.albedo as albedo

class MassBalanceDriver:
    def __init__(self, params):
        """
        Stores parameters and physical constants
        for accessing within mass balance functions.
        """
        self.params = params
        return

    # -------------------- WORKER FUNCTIONS --------------------
    #       These functions are called directly from main()
    def run_new_mass(self, state, forcings):
        """
        Runs processes to get snowfall, rainfall,
        and dry deposition totals.

        Returns
        -------
        snowfall, rainfall : jnp.ndarray
            (N_POINTS) arrays of snowfall and rainfall
            in kg m-2
        """
        # divide incoming precip to rain and snow
        rainfall, snowfall = self.get_precip_amounts(forcings)

        # add new snow to layers
        snowfall, state = self.add_accumulation(
            snowfall, rainfall, state, forcings
        )

        # append accumulation to past_snow
        new_past_snow = jnp.roll(
            state.past_snow, shift=-1, axis=1
        ).at[:, -1].set(snowfall)
        state = state._replace(past_snow=new_past_snow)

        # add dry deposition of light-absorbing particles
        state = self.add_dry_deposition(
            state, forcings
        )
        return rainfall, snowfall, state
    
    def run_daily_routines(self, state, forcings):
        """
        Checks if we are at an hour to run daily
        updates (e.g., albedo)
        """
        albedo_TOD = self.params.albedo_TOD

        # parse time index for daily functions (local_hour is per-point)
        is_day_start = forcings.local_hour == 0
        is_albedo_step = jnp.any(forcings.local_hour[:, None] == jnp.array(albedo_TOD), axis=-1)

        # either run or skip daily routines depending on hour of day
        state = jax.lax.cond(
            jnp.any(is_day_start | is_albedo_step),
            lambda s: self.get_daily_updates(s, forcings),
            lambda s: s,
            state
        )
        return state
    
    def run_vertical_processes(self, state, forcings, fluxes):
        """
        Runs vertical processes of melting (including
        penetrating shortwave radiation), percolation,
        refreezing, and thermal conduction.

        Returns
        -------
        mass_fluxes : dict
            Dictionary containing runoff and melt
        """

        # subsurface heating and melting
        state, melt_array, mass_to_route = self.heating_melting(state, fluxes)

        # percolate meltwater and route LAPs
        for var, data in mass_to_route.items():
            fluxes[var] = jnp.sum(data, axis=1)
        state, melt_runoff, fluxes = self.percolation(state, fluxes)
        state = self.route_particles(state, forcings, fluxes)
        
        # refreezing
        state = self.refreezing(state)

        # phase changes (e.g., sublimation, condensation)
        state, condensation_runoff, mass_fluxes = self.phase_changes(
            state, fluxes['latent_heat']
        )

        # check layer sizes for numeric stability before running temp profile
        state, dead_mass = layers.check_layer_sizes(state, self.params)
        
        # resolve temperature profile
        state = self.resolve_temperature_profile(state)

        # calculate total runoff and store mass fluxes together
        runoff = melt_runoff + condensation_runoff
        mass_fluxes['runoff'] = runoff 
        mass_fluxes['melt_2D'] = melt_array
        mass_fluxes['dead'] = dead_mass

        return state, mass_fluxes
    
    def run_state_updates(self, state, forcings):
        """
        Runs 'state' updates for densification,
        roughness, and grain size.

        Returns
        -------
        water_squeezed_out : jnp.ndarray
            (N_POINTS,) water that no longer has 
            pore space due to densification
        """
        # densification is only run daily
        is_day_start = forcings.local_hour == 0
        state, water_squeezed_out = jax.lax.cond(
            jnp.any(is_day_start),
            lambda s: self.densification(s),
            lambda s: (s, jnp.zeros_like(state.albedo)),
            state
        )

        # update surface roughness
        new_roughness = self.roughness(state)

        # update grain sizes
        new_grainsize = self.evolve_grain_size(state, forcings)

        # update age 
        new_age = state.lage + 1

        # store updated properties to state
        state = state._replace(
            lgrainsize = new_grainsize,
            roughness = new_roughness,
            lage = new_age
        )
        return state, water_squeezed_out

    def run_annual_routines(self, state, forcings):
        """
        Checks if it is the end of summer to run
        firn conversion routine.
        """
        params = self.params

        # are we in the end-of-summer window?
        is_summer_end_window = (forcings.doy >= params.start_end_summer) & \
            (forcings.doy <= params.start_end_summer + 60)
        is_midnight = forcings.local_hour == 0
        # does past snowfall surpass the threshold to consider winter?
        weather_trigger = jnp.sum(state.past_snow, axis=1) >= params.new_snow_threshold
        # put temporal triggers together
        time_to_merge = jnp.any(is_summer_end_window & is_midnight & weather_trigger)

        # only run end of summer if we met all temporal conditions
        state = jax.lax.cond(
            time_to_merge, 
            self.end_of_summer,
            lambda s: s, 
            state
        )

        # if it is the start of a year, reset firn converted trackers
        is_year_start = (forcings.doy == 0) & (forcings.local_hour == 0)
        new_firn_converted = jnp.where(is_year_start, False, state.annual_firn_converted)
        state = state._replace(annual_firn_converted = new_firn_converted)
        return state
    
    # -------------------- PHYSICS FUNCTIONS --------------------
    #  These routines are called from the above worker functions

    def get_precip_amounts(self, forcings):
        """
        Determines how much rain and snowfall occurred
        in a timestep.

        Returns:
        --------
        rain, snow : jnp.ndarray
            (N_POINTS) Specific mass of liquid and solid 
            precipitation to add [kg m-2]
        """
        # CONSTANTS
        SNOW_THRESHOLD_LOW = self.params.snow_threshold_low
        SNOW_THRESHOLD_HIGH = self.params.snow_threshold_high
        DENSITY_WATER = self.params.density_water

        # define rain vs snow scaling 
        rain_scale = jnp.linspace(0, 1, 20)
        temp_scale = jnp.linspace(SNOW_THRESHOLD_LOW, SNOW_THRESHOLD_HIGH, 20)

        # calculate fraction of rain
        fraction_rain = jnp.interp(forcings.temp, temp_scale, rain_scale)
        rain = forcings.tp * fraction_rain * DENSITY_WATER 
        snow = forcings.tp * (1-fraction_rain) * DENSITY_WATER

        # make sure amounts to the boundaries correctly
        rain = jnp.where(forcings.temp <= SNOW_THRESHOLD_LOW, 0, rain)
        snow = jnp.where(forcings.temp > SNOW_THRESHOLD_HIGH, 0, snow)
        
        return rain,snow # kg m-2

    def add_accumulation(self, snowfall, rainfall, state, forcings):
        """
        Adds snowfall to the layers either as a new layer
        or merges new snow with the top layer.
        
        Parameters
        ==========
        snowfall : float
            Fresh snow mass [kg m-2]

        Returns
        -------
        actual_snowfall : float
            Actual snow mass that was added [kg m-2]
        """
        params = self.params

        # grab forcings objects we need
        time_idx = forcings.time_idx
        wind = forcings.wind 
        temp = forcings.temp
        bcwet = forcings.bcwet
        ocwet = forcings.ocwet 
        dustwet = forcings.dustwet

        # add delayed snow to snowfall
        total_snowfall = snowfall + state.delayed_snow

        # check if using constant density for new snow
        if params.constant_snowfall_density:
            new_density = params.constant_snowfall_density
        else:
            # CROCUS formulation of density (Vionnet et al. 2012)
            new_density = jnp.maximum(
                109 + 6 * (temp - 0.) + 26 * jnp.sqrt(
                    jnp.maximum(wind, 1e-10)), 50)
        
        # check if using constant grain size for new snow
        if params.constant_freshgrainsize:
            new_grainsize = params.constant_freshgrainsize
        else:
            # CLM formulation of grain size (CLM5.0 Documentation)
            airtemp = temp
            new_grainsize = jnp.select(
                [airtemp <= -30, airtemp < 0],
                [54.5, 54.5 + 5 * (airtemp + 30)],
                default=204.5
            )

        # height and age of new layer
        new_height = total_snowfall/new_density
        new_age = jnp.zeros_like(total_snowfall)

        # wet deposition: scale by snow share of precip (all in kg m-2)
        total_precip = total_snowfall + rainfall
        snow_fraction = jnp.clip(total_snowfall / (total_precip + 1e-12), 0.0, 1.0)
        new_BC = bcwet * params.dt * snow_fraction
        new_OC = ocwet * params.dt * snow_fraction
        new_dust = dustwet * params.dt * snow_fraction

        # pack properties of new layer into a namespace
        new_layer = {
            'ltemp': temp,
            'lheight': total_snowfall / new_density,
            'ltype': jnp.full_like(temp, 0),
            'lice': total_snowfall,
            'ldensity': new_density,
            'lgrainsize': new_grainsize,
            'lBC': new_BC,
            'lOC': new_OC,
            'ldust': new_dust,
            'lage': new_age,
            'lwater': jnp.full_like(temp, 0),
            'lrefreeze': jnp.full_like(temp, 0),
            'ldrefreeze': jnp.full_like(temp, 0)
        }
        
        # define conditions for making a new layer for accumulation
        surf_not_snow = (state.ltype[:, 0] > 0)
        density_threshold = (state.ldensity[:, 0] > (new_density * 3))
        large_top_layer = (state.lheight[:, 0] > params.dz_toplayer * 2)
        new_layer_cond = surf_not_snow | density_threshold | large_top_layer

        # check for small surface snow layer (merge new snow with it no matter what)
        small_top_layer = (state.lheight[:, 0] < 1e-3) & (state.ltype[:, 0] == 0)

        # define masks for possible cases
        create_new_mask = new_layer_cond & (~small_top_layer) & (new_height >= 1e-3)
        delay_mask = new_layer_cond & (~small_top_layer) & (new_height < 1e-3)
        merge_new_mask = (~new_layer_cond) | small_top_layer

        action_taken_mask = create_new_mask | merge_new_mask

        # handle cases 1 & 3: create a new layer and merge snowfall with existing layer
        state = layers.add_top_layer(state, create_new_mask, new_layer)
        state = layers.merge_new_layer(state, merge_new_mask, new_layer, params)

        # handle case 2: delaying this snowfall to the next timestep
        updated_delayed_snow = jnp.where(delay_mask, total_snowfall, state.delayed_snow)
        updated_delayed_snow = jnp.where(action_taken_mask, 0.0, updated_delayed_snow)
        actual_snowfall = jnp.where(delay_mask, 0.0, total_snowfall)

        # update surface snow timestemp
        updated_last_snow = jnp.where(action_taken_mask, time_idx, state.last_snow)

        state = state._replace(
            delayed_snow = updated_delayed_snow,
            last_snow = updated_last_snow
        )
        
        # update layer depth from new layer heights
        state = layers.update_layer_props(state, params.density_ice)     
        
        # return actual snowfall that was added, including any delayed_snow
        return actual_snowfall, state

    def add_dry_deposition(self, state, forcings):
        """
        Adds dry deposition of light-absorbing particles
        to the surface layer.
        """
        dt = self.params.dt

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
    
    def get_daily_updates(self, state, forcings):
        """
        Executes daily update functions (albedo, surrounding 
        albedo, days since snowfall).
        """
        albedo_TOD = self.params.albedo_TOD
        is_day_start = forcings.local_hour == 0
        is_albedo_step = jnp.any(forcings.local_hour[:, None] == jnp.array(albedo_TOD), axis=-1)

        # === albedo ===
        new_albedo, new_annual_min_albedo = albedo.get_albedo(
            state, self.params, forcings
        )

        # === surrounding albedo ===
        ALBEDO_GROUND = self.params.albedo_ground
        ALBEDO_SNOW = self.params.albedo_fresh_snow

        # update max snow if snow mass is greater than running max
        current_snow = jnp.sum(state.lice * state.snow_mask, axis=1)
        new_annual_max_snow = jnp.maximum(current_snow, state.annual_max_snow)
        
        # get fraction of max annual snow at each point and scale albedo
        safe_max_snow = jnp.where(new_annual_max_snow > 0, new_annual_max_snow, 1.0)
        snow_fraction = jnp.clip(current_snow / safe_max_snow, 0.0, 1.0)
        new_albedo_surr = ALBEDO_GROUND + (ALBEDO_SNOW - ALBEDO_GROUND) * snow_fraction

        # === days since snowfall ===
        time_idx = jnp.full(state.lice.shape[0], forcings.time_idx)
        hours_since_snowfall = time_idx - state.last_snow
        new_days_since_snowfall = jnp.round(hours_since_snowfall / 24).astype(jnp.int32)

        # only update the properties requested based on hour of day
        updated_albedo = jnp.where(is_albedo_step, new_albedo, state.albedo)
        updated_min_albedo = jnp.where(
            is_albedo_step[:, None], new_annual_min_albedo, state.annual_min_albedo
        )
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
            albedo=updated_albedo,
            annual_min_albedo=updated_min_albedo,
            days_since_snowfall=new_days_since_snowfall,
            annual_max_snow=new_annual_max_snow,
            albedo_surr=new_albedo_surr
        )
    
    def heating_melting(self, state, fluxes):
        """
        Applies melt energy to the column. First
        warms layers to the melting point, then
        melts them. Melted mass becomes liquid water.
        
        Returns
        -------
        layermelt : jnp.ndarray
            (N_POINTS, N_LAYERS) melt per layer in kg m-2
        mass_to_route : dict
            Contains all mass from fully melted layers
            (ice, water, and particles)
        """
        params = self.params
        layers_idx = jnp.arange(state.lice.shape[1])[None, :]

        # CONSTANTS
        EXTINCT_SNOW = params.extinct_coef_snow 
        EXTINCT_ICE = params.extinct_coef_ice 
        CP_ICE = params.Cp_ice
        CP_WATER = params.Cp_water
        LH_RF = params.Lh_rf
        dt = params.dt

        # load the amount of heat added to each layer
        Q_abs_surface = fluxes['melt_energy'][:, None]
        Q_penetrating = fluxes['SWnet_penetrating'][:, None]
        extinct_coefs = jnp.where(state.ice_mask, EXTINCT_ICE, EXTINCT_SNOW)

        f_top = jnp.exp(-1 * extinct_coefs * (state.ldepth - state.lheight / 2))
        f_bottom = jnp.exp(-1 * extinct_coefs * (state.ldepth + state.lheight / 2))
        Q_abs_layer = Q_penetrating * (f_top - f_bottom)

        layer_heat = jnp.where(layers_idx > 0, Q_abs_layer, Q_abs_surface)

        # recalculate layer temperatures using effective heat capacity
        lmass = state.lice + state.lwater
        safe_lmass = jnp.where(lmass > 0, lmass, 1.0)
        cp_eff = jnp.where(
            lmass > 0,
            (state.lice * CP_ICE + state.lwater * CP_WATER) / safe_lmass, 1.0
        )

        # ---------- CASCADE FUNCTION ----------
        # transpose all arrays to (N_LAYERS, N_POINTS) for lax.scan
        scan_heat = jnp.transpose(layer_heat)
        scan_lice = jnp.transpose(state.lice)
        scan_lwater = jnp.transpose(state.lwater)
        scan_ltemp = jnp.transpose(state.ltemp)
        scan_lmass = jnp.transpose(safe_lmass)
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
            safe_denom = jnp.where(lmass * lcp > 0, lmass * lcp, 1.0)
            partial_warm_temp = jnp.where(lmass > 0, ltemp + (total_heat_in / safe_denom), ltemp)
            # clip ptemperature of points that were warmed past melting point
            intermediate_temp = jnp.where(warmed_past_zero, 0.0, partial_warm_temp)
            
            # leftover energy after melting to 0. converted to mass
            melt_energy_available = jnp.maximum(0.0, total_heat_in - energy_to_zero)
            potential_melt = melt_energy_available / LH_RF
            
            # actual melt is capped by how much solid ice physically exists in this layer
            actual_melt = jnp.minimum(potential_melt, lice)
            
            # calculate unspent melt energy that must cascade lower [W m-2]
            carry = jnp.maximum(
                0., (melt_energy_available - (actual_melt * LH_RF)) / dt
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

        # shrink layer height to reflect lost ice mass
        safe_density = jnp.where(properties['ldensity'] > 0, properties['ldensity'], 1.0)
        properties['lheight'] = properties['lice'] / safe_density

        # store the mass in the layers that are about to be deleted
        fully_melted_mask = properties['lice'] <= params.min_layer_mass
        mass_to_route = {}
        mass_to_route['meltwater'] = jnp.where(fully_melted_mask, properties['lwater'] + properties['lice'], 0)
        mass_to_route['BC'] = jnp.where(fully_melted_mask, properties['lBC'], 0)
        mass_to_route['OC'] = jnp.where(fully_melted_mask, properties['lOC'], 0)
        mass_to_route['dust'] = jnp.where(fully_melted_mask, properties['ldust'], 0)

        # remove the fully melted mass so it isn't double-counted
        properties['lwater'] = jnp.where(fully_melted_mask, 0.0, properties['lwater'])
        properties['lice'] = jnp.where(fully_melted_mask, 0.0, properties['lice'])
        
        # store updated properties
        state = state._replace(**properties)
        
        # layermelt is now actual melt amounts in (N_POINTS, N_LAYERS) shape
        layermelt = jnp.transpose(layermelt)

        # collapse grid to purge melted layers
        def remove_one(i, s):
            """Removes the top empty layer from each point"""
            fully_melted_mask = s.lice <= params.min_layer_mass
            melt_point_mask = jnp.any(fully_melted_mask, axis=1)
            melt_layer_idx = jnp.argmax(fully_melted_mask.astype(jnp.int32), axis=1)
            return layers.remove_layer(s, melt_point_mask, melt_layer_idx, params)

        state = jax.lax.fori_loop(0, 3, remove_one, state)

        return state, layermelt, mass_to_route
        
    def percolation(self, state, fluxes):
        """
        Percolates water downward using tipping-bucket
        approach. Layers never hold more than their
        irreducible water content.

        Returns
        -------
        runoff : jnp.ndarray
            Runoff of liquid water lost to system [kg m-2]
        fluxes : dict
            Mass fluxes into and out of domain including
            flow out of each layer 
        """
        properties = state._asdict()
        params = self.params 

        # CONSTANTS
        DENSITY_WATER = params.density_water
        DENSITY_ICE = params.density_ice

        # transpose arrays for layer cascade
        scan_lice = jnp.transpose(properties['lice'])
        scan_lwater = jnp.transpose(properties['lwater'])
        scan_lheight = jnp.transpose(properties['lheight'])
        scan_ldensity = jnp.transpose(properties['ldensity'])
        scan_ice_mask = jnp.transpose(properties['ice_mask'])

        # calculate porosity
        safe_lheight = jnp.where(scan_lheight > 0, scan_lheight, 1.0)
        vol_f_ice = scan_lice / (safe_lheight * DENSITY_ICE)
        porosity = jnp.maximum(0.0, 1.0 - vol_f_ice)

        # calculate irreducible water content
        if params.constant_irrwater:
            frac_irreduc = jnp.full_like(porosity, params.Sr)
        else:
            frac_irreduc = jnp.where(scan_ldensity > 500.0, params.Sr_dense, params.Sr_light)
        water_irreduc_capacity = porosity * scan_lheight * DENSITY_WATER * frac_irreduc

        # pack all the inputs
        layer_inputs = (scan_lwater, water_irreduc_capacity, scan_ice_mask)

        # define cascade function
        def _percolation_cascade(carry, inputs):
            # carry tracks: (current water flux in, cumulative runoff)
            q_in, current_runoff = carry
            lwater, capacity, is_barrier = inputs

            # if this is an ice layer, everything here and below runs off
            q_in_blocked = jnp.where(is_barrier, q_in, 0.0)
            q_in = jnp.where(is_barrier, 0.0, q_in)

            # remaining room before hitting irreducible water content
            # if negative, the layer is over capacity and must drain the excess
            available_room = capacity - lwater
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
        Routes LAPs vertically through the snow and firn
        layers according to water flow from percolation.

        """
        params = self.params
        properties = state._asdict()
        layers_idx = jnp.arange(state.lice.shape[1])[None, :]
        dt = params.dt

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

        # inject wet-deposited particles into water flowing in (all in kg m-2)
        total_precip = fluxes['rainfall'] + fluxes['snowfall']
        rain_fraction = jnp.clip(fluxes['rainfall'] / (total_precip + 1e-12), 0.0, 1.0)
        BC_wet_flux = forcings.bcwet * rain_fraction * dt
        OC_wet_flux = forcings.ocwet * rain_fraction * dt
        dust_wet_flux = forcings.dustwet * rain_fraction * dt

        scan_ice_mask = jnp.transpose(properties['ice_mask'])

        # pack all the inputs
        layer_inputs = (safe_mass, scan_mBC, scan_mOC, scan_mdust,
                        scan_q_out, scan_ice_mask, (layers_idx[0, :] == 0))

        # define cascade function
        def _particle_cascade(carry, inputs):
            BCin, OCin, dustin = carry
            mass, mBC, mOC, mdust, q_out, is_barrier, is_surface = inputs

            # add wet deposition if this is the top layer
            BCin = jnp.where(is_surface, BCin + BC_wet_flux, BCin)
            OCin = jnp.where(is_surface, OCin + OC_wet_flux, OCin)
            dustin = jnp.where(is_surface, dustin + dust_wet_flux, dustin)

            # ice layers block particle flow (same as water barrier) — pass BC to runoff
            BCin = jnp.where(is_barrier, 0.0, BCin)
            OCin = jnp.where(is_barrier, 0.0, OCin)
            dustin = jnp.where(is_barrier, 0.0, dustin)

            # instantly mix new particles into this layer
            cBC = jnp.where(mass > 0.0, mBC / mass, 0.0)
            cOC = jnp.where(mass > 0.0, mOC / mass, 0.0)
            cdust = jnp.where(mass > 0.0, mdust / mass, 0.0)

            # compute partition leaving the layer
            BCout_pot = self.params.ksp_BC * q_out * cBC
            OCout_pot = self.params.ksp_OC * q_out * cOC
            dustout_pot = self.params.ksp_dust * q_out * cdust

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

    def refreezing(self, state):
        """
        Calculates refreeze in layers that are both
        below freezing and contain liquid water content.
        """
        properties = state._asdict()
        params = self.params

        # CONSTANTS
        CP_ICE = params.Cp_ice
        CP_WATER = params.Cp_water
        DENSITY_ICE = params.density_ice
        LH_RF = params.Lh_rf

        ltemp = properties['ltemp']
        lwater = properties['lwater']
        lice = properties['lice']
        lheight = properties['lheight']

        # calculate actual heat capacity including water
        bulk_heat_capacity = (lice * CP_ICE) + (lwater * CP_WATER)
        safe_heat_capacity = jnp.where(
            bulk_heat_capacity > 0, bulk_heat_capacity, 1.
        )

        # potential based on available cold content [J m-2]
        E_cold = jnp.abs(ltemp) * bulk_heat_capacity 
        # potential based on available water to refreeze [J m-2]
        E_water = lwater * LH_RF
        # potential based on pore space available [J m-2]
        E_pore = jnp.maximum(0.0, DENSITY_ICE * lheight - lice) * LH_RF

        # calculate mass of refreeze [kg m-2]
        dm_rfz = jnp.minimum(
            E_cold, jnp.minimum(E_water, E_pore)
        ) / LH_RF

        # mask amount of refreeze to layers that were below 0.
        dm_rfz = jnp.where(ltemp < 0, dm_rfz, 0)

        # calculate temperature change due to refreezing
        dT_rfz = dm_rfz * LH_RF / safe_heat_capacity

        # update layer properties
        properties['lwater'] = jnp.maximum(0.0, lwater - dm_rfz) 
        properties['lice'] = lice + dm_rfz 
        properties['ltemp'] = jnp.minimum(0.0, ltemp + dT_rfz)

        # update layer refreeze quantities
        properties['lrefreeze'] += dm_rfz 
        properties['ldrefreeze'] = dm_rfz

        # store to state
        state = state._replace(**properties)

        # update density from refreeze
        state = layers.update_layer_props(state, DENSITY_ICE)
        return state
    
    def phase_changes(self, state, latent_heat):
        """
        Calculates mass lost or gained from latent heat
        exchange (sublimation, deposition, evaporation,
        or condensation).

        Returns
        -------
        total_runoff : jnp.ndarray
            (N_POINTS) runoff from condensation on ice
        surface_mass_fluxes : dict
            (N_POINTS) jnp.ndarrays of phase change 
            mass fluxes [kg m-2]
        """
        properties = state._asdict()
        params = self.params

        # CONSTANTS
        LV_SUB = params.Lv_sub
        LV_VAP = params.Lv_evap
        dt = params.dt

        # load inputs
        ice_mask = properties['ice_mask']
        lwater = properties['lwater']
        lice = properties['lice']
        ldensity = properties['ldensity']
        surftemp = properties['surftemp']

        # phase changes are vapor<-->solid if surface is below freezing
        is_sublimation_phase = surftemp < 0.0
        
        dm_sub_pot = (latent_heat * dt) / LV_SUB
        dm_vap_pot = (latent_heat * dt) / LV_VAP
        
        # calculate mass loss/gain potential
        dm_potential = jnp.where(is_sublimation_phase, dm_sub_pot, dm_vap_pot)

        # separate into mass gain (deposition, condensation) 
        mass_gain = jnp.maximum(0.0, dm_potential)
        # and mass loss (sublimation, evaporation)
        mass_loss_demand = jnp.abs(jnp.minimum(0.0, dm_potential))

        # MASS GAIN: add mass to solid / liquid of top layer
        lice_gain = jnp.where(is_sublimation_phase, mass_gain, 0.0)
        lwater_gain = jnp.where(~is_sublimation_phase, mass_gain, 0.0)

        # update state properties
        updated_lice = lice.at[:, 0].add(lice_gain)
        updated_lwater = lwater.at[:, 0].add(lwater_gain)

        # MASS LOSS: need to cascade in case an entire layer is sublimated
        target_mass_reservoir = jnp.transpose(jnp.where(
            is_sublimation_phase[:, None], updated_lice, updated_lwater
        ))
        
        # define cascade function
        def _phase_change_cascade(remaining_demand, layer_mass):
            # check how much mass the layer can give up
            actual_loss = jnp.minimum(remaining_demand, layer_mass)
            new_layer_mass = layer_mass - actual_loss

            # send the remainder to the next layer
            next_demand = remaining_demand - actual_loss
            return next_demand, (new_layer_mass, actual_loss)

        # execute cascade down the column
        unmet_demand, (new_mass, mass_lost) = jax.lax.scan(
            _phase_change_cascade, mass_loss_demand, target_mass_reservoir
        )
        new_mass = jnp.transpose(new_mass)
        mass_lost = jnp.transpose(mass_lost)

        # re-assign the updated reservoirs
        updated_lice = jnp.where(
            is_sublimation_phase[:, None], new_mass, updated_lice
        )
        updated_lwater = jnp.where(
            ~is_sublimation_phase[:, None], new_mass, updated_lwater
        )

        # calculate actual total mass lost for diagnostics
        total_actual_loss = jnp.sum(mass_lost, axis=1)

        # recalculate layer heights due to sublimation
        safe_ldensity = jnp.where(ldensity > 0, ldensity, 1.0)
        updated_lheight = jnp.maximum(0.0, updated_lice / safe_ldensity)

        # ice cannot hold water so it goes to runoff
        runoff_per_layer = jnp.where(ice_mask, updated_lwater, 0.0)
        updated_lwater = jnp.where(ice_mask, 0.0, updated_lwater)
        total_runoff = jnp.sum(runoff_per_layer, axis=1)

        # store properties back
        properties['lwater'] = updated_lwater
        properties['lice'] = updated_lice
        properties['lheight'] = updated_lheight

        # calculate actual sublimation and evaporation
        sublimation = jnp.where(is_sublimation_phase, total_actual_loss, 0.0)
        evaporation = jnp.where(~is_sublimation_phase, total_actual_loss, 0.0)
        
        surface_mass_fluxes = {
            'sublimation': sublimation,
            'deposition': lice_gain,
            'evaporation': evaporation,
            'condensation': lwater_gain
        }

        state = state._replace(**properties)
        return state, total_runoff, surface_mass_fluxes
        
    def resolve_temperature_profile(self, state):    
        """
        Resolves the temperature profile with vertical
        heat conduction following an explicit, linearized
        Forward-in-Time, Central-in-Space (FTCS) scheme.
        Uses a Dirichlet boundary condition at the top layer
        and the temperate boundary at the bottom.
        """   
        params = self.params

        # CONSTANTS
        CP_ICE = params.Cp_ice
        DENSITY_ICE = params.density_ice
        DENSITY_WATER = params.density_water
        TEMP_TEMP = params.temp_temp
        TEMP_DEPTH = params.temp_depth
        K_ICE = params.k_ice
        K_WATER = params.k_water
        K_AIR = params.k_air
        MAX_DT = params.max_temp_change

        # load inputs
        surftemp = state.surftemp
        lheight = state.lheight
        ldensity = state.ldensity
        ltemp = state.ltemp
        lice = state.lice
        lwater = state.lwater
        ldepth = state.ldepth
        ice_mask = state.ice_mask

        # determine temperate depth relative to ice surface
        snow_firn_heights = jnp.where(ice_mask, 0.0, lheight)
        ice_surf_depth = jnp.sum(snow_firn_heights, axis=1)
        temperate_depth = TEMP_DEPTH + ice_surf_depth
        is_temperate = ldepth >= temperate_depth[:, None]

        safe_lheight = jnp.where(lheight > 0, lheight, 1)

        # get thermal conductivity for every layer
        if params.method_conductivity in ['Sauter']:
            # handles snow or ice
            f_ice = (lice / DENSITY_ICE) / safe_lheight
            f_liq = (lwater / DENSITY_WATER) / safe_lheight
            f_air = jnp.clip(1 - f_ice - f_liq, 0, None)
            lcond = f_ice * K_ICE + f_liq * K_WATER + f_air * K_AIR
        else:
            # get snow and firn conductivity
            if params.method_conductivity in ['VanDusen']:
                lcond = 0.21e-01 + 0.42e-03 * ldensity + 0.22e-08 * ldensity**3
            elif params.method_conductivity in ['Douville']:
                lcond = 2.2 * jnp.power(ldensity / DENSITY_ICE, 1.88)
            elif params.method_conductivity in ['Jansson']:
                lcond = 0.02093 + 0.7953e-3 * ldensity + 1.512e-12 * ldensity **4
            elif params.method_conductivity in ['OstinAndersson']:
                lcond = -8.71e-3 + 0.439e-3 * ldensity + 1.05e-6 * ldensity**2
            
            # mask ice layers with constant conductivity
            lcond = jnp.where(ice_mask, K_ICE, lcond)

        # get timestep for heat equation
        dt_heat = params.dt / params.n_heat_steps
        dT_limit = MAX_DT / params.n_heat_steps

        # inter-layer spacing
        # dz is the distance between center of layer i and layer i+1
        dz = 0.5 * (lheight[:, :-1] + lheight[:, 1:])
        safe_dz = jnp.where(dz > 0, dz, 1.0)
        k_inter = 0.5 * (lcond[:, :-1] + lcond[:, 1:])

        # define the function to loop over dt_heat
        def _conduction_step(i, temps):
            # flux from surface boundary into layer 0 (1D)
            flux_surf = lcond[:, 0] * (surftemp - temps[:, 0]) / (0.5 * safe_lheight[:, 0])
            
            # flux matrix between all internal layer columns (2D)
            flux_inter = k_inter * (temps[:, :-1] - temps[:, 1:]) / safe_dz

            # top layer update logic
            raw_tm0 = CP_ICE * ldensity[:, 0] * safe_lheight[:, 0]
            safe_thermal_mass_0 = jnp.where(raw_tm0 > 0, raw_tm0, 1.0)
            dT_0 = (flux_surf - flux_inter[:, 0]) * dt_heat / safe_thermal_mass_0
            
            # mid-layer Updates Logic
            safe_thermal_mass_mid = CP_ICE * ldensity[:, 1:-1] * safe_lheight[:, 1:-1]
            
            # net flux difference: flux in - flux out
            net_flux_mid = flux_inter[:, :-1] - flux_inter[:, 1:]
            dT_mid = net_flux_mid * dt_heat / jnp.where(
                safe_thermal_mass_mid > 0, safe_thermal_mass_mid, 1.0)
            dT_mid = jnp.clip(dT_mid, -dT_limit, dT_limit)

            # assemble updated temperatures array
            next_temps = temps
            next_temps = next_temps.at[:, 0].add(dT_0)
            next_temps = next_temps.at[:, 1:-1].add(dT_mid)

            # top layer numerical stability checker fallback
            # very small top layer can experience extreme cooling / heating
            unstable_top = (next_temps[:, 0] > 0.0) | \
                    (next_temps[:, 0] < -50.0) | \
                    jnp.isnan(next_temps[:, 0])
            fallback_top_temp = 0.5 * (surftemp + next_temps[:, 1])
            next_temps = next_temps.at[:, 0].set(jnp.where(
                unstable_top, fallback_top_temp, next_temps[:, 0]))

            # overlay layers below TEMP_DEPTH with temperate temperature
            next_temps = jnp.where(is_temperate, TEMP_TEMP, next_temps)
            
            return next_temps

        # execute time-stepping loop
        final_temperatures = jax.lax.fori_loop(
            0, params.n_heat_steps, _conduction_step, ltemp
        )

        # save back to state
        state = state._replace(ltemp = final_temperatures)
        
        return state

    def densification(self, state):
        """
        Calculates densification of layers due to 
        compression from overlying mass.

        Returns
        -------
        squeezed_out : jnp.ndarray
            Water squeezed out as a result of decreasing
            pore space [kg m-2]
        """
        params = self.params

        # CONSTANTS
        GRAVITY = params.gravity
        R = params.R_gas
        VISCOSITY_SNOW = params.viscosity_snow
        rho = params.constant_snowfall_density
        DENSITY_FRESH_SNOW = rho if rho else 50
        DENSITY_ICE = params.density_ice
        DENSITY_WATER = params.density_water
        CTOK = params.celsius_to_kelvin
        dt = params.daily_dt

        # load inputs
        ldensity = state.ldensity
        ltemp = state.ltemp
        lice = state.lice
        lwater = state.lwater
        lmass = lice + lwater
        N_POINTS = lice.shape[0]

        # Boone / Anderson (1976) method (COSIPY)
        if params.method_densification in ['Boone']:
            # EMPIRICAL PARAMETERS
            c1 = params.Boone_c1
            c2 = params.Boone_c2
            c3 = params.Boone_c3
            c4 = params.Boone_c4
            c5 = params.Boone_c5

            # shift cumulative mass down by one (top layer has no weight above)
            cumulative_mass = jnp.cumsum(lmass, axis=1)[:, :-1]
            
            # fill top layer with zeros and calculate weight 
            weight_above = GRAVITY * jnp.hstack([jnp.zeros((N_POINTS, 1)), cumulative_mass])

            # get terms in Boone equation
            viscosity = VISCOSITY_SNOW * jnp.exp(c4 * (0.0 - ltemp) + c5 * ldensity)
            mass_term = weight_above / viscosity
            temp_term = -c2 * (0.0 - ltemp)
            dens_term = -c3 * jnp.maximum(0.0, ldensity - DENSITY_FRESH_SNOW)
            
            # calculate delta Rho for the entire matrix
            dRho = (mass_term + c1 * jnp.exp(temp_term + dens_term)) * ldensity * dt

        # Herron Langway (1980) method
        elif params.method_densification in ['HerronLangway']:
            # yearly accumulation is the maximum layer snow mass in mm w.e. yr-1
            a = layers.max_snow / (dt*365) # kg m-2 = mm w.e.
            k = jnp.zeros_like(ldensity)
            b = jnp.zeros_like(ldensity)
            ltemp_K = ltemp + CTOK

            b = jnp.where(ldensity < 550, 1, 0.5)
            k = jnp.where(
                ldensity < 550, 
                11 * jnp.exp(-10160 / (R * ltemp_K)),
                575 * jnp.exp(-21400 / (R * ltemp_K))
            )
            dRho = k * a**b * (DENSITY_ICE - ldensity) / DENSITY_ICE * dt

        # Kojima (1967) method (JULES)
        elif params.method_densification in ['Kojima']:
            NU_0 = 1e7      # Pa s
            RHO_0 = 50      # kg m-3
            k_S = 4000      # K
            T_m = 0. + CTOK
            ltemp_K = ltemp + CTOK

            # same weight_above calculation as Boone method
            cumulative_mass = jnp.cumsum(lmass, axis=1)[:, :-1]
            weight_above = GRAVITY * jnp.hstack([jnp.zeros((N_POINTS, 1)), cumulative_mass])

            # calculate terms in Kojima equation
            exp_term = jnp.exp(k_S / T_m - k_S / ltemp_K - ldensity / RHO_0)
            dRho = ldensity * weight_above / NU_0 * exp_term

        # calculated updated properties 
        new_ldensity = ldensity + dRho
        safe_new_ldensity = jnp.where(new_ldensity > 0, new_ldensity, 1.0)
        new_lheight = lice / safe_new_ldensity 

        # check if any water was squeezed out by densification
        if params.constant_irrwater:
            frac_irreduc = jnp.full_like(new_ldensity, params.Sr)
        else:
            frac_irreduc = jnp.where(new_ldensity > 500.0, params.Sr_dense, params.Sr_light)
        porosity = 1 - new_ldensity / DENSITY_ICE 
        water_irreduc = porosity * new_lheight * DENSITY_WATER * frac_irreduc
    
        # update water in squeezed_out and new_lwater
        squeezed_out = jnp.sum(jnp.where(
            lwater > water_irreduc, 
            lwater - water_irreduc,
            0), axis = 1)
        new_lwater = jnp.where(
            lwater > water_irreduc,
            water_irreduc,
            lwater
        )

        # store to state
        state = state._replace(
            ldensity = new_ldensity,
            lheight = new_lheight,
            lwater = new_lwater
        )

        # check if new firn or ice layers were created
        state = layers.update_layer_types(state, DENSITY_ICE)

        # add any water that was trapped in newly formed ice layers
        new_ice_mask = state.ltype == 2
        trapped_water_out = jnp.sum(jnp.where(
            new_ice_mask, state.lwater, 0.0
        ), axis=1)
        state = state._replace(lwater=jnp.where(new_ice_mask, 0.0, state.lwater))
        squeezed_out = squeezed_out + trapped_water_out

        # update ldepth and types
        state = layers.update_layer_props(state, DENSITY_ICE)

        return state, squeezed_out
    
    def roughness(self, state):
        """
        Determines the roughness length of the surface, 
        assuming the roughness of snow linearly degrades 
        with time in 60 days from that of fresh snow to firn.

        Returns
        -------
        roughness : jnp.ndarray
            (N_POINTS) surface roughness [m]]
        """
        params = self.params 

        # determine roughness from surface type
        surface_type = state.ltype[:, 0]
        roughness = jnp.minimum(
            params.roughness_fresh_snow + params.roughness_aging_rate * state.days_since_snowfall, 
            params.roughness_aged_snow
        )

        # overwrite firn and ice values
        roughness = jnp.where(surface_type == 1, params.roughness_firn, roughness)
        roughness = jnp.where(surface_type == 2, params.roughness_ice, roughness)

        # return roughness in m
        return roughness / 1000
    
    def evolve_grain_size(self, state, forcings):
        """
        Updates grain size according to wet and dry
        metamorphism, refreeze, and addition of fresh
        snow.

        Returns
        -------
        updated_grainsize : jnp.ndarray
            (N_POINTS, N_LAYERS) updated grainsize [um]
        """
        params = self.params

        # CONSTANTS
        WET_C = params.wet_grain_C
        PI = jnp.pi
        RFZ_GRAINSIZE = params.grainsize_rfz
        FIRN_GRAINSIZE = params.grainsize_firn
        ICE_GRAINSIZE = params.grainsize_ice
        CTOK = params.celsius_to_kelvin
        dt = params.dt

        # get temperatures
        airtemp = forcings.temp
        surftemp = state.surftemp

        # find fresh snow grainsize
        if params.constant_freshgrainsize:
            FRESH_GRAINSIZE = params.constant_freshgrainsize
        else:
            FRESH_GRAINSIZE = jnp.select(
                [airtemp <= -30, airtemp < 0],
                [54.5, 54.5 + 5 * (airtemp + 30)],
                default=204.5
            )[:, None]

        # define snow, firn and ice masks 
        snow_mask = state.snow_mask 
        ice_mask = state.ice_mask
            
        # grab layer masses everywhere
        lice = jnp.where(state.lice > 0, state.lice, 1.0)
        # differential refreeze: added this step
        ldrefreeze = state.ldrefreeze
        # "old snow" - non-refreeze and refreeze already accounted for
        lsnow = state.lice - ldrefreeze
        
        # define mass fractions of old snow and refreeze
        f_snow = lsnow / lice
        f_rfz = ldrefreeze / lice
        
        # calculate liquid water fraction everywhere
        lmass = state.lwater + state.lice
        safe_lmass = jnp.where(lmass > 0, lmass, 1.0)
        f_liq = state.lwater / safe_lmass    # fraction of total mass inc. liquid water

        # grab arrays needed for dry grain metamorphosis lookup
        dz = state.lheight
        T = state.ltemp + CTOK
        p = state.ldensity
        grainsize = state.lgrainsize
        
        # calculate surface temperature in K
        surftempK = surftemp + CTOK

        # sum layer heights safely to avoid div by 0
        dz_top_sum = dz[:, 0] + dz[:, 1]
        safe_dz_top_sum = jnp.where(dz_top_sum > 0.0, dz_top_sum, 1.0)
        safe_dz_top = jnp.where(dz[:, 0] > 0.0, dz[:, 0], 1.0)

        dz_upper_sum = dz[:, :-2] + dz[:, 1:-1]
        dz_lower_sum = dz[:, 1:-1] + dz[:, 2:]
        safe_upper_sum = jnp.where(dz_upper_sum > 0.0, dz_upper_sum, 1.0)
        safe_lower_sum = jnp.where(dz_lower_sum > 0.0, dz_lower_sum, 1.0)
        safe_dz_interior = jnp.where(dz[:, 1:-1] > 0.0, dz[:, 1:-1], 1.0)

        # DRY METAMORPHISM
        if params.constant_drdry:
            # apply constant drdry growth rate except where grainsize is too large
            drdry = jnp.where(
                grainsize < RFZ_GRAINSIZE,
                jnp.full_like(grainsize, params.constant_drdry * dt),
                jnp.zeros_like(grainsize)
            )
        else:
            # calculate dTdz in 2D
            dTdz = jnp.zeros_like(T)

            # top layer gradient utilizes surface temperature
            top_layer_val = (surftempK - (T[:, 0] * dz[:, 0] + T[:, 1] * dz[:, 1]) \
                             / safe_dz_top_sum) / safe_dz_top
            dTdz = dTdz.at[:, 0].set(top_layer_val)

            # interior layers using vectorized slice formulation
            t_upper = (T[:, :-2] * dz[:, :-2] + T[:, 1:-1] * dz[:, 1:-1]) / safe_upper_sum
            t_lower = (T[:, 1:-1] * dz[:, 1:-1] + T[:, 2:] * dz[:, 2:]) / safe_lower_sum
            interior_vals = (t_upper - t_lower) / safe_dz_interior
            # set interior values to 0.0 if there is no layer mass
            interior_vals = jnp.where(dz[:, 1:-1] > 0.0, interior_vals, 0.0)
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

            tau = params.interp_tau(input_matrix).reshape(state.lice.shape)
            kap = params.interp_kap(input_matrix).reshape(state.lice.shape)
            dr0 = params.interp_dr0(input_matrix).reshape(state.lice.shape)

            # calculate denominator in drdry equation
            avoid_div_zero_mask = (tau + grainsize) <= FRESH_GRAINSIZE
            denominator = jnp.where(
                avoid_div_zero_mask, 
                tau + 1e-6, # avoid 0 denominator
                tau + grainsize - FRESH_GRAINSIZE
            )
            safe_base = jnp.maximum(jnp.where(denominator > 0, tau / denominator, 1e-6), 1e-10)
            
            # determine actual dry grain growth rate from parameters
            safe_kap = jnp.maximum(kap, 1e-6)
            drdrydt = dr0 * jnp.power(safe_base, 1.0 / safe_kap) / dt
            drdry = drdrydt * dt

        # WET METAMORPHISM
        grainsize_m = grainsize / 1e6
        safe_grainsize_m_sq = jnp.where(grainsize_m > 0, grainsize_m ** 2, 1.0)
        drwetdt = WET_C * (f_liq ** 3) / (4.0 * PI * safe_grainsize_m_sq)
        drwet = drwetdt * dt * 1e6
        
        # accelerate grain growth?
        if params.option_accel_grains:
            F = jnp.exp(0.01 * state.ldensity)
            drwet = drwet * F

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

    def end_of_summer(self, state):
        """
        Checks prognostically if enough snow fell in the 
        previous days to constitute the start of the 
        accumulation season. If so, snow layers over a 
        certain age are transformed to firn and cumulative 
        refreeze is reset to 0.
        """
        params = self.params
        N_LAYERS = state.lice.shape[1]

        # load state (spatial) inputs
        annual_firn_converted = state.annual_firn_converted

        # points where firn has not been converted yet
        convert_firn_pt = ~annual_firn_converted

        # define function that will be looped to merge snow layers down
        def run_merger_loop(state):
            def _merge_snow_step(i, state):
                # evaluate on the fly if layer i and the next layer down are old snow
                is_layer_old_snow = (state.ltype[:, i] == 0) & (state.lage[:, i] >= params.firn_age)
                is_next_old_snow = (state.ltype[:, i+1] == 0) & (state.lage[:, i+1] >= params.firn_age)
                
                # only merge down if the column trigger is True and both layers are old snow
                active_merge_mask = convert_firn_pt & is_layer_old_snow & is_next_old_snow
                
                # call utility function to merge layers
                return layers.merge_existing_layers(state, active_merge_mask, i, params)
            
            state = jax.lax.fori_loop(0, N_LAYERS - 1, _merge_snow_step, state)

            # code the old snow as firn
            is_remaining_old_snow = (state.ltype == 0) & (state.lage >= params.firn_age)
            new_ltype = jnp.where(
                convert_firn_pt[:, None] & is_remaining_old_snow,
                1, state.ltype
            )
        
            # reset cumulative refreeze and annual albedo
            new_lrefreeze = jnp.zeros_like(state.lrefreeze)
            new_firn_converted = jnp.where(
                convert_firn_pt, True, annual_firn_converted
            )

            # store to state
            state = state._replace(
                ltype = new_ltype,
                lrefreeze = new_lrefreeze,
                annual_firn_converted = new_firn_converted
            )

            # update layer masks
            state = layers.update_layer_props(state, params.density_ice)
            return state

        # execute scanning function, if needed 
        any_point_merging = jnp.any(convert_firn_pt)
        state = jax.lax.cond(
            any_point_merging, run_merger_loop, lambda s: s, state
        )

        return state