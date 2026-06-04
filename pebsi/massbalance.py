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
        
        return rain,snow  # kg m-2

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

    # def get_grain_size(self):
    #     """
    #     Updates grain size according to wet and dry
    #     metamorphism, refreeze, and addition of fresh
    #     snow.
    #     """
    #     # get classes
    #     enbal = self.enbal
    #     layers = self.layers
    #     surface = self.surface
    #     args = self.args

    #     # CONSTANTS
    #     WET_C = self.args.wet_grain_C
    #     PI = np.pi
    #     RFZ_GRAINSIZE = args.rfz_grainsize
    #     FIRN_GRAINSIZE = args.firn_grainsize
    #     ICE_GRAINSIZE = args.ice_grainsize
    #     CTOK = args.celsius_to_kelvin
    #     dt = args.dt

    #     # get temperatures
    #     airtemp = enbal.tempC
    #     surftemp = surface.stemp

    #     # find fresh snow grainsize
    #     if args.constant_freshgrainsize:
    #         FRESH_GRAINSIZE = args.constant_freshgrainsize
    #     else:
    #         FRESH_GRAINSIZE = jnp.select(
    #             [airtemp <= -30, airtemp < 0],
    #             [54.5, 54.5 + 5 * (airtemp + 30)],
    #             default=204.5
    #         )[:, jnp.newaxis]

    #     # define snow, firn and ice masks 
    #     snow_mask = layers.snow_mask 
    #     firn_mask = layers.firn_mask 
    #     ice_mask = layers.ice_mask

    #     # exit function if there are no snow layers anywhere
    #     if not jnp.any(snow_mask):
    #         layers.lgrainsize[firn_mask] = FIRN_GRAINSIZE
    #         layers.lgrainsize[ice_mask] = ICE_GRAINSIZE
    #         return
            
    #     # grab layer masses
    #     m_total = layers.lice
    #     m_refreeze = layers.drefreeze       # differential refreeze: added this step
    #     m_snow = layers.lice - m_refreeze   # "old snow" (includes old refreeze)
        
    #     # define mass fractions of old snow and refreeze
    #     f_snow = m_snow / m_total
    #     f_rfz = m_refreeze / m_total
        
    #     # calculate liquid water fraction
    #     mw_total = layers.lwater + layers.lice
    #     f_liq = layers.lwater / mw_total    # fraction of total mass inc. liquid water

    #     # grab arrays needed for dry grain metamorphosis lookup
    #     dz = layers.lheight.copy()
    #     T = layers.ltemp.copy() + CTOK
    #     p = layers.ldensity.copy()
    #     grainsize = layers.lgrainsize.copy()
        
    #     # calculate surface temperature in K
    #     surftempK = surftemp + CTOK

    #     # DRY METAMORPHISM
    #     if args.constant_drdry:
    #         # apply constant drdry growth rate except where grainsize is too large
    #         drdry = jnp.ones_like(grainsize) * args.constant_drdry * dt
    #         drdry[grainsize >= RFZ_GRAINSIZE] = 0.0
    #     else:
    #         # calculate dTdz in 2D
    #         dTdz = jnp.zeros_like(T)
            
    #         # top layer gradient utilizes surface temperature
    #         dTdz[:, 0] = (surftempK - (T[:, 0] * dz[:, 0] + T[:, 1] * dz[:, 1]) \
    #                         / (dz[:, 0] + dz[:, 1])) / dz[:, 0]
            
    #         # interior layers using a vectorized slice formulation
    #         t_upper = (T[:, :-2] * dz[:, :-2] + T[:, 1:-1] * dz[:, 1:-1]) \
    #                         / (dz[:, :-2] + dz[:, 1:-1])
    #         t_lower = (T[:, 1:-1] * dz[:, 1:-1] + T[:, 2:] * dz[:, 2:]) \
    #                         / (dz[:, 1:-1] + dz[:, 2:])
    #         dTdz[:, 1:-1] = (t_upper - t_lower) / dz[:, 1:-1]
            
    #         # bottom layer gets assigned the same dTdz as the layer above
    #         dTdz[:, -1] = dTdz[:, -2]

    #         # take absolute value (direction does not matter)
    #         dTdz = jnp.abs(dTdz)

    #         # Fast matrix bounding to lookup table limits
    #         p = jnp.clip(p, 50.0, 400.0)
    #         dTdz = jnp.clip(dTdz, 0.0, 300.0)
    #         T = jnp.clip(T, 223.15, 273.15)

    #         # flatten matrices to feed your grid interpolators in a single parallel operation
    #         input_matrix = jnp.column_stack((T.ravel(), dTdz.ravel(), p.ravel()))

    #         tau = args.interp_tau(input_matrix).reshape(layers.shape)
    #         kap = args.interp_kap(input_matrix).reshape(layers.shape)
    #         dr0 = args.interp_dr0(input_matrix).reshape(layers.shape)

    #         # calculate denominator in drdry equation
    #         avoid_div_zero_mask = (tau + grainsize) <= FRESH_GRAINSIZE
    #         denominator = jnp.where(
    #             avoid_div_zero_mask, 
    #             tau + 1e-6, # avoid 0 denominator
    #             tau + grainsize - FRESH_GRAINSIZE
    #         )
            
    #         # determine actual dry grain growth rate from parameters
    #         drdrydt = dr0 * jnp.power(tau / denominator, 1.0 / kap) / dt
    #         drdry = drdrydt * dt

    #     # WET METAMORPHISM
    #     grainsize_m = grainsize / 1e6
    #     drwetdt = WET_C * (f_liq ** 3) / (4.0 * PI * (grainsize_m ** 2))
    #     drwet = drwetdt * dt * 1e6
        
    #     # accelerate grain growth?
    #     if args.option_accel_grains:
    #         F = jnp.exp(0.01 * layers.ldensity)
    #         drwet *= F

    #     # apply metamorphosis and refreezing 
    #     aged_grainsize = grainsize + drdry + drwet
    #     updated_grainsize = aged_grainsize * f_snow + RFZ_GRAINSIZE * f_rfz
    #     updated_grainsize = jnp.clip(updated_grainsize, None, RFZ_GRAINSIZE)

    #     # store the updated snow grainsize
    #     layers.lgrainsize[snow_mask] = updated_grainsize[snow_mask]
    #     layers.lgrainsize[firn_mask] = FIRN_GRAINSIZE
    #     layers.lgrainsize[ice_mask] = ICE_GRAINSIZE
    #     return

    # def subsurface_heating(self):
    #     """
    #     Calculates melt in subsurface layers (excluding
    #     layer 0) due to penetrating shortwave radiation.

    #     Returns
    #     -------
    #     layermelt : np.ndarray
    #         Subsurface melt for each layer [kg m-2]
    #     """
    #     # get classes
    #     layers = self.layers
    #     enbal = self.enbal
    #     args = self.args

    #     # check if this function can be skipped
    #     if layers.nlayers == 1: # only one layer: no subsurface to heat
    #         return [0.] # surface melt is filled in melting()
    #     if enbal.SWnet_penetrating < 1e-6: # no penetrating radiation
    #         return np.zeros(layers.nlayers)
        
    #     # CONSTANTS
    #     HEAT_CAPACITY_ICE = args.Cp_ice
    #     HEAT_CAPACITY_WATER = args.Cp_water
    #     LH_RF = args.Lh_rf

    #     # LAYERS IN
    #     ld = layers.ldepth.copy()
    #     lT = layers.ltemp.copy()
    #     lm = layers.lice.copy()
    #     lw = layers.lwater.copy()
    #     lmw = lm + lw

    #     # determine extinction coefficient from surface layer type
    #     if layers.ltype[0] == 'snow':
    #         EXTINCT_COEF = args.extinct_coef_snow
    #     else:
    #         EXTINCT_COEF = args.extinct_coef_ice

    #     # get layer boundaries
    #     d_bottom = ld 
    #     d_top = np.concatenate(([0], d_bottom[:-1]))

    #     # absorbed shortwave for each layer
    #     SWnet_pen = enbal.SWnet_penetrating
    #     SW_at_top = SWnet_pen * np.exp(-EXTINCT_COEF * d_top)
    #     SW_at_bottom = SWnet_pen * np.exp(-EXTINCT_COEF * d_bottom)
    #     layerSW = SW_at_top - SW_at_bottom
    #     layerSW[layerSW < 1e-6] = 0 # cut off tiny amounts of energy
    #     layerSW[0] = 0 # surface layer handled separately

    #     # recalculate layer temperatures, excluding the top layer (calculated separately)
    #     cp_eff = ((lm*HEAT_CAPACITY_ICE) + (lw*HEAT_CAPACITY_WATER)) / (lmw)
    #     lT[1:] += layerSW[1:]*self.dt/(lmw[1:]*cp_eff[1:])

    #     # calculate melt from temperatures above 0
    #     layermelt = np.zeros(layers.nlayers)
    #     leftover_melt = 0

    #     for layer in range (1, layers.nlayers):
    #         temp = lT[layer]

    #         # convert leftover melt to energy [J m-2]
    #         leftover_energy = leftover_melt * LH_RF

    #         if temp > 0.:
    #             # melting: calculate melt energy from layer temperature
    #             sensible_energy = temp * lmw[layer] * cp_eff[layer]
    #             total_energy = sensible_energy + leftover_energy
    #             melt = total_energy / LH_RF
    #             # set layer temp to the melting point
    #             lT[layer] = 0.
    #         else:
    #             # calculate energy needed to warm the layer to melting point
    #             required_energy = abs(temp) * lmw[layer] * cp_eff[layer]

    #             if leftover_energy >= required_energy:
    #                 # use leftover energy to warm to melting point and melt
    #                 lT[layer] = 0.
    #                 leftover_energy -= required_energy
    #                 melt = leftover_energy / LH_RF
    #             else:
    #                 # not enough energy to warm to melting point; warm partially
    #                 lT[layer] += leftover_energy / (lmw[layer] * cp_eff[layer])
    #                 melt = 0

    #         # cap melt at available layer mass
    #         if melt > lm[layer]:
    #             layermelt[layer] = lm[layer]
    #             leftover_melt = melt - lm[layer]
    #         else:
    #             layermelt[layer] = melt
    #             leftover_melt = 0

    #     # force surface layer melt to be 0 (calculated in melting)
    #     layermelt[0] = 0

    #     # LAYERS OUT
    #     layers.ltemp = lT
    #     return layermelt

    # def melting(self,subsurf_melt):
    #     """
    #     For cases when layers are melting. Can melt 
    #     multiple surface layers at once if Qm is 
    #     sufficiently high. Otherwise, adds the surface
    #     layer melt to the array containing subsurface 
    #     melt to return the total layer melt. 
        
    #     This function DOES NOT remove melted mass from 
    #     layers. That is done in percolation().

    #     Parameters
    #     ==========
    #     subsurf_melt : np.ndarray
    #         Subsurface melt for each layer [kg m-2]
        
    #     Returns
    #     -------
    #     layermelt : np.ndarray
    #         Melt for each layer [kg m-2]
    #     """
    #     # get classes
    #     layers = self.layers
    #     args = self.args

    #     # CONSTANTS
    #     LH_RF = args.Lh_rf

    #     # LAYERS IN
    #     lm = layers.lice.copy()
    #     layermelt = subsurf_melt.copy()       # mass of melt due to penetrating SW [kg m-2]
    #     initial_mass = np.sum(layers.lice + layers.lwater)

    #     # calculate surface melt
    #     surface_melt = max(0,self.surface.Qm*self.dt/LH_RF)     # mass of melt due to SEB [kg m-2]

    #     # check if melt by surface energy balance completely melts surface layer
    #     if surface_melt > lm[0]: 
    #         # distribute surface melt into next layers down
    #         layer = 0
    #         while surface_melt > 0 and layer < len(layermelt):
    #             capacity = lm[layer] - layermelt[layer]  # how much more this layer can take
    #             melt_added = min(surface_melt, capacity)
    #             layermelt[layer] += melt_added
    #             surface_melt -= melt_added
    #             layer += 1
    #     else:
    #         # only surface layer is melting or surface melt is 0
    #         layermelt[0] = surface_melt

    #     # check how many layers fully melted
    #     fully_melted = []
    #     if np.any(lm - layermelt <= 0):
    #         melted_subsurf = np.where(lm - layermelt <= 0)[0]
    #         for i in melted_subsurf:
    #             if i not in fully_melted:
    #                 fully_melted.append(i)
    #         fully_melted = np.array(fully_melted, dtype=int)
        
    #     # create melted layers class 
    #     self.melted_layers = MeltedLayers(layers, fully_melted)

    #     # remove layers that were completely melted 
    #     removed = 0 # accounts for indexes of layers changing with loop
    #     for layer in fully_melted:
    #         layers.remove_layer(layer-removed)
    #         removed += 1

    #     # remove fully melted layers from layermelt
    #     mask = np.ones(len(layermelt))
    #     mask[fully_melted] = False
    #     layermelt = layermelt[np.array(mask,dtype=bool)]

    #     # CHECK MASS CONSERVATION
    #     change = np.sum(layers.lice + layers.lwater) - initial_mass
    #     if len(fully_melted) > 0: # account for melted layers
    #         change += np.sum(self.melted_layers.mass)
    #     assert np.abs(change) < args.mb_threshold, f'melting failed mass conservation in {self.output.out_fn}'

    #     return layermelt
        
    # def percolation(self,layermelt,rainfall=0):
    #     """
    #     Updates the liquid water content in each layer
    #     with downward percolation and removes melted
    #     mass from layer dry mass.

    #     Parameters
    #     ==========
    #     layermelt: np.ndarray
    #         Array containing melt amount for each layer
    #     rainfall : float
    #         Additional liquid water input from 
    #         rainfall [kg m-2]

    #     Returns
    #     -------
    #     runoff : float
    #         Runoff of liquid water lost to system [kg m-2]
    #     """
    #     # get classes
    #     layers = self.layers
    #     args = self.args

    #     # CONSTANTS
    #     DENSITY_WATER = args.density_water
    #     DENSITY_ICE = args.density_ice
    #     FRAC_IRREDUC = args.Sr

    #     # get index of percolating (snow/firn) layers
    #     snow_firn_idx = np.concatenate([layers.snow_idx,layers.firn_idx])
    #     # check if there is an ice layer within the snow/firn
    #     if len(snow_firn_idx) > 0 and layers.ice_idx[0] < snow_firn_idx[-1]:
    #         if layers.ice_idx[0] == 0: 
    #             # surface ice layer: all melt/rain runs off
    #             snow_firn_idx = []
    #         else:
    #             # internal layer caused by densification/refreeze
    #             # flow stops (runs off) at ice lens
    #             snow_firn_idx = snow_firn_idx[:layers.ice_idx[0]]

    #     # initialize variables
    #     initial_mass = np.sum(layers.lice + layers.lwater)
    #     rain_bool = rainfall > 0
    #     runoff = 0  # any flow that leaves the point laterally

    #     # get incoming water flux
    #     if len(self.melted_layers.mass) > 0:
    #         # sum of rainfall and mass of fully melted layers
    #         water_in = rainfall + np.sum(self.melted_layers.mass)
    #     else:
    #         # no melted layers, incoming water is just rain
    #         water_in = rainfall

    #     if len(snow_firn_idx) > 0:
    #         # LAYERS IN
    #         lm = layers.lice.copy()[snow_firn_idx]
    #         lw = layers.lwater.copy()[snow_firn_idx]
    #         lh = layers.lheight.copy()[snow_firn_idx]
    #         layermelt_sf = layermelt[snow_firn_idx]

    #         # calculate volumetric fractions (theta)
    #         vol_f_ice = lm / (lh*DENSITY_ICE)
    #         porosity = 1 - vol_f_ice

    #         # remove / move snow melt to layer water
    #         lm -= layermelt_sf
    #         lh -= layermelt_sf / layers.ldensity[snow_firn_idx]
    #         lw += layermelt_sf

    #         # reduce layer refreeze (refreeze melts first)
    #         layers.lrefreeze[snow_firn_idx] -= layermelt_sf
    #         layers.lrefreeze[layers.lrefreeze < 0] = 0

    #         # initialize flow into the top layer
    #         q_out = water_in
    #         q_in_store = []
    #         q_out_store = []
    #         for layer in snow_firn_idx:
    #             # set flow in equal to flow out of the previous layer
    #             q_in = q_out

    #             # irreducible water content depends on density?
    #             if args.constant_irrwater:
    #                 water_irreduc = porosity[layer] * lh[layer] * DENSITY_WATER * FRAC_IRREDUC
    #             else:
    #                 if layers.ldensity[layer] > 500:
    #                     FRAC_IRREDUC = args.Sr_dense
    #                 else:
    #                     FRAC_IRREDUC = args.Sr_light
                        
    #             # calculate flow out of layer i
    #             if q_in < (water_irreduc - lw[layer]):
    #                 q_out = 0
    #             else:
    #                 q_out = q_in - (water_irreduc - lw[layer])

    #             # cannot be negative
    #             q_out = max(0,q_out)

    #             # layer mass balance
    #             lw[layer] += q_in - q_out
    #             q_in_store.append(q_in)
    #             q_out_store.append(q_out)

    #         # LAYERS OUT
    #         layers.lheight[snow_firn_idx] = lh
    #         layers.lwater[snow_firn_idx] = lw
    #         layers.lice[snow_firn_idx] = lm
    #         runoff += q_out + np.sum(layermelt[layers.ice_idx])

    #         # remove melted ice mass (only snow/firn mass was handled above)
    #         for layer in layers.ice_idx:
    #             layers.lice[layer] -= layermelt[layer]
    #             layers.lheight[layer] -= layermelt[layer] / layers.ldensity[layer]

    #         # move LAPs 
    #         if self.args.switch_LAPs == 1:
    #             self.move_LAPs(np.array(q_out_store),rain_bool,snow_firn_idx)
    #     else:
    #         # no percolation, but need to move melt to runoff
    #         layers.lice -= layermelt
    #         layers.lheight -= layermelt / layers.ldensity
    #         runoff += water_in + np.sum(layermelt)

    #     # make sure layers didn't get too small from removing melt
    #     layers.check_layer_sizes()

    #     # CHECK MASS CONSERVATION
    #     ins = water_in
    #     outs = runoff
    #     change = np.sum(layers.lice + layers.lwater) - initial_mass
    #     assert np.abs(change - (ins-outs)) < args.mb_threshold, f'percolation failed mass conservation in {self.output.out_fn}'
    #     return runoff
        
    # def move_LAPs(self,q_out,rain_bool,snow_firn_idx):
    #     """
    #     Moves LAPs vertically through the snow and firn
    #     layers according to water flow from percolation.

    #     Parameters
    #     ==========
    #     q_out : np.ndarray
    #         Water flow out of each layer [kg m-2]
    #     rain_bool : Bool
    #         Raining or not?
    #     snow_firn_idx : np.ndarray
    #         Indices of snow and firn layers
    #     """
    #     # get classes
    #     layers = self.layers
    #     enbal = self.enbal
    #     args = self.args

    #     # CONSTANTS
    #     PARTITION_COEF_BC = args.ksp_BC
    #     PARTITION_COEF_OC = args.ksp_OC
    #     PARTITION_COEF_DUST = args.ksp_dust
    #     dt = args.dt

    #     # LAYERS IN
    #     lw = layers.lwater[snow_firn_idx]
    #     lm = layers.lice[snow_firn_idx]

    #     # layer mass of each species in kg m-2
    #     mBC = layers.lBC[snow_firn_idx]
    #     mOC = layers.lOC[snow_firn_idx]
    #     mdust = layers.ldust[snow_firn_idx]

    #     # get wet deposition into top layer if it's raining
    #     if rain_bool and args.switch_LAPs == 1: # Switch runs have no BC
    #         mBC[0] += enbal.bcwet * dt
    #         mOC[0] += enbal.ocwet * dt
    #         mdust[0] += enbal.dustwet * dt

    #     # layer mass mixing ratio in kg kg-1
    #     cBC = mBC / (lw + lm)
    #     cOC = mOC / (lw + lm)
    #     cdust = mdust / (lw + lm)

    #     # add LAPs from fully melted layers
    #     if self.melted_layers != 0:
    #         m_BC_in_val = np.array(np.sum(self.melted_layers.BC))
    #         m_OC_in_val = np.array(np.sum(self.melted_layers.OC))
    #         m_dust_in_val = np.array(np.sum(self.melted_layers.dust))
    #     else:
    #         m_BC_in_val = np.array([0],dtype=float) 
    #         m_OC_in_val = np.array([0],dtype=float) 
    #         m_dust_in_val = np.array([0],dtype=float)

    #     # initiate arrays to store flow
    #     m_BC_in = np.zeros_like(mBC)
    #     m_BC_out = np.zeros_like(mBC)
    #     m_OC_in = np.zeros_like(mOC)
    #     m_OC_out = np.zeros_like(mOC)
    #     m_dust_in = np.zeros_like(mdust)
    #     m_dust_out = np.zeros_like(mdust)

    #     for i in range(len(mBC)):
    #         # inflow for this layer
    #         m_BC_in[i] = m_BC_in_val
    #         m_OC_in[i] = m_OC_in_val
    #         m_dust_in[i] = m_dust_in_val

    #         # potential outflow
    #         out_BC = PARTITION_COEF_BC * q_out[i] * cBC[i]
    #         out_OC = PARTITION_COEF_OC * q_out[i] * cOC[i]
    #         out_dust = PARTITION_COEF_DUST * q_out[i] * cdust[i]

    #         # outflow cannot exceed what was already there + what just flowed in
    #         m_BC_out[i] = min(out_BC, mBC[i] + m_BC_in[i])
    #         m_OC_out[i] = min(out_OC, mOC[i] + m_OC_in[i])
    #         m_dust_out[i] = min(out_dust, mdust[i] + m_dust_in[i])

    #         # set inflow for the next layer
    #         m_BC_in_val = m_BC_out[i]
    #         m_OC_in_val = m_OC_out[i]
    #         m_dust_in_val = m_dust_out[i]

    #     # mass balance on each constituent
    #     dmBC = m_BC_in - m_BC_out
    #     dmOC = m_OC_in - m_OC_out
    #     dmdust = m_dust_in - m_dust_out
    #     mBC += dmBC.astype(float)
    #     mOC += dmOC.astype(float)
    #     mdust += dmdust.astype(float)

    #     # LAYERS OUT
    #     layers.lBC[snow_firn_idx] = mBC
    #     layers.lOC[snow_firn_idx] = mOC
    #     layers.ldust[snow_firn_idx] = mdust
    #     return

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