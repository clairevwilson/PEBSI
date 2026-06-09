"""
Layers class for PEBSI

Tracks layer properties and contains utility
functions to maintain layer arrays.

@author: clairevwilson
"""
# Built-in libraries
import warnings, sys
warnings.simplefilter('error', RuntimeWarning)
# External libraries
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr
# Internal libraries
from util.config import ConfigError
from jax.debug import print as jax_print

class Layers():
    """
    Layer scheme for the 1D snowpack model.

    All mass terms are stored in kg m-2.
    """
    def __init__(self, args, prms, climate, terrain):
        """
        Initialize the layer properties (temperature, 
        density, water content, LAPs, etc.)

        Parameters
        ==========
        climate
            Class object from pebsi.climate
        args : command line arguments
        """
        # INPUTS
        self.climate = climate 
        self.args = args
        self.prms = prms
        self.N_POINTS = N_POINTS = terrain.N_POINTS
        self.N_LAYERS = N_LAYERS = args.max_nlayers
        self.shape = (N_POINTS, N_LAYERS)

        # load in initial depths of snow, firn and ice in m
        self.dz_snow = np.full(N_POINTS, prms.initial_snow_depth)
        if prms.initial_firn_depth.shape == (1, ):
            # if initial firn depth is a scalar, make it 0 below median glacier elevation
            # (rough approximation of where there should be firn)
            self.dz_firn = np.zeros(N_POINTS)
            self.dz_firn[terrain.elev_n > terrain.median_elev_n] = prms.initial_firn_depth
        else:
            self.dz_firn = prms.initial_firn_depth

        # calculate the layer depths based on initial snow, firn and ice depths
        self.make_layers()
        
        if args.debug:
            print(f'~ {N_LAYERS} layers initialized ~')
        return
    
    def make_layers(self):
        """
        Initializes layer depths based on an exponential
        growth function with prescribed rate of growth 
        and initial layer height. 
        """
        args = self.args
        N_POINTS = self.N_POINTS 
        N_LAYERS = self.N_LAYERS

        # CONSTANTS
        DZ_TOP = args.dz_toplayer
        DZ_SNOW = args.dz_snowlayer
        DZ_ICE = args.dz_icelayer
        LAYER_GROWTH = args.layer_growth

        snow_height = self.dz_snow 
        firn_height = self.dz_firn

        # initialize variables to get filled
        lheight = np.zeros(self.shape)
        ltype = np.zeros(self.shape)

        # define the exponential curve for indexing reference
        layer_indices = np.arange(N_LAYERS)

        # are snow layers a constant size?
        if args.option_uniform_snow:
            curve_snow = np.full(N_LAYERS, DZ_SNOW)
        else:
            curve_snow = DZ_TOP * np.exp(layer_indices * LAYER_GROWTH)

        # are ice layers a constant size?
        if args.option_uniform_ice:
            curve_ice = np.full(N_LAYERS, DZ_ICE)
        else:
            curve_ice = DZ_TOP * np.exp(layer_indices * LAYER_GROWTH)
    
        # create layers for each point
        for i in range(N_POINTS):
            snow_pt = snow_height[i]
            firn_pt = firn_height[i]

            curr_lyr = 0
            accum_depth = 0.0

            # allocate snow layers
            while accum_depth < snow_pt and curr_lyr < N_LAYERS:
                # determine height from layer index
                h = curve_snow[curr_lyr]
                if accum_depth + h > snow_pt:
                    h = snow_pt - accum_depth # Trim the last snow layer
                
                lheight[i, curr_lyr] = h
                ltype[i, curr_lyr] = 0
                accum_depth += h
                curr_lyr += 1

            # allocate firn layers
            if firn_pt > 0:
                # make firn layers of a constant height, approx 1 m
                n_firn = int(round(float(firn_pt), 0)) if firn_pt > 0.75 else 1
                h_firn = firn_pt / n_firn
                
                for _ in range(n_firn):
                    if curr_lyr >= N_LAYERS: break
                    lheight[i, curr_lyr] = h_firn
                    ltype[i, curr_lyr] = 1 # 1 = Firn
                    curr_lyr += 1

            # add ice layers until all layers are full with a thickness cap
            curve_ice_clipped = np.minimum(curve_ice, args.dz_icelayer)
            lheight[i] = np.where(np.arange(N_LAYERS) >= curr_lyr, 
                                  curve_ice_clipped, lheight[i])
            ltype[i] = np.where(np.arange(N_LAYERS) >= curr_lyr, 2, ltype[i])

        # calculate midpoint heights
        cum_height = np.cumsum(lheight, axis=1)
        ldepth = cum_height - (lheight / 2.0)

        # store to self
        self.ltype = np.array(ltype)            # LAYER TYPE (0, 1, or 2) [-]
        self.lheight = np.array(lheight)        # LAYER HEIGHT (dz) [m]
        self.ldepth = np.array(ldepth)          # LAYER DEPTH (midlayer) [m]

        # assign indices
        self.snow_mask = ltype==0       # type 0 = snow
        self.firn_mask = ltype==1       # type 1 = firn
        self.ice_mask = ltype==2        # type 2 = ice
        return

    def initialize_layers(self):
        """
        Initializes the layer temperature, density, 
        water content and grain size.

        Parameters:
        ==========
        snow_height : float
        firn_height : float
            Initial depth of snow and firn [m]
        
        Returns:
        --------
        ltemp, ldensity, lwater, lgrainsize : np.ndarray
            Arrays containing layer temperature [C], 
            density [kg m-3], water content [kg m-2],
            and grain size [um]
        """
        args = self.args
        snow_mask = self.snow_mask
        firn_mask = self.firn_mask
        ice_mask = self.ice_mask

        snow_height = self.dz_snow 
        firn_height = self.dz_firn

        # read in depth profiles
        temp_data = pd.read_csv(self.args.initial_temp_fn)
        density_data = pd.read_csv(self.args.initial_density_fn)
        grainsize_data = pd.read_csv(self.args.initial_grains_fn)

        # ===== TEMPERATURE [C] =====
        if args.initialize_temp == 'interpolate':
            ltemp = np.interp(
                self.ldepth.ravel(),
                temp_data['depth'],
                temp_data['temp']
            ).reshape(self.shape)
        elif args.initialize_temp == 'ripe':
            ltemp = np.zeros(self.shape)
        else:
            raise ConfigError('Invalid configuration: initialize_temp')
        
        # ===== GRAIN SIZE [um] =====
        lgrainsize = np.interp(
            self.ldepth.ravel(),
            grainsize_data['depth'],
            grainsize_data['grainsize']
        ).reshape(self.shape)
        lgrainsize[self.ltype == 1] = args.grainsize_firn
        lgrainsize[self.ltype == 2] = args.grainsize_ice

        # ===== DENSITY [kg m-3] =====
        if args.initialize_density == 'interpolate':
            # SNOW layers initialized by interpolation
            ldensity = np.interp(
                self.ldepth.ravel(),
                density_data['depth'],
                density_data['density']
            ).reshape(self.shape)

            # find the snow density at the bottom of each point
            bottom_snow_density = np.nanmax(ldensity * snow_mask, axis=1)
            bottom_snow_depth = np.nanmax(self.ldepth * snow_mask, axis=1)

            # if there is no snow, swap in constant firn_density
            bottom_snow_density = np.where(snow_height > 0, 
                                           bottom_snow_density, 
                                           args.density_firn)

            # compute the density slope from top to bottom of firn
            safe_firn_height = np.where(firn_height > 0, firn_height, 1.0)
            pslope = np.where(
                firn_height > 0,
                (args.density_ice - bottom_snow_density) / safe_firn_height,
                0.0,
            )[:, np.newaxis]

            # apply the firn interpolation
            bottom_snow_density = bottom_snow_density[:, np.newaxis]
            bottom_snow_depth = bottom_snow_depth[:, np.newaxis]
            firn_densities = bottom_snow_density + pslope* (self.ldepth - bottom_snow_depth)
            ldensity = np.where(firn_mask, firn_densities, ldensity)

            # set constant ice layer density
            ldensity = np.where(ice_mask, args.density_ice, ldensity)
        elif args.initialize_density == 'constant':
            ldensity = np.ones(self.shape) * args.density_snow
            ldensity = np.where(firn_mask, args.density_firn, ldensity)
            ldensity = np.where(ice_mask, args.density_ice, ldensity)
        else:
            raise ConfigError('Invalid configuration: initialize_density')
        
        # calculate dry mass from density
        lice = ldensity * self.lheight

        # WATER CONTENT [kg m-2]
        if args.initialize_water == 'dry':
            lwater = np.zeros(self.shape)
        elif args.initialize_water == 'saturated':
            porosity = 1 - ldensity / args.density_ice
            lwater = porosity * args.Sr * self.lheight * args.density_water
        else:
            raise ConfigError('Invalid configuration: initialize_water')
        
        # ===== AGE [days] =====
        firn_mask_int = self.firn_mask.astype(int)
        lage = np.cumsum(firn_mask_int, axis=1) * -365

        # firn ages count back in time; snow/ice initialized at 0
        lage = np.where(self.ltype >= 1, lage, 0.0)

        # ===== LAPs [kg m-2] =====
        self.initialize_LAPs()

        # ===== STORE EVERYTHING TO SELF =====
        # running maximum snow mass (reset each year)
        self.max_snow = np.sum(lice * self.snow_mask, axis=1)

        # sum masses for mass conservation checks
        self.mass_water = np.sum(lwater, axis=1)
        self.mass_ice = np.sum(lice, axis=1)
        self.mass = np.sum(lwater + lice, axis=1)

        # main properties
        self.ltemp = np.array(ltemp)                    # LAYER TEMPERATURE [C]
        self.ldensity = np.array(ldensity)              # LAYER DENSITY [kg m-3]
        self.lice = np.array(lice)     # LAYER ICE (SOLID) MASS [kg m-2]
        self.lwater = np.array(lwater)                  # LAYER WATER (LIQUID) MASS [kg m-2]
        self.lgrainsize = np.array(lgrainsize)          # LAYER GRAIN SIZE [um]
        self.drefreeze = np.zeros_like(self.ltemp)      # LAYER REFREEZE MASS ADDED PER TIMESTEP [kg m-2]
        self.lrefreeze = np.zeros_like(self.ltemp)      # LAYER REFREEZE MASS [kg m-2]
        self.lage = np.array(lage)      # LAYER AGE [days]
        return
    
    def initialize_LAPs(self):
        """
        Initializes light-absorbing particle content
        of the snow and firn layers.
        """
        args = self.args

        # CONSTANTS
        BC_FRESH = args.BC_freshsnow
        OC_FRESH = args.OC_freshsnow
        DUST_FRESH = args.dust_freshsnow

        # INPUTS
        lheight = self.lheight
        ldepth = self.ldepth

        if args.initialize_LAPs in ['clean']:
            # snowpack is clean; initialize as constant values
            lBC = BC_FRESH*lheight
            lOC = OC_FRESH*lheight
            ldust = DUST_FRESH*lheight 
        elif args.initialize_LAPs in ['interpolate']:
            # read in LAP data
            lap_data = pd.read_csv(args.initial_LAP_fn,index_col=0)

            # handle nans
            lap_data = lap_data.sort_index()
            bc_data = lap_data['BC'].dropna()
            oc_data = lap_data['OC'].dropna()
            dust_data = lap_data['dust'].dropna()

            # interpolate concentration by depth
            cBC = np.interp(
                ldepth.ravel(), bc_data.index, bc_data
            ).reshape(self.shape)
            cOC = np.interp(
                ldepth.ravel(), oc_data.index, oc_data
            ).reshape(self.shape)
            cdust = np.interp(
                ldepth.ravel(), dust_data.index, dust_data
            ).reshape(self.shape)

            # calculate mass from concentration
            lBC = cBC * lheight
            lOC = cOC * lheight
            ldust = cdust * lheight
        else:
            raise ConfigError('Invalid configuration: initialize_LAPs')
        
        # store to self
        self.lBC = lBC          # LAYER BLACK CARBON MASS [kg m-2]
        self.lOC = lOC          # LAYER ORGANIC CARBON MASS [kg m-2]
        self.ldust = ldust      # LAYER DUST MASS [kg m-2]
        return
    
# ========= UTILITY FUNCTIONS ==========
def add_top_layer(state, mask, new_layer,):
    """
    Adds a new layer to the top of layers
    for the points in mask.

    Parameters
    ==========
    state : NamedTuple
        Current model state containing point/layer properties
    mask : jnp.Array (N_POINTS,)
        Boolean mask for points where a new layer is added
    new_layer : dict
        Dictionary with new layer properties
    """
    # first need to make room. push the bottom layer mass into the basal reservoir
    ice_leaving = state.lice[:, state.lice.shape[1] - 1]
    water_leaving = state.lwater[:, state.lice.shape[1] - 1]
    mass_leaving = ice_leaving + water_leaving
    new_reservoir = jnp.where(
        mask, state.basal_reservoir + mass_leaving, state.basal_reservoir
    )
    state = state._replace(basal_reservoir = new_reservoir)

    # convert namespace to a dictionary of {attribute_name: array_values}
    properties = state._asdict()
    for attr, new_values in new_layer.items():
        data = properties[attr]
        
        # shift everything downwards by one
        shifted = data.at[:, 1:].set(data[:, :-1])
        
        # insert the new layer data to index 0
        shifted_new = shifted.at[:, 0].set(new_values)

        # mask logic
        properties[attr] = jnp.where(
            mask[:, None],
            shifted_new, 
            data
        )
    
    state = state._replace(**properties)
    return state

def add_bottom_layer(state, mask, args):
    """
    Fills in data for a new bottom layer added
    as a result of a different layer being removed.

    For points with ice layers, all ice layers have
    their mass redistributed evenly.

    For points with no ice layers, a new ice bottom
    layer is pulled up from the basal reservoir.

    Parameters
    ==========
    mask : jnp.Array (N_POINTS)
        Boolean mask for points where a new layer is added
    """
    properties = state._asdict()
    N_POINTS = state.lice.shape[0]
    ice_mask = properties['ice_mask']

    # calculate amount of ice at each point
    ice_masses = jnp.where(ice_mask, properties['lice'], 0.0)
    point_ice_mass = jnp.sum(ice_masses, axis=1, keepdims=True)

    # only redistribute mass if there is enough and reservoir is empty
    has_viable_ice = (point_ice_mass > args.min_glacier_depth).squeeze()
    empty_reservoir = state.basal_reservoir < 1e-3
    redistribute = mask & has_viable_ice & empty_reservoir
    use_reservoir = mask & ~empty_reservoir
    truly_dead = mask & ~has_viable_ice & empty_reservoir

    # case 1: redistribute mass across existing ice layers
    DZ_ICE = args.dz_icelayer
    DZ_TOP = args.dz_toplayer 
    LAYER_GROWTH = args.layer_growth

    layers_idx = jnp.arange(state.lice.shape[1])
    initial_ice_heights = DZ_TOP * jnp.exp(layers_idx * LAYER_GROWTH)

    # cap initial ice heights like we did in initialization
    initial_ice_heights = jnp.minimum(initial_ice_heights, DZ_ICE)
    
    # make layerheight 0 anywhere that isn't ice
    ice_heights_2D = jnp.where(ice_mask, initial_ice_heights[None, :], 0)

    # calculate fraction of ice height each layer should have
    sum_ice_heights = jnp.sum(ice_heights_2D, axis=1, keepdims=True)
    safe_sum = jnp.where(sum_ice_heights > 0, sum_ice_heights, 1.0)
    ice_fractions = ice_heights_2D / safe_sum

    # scale the weight of mass per layer (non-ice get weight of 0)
    mass_redistributed = point_ice_mass * ice_fractions
    
    # distribute the lost mass according to those exponential weights
    properties['lice'] = jnp.where(
        redistribute[:, None] & ice_mask,
        mass_redistributed,
        properties['lice']
    )
    properties['ldensity'] = jnp.where(
        redistribute[:, None] & ice_mask,
        args.density_ice,
        properties['ldensity']
    )

    # recalculate layer heights
    properties['lheight'] = properties['lice'] / properties['ldensity']
    
    # case 2: pull new ice layer from reservoir into bottom

    DENSITY_ICE = args.density_ice 
    TEMP_TEMP = args.temp_temp
    GRAINSIZE_ICE = args.grainsize_ice

    # create new ice layer
    new_bottom_layer = {
        'ldepth': properties['ldepth'][:, -1],
        'lheight': jnp.full(N_POINTS, DZ_ICE),
        'ldensity': jnp.full(N_POINTS, DENSITY_ICE),
        'lice': jnp.full(N_POINTS, DENSITY_ICE* DZ_ICE),
        'ltemp': jnp.full(N_POINTS, TEMP_TEMP),
        'ltype': jnp.full(N_POINTS, 2, dtype=jnp.int32),
        'lage': properties['lage'][:, -2] + 365,
        'lgrainsize': jnp.full(N_POINTS, GRAINSIZE_ICE),
        'lwater': jnp.zeros(N_POINTS),
        'lrefreeze': jnp.zeros(N_POINTS),
        'ldrefreeze': jnp.zeros(N_POINTS),
        'lBC': jnp.zeros(N_POINTS),
        'lOC': jnp.zeros(N_POINTS),
        'ldust': jnp.zeros(N_POINTS),
    }

    # make sure there is sufficient mass in the reservoir
    new_bottom_layer['lice'] = jnp.minimum(
        new_bottom_layer['lice'], properties['basal_reservoir']
    )

    # re-calculate the layer height in case mass changed
    new_bottom_layer['lheight'] = new_bottom_layer['lice'] \
                                 / new_bottom_layer['ldensity']

    for var in args.all_layer_vars:
        data = properties[var]
        new_data = new_bottom_layer[var]

        # stage the update
        updated_column = data.at[:, -1].set(new_data)

        # replace it only at points in mask
        properties[var] = jnp.where(
            use_reservoir[:, None],
            updated_column,
            data
        )

    # remove added mass from reservoir
    properties['basal_reservoir'] = jnp.where(
        use_reservoir,
        properties['basal_reservoir'] - new_bottom_layer['lice'],
        properties['basal_reservoir']
    )

    # check if any points are dead and fill them up with zeros if so    
    properties['lice'] = jnp.where(truly_dead[:, None], 0.0, properties['lice'])
    properties['lwater'] = jnp.where(truly_dead[:, None], 0.0, properties['lwater'])
    properties['lheight'] = jnp.where(truly_dead[:, None], 0.0, properties['lheight'])
    properties['ltype'] = jnp.where(truly_dead[:, None], 2, properties['ltype'])

    # save these properties to state
    state = state._replace(**properties)

    # update layers 
    state = update_layer_props(state, DENSITY_ICE)
    return state

def remove_layer(state, mask, idx, args):
    """
    Removes layers from layer idx (scalar or 
    1D array) for the points in mask.

    Parameters
    ==========
    mask : jnp.Array (N_POINTS)
        Boolean mask for points where a layer is removed
    idx : int
        Index of the layer to remove
    """
    properties = state._asdict()

    # load layer index array
    layers_idx = jnp.arange(state.lice.shape[1])

    # combine point mask with index mask
    # (shift layers below removed up by one)
    target_mask = mask[:, None] & (
        layers_idx[None, :] >= jnp.atleast_1d(idx)[:, None]
    )

    for var in args.all_layer_vars:
        data = properties[var]

        # shift everything upwards by one
        fully_shifted = data.at[:, :-1].set(data[:, 1:])

        # make sure new bottom layer is filled with no mass
        if var == 'ltype':
            fully_shifted = fully_shifted.at[:, -1].set(2)
        elif var == 'ldensity':
            fully_shifted = fully_shifted.at[:, -1].set(args.density_ice)
        else:
            fully_shifted = fully_shifted.at[:, -1].set(0.0)
        
        # replace it only at points / layers in mask
        properties[var] = jnp.where(
            target_mask,
            fully_shifted,
            data
        )

    # write these properties to state
    state = state._replace(**properties)

    # update ice mask
    state = update_layer_props(state, args.density_ice)

    # add a new bottom layer
    state = add_bottom_layer(state, mask, args)        
    return state

def split_layer(state, mask, idx, args):
    """
    Splits a single layer into two layers. Extensive
    properties are halved and intensive properties 
    are maintained.

    Parameters
    ==========
    mask : jnp.Array (N_POINTS)
        Boolean mask for points where a layer is split
    layer_to_split : int
        Index of the layer to split for points in mask
    """
    properties = state._asdict()

    # first need to make room. push the bottom layer mass into the basal reservoir
    mass_leaving = state.lice[:, state.lice.shape[1] - 1]
    properties['basal_reservoir'] = jnp.where(
        mask, state.basal_reservoir + mass_leaving, state.basal_reservoir
    )

    # load layer index array
    layers_idx = jnp.arange(state.lice.shape[1])

    # combine point mask with index mask
    # (shift points at or below index down by one)
    target_mask = mask[:, None] & (layers_idx > idx)[None, :]

    for var in args.all_layer_vars:
        data = properties[var]

        # shift everything downwards by one
        fully_shifted = data.at[:, 1:].set(data[:, :-1])

        # replace it only at points / layers in mask
        properties[var] = jnp.where(
            target_mask,
            fully_shifted,
            data
        )

    # halve extensive properties at layers that were copied
    halve_condition = (layers_idx == idx) | (layers_idx == idx + 1)
    halve_mask = mask[:, None] & halve_condition[None, :]

    for var in args.extensive_vars:
        data = properties[var]
        
        # calculate halved quantities at every point
        halved_data = data / 2.0
        
        # replace it only at points / layers in mask
        properties[var] = jnp.where(
            halve_mask,
            halved_data,
            properties[var] # use the shifted data
        )

    # recalculate lheight directly
    properties['lheight'] = properties['lice'] / properties['ldensity']

    # write these properties to state
    state = state._replace(**properties)

    # finally, send state through the profile updates utility
    state = update_layer_props(state, args.density_ice)
    
    return state

def merge_existing_layers(state, mask, idx, args):
    """
    Merges two layers into one. Extensive properties
    are added and intensive properties are averaged.

    Parameters
    ==========
    mask : jnp.Array (N_POINTS)
        Boolean mask for points where a new layer is added
    idx : int
        Index of the layer to merge with the layer
        beneath it for each point in mask
    """
    properties = state._asdict()
    layers_idx = jnp.arange(state.lice.shape[1])

    # idx is the layer being removed and merged down into target_idx
    target_idx = idx + 1

    # calculate mass weights between the two existing layers
    m_removed = properties['lice'][:, idx]
    m_target = properties['lice'][:, target_idx]
    m_total = m_removed + m_target
    m_safe = jnp.where(m_total > 0, m_total, 1.0)

    # combine point mask with index mask
    target_mask = mask[:, None] & (layers_idx == target_idx)[None, :]

    # take weighted average for intensive variables
    for var in args.intensive_vars:
        data = properties[var]
        removed_vals = data[:, idx]
        target_vals = data[:, target_idx]

        # calculate mass-weighted average of values (N_POINTS)
        weighted_avg = (target_vals * m_target + removed_vals * m_removed) / m_safe
        
        if var in ['ltype','lage']:
            weighted_avg = jnp.round(weighted_avg).astype(jnp.int32)

        # replace it only at points / layers in mask
        properties[var] = jnp.where(
            target_mask,
            weighted_avg[:, None],
            data
        )

    # sum for extensive variables
    for var in args.extensive_vars:
        data = properties[var]

        # sum values
        summed_data = data[:, idx] + data[:, target_idx]

        properties[var] = jnp.where(
            target_mask,
            summed_data[:, None],
            data
        )

    # combine point mask with index mask
    # (layers at or below target index move up one)
    shift_mask = mask[:, None] & (layers_idx >= idx)[None, :]

    for var in args.all_layer_vars:
        data = properties[var]

        # shift everything upwards by one
        fully_shifted = data.at[:, :-1].set(data[:, 1:])

        # make sure new bottom layer is filled with no mass
        if var == 'ltype':
            fully_shifted = fully_shifted.at[:, -1].set(2)
        elif var == 'ldensity': # fill with real value to avoid div 0
            fully_shifted = fully_shifted.at[:, -1].set(args.density_ice)
        else:
            fully_shifted = fully_shifted.at[:, -1].set(0.0)

        # replace it only at points / layers in mask
        properties[var] = jnp.where(
            shift_mask, fully_shifted, data
        )

    # update ice mask and layer height
    properties['ice_mask'] = properties['ltype'].astype(jnp.int32) == 2
    properties['lheight'] = properties['lice'] / properties['ldensity']
    
    # write these properties to state 
    state = state._replace(**properties)
    
    # add bottom layer to replaced removed
    state = add_bottom_layer(state, mask, args)

    # recalculate layer heights
    updated_lheight = state.lice / state.ldensity 
    state = state._replace(lheight=updated_lheight)

    # finally, send state through the profile updates utility
    state = update_layer_props(state, args.density_ice)
    
    return state

def merge_new_layer(state, mask, new_layer, args):
    """
    Merges accumulation into existing top layer
    for points in mask. Extensive properties are 
    added and intensive properties are averaged.

    Parameters
    ==========
    mask : jnp.Array (N_POINTS)
        Boolean mask for points where a new layer is added
    new_layer : object
        Namespace container with new layer properties
    """
    properties = state._asdict()
    layers_idx = jnp.arange(state.lice.shape[1])

    # calculate mass weights between the two existing layers
    m_new = new_layer['lice']
    m_target = properties['lice'][:, 0]
    m_total = m_new + m_target
    m_safe = jnp.where(m_total > 0, m_total, 1.0)

    # combine point mask with index mask
    target_mask = mask[:, None] & (layers_idx == 0)[None, :]

    # take weighted average for intensive variables
    for var in args.intensive_vars:
        data = properties[var]
        new_vals = new_layer[var]
        target_vals = data[:, 0]

        # calculate mass-weighted average of values (N_POINTS)
        weighted_avg = (target_vals * m_target + new_vals * m_new) / m_safe
        
        if var in ['ltype','lage']:
            weighted_avg = jnp.round(weighted_avg).astype(jnp.int32)

        # replace it only at points / layers in mask
        properties[var] = jnp.where(
            target_mask,
            weighted_avg[:, None],
            data
        )

    # sum for extensive variables
    for var in args.extensive_vars:
        data = properties[var]

        # sum values
        summed_data = data[:, 0] + new_layer[var]

        properties[var] = jnp.where(
            target_mask,
            summed_data[:, None],
            data
        )

    # write these properties to state 
    state = state._replace(**properties)

    # recalculate layer heights
    updated_lheight = state.lice / state.ldensity 
    state = state._replace(lheight=updated_lheight)

    # finally, send state through the profile updates utility
    state = update_layer_props(state, args.density_ice)
    
    return state

def check_layer_sizes(state, args):
    """
    Scans through layers sequentially from top to bottom.
    If a layer is below the minimum height threshold, it is 
    merged with the layer directly beneath it.
    """
    properties = state._asdict()
    n_points, n_layers = properties['lice'].shape

    # zero out dead layers
    threshold = args.mb_threshold / 1000.0
    dead_mask = properties['lice'] < threshold

    dead_mass = jnp.sum(jnp.where(
        dead_mask, properties['lice'] + properties['lwater'], 0.0
    ), axis=1)
    
    properties['lice'] = jnp.where(dead_mask, 0.0, properties['lice'])
    properties['lwater'] = jnp.where(dead_mask, 0.0, properties['lwater'])
    properties['lheight'] = jnp.where(dead_mask, 0.0, properties['lheight'])
    state = state._replace(**properties)

    # define function to scan for layers to merge
    def _scan_layers(current_state, idx):
        # always fetch the most up-to-date heights and types from evolving state
        dz = current_state.lheight[:, idx]
        curr_type = current_state.ltype[:, idx]
        next_type = current_state.ltype[:, idx + 1]
        
        # determine which spatial columns need a merge at this specific vertical index
        is_too_thin = dz < args.min_dz
        is_snow = curr_type == 0
        type_matches_below = curr_type == next_type
        force_small_snow = (curr_type == 0) & (next_type > 0)
        
        # build the boolean merge mask (N_POINTS)
        merge_mask = is_too_thin & (~is_snow | type_matches_below | force_small_snow)

        # merge layers, if there are layers to merge
        next_state = merge_existing_layers(current_state, merge_mask, idx, args)

        return next_state, None 
    
    layers_idx = jnp.arange(n_layers - 1) # don't include bottom layer
    state, _ = jax.lax.scan(_scan_layers, state, layers_idx)

    return state, dead_mass

def update_layer_props(state, DENSITY_ICE):
    """
    Recalculates nlayers, depths, and density. 
    Can specify to only update certain properties.

    Parameters
    ==========
    do : list-like
        List of any combination of depth, density to be updated
    """
    safe_lheight = jnp.where(state.lheight > 0, state.lheight, 1.0)
    
    new_ice_mask = state.ltype == 2
    new_density = jnp.where(
        new_ice_mask,
        DENSITY_ICE,
        state.lice / safe_lheight
    )

    lh = state.lheight
    new_depth = jnp.cumsum(lh, axis=1) - (lh / 2.0)

    # update state
    state = state._replace(
        ldepth = new_depth,
        ldensity = new_density,
        snow_mask = state.ltype == 0,
        firn_mask = state.ltype == 1,
        ice_mask = new_ice_mask
    )

    return state

def update_layer_types(state, DENSITY_ICE):
    """
    Checks if new ice layers have been created by 
    the densification of firn.
    """

    # find which firn layers have crossed the ice density threshold
    is_firn = state.ltype == 1
    density_check = state.ldensity >= DENSITY_ICE
    trans_mask = is_firn & density_check

    # apply the ice transformation
    new_ltype = jnp.where(trans_mask, 2, state.ltype)
    new_ldensity = jnp.where(trans_mask, DENSITY_ICE, state.ldensity)
    new_lheight = jnp.where(
        trans_mask, state.lice / new_ldensity, state.lheight
    )

    # update state
    state = state._replace(
        ltype = new_ltype, ldensity = new_ldensity,
        lheight = new_lheight,
    )
    return state