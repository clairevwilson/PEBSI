"""
Layers class for PEBSI

Tracks layer properties and contains utility
functions to maintain layer arrays.
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
from pebsi.config import ConfigError

class Layers():
    """
    Layer scheme for the 1D snowpack model.

    All mass terms are stored in kg m-2.
    """
    def __init__(self, params, terrain):
        """
        Initialize the layer properties (temperature, 
        density, water content, LAPs, etc.)
        """
        # INPUTS
        self.params = params
        self.N_POINTS = N_POINTS = terrain.N_POINTS
        self.N_LAYERS = N_LAYERS = params.max_nlayers
        self.shape = (N_POINTS, N_LAYERS)

        # load in initial depths of snow, firn and ice in m
        self.dz_snow = np.full(N_POINTS, params.initial_snow_depth)
        if params.initial_firn_depth.shape == (1, ):
            # if initial firn depth is a scalar, make it 0 below median glacier elevation
            # (rough approximation of where there should be firn)
            self.dz_firn = np.zeros(N_POINTS)
            self.dz_firn[terrain.elev_n > terrain.median_elev_n] = params.initial_firn_depth
        else:
            self.dz_firn = params.initial_firn_depth

        # calculate the layer depths based on initial snow, firn and ice depths
        self.make_layers()

        # reconcile ice thickness differences from dynamics
        if params.option_dynamics:
            self.apply_initial_ice_thickness(terrain)

        if params.debug:
            print(f'~ {N_LAYERS} layers initialized ~')
        return
    
    def make_layers(self):
        """
        Initializes layer depths based on an exponential
        growth function with prescribed rate of growth 
        and initial layer height. 
        """
        params = self.params
        N_POINTS = self.N_POINTS 
        N_LAYERS = self.N_LAYERS

        # CONSTANTS
        DZ_TOP = params.dz_toplayer
        DZ_SNOW = params.dz_snowlayer
        DZ_ICE = params.dz_icelayer
        LAYER_GROWTH = params.layer_growth

        snow_height = self.dz_snow 
        firn_height = self.dz_firn

        # initialize variables to get filled
        lheight = np.zeros(self.shape)
        ltype = np.zeros(self.shape)

        # define the exponential curve for indexing reference
        layer_indices = np.arange(N_LAYERS)
        curve_snow = DZ_TOP * np.exp(layer_indices * LAYER_GROWTH)
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
            curve_ice_clipped = np.minimum(curve_ice, params.dz_icelayer)
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

    def apply_initial_ice_thickness(self, terrain):
        """
        Reconciles the ice-typed portion of the column to each point's
        real initial ice thickness (terrain.thickness_n). Only needed
        when option_dynamics=True, otherwise ice thickness is an 
        arbitrary variable. 

        Assumes the ice thickness in the dataset is representative
        of firn and ice. Since firn layers are initialized separately,
        this function ONLY applies height changes to ice. Ice layers 
        are resized such that every layer is the same height and all
        ice + firn layers add up to the actual initial ice thickness.
        """
        min_height = self.params.min_dz

        n_ice_layers = np.sum(self.ice_mask, axis=1)  # (N_POINTS,)
        ice_only_thickness = np.maximum(terrain.thickness_n - self.dz_firn, 0.0)

        per_layer_height = np.where(
            n_ice_layers > 0, ice_only_thickness / np.maximum(n_ice_layers, 1), 0.0
        )
        per_layer_height = np.maximum(per_layer_height, min_height)

        self.lheight = np.where(self.ice_mask, per_layer_height[:, np.newaxis], self.lheight)

        # depths need recomputing everywhere since every ice layer's height changed
        cum_height = np.cumsum(self.lheight, axis=1)
        self.ldepth = cum_height - (self.lheight / 2.0)
        return

    def initialize_layers(self):
        """
        Initializes the layer temperature, density, 
        water content, grain size, LAPs, and age.
        """
        params = self.params
        snow_mask = self.snow_mask
        firn_mask = self.firn_mask
        ice_mask = self.ice_mask

        snow_height = self.dz_snow 
        firn_height = self.dz_firn

        # read in depth profiles
        temp_data = pd.read_csv(self.params.initial_temp_fn)
        density_data = pd.read_csv(self.params.initial_density_fn)
        grainsize_data = pd.read_csv(self.params.initial_grains_fn)

        # ===== TEMPERATURE [C] =====
        if params.initialize_temp == 'interpolate':
            ltemp = np.interp(
                self.ldepth.ravel(),
                temp_data['depth'],
                temp_data['temp']
            ).reshape(self.shape)
        elif params.initialize_temp == 'ripe':
            ltemp = np.zeros(self.shape)
        else:
            raise ConfigError('Invalid configuration: initialize_temp')
        
        # ===== GRAIN SIZE [um] =====
        lgrainsize = np.interp(
            self.ldepth.ravel(),
            grainsize_data['depth'],
            grainsize_data['grainsize']
        ).reshape(self.shape)
        lgrainsize[self.ltype == 1] = params.grainsize_firn
        lgrainsize[self.ltype == 2] = params.grainsize_ice

        # ===== DENSITY [kg m-3] =====
        if params.initialize_density == 'interpolate':
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
                                           params.density_firn)

            # compute the density slope from top to bottom of firn
            safe_firn_height = np.where(firn_height > 0, firn_height, 1.0)
            pslope = np.where(
                firn_height > 0,
                (params.density_ice - bottom_snow_density) / safe_firn_height,
                0.0,
            )[:, np.newaxis]

            # apply the firn interpolation
            bottom_snow_density = bottom_snow_density[:, np.newaxis]
            bottom_snow_depth = bottom_snow_depth[:, np.newaxis]
            firn_densities = bottom_snow_density + pslope* (self.ldepth - bottom_snow_depth)
            ldensity = np.where(firn_mask, firn_densities, ldensity)

            # set constant ice layer density
            ldensity = np.where(ice_mask, params.density_ice, ldensity)
        elif params.initialize_density == 'constant':
            ldensity = np.ones(self.shape) * params.density_snow
            ldensity = np.where(firn_mask, params.density_firn, ldensity)
            ldensity = np.where(ice_mask, params.density_ice, ldensity)
        else:
            raise ConfigError('Invalid configuration: initialize_density')
        
        # calculate dry mass from density
        lice = ldensity * self.lheight

        # WATER CONTENT [kg m-2]
        if params.initialize_water == 'dry':
            lwater = np.zeros(self.shape)
        elif params.initialize_water == 'saturated':
            porosity = 1 - ldensity / params.density_ice
            lwater = porosity * params.Sr * self.lheight * params.density_water
        else:
            raise ConfigError('Invalid configuration: initialize_water')
        
        # ===== AGE [days] =====
        firn_mask_int = self.firn_mask.astype(int)
        lage = np.cumsum(firn_mask_int, axis=1) * 365

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
        params = self.params

        # CONSTANTS
        BC_FRESH = params.BC_freshsnow
        OC_FRESH = params.OC_freshsnow
        DUST_FRESH = params.dust_freshsnow

        # INPUTS
        lheight = self.lheight
        ldepth = self.ldepth

        if params.initialize_LAPs in ['clean']:
            # snowpack is clean; initialize as constant values
            lBC = BC_FRESH * lheight
            lOC = OC_FRESH * lheight
            ldust = DUST_FRESH * lheight 
        elif params.initialize_LAPs in ['interpolate']:
            # read in LAP data
            lap_data = pd.read_csv(params.initial_LAP_fn,index_col=0)

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
def add_top_layer(state, mask, new_layer):
    """
    Adds a new layer to the top of layers
    for the points in mask.

    Parameters
    ==========
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

def add_bottom_layer(state, mask, params):
    """
    Fills in data for a new bottom layer added
    as a result of a different layer being removed.

    For points with a "viable" amount of ice (more than
    params.min_glacier_depth), all ice layers have their
    mass redistributed evenly across the existing ice layers.

    For points with ice mass that is too small to safely
    redistribute across many layers (numerical stability),
    all of the ice mass at that point is instead collapsed
    into a single ice layer (the topmost ice layer).
    Snow and firn layers are left untouched in this case --
    only the ice layers are affected, regardless of how small
    the total ice mass is.

    For points with no ice layers, a new ice bottom
    layer is pulled up from the basal reservoir.

    Parameters
    ==========
    mask : jnp.Array (N_POINTS)
        Boolean mask for points where a new layer is added
    """
    properties = state._asdict()
    N_POINTS = state.lice.shape[0]
    N_LAYERS = state.lice.shape[1]
    ice_mask = properties['ice_mask']

    # calculate amount of ice at each point
    ice_masses = jnp.where(ice_mask, properties['lice'], 0.0)
    point_ice_mass = jnp.sum(ice_masses, axis=1, keepdims=True)

    # determine where there is enough ice to redistribute
    has_viable_ice = (point_ice_mass > params.min_glacier_depth).squeeze()
    has_some_ice = (point_ice_mass > 0).squeeze()
    empty_reservoir = state.basal_reservoir < 1e-3

    redistribute = mask & has_viable_ice & empty_reservoir
    use_reservoir = mask & ~empty_reservoir
    collapse_to_single = mask & ~has_viable_ice & has_some_ice & empty_reservoir

    # case 1: redistribute mass across existing ice layers
    DZ_ICE = params.dz_icelayer
    DZ_TOP = params.dz_toplayer
    LAYER_GROWTH = params.layer_growth
    DENSITY_ICE = params.density_ice
    TEMP_TEMP = params.temp_temp
    GRAINSIZE_ICE = params.grainsize_ice

    layers_idx = jnp.arange(N_LAYERS)
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
        params.density_ice,
        properties['ldensity']
    )

    # case 2: collapse small ice mass into a single ice layer
    col_idx = jnp.arange(N_LAYERS)[None, :]
    ice_col_idx = jnp.where(ice_mask, col_idx, N_LAYERS)
    top_ice_layer_idx = jnp.min(ice_col_idx, axis=1)  # (N_POINTS,)
    top_ice_layer_idx = jnp.minimum(top_ice_layer_idx, N_LAYERS - 1)
    is_top_ice_layer = (col_idx == top_ice_layer_idx[:, None])

    # dump all the ice mass into the uppermost ice layer
    clear_other_ice_layers = collapse_to_single[:, None] & ice_mask & ~is_top_ice_layer
    set_top_ice_layer = collapse_to_single[:, None] & is_top_ice_layer

    properties['lice'] = jnp.where(
        clear_other_ice_layers, 0.0, properties['lice']
    )
    properties['lice'] = jnp.where(
        set_top_ice_layer, point_ice_mass, properties['lice']
    )

    # make sure that layer has ice density
    properties['ldensity'] = jnp.where(
        collapse_to_single[:, None] & ice_mask, DENSITY_ICE, properties['ldensity']
    )

    # merge all other mass terms into that top layer
    for merge_var in ('lwater', 'lrefreeze', 'ldrefreeze', 'lBC', 'lOC', 'ldust'):
        merge_data = properties[merge_var]
        merge_masses = jnp.where(
            collapse_to_single[:, None] & ice_mask, merge_data, 0.0
        )
        point_merge_total = jnp.sum(merge_masses, axis=1, keepdims=True)
        properties[merge_var] = jnp.where(
            clear_other_ice_layers, 0.0, properties[merge_var]
        )
        properties[merge_var] = jnp.where(
            set_top_ice_layer,
            jnp.broadcast_to(point_merge_total, properties[merge_var].shape),
            properties[merge_var]
        )

    safe_ldensity = jnp.where(properties['ldensity'] > 1e-3, properties['ldensity'], 1e-3)
    properties['lheight'] = jnp.where(
        (redistribute | collapse_to_single)[:, None] & ice_mask,
        properties['lice'] / safe_ldensity,
        properties['lheight']
    )

    # case 3: pull new ice layer from reservoir into bottom
    new_bottom_layer = {
        'ldepth': properties['ldepth'][:, -1],
        'lheight': jnp.full(N_POINTS, DZ_ICE),
        'ldensity': jnp.full(N_POINTS, DENSITY_ICE),
        'lice': jnp.full(N_POINTS, DENSITY_ICE * DZ_ICE),
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
    safe_bot_density = jnp.where(new_bottom_layer['ldensity'] > 0, new_bottom_layer['ldensity'], 1.0)
    new_bottom_layer['lheight'] = new_bottom_layer['lice'] / safe_bot_density

    for var in params.all_layer_vars:
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

    # save these properties to state
    state = state._replace(**properties)

    # update layers
    state = update_layer_props(state, DENSITY_ICE)
    return state

def remove_layer(state, mask, idx, params):
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
        layers_idx[None, :] >= idx[:, None]
    )

    for var in params.all_layer_vars:
        data = properties[var]

        # shift everything upwards by one
        fully_shifted = data.at[:, :-1].set(data[:, 1:])

        # make sure new bottom layer is filled with no mass
        if var == 'ltype':
            fully_shifted = fully_shifted.at[:, -1].set(2)
        elif var == 'ldensity':
            fully_shifted = fully_shifted.at[:, -1].set(params.density_ice)
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
    state = update_layer_props(state, params.density_ice)

    # add a new bottom layer
    state = add_bottom_layer(state, mask, params)        
    return state

def split_layer(state, mask, idx, params):
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

    for var in params.all_layer_vars:
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

    for var in params.extensive_vars:
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
    safe_ldensity = jnp.where(properties['ldensity'] > 1e-3, properties['ldensity'], 1e-3)
    properties['lheight'] = properties['lice'] / safe_ldensity

    # write these properties to state
    state = state._replace(**properties)

    # finally, send state through the profile updates utility
    state = update_layer_props(state, params.density_ice)
    
    return state

def merge_existing_layers(state, mask, idx, params):
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
    
    # floor at min_layer_mass
    m_safe = jnp.where(m_total > params.min_layer_mass, m_total, params.min_layer_mass)

    # combine point mask with index mask
    target_mask = mask[:, None] & (layers_idx == target_idx)[None, :]

    # take weighted average for intensive variables
    for var in params.intensive_vars:
        data = properties[var]
        removed_vals = data[:, idx]
        target_vals = data[:, target_idx]

        if var == 'ltype':
            # merged layer takes the type of whichever side has more mass
            merged_vals = jnp.where(m_removed > m_target, removed_vals, target_vals).astype(jnp.int32)
        else:
            # calculate mass-weighted average of values (N_POINTS)
            merged_vals = (target_vals * m_target + removed_vals * m_removed) / m_safe
            if var == 'lage':
                merged_vals = jnp.round(merged_vals).astype(jnp.int32)

        # replace it only at points / layers in mask
        properties[var] = jnp.where(
            target_mask,
            merged_vals[:, None],
            data
        )

    # sum for extensive variables
    for var in params.extensive_vars:
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

    # only do the shift when needed
    def _do_shift(props):
        props = dict(props)
        for var in params.all_layer_vars:
            data = props[var]

            # make sure new bottom layer is filled with no mass
            if var == 'ltype':
                fill_value = 2
            elif var == 'ldensity': # fill with real value to avoid div 0
                fill_value = params.density_ice
            else:
                fill_value = 0.0

            # shift everything upwards by one via slice + concatenate
            fill_col = jnp.full((data.shape[0], 1), fill_value, dtype=data.dtype)
            fully_shifted = jnp.concatenate([data[:, 1:], fill_col], axis=1)

            # replace it only at points / layers in mask
            props[var] = jnp.where(shift_mask, fully_shifted, data)
        return props

    properties = jax.lax.cond(jnp.any(mask), _do_shift, lambda p: p, properties)

    # update ice mask and layer height
    properties['ice_mask'] = properties['ltype'].astype(jnp.int32) == 2
    # floor at a small-but-nonzero density
    safe_ldensity = jnp.where(properties['ldensity'] > 1e-3, properties['ldensity'], 1e-3)
    properties['lheight'] = properties['lice'] / safe_ldensity

    # write these properties to state
    state = state._replace(**properties)

    # add bottom layer to replaced removed
    state = add_bottom_layer(state, mask, params)

    safe_ldensity = jnp.where(state.ldensity > 1e-3, state.ldensity, 1e-3)
    updated_lheight = state.lice / safe_ldensity
    state = state._replace(lheight=updated_lheight)

    # finally, send state through the profile updates utility
    state = update_layer_props(state, params.density_ice)

    return state


def merge_existing_layers_skipblock_probe(state, mask, idx, params, skip_block):
    """
    Debug-only: identical to (current, fixed) merge_existing_layers, except
    exactly one of its post-averaging blocks is replaced with a no-op:
      'shift': don't move any layers OR zero the removed one (post-averaging
        positions kept as-is -- NOTE this also breaks mass conservation,
        since the removed layer's mass is never zeroed, only relocated;
        see 'shift_zero_only' to separate that confound)
      'shift_zero_only': zero out the removed layer in place at idx (mass-
        conserving), but DON'T reindex/move later layers up to fill the gap
        -- isolates whether creating a genuinely near-empty layer matters,
        independent of the reindexing/data-movement itself
      'lheight_recompute_1': skip the lheight recompute right after the shift
      'add_bottom_layer': skip add_bottom_layer entirely
      'lheight_recompute_2': skip the lheight recompute right after add_bottom_layer
      'update_layer_props': skip the final update_layer_props call
      None: skip nothing (baseline, should match merge_existing_layers exactly)

    Unlike merge_existing_layers_probe (stop_after_phase truncates and
    returns early -- when called repeatedly across the layer scan, this
    compounds an artificial torn state every iteration, since later phases
    like add_bottom_layer/update_layer_props never run), this always
    completes every step except the one skip_block, so it's called
    repeatedly across the scan exactly like production. skip_block is a
    static Python string or None.

    Not used by any production path -- see jax_optimize.py's stage-bisection
    diagnostic.
    """
    properties = state._asdict()
    layers_idx = jnp.arange(state.lice.shape[1])
    target_idx = idx + 1

    m_removed = properties['lice'][:, idx]
    m_target = properties['lice'][:, target_idx]
    m_total = m_removed + m_target
    m_safe = jnp.where(m_total > params.min_layer_mass, m_total, params.min_layer_mass)

    target_mask = mask[:, None] & (layers_idx == target_idx)[None, :]

    for var in params.intensive_vars:
        data = properties[var]
        removed_vals = data[:, idx]
        target_vals = data[:, target_idx]
        weighted_avg = (target_vals * m_target + removed_vals * m_removed) / m_safe
        if var in ['ltype', 'lage']:
            weighted_avg = jnp.round(weighted_avg).astype(jnp.int32)
        properties[var] = jnp.where(target_mask, weighted_avg[:, None], data)

    for var in params.extensive_vars:
        data = properties[var]
        summed_data = data[:, idx] + data[:, target_idx]
        properties[var] = jnp.where(target_mask, summed_data[:, None], data)

    shift_mask = mask[:, None] & (layers_idx >= idx)[None, :]
    zero_mask = mask[:, None] & (layers_idx == idx)[None, :]

    def _do_shift(props):
        props = dict(props)
        for var in params.all_layer_vars:
            data = props[var]
            if var == 'ltype':
                fill_value = 2
            elif var == 'ldensity':
                fill_value = params.density_ice
            else:
                fill_value = 0.0
            fill_col = jnp.full((data.shape[0], 1), fill_value, dtype=data.dtype)
            fully_shifted = jnp.concatenate([data[:, 1:], fill_col], axis=1)
            props[var] = jnp.where(shift_mask, fully_shifted, data)
        return props

    def _do_zero_only(props):
        props = dict(props)
        for var in params.all_layer_vars:
            data = props[var]
            if var == 'ltype':
                fill_value = 2
            elif var == 'ldensity':
                fill_value = params.density_ice
            else:
                fill_value = 0.0
            filled = jnp.full_like(data, fill_value)
            props[var] = jnp.where(zero_mask, filled, data)
        return props

    if skip_block == 'shift_zero_only':
        properties = jax.lax.cond(jnp.any(mask), _do_zero_only, lambda p: p, properties)
    elif skip_block != 'shift':
        properties = jax.lax.cond(jnp.any(mask), _do_shift, lambda p: p, properties)

    properties['ice_mask'] = properties['ltype'].astype(jnp.int32) == 2
    if skip_block != 'lheight_recompute_1':
        safe_ldensity = jnp.where(properties['ldensity'] > 1e-3, properties['ldensity'], 1e-3)
        properties['lheight'] = properties['lice'] / safe_ldensity
    state = state._replace(**properties)

    if skip_block != 'add_bottom_layer':
        state = add_bottom_layer(state, mask, params)

    if skip_block != 'lheight_recompute_2':
        safe_ldensity = jnp.where(state.ldensity > 1e-3, state.ldensity, 1e-3)
        updated_lheight = state.lice / safe_ldensity
        state = state._replace(lheight=updated_lheight)

    if skip_block != 'update_layer_props':
        state = update_layer_props(state, params.density_ice)

    return state


def merge_existing_layers_probe(state, mask, idx, params, stop_after_phase):
    """
    Debug-only variant of merge_existing_layers for isolating which of its
    internal phases introduces a non-finite gradient:
      1: intensive-var weighted average + extensive-var sum
      2: + layer shift (reindexing), float-typed vars only
      3: + layer shift, int-typed vars too (ltype, lage -- isolates whether
         differentiating through their jnp.round().astype(int32) cast,
         combined with the shift's constant-fill + where-select, is
         specifically responsible)
      4: + ice_mask/lheight recompute #1, state reassign
      5: + add_bottom_layer
      6: + lheight recompute #2
      7: + update_layer_props (full -- matches merge_existing_layers exactly)
    stop_after_phase is a static Python int, 1-7.

    Not used by any production path -- see jax_optimize.py's stage-bisection
    diagnostic.
    """
    properties = state._asdict()
    layers_idx = jnp.arange(state.lice.shape[1])
    target_idx = idx + 1

    m_removed = properties['lice'][:, idx]
    m_target = properties['lice'][:, target_idx]
    m_total = m_removed + m_target
    m_safe = jnp.where(m_total > params.min_layer_mass, m_total, params.min_layer_mass)

    target_mask = mask[:, None] & (layers_idx == target_idx)[None, :]

    for var in params.intensive_vars:
        data = properties[var]
        removed_vals = data[:, idx]
        target_vals = data[:, target_idx]
        weighted_avg = (target_vals * m_target + removed_vals * m_removed) / m_safe
        if var in ['ltype', 'lage']:
            weighted_avg = jnp.round(weighted_avg).astype(jnp.int32)
        properties[var] = jnp.where(target_mask, weighted_avg[:, None], data)

    for var in params.extensive_vars:
        data = properties[var]
        summed_data = data[:, idx] + data[:, target_idx]
        properties[var] = jnp.where(target_mask, summed_data[:, None], data)

    if stop_after_phase == 1:
        return state._replace(**properties)

    def _shift_var(props, var):
        data = props[var]
        if var == 'ltype':
            fill_value = 2
        elif var == 'ldensity':
            fill_value = params.density_ice
        else:
            fill_value = 0.0
        fill_col = jnp.full((data.shape[0], 1), fill_value, dtype=data.dtype)
        fully_shifted = jnp.concatenate([data[:, 1:], fill_col], axis=1)
        return jnp.where(shift_mask, fully_shifted, data)

    shift_mask = mask[:, None] & (layers_idx >= idx)[None, :]

    def _do_float_shift(props):
        props = dict(props)
        for var in params.all_layer_vars:
            if var in ('ltype', 'lage'):
                continue
            props[var] = _shift_var(props, var)
        return props

    properties = jax.lax.cond(jnp.any(mask), _do_float_shift, lambda p: p, properties)

    if stop_after_phase == 2:
        return state._replace(**properties)

    for var in ('ltype', 'lage'):
        properties[var] = _shift_var(properties, var)

    if stop_after_phase == 3:
        return state._replace(**properties)

    properties['ice_mask'] = properties['ltype'].astype(jnp.int32) == 2
    safe_ldensity = jnp.where(properties['ldensity'] > 1e-3, properties['ldensity'], 1e-3)
    properties['lheight'] = properties['lice'] / safe_ldensity
    state = state._replace(**properties)
    if stop_after_phase == 4:
        return state

    state = add_bottom_layer(state, mask, params)
    if stop_after_phase == 5:
        return state

    safe_ldensity = jnp.where(state.ldensity > 1e-3, state.ldensity, 1e-3)
    updated_lheight = state.lice / safe_ldensity
    state = state._replace(lheight=updated_lheight)
    if stop_after_phase == 6:
        return state

    state = update_layer_props(state, params.density_ice)
    return state


# All 12 phase-1 variables (5 intensive + 7 extensive), in the order they're
# processed in merge_existing_layers_probe's phase-1 loops.
PHASE1_VAR_NAMES = [
    'ltemp', 'ldensity', 'lage', 'lgrainsize', 'ltype',       # intensive
    'lice', 'lwater', 'lBC', 'lOC', 'ldust', 'ldrefreeze', 'lrefreeze',  # extensive
]


def merge_existing_layers_phase1_skipvar_probe(state, mask, idx, params, skip_var):
    """
    Debug-only: runs ONLY phase 1 (weighted-average/extensive-sum -- no
    shift, no downstream at all, matching merge_existing_layers_probe's
    stop_after_phase=1), but skips writing back exactly one variable
    (leaves it at its pre-merge value for every point/layer, mask-selection
    included). Isolates which of the 12 phase-1 variables is responsible,
    since phase 1 as a whole was confirmed non-finite on a single, clean,
    isolated real merge event where nothing downstream even runs.

    skip_var is a static Python string (one of PHASE1_VAR_NAMES) or None
    (baseline, should match phase 1's confirmed non-finite result).

    Not used by any production path -- see jax_optimize.py's stage-bisection
    diagnostic.
    """
    properties = state._asdict()
    layers_idx = jnp.arange(state.lice.shape[1])
    target_idx = idx + 1

    m_removed = properties['lice'][:, idx]
    m_target = properties['lice'][:, target_idx]
    m_total = m_removed + m_target
    m_safe = jnp.where(m_total > params.min_layer_mass, m_total, params.min_layer_mass)

    target_mask = mask[:, None] & (layers_idx == target_idx)[None, :]

    for var in params.intensive_vars:
        if var == skip_var:
            continue
        data = properties[var]
        removed_vals = data[:, idx]
        target_vals = data[:, target_idx]
        weighted_avg = (target_vals * m_target + removed_vals * m_removed) / m_safe
        if var in ['ltype', 'lage']:
            weighted_avg = jnp.round(weighted_avg).astype(jnp.int32)
        properties[var] = jnp.where(target_mask, weighted_avg[:, None], data)

    for var in params.extensive_vars:
        if var == skip_var:
            continue
        data = properties[var]
        summed_data = data[:, idx] + data[:, target_idx]
        properties[var] = jnp.where(target_mask, summed_data[:, None], data)

    return state._replace(**properties)


def merge_existing_layers_var_probe(state, mask, idx, params, n_vars_shifted):
    """
    Debug-only: identical to merge_existing_layers_probe through phase 1
    (weighted-average/extensive-sum -- confirmed finite), then shifts only
    the first n_vars_shifted variables (in all_layer_vars order, excluding
    ltype/lage which were separately ruled out) instead of all of them.
    Isolates which specific variable's shift introduces the non-finite
    gradient, since shifting all of them (merge_existing_layers_probe phase 2)
    is confirmed non-finite while shifting none of them (phase 1) is finite.
    n_vars_shifted is a static Python int, 0 to len(float_vars).

    Not used by any production path -- see jax_optimize.py's stage-bisection
    diagnostic.
    """
    properties = state._asdict()
    layers_idx = jnp.arange(state.lice.shape[1])
    target_idx = idx + 1

    m_removed = properties['lice'][:, idx]
    m_target = properties['lice'][:, target_idx]
    m_total = m_removed + m_target
    m_safe = jnp.where(m_total > params.min_layer_mass, m_total, params.min_layer_mass)

    target_mask = mask[:, None] & (layers_idx == target_idx)[None, :]

    for var in params.intensive_vars:
        data = properties[var]
        removed_vals = data[:, idx]
        target_vals = data[:, target_idx]
        weighted_avg = (target_vals * m_target + removed_vals * m_removed) / m_safe
        if var in ['ltype', 'lage']:
            weighted_avg = jnp.round(weighted_avg).astype(jnp.int32)
        properties[var] = jnp.where(target_mask, weighted_avg[:, None], data)

    for var in params.extensive_vars:
        data = properties[var]
        summed_data = data[:, idx] + data[:, target_idx]
        properties[var] = jnp.where(target_mask, summed_data[:, None], data)

    shift_mask = mask[:, None] & (layers_idx >= idx)[None, :]
    float_vars = [v for v in params.all_layer_vars if v not in ('ltype', 'lage')]

    for var in float_vars[:n_vars_shifted]:
        data = properties[var]
        fully_shifted = data.at[:, :-1].set(data[:, 1:])
        if var == 'ldensity':
            fully_shifted = fully_shifted.at[:, -1].set(params.density_ice)
        else:
            fully_shifted = fully_shifted.at[:, -1].set(0.0)
        properties[var] = jnp.where(shift_mask, fully_shifted, data)

    return state._replace(**properties)


def merge_existing_layers_skip_var_probe(state, mask, idx, params, skip_var):
    """
    Debug-only: full and faithful merge_existing_layers (every phase, exactly
    matching production), except the shift step leaves `skip_var` un-shifted
    (a no-op for that one field) while every other variable and downstream
    step (lheight recompute, add_bottom_layer, update_layer_props) runs
    normally. Unlike merge_existing_layers_var_probe (which truncates a
    PREFIX of variables and returns early -- compounding torn-state
    inconsistency across repeated scan iterations, since every not-yet-
    shifted field stays stale at every merge site in the scan), this keeps
    every other field fully consistent at every iteration. Isolates whether
    skip_var's shift specifically is necessary for the non-finite gradient,
    without introducing an artifact of its own.

    skip_var is a static Python string (one of all_layer_vars), or None to
    shift everything (sanity-check baseline -- should behave identically to
    merge_existing_layers).

    Not used by any production path -- see jax_optimize.py's stage-bisection
    diagnostic.
    """
    properties = state._asdict()
    layers_idx = jnp.arange(state.lice.shape[1])
    target_idx = idx + 1

    m_removed = properties['lice'][:, idx]
    m_target = properties['lice'][:, target_idx]
    m_total = m_removed + m_target
    m_safe = jnp.where(m_total > params.min_layer_mass, m_total, params.min_layer_mass)

    target_mask = mask[:, None] & (layers_idx == target_idx)[None, :]

    for var in params.intensive_vars:
        data = properties[var]
        removed_vals = data[:, idx]
        target_vals = data[:, target_idx]
        weighted_avg = (target_vals * m_target + removed_vals * m_removed) / m_safe
        if var in ['ltype', 'lage']:
            weighted_avg = jnp.round(weighted_avg).astype(jnp.int32)
        properties[var] = jnp.where(target_mask, weighted_avg[:, None], data)

    for var in params.extensive_vars:
        data = properties[var]
        summed_data = data[:, idx] + data[:, target_idx]
        properties[var] = jnp.where(target_mask, summed_data[:, None], data)

    shift_mask = mask[:, None] & (layers_idx >= idx)[None, :]
    for var in params.all_layer_vars:
        if var == skip_var:
            continue
        data = properties[var]
        fully_shifted = data.at[:, :-1].set(data[:, 1:])
        if var == 'ltype':
            fully_shifted = fully_shifted.at[:, -1].set(2)
        elif var == 'ldensity':
            fully_shifted = fully_shifted.at[:, -1].set(params.density_ice)
        else:
            fully_shifted = fully_shifted.at[:, -1].set(0.0)
        properties[var] = jnp.where(shift_mask, fully_shifted, data)

    properties['ice_mask'] = properties['ltype'].astype(jnp.int32) == 2
    safe_ldensity = jnp.where(properties['ldensity'] > 1e-3, properties['ldensity'], 1e-3)
    properties['lheight'] = properties['lice'] / safe_ldensity
    state = state._replace(**properties)

    state = add_bottom_layer(state, mask, params)

    safe_ldensity = jnp.where(state.ldensity > 1e-3, state.ldensity, 1e-3)
    updated_lheight = state.lice / safe_ldensity
    state = state._replace(lheight=updated_lheight)

    state = update_layer_props(state, params.density_ice)
    return state


def merge_existing_layers_nvars_probe(state, mask, idx, params, n_vars_shifted):
    """
    Debug-only: like merge_existing_layers_skip_var_probe (full production
    fidelity -- every downstream step runs, no early return), but shifts
    only the FIRST n_vars_shifted variables (in float_vars order, i.e.
    all_layer_vars excluding ltype/lage) instead of skipping just one.
    Finds the minimum number of *simultaneously* shifted variables needed
    to trigger the non-finite gradient, given that omitting any single one
    of 12 (merge_existing_layers_skip_var_probe) was insufficient to fix it.

    Unlike merge_existing_layers_var_probe (which returns immediately after
    the partial shift, compounding torn-state inconsistency across repeated
    scan iterations), this keeps every other step fully consistent --
    avoiding that probe's forward-pass artifact.

    n_vars_shifted is a static Python int, 0 to len(float_vars) (12).

    Not used by any production path -- see jax_optimize.py's stage-bisection
    diagnostic.
    """
    properties = state._asdict()
    layers_idx = jnp.arange(state.lice.shape[1])
    target_idx = idx + 1

    m_removed = properties['lice'][:, idx]
    m_target = properties['lice'][:, target_idx]
    m_total = m_removed + m_target
    m_safe = jnp.where(m_total > params.min_layer_mass, m_total, params.min_layer_mass)

    target_mask = mask[:, None] & (layers_idx == target_idx)[None, :]

    for var in params.intensive_vars:
        data = properties[var]
        removed_vals = data[:, idx]
        target_vals = data[:, target_idx]
        weighted_avg = (target_vals * m_target + removed_vals * m_removed) / m_safe
        if var in ['ltype', 'lage']:
            weighted_avg = jnp.round(weighted_avg).astype(jnp.int32)
        properties[var] = jnp.where(target_mask, weighted_avg[:, None], data)

    for var in params.extensive_vars:
        data = properties[var]
        summed_data = data[:, idx] + data[:, target_idx]
        properties[var] = jnp.where(target_mask, summed_data[:, None], data)

    shift_mask = mask[:, None] & (layers_idx >= idx)[None, :]
    float_vars = [v for v in params.all_layer_vars if v not in ('ltype', 'lage')]
    shift_now = set(float_vars[:n_vars_shifted])

    for var in float_vars:
        if var not in shift_now:
            continue
        data = properties[var]
        fully_shifted = data.at[:, :-1].set(data[:, 1:])
        if var == 'ldensity':
            fully_shifted = fully_shifted.at[:, -1].set(params.density_ice)
        else:
            fully_shifted = fully_shifted.at[:, -1].set(0.0)
        properties[var] = jnp.where(shift_mask, fully_shifted, data)

    properties['ice_mask'] = properties['ltype'].astype(jnp.int32) == 2
    safe_ldensity = jnp.where(properties['ldensity'] > 1e-3, properties['ldensity'], 1e-3)
    properties['lheight'] = properties['lice'] / safe_ldensity
    state = state._replace(**properties)

    state = add_bottom_layer(state, mask, params)

    safe_ldensity = jnp.where(state.ldensity > 1e-3, state.ldensity, 1e-3)
    updated_lheight = state.lice / safe_ldensity
    state = state._replace(lheight=updated_lheight)

    state = update_layer_props(state, params.density_ice)
    return state


def merge_new_layer(state, mask, new_layer, params):
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
    # see merge_existing_layers -- floor at min_layer_mass, not just >0
    m_safe = jnp.where(m_total > params.min_layer_mass, m_total, params.min_layer_mass)

    # combine point mask with index mask
    target_mask = mask[:, None] & (layers_idx == 0)[None, :]

    # take weighted average for intensive variables
    for var in params.intensive_vars:
        data = properties[var]
        new_vals = new_layer[var]
        target_vals = data[:, 0]

        if var == 'ltype':
            # merged layer takes the type of whichever side has more mass
            merged_vals = jnp.where(m_new > m_target, new_vals, target_vals).astype(jnp.int32)
        else:
            # calculate mass-weighted average of values (N_POINTS)
            merged_vals = (target_vals * m_target + new_vals * m_new) / m_safe
            if var == 'lage':
                merged_vals = jnp.round(merged_vals).astype(jnp.int32)

        # replace it only at points / layers in mask
        properties[var] = jnp.where(
            target_mask,
            merged_vals[:, None],
            data
        )

    # sum for extensive variables
    for var in params.extensive_vars:
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

    safe_ldensity = jnp.where(state.ldensity > 1e-3, state.ldensity, 1e-3)
    updated_lheight = state.lice / safe_ldensity
    state = state._replace(lheight=updated_lheight)

    # finally, send state through the profile updates utility
    state = update_layer_props(state, params.density_ice)

    return state

def check_layer_sizes(state, params):
    """
    Scans through layers sequentially from top to bottom.
    If a layer is below the minimum height threshold, it is 
    merged with the layer directly beneath it.
    """
    properties = state._asdict()
    n_points, n_layers = properties['lice'].shape

    # zero out dead layers
    dead_mask = properties['lice'] < params.min_layer_mass

    dead_mass = jnp.sum(jnp.where(
        dead_mask, properties['lice'] + properties['lwater'], 0.0
    ), axis=1)
    
    properties['lice'] = jnp.where(dead_mask, 0.0, properties['lice'])
    properties['lwater'] = jnp.where(dead_mask, 0.0, properties['lwater'])
    properties['lheight'] = jnp.where(dead_mask, 0.0, properties['lheight'])
    properties['lBC'] = jnp.where(dead_mask, 0.0, properties['lBC'])
    properties['lOC'] = jnp.where(dead_mask, 0.0, properties['lOC'])
    properties['ldust'] = jnp.where(dead_mask, 0.0, properties['ldust'])
    state = state._replace(**properties)

    layer_indices = jnp.arange(n_layers)
    curve_snow = params.dz_toplayer * jnp.exp(layer_indices * params.layer_growth)
    min_height_by_depth = jnp.maximum(curve_snow, params.min_dz)

    # ice has stricter height requirement for numerical stability
    dt_heat = params.dt / params.n_heat_steps
    ice_stability_min = 2.0 * jnp.sqrt(
        4 * params.k_ice * dt_heat / (params.Cp_ice * params.density_ice)
    )
    min_height_ice = jnp.maximum(ice_stability_min, params.min_dz)

    # define function to scan for layers to merge
    def _scan_merge(carry, idx):
        current_state, already_merged = carry

        # always fetch the most up-to-date heights and types from evolving state
        dz = current_state.lheight[:, idx]
        curr_type = current_state.ltype[:, idx]
        next_type = current_state.ltype[:, idx + 1]

        # determine if layer is too thin for its position
        is_thin_snow = (curr_type == 0) & (dz < min_height_by_depth[idx])
        is_thin_any = (dz < min_height_ice)

        # determine which spatial columns need a merge at this specific vertical index
        is_snow = curr_type == 0
        type_matches_below = curr_type == next_type
        force_small_snow = (curr_type == 0) & (next_type > 0) & (dz < params.min_dz)

        # build the boolean merge mask (N_POINTS)
        # firn / ice layers only merge if they are below min_dz
        any_merge = is_thin_any & ~is_snow
        # snow layers merge if there is snow beneath; or if they are very small
        snow_merge = is_thin_snow & (type_matches_below | force_small_snow)

        # cap at one merge per point per call
        merge_mask = (any_merge | snow_merge) & ~already_merged

        # merge layers, if there are layers to merge
        next_state = merge_existing_layers(current_state, merge_mask, idx, params)
        next_already_merged = already_merged | merge_mask

        return (next_state, next_already_merged), None

    layers_idx = jnp.arange(n_layers - 1) # don't include bottom layer
    init_carry = (state, jnp.zeros(n_points, dtype=bool))
    (state, _), _ = jax.lax.scan(_scan_merge, init_carry, layers_idx)

    # split snow layers that have grown beyond twice their target size for their position
    def _scan_splits(current_state, idx):
        dz = current_state.lheight[:, idx]
        curr_type = current_state.ltype[:, idx]
        should_split = (curr_type == 0) & (dz > curve_snow[idx] * 2)
        return split_layer(current_state, should_split, idx, params), None

    state, _ = jax.lax.scan(_scan_splits, state, layers_idx)

    return state, dead_mass


def check_layer_sizes_probe(state, params, stop_after_phase, disable_any_merge=False, restrict_to_site=None):
    """
    Debug-only variant of check_layer_sizes for isolating which of its 3
    phases (1: dead-layer zeroing, 2: merge scan / merge_existing_layers,
    3: split scan / split_layer) introduces a non-finite gradient.
    stop_after_phase is a static Python int, 1-3. Returns state only.

    disable_any_merge (static bool): if True, forces any_merge (the
    is_thin_any & ~is_snow path -- firn/ice layers merging because they're
    below ice_stability_min, the numerical-stability height floor)
    permanently off, so only snow_merge (the force_small_snow / min_dz path
    this whole investigation started from) can ever trigger. Disabling this
    alone did NOT fix a real non-finite case, since snow_merge (via
    type_matches_below, not force_small_snow) was independently active at
    other sites in the same timestep -- see restrict_to_site to isolate a
    single merge event at a time instead.

    restrict_to_site (static int or None): if set, forces merge_mask to
    False for every point except this one -- isolates whether a single
    site's merge event, by itself, is sufficient to trigger a non-finite
    gradient, or whether it takes multiple sites merging simultaneously.

    Not used by any production path -- see jax_optimize.py's stage-bisection
    diagnostic.
    """
    properties = state._asdict()
    n_points, n_layers = properties['lice'].shape

    dead_mask = properties['lice'] < params.min_layer_mass
    properties['lice'] = jnp.where(dead_mask, 0.0, properties['lice'])
    properties['lwater'] = jnp.where(dead_mask, 0.0, properties['lwater'])
    properties['lheight'] = jnp.where(dead_mask, 0.0, properties['lheight'])
    properties['lBC'] = jnp.where(dead_mask, 0.0, properties['lBC'])
    properties['lOC'] = jnp.where(dead_mask, 0.0, properties['lOC'])
    properties['ldust'] = jnp.where(dead_mask, 0.0, properties['ldust'])
    state = state._replace(**properties)
    if stop_after_phase == 1:
        return state

    layer_indices = jnp.arange(n_layers)
    curve_snow = params.dz_toplayer * jnp.exp(layer_indices * params.layer_growth)
    min_height_by_depth = jnp.maximum(curve_snow, params.min_dz)

    dt_heat = params.dt / params.n_heat_steps
    ice_stability_min = 2.0 * jnp.sqrt(
        4 * params.k_ice * dt_heat / (params.Cp_ice * params.density_ice)
    )
    min_height_ice = jnp.maximum(ice_stability_min, params.min_dz)

    def _scan_merge(carry, idx):
        current_state, already_merged = carry
        dz = current_state.lheight[:, idx]
        curr_type = current_state.ltype[:, idx]
        next_type = current_state.ltype[:, idx + 1]

        is_thin_snow = (curr_type == 0) & (dz < min_height_by_depth[idx])
        is_thin_any = (dz < min_height_ice)

        is_snow = curr_type == 0
        type_matches_below = curr_type == next_type
        force_small_snow = (curr_type == 0) & (next_type > 0) & (dz < params.min_dz)

        any_merge = (is_thin_any & ~is_snow) & (not disable_any_merge)
        snow_merge = is_thin_snow & (type_matches_below | force_small_snow)
        merge_mask = (any_merge | snow_merge) & ~already_merged
        if restrict_to_site is not None:
            site_mask = jnp.arange(current_state.lice.shape[0]) == restrict_to_site
            merge_mask = merge_mask & site_mask

        next_state = merge_existing_layers(current_state, merge_mask, idx, params)
        next_already_merged = already_merged | merge_mask
        return (next_state, next_already_merged), None

    layers_idx = jnp.arange(n_layers - 1)
    init_carry = (state, jnp.zeros(n_points, dtype=bool))
    (state, _), _ = jax.lax.scan(_scan_merge, init_carry, layers_idx)
    if stop_after_phase == 2:
        return state

    def _scan_splits(current_state, idx):
        dz = current_state.lheight[:, idx]
        curr_type = current_state.ltype[:, idx]
        should_split = (curr_type == 0) & (dz > curve_snow[idx] * 2)
        return split_layer(current_state, should_split, idx, params), None

    state, _ = jax.lax.scan(_scan_splits, state, layers_idx)
    return state


def check_layer_sizes_no_merge_probe(state, params):
    """
    Debug-only variant of check_layer_sizes with the merge scan removed
    entirely (merge_existing_layers is never called -- not truncated, not
    masked to all-False, just absent from the traced graph). Dead-layer
    zeroing and the split scan run exactly as production.

    Every earlier bisection level (STAGE_BISECT down through
    MERGE_HOURS_BISECT/STATE_DUMP_HOUR in jax_optimize.py) narrowed the
    non-finite gradient down to merge_existing_layers via nested probes that
    each truncate or isolate a piece of the computation -- useful for
    localization, but every one of them changes the trajectory relative to
    a real run, and the chain never landed on a single conclusive culprit
    variable/block. This is the sanity check one level up: does a REAL,
    otherwise-untouched forward/backward simulation actually go non-finite
    only when merges happen, and go finite when this one probe removes
    merging (and nothing else)? See jax_optimize.py's single-site real test.

    Not used by any production path.
    """
    properties = state._asdict()

    dead_mask = properties['lice'] < params.min_layer_mass
    properties['lice'] = jnp.where(dead_mask, 0.0, properties['lice'])
    properties['lwater'] = jnp.where(dead_mask, 0.0, properties['lwater'])
    properties['lheight'] = jnp.where(dead_mask, 0.0, properties['lheight'])
    properties['lBC'] = jnp.where(dead_mask, 0.0, properties['lBC'])
    properties['lOC'] = jnp.where(dead_mask, 0.0, properties['lOC'])
    properties['ldust'] = jnp.where(dead_mask, 0.0, properties['ldust'])
    state = state._replace(**properties)

    n_layers = properties['lice'].shape[1]
    layer_indices = jnp.arange(n_layers)
    curve_snow = params.dz_toplayer * jnp.exp(layer_indices * params.layer_growth)

    def _scan_splits(current_state, idx):
        dz = current_state.lheight[:, idx]
        curr_type = current_state.ltype[:, idx]
        should_split = (curr_type == 0) & (dz > curve_snow[idx] * 2)
        return split_layer(current_state, should_split, idx, params), None

    layers_idx = jnp.arange(n_layers - 1)
    state, _ = jax.lax.scan(_scan_splits, state, layers_idx)
    return state


def check_layer_sizes_merge_internal_probe(state, params, stop_after_merge_phase, restrict_to_site=None):
    """
    Debug-only: runs check_layer_sizes' dead-layer zeroing in full, then the
    merge scan using merge_existing_layers_probe (same stop_after_phase for
    every merge in the scan) instead of merge_existing_layers, then returns
    -- no split scan, since this isolates merge_existing_layers' own
    internal structure (see merge_existing_layers_probe for phase meanings).

    restrict_to_site (static int or None): forces merge_mask False
    everywhere except this one site, and caps at one merge per point per
    call -- needed since a truncated (stop_after_merge_phase < 7) merge is
    called repeatedly across the scan, so without the cap a single
    restricted site that qualifies at multiple layer indices (common in
    practice) would still cascade through several truncated, torn-state
    merges instead of exactly one clean isolated one.

    Not used by any production path -- see jax_optimize.py's stage-bisection
    diagnostic.
    """
    properties = state._asdict()
    n_points, n_layers = properties['lice'].shape

    dead_mask = properties['lice'] < params.min_layer_mass
    properties['lice'] = jnp.where(dead_mask, 0.0, properties['lice'])
    properties['lwater'] = jnp.where(dead_mask, 0.0, properties['lwater'])
    properties['lheight'] = jnp.where(dead_mask, 0.0, properties['lheight'])
    properties['lBC'] = jnp.where(dead_mask, 0.0, properties['lBC'])
    properties['lOC'] = jnp.where(dead_mask, 0.0, properties['lOC'])
    properties['ldust'] = jnp.where(dead_mask, 0.0, properties['ldust'])
    state = state._replace(**properties)

    layer_indices = jnp.arange(n_layers)
    curve_snow = params.dz_toplayer * jnp.exp(layer_indices * params.layer_growth)
    min_height_by_depth = jnp.maximum(curve_snow, params.min_dz)

    dt_heat = params.dt / params.n_heat_steps
    ice_stability_min = 2.0 * jnp.sqrt(
        4 * params.k_ice * dt_heat / (params.Cp_ice * params.density_ice)
    )
    min_height_ice = jnp.maximum(ice_stability_min, params.min_dz)

    def _scan_merge(carry, idx):
        current_state, already_merged = carry
        dz = current_state.lheight[:, idx]
        curr_type = current_state.ltype[:, idx]
        next_type = current_state.ltype[:, idx + 1]

        is_thin_snow = (curr_type == 0) & (dz < min_height_by_depth[idx])
        is_thin_any = (dz < min_height_ice)

        is_snow = curr_type == 0
        type_matches_below = curr_type == next_type
        force_small_snow = (curr_type == 0) & (next_type > 0) & (dz < params.min_dz)

        any_merge = is_thin_any & ~is_snow
        snow_merge = is_thin_snow & (type_matches_below | force_small_snow)
        merge_mask = (any_merge | snow_merge) & ~already_merged
        if restrict_to_site is not None:
            site_mask = jnp.arange(current_state.lice.shape[0]) == restrict_to_site
            merge_mask = merge_mask & site_mask

        next_state = merge_existing_layers_probe(
            current_state, merge_mask, idx, params, stop_after_merge_phase
        )
        next_already_merged = already_merged | merge_mask
        return (next_state, next_already_merged), None

    layers_idx = jnp.arange(n_layers - 1)
    init_carry = (state, jnp.zeros(n_points, dtype=bool))
    (state, _), _ = jax.lax.scan(_scan_merge, init_carry, layers_idx)
    return state


def check_layer_sizes_merge_phase1_skipvar_probe(state, params, skip_var, restrict_to_site=None):
    """
    Debug-only: same idea as check_layer_sizes_merge_internal_probe, but
    calls merge_existing_layers_phase1_skipvar_probe (phase 1 only -- no
    shift, no downstream at all) instead -- isolates which of the 12
    phase-1 variables (PHASE1_VAR_NAMES) is responsible, since phase 1 as a
    whole was confirmed non-finite on a single, clean, isolated real merge.

    Not used by any production path -- see jax_optimize.py's stage-bisection
    diagnostic.
    """
    properties = state._asdict()
    n_points, n_layers = properties['lice'].shape

    dead_mask = properties['lice'] < params.min_layer_mass
    properties['lice'] = jnp.where(dead_mask, 0.0, properties['lice'])
    properties['lwater'] = jnp.where(dead_mask, 0.0, properties['lwater'])
    properties['lheight'] = jnp.where(dead_mask, 0.0, properties['lheight'])
    properties['lBC'] = jnp.where(dead_mask, 0.0, properties['lBC'])
    properties['lOC'] = jnp.where(dead_mask, 0.0, properties['lOC'])
    properties['ldust'] = jnp.where(dead_mask, 0.0, properties['ldust'])
    state = state._replace(**properties)

    layer_indices = jnp.arange(n_layers)
    curve_snow = params.dz_toplayer * jnp.exp(layer_indices * params.layer_growth)
    min_height_by_depth = jnp.maximum(curve_snow, params.min_dz)

    dt_heat = params.dt / params.n_heat_steps
    ice_stability_min = 2.0 * jnp.sqrt(
        4 * params.k_ice * dt_heat / (params.Cp_ice * params.density_ice)
    )
    min_height_ice = jnp.maximum(ice_stability_min, params.min_dz)

    def _scan_merge(carry, idx):
        current_state, already_merged = carry
        dz = current_state.lheight[:, idx]
        curr_type = current_state.ltype[:, idx]
        next_type = current_state.ltype[:, idx + 1]

        is_thin_snow = (curr_type == 0) & (dz < min_height_by_depth[idx])
        is_thin_any = (dz < min_height_ice)

        is_snow = curr_type == 0
        type_matches_below = curr_type == next_type
        force_small_snow = (curr_type == 0) & (next_type > 0) & (dz < params.min_dz)

        any_merge = is_thin_any & ~is_snow
        snow_merge = is_thin_snow & (type_matches_below | force_small_snow)
        merge_mask = (any_merge | snow_merge) & ~already_merged
        if restrict_to_site is not None:
            site_mask = jnp.arange(current_state.lice.shape[0]) == restrict_to_site
            merge_mask = merge_mask & site_mask

        next_state = merge_existing_layers_phase1_skipvar_probe(
            current_state, merge_mask, idx, params, skip_var
        )
        next_already_merged = already_merged | merge_mask
        return (next_state, next_already_merged), None

    layers_idx = jnp.arange(n_layers - 1)
    init_carry = (state, jnp.zeros(n_points, dtype=bool))
    (state, _), _ = jax.lax.scan(_scan_merge, init_carry, layers_idx)
    return state


def check_layer_sizes_merge_skipblock_probe(state, params, skip_block, restrict_to_site=None):
    """
    Debug-only: same as check_layer_sizes_merge_internal_probe, but calls
    merge_existing_layers_skipblock_probe (same skip_block for every merge
    in the scan) instead of merge_existing_layers_probe -- isolates which
    post-averaging block (shift, either lheight recompute, add_bottom_layer,
    update_layer_props) is responsible, WITHOUT the compounding-torn-state
    artifact of check_layer_sizes_merge_internal_probe (every merge always
    completes fully here, matching production, except the one skipped block).

    restrict_to_site (static int or None): forces merge_mask False
    everywhere except this one site, for testing a single isolated merge
    event -- also caps at one merge per point per call (like
    check_layer_sizes_probe), since a single restricted site can still
    qualify at multiple layer indices and would otherwise cascade.

    Not used by any production path -- see jax_optimize.py's stage-bisection
    diagnostic.
    """
    properties = state._asdict()
    n_points, n_layers = properties['lice'].shape

    dead_mask = properties['lice'] < params.min_layer_mass
    properties['lice'] = jnp.where(dead_mask, 0.0, properties['lice'])
    properties['lwater'] = jnp.where(dead_mask, 0.0, properties['lwater'])
    properties['lheight'] = jnp.where(dead_mask, 0.0, properties['lheight'])
    properties['lBC'] = jnp.where(dead_mask, 0.0, properties['lBC'])
    properties['lOC'] = jnp.where(dead_mask, 0.0, properties['lOC'])
    properties['ldust'] = jnp.where(dead_mask, 0.0, properties['ldust'])
    state = state._replace(**properties)

    layer_indices = jnp.arange(n_layers)
    curve_snow = params.dz_toplayer * jnp.exp(layer_indices * params.layer_growth)
    min_height_by_depth = jnp.maximum(curve_snow, params.min_dz)

    dt_heat = params.dt / params.n_heat_steps
    ice_stability_min = 2.0 * jnp.sqrt(
        4 * params.k_ice * dt_heat / (params.Cp_ice * params.density_ice)
    )
    min_height_ice = jnp.maximum(ice_stability_min, params.min_dz)

    def _scan_merge(carry, idx):
        current_state, already_merged = carry
        dz = current_state.lheight[:, idx]
        curr_type = current_state.ltype[:, idx]
        next_type = current_state.ltype[:, idx + 1]

        is_thin_snow = (curr_type == 0) & (dz < min_height_by_depth[idx])
        is_thin_any = (dz < min_height_ice)

        is_snow = curr_type == 0
        type_matches_below = curr_type == next_type
        force_small_snow = (curr_type == 0) & (next_type > 0) & (dz < params.min_dz)

        any_merge = is_thin_any & ~is_snow
        snow_merge = is_thin_snow & (type_matches_below | force_small_snow)
        merge_mask = (any_merge | snow_merge) & ~already_merged
        if restrict_to_site is not None:
            site_mask = jnp.arange(current_state.lice.shape[0]) == restrict_to_site
            merge_mask = merge_mask & site_mask

        next_state = merge_existing_layers_skipblock_probe(
            current_state, merge_mask, idx, params, skip_block
        )
        next_already_merged = already_merged | merge_mask
        return (next_state, next_already_merged), None

    layers_idx = jnp.arange(n_layers - 1)
    init_carry = (state, jnp.zeros(n_points, dtype=bool))
    (state, _), _ = jax.lax.scan(_scan_merge, init_carry, layers_idx)
    return state


def check_layer_sizes_merge_var_probe(state, params, n_vars_shifted):
    """
    Debug-only: same as check_layer_sizes_merge_internal_probe, but calls
    merge_existing_layers_var_probe (same n_vars_shifted for every merge in
    the scan) instead of merge_existing_layers_probe -- isolates which
    specific shifted variable introduces a non-finite gradient.

    Not used by any production path -- see jax_optimize.py's stage-bisection
    diagnostic.
    """
    properties = state._asdict()
    n_points, n_layers = properties['lice'].shape

    dead_mask = properties['lice'] < params.min_layer_mass
    properties['lice'] = jnp.where(dead_mask, 0.0, properties['lice'])
    properties['lwater'] = jnp.where(dead_mask, 0.0, properties['lwater'])
    properties['lheight'] = jnp.where(dead_mask, 0.0, properties['lheight'])
    properties['lBC'] = jnp.where(dead_mask, 0.0, properties['lBC'])
    properties['lOC'] = jnp.where(dead_mask, 0.0, properties['lOC'])
    properties['ldust'] = jnp.where(dead_mask, 0.0, properties['ldust'])
    state = state._replace(**properties)

    layer_indices = jnp.arange(n_layers)
    curve_snow = params.dz_toplayer * jnp.exp(layer_indices * params.layer_growth)
    min_height_by_depth = jnp.maximum(curve_snow, params.min_dz)

    dt_heat = params.dt / params.n_heat_steps
    ice_stability_min = 2.0 * jnp.sqrt(
        4 * params.k_ice * dt_heat / (params.Cp_ice * params.density_ice)
    )
    min_height_ice = jnp.maximum(ice_stability_min, params.min_dz)

    def _scan_merge(current_state, idx):
        dz = current_state.lheight[:, idx]
        curr_type = current_state.ltype[:, idx]
        next_type = current_state.ltype[:, idx + 1]

        is_thin_snow = (curr_type == 0) & (dz < min_height_by_depth[idx])
        is_thin_any = (dz < min_height_ice)

        is_snow = curr_type == 0
        type_matches_below = curr_type == next_type
        force_small_snow = (curr_type == 0) & (next_type > 0) & (dz < params.min_dz)

        any_merge = is_thin_any & ~is_snow
        snow_merge = is_thin_snow & (type_matches_below | force_small_snow)
        merge_mask = any_merge | snow_merge

        next_state = merge_existing_layers_var_probe(
            current_state, merge_mask, idx, params, n_vars_shifted
        )
        return next_state, None

    layers_idx = jnp.arange(n_layers - 1)
    state, _ = jax.lax.scan(_scan_merge, state, layers_idx)
    return state


def check_layer_sizes_merge_skipvar_probe(state, params, skip_var):
    """
    Debug-only: same as check_layer_sizes_merge_internal_probe, but calls
    merge_existing_layers_skip_var_probe (same skip_var for every merge in
    the scan) instead of merge_existing_layers_probe -- isolates whether a
    single variable's shift is necessary for the non-finite gradient,
    without the compounding-inconsistency artifact of
    check_layer_sizes_merge_var_probe (see merge_existing_layers_skip_var_probe).

    Not used by any production path -- see jax_optimize.py's stage-bisection
    diagnostic.
    """
    properties = state._asdict()
    n_points, n_layers = properties['lice'].shape

    dead_mask = properties['lice'] < params.min_layer_mass
    properties['lice'] = jnp.where(dead_mask, 0.0, properties['lice'])
    properties['lwater'] = jnp.where(dead_mask, 0.0, properties['lwater'])
    properties['lheight'] = jnp.where(dead_mask, 0.0, properties['lheight'])
    properties['lBC'] = jnp.where(dead_mask, 0.0, properties['lBC'])
    properties['lOC'] = jnp.where(dead_mask, 0.0, properties['lOC'])
    properties['ldust'] = jnp.where(dead_mask, 0.0, properties['ldust'])
    state = state._replace(**properties)

    layer_indices = jnp.arange(n_layers)
    curve_snow = params.dz_toplayer * jnp.exp(layer_indices * params.layer_growth)
    min_height_by_depth = jnp.maximum(curve_snow, params.min_dz)

    dt_heat = params.dt / params.n_heat_steps
    ice_stability_min = 2.0 * jnp.sqrt(
        4 * params.k_ice * dt_heat / (params.Cp_ice * params.density_ice)
    )
    min_height_ice = jnp.maximum(ice_stability_min, params.min_dz)

    def _scan_merge(current_state, idx):
        dz = current_state.lheight[:, idx]
        curr_type = current_state.ltype[:, idx]
        next_type = current_state.ltype[:, idx + 1]

        is_thin_snow = (curr_type == 0) & (dz < min_height_by_depth[idx])
        is_thin_any = (dz < min_height_ice)

        is_snow = curr_type == 0
        type_matches_below = curr_type == next_type
        force_small_snow = (curr_type == 0) & (next_type > 0) & (dz < params.min_dz)

        any_merge = is_thin_any & ~is_snow
        snow_merge = is_thin_snow & (type_matches_below | force_small_snow)
        merge_mask = any_merge | snow_merge

        next_state = merge_existing_layers_skip_var_probe(
            current_state, merge_mask, idx, params, skip_var
        )
        return next_state, None

    layers_idx = jnp.arange(n_layers - 1)
    state, _ = jax.lax.scan(_scan_merge, state, layers_idx)
    return state


def check_layer_sizes_merge_nvars_probe(state, params, n_vars_shifted):
    """
    Debug-only: same as check_layer_sizes_merge_internal_probe, but calls
    merge_existing_layers_nvars_probe (same n_vars_shifted for every merge
    in the scan) instead of merge_existing_layers_probe -- finds the
    minimum number of simultaneously shifted variables needed to trigger
    the non-finite gradient, with full downstream fidelity at every
    iteration (no compounding-inconsistency artifact).

    Not used by any production path -- see jax_optimize.py's stage-bisection
    diagnostic.
    """
    properties = state._asdict()
    n_points, n_layers = properties['lice'].shape

    dead_mask = properties['lice'] < params.min_layer_mass
    properties['lice'] = jnp.where(dead_mask, 0.0, properties['lice'])
    properties['lwater'] = jnp.where(dead_mask, 0.0, properties['lwater'])
    properties['lheight'] = jnp.where(dead_mask, 0.0, properties['lheight'])
    properties['lBC'] = jnp.where(dead_mask, 0.0, properties['lBC'])
    properties['lOC'] = jnp.where(dead_mask, 0.0, properties['lOC'])
    properties['ldust'] = jnp.where(dead_mask, 0.0, properties['ldust'])
    state = state._replace(**properties)

    layer_indices = jnp.arange(n_layers)
    curve_snow = params.dz_toplayer * jnp.exp(layer_indices * params.layer_growth)
    min_height_by_depth = jnp.maximum(curve_snow, params.min_dz)

    dt_heat = params.dt / params.n_heat_steps
    ice_stability_min = 2.0 * jnp.sqrt(
        4 * params.k_ice * dt_heat / (params.Cp_ice * params.density_ice)
    )
    min_height_ice = jnp.maximum(ice_stability_min, params.min_dz)

    def _scan_merge(current_state, idx):
        dz = current_state.lheight[:, idx]
        curr_type = current_state.ltype[:, idx]
        next_type = current_state.ltype[:, idx + 1]

        is_thin_snow = (curr_type == 0) & (dz < min_height_by_depth[idx])
        is_thin_any = (dz < min_height_ice)

        is_snow = curr_type == 0
        type_matches_below = curr_type == next_type
        force_small_snow = (curr_type == 0) & (next_type > 0) & (dz < params.min_dz)

        any_merge = is_thin_any & ~is_snow
        snow_merge = is_thin_snow & (type_matches_below | force_small_snow)
        merge_mask = any_merge | snow_merge

        next_state = merge_existing_layers_nvars_probe(
            current_state, merge_mask, idx, params, n_vars_shifted
        )
        return next_state, None

    layers_idx = jnp.arange(n_layers - 1)
    state, _ = jax.lax.scan(_scan_merge, state, layers_idx)
    return state


def apply_dynamics_mass_change(state, mask, dmass, params):
    """
    Reconciles a per-point ice-dynamics-only mass change into state. 
    dmass is the DYNAMICS-ONLY change in mass (excluding SMB which 
    is already accounted for in PEBSI).)

    Mass GAINED from dynamics (dmass > 0) goes entirely into
    basal_reservoir, uncapped.

    Mass LOST from dynamics (dmass < 0) is taken from basal_reservoir
    first, down to 0. Only once the reservoir is exhausted does the
    remainder come out of the ice-typed layers themselves, which are
    then redistributed with even heights.

    Parameters
    ==========
    mask : jnp.Array (N_POINTS,)
        Boolean mask for points to reconcile this coupling step
    dmass : jnp.Array (N_POINTS,)
        Ice-dynamics-only mass change, kg m-2 (positive = gain)
    """
    properties = state._asdict()
    ice_mask = properties['ice_mask']
    DENSITY_ICE = params.density_ice

    gain = jnp.maximum(dmass, 0.0)
    loss = jnp.maximum(-dmass, 0.0)

    # gains: straight into the reservoir
    reservoir = jnp.where(mask, properties['basal_reservoir'] + gain, properties['basal_reservoir'])

    # losses: deplete the reservoir first (down to 0)
    from_reservoir = jnp.minimum(loss, reservoir)
    reservoir = jnp.where(mask, reservoir - from_reservoir, reservoir)
    remaining_loss = loss - from_reservoir  # (N_POINTS,), 0 unless reservoir was exhausted

    # remainder: thin the ice layers themselves, evenly redistributed
    n_ice_layers = jnp.sum(ice_mask, axis=1)
    point_ice_mass = jnp.sum(jnp.where(ice_mask, properties['lice'], 0.0), axis=1)
    new_total_ice_mass = jnp.maximum(point_ice_mass - remaining_loss, 0.0)
    per_layer_mass = new_total_ice_mass / jnp.maximum(n_ice_layers, 1)

    thin_mask = mask & (remaining_loss > 0)
    properties['lice'] = jnp.where(
        thin_mask[:, None] & ice_mask, per_layer_mass[:, None], properties['lice']
    )
    properties['ldensity'] = jnp.where(
        thin_mask[:, None] & ice_mask, DENSITY_ICE, properties['ldensity']
    )
    safe_density = jnp.where(properties['ldensity'] > 0, properties['ldensity'], 1.0)
    properties['lheight'] = jnp.where(
        thin_mask[:, None] & ice_mask,
        properties['lice'] / safe_density,
        properties['lheight']
    )

    properties['basal_reservoir'] = reservoir
    state = state._replace(**properties)
    state = update_layer_props(state, DENSITY_ICE)
    return state

def update_layer_props(state, DENSITY_ICE):
    """
    Recalculates nlayers, depths, and density. 
    Can specify to only update certain properties.

    Parameters
    ==========
    do : list-like
        List of any combination of depth, density to be updated
    """
    # floor at a small-but-nonzero height
    safe_lheight = jnp.where(state.lheight > 1e-6, state.lheight, 1e-6)

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
    safe_new_ldensity = jnp.where(new_ldensity > 0, new_ldensity, 1.0)
    new_lheight = jnp.where(
        trans_mask, state.lice / safe_new_ldensity, state.lheight
    )

    # update state
    state = state._replace(
        ltype = new_ltype, ldensity = new_ldensity,
        lheight = new_lheight,
    )
    return state