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
import numpy as np
import pandas as pd
import xarray as xr
# Internal libraries
from util.config import ConfigError
# Handle GPU vs CPU
try:
    import cupy as cp
    xp = cp 
except:
    xp = np

class Layers():
    """
    Layer scheme for the 1D snowpack model.

    All mass terms are stored in kg m-2.
    """
    def __init__(self,core):
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
        self.climate = core.climate 
        self.args = args = core.args
        self.N_POINTS = N_POINTS = core.sd.n
        self.N_LAYERS = N_LAYERS = args.max_nlayers
        self.shape = (N_POINTS, N_LAYERS)

        # load in initial depths of snow, firn and ice in m
        dz_snow = xp.full(N_POINTS, args.initial_snow_depth)
        # *** rough approximation of where there should be firn
        dz_firn = xp.zeros(N_POINTS)
        dz_firn[core.sd.elev_n > core.sd.median_elev_n] = args.initial_firn_depth

        # calculate the layer depths based on initial snow, firn and ice depths
        self.make_layers(dz_snow,dz_firn)

        # initialize layer temperature, density, water and refreeze content
        self.initialize_layers(dz_snow,dz_firn)

        # initialize LAPs (black carbon, organic carbon, and dust)
        self.initialize_LAPs()

        # initialize layer ages
        self.initialize_age()

        # define intensive and extensive variables
        self.intensive_vars = ['ltemp','ldensity','lage','lgrainsize','ltype']
        self.extensive_vars = ['lice','lwater','lBC','lOC','ldust',
                               'drefreeze','lrefreeze']
        self.all_layer_vars = self.intensive_vars + self.extensive_vars + ['lheight','ldepth']

        # delayed snowfall
        self.delayed_snow = xp.zeros(N_POINTS)
        # running maximum snow mass (reset each year)
        self.max_snow = np.sum(self.lice * self.snow_mask, axis=1)
        # minimum albedo per year
        self.firn_albedos = xp.zeros(N_POINTS + 1)
        
        if args.debug:
            print(f'~ {N_LAYERS} layers initialized ~')
        return 
    
    def make_layers(self,snow_height,firn_height):
        """
        Initializes layer depths based on an exponential
        growth function with prescribed rate of growth 
        and initial layer height. 

        Parameters
        ==========
        snow_height : float
        firn_height : float
        ice_height : float
            Initial depth of snow, firn, and ice [m]

        Returns
        -------
        lheight : np.ndarray
            Height of the layer [m]
        ldepth : np.ndarray
            Depth of the middle of the layer [m]
        ltype : np.ndarray
            Layers types (snow, firn, ice)
        """
        args = self.args
        N_POINTS = self.N_POINTS 
        N_LAYERS = self.N_LAYERS

        # CONSTANTS
        DZ_TOP = args.dz_toplayer
        DZ_SNOW = args.dz_snowlayer
        DZ_ICE = args.dz_icelayer
        LAYER_GROWTH = args.layer_growth

        # initialize variables to get filled
        lheight = xp.zeros(self.shape)
        ltype = xp.zeros(self.shape, dtype=xp.int32)

        # define the exponential curve for indexing reference
        layer_indices = xp.arange(N_LAYERS)

        # are snow layers a constant size?
        if args.option_uniform_snow:
            curve_snow = xp.full(N_LAYERS, DZ_SNOW)
        else:
            curve_snow = DZ_TOP * xp.exp(layer_indices * LAYER_GROWTH)

        # are ice layers a constant size?
        if args.option_uniform_ice:
            curve_ice = xp.full(N_LAYERS, DZ_ICE)
        else:
            curve_ice = DZ_TOP * xp.exp(layer_indices * LAYER_GROWTH)
    
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
            curve_ice_clipped = xp.minimum(curve_ice, args.dz_icelayer)
            lheight[i] = xp.where(xp.arange(N_LAYERS) >= curr_lyr, 
                                  curve_ice_clipped, lheight[i])
            ltype[i] = xp.where(xp.arange(N_LAYERS) >= curr_lyr, 2, ltype[i])

        # calculate midpoint heights
        cum_height = xp.cumsum(lheight, axis=1)
        ldepth = cum_height - (lheight / 2.0)

        # store to self
        self.ltype = xp.array(ltype)            # LAYER TYPE (0, 1, or 2) [-]
        self.lheight = xp.array(lheight)        # LAYER HEIGHT (dz) [m]
        self.ldepth = xp.array(ldepth)          # LAYER DEPTH (midlayer) [m]

        # assign indices
        self.snow_mask = ltype==0       # type 0 = snow
        self.firn_mask = ltype==1       # type 1 = firn
        self.ice_mask = ltype==2        # type 2 = ice
        return

    def initialize_layers(self,snow_height,firn_height):
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

        # read in depth profiles
        temp_data = pd.read_csv(self.args.initial_temp_fn)
        density_data = pd.read_csv(self.args.initial_density_fn)
        grainsize_data = pd.read_csv(self.args.initial_grains_fn)

        # TEMPERATURE [C]
        if args.initialize_temp == 'interpolate':
            ltemp = xp.interp(
                self.ldepth.ravel(),
                temp_data['depth'],
                temp_data['temp']
            ).reshape(self.shape)
        elif args.initialize_temp == 'ripe':
            ltemp = xp.zeros(self.shape)
        else:
            raise ConfigError('Invalid configuration: initialize_temp')
        
        # GRAIN SIZE [um]
        lgrainsize = xp.interp(
            self.ldepth.ravel(),
            grainsize_data['depth'],
            grainsize_data['grainsize']
        ).reshape(self.shape)
        lgrainsize[self.ltype == 1] = args.firn_grainsize
        lgrainsize[self.ltype == 2] = args.ice_grainsize

        # DENSITY [kg m-3]
        if args.initialize_density == 'interpolate':
            # SNOW layers initialized by interpolation
            ldensity = xp.interp(
                self.ldepth.ravel(),
                density_data['depth'],
                density_data['density']
            ).reshape(self.shape)

            # find the snow density at the bottom of each point
            bottom_snow_density = xp.nanmax(ldensity * snow_mask, axis=1)
            bottom_snow_depth = xp.nanmax(self.ldepth * snow_mask, axis=1)

            # if there is no snow, swap in constant firn_density
            bottom_snow_density = xp.where(snow_height > 0, 
                                           bottom_snow_density, 
                                           args.density_firn)

            # compute the density slope from top to bottom of firn
            firn_height_no0 = xp.where(firn_height > 0, firn_height, 1.0)
            pslope = xp.where(
                firn_height > 0,
                (args.density_ice - bottom_snow_density) / firn_height_no0,
                0.0,
            )[:, xp.newaxis]

            # apply the firn interpolation
            bottom_snow_density = bottom_snow_density[:, xp.newaxis]
            bottom_snow_depth = bottom_snow_depth[:, xp.newaxis]
            firn_densities = bottom_snow_density + pslope* (self.ldepth - bottom_snow_depth)
            ldensity = xp.where(firn_mask, firn_densities, ldensity)

            # set constant ice layer density
            ldensity = xp.where(ice_mask, args.density_ice, ldensity)
        elif args.initialize_density == 'constant':
            ldensity = np.ones(self.shape) * args.density_snow
            ldensity = xp.where(firn_mask, args.density_firn, ldensity)
            ldensity = xp.where(ice_mask, args.density_ice, ldensity)
        else:
            raise ConfigError('Invalid configuration: initialize_density')

        # WATER CONTENT [kg m-2]
        if args.initialize_water == 'dry':
            lwater = np.zeros(self.shape)
        elif args.initialize_water == 'saturated':
            porosity = 1 - ldensity / args.density_ice
            lwater = porosity * args.Sr * self.lheight * args.density_water
        else:
            raise ConfigError('Invalid configuration: initialize_water')
        
        # store to self
        self.ltemp = xp.array(ltemp)                    # LAYER TEMPERATURE [C]
        self.ldensity = xp.array(ldensity)              # LAYER DENSITY [kg m-3]
        self.lice = xp.array(ldensity*self.lheight)     # LAYER ICE (SOLID) MASS [kg m-2]
        self.lwater = xp.array(lwater)                  # LAYER WATER (LIQUID) MASS [kg m-2]
        self.lgrainsize = xp.array(lgrainsize)          # LAYER GRAIN SIZE [um]
        self.drefreeze = np.zeros_like(self.ltemp)      # LAYER REFREEZE MASS ADDED PER TIMESTEP [kg m-2]
        self.lrefreeze = np.zeros_like(self.ltemp)      # LAYER REFREEZE MASS [kg m-2]
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
    
    def initialize_age(self):
        """
        Initializes the age of each layer in days.
        """
        # set firn layer ages counting back from start year
        firn_mask_int = self.firn_mask.astype(int)
        lage = xp.cumsum(firn_mask_int, axis=1) * -365

        # firn ages count back in time; snow/ice initialized at 0
        lage = xp.where(self.ltype >= 1, lage, 0.0)

        # store to self
        self.lage = xp.array(lage, dtype=xp.int32)  # LAYER AGE [days]
        return lage
    
    # ========= UTILITY FUNCTIONS ==========
    def add_top_layer(self, mask, new_layer):
        """
        Adds a new layer to the top of layers
        for the points in mask.

        Parameters
        ==========
        mask : xp.Array (N_POINTS)
            Boolean mask for points where a new layer is added
        new_layer : object
            Namespace container with new layer properties
        """
        # convert namespace to a dictionary of {attribute_name: array_values}
        properties = vars(new_layer)

        # loop through vars (e.g., ltemp, ldensity)
        for attr, new_values in properties.items():
            # grab the existing array for this var
            target_array = getattr(self, attr)
            
            # shift layers down one idx for points where a layer is added
            target_array[mask, 1:] = target_array[mask, :-1]
            
            # insert the new layer data
            target_array[mask, 0] = new_values[mask]
        return
    
    def add_bottom_layer(self, mask):
        """
        Fills in data for a new bottom layer added
        as a result of a different layer being removed
        for the points in mask. Data is copied from
        the layer that was previously the bottom (-2).

        Parameters
        ==========
        mask : xp.Array (N_POINTS)
            Boolean mask for points where a new layer is added
        """
        for var in self.all_layer_vars:
            data = getattr(self, var)
            previous_bottom = data[mask, -2]
            data[mask, -1] = previous_bottom 
        return
    
    def remove_layer(self, mask, idx):
        """
        Removes a single layer from layers class
        for the points in mask.

        Parameters
        ==========
        mask : xp.Array (N_POINTS)
            Boolean mask for points where a layer is removed
        idx : int
            Index of the layer to remove
        """
        # shift everything below the removed layer upwards by 1
        for var in self.all_layer_vars:
            data = getattr(self, var)
            data[mask, idx:-1] = data[mask, idx + 1:]

        # add a new bottom layer
        self.add_bottom_layer(mask)        
        return
    
    def split_layer(self, mask, idx):
        """
        Splits a single layer into two layers. Extensive
        properties are halved and intensive properties 
        are maintained.

        Parameters
        ==========
        mask : xp.Array (N_POINTS)
            Boolean mask for points where a layer is split
        layer_to_split : int
            Index of the layer to split for points in mask
        """
        # shift everything below the split layer downwards by 1
        for var in self.all_layer_vars:
            data = getattr(self, var)
            data[mask, idx + 1:] = data[mask, idx:-1]

        # intensive variables were already copied correctly
        # halve the extensive properties in the split layers
        for var in self.extensive_vars:
            data = getattr(self, var)
            data[mask, idx:idx+2] /= 2

        # make sure depth, density and type are updated
        self.lheight = self.lice / self.ldensity
        self.update_layer_props()
        return

    def merge_existing_layers(self, mask, idx):
        """
        Merges two layers into one. Extensive properties
        are added and intensive properties are averaged.

        Parameters
        ==========
        mask : xp.Array (N_POINTS)
            Boolean mask for points where a new layer is added
        idx : int
            Index of the layer to merge with the layer
            beneath it for each point in mask
        """
        # idx is the layer being removed and merged down into target_idx
        target_idx = idx + 1

        # calculate mass weights between the two existing layers
        m_removed = self.lice[mask, idx]
        m_target = self.lice[mask, target_idx]
        m_total = m_removed + m_target

        # merge intensive properties with weighted mean
        for var in self.intensive_vars:
            data = getattr(self, var)
            target = data[mask, target_idx]
            removed = data[mask, idx]
            weighted_avg = (target * m_target + removed * m_removed) / m_total
            
            # make sure ltype is int type
            if var == 'ltype': weighted_avg = weighted_avg.astype(int)
                
            # store the weighted average
            data[mask, target_idx] = weighted_avg

        # merge extensive properties into the lower layer (target_idx)
        for var in self.extensive_vars:
            data = getattr(self, var)
            target = data[mask, target_idx]
            removed = data[mask, idx]
            data[mask, target_idx] = removed + target

        # recalculate height from averaged density 
        self.lheight = self.lice / self.ldensity

        # shift everything below the removed layer upwards by 1
        for var in self.all_layer_vars:
            data = getattr(self, var)
            data[mask, idx:-1] = data[mask, idx + 1:]

        # lost a layer, so need to create new bottom layer
        self.add_bottom_layer(mask)

        # make sure depth, density and type are updated
        self.update_layer_props()
        return
    
    def merge_new_layer(self, mask, new_layer):
        """
        Merges accumulation into existing top layer
        for points in mask. Extensive properties are 
        added and intensive properties are averaged.

        Parameters
        ==========
        mask : xp.Array (N_POINTS)
            Boolean mask for points where a new layer is added
        new_layer : object
            Namespace container with new layer properties
        """
        # convert namespace to a dictionary of {attribute_name: array_values}
        properties = vars(new_layer)

        # calculate mass weights for the merge cells
        m_old = self.lice[mask, 0]
        m_new = new_layer.lice[mask]
        m_total = m_old + m_new

        # take mass-weighted mean for intensive properties
        for var in self.intensive_vars:
            data = getattr(self, var)
            existing = data[mask, 0]
            new = properties[var][mask]
            weighted_avg = (existing * m_old + new * m_new) / (m_total)

            # make sure ltype is int type
            if var == 'ltype': weighted_avg = weighted_avg.astype(int)

            # store the weighted average
            data[mask, 0] = weighted_avg

        # sum extensive properties
        for var in self.extensive_vars:
            data = getattr(self, var)
            existing = data[mask, 0]
            new = properties[var][mask]
            data[mask, 0] = existing + new

        # recalculate heights, depths, and masks
        self.lheight = self.lice / self.ldensity 
        self.update_layer_props()
        return
    
    def check_layer_sizes(self):
        """
        Scans through layers sequentially from top to bottom.
        If a layer is below the minimum height threshold, it is 
        merged with the layer directly beneath it.
        """
        args = self.args

        # remove dead layers (mass ~ 0) across the entire grid
        # run this from bottom up so layer indices don't shift
        for idx in reversed(range(self.N_LAYERS)):
            dead_mask = self.lice[:, idx] < (args.mb_threshold / 1000.0)
            if xp.any(dead_mask):
                self.remove_layer(dead_mask, idx)

        # sequential check for miniscule layers
        # only loop to idx -2 because bottom layer cannot be merged
        idx = 0
        while idx < self.N_LAYERS - 1:
            dz = self.lheight[:, idx]
            
            # identify where the current layer is too thin
            merge_mask = dz < args.min_dz
            
            # only merge snow if it matches the type underneath
            # ice or firn merge unconditionally
            is_snow = self.ltype[:, idx] == 0
            type_matches_below = self.ltype[:, idx] == self.ltype[:, idx + 1]
            merge_mask &= (~is_snow | type_matches_below)

            if xp.any(merge_mask):
                # merge any points where this layer msut be merged
                self.merge_existing_layers(merge_mask, idx)
                
                # do not update idx since layers just shifted up
                continue
            else:
                # no profiles were merged, so move to the next layer idx
                idx += 1
            
        return
    
    def update_layer_props(self,do=['depth','density']):
        """
        Recalculates nlayers, depths, and density. 
        Can specify to only update certain properties.

        Parameters
        ==========
        do : list-like
            List of any combination of depth, density to be updated
        """
        # CONSTANTS
        DENSITY_ICE = self.args.density_ice

        self.snow_mask = self.ltype == 0
        self.firn_mask = self.ltype == 1
        self.ice_mask = self.ltype == 2
        
        if 'depth' in do:
            lh = self.lheight.copy()
            self.ldepth = xp.cumsum(lh, axis=1) - (lh / 2.0)
        if 'density' in do:
            self.ldensity = xp.where(
                self.ice_mask,
                DENSITY_ICE,
                self.lice / self.lheight
            )
        return
    
    def update_layer_types(self):
        """
        Checks if new ice layers have been created by 
        the densification of firn.
        """
        args = self.args
        DENSITY_ICE = args.density_ice

        # find which firn layers have crossed the ice density threshold
        is_firn = self.ltype == 'firn'
        density_check = self.ldensity >= DENSITY_ICE
        trans_mask = is_firn & density_check

        # apply the ice transformation
        self.ltype[trans_mask] = 'ice'
        self.ldensity[trans_mask] = DENSITY_ICE

        # bound density of remaining snow layers (superimposed ice patch)
        too_dense_snow = self.snow_mask & (self.ldensity > DENSITY_ICE)
        self.ldensity[too_dense_snow] = DENSITY_ICE

        return