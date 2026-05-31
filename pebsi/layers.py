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
        # set firn layer ages counting back from start year
        firn_mask_int = self.firn_mask.astype(int)
        lage = xp.cumsum(firn_mask_int, axis=1) * -365

        # firn ages count back in time; snow/ice initialized at 0
        lage = xp.where(self.ltype >= 1, lage, 0.0)

        # store to self
        self.lage = xp.array(lage, dtype=xp.int32)  # LAYER AGE [days]
        return lage
    
    # ========= UTILITY FUNCTIONS ==========
    def add_layers(self,layers_to_add):
        """
        Adds layers to layers class.

        Parameters
        ==========
        layers_to_add : pd.Dataframe
            Contains temperature 'T', water mass 'w', 
            solid mass 'm', height 'h', type 't', 
            grain size 'g', timestep 'time',
            and impurities 'BC','OC' and 'dust'
        """
        # self.nlayers += len(layers_to_add.loc['T'].values)
        # self.ltemp = np.append(layers_to_add.loc['T'].values,self.ltemp).astype(float)
        # self.lwater = np.append(layers_to_add.loc['w'].values,self.lwater).astype(float)
        # self.lheight = np.append(layers_to_add.loc['h'].values,self.lheight).astype(float)
        # self.ltype = np.append(layers_to_add.loc['t'].values,self.ltype)
        # self.lice = np.append(layers_to_add.loc['m'].values,self.lice).astype(float)
        # new_layer_age = layers_to_add.loc['time'].values
        # self.lage = np.array(pd.to_datetime(np.append(new_layer_age,self.lage)))
        # self.lgrainsize = np.append(layers_to_add.loc['g'].values,self.lgrainsize).astype(float)
        # new_layer_BC = layers_to_add.loc['BC'].values.astype(float)*self.lheight[0]
        # self.lBC = np.append(new_layer_BC,self.lBC)
        # new_layer_OC = layers_to_add.loc['OC'].values.astype(float)*self.lheight[0]
        # self.lOC = np.append(new_layer_OC,self.lOC)
        # new_layer_dust = layers_to_add.loc['dust'].values.astype(float)*self.lheight[0]
        # self.ldust = np.append(new_layer_dust,self.ldust)
        # # new layers start with 0 refreeze
        # self.drefreeze = np.append(0,self.drefreeze) 
        # self.lrefreeze = np.append(0,self.lrefreeze)
        # self.update_layer_props()
        return
    
    def remove_layer(self,layer_to_remove):
        """
        Removes a single layer from layers class.

        Parameters
        ==========
        layer_to_remove : int
            index of layer to remove
        """
        # self.nlayers -= 1
        # self.ltemp = np.delete(self.ltemp,layer_to_remove)
        # self.lwater = np.delete(self.lwater,layer_to_remove)
        # self.lheight = np.delete(self.lheight,layer_to_remove)
        # self.ltype = np.delete(self.ltype,layer_to_remove)
        # self.lice = np.delete(self.lice,layer_to_remove)
        # self.lage = np.delete(self.lage,layer_to_remove)
        # self.drefreeze = np.delete(self.drefreeze,layer_to_remove)
        # self.lrefreeze = np.delete(self.lrefreeze,layer_to_remove)
        # self.lgrainsize = np.delete(self.lgrainsize,layer_to_remove)
        # self.lBC = np.delete(self.lBC,layer_to_remove)
        # self.lOC = np.delete(self.lOC,layer_to_remove)
        # self.ldust = np.delete(self.ldust,layer_to_remove)
        # self.update_layer_props()
        return
    
    def split_layer(self,layer_to_split):
        """
        Splits a single layer into two layers. Extensive
        properties are halved and intensive properties 
        are maintained.

        Parameters
        ==========
        layer_to_split : int
            Index of the layer to split
        """
        # args = self.args
        # if (self.nlayers+1) > args.max_nlayers and 'layers' in args.store_vars:
        #     raise ConfigError('Too many layers: increase max_nlayers')
        # l = layer_to_split
        # self.nlayers += 1
        # self.ltemp = np.insert(self.ltemp,l,self.ltemp[l])
        # self.ltype = np.insert(self.ltype,l,self.ltype[l])
        # self.lgrainsize = np.insert(self.lgrainsize,l,self.lgrainsize[l])
        # self.lwater[l] = self.lwater[l]/2
        # self.lwater = np.insert(self.lwater,l,self.lwater[l])
        # self.lheight[l] = self.lheight[l]/2
        # self.lheight = np.insert(self.lheight,l,self.lheight[l])
        # self.lice[l] = self.lice[l]/2
        # self.lice = np.insert(self.lice,l,self.lice[l])
        # self.lage = np.insert(self.lage,l,self.lage[l])
        # self.drefreeze[l] = self.drefreeze[l]/2
        # self.drefreeze = np.insert(self.drefreeze,l,self.drefreeze[l])
        # self.lrefreeze[l] = self.lrefreeze[l]/2
        # self.lrefreeze = np.insert(self.lrefreeze,l,self.lrefreeze[l])
        # self.lBC[l] = self.lBC[l]/2
        # self.lBC = np.insert(self.lBC,l,self.lBC[l])
        # self.lOC[l] = self.lOC[l]/2
        # self.lOC = np.insert(self.lOC,l,self.lOC[l])
        # self.ldust[l] = self.ldust[l]/2
        # self.ldust = np.insert(self.ldust,l,self.ldust[l])
        # self.update_layer_props()
        return

    def merge_layers(self,layer_to_merge):
        """
        Merges two layers into one. Extensive properties
        are added and intensive properties are averaged.

        Parameters
        ==========
        layer_to_merge : int
            Index of the layer to merge with the layer below
        """
        # l = layer_to_merge
        # self.ldensity[l+1] = np.sum(self.ldensity[l:l+2]*self.lice[l:l+2])/np.sum(self.lice[l:l+2])
        # self.lwater[l+1] = np.sum(self.lwater[l:l+2])
        # self.ltemp[l+1] = np.mean(self.ltemp[l:l+2])
        # self.lheight[l+1] = np.sum(self.lheight[l:l+2])
        # self.lice[l+1] = np.sum(self.lice[l:l+2])
        # self.drefreeze[l+1] = np.sum(self.drefreeze[l:l+2])
        # self.lrefreeze[l+1] = np.sum(self.lrefreeze[l:l+2])
        # self.lgrainsize[l+1] = np.sum(self.lgrainsize[l:l+2]*self.lice[l:l+2])/np.sum(self.lice[l:l+2])
        # self.lBC[l+1] = np.sum(self.lBC[l:l+2])
        # self.lOC[l+1] = np.sum(self.lOC[l:l+2])
        # self.ldust[l+1] = np.sum(self.ldust[l:l+2])

        # # get new layer weighted mean age
        # if self.lage[l] != self.lage[l+1]:
        #     decimal_time = self.to_decimal_year(self.lage[l:l+2])
        #     mean_time = np.sum(decimal_time*self.lice[l:l+2])/np.sum(self.lice[l:l+2])
        #     self.lage[l+1] = self.from_decimal_year(mean_time)
        # self.remove_layer(l)
        return
    
    def check_layer_sizes(self):
        """
        Checks the layer heights against the initial sizes.
        
        If layers have become too small (less than half their
        original size), they are merged with the layer below.
        
        If layers have become too large (more than double their
        original size), they are split into two layers.
        """
        args = self.args

        # define initial mass for conservation check
        initial_mass = np.sum(self.lice + self.lwater)

        # layer heights
        if self.ltype[0] in ['snow','firn']:
            DZ0 = args.dz_toplayer
        else: # if there is only ice, make the minimum layer size larger
            DZ0 = args.min_dz_ice
        min_heights = lambda i: DZ0 * np.exp((i-1)*args.layer_growth)/2
        max_heights = lambda i: DZ0 * np.exp((i-1)*args.layer_growth)*2

        # check if using uniform layers
        if args.option_uniform_snow:
            # only check minimum size
            layer = 0
            while layer < len(self.snow_mask):
                dz = self.lheight[layer]
                if dz < args.min_dz and self.ltype[layer]==self.ltype[layer+1]:
                    # layer too small: merge if it is the same type as the layer underneath
                    self.merge_layers(layer)
                layer += 1
        if args.option_uniform_ice:
            # update ice_mask
            self.ice_mask = np.where(self.ltype=='ice')[0]

            # only check minimum size
            layer = self.ice_mask[0]
            while layer < len(self.ice_mask) - 1:
                # don't check the bottom layer
                dz = self.lheight[layer]
                if dz < args.min_dz:
                    self.merge_layers(layer)
                layer += 1
            if args.option_uniform_snow:
                return

        # loop through layers
        layer = 0 
        while layer < self.nlayers:
            # reinitiaze layer split
            layer_split = False

            # get height of current layer
            dz = self.lheight[layer]

            # remove any 0 mass layers
            if self.lice[layer] < args.mb_threshold / 1000:
                self.remove_layer(layer)

            # SNOW layers
            if self.ltype[layer] in ['snow']:
                if dz < min_heights(layer) and self.ltype[layer]==self.ltype[layer+1]:
                    # layer too small: merge if it is the same type as the layer underneath
                    self.merge_layers(layer)
                elif dz > max_heights(layer):
                    # layer too big: split into two equal size layers
                    self.split_layer(layer)
                    layer_split = True
            
            # FIRN layers can be any size and are not handled

            # ICE layers
            if self.ltype[layer] in ['ice'] and not args.option_uniform_ice:
                layer_check = layer - len(self.firn_mask)
                if dz < min_heights(layer_check) and layer < self.nlayers - 1:
                    # layer too small: merge if it is not the bottom layer
                    self.merge_layers(layer)
                elif dz > max_heights(layer_check):
                    # layer too big: split into two equal size layers
                    self.split_layer(layer)
                    layer_split = True
            
            # advance index unless a layer was added via splitting
            if not layer_split:
                layer += 1

        # CHECK MASS CONSERVATION
        change = np.sum(self.lice + self.lwater) - initial_mass
        assert np.abs(change) < args.mb_threshold, f'check_layers failed mass conservation in {self.args.output_fn}'
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
        densification of firn.
        """
        args = self.args

        # CONSTANTS
        DENSITY_ICE = args.density_ice
        DZ_CHECK = args.min_dz_ice

        layer = 0
        while layer < self.nlayers:
            density_check = self.ldensity[layer] >= DENSITY_ICE
            # firn -> ice
            if density_check and self.ltype[layer] == 'firn':
                self.ltype[layer] = 'ice'
                self.ldensity[layer] = DENSITY_ICE
                # merge into ice below if layer is smaller than 1 meter
                if self.lheight[layer] < DZ_CHECK and self.ltype[layer+1] in ['ice']:
                    self.merge_layers(layer)
            # snow -> ice (occurs with rapid densification, no firn because it is already ice)
            if density_check and self.ltype[layer] == 'snow' and len(self.firn_mask) == 0:
                self.ltype[layer] = 'ice'
                self.ldensity[layer] = DENSITY_ICE
                # merge into ice below if layer is smaller than 1 meter
                if self.lheight[layer] < DZ_CHECK and self.ltype[layer+1] in ['ice']:
                    self.merge_layers(layer)
            # advance layer if it fails the new ice check
            else:
                layer += 1

        # bound density of superimposed ice
        self.ldensity[self.snow_mask][self.ldensity[self.snow_mask] > DENSITY_ICE] = DENSITY_ICE
        return