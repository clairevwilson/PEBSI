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
# Local libraries
import pebsi.input as prms

class Layers():
    """
    Layer scheme for the 1D snowpack model.

    All mass terms are stored in kg m-2.
    """
    def __init__(self,climate,args):
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

        # load in initial depths of snow, firn and ice in m
        dz_snow = args.initial_snow_depth
        dz_firn = args.initial_firn_depth
        dz_ice = prms.initial_ice_depth

        # calculate the layer depths based on initial snow, firn and ice depths
        lheight,ldepth,ltype,nlayers = self.make_layers(dz_snow,dz_firn,dz_ice)
        self.nlayers = nlayers              # NUMBER OF LAYERS
        self.ltype = ltype                  # LAYER TYPE (snow, firn, or ice)
        self.lheight = lheight              # LAYER HEIGHT (dz) [m]
        self.ldepth = ldepth                # LAYER DEPTH (midlayer) [m]

        # initialize layer temperature, density, water content
        ltemp,ldensity,lwater,lgrainsize = self.initialize_layers(dz_snow,dz_firn)
        self.ltemp = ltemp                  # LAYER TEMPERATURE [C]
        self.ldensity = ldensity            # LAYER DENSITY [kg m-3]
        self.lice = ldensity*lheight        # LAYER ICE (SOLID) MASS [kg m-2]
        self.lwater = lwater                # LAYER WATER (LIQUID) MASS [kg m-2]
        self.lgrainsize = lgrainsize        # LAYER GRAIN SIZE [um]

        # initialize LAPs (black carbon, organic carbon, and dust)
        if prms.switch_LAPs == 1:
            lBC,lOC,ldust = self.initialize_LAPs()
        else:
            lBC = np.zeros(self.nlayers)
            lOC = np.zeros(self.nlayers)
            ldust = np.zeros(self.nlayers)
        self.lBC = lBC                      # LAYER BLACK CARBON MASS [kg m-2]
        self.lOC = lOC                      # LAYER ORGANIC CARBON MASS [kg m-2]
        self.ldust = ldust                  # LAYER DUST MASS [kg m-2]
       
        # refreeze content
        self.drefreeze = np.zeros_like(self.ltemp)   # LAYER MASS OF REFREEZE ADDED THIS TIMESTEP [kg m-2]
        self.lrefreeze = np.zeros_like(self.ltemp)   # LAYER MASS OF REFREEZE [kg m-2]

        # layer age
        time_0 = pd.to_datetime(self.args.startdate)
        self.lage = np.array([time_0 for _ in range(self.nlayers)])        # LAYER AGE [date of creation]

        # initialize bucket for 'delayed snow' and running max snow mass
        self.delayed_snow = 0
        self.max_snow = np.sum(self.lice[self.snow_idx])
        
        if args.debug:
            print(f'~ {self.nlayers} layers initialized ~')
        return 
    
    def make_layers(self,snow_height,firn_height,ice_height):
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
        # CONSTANTS
        DZ_TOP = prms.dz_toplayer
        LAYER_GROWTH = prms.layer_growth

        # initialize variables to get looped
        lheight = []
        ltype = []
        current_depth = 0
        layer = 0

        # make exponentially growing snow layers
        while current_depth < snow_height:
            lheight.append(DZ_TOP * np.exp(layer*LAYER_GROWTH))
            ltype.append('snow')
            layer += 1
            current_depth = np.sum(lheight)
        if snow_height > 0:
            lheight[-1] = lheight[-1] - (current_depth-snow_height)
            # can end up with a small bottom layer: merge it
            if lheight[-1] < lheight[-2] / 2:
                lheight[-2] += lheight[-1]
                lheight = lheight[:-1]
                ltype = ltype[:-1]
                layer -= 1
    
        # add firn layers
        if firn_height > 0.75:
            n_firn_layers = int(round(firn_height,0))
            lheight.extend([firn_height/n_firn_layers]*n_firn_layers)
            ltype.extend(['firn']*n_firn_layers)
        elif firn_height > 0:
            lheight.extend([firn_height])
            ltype.extend(['firn'])

        # add ice layers
        current_depth = 0
        while current_depth < ice_height:
            lheight.append(DZ_TOP * np.exp(layer*LAYER_GROWTH))
            ltype.append('ice')
            layer += 1
            ice_idx = np.where(np.array(ltype)=='ice')[0]
            current_depth = np.sum(np.array(lheight)[ice_idx])
        lheight[-1] = lheight[-1] - (current_depth-ice_height)
        
        # get depth of layers (distance from surface to midpoint of layer) [m]
        nlayers = len(lheight)
        if 'layers' in prms.store_vars:
            assert nlayers <= prms.max_nlayers, f'Need >= {nlayers} in prms.max_nlayers'
        ldepth = [np.sum(lheight[:i+1])-(lheight[i]/2) for i in range(nlayers)]

        # make into arrays
        lheight = np.array(lheight)
        ldepth = np.array(ldepth)
        ltype = np.array(ltype)

        # assign indices
        self.snow_idx = np.where(ltype=='snow')[0]
        self.firn_idx = np.where(ltype=='firn')[0]
        self.ice_idx = np.where(ltype=='ice')[0]
        return np.array(lheight), np.array(ldepth), np.array(ltype), nlayers

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
        snow_idx = self.snow_idx
        firn_idx = self.firn_idx
        ice_idx = self.ice_idx

        # read in depth profiles
        temp_data = pd.read_csv(self.args.initial_temp_fp)
        density_data = pd.read_csv(self.args.initial_density_fp)
        grainsize_data = pd.read_csv(self.args.initial_grains_fp)

        # TEMPERATURE [C]
        if prms.initialize_temp == 'interpolate':
            ltemp = np.interp(self.ldepth,temp_data['depth'],temp_data['temp'])
        elif prms.initialize_temp == 'ripe':
            ltemp = np.ones(self.nlayers)*0
        else:
            print('Choose between ripe and interpolate in initialize_temp')
            self.exit()
        
        # GRAIN SIZE [um]
        lgrainsize = np.interp(self.ldepth,grainsize_data['depth'],
                               grainsize_data['grainsize'])
        lgrainsize[self.ltype == 'firn'] = prms.firn_grainsize
        lgrainsize[self.ltype == 'ice'] = prms.ice_grainsize

        # DENSITY [kg m-3]
        if prms.initialize_density == 'interpolate':
            # SNOW layers initialized by interpolation
            ldensity = np.interp(self.ldepth[snow_idx],
                                density_data['depth'],density_data['density'])
            if len(firn_idx) > 0:
                # calculate firn density slope from snow --> ice
                if snow_height > 0 and firn_height > 0:
                    pslope = (prms.density_ice - ldensity[-1]) / (
                        self.ldepth[ice_idx[0]]-self.ldepth[snow_idx[-1]])
                # no snow: set boundary tp constant density_firn
                elif firn_height > 0:
                    pslope = (prms.density_ice - prms.density_firn)/(firn_height)
        elif prms.initialize_density == 'constant':
            ldensity = np.ones_like(snow_idx) * prms.density_snow
        else:
            print('Choose between constant and interpolate in initialize_density')
            self.exit()
            
        # append firn and ice layer densities
        for (type,depth) in zip(self.ltype,self.ldepth):
            if type in ['firn']: 
                ldensity = np.append(ldensity,
                    ldensity[snow_idx[-1]] + pslope*(depth-self.ldepth[snow_idx[-1]]))
            elif type in ['ice']:
                ldensity = np.append(ldensity,prms.density_ice)

        # WATER CONTENT [kg m-2]
        if prms.initialize_water == 'dry':
            lwater = np.zeros(self.nlayers)
        elif prms.initialize_water == 'saturated':
            porosity = 1 - ldensity / prms.density_ice
            lwater = porosity * prms.Sr * self.lheight * prms.density_water
        else:
            print('Choose between dry and saturated in initialize_water')
            self.exit()

        return ltemp,ldensity,lwater,lgrainsize
    
    def initialize_LAPs(self):
        """
        Initializes light-absorbing particle content
        of the snow and firn layers.
        """
        # CONSTANTS
        BC_FRESH = prms.BC_freshsnow
        OC_FRESH = prms.OC_freshsnow
        DUST_FRESH = prms.dust_freshsnow

        # INPUTS
        n = self.nlayers
        lheight = self.lheight
        ldepth = self.ldepth

        if prms.initialize_LAPs in ['clean']:
            # snowpack is clean; initialize as constant values
            lBC = np.ones(n)*BC_FRESH*lheight
            lOC = np.ones(n)*OC_FRESH*lheight
            ldust = np.ones(n)*DUST_FRESH*lheight 
        elif prms.initialize_LAPs in ['interpolate']:
            # read in LAP data
            lap_data = pd.read_csv(self.args.initial_LAP_fp,index_col=0)

            # add boundaries for interpolation
            lap_data.loc[0,'BC'] = BC_FRESH
            lap_data.loc[0,'OC'] = OC_FRESH
            lap_data.loc[0,'dust'] = DUST_FRESH
            lap_data.loc[100,'BC'] = BC_FRESH
            lap_data.loc[100,'OC'] = OC_FRESH
            lap_data.loc[100,'dust'] = DUST_FRESH
            lap_data = lap_data.sort_index()

            # interpolate concentration by depth
            data_depth = lap_data.index.to_numpy()
            cBC = np.interp(ldepth,data_depth,lap_data['BC'].values.flatten())
            cOC = np.interp(ldepth,data_depth,lap_data['OC'].values.flatten())
            cdust = np.interp(ldepth,data_depth,lap_data['dust'].values.flatten())

            # calculate mass from concentration
            lBC = cBC * lheight
            lOC = cOC * lheight
            ldust = cdust * lheight
        else:
            print('Choose between clean and interpolate in initialize_LAPs')
            self.exit()
        lBC[self.ice_idx] = 0
        lOC[self.ice_idx] = 0
        ldust[self.ice_idx] = 0
        return lBC, lOC, ldust
    
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
        self.nlayers += len(layers_to_add.loc['T'].values)
        self.ltemp = np.append(layers_to_add.loc['T'].values,self.ltemp).astype(float)
        self.lwater = np.append(layers_to_add.loc['w'].values,self.lwater).astype(float)
        self.lheight = np.append(layers_to_add.loc['h'].values,self.lheight).astype(float)
        self.ltype = np.append(layers_to_add.loc['t'].values,self.ltype)
        self.lice = np.append(layers_to_add.loc['m'].values,self.lice).astype(float)
        new_layer_age = layers_to_add.loc['time'].values
        self.lage = np.array(pd.to_datetime(np.append(new_layer_age,self.lage)))
        self.lgrainsize = np.append(layers_to_add.loc['g'].values,self.lgrainsize).astype(float)
        new_layer_BC = layers_to_add.loc['BC'].values.astype(float)*self.lheight[0]
        self.lBC = np.append(new_layer_BC,self.lBC)
        new_layer_OC = layers_to_add.loc['OC'].values.astype(float)*self.lheight[0]
        self.lOC = np.append(new_layer_OC,self.lOC)
        new_layer_dust = layers_to_add.loc['dust'].values.astype(float)*self.lheight[0]
        self.ldust = np.append(new_layer_dust,self.ldust)
        # new layers start with 0 refreeze
        self.drefreeze = np.append(0,self.drefreeze) 
        self.lrefreeze = np.append(0,self.lrefreeze)
        self.update_layer_props()
        return
    
    def remove_layer(self,layer_to_remove):
        """
        Removes a single layer from layers class.

        Parameters
        ==========
        layer_to_remove : int
            index of layer to remove
        """
        self.nlayers -= 1
        self.ltemp = np.delete(self.ltemp,layer_to_remove)
        self.lwater = np.delete(self.lwater,layer_to_remove)
        self.lheight = np.delete(self.lheight,layer_to_remove)
        self.ltype = np.delete(self.ltype,layer_to_remove)
        self.lice = np.delete(self.lice,layer_to_remove)
        self.lage = np.delete(self.lage,layer_to_remove)
        self.drefreeze = np.delete(self.drefreeze,layer_to_remove)
        self.lrefreeze = np.delete(self.lrefreeze,layer_to_remove)
        self.lgrainsize = np.delete(self.lgrainsize,layer_to_remove)
        self.lBC = np.delete(self.lBC,layer_to_remove)
        self.lOC = np.delete(self.lOC,layer_to_remove)
        self.ldust = np.delete(self.ldust,layer_to_remove)
        self.update_layer_props()
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
        if (self.nlayers+1) > prms.max_nlayers and 'layers' in prms.store_vars:
            print(f'Need bigger max_nlayers: currently have {self.nlayers+1} layers')
            self.exit()
        l = layer_to_split
        self.nlayers += 1
        self.ltemp = np.insert(self.ltemp,l,self.ltemp[l])
        self.ltype = np.insert(self.ltype,l,self.ltype[l])
        self.lgrainsize = np.insert(self.lgrainsize,l,self.lgrainsize[l])
        self.lwater[l] = self.lwater[l]/2
        self.lwater = np.insert(self.lwater,l,self.lwater[l])
        self.lheight[l] = self.lheight[l]/2
        self.lheight = np.insert(self.lheight,l,self.lheight[l])
        self.lice[l] = self.lice[l]/2
        self.lice = np.insert(self.lice,l,self.lice[l])
        self.lage = np.insert(self.lage,l,self.lage[l])
        self.drefreeze[l] = self.drefreeze[l]/2
        self.drefreeze = np.insert(self.drefreeze,l,self.drefreeze[l])
        self.lrefreeze[l] = self.lrefreeze[l]/2
        self.lrefreeze = np.insert(self.lrefreeze,l,self.lrefreeze[l])
        self.lBC[l] = self.lBC[l]/2
        self.lBC = np.insert(self.lBC,l,self.lBC[l])
        self.lOC[l] = self.lOC[l]/2
        self.lOC = np.insert(self.lOC,l,self.lOC[l])
        self.ldust[l] = self.ldust[l]/2
        self.ldust = np.insert(self.ldust,l,self.ldust[l])
        self.update_layer_props()
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
        l = layer_to_merge
        self.ldensity[l+1] = np.sum(self.ldensity[l:l+2]*self.lice[l:l+2])/np.sum(self.lice[l:l+2])
        self.lwater[l+1] = np.sum(self.lwater[l:l+2])
        self.ltemp[l+1] = np.mean(self.ltemp[l:l+2])
        self.lheight[l+1] = np.sum(self.lheight[l:l+2])
        self.lice[l+1] = np.sum(self.lice[l:l+2])
        self.drefreeze[l+1] = np.sum(self.drefreeze[l:l+2])
        self.lrefreeze[l+1] = np.sum(self.lrefreeze[l:l+2])
        self.lgrainsize[l+1] = np.sum(self.lgrainsize[l:l+2]*self.lice[l:l+2])/np.sum(self.lice[l:l+2])
        self.lBC[l+1] = np.sum(self.lBC[l:l+2])
        self.lOC[l+1] = np.sum(self.lOC[l:l+2])
        self.ldust[l+1] = np.sum(self.ldust[l:l+2])

        # get new layer weighted mean age
        if self.lage[l] != self.lage[l+1]:
            decimal_time = self.to_decimal_year(self.lage[l:l+2])
            mean_time = np.sum(decimal_time*self.lice[l:l+2])/np.sum(self.lice[l:l+2])
            self.lage[l+1] = self.from_decimal_year(mean_time)
        self.remove_layer(l)
        return
    
    def check_layer_sizes(self):
        """
        Checks the layer heights against the initial sizes.
        
        If layers have become too small (less than half their
        original size), they are merged with the layer below.
        
        If layers have become too large (more than double their
        original size), they are split into two layers.
        """
        # define initial mass for conservation check
        initial_mass = np.sum(self.lice + self.lwater)
 
        # layer heights
        if self.ltype[0] in ['snow','firn']:
            DZ0 = prms.dz_toplayer
        else: # if there is only ice, make the minimum layer size larger
            DZ0 = prms.min_dz_ice
        min_heights = lambda i: DZ0 * np.exp((i-1)*prms.layer_growth)/2
        max_heights = lambda i: DZ0 * np.exp((i-1)*prms.layer_growth)*2

        # loop through layers
        layer = 0 
        while layer < self.nlayers:
            # reinitiaze layer split
            layer_split = False

            # get height of current layer
            dz = self.lheight[layer]

            # remove any 0 mass layers
            if self.lice[layer] < prms.mb_threshold / 1000:
                self.remove_layer(layer)

            # check snow layers
            if self.ltype[layer] in ['snow']:
                if dz < min_heights(layer) and self.ltype[layer]==self.ltype[layer+1]:
                    # layer too small: merge if it is the same type as the layer underneath
                    self.merge_layers(layer)
                elif dz > max_heights(layer):
                    # layer too big: split into two equal size layers
                    self.split_layer(layer)
                    layer_split = True
            # firn layers can be any size
            # check ice layers
            if self.ltype[layer] in ['ice']:
                layer_check = layer - len(self.firn_idx)
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
        assert np.abs(change) < prms.mb_threshold, f'check_layers failed mass conservation in {self.args.out}'
        return
    
    def update_layer_props(self,do=['depth','density']):
        """
        Recalculates nlayers, depths, and density. 
        Can specify to only update certain properties.

        Parameters
        ==========
        do : list-like
            List of any combination of depth to be updated
        """
        self.nlayers = len(self.lheight)
        self.snow_idx = np.where(self.ltype=='snow')[0]
        self.firn_idx = np.where(self.ltype=='firn')[0]
        self.ice_idx = np.where(self.ltype=='ice')[0]
        
        lh = self.lheight.copy()
        if 'depth' in do:
            self.ldepth = np.array([np.sum(lh[:i+1])-(lh[i]/2) for i in range(self.nlayers)])
        if 'density' in do:
            self.ldensity = self.lice / self.lheight
        return
    
    def update_layer_types(self):
        """
        Checks if new ice layers have been created by 
        densification of firn.
        """
        # CONSTANTS
        DENSITY_ICE = prms.density_ice
        DZ_CHECK = prms.min_dz_ice

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
            if density_check and self.ltype[layer] == 'snow' and len(self.firn_idx) == 0:
                self.ltype[layer] = 'ice'
                self.ldensity[layer] = DENSITY_ICE
                # merge into ice below if layer is smaller than 1 meter
                if self.lheight[layer] < DZ_CHECK and self.ltype[layer+1] in ['ice']:
                    self.merge_layers(layer)
            # advance layer if it fails the new ice check
            else:
                layer += 1

        # bound density of superimposed ice
        self.ldensity[self.snow_idx][self.ldensity[self.snow_idx] > DENSITY_ICE] = DENSITY_ICE
        return
    
    def to_decimal_year(self, dates):
        dates = pd.to_datetime(dates)
        year = dates.year
        start_of_year = pd.to_datetime(year.astype(str) + '-01-01')
        end_of_year = pd.to_datetime((year + 1).astype(str) + '-01-01')
        year_elapsed = (dates - start_of_year).total_seconds()
        year_duration = (end_of_year - start_of_year).total_seconds()
        return year + year_elapsed / year_duration
    
    def from_decimal_year(self, decimal_years):
        years = decimal_years.astype(int)
        start_of_year = pd.to_datetime(years.astype(str) + '-01-01')
        end_of_year = pd.to_datetime((years + 1).astype(str) + '-01-01')
        year_fraction = decimal_years - years
        year_duration = (end_of_year - start_of_year).total_seconds()
        fractional_seconds = year_fraction * year_duration
        return start_of_year + pd.to_timedelta(fractional_seconds, unit='s')
    
    def exit(self):
        if self.args.debug:
            print('Failed in layers')
            print('Layer temperature:',self.ltemp)
            print('Layer density:',self.ldensity)
        sys.exit()