"""
Surface class for PEBSI

Calculates the surface properties such
as albedo and surface temperature.

@author: clairevwilson
"""
# Built-in libraries
import sys, os
import yaml
import copy
# External libraries
import numpy as np
import pandas as pd
from scipy.optimize import brentq
import suncalc
# Internal libraries
from util.config import ConfigError

# Make SNICAR find-able
sys.path.append(os.getcwd()+'/biosnicar-py/')
sys.path.append(os.getcwd()+'/snicar-fx/src/')

class Surface():
    """
    Tracks properties of the surface including
    surface temperature, type, and albedo.
    """ 
    def __init__(self,layers,time,args,climate):
        # add args and climate to surface class
        self.args = args
        self.climate = climate

        # get the SNICAR function from input
        if args.method_snicar in ['bioSNICAR']:
            self.run_SNICAR = self.run_bioSNICAR
            self.snicar_base_fn = args.biosnicar_input_fn
            self.snicar_fn = args.biosnicar_input_fn
        elif args.method_snicar in ['SNICARfx']:
            self.run_SNICAR = self.run_SNICARfx 
            self.snicar_base_fn = args.snicarfx_input_fn
            self.snicar_fn = args.snicarfx_input_fn
        else:
            raise ConfigError('Invalid SNICAR method')

        # initialize surface properties
        self.stemp = args.surftemp_guess
        self.days_since_snowfall = 0
        self.snow_timestamp = time[0]
        self.stype = layers.ltype[0]

        # set initial albedo based on surface type
        self.albedo_dict = {'snow':args.albedo_fresh_snow,
                            'firn':args.albedo_firn,
                            'ice':args.albedo_ice}
        self.bba = self.albedo_dict[self.stype]
        self.vis_a = self.bba # visible albedo is only used for output comparison
        # when albedo is a scalar, make spectral_weights a scalar of 1
        self.albedo = [self.bba]
        self.spectral_weights = np.ones(1)

        # get shading df and initialize surrounding albedo
        self.shading_df = pd.read_csv(args.shading_fn,index_col=0)
        self.shading_df.index = pd.to_datetime(self.shading_df.index)
        self.albedo_surr = args.albedo_fresh_snow

        # output spectral albedo dataframe
        if args.store_bands:
            bands = np.arange(0,480).astype(str)
            self.albedo_df = pd.DataFrame(np.zeros((0,480)),columns=bands)

        # get the underlying ice spectrum
        clean_ice = pd.read_csv(args.clean_ice_fn,names=[''])

        # find albedo of the base spectrum from the filename
        albedo_string = args.clean_ice_fn.split('bba')[-1].split('.')[0]
        bba = int(albedo_string) / (10 ** len(albedo_string))

        # scale the new spectrum by the ice albedo
        ice_point_spectrum = clean_ice * args.albedo_ice / bba

        # name file for ice spectrum
        clean_ice_fn = args.clean_ice_fn.split('/')[-1]
        self.ice_spectrum_fn = args.clean_ice_fn.replace(clean_ice_fn,f'ice_spectrum_{args.task_id}{args.site}.csv')

        # store new spectrum (will be deleted after run completion)
        df_spectrum = pd.DataFrame(ice_point_spectrum)
        df_spectrum.to_csv(self.ice_spectrum_fn, index=False, header=False)

        # parallel runs need separate input files to access
        if args.task_id != -1:
            self.snicar_fn = self.snicar_fn.replace('inputs',f'inputs_{args.task_id}{args.site}')
        else:
            self.snicar_fn = self.snicar_fn.replace('inputs',f'inputs_inuse')

        # check inputs file works
        if not os.path.exists(self.snicar_fn):
            # no input file: create one from inputs.yaml
            self.reset_SNICAR()
        try:
            # check if SNICAR imports properly
            with HiddenPrints():
                from biosnicar import get_albedo
                _,_ = get_albedo.get('adding-doubling',plot=False,validate=False)
        except:
            # problem in the SNICAR input file: create a new one
            self.reset_SNICAR()
        self.snicar_initialized = False

        # need some initial value for cloud cover and annual minimum albedo
        self.tcc = 1
        self.min_annual_albedo = 1
        return
    
    def daily_updates(self,layers,timestamp):
        """
        Updates daily-evolving surface properties (grain
        size, surface type and days since snowfall)

        Parameters
        ----------
        layers
            Class object from pebsi.layers
        airtemp : float 
            Air temperature [C]
        surftemp : float
            Surface temperature [C]
        timestamp : pd.Datetime
            Current timestep
        """
        self.stype = layers.ltype[0]
        self.days_since_snowfall = (timestamp - self.snow_timestamp)/pd.Timedelta(days=1)
        self.get_surr_albedo(layers,timestamp)
        return
    
    def get_surftemp(self,enbal,layers):
        """
        Iteratively solves energy balance equation
        for the surface temperature.
        
        There are three cases:
        (1) LWout data is input
                surftemp is derived from data
        (2) Qm is positive with surftemp = 0. 
                excess Qm is used to warm layers to the
                melting point or melt layers, depending 
                on layer temperatures
        (3) Qm is negative with surftemp = 0.
                snowpack is cooling and surftemp is 
                lowered to balance Qm
        
        Parameters
        ----------
        enbal
            Class object from pebsi.energybalance
        layers
            Class object from pebsi.layers
        """
        args = self.args 

        # CONSTANTS
        STEFAN_BOLTZMANN = args.sigma_SB
        HEAT_CAPACITY_ICE = args.Cp_ice
        CTOK = args.celsius_to_kelvin
        dt = args.dt

        # define target function to solve energy balance
        def target_func(t):
            return enbal.surface_EB(t, self)

        if not enbal.nanLWout:
            # CASE (1): surftemp from LW data
            self.stemp = np.power(np.abs(enbal.LWout_ds/(dt*STEFAN_BOLTZMANN)),1/4) - CTOK
            Qm = target_func(self.stemp)
        else:
            Qm_check = target_func(0)
            # if Qm>0 with surftemp=0, the surface is melting or warming.
            # if Qm<0 with surftemp=0, the surface is cooling.
            cooling = True if Qm_check < 0 else False
            if not cooling:
                # CASE (2): Energy toward the surface
                self.stemp = 0
                Qm = Qm_check
                if layers.ltemp[0] < 0.: 
                    # warm the top layer to the melting point
                    temp_change = Qm_check*dt/(HEAT_CAPACITY_ICE*layers.lice[0])
                    layers.ltemp[0] += temp_change

                    # temp change can raise layer above melting point
                    if layers.ltemp[0] > 0.:
                        # leave excess energy in the melt energy
                        Qm = layers.ltemp[0]*HEAT_CAPACITY_ICE*layers.lice[0]/dt
                        layers.ltemp[0] = 0.

                        # if top layer is melted, warm the next layer
                        if Qm*dt/args.Lh_rf > layers.lice[0] and layers.ltemp[1] < 0.:
                            leftover = Qm*dt/args.Lh_rf - layers.lice[0]
                            layers.ltemp[1] += leftover/(HEAT_CAPACITY_ICE*layers.lice[1])
                    else:
                        # all energy was used up
                        Qm = 0

            elif cooling:
                # CASE (3): Energy away from surface
                # check cold boundary (-60°C)
                eb_at_60 = target_func(-60)
                if eb_at_60 <= 0:
                    # met cold boundary: stay there
                    self.stemp = -60
                    self.Qm = eb_at_60
                
                # apply cooling methods to determine surf temp
                elif args.method_cooling in ['minimize']:
                    # run minimization on EB function
                    self.stemp = brentq(target_func, -60, 0, xtol=1e-3)

                elif args.method_cooling in ['iterative']:
                    # loop to iteratively calculate surftemp
                    n_iters = 0
                    while True:
                        # initial check of Qm comparing to previous surftemp
                        Qm_check = enbal.surface_EB(self.stemp,self)

                        # adaptive surface temp step size (minimum 0.02, maximum 1)
                        step = min(1, max(0.02, abs(Qm_check) * 0.05))
                        
                        # check direction of flux at that temperature and adjust
                        if Qm_check > 0.5:
                            self.stemp += step
                        elif Qm_check < -0.5:
                            self.stemp -= step

                        # surftemp cannot go below -60
                        self.stemp = max(-60,self.stemp)

                        # count iteration
                        n_iters += 1

                        # break loop if Qm is ~0 or after 10 iterations
                        if abs(Qm_check) < 0.5 or n_iters > 10:
                            # if temp is still bottoming out at -60, resolve minimization
                            if self.stemp == -60 or n_iters > 10:
                                self.stemp = brentq(target_func, -60, 0, xtol=1e-3)
                            break

                # if cooling, Qm must be 0
                Qm = 0

        # update surface balance terms with new surftemp
        self.Qm = Qm
        enbal.surface_EB(self.stemp,self)
        self.tcc = enbal.tcc
        return

    def get_albedo(self,layers,timestamp):
        """
        Checks switches and gets albedo with the correct
        method. If LAPs or grain size are tracked, albedo
        comes from SNICAR, otherwise it is parameterized 
        by surface type or surface age.
        
        Parameters
        ----------
        layers
            Class object from pebsi.layers
        timestamp : pd.Datetime
            Current timestep
        """
        args = self.args

        # CONSTANTS
        ALBEDO_FIRN = args.albedo_firn
        ALBEDO_FRESH_SNOW = args.albedo_fresh_snow
        DEG_RATE = args.albedo_deg_rate
        
        # update surface type
        self.stype = layers.ltype[0]

        # determine the method to get albedo from switches
        if self.stype == 'snow':
            if args.switch_melt == 0:
                if args.switch_LAPs == 0:
                    # SURFACE TYPE ONLY
                    self.albedo = self.albedo_dict[self.stype]
                    self.bba = self.albedo
                elif args.switch_LAPs == 1:
                    # LAPs ON, GRAIN SIZE OFF
                    albedo,sw = self.run_SNICAR(layers,timestamp,override_grainsize=True)
                    self.albedo = albedo
                    self.spectral_weights = sw
            elif args.switch_melt == 1:
                # BASIC DEGRADATION RATE
                age = self.days_since_snowfall
                albedo_aging = (ALBEDO_FRESH_SNOW - ALBEDO_FIRN)*(np.exp(-age/DEG_RATE))
                self.albedo = max(ALBEDO_FIRN + albedo_aging,ALBEDO_FIRN)
                self.bba = self.albedo
            elif args.switch_melt == 2:
                if args.switch_LAPs == 0:
                    # LAPs OFF, GRAIN SIZE ON
                    albedo,sw = self.run_SNICAR(layers,timestamp,override_LAPs=True)
                    self.albedo = albedo
                    self.spectral_weights = sw
                elif args.switch_LAPs == 1:
                    # LAPs ON, GRAIN SIZE ON
                    self.albedo,self.spectral_weights = self.run_SNICAR(layers,timestamp)
        elif self.stype == 'firn':
            # try to retrieve firn albedo from past years of the simulation
            year = pd.to_datetime(layers.lage[0]).year
            if year in layers.firn_albedos:
                # found previous albedo
                self.albedo = layers.firn_albedos[year]
            else:
                # if year is not in the dict, use default
                self.albedo = self.albedo_dict[self.stype]
            self.bba = self.albedo
            self.spectral_weights = np.ones(1)
        else:
            self.albedo = self.albedo_dict[self.stype]
            self.bba = self.albedo
            self.spectral_weights = np.ones(1)

        # make albedo a list
        if '__iter__' not in dir(self.albedo):
            self.albedo = [self.albedo]

        if self.bba < self.min_annual_albedo:
            self.min_annual_albedo = self.bba

        # store
        if args.store_bands:
            if '__iter__' not in dir(self.albedo):
                self.albedo = np.ones(480) * self.albedo
                self.spectral_weights = np.ones(480)
            self.albedo_df.loc[timestamp] = self.albedo.copy()
        return
    
    def initialize_SNICARfx(self):
        # SNICARfx imports
        from snicarfx import snicarfx_wrapper
        from snicarfx.core.config_validator import Config
        
        # open the inputs file        
        with open(self.snicar_fn, 'r') as f:
            self.inputs = yaml.safe_load(f)
        self.SNICAR = snicarfx_wrapper.run_two_stream
        self.snicar_initialized = True

        # verify the YAML using SNICAR functionality
        # this is the proper way of doing it but has issues:
        # crashes on units being None for LAPs
        #    >>> probably need to ask developer
        # as is, it works if you just open inputs and don't validate
        # self.inputs = Config.validate_yaml_file(self.snicar_fn).model_dump()
        return 
    
    def run_SNICARfx(self,layers,timestamp,
                   override_grainsize=False,override_LAPs=False):
        """
        Runs SNICAR model to retrieve broadband albedo. 

        Parameters
        ----------
        layers
            Class object from pebsi.layers
        nlayers : int
            Number of layers to include in the 
            calculation
        max_depth : float
            Maximum depth of layers to include 
            in the calculation
            ** Specify nlayers OR max_depth **
        override_grainsize : Bool
            If True, use constant average grainsize 
            specified in input.py
        override_LAPs: Bool
            If True, use constant LAP concentrations 
            specified in input.py

        Returns
        -------
        albedo : np.ndarray
            Spectral albedo
        spectral_weights : np.ndarray
            Wights of each spectral band
        """
        # check if we already validated inputs
        if not self.snicar_initialized:
            self.initialize_SNICARfx()
        args = self.args 

        # CONSTANTS
        AVG_GRAINSIZE = args.average_grainsize
        DIFFUSE_CLOUD_LIMIT = args.diffuse_cloud_limit
        DENSITY_FIRN = args.density_firn

        # get layers to include in the calculation (top 1m of non-ice layers)
        nlayers = np.where(layers.ldepth >= 1)[0][0] + 1
        if layers.ldensity[nlayers-1] > DENSITY_FIRN:
            # only consider firn or ice layers
            nlayers = np.where(layers.ltype != 'ice')[0][-1] + 1
        idx = np.arange(nlayers)

        # unpack layer variables (need to be stored as lists)
        lheight = layers.lheight[idx].astype(float).tolist()
        ldensity = layers.ldensity[idx].astype(float).tolist()
        lgrainsize = layers.lgrainsize[idx].astype(int)
        lwater = layers.lwater[idx] / (layers.lice[idx]+layers.lwater[idx])

        # specific surface area
        ssa = 3 / (lgrainsize/1e6 * np.array(ldensity))
        ssa[ssa > 100] = 100
        ssa = (ssa.astype(float)).tolist()

        # check if grains in each layer are rounded
        shapes = np.ones(nlayers)*0
        # shapes[lwater >= porosity * FRAC_IRREDUC] = 0
        shapes = (shapes.astype(int)).tolist()

        # convert LAPs from mass to concentration in ppb
        BC = layers.lBC[idx] / layers.lheight[idx] * 1e6
        OC = layers.lOC[idx] / layers.lheight[idx] * 1e6
        dust1 = layers.ldust[idx] / layers.lheight[idx] * 1e6 * args.ratio_DU_bin1
        dust2 = layers.ldust[idx] / layers.lheight[idx] * 1e6 * args.ratio_DU_bin2
        dust3 = layers.ldust[idx] / layers.lheight[idx] * 1e6 * args.ratio_DU_bin3
        dust4 = layers.ldust[idx] / layers.lheight[idx] * 1e6 * args.ratio_DU_bin4
        dust5 = layers.ldust[idx] / layers.lheight[idx] * 1e6 * args.ratio_DU_bin5

        # convert arrays to lists for making input file
        lBC = (BC.astype(float)).tolist()
        lOC = (OC.astype(float)).tolist()
        ldust1 = (dust1.astype(float)).tolist()
        ldust2 = (dust2.astype(float)).tolist()
        ldust3 = (dust3.astype(float)).tolist()
        ldust4 = (dust4.astype(float)).tolist()
        ldust5 = (dust5.astype(float)).tolist()

        # override options for switch runs
        if override_grainsize:
            # overrides grainsize with the average value in prms
            lgrainsize = [AVG_GRAINSIZE for _ in idx]
        if override_LAPs:
            # overrides LAPs with fresh snow values
            lBC = [args.BC_freshsnow*1e6 for _ in idx]
            lOC = [args.OC_freshsnow*1e6 for _ in idx]
            ldust1 = np.array([args.dust_freshsnow*1e6 for _ in idx]).tolist()
            ldust2 = ldust1.copy()
            ldust3 = ldust1.copy()
            ldust4 = ldust1.copy()
            ldust5 = ldust1.copy()

        # copy inputs for this timestep
        inputs = copy.deepcopy(self.inputs)

        # LIGHT_ABSORBING_PARTICLES
        inputs['LIGHT_ABSORBING_PARTICLES']['BC']['CONC'] = lBC
        inputs['LIGHT_ABSORBING_PARTICLES']['OC']['CONC'] = lOC
        inputs['LIGHT_ABSORBING_PARTICLES']['DUST1']['CONC'] = ldust1
        inputs['LIGHT_ABSORBING_PARTICLES']['DUST2']['CONC'] = ldust2
        inputs['LIGHT_ABSORBING_PARTICLES']['DUST3']['CONC'] = ldust3
        inputs['LIGHT_ABSORBING_PARTICLES']['DUST4']['CONC'] = ldust4
        inputs['LIGHT_ABSORBING_PARTICLES']['DUST5']['CONC'] = ldust5

        # ICE
        inputs['ICE']['THICKNESS'] = lheight
        inputs['ICE']['DENSITY'] = ldensity
        inputs['ICE']['SPECIFIC_SURFACE_AREA'] = ssa
        if args.include_LWC_SNICAR:
            inputs['ICE']['LWC'] = lwater.tolist()
        else:
            inputs['ICE']['LWC'] = [0]*nlayers
        inputs['ICE']['GRAIN_SHAPE'] = shapes
        inputs['ICE']['LAYER_TYPE'] = [0]*nlayers

        # filepath for ice albedo
        # list_doc['PATHS']['SFC'] = self.ice_spectrum_fn.split('biosnicar-py/')[-1]

        # solar zenith angle
        lat = self.climate.lat
        lon = self.climate.lon
        time_UTC = timestamp - self.args.timezone
        altitude_angle = suncalc.get_position(time_UTC,lon,lat)['altitude']
        zenith = 180/np.pi * (np.pi/2 - altitude_angle) if altitude_angle > 0 else 89
        inputs['RTM']['SZA'] = int(zenith)
        inputs['RTM']['DIRECT'] = 0 if self.tcc > DIFFUSE_CLOUD_LIMIT else 1
       
        # run get_albedo from SNICAR
        outputs = self.SNICAR(inputs)
        spectral_weights = outputs.spectral_weights
        albedo = outputs.albedo
        args.wvs = outputs.wavelengths * 1e6

        # find broadband albedo from spectral albedo
        self.bba = np.sum(albedo * spectral_weights) / np.sum(spectral_weights)
        
        # calculate visible albedo
        vis_idx = np.where((args.wvs <= 0.75) & (args.wvs >= 0.4))[0]
        self.vis_a = np.sum(albedo[vis_idx] * spectral_weights[vis_idx]) / np.sum(spectral_weights[vis_idx])
        return albedo,spectral_weights
    
    def initialize_bioSNICAR(self):
        with HiddenPrints():
            from biosnicar import get_albedo
            self.model = get_albedo.get

        with open(self.snicar_fn, 'r') as f:
            self.inputs = yaml.safe_load(f)
        
        return
    
    def run_bioSNICAR(self,layers,timestamp,
                   override_grainsize=False,override_LAPs=False):
        """
        Runs SNICAR model to retrieve broadband albedo. 

        Parameters
        ----------
        layers
            Class object from pebsi.layers
        nlayers : int
            Number of layers to include in the 
            calculation
        max_depth : float
            Maximum depth of layers to include 
            in the calculation
            ** Specify nlayers OR max_depth **
        override_grainsize : Bool
            If True, use constant average grainsize 
            specified in input.py
        override_LAPs: Bool
            If True, use constant LAP concentrations 
            specified in input.py

        Returns
        -------
        albedo : np.ndarray
            Spectral albedo
        spectral_weights : np.ndarray
            Wights of each spectral band
        """
        # check if we already validated inputs
        if not self.snicar_initialized:
            self.initialize_bioSNICAR()
        args = self.args 

        # CONSTANTS
        AVG_GRAINSIZE = args.average_grainsize
        DIFFUSE_CLOUD_LIMIT = args.diffuse_cloud_limit
        DENSITY_WATER = args.density_water
        DENSITY_FIRN = args.density_firn
        DENSITY_ICE = args.density_ice
        FRAC_IRREDUC = args.Sr

        # get layers to include in the calculation (top 1m of non-ice layers)
        nlayers = np.where(layers.ldepth >= 1)[0][0] + 1
        if layers.ldensity[nlayers-1] > DENSITY_FIRN:
            # only consider firn or ice layers
            nlayers = np.where(layers.ltype != 'ice')[0][-1] + 1
        idx = np.arange(nlayers)

        # unpack layer variables (need to be stored as lists)
        lheight = layers.lheight[idx].astype(float).tolist()
        ldensity = layers.ldensity[idx].astype(float).tolist()
        lgrainsize = layers.lgrainsize[idx].astype(int)
        lwater = layers.lwater[idx] / (layers.lice[idx]+layers.lwater[idx])
        lrefreeze = layers.lrefreeze[idx].astype(float)

        # grain size files are every 1um up to 1500um, then every 500
        idx_1500 = lgrainsize>1500
        lgrainsize[idx_1500] = np.round(lgrainsize[idx_1500]/500) * 500
        lgrainsize[lgrainsize < 30] = 30    # cap minimum grain size
        lgrainsize = lgrainsize.tolist()    # make array a list

        # convert LAPs from mass to concentration in ppb
        BC = layers.lBC[idx] / layers.lheight[idx] * 1e6
        OC = layers.lOC[idx] / layers.lheight[idx] * 1e6
        dust1 = layers.ldust[idx] / layers.lheight[idx] * 1e6 * args.ratio_DU_bin1
        dust2 = layers.ldust[idx] / layers.lheight[idx] * 1e6 * args.ratio_DU_bin2
        dust3 = layers.ldust[idx] / layers.lheight[idx] * 1e6 * args.ratio_DU_bin3
        dust4 = layers.ldust[idx] / layers.lheight[idx] * 1e6 * args.ratio_DU_bin4
        dust5 = layers.ldust[idx] / layers.lheight[idx] * 1e6 * args.ratio_DU_bin5

        # convert arrays to lists for making input file
        lBC = (BC.astype(float)).tolist()
        lOC = (OC.astype(float)).tolist()
        ldust1 = (dust1.astype(float)).tolist()
        ldust2 = (dust2.astype(float)).tolist()
        ldust3 = (dust3.astype(float)).tolist()
        ldust4 = (dust4.astype(float)).tolist()
        ldust5 = (dust5.astype(float)).tolist()

        # override options for switch runs
        if override_grainsize:
            # overrides grainsize with the average value in prms
            lgrainsize = [AVG_GRAINSIZE for _ in idx]
        if override_LAPs:
            # overrides LAPs with fresh snow values
            lBC = [args.BC_freshsnow*1e6 for _ in idx]
            lOC = [args.OC_freshsnow*1e6 for _ in idx]
            ldust1 = np.array([args.dust_freshsnow*1e6 for _ in idx]).tolist()
            ldust2 = ldust1.copy()
            ldust3 = ldust1.copy()
            ldust4 = ldust1.copy()
            ldust5 = ldust1.copy()

        # copy inputs
        inputs = copy.deepcopy(self.inputs)

        # update changing layer variables
        inputs['IMPURITIES']['BC']['CONC'] = lBC
        inputs['IMPURITIES']['OC']['CONC'] = lOC
        inputs['IMPURITIES']['DUST1']['CONC'] = ldust1
        inputs['IMPURITIES']['DUST2']['CONC'] = ldust2
        inputs['IMPURITIES']['DUST3']['CONC'] = ldust3
        inputs['IMPURITIES']['DUST4']['CONC'] = ldust4
        inputs['IMPURITIES']['DUST5']['CONC'] = ldust5
        inputs['ICE']['DZ'] = lheight
        inputs['ICE']['RHO'] = ldensity
        inputs['ICE']['RDS'] = lgrainsize
        if args.include_LWC_SNICAR:
            inputs['ICE']['LAYER_TYPE'][0] = 4
            inputs['ICE']['LWC'] = lwater.tolist()
        else:
            inputs['ICE']['LAYER_TYPE'][0] = 0
            inputs['ICE']['LWC'] = [0]*nlayers

        # the following variables are constants for the n layers
        ice_variables = ['LAYER_TYPE','SHP','HEX_SIDE','HEX_LENGTH',
                         'SHP_FCTR','WATER_COATING','AR','CDOM']
        # option to change shape in inputs
        shapes = np.ones(nlayers, dtype=int) * 2
        shapes[(lwater > 0) | (lrefreeze > 0)] = 0
        aspect_ratios = np.ones(nlayers, dtype=int) * 0.01
        aspect_ratios[(lwater > 0) | (lrefreeze > 0)] = 0
        inputs['ICE']['SHP'] = shapes[idx].tolist()
        inputs['ICE']['AR'] = aspect_ratios[idx].tolist()
        for var in ice_variables:
            inputs['ICE'][var] = [inputs['ICE'][var][0]] * nlayers

        # filepath for ice albedo
        inputs['PATHS']['SFC'] = self.ice_spectrum_fn.split('biosnicar-py/')[-1]

        # solar zenith angle
        lat = self.climate.lat
        lon = self.climate.lon
        time_UTC = timestamp - self.args.timezone
        altitude_angle = suncalc.get_position(time_UTC,lon,lat)['altitude']
        zenith = 180/np.pi * (np.pi/2 - altitude_angle) if altitude_angle > 0 else 89
        inputs['RTM']['SOLZEN'] = int(zenith)
        inputs['RTM']['DIRECT'] = 0 if self.tcc > DIFFUSE_CLOUD_LIMIT else 1
        
        # run get_albedo from SNICAR
        with HiddenPrints():
            self.model.inputs = inputs
            albedo,spectral_weights = self.model('adding-doubling',plot=False,validate=False)

        # find broadband albedo from spectral albedo
        self.bba = np.sum(albedo * spectral_weights) / np.sum(spectral_weights)
        
        # calculate visible albedo
        vis_idx = np.where((args.wvs <= 0.75) & (args.wvs >= 0.4))[0]
        self.vis_a = np.sum(albedo[vis_idx] * spectral_weights[vis_idx]) / np.sum(spectral_weights[vis_idx])
        return albedo, spectral_weights
    
    def reset_SNICAR(self):
        """
        Checks if SNICAR inputs file is functional.
        If not, generates a new one from a default
        file which is never updated.

        Parameters
        ----------
        fn : str
            Filepath to the inputs.yaml file
        """
        base_filepath = os.path.join(os.getcwd(), self.snicar_base_fn)
        id_filepath = os.path.join(os.getcwd(), self.snicar_fn)

        # remove old file if it exists
        if os.path.exists(id_filepath):
            os.remove(id_filepath)

        # open the base inputs file
        with open(base_filepath, 'rb') as src_file:
            file_contents = src_file.read()

        # copy the base inputs file to fn
        with open(id_filepath, 'wb') as dest_file:
            dest_file.write(file_contents)
        return
    
    def get_surr_albedo(self,layers,timestamp):
        """
        Calculates surrounding albedo by scaling between
        ground albedo and fresh snow albedo using
        the current percentage of the maximum annual 
        snowfall as a proxy.

        Parameters
        ----------
        layers
            Class object from pebsi.layers
        time : pd.Timestamp
            Current timestep
        """
        # CONSTANTS
        ALBEDO_GROUND = self.args.albedo_ground
        ALBEDO_SNOW = self.args.albedo_fresh_snow

        # reset max snowdepth yearly
        if timestamp.month == 1 + timestamp.day == 1 + timestamp.hour == 0:
            layers.max_snow = 0

        # check if max_snow has been exceeded
        current_snow = np.sum(layers.lice[layers.snow_idx])
        layers.max_snow = max(current_snow, layers.max_snow)
        
        # scale surrounding albedo based on snowdepth
        if layers.max_snow <= 0:
            albedo_surr = ALBEDO_GROUND
        else:
            albedo_surr = np.interp(current_snow,
                                np.array([0, layers.max_snow]),
                                np.array([ALBEDO_GROUND,ALBEDO_SNOW]))
        self.albedo_surr = albedo_surr
        return

class HiddenPrints:
    """
    Class to hide prints when running SNICAR
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self,exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        return