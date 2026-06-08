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
import equinox as eqx
import numpy as np
import jax 
import jax.numpy as jnp
import pandas as pd

class SNICAREmulator(eqx.Module):
    layers: list
 
    def __init__(self, in_dim, key):
        keys = jax.random.split(key, 4)
        self.layers = [
            eqx.nn.Linear(in_dim, 128, key=keys[0]), eqx.nn.LayerNorm((128,)),
            eqx.nn.Linear(128,    128, key=keys[1]), eqx.nn.LayerNorm((128,)),
            eqx.nn.Linear(128,     64, key=keys[2]), eqx.nn.LayerNorm((64,)),
            eqx.nn.Linear(64,       1, key=keys[3]),
        ]
 
    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.gelu(layer(x)) if isinstance(layer, eqx.nn.Linear) else layer(x)
        return jax.nn.sigmoid(self.layers[-1](x)).squeeze()
    
model = eqx.tree_deserialise_leaves(
    'snicar_emulator.eqx',
    eqx.tree_at(lambda m: m, SNICAREmulator(25, jax.random.PRNGKey(0)), 
                replace_fn=lambda x: x.astype(jnp.float32) if eqx.is_array(x) else x)
)
norm  = np.load('snicar_norm.npz')
mu, sigma = jnp.array(norm['mu']), jnp.array(norm['sigma'])

def get_albedo(state, args, solar_zenith):
    lheight = state.lheight[:, :4]
    ldensity = state.ldensity[:, :4]
    lgrainsize = state.lgrainsize[:, :4]
    lBC = state.lBC[:, :4]
    lOC = state.lOC[:, :4]
    ldust = state.ldust[:, :4]

    cBC = lBC / lheight * 1e6
    cOC = lOC / lheight * 1e6
    cdust = ldust / lheight * 1e6

    # stack inputs
    X = jnp.concatenate([
        jnp.stack([lgrainsize[:,i], ldensity[:,i], lheight[:,i],
                cBC[:,i], cOC[:,i], cdust[:,i]], axis=1)
        for i in range(4)
    ] + [solar_zenith[:, None]], axis=1)
    
    # calculate albedo from emulator
    albedo = jax.vmap(model)((X - mu) / sigma)  # (N_POINTS,)

    final_albedo = jnp.where(
        state.ltype[:, 0] == 0,
        albedo,
        jnp.where(state.ltype[:, 0] == 1,
                  args.albedo_firn, args.albedo_ice)
    )
    return final_albedo


# Make SNICAR find-able
sys.path.append(os.getcwd()+'/biosnicar-py/')
sys.path.append(os.getcwd()+'/snicar-fx/src/')

class Surface():
    """
    Tracks properties of the surface including
    surface temperature, type, and albedo.
    """ 
    def __init__(self,core, layers):
        # add args and climate to surface class
        self.args = args = core.args
        self.climate = climate = core.climate
        self.dates = climate.dates

        # get the SNICAR function from input
        if args.method_snicar in ['bioSNICAR']:
            self.run_SNICAR = self.run_bioSNICAR
            self.initialize_bioSNICAR()
        elif args.method_snicar in ['SNICARfx']:
            self.run_SNICAR = self.run_SNICARfx 
            self.initialize_SNICARfx()
        elif args.method_snicar in ['emulator']:
            self.emulator = args.SNICAR_emulator 
            self.SWin_emulator = climate.SW_emulator_input
            self.PDD_emulator = climate.PDD_emulator_input 
            self.BC_emulator = climate.BC_emulator_input
        else:
            raise ConfigError('Invalid SNICAR method')

        # initialize surface properties
        # all placeholders: will be updated in first timestep
        N_POINTS = core.sd.n
        self.days_since_snowfall = xp.zeros(N_POINTS)
        self.snow_timestamp = xp.zeros(N_POINTS)
        self.stemp = xp.full(N_POINTS, args.surftemp_guess)
        self.stype = layers.ltype[:, 0]
        self.albedo_surr = xp.full(N_POINTS, args.albedo_fresh_snow)
        self.tcc = xp.ones(N_POINTS)
        self.min_annual_albedo = xp.ones(N_POINTS)

        # set initial albedo based on surface type
        self.albedo_dict = {0:args.albedo_fresh_snow,
                            1:args.albedo_firn,
                            2:args.albedo_ice}
        self.bba = xp.full(N_POINTS, self.albedo_dict[self.stype[0]])
        self.vis_a = self.bba

        # when albedo is a scalar, make spectral_weights a scalar of 1
        self.albedo = xp.full(N_POINTS, self.bba)
        self.spectral_weights = xp.ones(N_POINTS)
        return
    
    def daily_updates(self,layers,ti):
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
        self.stype = layers.ltype[:, 0]
        dt_to_day = self.args.daily_dt / self.args.dt
        self.days_since_snowfall = (ti - self.snow_timestamp) / dt_to_day
        self.get_surr_albedo(layers, self.dates[ti])
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
            # emulator overrides switches
            if args.method_snicar in ['emulator']:
                date = timestamp.replace(hour=0)
                features = self.SNICAR_inputs.loc[date].to_frame().T
                features.insert(2, 'days_since_acc', self.days_since_snowfall)
                self.albedo = self.emulator.predict(features)
                self.spectral_weights = np.ones(1)
            elif args.switch_melt == 0:
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
        self.inputs = self.args.snicar_inputs
        self.SNICAR = snicarfx_wrapper.run_two_stream

        # verify the YAML using SNICAR functionality
        # this is the proper way of doing it but has issues:
        # crashes on units being None for LAPs
        #    >>> probably need to ask developer
        # as is, it works if you just open inputs and don't validate
        # self.inputs = Config.validate_yaml_file(self.args.snicar_fn).model_dump()
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
        # list_doc['PATHS']['SFC'] = self.args.ice_spectrum_fn.split('biosnicar-py/')[-1]

        # solar zenith angle
        lat = self.climate.lat
        lon = self.climate.lon
        time_UTC = timestamp - self.args.timezone
        altitude_angle = suncalc.get_position(time_UTC,lon,lat)['altitude']
        zenith = 180/np.pi * (np.pi/2 - altitude_angle) if altitude_angle > 0 else 89
        inputs['RTM']['SZA'] = int(zenith)
        inputs['RTM']['DIRECT'] = 0 if self.tcc > DIFFUSE_CLOUD_LIMIT else 1
       
        # run SNICAR
        outputs = self.SNICAR(inputs)

        # grab arrays from outputs
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
            self.SNICAR = get_albedo.get

        self.inputs = self.args.snicar_inputs
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
        ice_variables = ['LAYER_TYPE','HEX_SIDE','HEX_LENGTH',
                         'SHP_FCTR','WATER_COATING','CDOM']
        # option to change shape in inputs
        porosity = 1 - layers.lice[0] / (lheight[0]*DENSITY_ICE)
        no_water = lwater[0] < porosity * FRAC_IRREDUC
        shapes = np.ones(nlayers, dtype=int) * 2
        shapes[(no_water) | (lrefreeze > 0)] = 0
        aspect_ratios = np.ones(nlayers, dtype=int) * 0.01
        aspect_ratios[(no_water) | (lrefreeze > 0)] = 0
        inputs['ICE']['SHP'] = shapes[idx].tolist()
        inputs['ICE']['AR'] = aspect_ratios[idx].tolist()
        for var in ice_variables:
            inputs['ICE'][var] = [inputs['ICE'][var][0]] * nlayers

        # filepath for ice albedo
        inputs['PATHS']['SFC'] = self.args.ice_spectrum_fn.split('biosnicar-py/')[-1]

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
            albedo,spectral_weights = self.SNICAR(inputs)

        # find broadband albedo from spectral albedo
        self.bba = np.sum(albedo * spectral_weights) / np.sum(spectral_weights)
        
        # calculate visible albedo
        vis_idx = np.where((args.wvs <= 0.75) & (args.wvs >= 0.4))[0]
        self.vis_a = np.sum(albedo[vis_idx] * spectral_weights[vis_idx]) / np.sum(spectral_weights[vis_idx])
        return albedo, spectral_weights