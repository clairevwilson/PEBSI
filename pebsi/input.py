# Built-in libraries
import os
import socket
# External libraries
import numpy as np
import pandas as pd
import xarray as xr

# ========== USER OPTIONS ========== 
glac_no = '00.00000'    # RGI glacier ID
use_AWS = False         # Default to using AWS data?
debug = False           # Default to printing monthly model status?
store_data = False      # Default to saving data?

# ========== DIRECTORIES AND FILEPATHS ========== 
machine = socket.gethostname()
# GLACIER
metadata_fp = 'data/glacier_metadata.csv'                   # Glacier metadata filename
site_fp = 'data/by_glacier/GLACIER/site_constants.csv'      # Generalized glacier site info filepath
RGI_fp = '../RGI/rgi60/00_rgi60_attribs/'                   # Randolph Glacier Inventory filepath
AWS_fp = '../climate_data/AWS/Processed/'                   # Weather station data filepath
AWS_metadata_fn = 'data/aws_metadata.txt'                   # Weather station metadata filename
# SNICAR
grainsize_fp = 'data/grainsize/drygrainsize(SSAin=##).nc'   # Grain size evolution lookup table filepath
snicar_input_fp = 'biosnicar-py/biosnicar/inputs.yaml'      # SNICAR input filepath
clean_ice_fp = 'biosnicar-py/Data/OP_data/480band/r_sfc/gulkana_cleanice_avg_bba3732.csv' # Ice spectrum filepath
# INITIAL CONDITIONS
initial_temp_fp = 'data/sample_initial_temp.csv'            # Initial temperature profile filepath
initial_density_fp = 'data/sample_initial_density.csv'      # Initial density profile filepath
initial_grains_fp = 'data/sample_initial_grains.csv'        # Initial grain size profile filepath
initial_LAP_fp = 'data/sample_initial_laps.csv'             # Initial LAP content # f'/../Data/Nagorski/May_Mend-2_BC.csv'
# SHADING
dem_fp = '../data/dems/GLACIER_dem.tif'                     # Generalized DEM filepath
shading_fp = 'data/by_glacier/GLACIER/shade/GLACIERSITE_shade.csv'# Generalized shading filepath
# CLIMATE
bias_fp = 'data/bias_adjustment/METHOD_VAR.csv'             # Generalized bias adjustment filepath
climate_fp = '../climate_data/'                             # Climate data filepath
# OUTPUT
output_filepath = '../Output/'                              # Output filepath
albedo_out_fp = '../Output/EB/albedo.csv'                   # Output spectral albedo filepath

# ========== CLIMATE AND TIME INPUTS ========== 
# TIME
startdate = pd.to_datetime('2024-04-20 00:00:00') 
enddate = pd.to_datetime('2025-04-20 00:00:00')

# WEATHER STATION
use_AWS_site = False                        # True to override site (lat, lon, etc.) to the AWS site

# REANALYSIS DATA
reanalysis = 'MERRA2'                       # 'MERRA2' ('ERA5-hourly' ***** BROKEN)
MERRA2_filetag = False                      # False or string to follow 'MERRA2_VAR_' in MERRA2 filename
bias_vars = ['wind','SWin','temp','rh']     # Vars to correct by quantile mapping
    
# ========== MODEL OPTIONS ========== 
# INITIALIATION
initialize_temp = 'interpolate'     # 'interpolate' or 'ripe'
initialize_density = 'interpolate'  # 'interpolate' or 'constant'
initialize_LAPs = 'clean'           # 'interpolate' or 'clean' 
initialize_water = 'dry'            # 'dry' or 'saturated'
surftemp_guess =  -10               # guess for surface temperature of first timestep [C]
initial_snow_depth = 1              # default amount of initial snow [m]
initial_firn_depth = 0              # default amount of initial firn [m]
initial_ice_depth = 200             # default amount of initial ice [m]
# Initial depths of snow and firn may be specified in site_constants or the command line using --s0, --f0

# OUTPUT
store_vars = ['MB','EB','temp','layers']  # Variables to store of the possible set: ['MB','EB','temp','layers','SW]
store_bands = False         # Store spectral albedo .csv
store_climate = False       # Store climate dataset .nc

# METHODS
method_turbulent = 'BulkRichardson'     # 'MO-similarity' or 'BulkRichardson' 
method_diffuse = 'Wohlfahrt'            # 'Wohlfahrt', 'none'
method_heateq = 'Crank-Nicholson'       # 'Crank-Nicholson'
method_densification = 'Boone'          # 'Boone', 'HerronLangway', 'Kojima'
method_cooling = 'iterative'            # 'minimize','iterative' (fast)
method_ground = 'MolgHardy'             # 'MolgHardy'
method_conductivity = 'Douville'        # 'Sauter', 'Douville','Jansson','OstinAndersson','VanDusen'

# OPTIONAL MODULES
option_SWpen = True                     # Calculate penetration of shortwave radiation?

# CONSTANT SWITCHES
constant_snowfall_density = False       # False or density [kg m-3]
constant_freshgrainsize = 54.5          # False or grain size [um] (Kuipers Munneke (2011): 54.5)
constant_drdry = False                  # False or dry metamorphism grain size growth rate [um s-1] (1e-4 seems reasonable)

# ALBEDO SWITCHES
switch_snow = 1             # 0 to turn off fresh snow feedback; 1 to include it
switch_melt = 2             # 0 to turn off melt feedback; 1 for simple degradation; 2 for grain size evolution
switch_LAPs = 1             # 0 to turn off LAPs; 1 to turn on

# ========== INTERNAL CONFIGURATION ========== 
# TIMESTEP
dt = 3600                   # Model timestep [s]
daily_dt = 3600*24          # Seconds in a day [s]
n_heat_steps = 5            # Number of times to run heat equation per dt [-]

# ALBEDO BANDS
wvs = np.round(np.arange(0.2,5,0.01),2) # 480 bands used by SNICAR
band_indices = {}           # dictionary for storing spectral albedo
for i in np.arange(0,480):
    band_indices['Band '+str(i)] = np.array([i])
initSSA = 80   # estimate of Specific Surface Area of fresh snowfall (60, 80 or 100)
grainsize_ds = xr.open_dataset(grainsize_fp.replace('##',str(initSSA)))

# ========== PARAMETERS and CONSTANTS ==========
# <<<<<< Climate downscaling >>>>>
sky_view = 0.95             # Sky-view factor [-]
wind_factor = 1             # Wind factor [-]
kp = 2.25                   # Precipitation factor [-]
precgrad = 0.000130         # Precipitation gradient with elevation [% m-1]
lapserate = -0.0065         # Temperature lapse rate for both gcm to glacier and on glacier between elevation bins [C m-1]
albedo_ice = 0.47           # Ice albedo [-] 
snow_threshold_low = 0.2    # Lower threshold for linear snow-rain scaling [C]
snow_threshold_high = 2.2   # Upper threshold for linear snow-rain scaling [C]
wind_ref_height = 10 if reanalysis in ['ERA5-hourly'] else 2  # Reference height for wind speed [m]
# <<<<<< Numerical >>>>>
dz_toplayer = 0.03          # Thickness of the uppermost layer [m]
layer_growth = 0.5          # Rate of exponential growth of layer size (smaller layer growth = more layers) recommend 0.3-.6
max_nlayers = 80            # Maximum number of vertical layers allowed (more layers --> larger file size)
max_dz = 2                  # Max layer thickness
min_dz_ice = 0.5            # Thickness of uppermost layer when surface is ice [m]
mb_threshold = 0.1          # Threshold to consider not conserving mass (kg m-2 = mm w.e.)
min_glacier_depth = 2       # Minimum ice depth to end the model run [m]
max_temp_change = 5         # Maximum possible temperature change in a timestep for a single layer [K hr-1]
# <<<<<< Boundary conditions >>>>>
temp_temp = 0               # Temperature of temperate ice [C]
temp_depth = 10             # Depth of temperate ice [m]
# <<<<<< Physical properties of snow, ice, water and air >>>>>
density_water = 1000        # Density of water [kg m-3]
density_ice = 900           # Density of ice [kg m-3]
density_firn = 700          # Density threshold for firn [kg m-3]
density_snow = 500          # Average density of snow if held constant [kg m-3]
k_air = 0.023               # Thermal conductivity of air [W K-1 m-1] (Mellor, 1997)
k_ice = 2.25                # Thermal conductivity of ice [W K-1 m-1]
k_water = 0.56              # Thermal conductivity of water [W K-1 m-1]
Cp_water = 4184             # Isobaric heat capacity of water [J kg-1 K-1]
Cp_air = 1005               # Isobaric heat capacity of air [J kg-1 K-1]
Cp_ice = 2050               # Isobaric heat capacity of ice [J kg-1 K-1]
Lv_evap = 2514000           # latent heat of evaporation [J kg-1]
Lv_sub = 2849000            # latent heat of sublimation [J kg-1]
Lh_rf = 333550              # Latent heat of fusion of ice [J kg-1]
viscosity_snow = 3.7e7      # Viscosity of snow [Pa-s]
firn_grainsize = 2000       # Grain size of firn [um]
rfz_grainsize = 1500        # Grain size of refrozen snow [um]
ice_grainsize = 5000        # Grain size of ice [um] (placeholder; unused)
frac_absrad_snow = 0.9      # Fraction of shortwave absorbed radiation for snow [-] 
frac_absrad_ice = 0.8       # Fraction of shortwave absorbed radiation for ice/firn [-] 
extinct_coef_snow = 17.1    # Extinction coefficient for snow [-]
extinct_coef_ice = 2.5      # Extinction coefficient for ice/firn [-]
# <<<<<< Universal constants >>>>>
gravity = 9.81              # Gravity [m s-2]
karman = 0.4                # von Karman's constant [-]
sigma_SB = 5.67037e-8       # Stefan-Boltzmann constant [W m-2 K-4]
solar_constant = 1367       # Solar constant [W m-2]
# <<<<<< Ideal gas law >>>>>
R_gas = 8.3144598           # Universal gas constant [J mol-1 K-1]
molarmass_air = 0.0289644   # Molar mass of Earth's air [kg mol-1]
pressure_std = 101325       # Standard pressure [Pa]
temp_std = 293.15           # Standard temperature [K]
density_std = 1.225         # Air density at sea level [kg m-3]
# <<<<<< Model parameterizations >>>>>
Boone_c5 = 0.016            # Densification parameter [m3 kg-1]
roughness_fresh_snow = 0.24 # Surface roughness length for fresh snow [mm] (Moelg et al. 2012, TC)
roughness_aged_snow = 10    # Surface roughness length for aged snow [mm]
roughness_firn = 4          # Surface roughness length for firn [mm] (Moelg et al. 2012, TC)
roughness_ice = 20          # Surface roughness length for ice [mm] (Moelg et al. 2012, TC)
roughness_aging_rate = 0.5  # Rate in mm/day fresh --> aged snow (60 days from 0.24 to 4.0 => 0.06267)
wet_snow_C = 4.22e-13       # Constant for wet snow metamorphosis [m3 s-1]
Sr = 0.033                  # Fraction of irreducible water content for percolation [-]
albedo_ground = 0.2         # Albedo of ground [-]
# <<<<<< SNICAR >>>>>
albedo_TOD = [14]           # List of time(s) of day to calculate albedo [hr] 
diffuse_cloud_limit = 0.6   # Threshold to consider cloudy vs clear-sky in SNICAR [-]
include_LWC_SNICAR = False  # Include liquid water in SNICAR? (slush)
grainshape_SNICAR = 0       # 0: sphere, 1: spheroid, 2: hexagonal plate, 3: koch snowflake, 4: hexagonal prisms
# <<<<<< Constants for switch runs >>>>>
albedo_deg_rate = 15        # Rate of exponential decay of albedo
average_grainsize = 300     # Grainsize to treat as constant if switch_melt is 0 [um]
albedo_fresh_snow = 0.85    # Albedo of fresh snow for exponential method [-] (Moelg et al. 2012, TC)
albedo_firn = 0.5           # Albedo of firn [-]
# <<<<<< BC and dust >>>>>
# 1 kg m-3 = 1e6 ppb = ng g-1 = ug L-1
ksp_BC = 1                  # Meltwater scavenging efficiency of BC [-] (0.1-0.2 from CLM5)
ksp_OC = 1                  # Meltwater scavenging efficiency of OC [-] (0.1-0.2 from CLM5)
ksp_dust = 0.01             # Meltwater scavenging efficiency of dust [-] (0.015 from CLM5)
BC_freshsnow = 0            # Concentration of BC in fresh snow for initialization [kg m-3]
OC_freshsnow = 0            # Concentration of OC in fresh snow for initialization [kg m-3]
dust_freshsnow = 0          # Concentration of dust in fresh snow for initilization [kg m-3]
adjust_deposition = False   # Adjust deposition according to preprocessed factor
# <<<<<< MERRA-2: LAP binning >>>>>
ratio_BC2_BCtot = 2.08      # Ratio to transform BC bin 2 deposition to total BC
ratio_OC2_OCtot = 1.54      # Ratio to transform OC bin 2 deposition to total OC
ratio_DU3_DUtot = 3         # Ratio to transform dust bin 3 deposition to total dust
ratio_DU_bin1 = 0.0751      # Ratio to transform total dust to SNICAR Bin 1 (0.05-0.5um)
ratio_DU_bin2 = 0.20535     # " SNICAR Bin 2 (0.5-1.25um)
ratio_DU_bin3 = 0.481675    # " SNICAR Bin 3 (1.25-2.5um)
ratio_DU_bin4 = 0.203775    # " SNICAR Bin 4 (2.5-5um)
ratio_DU_bin5 = 0.034       # " SNICAR Bin 5 (5-50um)
# <<<<<< End-of-summer >>>>>
start_end_summer = 228      # Julian day of year to start checking for end of summer (snow -> firn)
new_snow_threshold = 0.02   # Threshold for new snow to consider the start of winter [m w.e.]
new_snow_days = 10          # Number of days to sum snow over and compare against threshold [d]
firn_age = 60               # Number of days old a snow layer has to be to turn it into firn [d]

# ========== OTHER PYGEM INPUTS ========== 
# rgi_regionsO1 = [1]
# rgi_regionsO2 = [2]
# rgi_glac_number = 'all'

# # Types of glaciers to include (True) or exclude (False)
# include_landterm = True                # Switch to include land-terminating glaciers
# include_laketerm = True                # Switch to include lake-terminating glaciers
# include_tidewater = True               # Switch to include tidewater glaciers
# ignore_calving = False                 # Switch to ignore calving and treat tidewater glaciers as land-terminating
# oggm_base_url = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.4/L1-L2_files/elev_bands/'
# logging_level = 'DEBUG' # DEBUG, INFO, WARNING, ERROR, WORKFLOW, CRITICAL (recommended WORKFLOW)
# option_leapyear = 1 # 0 to exclude leap years

# # Reference period runs (runs up to present)
# ref_startyear = 1980                # first year of model run (reference dataset)
# ref_endyear = 2019                  # last year of model run (reference dataset)
# ref_wateryear = 'calendar'          # options for years: 'calendar', 'hydro', 'custom'
# ref_spinupyears = 0                 # spin up years

# # This is where the simulation runs climate data will be set up once we're there
# # Simulation runs (refers to period of simulation and needed separately from reference year to account for bias adjustments)
# gcm_startyear = 1980            # first year of model run (simulation dataset)
# gcm_endyear = 2019              # last year of model run (simulation dataset)
# gcm_wateryear = 'calendar'      # options for years: 'calendar', 'hydro', 'custom'
# gcm_spinupyears = 0             # spin up years for simulation (output not set up for spinup years at present)
# constantarea_years = 0          # number of years to not let the area or volume change
# if gcm_spinupyears > 0:
#     assert 0==1, 'Code needs to be tested to enure spinup years are correctly accounted for in output files'

# GRAVEYARD
# WAYS OF MAKING BIN_ELEV
# dynamics = False
# if dynamics:
#     gdir = oggm.single_flowline_glacier_directory(glac_no, logging_level='CRITICAL') #,has_internet=False
#     all_fls = oggm.get_glacier_zwh(gdir)
#     fls = all_fls.iloc[np.nonzero(all_fls['h'].to_numpy())] # remove empty bins
#     bin_indices = np.linspace(len(fls.index)-1,0,n_bins,dtype=int)
#     bin_elev = fls.iloc[bin_indices]['z'].to_numpy()
#     bin_ice_depth = fls.iloc[bin_indices]['h'].to_numpy()
# bin_elev = np.array([1270,1385,1470,1585,1680,1779]) # From Takeuchi 2009
# bin_elev = np.array([1526,1693,1854])
# bin_ice_depth = np.ones(len(bin_elev)) * 200

# temp_bias_adjust = False                        # Adjust MERRA-2 temperature linearly? ***** probably can delete these
# temp_bias_slope = 0.57596                       # Slope of MERRA-2 --> ON-ICE AWS
# temp_bias_intercept = 1.799                     # Intercept of MERRA-2 --> ON-ICE AWS
# in climate.py under if eb_prms.reanalysis == 'MERRA2':
            # Correct MERRA-2 temperature bias
            # temp_filled = True if not self.args.use_AWS else 'temp' in self.need_vars
            # if eb_prms.temp_bias_adjust and temp_filled:
            #     self.adjust_temp_bias()