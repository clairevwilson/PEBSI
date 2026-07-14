"""
Defaults for PEBSI

Contains directories and filepaths, options for
handling climate inputs, model physics, and outputs,
internal configuration information, and physical
constants and parameters.

Any updates to made to these variables can also be
specified in config.yaml under the same name.

E.g.: to change the initial temperature data, add 
      initial_temp_fn: 'xxx.csv' to your config.yaml.

Anything with a commented $ can also be flagged from 
the command line. This will overwrite if the same
variable is present in config.yaml.

Anything with a commented * (also present in the 
variable dynamic_variables) can either be input 
as a scalar or as an array of length (N_POINTS).
"""
import socket

# ====================================================================================================================
#                             USER OPTIONS (CAN ALL BE FLAGGED FROM COMMAND LINE)
# ====================================================================================================================
use_config = True       #$ -c, --use_config         Use configuration file?
output_fp = None        #$ -out, --output_fp        Output file path (None for a default, descriptive name)
rgi_ids = None          #$ -ids, --rgi_ids          List of RGI glacier IDs to simulate
rgi_region = 1          #$ -reg, --rgi_region       RGI region to run (rgi_ids overrides this)
sites = None            #$ --sites                  Names of sites to use with same length as rgi_ids
use_aws = False         #$ -use_aws                 Use AWS data?
store_data = False      #$ -store_data              Store output?
debug = False           #$ -debug                   Print debug statements?
progress_bar = False    #$ -pb, --progress_bar      Show progress bar?

# ====================================================================================================================
#                      DIRECTORIES AND FILEPATHS (ALL FILEPATHS ARE RELATIVE TO PEBSI/)
# ====================================================================================================================
# get machine this simulation is running on
machine = socket.gethostname()

# =========================================== USER EDITABLE ==========================================================
#           The following filepaths can be absolute or relative to the current directory, PEBSI.
#                           Thus in config.yaml you can use absolute filepaths 
#                   (e.g., if you want to store the output to a different location.)
# ====================================================================================================================

# GENERAL
config_fn = 'config.yaml'           #$ -cf, --config_fn     # Configuration .yaml file    
output_fp = '../Output/'                                    # General output filepath
output_fn = '{g}_{i}.zarr'                                  # Output file name, to be formatted by glacier ID and point index

# GLACIER
rgi_fp = '../RGI/rgi60/00_rgi60_attribs/'                   # Randolph Glacier Inventory attributes filepath
cop30_vrt_path = '../data/dems/COP30/COP30_reg{r}.vrt'      # Path to VRT file for COP30 DEMs, inside cop30_fp
dem_fn = None                                               # DEM filename to override COP30 (e.g. for modeling one glacier)
shading_fp = '../data/shading/'                             # Shading data filepath for output of shading model
sample_shading_fp = 'data/sample_data/'                     # Shading filepath for the test glacier (region 00)
shading_fn = '{gid}_shadows.zarr'                           # Shading data filename template

# CLIMATE
climate_fp = '../climate_data/'                             # General climate data filepath
aws_fn = 'data/sample_data/sample_aws.csv'                  # Sample weather station filename

# INITIAL CONDITIONS
initial_temp_fn = 'data/init/initial_temp_profile.csv'      # Initial temperature profile filepath
initial_density_fn = 'data/init/initial_density_profile.csv' # Initial density profile filepath
initial_grains_fn = 'data/init/initial_grainsize_profile.csv' # Initial grain size profile filepath
initial_LAP_fn = 'data/init/initial_laps_profile.csv'       # Initial LAP content

# =============================================== INTERNAL ===========================================================
#       The following filepaths are internal to the model and should be relative to the current directory.
#                            Don't edit these unless you have good reason to.
# ====================================================================================================================

# UTILITY
metadata_fn = 'data/glacier_metadata.csv'                   # Glacier metadata filename containing site information
grainsize_fn = 'data/grainsize/drygrainsizeSSAin{s}.nc'     # Grain size evolution lookup table filepath
emulator_fn = 'data/snicar_emulator.npz'                    # SNICAR emulator .npz file (CANNOT BE CHANGED IN CONFIG.YAML)

# CLIMATE
merra2_laps_fn = 'MERRA2/reg{r}_{sp}_regression_map.nc'     # Regional file of BC2-->BCtot and OC2-->OCtot ratios
ukesm_merra_laps_fn = 'ukesm_merra2_reg{r}_{sp}{t}.nc'      # Regional file of UK-ESM-->MERRA-2 deposition ratio
ukesm_fp = '../UKESM/dr401_GFED/'                           # UK-ESM deposition data filepath relative to climate_fp plus one level
ukesm_fn = 'sum_{sp}_{t}deposition_kgm-2s-1.nc'             # UK-ESM deposition data filename
bias_fn = 'data/quantile_cdfs.nc'                           # Bias adjustment file name

# ====================================================================================================================
#                                            CLIMATE AND TIME INPUTS
# ====================================================================================================================
# TEMPORAL
start_date = '2024-04-20 00:00'             #$ -start, --start_date     Simulation start time in UTC
end_date = '2024-04-22 00:00'               #$ -end, --end_date         Simulation end time in UTC
temporal_chunks = 8760                      # Number of chunks to split up simulation into
                                            # chunk size should be close to divisible by 8760 for clean spin-up
n_spinup_years = 1                          # Number of years to repeat forcings to spin-up the model

# SPATIAL 
n_points = 6                                # Number of points to divide domain into

# WEATHER STATION
aws_elev = None                             # Elevation of the weather station [m a.s.l.]

# CLIMATE DATA
climate_source = 'MERRA2'                   # 'MERRA2'
deposition_data = None                      # None or 'UKESM'
ukesm_vn = (                                # Name of var in UKESM data
        'tendency_of_atmosphere_'
        'mass_content_of_{sp}_dry_aerosol_'
        'particles_due_to_{t}_deposition')   
wind_ref_height = 2                         # Reference height for wind speed [m]

# BIAS CORRECTION
bias_vars = []                              # Vars to correct by quantile mapping (only applied to reanalysis data)

# PERTURBATIONS
temp_perturb = 0                            # Additive factor to apply to model temperature
tp_perturb = 1                              # Multiplicative factor to apply to precipitation

# ====================================================================================================================
#                                            MODEL OPTIONS
# ====================================================================================================================
# INITIALIATION
initialize_temp = 'interpolate'         # 'interpolate' or 'ripe'
initialize_density = 'interpolate'      # 'interpolate' or 'constant'
initialize_LAPs = 'interpolate'         # 'interpolate' or 'clean' 
initialize_water = 'dry'                # 'dry' or 'saturated'
surftemp_guess =  -10                   # guess for surface temperature of first timestep [C]
initial_snow_depth = 1                  # default amount of initial snow [m]
initial_firn_depth = 10                 # default amount of initial firn [m] (only for sites identified as accumulation area)

# OUTPUT
store_vars = ['MB','EB','climate']      # Variables to store of the possible set: ['MB','EB','layers','climate']
differentiable = False                  # Enable gradient checkpointing inside inner scan (set True for optimization)

# METHODS
method_distribute = 'grid'              # 'grid', 'sites'
method_densification = 'Boone'          # 'Boone', 'HerronLangway', 'Kojima'
method_conductivity = 'Douville'        # 'Sauter', 'Douville', 'Jansson', 'OstinAndersson', 'VanDusen'

# OPTIONAL MODULES
option_SWpen = True                     # Calculate penetration of shortwave radiation?
option_flat_plates = False              # Use flat grain shapes in SNICAR emulator or only spherical?
option_accel_grains = False             # Accelerate wet grain metamorphosis?

# CONSTANT SWITCHES
constant_snowfall_density = False       # False or density [kg m-3]
constant_freshgrainsize = False         # False or grain size [um] (Kuipers Munneke (2011): 54.5)
constant_drdry = False                  # False or dry metamorphism grain size growth rate [um s-1] (1e-4 seems reasonable)
constant_irrwater = True                # True or False to estimate from density (Sr_light and Sr_dense)

# ====================================================================================================================
#                                             INTERNAL CONFIGURATION
# ====================================================================================================================
# TIMESTEP
dt = 3600                   # Model timestep [s]
daily_dt = 86400            # Seconds in a day [s]
n_heat_steps = 5            # Number of times to run heat equation per dt [-]

# GRAIN SIZE INITIALIZATION
initSSA = 80   # estimate of Specific Surface Area of fresh snowfall (60, 80 or 100)

# INTENSIVE AND EXTENSIVE LAYER VARIABLES
intensive_vars = ['ltemp','ldensity','lage','lgrainsize','ltype']
extensive_vars = ['lice','lwater','lBC','lOC','ldust','ldrefreeze','lrefreeze']
all_layer_vars = intensive_vars + extensive_vars + ['lheight','ldepth']

# ====================================================================================================================
#                                            PARAMETERS AND CONSTANTS
# ====================================================================================================================

# <<<<<< Climate downscaling >>>>>>
wind_factor = 1             #* Wind factor [-]
kp = 2.25                   #* Precipitation factor [-]
dust_factor = 1             #* Dust factor [-]
precgrad = 0                #* Precipitation gradient with elevation [% m-1] 
lapse_rate = -6.5           #* Temperature lapse rate for both gcm to glacier and on glacier between elevation bins [K km-1]
albedo_fresh_snow = 0.85    #* Albedo of fresh snow [-]
albedo_ice = 0.47           #* Albedo of ice [-] 
albedo_firn = 0.4           #* Albedo of firn [-]
snow_threshold_low = 0.2    #* Lower threshold for linear snow-rain scaling [C]
snow_threshold_high = 2.2   #* Upper threshold for linear snow-rain scaling [C]
# <<<<<< Boundary conditions >>>>>>
temp_temp = 0               #* Temperature of temperate ice [C]
temp_depth = 10             #* Depth of temperate ice [m]
# <<<<<< Numerical >>>>>>
dz_toplayer = 0.05          # Thickness of the uppermost layer [m]
dz_snowlayer = 0.1          # Thickness of snow layers if option_uniform_snow [m]
dz_icelayer = 5             # Thickness of ice layers if option_uniform_ice [m]
layer_growth = 0.3          # Rate of exponential growth of layer size (smaller layer growth = more layers) recommend 0.3-.6
max_nlayers = 50            # Maximum number of vertical layers allowed (more layers --> larger file size)
min_area = 2                # Minimum area of glacier to simulate if running an entire region [km2]
min_dz = 0.01               # Minimum size a layer can be before it is merged with layer below, regardless of option_uniform [m]
min_layer_mass = 0.001      # Threshold below which to discard layer mass [kg m-2 = mm w.e.]
min_glacier_depth = 1000    # Minimum ice depth to end the simulation (fills up with zeros) [kg m-2 = mm w.e.]
max_temp_change = 2         # Maximum possible temperature change in a timestep for a single layer [K hr-1]
# <<<<<< Physical properties of snow, ice, water and air >>>>>>
density_water = 1000        # Density of water [kg m-3]
density_ice = 900           # Density of ice [kg m-3]
density_firn = 700          # Density threshold for firn [kg m-3]
density_snow = 500          # Average density of snow if held constant [kg m-3]
density_fresh_snow = 100    # Density of fresh snow for grain growth factor [kg m-3]
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
grainsize_firn = 5000       # Grain size of firn [um]
grainsize_rfz = 1500        # Grain size of refrozen snow [um]
grainsize_ice = 5000        # Grain size of ice [um] (placeholder; unused)
frac_absrad_snow = 0.9      # Fraction of shortwave absorbed radiation for snow [-] 
frac_absrad_ice = 1         # Fraction of shortwave absorbed radiation for ice/firn [-] 
extinct_coef_snow = 17.1    # Extinction coefficient for snow [-]
extinct_coef_ice = 2.5      # Extinction coefficient for ice/firn [-]
surf_emissivity = 1         # Emissivity of ice
# <<<<<< Universal constants >>>>>>
gravity = 9.81              # Gravity [m s-2]
karman = 0.4                # von Karman's constant [-]
sigma_SB = 5.67037e-8       # Stefan-Boltzmann constant [W m-2 K-4]
solar_constant = 1367       # Solar constant [W m-2]
# <<<<<< Unit conversions >>>>>>
celsius_to_kelvin = 273.15
seconds_per_hour = 3600
# <<<<<< Ideal gas law >>>>>>
R_gas = 8.3144598           # Universal gas constant [J mol-1 K-1]
molarmass_air = 0.0289644   # Molar mass of Earth's air [kg mol-1]
# <<<<<< Model parameterizations >>>>>>
Boone_c1 = 2.7e-6           # Densification c1 [s-1]
Boone_c2 = 0.042            # Densification c2 [K-1]
Boone_c3 = 0.046            # Densification c3 [m3 kg-1]
Boone_c4 = 0.081            # Densification c4 [K-1]
Boone_c5 = 0.016            # Densification parameter [m3 kg-1]
roughness_fresh_snow = 0.24 #* Surface roughness length for fresh snow [mm] (Moelg et al. 2012, TC)
roughness_aged_snow = 10    #* Surface roughness length for aged snow [mm]
roughness_firn = 4          #* Surface roughness length for firn [mm] (Moelg et al. 2012, TC)
roughness_ice = 20          #* Surface roughness length for ice [mm] (Moelg et al. 2012, TC)
roughness_aging_rate = 0.5  #* Rate in mm/day fresh --> aged snow (60 days from 0.24 to 4.0 => 0.06267)
wet_grain_C = 4.22e-13      # Constant for wet snow metamorphosis [m3 s-1]   4.22e-13
Sr = 0.05                   # Fraction of irreducible water content for percolation [-]
Sr_dense = 0.12             # Irreducible water content fraction for dense snow (>500 kg m-3) (0.12)
Sr_light = 0.033            # Irreducible water content fraction for less dense snow (<= 500 kg m-3) (0.033)
albedo_ground = 0.2         #* Albedo of surrounding bedrock [-]
# <<<<<< SNICAR >>>>>
albedo_TOD = [14]           # List of time(s) of day to calculate albedo [hr] 
diffuse_cloud_limit = 0.6   # Threshold to consider cloudy vs clear-sky in SNICAR [-]
# <<<<<< BC and dust >>>>>
# 1 kg m-3 = 1e6 ppb = ng g-1 = ug L-1
ksp_BC = 1                  #* Meltwater scavenging efficiency of BC [-] (0.1-0.2 from CLM5)
ksp_OC = 1                  #* Meltwater scavenging efficiency of OC [-] (0.1-0.2 from CLM5)
ksp_dust = 0.01             #* Meltwater scavenging efficiency of dust [-] (0.015 from CLM5)
BC_freshsnow = 0            # Concentration of BC in fresh snow for initialization [kg m-3]
OC_freshsnow = 0            # Concentration of OC in fresh snow for initialization [kg m-3]
dust_freshsnow = 0          # Concentration of dust in fresh snow for initilization [kg m-3]
# <<<<<< MERRA-2: LAP binning >>>>>
ratio_DU3_DUtot = 3         # Ratio to transform dust bin 3 deposition to total dust
merra_lat_res = 0.5         # Resolution of MERRA-2 latitudinally [deg]
merra_lon_res = 0.625       # Resolution of MERRA-2 longitudinally [deg]
# <<<<<< End-of-summer >>>>>
start_end_summer = 228      # Julian day of year to start checking for end of summer (snow -> firn)
new_snow_threshold = 50     # Threshold for new snow to consider the start of winter [kg m-2 = mm w.e.]
new_snow_days = 10          # Number of days to sum snow over and compare against threshold [d]
firn_age = 60               # Number of days old a snow layer has to be to turn it into firn [d]