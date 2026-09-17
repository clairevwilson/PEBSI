"""
Project-wide parameters shared across scripts that build and run PEBSI
simulations (AD_optimize.py, loss_surface.py, point_density_rule.py, and
anything else that assembles a config rather than just reading model
output).

Machine-dependent filepaths live here too: any simulation-building
script can refer to HOST_PATHS[host] to find common filepaths for the
systems used in development.
"""
import socket

if 'trace' in socket.gethostname():
    host = 'trace'
else:
    host = 'bridges'

HOST_PATHS = {
    'trace': dict(
        climate_fp='/trace/group/rounce/cvwilson/climate_data/',
        rgi_fp='/trace/group/rounce/shared/RGI/rgi60/00_rgi60_attribs/',
        output_fp='/trace/group/rounce/cvwilson/Output/AD_optimize/',
        cop30_vrt_path='/trace/group/rounce/cvwilson/dems/RGI1_DEM/rgi_dem.vrt',
        shading_fp='/trace/group/rounce/cvwilson/shading/',
    ),

    'bridges': dict(
        climate_fp='/ocean/projects/ees260009p/cwilson4/climate_data/',
        rgi_fp='/ocean/projects/ees260009p/cwilson4/RGI/rgi60/00_rgi60_attribs/',
        output_fp='/ocean/projects/ees260009p/cwilson4/Output/AD_optimize/',
        cop30_vrt_path='/ocean/projects/ees260009p/cwilson4/data/dems/COP30/COP30_reg01.vrt',
        shading_fp='/ocean/projects/ees260009p/cwilson4/data/shading/',
        ice_albedo_fn='/ocean/projects/ees260009p/cwilson4/data/ice_albedo/{gid}_albedo.tif',
        thickness_fn='/ocean/projects/ees260009p/cwilson4/data/ice_thickness/RGI60-01/RGI60-{gid}_thickness.tif',
        windmap_fn='/ocean/projects/ees260009p/cwilson4/data/windmapper/{gid}.nc',
    ),
}

# RGI IDs for the named glaciers everything else refers to by name
translate_rgi = {
                 'gulkana':{'6': '01.00570', '7':'01.05299'}, # GULKANA
                 'kahiltna':{'6':'01.22193','7':'01.04282'}, # KAHILTNA
                 'kennicott':{'6':'01.15645','7':'01.05740'}, # KENNICOTT
                 'wolverine':{'6':'01.09162','7':'01.11350'}, # WOLVERINE
                 'lemon_creek':{'6':'01.01104','7':'01.19406'}, # LEMON CREEK
                 'taku':{'6':'01.01390','7':'01.19709'}, # TAKU
                 }
translate_names = {}
for glacier, items in translate_rgi.items():
    translate_names[items['6']] = glacier

# the five glaciers AD_optimize calibrates against, reused by anything else
# that needs to run or score all of them together (loss_surface.py)
GLACIERS = ['gulkana', 'kahiltna', 'kennicott', 'wolverine', 'lemon_creek']

# kp/wind_factor starting point for calibration, and the parameter values
# the point-density rule and other sanity checks were run at
baseline = {'kp': 2.5, 'wind_factor': 2.5}

# physics settings shared with the distributed reference run (config.yaml).
# Anything that builds a config to run PEBSI on these glaciers starts from
# this rather than repeating the physics options by hand.
BASE_CONFIG = dict(
    option_ice_albedo_tif=True,
    option_windmaps=True,
    option_accel_grains=True,
    option_flat_plates=True,
    option_dynamics=False,
    constant_freshgrainsize=54.5,
    constant_irrwater=True,
    precgrad=0.000100,
    bias_vars=['temp'],
    max_nlayers=25,
)
