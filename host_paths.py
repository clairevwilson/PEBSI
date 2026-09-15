"""
Machine-dependent filepaths for PEBSI.

Kept apart from AD_optimize so analysis and plotting scripts can find
the data without importing the model, and with it JAX and CUDA.
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
