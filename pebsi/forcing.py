"""
Forcing expansion and elevation adjustment for PEBSI

Expands per-cell (N_UNIQUE,) climate slices to per-point (N_POINTS,)
and applies elevation-dependent corrections and per-point parameters.
All functions operate on JAX arrays and are called inside the time loop.

@author: clairevwilson
"""
import jax.numpy as jnp

def expand_forcings(forcings, point_attrs):
    """
    Expands all per-cell (N_UNIQUE,) forcing fields to (N_POINTS,)
    using point_attrs.cell_idx, then replaces the original fields.
    Fields that are already (N_POINTS,) (terrain-derived: shadow_mask,
    solar_azimuth, solar_zenith) are left untouched.
    """
    idx = point_attrs.cell_idx
    return forcings._replace(
        temp=forcings.temp[idx],
        tp=forcings.tp[idx],
        wind=forcings.wind[idx],
        winddir=forcings.winddir[idx],
        rh=forcings.rh[idx],
        sp=forcings.sp[idx],
        tcc=forcings.tcc[idx],
        shortwave_in=forcings.shortwave_in[idx],
        longwave_in=forcings.longwave_in[idx],
        bcwet=forcings.bcwet[idx],
        bcdry=forcings.bcdry[idx],
        ocwet=forcings.ocwet[idx],
        ocdry=forcings.ocdry[idx],
        dustwet=forcings.dustwet[idx],
        dustdry=forcings.dustdry[idx],
        local_hour=forcings.local_hour[idx],
    )

def adjust_temperature(forcings, point_attrs, params):
    """Applies lapse rate correction to bring temperature to point elevation."""
    lapse_rate = params.lapse_rate / 1000
    temp_elev = point_attrs.temp_elev[point_attrs.cell_idx]
    temp = forcings.temp + lapse_rate * (point_attrs.elevation - temp_elev)
    return forcings._replace(temp=temp)

def adjust_precipitation(forcings, point_attrs, params):
    """Applies elevation gradient and kp factor to precipitation."""
    elev_change = point_attrs.elevation - point_attrs.median_elev
    tp = forcings.tp * (1 + params.precgrad * (elev_change))
    return forcings._replace(tp=tp)

def adjust_pressure(forcings, point_attrs, params):
    """Adjusts surface pressure to point elevation via barometric law."""
    lapse_rate = params.lapse_rate / 1000
    CTOK = params.celsius_to_kelvin
    temp_K = forcings.temp + CTOK

    # translate temperature back to the sp elevation from the point elev
    sp_elev = point_attrs.sp_elev[point_attrs.cell_idx]
    temp_sp_K = temp_K + lapse_rate * (sp_elev - point_attrs.elevation)

    exponent = -params.gravity * params.molarmass_air / (params.R_gas * lapse_rate)
    return forcings._replace(sp=forcings.sp * (temp_K / temp_sp_K) ** exponent)

def adjust_longwave(forcings, point_attrs, params):
    """
    Adjusts incoming longwave to point elevation using the Brutsaert (1975)
    clear-sky emissivity parameterization.
    """
    lapse_rate = params.lapse_rate / 1000
    CTOK = params.celsius_to_kelvin

    # translate temperature back to the LWin elevation from the point elevation
    point_elev = point_attrs.LWin_elev[point_attrs.cell_idx]
    temp_cell = forcings.temp + lapse_rate * (point_elev - point_attrs.elevation)

    def sat_vp(t_C):
        return 610.94 * jnp.exp(17.625 * t_C / (t_C + 243.04))

    def emissivity(t_C, rh):
        e_hPa = sat_vp(t_C) * (rh / 100) / 100
        return 1.24 * (e_hPa / (t_C + CTOK)) ** (1.0 / 7.0)

    eps_site = emissivity(forcings.temp, forcings.rh)
    eps_cell = emissivity(temp_cell, forcings.rh)
    delta = (eps_site * params.sigma_SB * (forcings.temp + CTOK)**4
             - eps_cell * params.sigma_SB * (temp_cell + CTOK)**4) * params.seconds_per_hour
    return forcings._replace(longwave_in=forcings.longwave_in + delta)

def apply_parameters(forcings, params):
    """Applies per-point wind and dust scaling factors."""
    return forcings._replace(
        wind=forcings.wind * params.wind_factor,
        dustdry=forcings.dustdry * params.dust_factor,
        tp=forcings.tp * params.kp
    )

def domain_expansion(forcings, point_attrs, params):
    """
    Full pipeline: expand cell → point, then apply all elevation
    corrections and per-point parameters.
    """
    forcings = expand_forcings(forcings, point_attrs)
    forcings = adjust_temperature(forcings, point_attrs, params)
    forcings = adjust_precipitation(forcings, point_attrs, params)
    forcings = adjust_pressure(forcings, point_attrs, params)
    forcings = adjust_longwave(forcings, point_attrs, params)
    forcings = apply_parameters(forcings, params)
    return forcings
