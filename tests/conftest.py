"""
Shared fixtures for PEBSI tests.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from types import SimpleNamespace

import util.defaults as D
from pebsi.state import GlacierState, ClimateState, PointAttributes
from pebsi.massbalance import MassBalanceDriver
from pebsi.energybalance import EnergyBalanceDriver

jax.config.update("jax_enable_x64", True)

N = 4        # N_POINTS
L = 12       # N_LAYERS
N_SNOW = 3   # snow layers on top


@pytest.fixture(scope='session')
def params():
    p = SimpleNamespace(**{k: v for k, v in vars(D).items() if not k.startswith('_')})
    # broadcast scalar dynamic params to (N,) so they work per-point
    p.kp           = np.ones(N) * D.kp
    p.wind_factor  = np.ones(N) * D.wind_factor
    p.dust_factor  = np.ones(N) * D.dust_factor
    p.surftemp_guess = -5.0
    return p


def _make_state(params, surftemp=-5.0, ltemp=-2.0, lwater_frac=0.0,
                n_snow=N_SNOW, n_layers=L, n_points=N):
    """Build a minimal GlacierState with n_snow snow layers over ice."""
    ltype = jnp.concatenate([
        jnp.zeros((n_points, n_snow), dtype=jnp.int32),
        jnp.full((n_points, n_layers - n_snow), 2, dtype=jnp.int32),
    ], axis=1)

    density   = jnp.where(ltype == 0, 300.0, 900.0).astype(jnp.float64)
    lheight   = jnp.where(ltype == 0, 0.2,   5.0  ).astype(jnp.float64)
    lice      = (density * lheight).astype(jnp.float64)
    # water can only exist in porous (snow/firn) layers, not solid ice
    lwater    = jnp.where(ltype == 2, 0.0, lice * lwater_frac).astype(jnp.float64)
    ltemp_arr = jnp.full((n_points, n_layers), ltemp, dtype=jnp.float64)
    ldepth    = jnp.cumsum(lheight, axis=1) - lheight / 2.0

    return GlacierState(
        albedo                = jnp.full((n_points,), 0.7,   dtype=jnp.float64),
        albedo_surr           = jnp.full((n_points,), 0.3,   dtype=jnp.float64),
        surftemp              = jnp.full((n_points,), surftemp, dtype=jnp.float64),
        roughness             = jnp.full((n_points,), 1e-3,  dtype=jnp.float64),
        last_snow             = jnp.zeros((n_points,), dtype=jnp.int32),
        annual_firn_converted = jnp.zeros((n_points,), dtype=bool),
        annual_min_albedo     = jnp.ones((n_points, 1), dtype=jnp.float64),
        annual_max_snow       = jnp.full((n_points,), 100.0, dtype=jnp.float64),
        days_since_snowfall   = jnp.zeros((n_points,), dtype=jnp.int32),
        delayed_snow          = jnp.zeros((n_points,), dtype=jnp.float64),
        cum_mass_error        = jnp.zeros((n_points,), dtype=jnp.float64),
        basal_reservoir       = jnp.full((n_points,), 1e6,  dtype=jnp.float64),
        past_snow             = jnp.zeros((n_points, params.new_snow_days * 24), dtype=jnp.float64),
        lheight    = lheight,
        ldepth     = ldepth,
        snow_mask  = ltype == 0,
        firn_mask  = ltype == 1,
        ice_mask   = ltype == 2,
        ldensity   = density,
        ltemp      = ltemp_arr,
        ltype      = ltype,
        lage       = jnp.full((n_points, n_layers), 10, dtype=jnp.int32),
        lgrainsize = jnp.full((n_points, n_layers), 500.0, dtype=jnp.float64),
        lice       = lice,
        lwater     = lwater,
        lrefreeze  = jnp.zeros((n_points, n_layers), dtype=jnp.float64),
        ldrefreeze = jnp.zeros((n_points, n_layers), dtype=jnp.float64),
        lBC        = jnp.zeros((n_points, n_layers), dtype=jnp.float64),
        lOC        = jnp.zeros((n_points, n_layers), dtype=jnp.float64),
        ldust      = jnp.zeros((n_points, n_layers), dtype=jnp.float64),
    )


def _make_forcings(tempC=0.0, tp=0.001, wind=3.0, sp=85000.0, rh=80.0,
                   sw=500.0, lw=300.0, n_points=N):
    """Build a minimal one-timestep ClimateState (scalars for time, N for space)."""
    return ClimateState(
        time_idx     = jnp.array(0, dtype=jnp.int32),
        year         = jnp.array(2020, dtype=jnp.int32),
        month        = jnp.array(7, dtype=jnp.int32),
        day          = jnp.array(1, dtype=jnp.int32),
        hour         = jnp.array(12, dtype=jnp.int32),
        local_hour   = jnp.full((n_points,), 12, dtype=jnp.int32),
        doy          = jnp.array(183, dtype=jnp.int32),
        temp         = jnp.full((n_points,), tempC, dtype=jnp.float64),
        tp           = jnp.full((n_points,), tp,    dtype=jnp.float64),
        wind         = jnp.full((n_points,), wind,  dtype=jnp.float64),
        winddir      = jnp.zeros((n_points,), dtype=jnp.float64),
        sp           = jnp.full((n_points,), sp,    dtype=jnp.float64),
        rh           = jnp.full((n_points,), rh,    dtype=jnp.float64),
        tcc          = jnp.full((n_points,), 0.5,   dtype=jnp.float64),
        shortwave_in = jnp.full((n_points,), sw * 3600.0, dtype=jnp.float64),
        longwave_in  = jnp.full((n_points,), lw * 3600.0, dtype=jnp.float64),
        shadow_mask  = jnp.ones((n_points,), dtype=bool),
        solar_azimuth= jnp.full((n_points,), 0.5,  dtype=jnp.float64),
        solar_zenith = jnp.full((n_points,), 0.8,  dtype=jnp.float64),
        bcdry        = jnp.zeros((n_points,), dtype=jnp.float64),
        bcwet        = jnp.zeros((n_points,), dtype=jnp.float64),
        ocdry        = jnp.zeros((n_points,), dtype=jnp.float64),
        ocwet        = jnp.zeros((n_points,), dtype=jnp.float64),
        dustdry      = jnp.zeros((n_points,), dtype=jnp.float64),
        dustwet      = jnp.zeros((n_points,), dtype=jnp.float64),
    )


def _make_point_attrs(n_points=N):
    # one unique cell shared by all points
    return PointAttributes(
        latitude       = jnp.full((n_points,), 63.0,   dtype=jnp.float64),
        longitude      = jnp.full((n_points,), -145.0, dtype=jnp.float64),
        elevation      = jnp.full((n_points,), 1500.0, dtype=jnp.float64),
        slope          = jnp.zeros((n_points,), dtype=jnp.float64),
        aspect         = jnp.zeros((n_points,), dtype=jnp.float64),
        sky_view_factor= jnp.ones((n_points,),  dtype=jnp.float64),
        median_elev    = jnp.full((n_points,), 1500.0, dtype=jnp.float64),
        cell_idx       = jnp.zeros((n_points,), dtype=jnp.int32),
        gcm_elev       = jnp.array([1500.0],    dtype=jnp.float64),
        temp_elev      = jnp.array([1500.0],    dtype=jnp.float64),
        sp_elev        = jnp.array([1500.0],    dtype=jnp.float64),
        LWin_elev      = jnp.array([1500.0],    dtype=jnp.float64),
    )


@pytest.fixture
def state(params):
    return _make_state(params)


@pytest.fixture
def wet_state(params):
    """Column with liquid water already in snow layers (for percolation/refreezing)."""
    return _make_state(params, lwater_frac=0.05)


@pytest.fixture
def melting_state(params):
    """Surface at 0C, ready to melt."""
    return _make_state(params, surftemp=0.0, ltemp=0.0)


@pytest.fixture
def forcings():
    return _make_forcings()


@pytest.fixture
def point_attrs():
    return _make_point_attrs()


@pytest.fixture
def mb(params):
    return MassBalanceDriver(params)


@pytest.fixture
def eb(params):
    return EnergyBalanceDriver(params)


def total_solid_liquid(state):
    """Total mass (ice + water) summed over all layers, per point."""
    return jnp.sum(state.lice + state.lwater, axis=1)
