"""
Physical plausibility tests for PEBSI.

These check that individual functions obey basic physical constraints
(sign conventions, bounds, direction of fluxes, etc.) independent of
mass conservation.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import _make_state, _make_forcings, _make_point_attrs


# ------------------------------------------------------------------ #
#  energy balance — surface temperature                               #
# ------------------------------------------------------------------ #

def test_surftemp_always_leq_zero(eb, params, point_attrs):
    """Energy balance root-finder must never return surftemp > 0."""
    state = _make_state(params, surftemp=-3.0)
    forcings = _make_forcings(tempC=5.0, sw=800.0)   # warm sunny day
    state_out, _ = eb.solve_energy_balance(state, forcings, point_attrs)
    assert jnp.all(state_out.surftemp <= 0.0 + 1e-6)


def test_surftemp_equals_zero_when_melting(eb, params, point_attrs):
    """When net energy at T=0 is positive (melting), surftemp is pinned to 0."""
    state = _make_state(params, surftemp=0.0, ltemp=0.0)
    forcings = _make_forcings(tempC=10.0, sw=1000.0, rh=50.0)
    state_out, fluxes = eb.solve_energy_balance(state, forcings, point_attrs)
    melting_mask = fluxes['melt_energy'] > 0
    # wherever melt_energy > 0, surftemp must be 0
    assert jnp.all(jnp.where(melting_mask, state_out.surftemp == 0.0, True))


def test_surftemp_colder_when_more_radiation_lost(eb, params, point_attrs):
    """Reducing incoming SW should produce equal or lower surface temperature."""
    state = _make_state(params, surftemp=-5.0)
    f_high = _make_forcings(sw=600.0, tempC=-2.0)
    f_low  = _make_forcings(sw=100.0, tempC=-2.0)
    s_high, _ = eb.solve_energy_balance(state, f_high, point_attrs)
    s_low,  _ = eb.solve_energy_balance(state, f_low,  point_attrs)
    assert jnp.all(s_high.surftemp >= s_low.surftemp - 0.1)


# ------------------------------------------------------------------ #
#  energy balance — individual flux signs                             #
# ------------------------------------------------------------------ #

def test_longwave_out_always_negative(eb, params):
    """LWout is emitted radiation — always a negative flux."""
    for temp in [-20.0, -5.0, 0.0]:
        state = _make_state(params, surftemp=temp)
        forcings = _make_forcings()
        LWin, LWout = eb.get_LW(state, forcings)
        assert jnp.all(LWout < 0), f'LWout positive at surftemp={temp}'


def test_longwave_out_increases_with_surftemp(eb, params):
    """Warmer surface emits more longwave (Stefan-Boltzmann)."""
    state_cold = _make_state(params, surftemp=-20.0)
    state_warm = _make_state(params, surftemp=-1.0)
    forcings = _make_forcings()
    _, LWout_cold = eb.get_LW(state_cold, forcings)
    _, LWout_warm = eb.get_LW(state_warm, forcings)
    # LWout is negative, so warm surface is more negative (greater emission)
    assert jnp.all(LWout_warm < LWout_cold)


def test_sensible_heat_direction(eb, params, point_attrs):
    """Sensible heat is positive (into surface) when air is warmer than surface."""
    state = _make_state(params, surftemp=-10.0)
    forcings = _make_forcings(tempC=2.0, wind=4.0)
    Qs, _ = eb.get_turbulent(state, forcings, point_attrs)
    assert jnp.all(Qs > 0)


def test_sensible_heat_negative_when_surface_warmer(eb, params, point_attrs):
    """Sensible heat is negative (out of surface) when surface is warmer than air."""
    # surface at 0C, air at -10C
    state = _make_state(params, surftemp=0.0)
    forcings = _make_forcings(tempC=-10.0, wind=4.0)
    Qs, _ = eb.get_turbulent(state, forcings, point_attrs)
    assert jnp.all(Qs < 0)


def test_sensible_heat_zero_at_zero_wind(eb, params, point_attrs):
    """No turbulent exchange when wind speed is zero."""
    state = _make_state(params, surftemp=-5.0)
    forcings = _make_forcings(tempC=2.0, wind=0.0)
    Qs, Ql = eb.get_turbulent(state, forcings, point_attrs)
    np.testing.assert_allclose(Qs, 0.0, atol=1e-6)
    np.testing.assert_allclose(Ql, 0.0, atol=1e-6)


def test_shortwave_zero_when_shaded(eb, params, point_attrs):
    """Shadowed points receive no direct SW — only diffuse + terrain."""
    state = _make_state(params, surftemp=-5.0)
    forcings = _make_forcings(sw=800.0)
    forcings_shaded = forcings._replace(shadow_mask=jnp.zeros((4,), dtype=bool))
    SWin_sunny,  _ = eb.get_SW(state, forcings,        point_attrs)
    SWin_shaded, _ = eb.get_SW(state, forcings_shaded, point_attrs)
    assert jnp.all(SWin_sunny >= SWin_shaded)


def test_rain_flux_zero_when_no_precip(eb, params):
    """Rain heat flux is zero when there is no precipitation."""
    state = _make_state(params, surftemp=-2.0)
    forcings = _make_forcings(tp=0.0)
    Qp = eb.get_rain(state, forcings)
    np.testing.assert_allclose(Qp, 0.0, atol=1e-9)


def test_rain_flux_positive_when_warm_rain(eb, params):
    """Warm rain (air warmer than surface) brings heat to surface."""
    state = _make_state(params, surftemp=-5.0)
    forcings = _make_forcings(tempC=5.0, tp=0.01)
    Qp = eb.get_rain(state, forcings)
    assert jnp.all(Qp > 0)


# ------------------------------------------------------------------ #
#  mass balance — physical constraints                                #
# ------------------------------------------------------------------ #

def test_melt_only_when_energy_positive(mb, params):
    """Layers should not melt when total energy input is negative."""
    state = _make_state(params, ltemp=-5.0)
    fluxes = {
        'melt_energy':       jnp.full((4,), -100.0),
        'SWnet_penetrating': jnp.zeros((4,)),
    }
    state_out, layermelt, _ = mb.heating_melting(state, fluxes)
    np.testing.assert_allclose(layermelt, 0.0, atol=1e-9)


def test_refreezing_not_above_freezing(mb, params):
    """No refreezing in layers that are already at or above 0C."""
    state = _make_state(params, ltemp=0.0, lwater_frac=0.1)
    ice_before = state.lice
    state_out = mb.refreezing(state)
    ice_gained = state_out.lice - ice_before
    np.testing.assert_allclose(ice_gained, 0.0, atol=1e-9)


def test_density_stays_physical(mb, params):
    """Layer density stays between fresh snow density and ice density after densification."""
    state = _make_state(params)
    state_out, _ = mb.densification(state)
    rho = state_out.ldensity
    active = state_out.lice > params.min_layer_mass
    assert jnp.all(jnp.where(active, rho >= params.density_fresh_snow * 0.9, True))
    assert jnp.all(jnp.where(active, rho <= params.density_ice + 1.0, True))


def test_layer_height_positive(mb, params):
    """All active layers should have positive height after heating."""
    state = _make_state(params, ltemp=0.0)
    fluxes = {
        'melt_energy':       jnp.full((4,), 300.0),
        'SWnet_penetrating': jnp.zeros((4,)),
    }
    state_out, _, _ = mb.heating_melting(state, fluxes)
    active = state_out.lice > params.min_layer_mass
    assert jnp.all(jnp.where(active, state_out.lheight > 0, True))


def test_no_water_in_cold_layers_after_refreezing(mb, params):
    """Any layer still below 0C after refreezing must have no water left."""
    state = _make_state(params, ltemp=-15.0, lwater_frac=0.05)
    state_out = mb.refreezing(state)
    # if a layer is still cold after refreezing, all water must have frozen
    still_cold = state_out.ltemp < -1e-3
    residual_water = jnp.where(still_cold, state_out.lwater, 0.0)
    np.testing.assert_allclose(residual_water, 0.0, atol=1e-6)
