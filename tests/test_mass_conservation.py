"""
Mass conservation tests for PEBSI.

Each test isolates one process and checks that mass is neither
created nor destroyed. The tolerance is set to account for the
min_layer_mass threshold at which layers are discarded (~0.001 kg m-2).
"""
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import _make_state, _make_forcings, total_solid_liquid

ATOL = 0.001  # kg m-2; covers min_layer_mass discards


# ------------------------------------------------------------------ #
#  heating_melting                                                     #
# ------------------------------------------------------------------ #

def test_heating_melting_no_net_creation(mb, params):
    """Ice + water + routed meltwater equals initial ice + water."""
    state = _make_state(params, ltemp=-1.0)
    melt_energy = jnp.full((4,), 200.0)   # W m-2, enough to melt surface
    fluxes = {
        'melt_energy': melt_energy,
        'SWnet_penetrating': jnp.zeros((4,)),
    }
    mass_before = total_solid_liquid(state)
    state_out, _, mass_to_route = mb.heating_melting(state, fluxes)
    routed = jnp.sum(mass_to_route['meltwater'], axis=1)
    mass_after = total_solid_liquid(state_out) + routed
    np.testing.assert_allclose(mass_before, mass_after, atol=ATOL)


def test_heating_melting_no_melt_when_frozen(mb, params):
    """No melt when energy is negative (surface is cooling)."""
    state = _make_state(params, ltemp=-10.0)
    fluxes = {
        'melt_energy': jnp.full((4,), -50.0),
        'SWnet_penetrating': jnp.zeros((4,)),
    }
    state_out, layermelt, _ = mb.heating_melting(state, fluxes)
    assert float(jnp.sum(layermelt)) == 0.0


def test_heating_melting_melt_bounded_by_ice(mb, params):
    """Actual melt never exceeds available ice mass."""
    state = _make_state(params, ltemp=0.0)
    fluxes = {
        'melt_energy': jnp.full((4,), 1e8),   # absurdly large energy
        'SWnet_penetrating': jnp.zeros((4,)),
    }
    state_out, layermelt, _ = mb.heating_melting(state, fluxes)
    # layermelt is ice->water conversion; mass_to_route is that same water leaving the
    # domain — don't double-count
    ice_before = jnp.sum(state.lice, axis=1)
    assert jnp.all(jnp.sum(layermelt, axis=1) <= ice_before + ATOL)


def test_heating_melting_lheight_shrinks(mb, params):
    """Layer height decreases after ice is lost (density fixed)."""
    state = _make_state(params, ltemp=0.0)
    fluxes = {
        'melt_energy': jnp.full((4,), 500.0),
        'SWnet_penetrating': jnp.zeros((4,)),
    }
    total_height_before = jnp.sum(state.lheight, axis=1)
    state_out, _, _ = mb.heating_melting(state, fluxes)
    total_height_after = jnp.sum(state_out.lheight, axis=1)
    assert jnp.all(total_height_after <= total_height_before + 1e-6)


# ------------------------------------------------------------------ #
#  percolation                                                         #
# ------------------------------------------------------------------ #

def test_percolation_conserves_mass(mb, params):
    """Water before + rainfall = water remaining in layers + runoff."""
    state = _make_state(params, ltemp=0.0, lwater_frac=0.1)
    rainfall = jnp.full((4,), 5.0)
    fluxes = {
        'rainfall': rainfall,
        'meltwater': jnp.zeros((4,)),
    }
    water_before = jnp.sum(state.lwater, axis=1)
    state_out, runoff, _ = mb.percolation(state, fluxes)
    water_after = jnp.sum(state_out.lwater, axis=1)
    np.testing.assert_allclose(water_before + rainfall, water_after + runoff, atol=ATOL)


def test_percolation_water_moves_down_only(mb, params):
    """Water in the top layer should not increase above what was added."""
    state = _make_state(params, ltemp=0.0, lwater_frac=0.05)
    fluxes = {
        'rainfall': jnp.zeros((4,)),
        'meltwater': jnp.zeros((4,)),
    }
    top_water_before = state.lwater[:, 0]
    state_out, _, _ = mb.percolation(state, fluxes)
    top_water_after = state_out.lwater[:, 0]
    assert jnp.all(top_water_after <= top_water_before + ATOL)


# ------------------------------------------------------------------ #
#  refreezing                                                          #
# ------------------------------------------------------------------ #

def test_refreezing_conserves_mass(mb, params):
    """Refreezing converts water to ice with no net mass change."""
    # cold layers with some water
    state = _make_state(params, ltemp=-5.0, lwater_frac=0.1)
    mass_before = total_solid_liquid(state)
    state_out = mb.refreezing(state)
    mass_after = total_solid_liquid(state_out)
    np.testing.assert_allclose(mass_before, mass_after, atol=ATOL)


def test_refreezing_only_at_cold_layers(mb, params):
    """No refreezing when all layers are at 0C."""
    state = _make_state(params, ltemp=0.0, lwater_frac=0.1)
    water_before = jnp.sum(state.lwater, axis=1)
    state_out = mb.refreezing(state)
    water_after = jnp.sum(state_out.lwater, axis=1)
    np.testing.assert_allclose(water_before, water_after, atol=ATOL)


def test_refreezing_limited_by_water(mb, params):
    """Ice gained never exceeds water available."""
    state = _make_state(params, ltemp=-20.0, lwater_frac=0.05)
    water_before = state.lwater
    state_out = mb.refreezing(state)
    ice_gained = state_out.lice - state.lice
    assert jnp.all(ice_gained <= water_before + ATOL)


# ------------------------------------------------------------------ #
#  phase_changes                                                       #
# ------------------------------------------------------------------ #

def test_sublimation_removes_ice(mb, params):
    """Negative latent heat on frozen surface removes ice from top layer."""
    state = _make_state(params, surftemp=-5.0)
    latent_heat = jnp.full((4,), -50.0)   # W m-2, outgoing (sublimation)
    ice_before = state.lice[:, 0]
    state_out, _, mass_fluxes = mb.phase_changes(state, latent_heat)
    ice_after = state_out.lice[:, 0]
    assert jnp.all(ice_after <= ice_before + ATOL)
    assert jnp.all(mass_fluxes['sublimation'] >= 0.0)


def test_deposition_adds_ice(mb, params):
    """Positive latent heat on frozen surface adds ice to top layer."""
    state = _make_state(params, surftemp=-5.0)
    latent_heat = jnp.full((4,), 50.0)    # W m-2, incoming (deposition)
    ice_before = state.lice[:, 0]
    state_out, _, mass_fluxes = mb.phase_changes(state, latent_heat)
    ice_after = state_out.lice[:, 0]
    assert jnp.all(ice_after >= ice_before - ATOL)
    assert jnp.all(mass_fluxes['deposition'] >= 0.0)


def test_phase_changes_mass_flux_tracks_delta(mb, params):
    """Mass flux reported == actual change in layer mass."""
    state = _make_state(params, surftemp=-3.0)
    latent_heat = jnp.full((4,), -30.0)
    ice_before = jnp.sum(state.lice, axis=1)
    state_out, condensation_runoff, mass_fluxes = mb.phase_changes(state, latent_heat)
    ice_after = jnp.sum(state_out.lice, axis=1)
    reported = mass_fluxes['sublimation'] - mass_fluxes['deposition']
    actual_loss = ice_before - ice_after
    np.testing.assert_allclose(actual_loss, reported, atol=ATOL)


# ------------------------------------------------------------------ #
#  densification                                                       #
# ------------------------------------------------------------------ #

def test_densification_conserves_ice_mass(mb, params):
    """Densification compacts layers but does not create or destroy ice."""
    state = _make_state(params)
    ice_before = jnp.sum(state.lice, axis=1)
    state_out, _ = mb.densification(state)
    ice_after = jnp.sum(state_out.lice, axis=1)
    np.testing.assert_allclose(ice_before, ice_after, atol=ATOL)


def test_densification_increases_density(mb, params):
    """Snow density should be >= initial density after compaction."""
    state = _make_state(params)
    snow_density_before = state.ldensity[:, 0]   # top snow layer
    state_out, _ = mb.densification(state)
    snow_density_after = state_out.ldensity[:, 0]
    assert jnp.all(snow_density_after >= snow_density_before - 1.0)


def test_densification_height_decreases(mb, params):
    """Total column height should not increase after densification."""
    state = _make_state(params)
    height_before = jnp.sum(state.lheight, axis=1)
    state_out, _ = mb.densification(state)
    height_after = jnp.sum(state_out.lheight, axis=1)
    assert jnp.all(height_after <= height_before + 1e-4)


# ------------------------------------------------------------------ #
#  add_accumulation                                                    #
# ------------------------------------------------------------------ #

def test_accumulation_adds_mass(mb, params):
    """Snowfall increases total ice mass by the snowfall amount."""
    state = _make_state(params)
    forcings = _make_forcings(tempC=-5.0, tp=0.005)    # guaranteed snow
    snowfall, rainfall = mb.get_precip_amounts(forcings)  # kg m-2
    mass_before = jnp.sum(state.lice, axis=1)
    _, state_out = mb.add_accumulation(snowfall, rainfall, state, forcings)
    mass_after = jnp.sum(state_out.lice, axis=1)
    np.testing.assert_allclose(mass_after - mass_before, snowfall, atol=ATOL)


# ------------------------------------------------------------------ #
#  full vertical pipeline                                              #
# ------------------------------------------------------------------ #

def test_run_vertical_processes_conserves_mass(mb, params):
    """Net mass balance of the full vertical pipeline equals reported fluxes."""
    state = _make_state(params, ltemp=0.0, lwater_frac=0.02)
    fluxes = {
        'rainfall':          jnp.full((4,), 1.0),
        'snowfall':          jnp.zeros((4,)),
        'melt_energy':       jnp.full((4,), 100.0),
        'SWnet_penetrating': jnp.zeros((4,)),
        'latent_heat':       jnp.full((4,), -10.0),
        'BC':                jnp.zeros((4,)),
        'OC':                jnp.zeros((4,)),
        'dust':              jnp.zeros((4,)),
    }
    mass_before = total_solid_liquid(state) + state.basal_reservoir
    forcings = _make_forcings()
    state_out, mass_fluxes = mb.run_vertical_processes(state, forcings, fluxes)
    mass_after = total_solid_liquid(state_out) + state_out.basal_reservoir

    mass_in  = fluxes['rainfall'] + mass_fluxes.get('condensation', 0) \
                                  + mass_fluxes.get('deposition', 0)
    mass_out = mass_fluxes['runoff'] + mass_fluxes.get('sublimation', 0) \
                                     + mass_fluxes.get('evaporation', 0)
    expected_after = mass_before + mass_in - mass_out
    np.testing.assert_allclose(mass_after, expected_after, atol=0.1)
