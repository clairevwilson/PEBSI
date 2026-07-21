"""
Tests for layer utility functions in pebsi/physics/layers.py.

Every function that structurally modifies the column (add, remove, split,
merge, check sizes) should:
  1. Conserve mass (ice + water + reservoir).
  2. Produce a column of the correct shape.
  3. Only affect points in the mask.
  4. Leave the column in a self-consistent state (height = ice/density, etc.).
"""
import jax.numpy as jnp
import numpy as np
import pytest

import pebsi.physics.layers as layers
from conftest import _make_state

_LAYER_FIELDS = [
    'lheight', 'ldensity', 'lice', 'lwater', 'ltemp', 'ltype', 'lage',
    'lgrainsize', 'lrefreeze', 'ldrefreeze', 'lBC', 'lOC', 'ldust', 'ldepth',
]

ATOL = 0.01   # kg m-2


def _total_mass(state):
    """Ice + water in all layers + basal reservoir, per point."""
    return (jnp.sum(state.lice + state.lwater, axis=1)
            + state.basal_reservoir)


def _all_mask(n=4):
    return jnp.ones(n, dtype=bool)


def _no_mask(n=4):
    return jnp.zeros(n, dtype=bool)


def _half_mask(n=4):
    """Only the first half of points."""
    m = jnp.zeros(n, dtype=bool)
    return m.at[:n//2].set(True)


# ================================================================== #
#  add_top_layer (insert at top)                                      #
# ================================================================== #

def _new_snow_layer(n=4, lice=7.5):
    return {
        'lheight':   jnp.full(n, lice / 150.0),
        'ldensity':  jnp.full(n, 150.0),
        'lice':      jnp.full(n, lice),
        'lwater':    jnp.zeros(n),
        'ltemp':     jnp.zeros(n),
        'ltype':     jnp.zeros(n, dtype=jnp.int32),
        'lage':      jnp.zeros(n, dtype=jnp.int32),
        'lgrainsize':jnp.full(n, 200.0),
        'lrefreeze': jnp.zeros(n),
        'ldrefreeze':jnp.zeros(n),
        'lBC':       jnp.zeros(n),
        'lOC':       jnp.zeros(n),
        'ldust':     jnp.zeros(n),
        'ldepth':    jnp.zeros(n),
    }


def test_add_top_layer_mass_accounting(params):
    """add_top_layer pushes bottom to reservoir; total (layers + reservoir) grows by new layer mass."""
    state = _make_state(params)
    new_ice = 7.5
    new_layer = _new_snow_layer(lice=new_ice)
    mass_before = _total_mass(state)
    state_out = layers.add_top_layer(state, _all_mask(), new_layer)
    mass_after = _total_mass(state_out)
    np.testing.assert_allclose(mass_after, mass_before + new_ice, atol=ATOL)


def test_add_top_layer_only_affects_masked_points(params):
    """Points not in mask are unchanged."""
    state = _make_state(params)
    mask = _half_mask()
    state_out = layers.add_top_layer(state, mask, _new_snow_layer())
    unmasked = ~mask
    np.testing.assert_allclose(
        state_out.lice[unmasked, 0], state.lice[unmasked, 0], atol=ATOL
    )


def test_add_top_layer_new_layer_at_top(params):
    """After adding, the new layer's mass appears at index 0."""
    state = _make_state(params)
    new_ice = 8.0
    state_out = layers.add_top_layer(state, _all_mask(), _new_snow_layer(lice=new_ice))
    np.testing.assert_allclose(state_out.lice[:, 0], new_ice, atol=ATOL)


def test_add_top_layer_shifts_old_top_to_index_1(params):
    """The layer that was at index 0 should now be at index 1."""
    state = _make_state(params)
    old_top_ice = state.lice[:, 0]
    state_out = layers.add_top_layer(state, _all_mask(), _new_snow_layer())
    np.testing.assert_allclose(state_out.lice[:, 1], old_top_ice, atol=ATOL)


# ================================================================== #
#  remove_layer                                                        #
# ================================================================== #
#
# remove_layer is called AFTER a layer's ice has already been zeroed
# (e.g., after heating_melting empties the layer). The function's job
# is to collapse the empty slot and refill the bottom from the glacier
# reservoir. Testing it on a pre-zeroed layer matches actual usage.

def _zero_top_layer(state):
    """Zero out the top layer — simulating a fully-melted layer."""
    return state._replace(
        lice=state.lice.at[:, 0].set(0.0),
        lwater=state.lwater.at[:, 0].set(0.0),
        lheight=state.lheight.at[:, 0].set(0.0),
    )


def test_remove_layer_conserves_mass_when_empty(params):
    """Removing an already-empty layer conserves total mass (the normal call site)."""
    state = _zero_top_layer(_make_state(params))
    idx = jnp.zeros(4, dtype=jnp.int32)
    mass_before = _total_mass(state)
    state_out = layers.remove_layer(state, _all_mask(), idx, params)
    mass_after = _total_mass(state_out)
    np.testing.assert_allclose(mass_before, mass_after, atol=ATOL)


def test_remove_layer_discards_non_empty_layer_mass(params):
    """Removing a layer with mass causes that mass to leave the domain (no output flux)."""
    state = _make_state(params)
    removed_mass = state.lice[:, 0] + state.lwater[:, 0]   # 60 kg/m² snow
    mass_before = _total_mass(state)
    idx = jnp.zeros(4, dtype=jnp.int32)
    state_out = layers.remove_layer(state, _all_mask(), idx, params)
    mass_after = _total_mass(state_out)
    np.testing.assert_allclose(mass_after, mass_before - removed_mass, atol=ATOL)


def test_remove_layer_shifts_column_up(params):
    """After removing index 0, what was at index 1 is now at index 0."""
    state = _zero_top_layer(_make_state(params))
    old_idx1_ice = state.lice[:, 1]
    idx = jnp.zeros(4, dtype=jnp.int32)
    state_out = layers.remove_layer(state, _all_mask(), idx, params)
    np.testing.assert_allclose(state_out.lice[:, 0], old_idx1_ice, atol=ATOL)


def test_remove_layer_only_affects_masked_points(params):
    """Points outside the mask keep their original column."""
    state = _zero_top_layer(_make_state(params))
    mask = _half_mask()
    idx = jnp.zeros(4, dtype=jnp.int32)
    state_out = layers.remove_layer(state, mask, idx, params)
    unmasked = ~mask
    np.testing.assert_allclose(
        state_out.lice[unmasked, 0], state.lice[unmasked, 0], atol=ATOL
    )


# ================================================================== #
#  split_layer                                                         #
# ================================================================== #

def test_split_layer_conserves_mass(params):
    """Splitting pushes bottom to reservoir — total mass (incl. reservoir) unchanged."""
    state = _make_state(params)
    mass_before = _total_mass(state)
    state_out = layers.split_layer(state, _all_mask(), 0, params)
    mass_after = _total_mass(state_out)
    np.testing.assert_allclose(mass_before, mass_after, atol=ATOL)


def test_split_layer_halves_ice(params):
    """The two halves of the split layer each hold ~half the original ice."""
    state = _make_state(params)
    original_top_ice = state.lice[:, 0]
    state_out = layers.split_layer(state, _all_mask(), 0, params)
    np.testing.assert_allclose(state_out.lice[:, 0], original_top_ice / 2, atol=ATOL)
    np.testing.assert_allclose(state_out.lice[:, 1], original_top_ice / 2, atol=ATOL)


def test_split_layer_preserves_intensive_props(params):
    """Temperature and density are unchanged after splitting."""
    state = _make_state(params)
    state_out = layers.split_layer(state, _all_mask(), 0, params)
    np.testing.assert_allclose(state_out.ltemp[:, 0],    state.ltemp[:, 0],    atol=1e-6)
    np.testing.assert_allclose(state_out.ldensity[:, 0], state.ldensity[:, 0], atol=1e-6)


def test_split_layer_only_affects_masked_points(params):
    """Unmasked points are not modified."""
    state = _make_state(params)
    mask = _half_mask()
    state_out = layers.split_layer(state, mask, 0, params)
    unmasked = ~mask
    np.testing.assert_allclose(
        state_out.lice[unmasked, 0], state.lice[unmasked, 0], atol=ATOL
    )


# ================================================================== #
#  merge_existing_layers                                               #
# ================================================================== #

def test_merge_existing_conserves_mass(params):
    """Merging two layers conserves total column + reservoir mass."""
    state = _make_state(params)
    mass_before = _total_mass(state)
    state_out = layers.merge_existing_layers(state, _all_mask(), 0, params)
    mass_after = _total_mass(state_out)
    np.testing.assert_allclose(mass_before, mass_after, atol=ATOL)


def test_merge_existing_combines_ice(params):
    """Ice in merged layer equals sum of the two originals."""
    state = _make_state(params)
    combined = state.lice[:, 0] + state.lice[:, 1]
    state_out = layers.merge_existing_layers(state, _all_mask(), 0, params)
    np.testing.assert_allclose(state_out.lice[:, 0], combined, atol=ATOL)


def test_merge_existing_weighted_temperature(params):
    """Merged temperature is mass-weighted average of the two layers."""
    state = _make_state(params)
    # give the two layers different temperatures
    ltemp = state.ltemp.at[:, 0].set(-10.0).at[:, 1].set(-2.0)
    state = state._replace(ltemp=ltemp)
    m0 = state.lice[:, 0]
    m1 = state.lice[:, 1]
    expected_temp = (m0 * -10.0 + m1 * -2.0) / (m0 + m1)

    state_out = layers.merge_existing_layers(state, _all_mask(), 0, params)
    np.testing.assert_allclose(state_out.ltemp[:, 0], expected_temp, atol=1e-3)


def test_merge_existing_only_affects_masked_points(params):
    """Unmasked points keep their original layer 0 ice."""
    state = _make_state(params)
    mask = _half_mask()
    state_out = layers.merge_existing_layers(state, mask, 0, params)
    unmasked = ~mask
    np.testing.assert_allclose(
        state_out.lice[unmasked, 0], state.lice[unmasked, 0], atol=ATOL
    )


# ================================================================== #
#  check_layer_sizes                                                   #
# ================================================================== #

def test_check_layer_sizes_conserves_mass(params):
    """Merging thin layers conserves total mass (modulo min_layer_mass discards)."""
    state = _make_state(params)
    # make top layer very thin to trigger a merge
    thin_lice   = state.lice.at[:, 0].set(0.005)
    thin_lheight = state.lheight.at[:, 0].set(0.005 / 300.0)
    state = state._replace(lice=thin_lice, lheight=thin_lheight)
    mass_before = _total_mass(state)
    state_out, dead_mass = layers.check_layer_sizes(state, params)
    mass_after = _total_mass(state_out) + dead_mass
    np.testing.assert_allclose(mass_before, mass_after, atol=ATOL)


def test_check_layer_sizes_removes_thin_layers(params):
    """A layer below min_dz should be merged into the layer below it."""
    state = _make_state(params)
    thin_lice    = state.lice.at[:, 0].set(0.005)
    thin_lheight = state.lheight.at[:, 0].set(1e-4)   # well below min_dz
    state = state._replace(lice=thin_lice, lheight=thin_lheight)
    state_out, _ = layers.check_layer_sizes(state, params)
    # the very thin layer should be gone; its ice should now be in layer 0
    # (absorbed into what was layer 1)
    assert float(jnp.min(state_out.lheight[:, 0])) >= params.min_dz - ATOL


def test_check_layer_sizes_dead_mass_is_positive(params):
    """Discarded mass (below min_layer_mass) is reported as non-negative."""
    state = _make_state(params)
    # put a near-zero layer
    state = state._replace(
        lice=state.lice.at[:, 0].set(params.min_layer_mass * 0.5)
    )
    _, dead_mass = layers.check_layer_sizes(state, params)
    assert jnp.all(dead_mass >= 0.0)


# ================================================================== #
#  update_layer_props                                                  #
# ================================================================== #

def test_update_layer_props_depth_monotone(params):
    """Layer midpoint depths should be strictly increasing from top to bottom."""
    state = _make_state(params)
    state_out = layers.update_layer_props(state, params.density_ice)
    depth_diffs = jnp.diff(state_out.ldepth, axis=1)
    assert jnp.all(depth_diffs >= 0.0)


def test_update_layer_props_density_from_height(params):
    """For non-ice layers, density should equal lice / lheight."""
    state = _make_state(params)
    state_out = layers.update_layer_props(state, params.density_ice)
    active_snow = (state_out.ltype == 0) & (state_out.lheight > 0)
    safe_h = jnp.where(active_snow, state_out.lheight, 1.0)
    computed_density = state_out.lice / safe_h
    np.testing.assert_allclose(
        jnp.where(active_snow, state_out.ldensity, 0.0),
        jnp.where(active_snow, computed_density, 0.0),
        atol=1e-3
    )


def test_update_layer_props_masks_consistent(params):
    """snow_mask, firn_mask, ice_mask are mutually exclusive and exhaustive."""
    state = _make_state(params)
    state_out = layers.update_layer_props(state, params.density_ice)
    n_types = state_out.snow_mask.astype(int) + \
              state_out.firn_mask.astype(int) + \
              state_out.ice_mask.astype(int)
    assert jnp.all(n_types == 1)
