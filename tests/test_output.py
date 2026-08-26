"""
Tests for output-resolution aggregation (output_freq: hourly/daily/monthly).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from collections import namedtuple

import jax.numpy as jnp
import numpy as np
import pytest

from pebsi.state import OUTPUT_GROUPS, AGG_METHOD


def test_agg_method_covers_every_output_group_field():
    """Every field reachable through store_vars must have an aggregation rule,
    so a newly added output variable can't silently fall back to a default."""
    all_fields = {f for fields in OUTPUT_GROUPS.values() for f in fields}
    missing = all_fields - set(AGG_METHOD)
    assert not missing, f'Fields missing from AGG_METHOD: {sorted(missing)}'


def test_agg_method_values_are_valid():
    valid = {'sum', 'mean', 'last'}
    invalid = {f: m for f, m in AGG_METHOD.items() if m not in valid}
    assert not invalid, f'Invalid aggregation methods: {invalid}'


def _aggregate_period(hourly_records, agg_method=AGG_METHOD):
    """Mirrors pebsi.main.main()'s aggregate_period closure, so the
    AGG_METHOD contract is exercised the same way it is inside the scan."""
    agg = {}
    for field in hourly_records._fields:
        vals = getattr(hourly_records, field)
        method = agg_method.get(field, 'last')
        if method == 'sum':
            agg[field] = jnp.sum(vals, axis=0)
        elif method == 'mean':
            agg[field] = jnp.mean(vals, axis=0)
        else:
            agg[field] = vals[-1]
    return type(hourly_records)(**agg)


@pytest.fixture
def hourly_records():
    """24 hourly rows x 3 points for a representative field from each
    aggregation category: a flux (melt), a cumulative total (total_mass),
    a state snapshot (surftemp), and a categorical code (surftype)."""
    Records = namedtuple('Records', ['melt', 'total_mass', 'surftemp', 'surftype'])
    hours = jnp.arange(24, dtype=jnp.float64)[:, None]  # (24, 1)
    ones = jnp.ones((24, 3), dtype=jnp.float64)
    return Records(
        melt=ones * 2.0,                                   # 2.0 m w.e. melt every hour
        total_mass=(1000.0 - hours) * ones,                # decreasing running total
        surftemp=(-10.0 + hours) * ones,                    # warms through the day
        surftype=jnp.where(hours < 12, 0, 1).astype(jnp.int32) * jnp.ones((24, 3), dtype=jnp.int32),
    )


def test_flux_variable_is_summed(hourly_records):
    period = _aggregate_period(hourly_records)
    np.testing.assert_allclose(np.array(period.melt), 2.0 * 24)


def test_cumulative_variable_takes_last_value(hourly_records):
    period = _aggregate_period(hourly_records)
    np.testing.assert_allclose(np.array(period.total_mass), 1000.0 - 23)


def test_state_variable_is_averaged(hourly_records):
    period = _aggregate_period(hourly_records)
    expected = np.mean(-10.0 + np.arange(24))
    np.testing.assert_allclose(np.array(period.surftemp), expected)


def test_categorical_variable_takes_last_value(hourly_records):
    period = _aggregate_period(hourly_records)
    assert np.all(np.array(period.surftype) == 1)
