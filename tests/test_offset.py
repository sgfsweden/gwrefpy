"""Tests for time equivalent grouping with different aggregation methods."""

import numpy as np
import pandas as pd
import pytest

from gwrefpy import analyze_offsets
from gwrefpy.methods.timeseries import groupby_time_equivalents


@pytest.fixture
def sample_data_with_multiple_values():
    """Create sample data with multiple values per time equivalent for aggregation."""
    # Create observation data with multiple measurements per day
    obs = pd.Series(
        index=[
            pd.Timestamp("2020-01-07 09:00"),
            pd.Timestamp("2020-01-07 15:00"),
            pd.Timestamp("2020-02-01 11:00"),
            pd.Timestamp("2020-02-01 14:00"),
            pd.Timestamp("2020-02-25 09:00"),
            pd.Timestamp("2020-02-25 16:00"),
        ],
        data=[10.0, 30.0, 25.0, 5.0, 15.0, 20.0],
        name="obs",
    )

    # Create reference data with multiple measurements per day
    ref = pd.Series(
        index=[
            pd.Timestamp("2020-01-07 08:00"),
            pd.Timestamp("2020-01-07 14:00"),
            pd.Timestamp("2020-02-01 10:00"),
            pd.Timestamp("2020-02-01 15:00"),
            pd.Timestamp("2020-02-25 08:00"),
            pd.Timestamp("2020-02-25 17:00"),
        ],
        data=[8.0, 12.0, 11.0, 6.0, 14.0, 18.0],
        name="ref",
    )

    return obs, ref


def test_aggregation_mean(sample_data_with_multiple_values):
    """Test that mean aggregation works correctly."""
    obs, ref = sample_data_with_multiple_values

    ref_ts, obs_ts, n = groupby_time_equivalents(obs, ref, "1D", aggregation="mean")

    assert n == 3
    assert len(ref_ts) == 3
    assert len(obs_ts) == 3

    assert obs_ts.iloc[0] == np.mean([10, 30])
    assert obs_ts.iloc[1] == np.mean([25, 5])
    assert obs_ts.iloc[2] == np.mean([15, 20])


def test_aggregation_median(sample_data_with_multiple_values):
    """Test that median aggregation works correctly."""
    obs, ref = sample_data_with_multiple_values

    ref_ts, obs_ts, n = groupby_time_equivalents(obs, ref, "1D", aggregation="median")

    assert n == 3
    assert len(ref_ts) == 3
    assert len(obs_ts) == 3

    assert obs_ts.iloc[0] == np.median([10, 30])
    assert obs_ts.iloc[1] == np.median([25, 5])
    assert obs_ts.iloc[2] == np.median([15, 20])


def test_aggregation_min(sample_data_with_multiple_values):
    """Test that min aggregation works correctly."""
    obs, ref = sample_data_with_multiple_values

    ref_ts, obs_ts, n = groupby_time_equivalents(obs, ref, "1D", aggregation="min")

    assert n == 3
    assert len(ref_ts) == 3
    assert len(obs_ts) == 3

    assert obs_ts.iloc[0] == np.min([10, 30])
    assert obs_ts.iloc[1] == np.min([25, 5])
    assert obs_ts.iloc[2] == np.min([15, 20])


def test_aggregation_max(sample_data_with_multiple_values):
    """Test that max aggregation works correctly."""
    obs, ref = sample_data_with_multiple_values

    ref_ts, obs_ts, n = groupby_time_equivalents(obs, ref, "1D", aggregation="max")

    assert n == 3
    assert len(ref_ts) == 3
    assert len(obs_ts) == 3

    assert obs_ts.iloc[0] == np.max([10, 30])
    assert obs_ts.iloc[1] == np.max([25, 5])
    assert obs_ts.iloc[2] == np.max([15, 20])


def test_default_aggregation_is_mean():
    """Test that default aggregation is 'mean' when not specified."""
    obs = pd.Series(
        index=[
            pd.Timestamp("2020-01-07 09:00"),
            pd.Timestamp("2020-01-07 15:00"),
        ],
        data=[10.0, 30.0],
        name="obs",
    )
    ref = pd.Series(
        index=[
            pd.Timestamp("2020-01-07 08:00"),
            pd.Timestamp("2020-01-07 14:00"),
        ],
        data=[8.0, 12.0],
        name="ref",
    )

    # Without specifying aggregation (should use mean)
    ref_ts_default, obs_ts_default, n_default = groupby_time_equivalents(obs, ref, "1D")

    # With explicit mean aggregation
    ref_ts_mean, obs_ts_mean, n_mean = groupby_time_equivalents(
        obs, ref, "1D", aggregation="mean"
    )

    assert n_default == n_mean
    assert ref_ts_default.equals(ref_ts_mean)
    assert obs_ts_default.equals(obs_ts_mean)


def test_aggregation_with_single_value_groups():
    """Test that aggregation methods work correctly when groups have single values."""
    obs = pd.Series(
        index=[
            pd.Timestamp("2020-01-07"),
            pd.Timestamp("2020-02-01"),
            pd.Timestamp("2020-02-25"),
        ],
        data=[11.4, 11.7, 11.8],
        name="obs",
    )
    ref = pd.Series(
        index=[
            pd.Timestamp("2020-01-08"),
            pd.Timestamp("2020-02-03"),
            pd.Timestamp("2020-02-25"),
        ],
        data=[8.9, 9.2, 9.4],
        name="ref",
    )

    # All aggregation methods should give same result for single values
    aggregations = ["mean", "median", "min", "max"]
    results = []

    for agg in aggregations:
        ref_te, obs_te, n = groupby_time_equivalents(
            obs,
            ref,
            "3.5D",
            aggregation=agg,  # type: ignore
        )
        results.append((ref_te, obs_te, n))

    # All should be equal since each group has only one value
    for i in range(1, len(results)):
        assert results[i][2] == results[0][2]  # Same n
        assert results[i][0].equals(results[0][0])  # Same ref_te
        assert results[i][1].equals(results[0][1])  # Same obs_te


def test_aggregation_preserves_empty_results():
    """Test that all aggregation methods return empty results when no pairs found."""
    obs = pd.Series(
        index=[
            pd.Timestamp("2020-01-07"),
            pd.Timestamp("2020-02-01"),
        ],
        data=[11.4, 11.7],
        name="obs",
    )
    ref = pd.Series(
        index=[
            pd.Timestamp("2024-01-08"),  # Different year
            pd.Timestamp("2024-02-03"),
        ],
        data=[8.9, 9.2],
        name="ref",
    )

    aggregations = ["mean", "median", "min", "max"]

    for agg in aggregations:
        ref_te, obs_te, n = groupby_time_equivalents(
            obs,
            ref,
            "7D",
            aggregation=agg,  # type: ignore
        )
        assert n == 0
        assert len(ref_te) == 0
        assert len(obs_te) == 0


def test_strandangers_example(strandangers_example) -> None:
    obs, ref = strandangers_example

    ref_te, obs_te, n = groupby_time_equivalents(obs, ref, "3.5D")
    assert n == 3
    assert ref_te.tolist() == [8.9, 9.2, 9.4]
    assert obs_te.tolist() == [11.4, 11.7, 11.8]


def test_groupby_time_equivalents_no_pairs() -> None:
    obs = pd.Series(
        index=[
            pd.Timestamp("2020-01-07"),
            pd.Timestamp("2020-02-01"),
            pd.Timestamp("2020-02-25"),
        ],
        data=[11.4, 11.7, 11.8],
        name="obs",
    )
    ref = pd.Series(
        index=[
            pd.Timestamp("2024-01-08"),
            pd.Timestamp("2024-02-03"),
            pd.Timestamp("2024-02-08"),
            pd.Timestamp("2024-02-25"),
            pd.Timestamp("2024-02-28"),
        ],
        data=[8.9, 9.2, 9.3, 9.3, 9.5],
        name="ref",
    )
    ref_te, obs_te, n = groupby_time_equivalents(obs, ref, "7D")
    assert n == 0
    assert ref_te.tolist() == []
    assert obs_te.tolist() == []


def test_test_offsets(strandangers_example) -> None:
    obs, ref = strandangers_example
    offsets = ["0D", "1D", "3.5D", "5D", "7D"]
    result = analyze_offsets(ref, obs, offsets)
    assert len(result) == len(offsets)
    assert result.index.tolist() == offsets
    assert result.name == "n_pairs"

    assert result.loc["0D"] == 1
    assert result.loc["3.5D"] == 3
