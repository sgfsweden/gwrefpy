"""Tests for time equivalent grouping with different aggregation methods."""

import numpy as np
import pandas as pd
import pytest

from gwrefpy import analyze_offsets
from gwrefpy.methods.timeseries import groupby_time_equivalents


@pytest.fixture
def sample_data_with_multiple_values():
    """Create sample data with multiple values per time equivalent for aggregation."""
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


# ---------------------------------------------------------------------------
# Consecutive method: aggregation tests (obs values are also grouped per window)
# ---------------------------------------------------------------------------


def test_aggregation_mean(sample_data_with_multiple_values):
    """Consecutive method: mean aggregation groups both obs and ref per window."""
    obs, ref = sample_data_with_multiple_values

    ref_ts, obs_ts, n = groupby_time_equivalents(
        obs, ref, "1D", aggregation="mean", method="consecutive"
    )

    assert n == 3
    assert len(ref_ts) == 3
    assert len(obs_ts) == 3

    assert obs_ts.iloc[0] == np.mean([10, 30])
    assert obs_ts.iloc[1] == np.mean([25, 5])
    assert obs_ts.iloc[2] == np.mean([15, 20])


def test_aggregation_median(sample_data_with_multiple_values):
    """Consecutive method: median aggregation."""
    obs, ref = sample_data_with_multiple_values

    ref_ts, obs_ts, n = groupby_time_equivalents(
        obs, ref, "1D", aggregation="median", method="consecutive"
    )

    assert n == 3
    assert len(ref_ts) == 3
    assert len(obs_ts) == 3

    assert obs_ts.iloc[0] == np.median([10, 30])
    assert obs_ts.iloc[1] == np.median([25, 5])
    assert obs_ts.iloc[2] == np.median([15, 20])


def test_aggregation_min(sample_data_with_multiple_values):
    """Consecutive method: min aggregation."""
    obs, ref = sample_data_with_multiple_values

    ref_ts, obs_ts, n = groupby_time_equivalents(
        obs, ref, "1D", aggregation="min", method="consecutive"
    )

    assert n == 3
    assert len(ref_ts) == 3
    assert len(obs_ts) == 3

    assert obs_ts.iloc[0] == np.min([10, 30])
    assert obs_ts.iloc[1] == np.min([25, 5])
    assert obs_ts.iloc[2] == np.min([15, 20])


def test_aggregation_max(sample_data_with_multiple_values):
    """Consecutive method: max aggregation."""
    obs, ref = sample_data_with_multiple_values

    ref_ts, obs_ts, n = groupby_time_equivalents(
        obs, ref, "1D", aggregation="max", method="consecutive"
    )

    assert n == 3
    assert len(ref_ts) == 3
    assert len(obs_ts) == 3

    assert obs_ts.iloc[0] == np.max([10, 30])
    assert obs_ts.iloc[1] == np.max([25, 5])
    assert obs_ts.iloc[2] == np.max([15, 20])


# ---------------------------------------------------------------------------
# Anchor method: aggregation tests (each obs time is its own group)
# ---------------------------------------------------------------------------


def test_anchor_aggregation_aggregates_ref_not_obs(sample_data_with_multiple_values):
    """Anchor method: ref values within ±offset are aggregated; obs values are not."""
    obs, ref = sample_data_with_multiple_values

    ref_ts, obs_ts, n = groupby_time_equivalents(
        obs, ref, "1D", aggregation="mean", method="anchor"
    )

    # One output pair per obs time point
    assert n == 6
    assert len(ref_ts) == 6
    assert len(obs_ts) == 6

    # Obs values are preserved as-is
    assert obs_ts.iloc[0] == 10.0
    assert obs_ts.iloc[1] == 30.0

    # Ref values are aggregated across all refs within ±1D of each obs anchor
    # Both ref points on Jan-07 fall within ±1D of any Jan-07 obs time
    assert ref_ts.iloc[0] == np.mean([8.0, 12.0])
    assert ref_ts.iloc[1] == np.mean([8.0, 12.0])


def test_anchor_method_dense_continuous_data():
    """Anchor method creates one group per obs day even with continuous ref data."""
    obs = pd.Series(
        index=[
            pd.Timestamp("2018-01-04 00:00:00"),
            pd.Timestamp("2018-01-05 00:00:00"),
            pd.Timestamp("2018-01-06 00:00:00"),
        ],
        data=[27.582109, 27.602150, 27.523178],
        name="obs",
    )
    ref_indices = []
    ref_data = []
    for day in [4, 5, 6]:
        for hour in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]:
            second = 1 if hour in (0, 2) else 0
            ref_indices.append(
                pd.Timestamp(f"2018-01-0{day} {hour:02d}:00:{second:02d}")
            )
            ref_data.append(27.0 + hour * 0.01)
    ref = pd.Series(index=ref_indices, data=ref_data, name="ref")

    ref_te, obs_te, n = groupby_time_equivalents(obs, ref, "0.5D", method="anchor")

    # One group per daily obs point — consecutive would chain everything into 1
    assert n == 3


def test_methods_differ_on_dense_continuous_data():
    """Anchor gives 3 groups; consecutive chains all into 1 for dense ref data."""
    obs = pd.Series(
        index=[
            pd.Timestamp("2018-01-04 00:00:00"),
            pd.Timestamp("2018-01-05 00:00:00"),
            pd.Timestamp("2018-01-06 00:00:00"),
        ],
        data=[27.582109, 27.602150, 27.523178],
        name="obs",
    )
    ref_indices = []
    ref_data = []
    for day in [4, 5, 6]:
        for hour in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]:
            second = 1 if hour in (0, 2) else 0
            ref_indices.append(
                pd.Timestamp(f"2018-01-0{day} {hour:02d}:00:{second:02d}")
            )
            ref_data.append(27.0 + hour * 0.01)
    ref = pd.Series(index=ref_indices, data=ref_data, name="ref")

    _, _, n_anchor = groupby_time_equivalents(obs, ref, "0.5D", method="anchor")
    _, _, n_consec = groupby_time_equivalents(obs, ref, "0.5D", method="consecutive")

    assert n_anchor == 3
    assert n_consec == 1


def test_default_method_is_anchor():
    """Default method must be anchor."""
    obs = pd.Series(
        index=[pd.Timestamp("2020-01-07"), pd.Timestamp("2020-02-01")],
        data=[11.4, 11.7],
        name="obs",
    )
    ref = pd.Series(
        index=[pd.Timestamp("2020-01-08"), pd.Timestamp("2020-02-03")],
        data=[8.9, 9.2],
        name="ref",
    )
    ref_default, obs_default, n_default = groupby_time_equivalents(obs, ref, "3.5D")
    ref_anchor, obs_anchor, n_anchor = groupby_time_equivalents(
        obs, ref, "3.5D", method="anchor"
    )

    assert n_default == n_anchor
    pd.testing.assert_series_equal(ref_default, ref_anchor)
    pd.testing.assert_series_equal(obs_default, obs_anchor)


# ---------------------------------------------------------------------------
# Shared behaviour tests
# ---------------------------------------------------------------------------


def test_default_aggregation_is_mean():
    """Default aggregation is 'mean' when not specified."""
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

    ref_ts_default, obs_ts_default, n_default = groupby_time_equivalents(obs, ref, "1D")
    ref_ts_mean, obs_ts_mean, n_mean = groupby_time_equivalents(
        obs, ref, "1D", aggregation="mean"
    )

    assert n_default == n_mean
    assert ref_ts_default.equals(ref_ts_mean)
    assert obs_ts_default.equals(obs_ts_mean)


def test_aggregation_with_single_value_groups():
    """All aggregation methods give identical results when each group has one value."""
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

    aggregations = ["mean", "median", "min", "max"]
    results = []
    for agg in aggregations:
        ref_te, obs_te, n = groupby_time_equivalents(obs, ref, "3.5D", aggregation=agg)  # type: ignore
        results.append((ref_te, obs_te, n))

    for i in range(1, len(results)):
        assert results[i][2] == results[0][2]
        assert results[i][0].equals(results[0][0])
        assert results[i][1].equals(results[0][1])


def test_aggregation_preserves_empty_results():
    """All aggregation methods return empty results when no pairs are found."""
    obs = pd.Series(
        index=[pd.Timestamp("2020-01-07"), pd.Timestamp("2020-02-01")],
        data=[11.4, 11.7],
        name="obs",
    )
    ref = pd.Series(
        index=[pd.Timestamp("2024-01-08"), pd.Timestamp("2024-02-03")],
        data=[8.9, 9.2],
        name="ref",
    )
    for agg in ["mean", "median", "min", "max"]:
        ref_te, obs_te, n = groupby_time_equivalents(obs, ref, "7D", aggregation=agg)  # type: ignore
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
    result = analyze_offsets(obs, ref, offsets)
    assert len(result) == len(offsets)
    assert result.index.tolist() == offsets
    assert result.name == "n_pairs"

    assert result.loc["0D"] == 1
    assert result.loc["3.5D"] == 3


def test_test_offsets_consecutive(strandangers_example) -> None:
    obs, ref = strandangers_example
    offsets = ["0D", "1D", "3.5D", "5D", "7D"]
    result = analyze_offsets(obs, ref, offsets, method="consecutive")
    assert result.loc["0D"] == 1
    assert result.loc["3.5D"] == 3
