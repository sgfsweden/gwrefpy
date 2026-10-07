from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

from ..well import Well


def groupby_time_equivalents(
    obs_timeseries: pd.Series,
    ref_timeseries: pd.Series,
    offset: pd.DateOffset | pd.Timedelta | str,
    aggregation: Literal["mean", "median", "min", "max"] = "mean",
    method: Literal["anchor", "consecutive"] = "anchor",
) -> tuple[pd.Series, pd.Series, int]:
    """
    Groups the reference and observation timeseries by their time equivalents.

    Parameters
    ----------
    obs_timeseries: pd.Series
        The observed timeseries data.
    ref_timeseries : pd.Series
        The reference timeseries data.
    offset: pd.DateOffset | pd.Timedelta | str
        Maximum date offset to allow for matching reference points to each observation
        point.
    aggregation: Literal["mean", "median", "min", "max"], optional
        The aggregation method to use when grouping data points within time equivalents.
        Default is "mean".
    method: Literal["anchor", "consecutive"], optional
        The grouping method to use.

        ``"anchor"`` (default): each observation time point is the center of a
        ``±offset`` window; all reference points inside that window are aggregated into
        one output pair per observation time.

        ``"consecutive"``: all timestamps from both series are merged and sorted, then
        split into groups wherever the gap between consecutive timestamps exceeds
        ``offset``. Both reference and observation values within each group are
        aggregated together.

    Returns
    -------
    pd.Series
        Reference time series data grouped by their time equivalents.
    pd.Series
        Observed time series data grouped by their time equivalents.
    int
        Number of grouped pairs of data points.
    """
    if not obs_timeseries.name or obs_timeseries.name != "obs":
        obs_timeseries = obs_timeseries.rename("obs")
    if not ref_timeseries.name or ref_timeseries.name != "ref":
        ref_timeseries = ref_timeseries.rename("ref")

    if isinstance(offset, str):
        offset_td = pd.Timedelta(offset)
    elif isinstance(offset, pd.DateOffset):
        offset_td = pd.Timedelta(days=offset.days if hasattr(offset, "days") else 1)
    else:
        offset_td = offset

    if method == "anchor":
        return _anchor_method(obs_timeseries, ref_timeseries, offset_td, aggregation)
    elif method == "consecutive":
        return _consecutive_method(obs_timeseries, ref_timeseries, offset, aggregation)
    else:
        raise ValueError(
            f"Unknown method: {method!r}. Must be 'anchor' or 'consecutive'."
        )


def _anchor_method(
    obs_timeseries: pd.Series,
    ref_timeseries: pd.Series,
    offset_td: pd.Timedelta,
    aggregation: str,
) -> tuple[pd.Series, pd.Series, int]:
    if obs_timeseries.empty or ref_timeseries.empty:
        return (
            pd.Series([], dtype=float, name="ref"),
            pd.Series([], dtype=float, name="obs"),
            0,
        )

    obs_arr = obs_timeseries.index.values.astype("int64")
    ref_arr = ref_timeseries.index.values.astype("int64")
    offset_ns = int(offset_td.total_seconds() * 1e9)

    # (n_obs × n_ref) boolean mask of matching pairs
    diff = np.abs(obs_arr[:, None] - ref_arr[None, :])
    mask = diff <= offset_ns

    valid = mask.any(axis=1)
    if not valid.any():
        return (
            pd.Series([], dtype=float, name="ref"),
            pd.Series([], dtype=float, name="obs"),
            0,
        )

    # Replace non-matching ref values with NaN, then aggregate only matched rows
    ref_vals = ref_timeseries.values[None, :].astype(float)
    masked_valid = np.where(mask[valid], ref_vals, np.nan)
    agg_fn = {
        "mean": np.nanmean,
        "median": np.nanmedian,
        "min": np.nanmin,
        "max": np.nanmax,
    }[aggregation]
    agg_vals = agg_fn(masked_valid, axis=1)

    ref_series = pd.Series(agg_vals, index=obs_timeseries.index[valid], name="ref")
    obs_series = obs_timeseries.iloc[np.where(valid)[0]].rename("obs")
    return ref_series, obs_series, int(valid.sum())


def _consecutive_method(
    obs_timeseries: pd.Series,
    ref_timeseries: pd.Series,
    offset: pd.DateOffset | pd.Timedelta | str,
    aggregation: str,
) -> tuple[pd.Series, pd.Series, int]:
    time_equivalents = _create_time_equivalents(
        obs_timeseries.index, ref_timeseries.index, offset
    )
    combined = pd.concat([obs_timeseries, ref_timeseries], axis="columns")
    combined_time_eqs = combined.set_index(time_equivalents, drop=True)
    grouped = combined_time_eqs.groupby(combined_time_eqs.index)
    time_eq_aggregated = getattr(grouped, aggregation)().dropna()
    return (
        time_eq_aggregated[ref_timeseries.name],
        time_eq_aggregated[obs_timeseries.name],
        len(time_eq_aggregated),
    )


def _create_time_equivalents(
    ref_index: pd.DatetimeIndex,
    obs_index: pd.DatetimeIndex,
    offset: pd.DateOffset | pd.Timedelta | str,
) -> pd.Series:
    timestamps = ref_index.union(obs_index).to_series().sort_index().index
    ts_diffs = timestamps.diff()
    starts = ts_diffs > offset
    starts[0] = True
    return pd.Series(index=timestamps, data=np.cumsum(starts), name="time_equivalents")


def analyze_offsets(
    obs: pd.Series | Well,
    ref: pd.Series | Well,
    offsets: Sequence[pd.DateOffset | pd.Timedelta | str],
    method: Literal["anchor", "consecutive"] = "anchor",
) -> pd.Series:
    """
    Tests the grouping of time series data by different offsets. This can be helpful
    when choosing an offset.

    Parameters
    ----------
    obs: pd.Series | Well
        The observed time series data.
    ref: pd.Series | Well
        The reference time series data.
    offsets: list[pd.DateOffset | pd.Timedelta | str]
        The list of offsets to test.
    method: Literal["anchor", "consecutive"], optional
        The grouping method to use (default is "anchor").

    Returns
    -------
    pd.Series
        The number of grouped pairs of data points for each offset.
    """
    if isinstance(obs, Well):
        obs = obs.timeseries
    if isinstance(ref, Well):
        ref = ref.timeseries

    data = []
    idx = []
    for offset in offsets:
        _, _, n_pairs = groupby_time_equivalents(obs, ref, offset, method=method)
        data.append(n_pairs)
        if isinstance(offset, pd.DateOffset | pd.Timedelta):
            idx.append(str(offset))
        else:
            idx.append(offset)
    return pd.Series(index=idx, data=data, name="n_pairs")
