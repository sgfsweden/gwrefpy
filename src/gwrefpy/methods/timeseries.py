from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..well import Well


def groupby_time_equivalents(
    obs_timeseries: pd.Series,
    ref_timeseries: pd.Series,
    offset: pd.DateOffset | pd.Timedelta | str,
) -> tuple[pd.Series, pd.Series, int]:
    """
    Groups the reference and observation timeseries by their time equivalents.
    Currently, this function uses the mean to aggregate within each time equivalent.

    Parameters
    ----------
    obs_timeseries: pd.Series
        The observed timeseries data.
    ref_timeseries : pd.Series
        The reference timeseries data.
    offset: pd.DateOffset | pd.Timedelta | str
        Maximum date offset to allow to group pairs of data points.

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
        obs_timeseries.name = "obs"
    if not ref_timeseries.name or ref_timeseries.name != "ref":
        ref_timeseries.name = "ref"

    time_equivalents = _create_time_equivalents(
        obs_timeseries.index, ref_timeseries.index, offset
    )

    combined = pd.concat([obs_timeseries, ref_timeseries], axis="columns")
    combined_time_eqs = combined.set_index(time_equivalents, drop=True)
    time_eq_means = combined_time_eqs.groupby(combined_time_eqs.index).mean()

    time_eq_means = time_eq_means.dropna()

    return (
        time_eq_means[ref_timeseries.name],
        time_eq_means[obs_timeseries.name],
        len(time_eq_means),
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
    ref: pd.Series | Well,
    obs: pd.Series | Well,
    offsets: Sequence[pd.DateOffset | pd.Timedelta | str],
) -> pd.Series:
    """
    Tests the grouping of time series data by different offsets. This can be helpful
    when choosing an offset.

    Parameters
    ----------
    ref: pd.Series | Well
        The reference time series data.
    obs: pd.Series | Well
        The observed time series data.
    offsets: list[pd.DateOffset | pd.Timedelta | str]
        The list of offsets to test.

    Returns
    -------
    pd.Series
        The number of grouped pairs of data points for each offset.

    """
    if isinstance(ref, Well):
        ref = ref.timeseries
    if isinstance(obs, Well):
        obs = obs.timeseries

    data = []
    idx = []
    for offset in offsets:
        _, _, n_pairs = groupby_time_equivalents(ref, obs, offset)
        data.append(n_pairs)
        if isinstance(offset, (pd.DateOffset | pd.Timedelta)):
            idx.append(str(offset))
        else:
            idx.append(offset)
    return pd.Series(index=idx, data=data, name="n_pairs")
