import logging

import numpy as np
import pandas as pd
import scipy as sp

from ..fitresults import FitResultData, LinRegResult
from ..methods.common import _get_gwrefs_stats, compute_residual_std_error
from ..methods.timeseries import groupby_time_equivalents
from ..well import Well

logger = logging.getLogger(__name__)


def linregressfit(
    obs_well: Well,
    ref_well: Well,
    offset: pd.DateOffset | pd.Timedelta | str,
    tmin: pd.Timestamp | str | None = None,
    tmax: pd.Timestamp | str | None = None,
    name: str | None = None,
    p=0.95,
    aggregation="mean",
):
    """
    Perform linear regression fit between reference and observation well time series.

    Parameters
    ----------
    obs_well : Well
        The observation well object containing the time series data.
    ref_well : Well
        The reference well object containing the time series data.
    offset: pd.DateOffset | pd.Timedelta | str
        The offset to apply when grouping the time series into time equivalents.
    tmin: pd.Timestamp | str | None = None
        The minimum timestamp for the calibration period.
    tmax: pd.Timestamp | str | None = None
        The maximum timestamp for the calibration period.
    name: str | None = None
        An optional name for the fit result.
    p : float, optional
        The confidence level for the prediction interval (default is 0.95).
    aggregation : str, optional
        The aggregation method to use when grouping data points within time
        equivalents (default is "mean"). Can be "mean", "median", "min", or "max".

    Returns
    -------
    fit_result : FitResultData
        A `FitResultData` object containing the results of the linear regression fit.
    """

    # Groupby time equivalents with given offset
    if ref_well.timeseries is None or obs_well.timeseries is None:
        logger.critical("Missing time series data for for either ref or obs well")
        return None

    ref_timeseries, obs_timeseries, n = groupby_time_equivalents(
        obs_well.timeseries.loc[tmin:tmax],
        ref_well.timeseries.loc[tmin:tmax],
        offset,
        aggregation,
    )

    res = sp.stats.linregress(ref_timeseries, obs_timeseries)
    linreg = LinRegResult(
        slope=res.slope,
        intercept=res.intercept,
        rvalue=res.rvalue,
        pvalue=res.pvalue,
        stderr=res.stderr,
    )

    stderr = compute_residual_std_error(
        ref_timeseries, obs_timeseries, n, lambda x: linreg.slope * x + linreg.intercept
    )

    pred_const, t_a = _get_gwrefs_stats(p, n, stderr)

    rmse = np.sqrt(
        np.mean(
            (obs_timeseries - (linreg.slope * ref_timeseries + linreg.intercept)) ** 2
        )
    )

    # Create and return a FitResultData object with the regression results
    if tmin is None:
        logger.warning("tmin is None, setting to min common time of both wells")
        tmin = max(obs_well.timeseries.index.min(), ref_well.timeseries.index.min())
    if tmax is None:
        logger.warning("tmax is None, setting to max common time of both wells")
        tmax = min(obs_well.timeseries.index.max(), ref_well.timeseries.index.max())
    fit_result = FitResultData(
        obs_well=obs_well,
        ref_well=ref_well,
        rmse=rmse,
        n=n,
        fit_method=linreg,
        t_a=t_a,
        stderr=stderr,
        pred_const=pred_const,
        p=p,
        offset=offset,
        aggregation=aggregation,
        tmin=tmin,
        tmax=tmax,
        name=name,
    )
    return fit_result
