import logging

import numpy as np
import pandas as pd
import scipy as sp

from ..fitresults import FitResultData, LinRegResult
from ..methods.timeseries import groupby_time_equivalents
from ..well import Well

logger = logging.getLogger(__name__)


def linregressfit(
    obs_well: Well,
    ref_well: Well,
    offset: pd.DateOffset | pd.Timedelta | str,
    tmin: pd.Timestamp | str | None = None,
    tmax: pd.Timestamp | str | None = None,
    p=0.95,
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
    p : float, optional
        The confidence level for the prediction interval (default is 0.95).

    Returns
    -------
    fit_result : FitResultData
        A `FitResultData` object containing the results of the linear regression fit.
    """

    def _t_inv(probability, degrees_freedom):
        """
        Mimics Excel's T.INV function.
        Returns the t-value for the given probability and degrees of freedom.
        """
        return -sp.stats.t.ppf(probability, degrees_freedom)

    def _get_gwrefs_stats(p, n, stderr):
        ta = _t_inv((1 - p) / 2, n - 1)
        pc = ta * stderr * np.sqrt(1 + 1 / n)
        return pc, ta

    def compute_residual_std_error(x, y, a, b, n):
        y_pred = a * x + b
        residuals = y - y_pred

        stderr = np.sum(residuals**2) - np.sum(
            residuals * (x - np.mean(x))
        ) ** 2 / np.sum((x - np.mean(x)) ** 2)
        stderr *= 1 / (n - 2)
        stderr = np.sqrt(stderr)

        return stderr

    # Groupby time equivalents with given offset
    if ref_well.timeseries is None or obs_well.timeseries is None:
        logger.critical("Missing time series data for for either ref or obs well")
        return None

    ref_timeseries, obs_timeseries, n = groupby_time_equivalents(
        obs_well.timeseries.loc[tmin:tmax], ref_well.timeseries.loc[tmin:tmax], offset
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
        ref_timeseries, obs_timeseries, linreg.slope, linreg.intercept, n
    )

    pred_const, t_a = _get_gwrefs_stats(p, n, stderr)

    rmse = np.sqrt(
        np.mean(
            (obs_timeseries - (linreg.slope * ref_timeseries + linreg.intercept)) ** 2
        )
    )

    # Create and return a FitResultData object with the regression results
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
        tmin=tmin,
        tmax=tmax,
    )
    return fit_result


def linregress_to_dict(fit_result):
    linreg = fit_result.fit_method
    return {
        "slope": linreg.slope,
        "intercept": linreg.intercept,
        "rvalue": linreg.rvalue,
        "pvalue": linreg.pvalue,
        "stderr": linreg.stderr,
    }
