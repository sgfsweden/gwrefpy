import logging

import numpy as np
import pandas as pd
import scipy as sp

from ..methods.common import (
    FitMethodBase,
    _get_gwrefs_stats,
    _validate_timeseries_len,
    compute_residual_std_error,
)

logger = logging.getLogger(__name__)


class Linear(FitMethodBase):
    """
    A mixin class that provides linear regression fitting functionality to a
    base fit method class.

    This class is intended to be used as a mixin for classes that implement the
    `FitMethodBase` interface. It provides methods for performing linear regression
    fits and computing related statistics.

    Attributes
    ----------
    None

    Methods
    -------
    fit(ref_timeseries, obs_timeseries, p=0.95)
        Perform linear regression fit between reference and observation well
        time series.
    """

    _scalar_keys = (
        "slope",
        "intercept",
        "rvalue",
        "pvalue",
        "slope_stderr",
        "stderr",
        "pred_const",
        "t_a",
        "rmse",
    )
    _array_keys = ()

    def __init__(self):
        super().__init__()
        self.slope = None
        self.intercept = None
        self.pvalue = None
        # Standard error of the slope; ``stderr`` is the residual standard error
        self.slope_stderr = None
        self.pred_const = None

    def __str__(self):
        if self.slope is None:
            return "Linear()"
        return (
            f"Linear(slope={self.slope:.4f}, intercept={self.intercept:.4f}, "
            f"rvalue={self.rvalue:.4f}, pvalue={self.pvalue:.4f}, "
            f"stderr={self.stderr:.4f})"
        )

    def __repr__(self):
        return str(self)

    def fit_str(self):
        fit_lines = [
            f"{'R²':<15} {self.rvalue**2:<12.4f} Coefficient of Determination",
            f"{'R-value':<15} {self.rvalue:<12.4f} Correlation Coefficient",
            f"{'Slope':<15} {self.slope:<12.4f} Linear Regression Slope",
            f"{'Intercept':<15} {self.intercept:<12.4f} Linear Regression Intercept",
            f"{'P-value':<15} {self.pvalue:<12.4f} Statistical Significance",
        ]
        return fit_lines

    def html_fit_str(self):
        return f"""
            <tr>
                <td>R²</td>
                <td>{self.rvalue**2:.4f}</td>
                <td>Coefficient of Determination</td>
            </tr>
            <tr>
                <td>R-value</td>
                <td>{self.rvalue:.4f}</td>
                <td>Correlation Coefficient</td>
            </tr>
            <tr>
                <td>Slope</td>
                <td>{self.slope:.4f}</td>
                <td>Linear Regression Slope</td>
            </tr>
            <tr>
                <td>Intercept</td>
                <td>{self.intercept:.4f}</td>
                <td>Linear Regression Intercept</td>
            </tr>
            <tr>
                <td>P-value</td>
                <td>{self.pvalue:.4f}</td>
                <td>Statistical Significance</td>
            </tr>
            """

    def fit(
        self,
        obs_timeseries: pd.Series,
        ref_timeseries: pd.Series,
        p: float = 0.95,
    ):
        """
        Perform linear regression fit between reference and observation well
        time series.

        Parameters
        ----------
        ref_timeseries : pd.Series
            The reference time series data.
        obs_timeseries : pd.Series
            The observation time series data.
        p : float, optional
            The confidence level for the prediction interval (default is 0.95).

        Returns
        -------
        None
            The results of the fit are stored in the instance attributes.

        References
        ----------
        .. [1] scipy.stats.linregress documentation:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html
        """
        # Validate input time series
        n = len(ref_timeseries)
        _validate_timeseries_len(n, 1, "Linear Regression")

        # Fit the linear regression model using scipy.stats.linregress
        res = sp.stats.linregress(ref_timeseries, obs_timeseries)

        # Compute the residual standard error
        stderr = compute_residual_std_error(
            ref_timeseries, obs_timeseries, n, lambda x: res.slope * x + res.intercept
        )

        # Compute prediction interval constants
        pred_const, t_a = _get_gwrefs_stats(p, n, stderr)

        # Compute the root mean square error (RMSE)
        rmse = np.sqrt(
            np.mean(
                (obs_timeseries - (res.slope * ref_timeseries + res.intercept)) ** 2
            )
        )

        # Store the results in the instance attributes
        self.slope = res.slope
        self.intercept = res.intercept
        self.rvalue = res.rvalue
        self.pvalue = res.pvalue
        self.slope_stderr = res.stderr
        self.stderr = stderr
        self.pred_const = pred_const
        self.t_a = t_a
        self.rmse = rmse

    def fit_timeseries(self, timeseries: pd.Series) -> pd.Series:
        """
        Apply the linear regression fit to a given series.

        Parameters
        ----------
        timeseries : pd.Series
            The input series to apply the fit to.

        Returns
        -------
        pd.Series
            The fitted values based on the input series.
        """
        return timeseries.apply(lambda x: self.slope * x + self.intercept)

    def get_prediction_constant(self, ref_series: pd.Series | None = None) -> float:
        """
        Get the prediction constant for the linear regression fit.

        Returns
        -------
        float
            The prediction constant.
        """
        return self.pred_const
