import uuid

import numpy as np
import pandas as pd

from .methods.chebyshev import Chebyshev
from .methods.common import (
    _validate_timeseries_len,
    compute_polyfit_residual_std_error,
    compute_residual_std_error,
)
from .methods.linregressfit import Linear
from .methods.npolyfit import NPoly
from .methods.timeseries import groupby_time_equivalents
from .utils.conversions import datetime_to_float
from .well import Well


class FitResultData:
    """
    This class contains all information that is required to reproduce a fit between
    a reference well and an observation well.

    Parameters
    ----------
    ref_well : Well
        The reference well object containing the time series data.
    obs_well : Well
        The observation well object containing the time series data.
    n : int
        The number of data points used in the fit.
    fit_method : Linear | NPoly | Chebyshev
        The method used for fitting (e.g., linreg).
    p : float
        The confidence level used in the fit.
    offset: pd.DateOffset | pd.Timedelta | str
        Allowed offset when grouping data points within time equivalents.
    aggregation: str
        The aggregation method used when grouping data points within time
        equivalents ("mean", "median", "min", or "max").
    te_method: str
        The time equivalent grouping method used ("anchor" or "consecutive").
    tmin: pd.Timestamp | str | None
        The minimum timestamp for the calibration period.
    tmax: pd.Timestamp | str | None
        The maximum timestamp for the calibration period.
    name: str
        A unique name for the FitResultData object. If None, a UUID will be generated.
    """

    def __init__(
        self,
        obs_well: Well,
        ref_well: Well,
        n: int,
        fit_method: Linear | NPoly | Chebyshev,
        p: float,
        offset: pd.DateOffset | pd.Timedelta | str,
        aggregation: str,
        te_method: str = "anchor",
        tmin: pd.Timestamp | str | None = None,
        tmax: pd.Timestamp | str | None = None,
        shift: pd.Timedelta | str | None = None,
        name: str | None = None,
    ):
        """
        Initialize a FitResultData object to store the results of a fit between.
        """
        self.obs_well = obs_well
        self.ref_well = ref_well
        self.n = n
        self.fit_method = fit_method
        self.p = p
        self.offset = offset
        self.aggregation = aggregation
        self.te_method = te_method
        self.tmin = tmin
        self.tmax = tmax
        self.shift = shift
        self.name = name if name is not None else str(uuid.uuid4())

    def __str__(self):
        """Return a nicely formatted table representation of the fit results."""
        # Header
        header = f"Fit Results: {self.obs_well.name} ~ {self.ref_well.name}"
        fit_method = f"{self.fit_method.__class__.__name__}"
        separator = "=" * max([len(header), len(fit_method)])

        # Build the table content
        if not hasattr(self.fit_method, "fit_str"):
            raise ValueError("No valid fit method provided.")
        fit_lines = self.fit_method.fit_str()
        lines = [
            header,
            separator,
            f"{'Statistic':<15} {'Value':<12} {'Description'}",
            "-" * 50,
            f"{'RMSE':<15} {self.rmse:<12.4f} Root Mean Square Error",
            *fit_lines,
            f"{'N':<15} {self.n:<12d} Number of Data Points",
            f"{'Std Error':<15} {self.stderr:<12.4f} Standard Error",
            f"{'Confidence':<15} {self.p * 100:<12.1f}% Confidence Level",
            "",
            f"Calibration Period: {self.tmin} to {self.tmax}",
            f"Time Offset: {self.offset}",
            f"Shift: {self.shift}",
            f"Aggregation Method: {self.aggregation}",
            f"Grouping Method: {self.te_method}",
        ]

        return "\n".join(lines)

    def _repr_html_(self):
        """Return HTML representation for Jupyter notebooks."""
        if hasattr(self.fit_method, "html_fit_str"):
            fit_lines = self.fit_method.html_fit_str()
            return f"""
                    <div>
                        <strong>Fit Results: {self.obs_well.name} ~
                                {self.ref_well.name}</strong>
                        <table>
                            <thead>
                                <tr>
                                    <th style="text-align: left">Statistic</th>
                                    <th style="text-align: left">Value</th>
                                    <th style="text-align: left">Description</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>RMSE</td>
                                    <td>{self.rmse:.4f}</td>
                                    <td>Root Mean Square Error</td>
                                </tr>
                                {"".join(fit_lines)}
                                <tr>
                                    <td>N</td>
                                    <td>{self.n}</td>
                                    <td>Number of Data Points</td>
                                </tr>
                                <tr>
                                    <td>Std Error</td>
                                    <td>{self.stderr:.4f}</td>
                                    <td>Standard Error</td>
                                </tr>
                                <tr>
                                    <td>Confidence</td>
                                    <td>{self.p * 100:.1f}%</td>
                                    <td>Confidence Level</td>
                                </tr>
                            </tbody>
                        </table>
                        <p>
                            Calibration Period: {self.tmin} to {self.tmax}<br>
                            Time Offset: {self.offset}<br>
                            Shift: {self.shift}<br>
                            Aggregation Method: {self.aggregation}
                        </p>
                    </div>
                """
        else:
            return "<div><strong>No fit method available.</strong></div>"

    def __repr__(self):
        """Return a concise representation for debugging."""
        return (
            f"FitResultData(ref_well='{self.ref_well.name}', "
            f"obs_well='{self.obs_well.name}', "
            f"rmse={self.fit_method.rmse:.4f}, n={self.n})"
        )

    @property
    def rmse(self):
        return self.fit_method.rmse

    @property
    def rvalue(self):
        return self.fit_method.rvalue

    @property
    def stderr(self):
        return self.fit_method.stderr

    @property
    def t_a(self):
        return self.fit_method.t_a

    def fit_timeseries(self, ref_series: pd.Series | None = None) -> pd.Series:
        """
        Apply the fit method to a reference time series to get the fitted values.

        Parameters
        ----------
        ref_series : pd.Series | None
            Reference values to evaluate. Defaults to the reference well's current
            time series.

        Returns
        -------
        pd.Series
            The fitted values based on the reference series.
        """
        if ref_series is None:
            ref_series = self.ref_well.timeseries
        if hasattr(self.fit_method, "fit_timeseries"):
            return self.fit_method.fit_timeseries(ref_series)
        else:
            raise NotImplementedError(
                f"Fitting method {self.fit_method.__class__.__name__} is not "
                f"implemented"
            )

    def get_fit_timeseries(self, ref_series: pd.Series | None = None) -> pd.Series:
        """
        Get the fitted time series for the reference well.

        Returns
        -------
        pd.Series
            The fitted time series for the reference well.
        """

        return self.fit_timeseries(ref_series)

    def get_prediction_constant(
        self, ref_series: pd.Series | None = None
    ) -> float | pd.Series:
        """Calculate prediction widths using the fit's original training data.

        For polynomial fits, widths are calculated at each value in ``ref_series``
        using the leverage operator saved when the fit was trained. The training
        window is not expanded when new reference values are supplied.
        """
        if ref_series is None:
            ref_series = self.ref_well.timeseries
        return self.fit_method.get_prediction_constant(ref_series)

    def get_upper_confidence_bound(
        self, ref_series: pd.Series | None = None
    ) -> pd.Series:
        """
        Calculate the upper confidence bound based on the fit method and RMSE.

        Parameters
        ----------
        ref_series : pd.Series | None
            Reference values to evaluate. Defaults to the reference well's current
            time series.

        Returns
        -------
        pd.Series
            The upper confidence bound based on the fit method and RMSE.
        """
        fitted_values = self.fit_method.fit_timeseries(ref_series)
        return fitted_values + self.fit_method.get_prediction_constant(ref_series)

    def get_lower_confidence_bound(
        self, ref_series: pd.Series | None = None
    ) -> pd.Series:
        """
        Calculate the lower confidence bound based on the fit method and RMSE.

        Parameters
        ----------
        ref_series : pd.Series | None
            Reference values to evaluate. Defaults to the reference well's current
            time series.

        Returns
        -------
        pd.Series
            The lower confidence bound based on the fit method and RMSE.
        """
        fitted_values = self.fit_method.fit_timeseries(ref_series)
        return fitted_values - self.fit_method.get_prediction_constant(ref_series)

    def fit_outliers(self, ref_series: pd.Series | None = None) -> pd.Series:
        """
        Calculate the outliers based on the fit method and RMSE.

        Returns
        -------
        pd.Series
            The outlier values based on the fit method and RMSE.
        """
        if hasattr(self.fit_method, "fit_timeseries"):
            if ref_series is None:
                ref_series = self.ref_well.timeseries
            fitted_values = self.fit_method.fit_timeseries(ref_series)
            prediction_constant = self.fit_method.get_prediction_constant(ref_series)
            if self.shift is not None:
                obs_timeseries = self.obs_well.shift_timeseries(self.shift)
            else:
                obs_timeseries = self.obs_well.timeseries
            outliers = pd.Series(
                abs(obs_timeseries - fitted_values) > prediction_constant,
                index=obs_timeseries.index,
            )
            return outliers
        else:
            raise NotImplementedError(
                f"Fitting method {self.fit_method.__class__.__name__} is not "
                f"implemented"
            )

    def test_fit(
        self,
        ref_series: pd.Series,
        offset: pd.DateOffset | pd.Timedelta | str,
        tmin: pd.Timestamp | str | None = None,
        tmax: pd.Timestamp | str | None = None,
        aggregation="mean",
        te_method="anchor",
    ) -> tuple[float, float]:
        """
        Test the fit method on a given reference series.

        Parameters
        ----------
        ref_series : pd.Series
            The reference series to test the fit on.
        offset: pd.DateOffset | pd.Timedelta | str
            The offset to apply when grouping the time series into time equivalents.
        tmin: pd.Timestamp | str | None = None
            The minimum timestamp for the calibration period.
        tmax: pd.Timestamp | str | None = None
            The maximum timestamp for the calibration period.
        aggregation : str, optional
            The aggregation method to use when grouping data points within time
            equivalents (default is "mean"). Can be "mean", "median", "min", or "max".
        te_method : str, optional
            The time equivalent grouping method (default is "anchor"). Can be "anchor"
            or "consecutive".

        Returns
        -------
        stderr : float
            The standard error of the fit on the given reference series.
        rmse : float
            The root mean square error of the fit on the given reference series.
        """
        # Validate input
        if not isinstance(ref_series, pd.Series):
            raise ValueError("ref_series must be a pandas Series.")

        # Apply the fit method to the reference series
        if hasattr(self.fit_method, "fit_timeseries"):
            # Group by time equivalents with given offset
            ref_timeseries, obs_timeseries, n = groupby_time_equivalents(
                self.obs_well.timeseries.loc[tmin:tmax],
                ref_series.loc[tmin:tmax],
                offset,
                aggregation,
                te_method,
            )

            # Calculate fitted values and residuals
            fitted_values = self.fit_method.fit_timeseries(ref_timeseries)
            residuals = obs_timeseries - fitted_values

            # Compute stderr and rmse
            if isinstance(self.fit_method, NPoly | Chebyshev):
                method_name = self.fit_method.__class__.__name__
                _validate_timeseries_len(n, self.fit_method.degree, method_name)
                stderr = compute_polyfit_residual_std_error(
                    obs_timeseries.to_numpy(),
                    fitted_values.to_numpy(),
                    n,
                    self.fit_method.degree,
                )
            else:
                stderr = compute_residual_std_error(
                    ref_timeseries,
                    obs_timeseries,
                    n,
                    lambda x: self.fit_method.fit_timeseries(x),
                )
            rmse = np.sqrt(np.mean(residuals**2))
            return stderr, rmse
        else:
            raise NotImplementedError(
                f"Fitting method {self.fit_method.__class__.__name__} is not "
                f"implemented"
            )

    def has_well(self, well: Well) -> bool:
        """
        Check if the FitResultData object involves the given well.

        Parameters
        ----------
        well : Well
            The well to check.

        Returns
        -------
        bool
            True if the well is either the reference or observation well,
            False otherwise.
        """
        return self.ref_well == well or self.obs_well == well

    def _to_dict(self) -> dict:
        """
        Convert the FitResultData object to a dictionary.

        Returns
        -------
        dict
            A dictionary representation of the FitResultData object.
        """
        dict_representation = {
            "ref_well": self.ref_well.name,
            "obs_well": self.obs_well.name,
            "rmse": self.rmse,
            "n": self.n,
            "t_a": self.t_a,
            "fit_method": str(self.fit_method.__class__.__name__),
            "stderr": self.stderr,
            "pred_const": (
                self.pred_const.tolist()
                if isinstance(self.pred_const, pd.Series)
                else self.pred_const
            ),
            "pred_const_index": (
                [datetime_to_float(timestamp) for timestamp in self.pred_const.index]
                if isinstance(self.pred_const, pd.Series)
                else None
            ),
            "prediction_leverage_matrix": (
                self.prediction_leverage_matrix.tolist()
                if self.prediction_leverage_matrix is not None
                else None
            ),
            "p": self.p,
            "offset": self.offset,
            "aggregation": self.aggregation,
            "te_method": self.te_method,
            "tmin": datetime_to_float(self.tmin),
            "tmax": datetime_to_float(self.tmax),
            "name": self.name,
        }

        if hasattr(self.fit_method, "_to_dict"):
            dict_representation[self.fit_method.__class__.__name__] = (
                self.fit_method._to_dict()
            )

        return dict_representation


def _unpack_dict_fit_method(
    data: dict,
) -> Linear | NPoly | Chebyshev:
    """
    Unpack a dictionary representation of a fit method into a Linear object.

    Parameters
    ----------
    data : dict
        A dictionary containing the fit method data.

    Returns
    -------
    Linear | NPoly | Chebyshev
        The unpacked fitting method object.
    """
    fit_method_name = data.get("fit_method", None)
    if fit_method_name == "Linear":
        linreg_data = data.get("Linear", {})
        return Linear.from_dict(linreg_data)
    elif fit_method_name == "NPoly":
        npoly_data = data.get("NPoly", {})
        return NPoly.from_dict(npoly_data)
    elif fit_method_name == "Chebyshev":
        chebyshev_data = data.get("Chebyshev", {})
        return Chebyshev.from_dict(chebyshev_data)
    else:
        raise ValueError(f"Unsupported fit method: {fit_method_name}")
