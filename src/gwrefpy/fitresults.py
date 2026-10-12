import uuid

import numpy as np
import pandas as pd

from .methods.linregressfit import Linear
from .methods.npolyfit import NPolyFit
from .utils.conversions import datetime_to_float
from .well import Well


class ChebyshevFitResult:
    """
    This class contains the results of a Chebyshev polynomial fit.

    Parameters
    ----------
    coefficients : np.ndarray[float]
        The coefficients of the Chebyshev polynomial.
    """

    def __init__(self, coefficients: np.array):
        self.coefficients = coefficients
        self.degree = len(coefficients) - 1

    def __str__(self):
        return (
            f"ChebyshevFitResult(degree={self.degree}, "
            f"coefficients={self.coefficients})"
        )

    def __repr__(self):
        return (
            f"ChebyshevFitResult(degree={self.degree}, "
            f"coefficients={self.coefficients})"
        )

    def fit_str(self):
        fit_lines = [
            f"{'Degree':<15} {self.degree:<12d} Degree of Chebyshev Polynomial Fit",
            *[
                f"{'Coefficient ' + str(i):<15} {coef:<12.4f} Chebyshev "
                f"Polynomial Coefficient"
                for i, coef in enumerate(self.coefficients)
            ],
        ]
        return fit_lines

    def html_fit_str(self):
        return f"""
                <tr>
                    <td>Degree</td>
                    <td>{self.degree}</td>
                    <td>Degree of Chebyshev Polynomial Fit</td>
                </tr>
                {
            "".join(
                [
                    f"<tr><td>Coefficient {i}</td>"
                    f"<td>{coef:.4f}</td>"
                    f"<td>Chebyshev Polynomial Coefficient</td>"
                    f"</tr>"
                    for i, coef in enumerate(self.coefficients)
                ]
            )
        }
                """

    def fit_timeseries(self, timeseries: pd.Series) -> pd.Series:
        """
        Apply the Chebyshev polynomial fit to a given series.

        Parameters
        ----------
        timeseries : pd.Series
            The input series to apply the fit to.

        Returns
        -------
        pd.Series
            The fitted values based on the input series.
        """
        return timeseries.apply(
            lambda x: np.polynomial.chebyshev.chebval(
                x, self.coefficients, tensor=False
            )
        )

    def to_dict(self) -> dict:
        """
        Convert the ChebyshevFitResult object to a dictionary.

        Returns
        -------
        dict
            A dictionary representation of the ChebyshevFitResult object.
        """
        return {
            "coefficients": self.coefficients.tolist(),
        }


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
    rmse : float
        The root mean square error of the fit.
    n : int
        The number of data points used in the fit.
    fit_method : Linear | NPolyFit | ChebyshevFitResult
        The method used for fitting (e.g., linreg).
    t_a : float
        The t-value for the given confidence level and degrees of freedom.
    stderr : float
        The standard error of the regression.
    pred_const : float | pd.Series
        The prediction constant for the confidence interval. Polynomial fits use a
        value for each reference time-series point.
    prediction_leverage_matrix : np.ndarray | None
        The leverage operator derived from the training data for polynomial fits.
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
        fit_method: Linear | NPolyFit | ChebyshevFitResult,
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
            f"rmse={self.rmse:.4f}, n={self.n})"
        )

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
        if not isinstance(self.fit_method, ChebyshevFitResult):
            return self.fit_method.get_prediction_constant(ref_series)

        was_provided = ref_series is not None
        if ref_series is None:
            ref_series = self.ref_well.timeseries
        if not isinstance(ref_series, pd.Series):
            raise TypeError("ref_series must be a pandas Series.")

        if self.prediction_leverage_matrix is None:
            if (
                isinstance(self.pred_const, pd.Series)
                and ref_series.index.isin(self.pred_const.index).all()
            ):
                return self.pred_const.reindex(ref_series.index)
            if not was_provided and not isinstance(self.pred_const, pd.Series):
                return self.pred_const
            raise ValueError(
                "This fit does not contain the training leverage data required to "
                "calculate prediction widths for new reference values. Refit the "
                "model to enable this API."
            )

        if isinstance(self.fit_method, NPolyFit):
            design_matrix = np.vander(
                ref_series.to_numpy(), len(self.fit_method.coefficients)
            )
        else:
            design_matrix = np.polynomial.chebyshev.chebvander(
                ref_series.to_numpy(), self.fit_method.degree
            )
        leverage = np.einsum(
            "ij,jk,ik->i",
            design_matrix,
            self.prediction_leverage_matrix,
            design_matrix,
        )
        widths = self.t_a * self.stderr * np.sqrt(1 + leverage)
        return pd.Series(widths, index=ref_series.index, name="pred_const")

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
) -> Linear | NPolyFit | ChebyshevFitResult:
    """
    Unpack a dictionary representation of a fit method into a Linear object.

    Parameters
    ----------
    data : dict
        A dictionary containing the fit method data.

    Returns
    -------
    Linear | NPolyFit | ChebyshevFitResult
        The unpacked fitting method object.
    """
    fit_method_name = data.get("fit_method", None)
    if fit_method_name == "Linear":
        linreg_data = data.get("Linear", {})
        return Linear(
            slope=linreg_data.get("slope", 0.0),
            intercept=linreg_data.get("intercept", 0.0),
            rvalue=linreg_data.get("rvalue", 0.0),
            pvalue=linreg_data.get("pvalue", 0.0),
            stderr=linreg_data.get("stderr", 0.0),
        )
    elif fit_method_name == "NPolyFit":
        npoly_data = data.get("NPolyFit", {})
        return NPolyFit(
            coefficients=np.array(npoly_data.get("coefficients", [])),
        )
    elif fit_method_name == "ChebyshevFitResult":
        chebyshev_data = data.get("ChebyshevFitResult", {})
        return ChebyshevFitResult(
            coefficients=np.array(chebyshev_data.get("coefficients", [])),
        )
    else:
        raise ValueError(f"Unsupported fit method: {fit_method_name}")
