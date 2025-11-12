import numpy as np
import pandas as pd

from .utils.conversions import datetime_to_float
from .well import Well


class LinRegResult:
    """
    This class contains the results of a linear regression fit.

    Parameters
    ----------
    slope : float
        The slope of the regression line.
    intercept : float
        The intercept of the regression line.
    rvalue : float
        The correlation coefficient.
    pvalue : float
        The two-sided p-value for a hypothesis test whose null hypothesis is
        that the slope is zero.
    stderr : float
        The standard error of the estimated slope.
    """

    def __init__(
        self,
        slope: float,
        intercept: float,
        rvalue: float,
        pvalue: float,
        stderr: float,
    ):
        self.slope = slope
        self.intercept = intercept
        self.rvalue = rvalue
        self.pvalue = pvalue
        self.stderr = stderr

    def __str__(self):
        return (
            f"LinRegResult(slope={self.slope:.4f}, intercept={self.intercept:.4f}, "
            f"rvalue={self.rvalue:.4f}, pvalue={self.pvalue:.4f}, "
            f"stderr={self.stderr:.4f})"
        )

    def __repr__(self):
        return (
            f"LinRegResult(slope={self.slope:.4f}, intercept={self.intercept:.4f}, "
            f"rvalue={self.rvalue:.4f}, pvalue={self.pvalue:.4f}, "
            f"stderr={self.stderr:.4f})"
        )


class NPolyFitResult:
    """
    This class contains the results of a n-degree polynomial fit.

    Parameters
    ----------
    coefficients : np.ndarray[float]
        The coefficients of the polynomial, ordered from highest degree to lowest.
    """

    def __init__(self, coefficients: np.array):
        self.coefficients = coefficients
        self.degree = len(coefficients) - 1

    def __str__(self):
        return f"NPolyFitResult(coefficients={self.coefficients})"

    def __repr__(self):
        return f"NPolyFitResult(coefficients={self.coefficients})"


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
    fit_method : LinRegResult | NPolyFitResult
        The method used for fitting (e.g., linreg).
    t_a : float
        The t-value for the given confidence level and degrees of freedom.
    stderr : float
        The standard error of the regression.
    pred_const : float
        The prediction constant for the confidence interval.
    p : float
        The confidence level used in the fit.
    offset: pd.DateOffset | pd.Timedelta | str
        Allowed offset when grouping data points within time equivalents.
    aggregation: str
        The aggregation method used when grouping data points within time
        equivalents ("mean", "median", "min", or "max").
    tmin: pd.Timestamp | str | None
        The minimum timestamp for the calibration period.
    tmax: pd.Timestamp | str | None
        The maximum timestamp for the calibration period.
    """

    def __init__(
        self,
        obs_well: Well,
        ref_well: Well,
        rmse: float,
        n: int,
        fit_method: LinRegResult | NPolyFitResult,
        t_a: float,
        stderr: float,
        pred_const: float,
        p: float,
        offset: pd.DateOffset | pd.Timedelta | str,
        aggregation: str,
        tmin: pd.Timestamp | str | None,
        tmax: pd.Timestamp | str | None,
    ):
        """
        Initialize a FitResultData object to store the results of a fit between.
        """
        self.obs_well = obs_well
        self.ref_well = ref_well
        self.rmse = rmse
        self.n = n
        self.fit_method = fit_method
        self.t_a = t_a
        self.stderr = stderr
        self.pred_const = pred_const
        self.p = p
        self.offset = offset
        self.aggregation = aggregation
        self.tmin = tmin
        self.tmax = tmax

    def __str__(self):
        """Return a nicely formatted table representation of the fit results."""
        # Header
        header = f"Fit Results: {self.obs_well.name} ~ {self.ref_well.name}"
        fit_method = f"{self.fit_method.__class__.__name__}"
        separator = "=" * max([len(header), len(fit_method)])

        # Build the table content
        if isinstance(self.fit_method, LinRegResult):
            fit_lines = [
                f"{'R²':<15} {self.fit_method.rvalue ** 2:<12.4f} "
                f"Coefficient of Determination",
                f"{'R-value':<15} {self.fit_method.rvalue:<12.4f} Correlation "
                f"Coefficient",
                f"{'Slope':<15} {self.fit_method.slope:<12.4f} Linear Regression Slope",
                f"{'Intercept':<15} {self.fit_method.intercept:<12.4f} "
                f"Linear Regression Intercept",
                f"{'P-value':<15} {self.fit_method.pvalue:<12.4f} Statistical "
                f"Significance",
            ]
        elif isinstance(self.fit_method, NPolyFitResult):
            fit_lines = [
                f"{'Degree':<15} {self.fit_method.degree:<12d} Degree of Polynomial "
                f"Fit",
                *[
                    f"{'Coefficient ' + str(i):<15} {coef:<12.4f} Polynomial "
                    f"Coefficient"
                    for i, coef in enumerate(self.fit_method.coefficients)
                ],
            ]
        else:
            fit_lines = []
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
            f"Aggregation Method: {self.aggregation}",
        ]

        return "\n".join(lines)

    def _repr_html_(self):
        """Return HTML representation for Jupyter notebooks."""
        return f"""
        <div>
            <strong>Fit Results: {self.obs_well.name} ~ {self.ref_well.name}</strong>
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
                    <tr>
                        <td>R²</td>
                        <td>{self.fit_method.rvalue**2:.4f}</td>
                        <td>Coefficient of Determination</td>
                    </tr>
                    <tr>
                        <td>R-value</td>
                        <td>{self.fit_method.rvalue:.4f}</td>
                        <td>Correlation Coefficient</td>
                    </tr>
                    <tr>
                        <td>Slope</td>
                        <td>{self.fit_method.slope:.4f}</td>
                        <td>Linear Regression Slope</td>
                    </tr>
                    <tr>
                        <td>Intercept</td>
                        <td>{self.fit_method.intercept:.4f}</td>
                        <td>Linear Regression Intercept</td>
                    </tr>
                    <tr>
                        <td>P-value</td>
                        <td>{self.fit_method.pvalue:.4f}</td>
                        <td>Statistical Significance</td>
                    </tr>
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
                Aggregation Method: {self.aggregation}
            </p>
        </div>
        """

    def __repr__(self):
        """Return a concise representation for debugging."""
        return (
            f"FitResultData(ref_well='{self.ref_well.name}', "
            f"obs_well='{self.obs_well.name}', "
            f"rmse={self.rmse:.4f}, r²={self.fit_method.rvalue**2:.4f}, n={self.n})"
        )

    def fit_timeseries(self) -> pd.Series:
        """
        Apply the fit method to a reference time series to get the fitted values.

        Parameters
        ----------


        Returns
        -------
        pd.Series
            The fitted values based on the reference series.
        """
        if isinstance(self.fit_method, LinRegResult):
            return self.ref_well.timeseries.apply(
                lambda x: self.fit_method.slope * x + self.fit_method.intercept
            )
        elif isinstance(self.fit_method, NPolyFitResult):
            return self.ref_well.timeseries.apply(
                lambda x: np.polyval(self.fit_method.coefficients, x)
            )
        else:
            raise NotImplementedError(
                f"Fitting method {self.fit_method.__class__.__name__} is not "
                f"implemented"
            )

    def get_fit_timeseries(self) -> pd.Series:
        """
        Get the fitted time series for the reference well.

        Returns
        -------
        pd.Series
            The fitted time series for the reference well.
        """

        return self.fit_timeseries()

    def get_upper_confidence_bound(self) -> pd.Series:
        """
        Calculate the upper confidence bound based on the fit method and RMSE.

        Returns
        -------
        pd.Series
            The upper confidence bound based on the fit method and RMSE.
        """
        fitted_values = self.fit_timeseries()
        return fitted_values + self.pred_const

    def get_lower_confidence_bound(self) -> pd.Series:
        """
        Calculate the lower confidence bound based on the fit method and RMSE.

        Returns
        -------
        pd.Series
            The lower confidence bound based on the fit method and RMSE.
        """
        fitted_values = self.fit_timeseries()
        return fitted_values - self.pred_const

    def fit_outliers(self) -> pd.Series:
        """
        Calculate the outliers based on the fit method and RMSE.

        Returns
        -------
        pd.Series
            The outlier values based on the fit method and RMSE.
        """
        if isinstance(self.fit_method, LinRegResult):
            fitted_values = self.fit_timeseries()
            outliers = pd.Series(
                abs(self.obs_well.timeseries - fitted_values) > self.pred_const,
                index=self.obs_well.timeseries.index,
            )
            return outliers
        elif isinstance(self.fit_method, NPolyFitResult):
            fitted_values = self.fit_timeseries()
            outliers = pd.Series(
                abs(self.obs_well.timeseries - fitted_values) > self.pred_const,
                index=self.obs_well.timeseries.index,
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
            "pred_const": self.pred_const,
            "p": self.p,
            "offset": self.offset,
            "aggregation": self.aggregation,
            "tmin": datetime_to_float(self.tmin),
            "tmax": datetime_to_float(self.tmax),
        }

        if dict_representation["fit_method"] == "LinRegResult":
            dict_representation["LinRegResult"] = {
                "slope": self.fit_method.slope,
                "intercept": self.fit_method.intercept,
                "rvalue": self.fit_method.rvalue,
                "pvalue": self.fit_method.pvalue,
                "stderr": self.fit_method.stderr,
            }
        elif dict_representation["fit_method"] == "NPolyFitResult":
            dict_representation["NPolyFitResult"] = {
                "coefficients": self.fit_method.coefficients.tolist(),
            }

        return dict_representation


def _unpack_dict_fit_method(data: dict) -> LinRegResult | NPolyFitResult:
    """
    Unpack a dictionary representation of a fit method into a LinRegResult object.

    Parameters
    ----------
    data : dict
        A dictionary containing the fit method data.

    Returns
    -------
    LinRegResult | NPolyFitResult
        The unpacked fitting method object.
    """
    fit_method_name = data.get("fit_method", None)
    if fit_method_name == "LinRegResult":
        linreg_data = data.get("LinRegResult", {})
        return LinRegResult(
            slope=linreg_data.get("slope", 0.0),
            intercept=linreg_data.get("intercept", 0.0),
            rvalue=linreg_data.get("rvalue", 0.0),
            pvalue=linreg_data.get("pvalue", 0.0),
            stderr=linreg_data.get("stderr", 0.0),
        )
    elif fit_method_name == "NPolyFitResult":
        npoly_data = data.get("NPolyFitResult", {})
        return NPolyFitResult(
            coefficients=np.array(npoly_data.get("coefficients", [])),
        )
    else:
        raise ValueError(f"Unsupported fit method: {fit_method_name}")
