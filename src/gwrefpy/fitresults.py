import uuid
from typing import cast

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure, SubFigure

from .methods.common import compute_residual_std_error
from .methods.timeseries import groupby_time_equivalents
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

    def to_dict(self) -> dict:
        """
        Convert the LinRegResult object to a dictionary.

        Returns
        -------
        dict
            A dictionary representation of the LinRegResult object.
        """
        return {
            "slope": self.slope,
            "intercept": self.intercept,
            "rvalue": self.rvalue,
            "pvalue": self.pvalue,
            "stderr": self.stderr,
        }


class NPolyFitResult:
    """
    This class contains the results of a n-degree polynomial fit.

    Parameters
    ----------
    coefficients : np.ndarray[float]
        The coefficients of the polynomial.
    """

    def __init__(self, coefficients: np.array):
        self.coefficients = coefficients
        self.degree = len(coefficients) - 1

    def __str__(self):
        return f"NPolyFitResult(degree={self.degree}, coefficients={self.coefficients})"

    def __repr__(self):
        return f"NPolyFitResult(degree={self.degree}, coefficients={self.coefficients})"

    def fit_str(self):
        fit_lines = [
            f"{'Degree':<15} {self.degree:<12d} Degree of Polynomial Fit",
            *[
                f"{'Coefficient ' + str(i):<15} {coef:<12.4f} Polynomial Coefficient"
                for i, coef in enumerate(self.coefficients)
            ],
        ]
        return fit_lines

    def html_fit_str(self):
        return f"""
                <tr>
                    <td>Degree</td>
                    <td>{self.degree}</td>
                    <td>Degree of Polynomial Fit</td>
                </tr>
                {
            "".join(
                [
                    f"<tr><td>Coefficient {i}</td>"
                    f"<td>{coef:.4f}</td>"
                    f"<td>Polynomial Coefficient</td>"
                    f"</tr>"
                    for i, coef in enumerate(self.coefficients)
                ]
            )
        }
                """

    def fit_timeseries(self, timeseries: pd.Series) -> pd.Series:
        """
        Apply the polynomial fit to a given series.

        Parameters
        ----------
        timeseries : pd.Series
            The input series to apply the fit to.

        Returns
        -------
        pd.Series
            The fitted values based on the input series.
        """
        return timeseries.apply(lambda x: np.polyval(self.coefficients, x))

    def to_dict(self) -> dict:
        """
        Convert the NPolyFitResult object to a dictionary.

        Returns
        -------
        dict
            A dictionary representation of the NPolyFitResult object.
        """
        return {
            "coefficients": self.coefficients.tolist(),
        }


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
        rmse: float,
        n: int,
        fit_method: LinRegResult | NPolyFitResult | ChebyshevFitResult,
        t_a: float,
        stderr: float,
        pred_const: float,
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
        self.rmse = rmse
        self.n = n
        self.fit_method = fit_method
        self.t_a = t_a
        self.stderr = stderr
        self.pred_const = pred_const
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
        if hasattr(self.fit_method, "fit_timeseries"):
            return self.fit_method.fit_timeseries(self.ref_well.timeseries)
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
        if hasattr(self.fit_method, "fit_timeseries"):
            fitted_values = self.fit_timeseries()
            if self.shift is not None:
                obs_timeseries = self.obs_well.shift_timeseries(self.shift)
            else:
                obs_timeseries = self.obs_well.timeseries
            outliers = pd.Series(
                abs(obs_timeseries - fitted_values) > self.pred_const,
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

    def plot(
        self,
        title: str = "",
        xlabel: str = "Time",
        ylabel: str = "Measurements",
        mark_outliers: bool = True,
        show_initiation_period: bool = False,
        plot_ref_well: bool = False,
        plot_style: str | None = None,
        color_style: str | None = None,
        save_path: str | None = None,
        num: int = 6,
        ax: Axes | None = None,
        offset_text: dict[str, float] | None = None,
        tmin: str | pd.Timestamp | None = None,
        tmax: str | pd.Timestamp | None = None,
        **kwargs,
    ) -> tuple[Figure | SubFigure, Axes]:
        """
        Plot this fit result.

        This is a convenience method that delegates to Plotter.plot_fits().
        See Plotter.plot_fits() for full parameter documentation.

        Parameters
        ----------
        title : str
            The title of the plot.
        xlabel : str
            The label for the x-axis.
        ylabel : str
            The label for the y-axis.
        mark_outliers : bool
            If True, outliers will be marked on the plot.
        show_initiation_period : bool
            If True, the initiation period will be shaded.
        plot_ref_well : bool
            If True, the reference well data will be plotted.
        plot_style : str | None
            Style of the plot ("fancy", "scientific", or None).
        color_style : str | None
            Color style ("color", "monochrome", or None).
        save_path : str | None
            If provided, the plot will be saved to this path.
        num : int
            Number of ticks on the x-axis.
        ax : matplotlib.axes.Axes | None
            Optional existing Axes to plot on.
        offset_text : dict[str, float] | None
            Vertical offset for text labels.
        tmin : str | pd.Timestamp | None
            Minimum time for the plot. If provided, the time series will be sliced
            to only show data from this time onwards. Can be a string like "2021-01-01"
            or a pd.Timestamp. Default is None (no minimum time limit).
        tmax : str | pd.Timestamp | None
            Maximum time for the plot. If provided, the time series will be sliced
            to only show data up to this time. Can be a string like "2021-01-01"
            or a pd.Timestamp. Default is None (no maximum time limit).
        **kwargs : dict
            Additional matplotlib kwargs (e.g., figsize, dpi).

        Returns
        -------
        tuple[Figure | SubFigure, Axes]
            The figure and axes objects.
        """
        from .plotter import _FitPlotter

        plotter = _FitPlotter(self)
        # plot_separately=False ensures return type is tuple[Figure | SubFigure, Axes]
        result = plotter.plot_fits(
            fits=self,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            mark_outliers=mark_outliers,
            show_initiation_period=show_initiation_period,
            plot_ref_well=plot_ref_well,
            plot_style=plot_style,
            color_style=color_style,
            save_path=save_path,
            num=num,
            plot_separately=False,
            ax=ax,
            offset_text=offset_text,
            tmin=tmin,
            tmax=tmax,
            **kwargs,
        )
        return cast(tuple[Figure | SubFigure, Axes], result)

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
            "te_method": self.te_method,
            "tmin": datetime_to_float(self.tmin),
            "tmax": datetime_to_float(self.tmax),
            "name": self.name,
        }

        if hasattr(self.fit_method, "to_dict"):
            dict_representation[self.fit_method.__class__.__name__] = (
                self.fit_method.to_dict()
            )

        return dict_representation


def _unpack_dict_fit_method(
    data: dict,
) -> LinRegResult | NPolyFitResult | ChebyshevFitResult:
    """
    Unpack a dictionary representation of a fit method into a LinRegResult object.

    Parameters
    ----------
    data : dict
        A dictionary containing the fit method data.

    Returns
    -------
    LinRegResult | NPolyFitResult | ChebyshevFitResult
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
    elif fit_method_name == "ChebyshevFitResult":
        chebyshev_data = data.get("ChebyshevFitResult", {})
        return ChebyshevFitResult(
            coefficients=np.array(chebyshev_data.get("coefficients", [])),
        )
    else:
        raise ValueError(f"Unsupported fit method: {fit_method_name}")
