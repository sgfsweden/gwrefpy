import logging

import numpy as np
import pandas as pd

from ..methods.common import (
    FitMethodBase,
    _get_gwrefs_polyfit_stats,
    _validate_fit_rank,
    _validate_timeseries_len,
    compute_polyfit_residual_std_error,
)

logger = logging.getLogger(__name__)


class Chebyshev(FitMethodBase):
    def __init__(self, degree: int = 4):
        super().__init__()
        self.coefficients = None
        self.degree = degree
        self.rvalue = None
        self.rank = None
        self.stderr = None
        self.leverage_operator = None
        self.t_a = None
        self.rmse = None

    def __str__(self):
        return f"Chebyshev(degree={self.degree}, coefficients={self.coefficients})"

    def __repr__(self):
        return f"Chebyshev(degree={self.degree}, coefficients={self.coefficients})"

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

    def fit(
        self,
        obs_timeseries: pd.Series,
        ref_timeseries: pd.Series,
        p=0.95,
    ):
        """
        Perform Chebyshev polynomial fit between reference and observation well time
        series.

        Parameters
        ----------
        obs_timeseries : pd.Series
            The observation time series data.
        ref_timeseries : pd.Series
            The reference time series data.
        p : float, optional
            The confidence level for the prediction interval (default is 0.95).

        Returns
        -------
        None
            The method updates the instance attributes with the fit results.

        References
        ----------
        .. [1] numpy.polynomial.chebyshev.chebfit documentation:
        https://numpy.org/doc/stable/reference/generated/numpy.polynomial.chebyshev.chebfit.html
        """

        # Validate timeseries input
        n = len(obs_timeseries)
        _validate_timeseries_len(n, self.degree, "Chebyshev")

        # Perform Nth degree polynomial fit
        coefficients, res_list = np.polynomial.chebyshev.chebfit(
            ref_timeseries.values, obs_timeseries.values, self.degree, full=True
        )
        residuals, rank, singular_values, rcond = res_list
        _validate_fit_rank(rank, self.degree, "Chebyshev")

        # Compute residual standard error
        stderr = compute_polyfit_residual_std_error(
            obs_timeseries.values,
            np.polynomial.chebyshev.chebval(ref_timeseries.values, coefficients),
            n,
            self.degree,
        )
        training_matrix = np.polynomial.chebyshev.chebvander(
            ref_timeseries.values, self.degree
        )
        t_a, leverage_operator = _get_gwrefs_polyfit_stats(p, n, training_matrix)

        rmse = np.sqrt(
            np.mean(
                (
                    obs_timeseries.values
                    - np.polynomial.chebyshev.chebval(
                        ref_timeseries.values, coefficients
                    )
                )
                ** 2
            )
        )
        ss_total = np.sum((obs_timeseries.values - np.mean(obs_timeseries.values)) ** 2)
        r_squared = 1 - (residuals / ss_total)

        # Store the results in the instance attributes
        self.coefficients = coefficients
        self.rvalue = r_squared
        self.rank = rank
        self.stderr = stderr
        self.leverage_operator = leverage_operator
        self.t_a = t_a
        self.rmse = rmse

    def get_prediction_constant(self, ref_series: pd.Series) -> float | pd.Series:
        """Calculate prediction widths using the fit's original training data."""

        design_matrix = np.polynomial.chebyshev.chebvander(
            ref_series.to_numpy(), self.degree
        )

        leverage = np.einsum(
            "ij,jk,ik->i",
            design_matrix,
            self.leverage_operator,
            design_matrix,
        )
        widths = self.t_a * self.stderr * np.sqrt(1 + leverage)
        return pd.Series(widths, index=ref_series.index, name="pred_const")
