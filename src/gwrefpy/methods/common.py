import numpy as np
import scipy as sp


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


def _get_gwrefs_polyfit_stats(p, n, stderr, training_matrix, prediction_matrix):
    degrees_freedom = n - training_matrix.shape[1]
    ta = _t_inv((1 - p) / 2, degrees_freedom)

    column_scales = np.linalg.norm(training_matrix, axis=0)
    column_scales[column_scales == 0] = 1
    scaled_training_matrix = training_matrix / column_scales
    scaled_training_pinv = np.linalg.pinv(scaled_training_matrix)
    leverage_operator = (scaled_training_pinv @ scaled_training_pinv.T) / np.outer(
        column_scales, column_scales
    )
    leverage = np.einsum(
        "ij,jk,ik->i", prediction_matrix, leverage_operator, prediction_matrix
    )
    pred_const = ta * stderr * np.sqrt(1 + leverage)
    return pred_const, ta, leverage_operator


def compute_polyfit_residual_std_error(y, y_pred, n, degree):
    degrees_freedom = n - degree - 1
    return np.sqrt(np.sum((y - y_pred) ** 2) / degrees_freedom)


def _validate_fit_rank(rank, degree, method):
    if rank < degree + 1:
        raise ValueError(
            f"Cannot compute confidence statistics for {method}: the degree "
            f"{degree} design matrix is rank deficient ({rank} of {degree + 1})."
        )


def compute_residual_std_error(x, y, n, fit_method_func):
    """
    Computes the residual standard error of a fit.

    Parameters
    ----------
    x : pd.Series or np.ndarray
        The independent variable data.
    y : pd.Series or np.ndarray
        The dependent variable data.
    n : int
        The number of data points.
    fit_method_func : callable
        A function that takes x as input and returns the fitted y values.

    Returns
    -------
    float
        The residual standard error.

    """
    y_pred = fit_method_func(x)
    residuals = y - y_pred

    stderr = np.sum(residuals**2) - np.sum(residuals * (x - np.mean(x))) ** 2 / np.sum(
        (x - np.mean(x)) ** 2
    )
    stderr *= 1 / (n - 2)
    stderr = np.sqrt(stderr)

    return stderr


def _validate_timeseries_len(n, degree, method):
    if n < degree + 1:
        raise ValueError(
            f"Not enough data points ({n}) to fit a {method} of degree {degree}. "
            "At least degree + 1 data points are required."
        )

    if n < degree + 2:
        raise ValueError(
            f"Not enough data points ({n}) to compute statistics for {method} of "
            f"degree {degree}. At least degree + 2 data points are required."
        )

    if n < 3:
        raise ValueError(
            f"Not enough data points ({n}) to compute statistics for {method}. "
            "At least 3 data points are required."
        )


def _validate_input_timeseries(obs_timeseries, ref_timeseries):
    if obs_timeseries.empty:
        raise ValueError("The observation time series is empty.")
    if ref_timeseries.empty:
        raise ValueError("The reference time series is empty.")
    if obs_timeseries.equals(ref_timeseries):
        raise ValueError("The observation and reference time series are identical.")
