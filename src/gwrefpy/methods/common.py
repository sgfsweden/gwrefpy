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
