import logging
import warnings
from typing import Literal

import pandas as pd

from .fitcollection import METHODS, FitCollection
from .fitresults import FitResultData
from .methods.chebyshev import chebyshevfit
from .methods.linregressfit import linregressfit
from .methods.npolyfit import npolyfit
from .well import Well

logger = logging.getLogger(__name__)


def _default_fit_name(obs_well: Well, ref_well: Well, method: str, degree: int) -> str:
    """Readable default fit name, e.g. ``"obs~ref:npolyfit4"``."""
    label = method
    if method in ("npolyfit", "chebyshev"):
        label += str(degree)
    return f"{obs_well.name}~{ref_well.name}:{label}"


class FitBase:
    def __init__(self):
        self.ref_wells = None
        self.name = None
        self.fits: FitCollection = FitCollection._root()

    def fit(
        self,
        obs_well: Well | list[Well] | str | list[str],
        ref_well: Well | list[Well] | str | list[str],
        offset: pd.DateOffset | pd.Timedelta | str,
        aggregation: Literal["mean", "median", "min", "max"] = "mean",
        p: float = 0.95,
        method: Literal[
            "linearregression", "npolyfit", "chebyshev"
        ] = "linearregression",
        tmin: pd.Timestamp | str | None = None,
        tmax: pd.Timestamp | str | None = None,
        shift: pd.Timedelta | str | None = None,
        name: str | list[str] | None = None,
        report: bool = True,
        **kwargs,
    ) -> FitResultData | FitCollection:
        """
        Fit reference well(s) to observation well(s) using regression.

        Parameters
        ----------
        obs_well : Well | list[Well] | str | list[str]
            The observation well(s) to use for fitting. Can be Well objects,
            well names (strings), or lists of either. If a list is provided,
            each well will be paired with the corresponding reference well by index.
        ref_well : Well | list[Well] | str | list[str]
            The reference well(s) to use for fitting. Can be Well objects,
            well names (strings), or lists of either. If a list is provided,
            each well will be paired with the corresponding observation well by index.
        offset: pd.DateOffset | pd.Timedelta | str
            The offset to apply to the time series when grouping within time
            equivalents.
        aggregation: Literal["mean", "median", "min", "max"], optional
            The aggregation method to use when grouping data points within time
            equivalents (default is "mean").
        p : float, optional
            The confidence level for the fit (default is 0.95).
        method : Literal["linearregression", "npolyfit", "chebyshev"]
            Method with which to perform regression. Currently supports
            linear regression, N-th degree polynomial fit, and Chebyshev polynomial fit.
        tmin: pd.Timestamp | str | None = None
            Minimum time for calibration period.
        tmax: pd.Timestamp | str | None = None
            Maximum time for calibration period.
        shift : pd.Timedelta | str | None, optional
            An optional time shift to apply to the observation well time series before
            fitting.
        name : str | list[str] | None, optional
            An optional name or list of names for the fit result(s). If lists of
            wells are provided, names must be a list matching in length. If None,
            default names like ``"obs~ref:linearregression"`` are assigned. A fit
            replaces any existing fit with the same name.
        report: bool, optional
            Whether to print fit results summary (default is True).
        **kwargs
            Additional keyword arguments to pass to the fitting method.
            For example, you can use `degree` (default is 4) when using the `npolyfit`
            or `chebyshev` methods.

        Returns
        -------
        FitResultData | FitCollection
            If single wells are provided, returns a single FitResultData object.
            If lists of wells are provided, returns a detached FitCollection with
            the fit for each obs_well/ref_well pair.

        Raises
        ------
        ValueError
            If lists are provided but have different lengths.
        """
        # Resolve wells (convert strings to Well objects and normalize to lists)
        obs_wells = self._resolve_wells(obs_well)
        ref_wells = self._resolve_wells(ref_well)

        # Handle single well case
        if len(obs_wells) == 1 and len(ref_wells) == 1:
            result = self._fit(
                obs_wells[0],
                ref_wells[0],
                offset,
                p,
                method,
                tmin,
                tmax,
                shift,
                name,
                aggregation,
                **kwargs,
            )
            logger.info(
                f"Fitting model '{self.name}' using reference well "
                f"'{ref_wells[0].name}' and observation well '{obs_wells[0].name}'."
            )
            if report:
                self._display_result(result)
            return result

        # Validate that lists have the same length
        if len(obs_wells) != len(ref_wells):
            error_msg = (
                f"obs_well list length ({len(obs_wells)}) must match "
                f"ref_well list length ({len(ref_wells)})"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Validate the name parameter if provided as a list
        if isinstance(name, list):
            if len(name) != len(obs_wells):
                error_msg = (
                    f"name list length ({len(name)}) must match "
                    f"obs_well/ref_well list length ({len(obs_wells)})"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

        # Perform fitting for each pair
        results = []
        if isinstance(name, str):
            # Fit names are unique, so one name would make each fit replace the last
            error_msg = (
                "When fitting lists of wells, pass a list of names (one per pair) "
                "or None for default names."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        if name is None:
            name = [None] * len(obs_wells)
        for obs_w, ref_w, fit_name in zip(obs_wells, ref_wells, name, strict=True):
            result = self._fit(
                obs_w,
                ref_w,
                offset,
                p,
                method,
                tmin,
                tmax,
                shift,
                fit_name,
                aggregation,
                **kwargs,
            )
            results.append(result)
            logger.info(
                f"Fitting model '{self.name}' using reference well '{ref_w.name}' "
                f"and observation well '{obs_w.name}'."
            )
            if report:
                self._display_result(result)

        return FitCollection(results)

    def _fit(
        self,
        obs_well: Well,
        ref_well: Well,
        offset: pd.DateOffset | pd.Timedelta | str,
        p: float = 0.95,
        method: Literal[
            "linearregression", "npolyfit", "chebyshev"
        ] = "linearregression",
        tmin: pd.Timestamp | str | None = None,
        tmax: pd.Timestamp | str | None = None,
        shift: pd.Timedelta | str | None = None,
        name: str | None = None,
        aggregation: Literal["mean", "median", "min", "max"] = "mean",
        **kwargs,
    ) -> FitResultData:
        # Check that the ref_well is a reference well
        if not ref_well.is_reference:
            logger.error(f"The well '{ref_well.name}' is not a reference well.")
            raise ValueError(f"The well '{ref_well.name}' is not a reference well.")

        # Check that the obs_well is an observation well
        if obs_well.is_reference:
            logger.error(f"The well '{obs_well.name}' is not an observation well.")
            raise ValueError(f"The well '{obs_well.name}' is not an observation well.")

        degree = kwargs.get("degree", 4)
        if name is None:
            name = _default_fit_name(obs_well, ref_well, method, degree)

        fit = None
        if method == "linearregression":
            logger.debug("Using linear regression method for fitting.")
            fit = linregressfit(
                obs_well, ref_well, offset, tmin, tmax, shift, name, p, aggregation
            )
        elif method == "npolyfit":
            logger.debug("Using Nth degree polynomial fit method for fitting.")
            fit = npolyfit(
                obs_well,
                ref_well,
                offset,
                degree,
                tmin,
                tmax,
                shift,
                name,
                p,
                aggregation,
            )
        elif method == "chebyshev":
            logger.debug("Using Chebyshev polynomial fit method for fitting.")
            fit = chebyshevfit(
                obs_well,
                ref_well,
                offset,
                degree,
                tmin,
                tmax,
                shift,
                name,
                p,
                aggregation,
            )
        if fit is None:
            logger.error(f"Fitting method '{method}' is not implemented.")
            raise NotImplementedError(f"Fitting method '{method}' is not implemented.")

        self.fits._add_or_replace(fit)
        logger.debug(f"Fit completed for model '{self.name}' with RMSE {fit.rmse}.")
        return fit

    def best_fit(
        self,
        obs_well: str | Well,
        ref_wells: list[str | Well] | None = None,
        method: Literal[
            "linearregression", "npolyfit", "chebyshev"
        ] = "linearregression",
        skip_raise_error: bool = False,
        **kwargs,
    ) -> FitResultData:
        """
        Find the best fit for the model using the provided wells.
        Note that all successful fits will be stored in the model's `fits` attribute.
        The best fit is determined by the lowest RMSE value.

        Parameters
        ----------
        obs_well : Well or list of Well or None, optional
            The observation well to use for fitting.
        ref_wells : Well or list of Well or None, optional
            The reference wells to test. If None, all reference wells in the
            model will be used (default is None).
        method : Literal["linearregression", "npolyfit", "chebyshev"]
            Method with which to perform regression. Currently only supports
            linear regression.
        skip_raise_error: bool, optional
            If True, will skip wells that raise exceptions during fitting and continue.
            If False, will raise the exception and stop execution (default is False).
        **kwargs
            Keyword arguments to pass to the fitting method. For example, you can use
            `offset`, `p`, `tmin`, `tmax`, and `aggregation` to control the fitting

        Returns
        -------
        FitResultData
            Returns the best fit for the given observation well.
        """
        return self._best_fit(
            obs_well, ref_wells, method, skip_raise_error=skip_raise_error, **kwargs
        )

    def _best_fit(
        self,
        obs_well: str | Well,
        ref_wells: list[str | Well] | None = None,
        method: Literal[
            "linearregression", "npolyfit", "chebyshev"
        ] = "linearregression",
        skip_raise_error: bool = False,
        **kwargs,
    ) -> FitResultData:
        """
        The internal method to find the best fit.

        Parameters
        ----------
        obs_well : Well or list of Well or None, optional
            The observation well to use for fitting.
        ref_wells : Well or list of Well or None, optional
            The reference wells to test. If None, all reference wells in the
            model will be used (default is None).
        method : Literal["linearregression", "npolyfit", "chebyshev"]
            Method with which to perform regression. Currently only supports
            linear regression.
        skip_raise_error: bool, optional
            If True, will skip wells that raise exceptions during fitting and continue.
            If False, will raise the exception and stop execution (default is False).
        **kwargs
            Keyword arguments to pass to the fitting method. For example, you can use
            `offset`, `p`, `tmin`, `tmax`, and `aggregation` to control the fitting

        Returns
        -------
        FitResultData
            Returns the best fit for the given arguments.
        """
        if isinstance(ref_wells, list) and len(ref_wells) < 1:
            logger.error("ref_wells list cannot be empty.")
            raise ValueError("ref_wells list cannot be empty.")

        target_obs_well: Well
        if isinstance(obs_well, str):
            target_obs_well = self.get_wells(obs_well)
            if isinstance(target_obs_well, list):
                logger.error(
                    "obs_well parameter must resolve to a single well, not a list."
                )
                raise ValueError(
                    "obs_well parameter must resolve to a single well, not a list."
                )
        elif isinstance(obs_well, Well):
            target_obs_well = obs_well
        if ref_wells is None:
            target_ref_wells = self.ref_wells
            if len(target_ref_wells) < 1:
                logger.error("No reference wells available in the model.")
                raise ValueError("No reference wells available in the model.")
        else:
            target_ref_wells: list[Well] = []
            for rw in ref_wells:
                if isinstance(rw, str):
                    target_ref_wells.append(self.get_wells(rw))  # type: ignore
                elif isinstance(rw, Well):
                    target_ref_wells.append(rw)
                else:
                    logger.error(
                        f"Unsupported type for {rw}. Supported types are Well or str"
                    )
                    raise TypeError(
                        f"Unsupported type for {rw}. Supported types are Well or str"
                    )

        local_fits: list[FitResultData] = []
        if skip_raise_error:
            for ref_well in target_ref_wells:
                logger.debug(
                    f"Testing fit for observation well '{target_obs_well.name}' "
                    f"and reference well '{ref_well.name}'."
                )
                try:
                    fit = self._fit(target_obs_well, ref_well, method=method, **kwargs)
                    local_fits.append(fit)
                    logger.debug(
                        f"Fit result for observation well '{target_obs_well.name}' and"
                        f"reference well '{ref_well.name}': RMSE={fit.rmse}"
                    )
                except Exception as e:
                    logger.error(
                        f"Error occurred while fitting observation well "
                        f"'{target_obs_well.name}' "
                        f"and reference well '{ref_well.name}': {e}"
                    )
        else:
            for ref_well in target_ref_wells:
                logger.debug(
                    f"Testing fit for observation well '{target_obs_well.name}' "
                    f"and reference well '{ref_well.name}'."
                )
                fit = self._fit(target_obs_well, ref_well, method=method, **kwargs)
                local_fits.append(fit)
                logger.debug(
                    f"Fit result for observation well '{target_obs_well.name}' and"
                    f"reference well '{ref_well.name}': RMSE={fit.rmse}"
                )

        if not local_fits:
            logger.error(
                f"No successful fits found for observation well "
                f"'{target_obs_well.name}' "
                f"with the provided reference wells."
            )
            raise ValueError(
                f"No successful fits found for observation well "
                f"'{target_obs_well.name}' "
                f"with the provided reference wells."
            )

        return min(local_fits, key=lambda x: x.rmse)

    def get_fits(
        self, well: Well | str, method: str | None = None
    ) -> list[FitResultData] | FitResultData | None:
        """
        Get all fit results involving a specific well.

        .. deprecated:: 1.1.0
            Use ``model.fits.filter(obs=..., ref=..., method=...)`` instead.

        Parameters
        ----------
        well : Well | str
            The well to check.
        method : str | None, optional
            The fitting method to filter by (default is None, which returns all
            fits involving the well). Options are "linearregression", "npolyfit", and
            "chebyshev".

        Returns
        -------
        list[FitResultData] | FitResultData | None
            A list of fit results involving the specified well.
        """
        warnings.warn(
            "get_fits() is deprecated and will be removed in 2.0. "
            "Use model.fits.filter(obs=..., ref=..., method=...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        target_well: Well
        if isinstance(well, str):
            target_well = self.get_wells(well)  # type: ignore
        elif isinstance(well, Well):
            target_well = well
        else:
            logger.error("Parameter 'well' must be a Well instance or a string.")
            raise TypeError("Parameter 'well' must be a Well instance or a string.")

        # Unknown methods were ignored before the deprecation; keep doing so
        if method not in METHODS:
            method = None
        fit_list = list(
            self.fits._select(lambda fit: fit.has_well(target_well), method=method)
        )
        return (
            fit_list
            if len(fit_list) > 1
            else (fit_list[0] if len(fit_list) == 1 else None)
        )

    def remove_fits_by_n(self, obs_well: Well | str, n: int) -> None:
        """
        Remove fit results for a specific observation well. Keeps only the best n fits
        based on RMSE.

        .. deprecated:: 1.1.0
            Use ``model.fits.keep_best(n, by="obs")`` instead.

        Parameters
        ----------
        obs_well : Well | str
            The observation well to check.
        n : int
            The n value of the fit results to remove.

        Returns
        -------
        None
            Removes fits from the model's fit list.
        """
        warnings.warn(
            "remove_fits_by_n() is deprecated and will be removed in 2.0. "
            'Use model.fits.keep_best(n, by="obs") instead.',
            DeprecationWarning,
            stacklevel=2,
        )
        # Check that n is a positive integer
        if not isinstance(n, int) or n < 1:
            logger.error("Parameter 'n' must be a positive integer.")
            raise ValueError("Parameter 'n' must be a positive integer.")

        # Resolve the observation well
        target_well: Well
        if isinstance(obs_well, str):
            target_well = self.get_wells(obs_well)
        elif isinstance(obs_well, Well):
            target_well = obs_well
        else:
            logger.error("Parameter 'obs_well' must be a Well instance or a string.")
            raise TypeError("Parameter 'obs_well' must be a Well instance or a string.")

        # Make sure the target well is an observation well
        if target_well.is_reference:
            logger.error(f"The well '{target_well.name}' is not an observation well.")
            raise ValueError(
                f"The well '{target_well.name}' is not an observation well."
            )

        obs_fits = self.fits._select(obs=target_well)
        if not obs_fits:
            logger.warning(f"No fits found for well '{target_well.name}'.")
            return

        if len(obs_fits) <= n:
            logger.warning(
                f"Number of fits for well '{target_well.name}' ({len(obs_fits)}) "
                f"is less than or equal to n ({n}). No fits removed."
            )
            return

        best = obs_fits.top(n)
        self.fits.remove([fit for fit in obs_fits if fit not in best])

        logger.info(
            f"Removed fits for well '{target_well.name}' to retain only the best {n} "
            f"fits."
        )

    def _resolve_wells(self, obs_well):
        raise NotImplementedError("_resolve_wells must be implemented in subclass")

    def get_wells(self, obs_well):
        raise NotImplementedError("get_wells must be implemented in subclass")

    def _display_result(self, result):
        raise NotImplementedError("_display_result must be implemented in subclass")
