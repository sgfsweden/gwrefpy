import copy
import logging
import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal

import pandas as pd

from .fitcollection import FitCollection
from .fitresults import FitResultData
from .methods.chebyshev import Chebyshev
from .methods.common import FitMethodBase, _validate_input_timeseries
from .methods.linregressfit import Linear
from .methods.npolyfit import NPoly
from .methods.timeseries import groupby_time_equivalents
from .well import Well

if TYPE_CHECKING:
    from .wellcollection import WellCollection

logger = logging.getLogger(__name__)

# Maps the method names accepted by the deprecated get_fits() to the current ones
LEGACY_METHOD_NAMES = {
    "linearregression": "linear",
    "npolyfit": "npoly",
    "chebyshev": "chebyshev",
}


def _method_label(method: FitMethodBase) -> str:
    """Short readable label for a fit method, e.g. ``"Linear"`` or ``"NPoly(4)"``."""
    if isinstance(method, NPoly):
        return f"NPoly({method.degree})"
    if isinstance(method, Chebyshev):
        return f"Chebyshev({method.degree})"
    if isinstance(method, Linear):
        return "Linear"
    return method.__class__.__name__.lower()


def _default_fit_name(obs_well: Well, ref_well: Well, method: FitMethodBase) -> str:
    """Readable default fit name, e.g. ``"obs~ref:NPoly(4)"``."""
    return f"{obs_well.name}~{ref_well.name}:{_method_label(method)}"


class FitBase:
    wells: "WellCollection"

    def __init__(self):
        self.ref_wells = None
        self.name = None
        self.fits: FitCollection = FitCollection._root()

    def fit(
        self,
        obs_well: Well | str | Iterable[Well | str],
        ref_well: Well | str | Iterable[Well | str],
        offset: pd.DateOffset | pd.Timedelta | str,
        aggregation: Literal["mean", "median", "min", "max"] = "mean",
        p: float = 0.95,
        method: FitMethodBase | None = None,
        tmin: pd.Timestamp | str | None = None,
        tmax: pd.Timestamp | str | None = None,
        shift: pd.Timedelta | str | None = None,
        name: str | list[str] | None = None,
        report: bool = True,
        skip_raise_error: bool = False,
        **kwargs,
    ) -> FitResultData | FitCollection:
        """
        Fit reference well(s) to observation well(s) using regression.

        Wells are paired up as follows: a single well on one side is paired with
        every well on the other side, and two lists of the same length are paired
        by index. For example, ``model.fit("obs", model.wells.ref(), offset="0D")``
        fits ``obs`` against every reference well, and ``.best()`` on the result
        picks the fit with the lowest RMSE.

        Parameters
        ----------
        obs_well : Well | str | Iterable[Well | str]
            The observation well(s) to use for fitting. Can be Well objects,
            well names (strings), or a list or WellCollection of either.
        ref_well : Well | str | Iterable[Well | str]
            The reference well(s) to use for fitting. Can be Well objects,
            well names (strings), or a list or WellCollection of either.
        offset: pd.DateOffset | pd.Timedelta | str
            The offset to apply to the time series when grouping within time
            equivalents.
        aggregation: Literal["mean", "median", "min", "max"], optional
            The aggregation method to use when grouping data points within time
            equivalents (default is "mean").
        p : float, optional
            The confidence level for the fit (default is 0.95).
        method : FitMethodBase | None
            Method with which to perform regression. Currently supports
            Linear() linear regression, NPoly() N-th degree polynomial fit, and
            Chebyshev() Chebyshev polynomial fit.
        tmin: pd.Timestamp | str | None = None
            Minimum time for calibration period.
        tmax: pd.Timestamp | str | None = None
            Maximum time for calibration period.
        shift : pd.Timedelta | str | None, optional
            An optional time shift to apply to the observation well time series before
            fitting.
        name : str | list[str] | None, optional
            An optional name or list of names for the fit result(s). If lists of
            wells are provided, names must be a list with one name per pair. If None,
            default names like ``"obs~ref:Linear"`` or ``"obs~ref:NPoly(4)"`` are
            assigned. A fit replaces any existing fit with the same name.
        report: bool, optional
            Whether to print fit results summary (default is True).
        skip_raise_error: bool, optional
            When fitting lists of wells, skip (and log) pairs that fail instead of
            raising. Has no effect when two single wells are given (default is
            False).
        **kwargs
            Additional keyword arguments, e.g. ``te_method`` ("anchor" or
            "consecutive") for grouping the time equivalents.

        Returns
        -------
        FitResultData | FitCollection
            If two single wells are provided, returns a single FitResultData object.
            If a list or collection is provided for either argument, returns a
            detached FitCollection with the fit for each obs_well/ref_well pair,
            even when there is only one pair.

        Raises
        ------
        ValueError
            If a well list is empty, if two lists longer than one have different
            lengths, or if ``skip_raise_error`` is True and every pair fails.
        """
        # Two single wells give a single fit, anything list-like gives a collection
        single = isinstance(obs_well, str | Well) and isinstance(ref_well, str | Well)

        # Resolve wells (convert strings to Well objects and normalize to lists)
        obs_wells = self._resolve_wells(obs_well)
        ref_wells = self._resolve_wells(ref_well)
        if not obs_wells or not ref_wells:
            error_msg = "obs_well and ref_well cannot be empty."
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Pair a single well on either side with every well on the other side
        if len(obs_wells) == 1:
            obs_wells = obs_wells * len(ref_wells)
        elif len(ref_wells) == 1:
            ref_wells = ref_wells * len(obs_wells)

        # Validate that lists have the same length
        if len(obs_wells) != len(ref_wells):
            error_msg = (
                f"obs_well list length ({len(obs_wells)}) must match "
                f"ref_well list length ({len(ref_wells)})"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Validate the name parameter
        if single:
            names = [name]
        elif isinstance(name, str):
            # Fit names are unique, so one name would make each fit replace the last
            error_msg = (
                "When fitting lists of wells, pass a list of names (one per pair) "
                "or None for default names."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        elif name is None:
            names = [None] * len(obs_wells)
        elif len(name) != len(obs_wells):
            error_msg = (
                f"name list length ({len(name)}) must match "
                f"the number of obs_well/ref_well pairs ({len(obs_wells)})"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        else:
            names = name

        # Perform fitting for each pair
        results = []
        for obs_w, ref_w, fit_name in zip(obs_wells, ref_wells, names, strict=True):
            try:
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
            except Exception as e:
                if single or not skip_raise_error:
                    raise
                logger.warning(
                    f"Skipping fit of observation well '{obs_w.name}' and "
                    f"reference well '{ref_w.name}': {e}"
                )
                continue
            results.append(result)
            logger.info(
                f"Fitting model '{self.name}' using reference well '{ref_w.name}' "
                f"and observation well '{obs_w.name}'."
            )
            if report:
                self._display_result(result)

        if not results:
            error_msg = "No successful fits for the given obs_well/ref_well pairs."
            logger.error(error_msg)
            raise ValueError(error_msg)
        if single:
            return results[0]
        return FitCollection(results)

    def _fit(
        self,
        obs_well: Well,
        ref_well: Well,
        offset: pd.DateOffset | pd.Timedelta | str,
        p: float = 0.95,
        method: FitMethodBase | None = None,
        tmin: pd.Timestamp | str | None = None,
        tmax: pd.Timestamp | str | None = None,
        shift: pd.Timedelta | str | None = None,
        name: str | None = None,
        aggregation: Literal["mean", "median", "min", "max"] = "mean",
        te_method: Literal["anchor", "consecutive"] = "anchor",
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

        # Check method type
        if method is None:
            method = Linear()
        if not isinstance(method, FitMethodBase):
            error_msg = (
                f"Invalid method type: {type(method)}. "
                "Must be an instance of FitMethodBase."
            )
            logger.error(error_msg)
            raise TypeError(error_msg)
        # Fitting stores the results on the method, so each fit gets its own copy
        method = copy.deepcopy(method)

        if name is None:
            name = _default_fit_name(obs_well, ref_well, method)

        # Validate timeseries input
        _validate_input_timeseries(obs_well.timeseries, ref_well.timeseries)

        if shift is not None:
            obs_timeseries = obs_well.shift_timeseries(shift)
        else:
            obs_timeseries = obs_well.timeseries

        ref_timeseries, obs_timeseries, n = groupby_time_equivalents(
            obs_timeseries.loc[tmin:tmax],
            ref_well.timeseries.loc[tmin:tmax],
            offset,
            aggregation,
            te_method,
        )
        if tmin is None:
            logger.info("tmin is None, setting to min common time of both wells")
            tmin = obs_timeseries.index[0]
        if tmax is None:
            logger.info("tmax is None, setting to max common time of both wells")
            tmax = obs_timeseries.index[-1]

        method.fit(obs_timeseries, ref_timeseries, p=p)
        fit = FitResultData(
            obs_well=obs_well,
            ref_well=ref_well,
            n=len(obs_timeseries),
            fit_method=method,
            p=p,
            offset=offset,
            aggregation=aggregation,
            te_method=te_method,
            tmin=tmin,
            tmax=tmax,
            shift=shift,
            name=name,
        )

        self.fits._add_or_replace(fit)
        logger.debug(
            f"Fit completed for model '{self.name}' with RMSE {fit.fit_method.rmse}."
        )
        return fit

    def best_fit(
        self,
        obs_well: str | Well,
        ref_wells: list[str | Well] | None = None,
        method: FitMethodBase | None = None,
        skip_raise_error: bool = False,
        **kwargs,
    ) -> FitResultData:
        """
        Fit an observation well against several reference wells and return the
        fit with the lowest RMSE. All successful fits are added to ``model.fits``.

        .. deprecated:: 1.1.0
            Use ``model.fit(obs_well, ref_wells, ...).best()`` instead.

        Parameters
        ----------
        obs_well : str | Well
            The observation well to use for fitting.
        ref_wells : list[str | Well] | None, optional
            The reference wells to test. If None, all reference wells in the
            model will be used (default is None).
        method : FitMethodBase | None, optional
            Method with which to perform regression, e.g. ``NPoly(degree=2)``
            (default is ``Linear()``).
        skip_raise_error: bool, optional
            If True, will skip wells that raise exceptions during fitting and continue.
            If False, will raise the exception and stop execution (default is False).
        **kwargs
            Keyword arguments to pass to ``fit()``, e.g. ``offset``, ``p``, ``tmin``,
            ``tmax`` and ``aggregation``.

        Returns
        -------
        FitResultData
            Returns the best fit for the given observation well.
        """
        # TODO: remove in v2
        warnings.warn(
            "best_fit() is deprecated and will be removed in 2.0. "
            "Use model.fit(obs_well, ref_wells, ...).best() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if ref_wells is None:
            ref_wells = list(self.wells.ref())
        # best_fit never printed reports, so a passed report argument is ignored
        kwargs.pop("report", None)
        return self.fit(
            obs_well,
            ref_wells,
            method=method,
            report=False,
            skip_raise_error=skip_raise_error,
            **kwargs,
        ).best()

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
            target_well = self._get_well(well)
        elif isinstance(well, Well):
            target_well = well
        else:
            logger.error("Parameter 'well' must be a Well instance or a string.")
            raise TypeError("Parameter 'well' must be a Well instance or a string.")

        # Unknown methods were ignored before the deprecation; keep doing so
        method = LEGACY_METHOD_NAMES.get(method) if method is not None else None
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
            target_well = self._get_well(obs_well)
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

    def _get_well(self, name):
        raise NotImplementedError("_get_well must be implemented in subclass")

    def _display_result(self, result):
        raise NotImplementedError("_display_result must be implemented in subclass")
