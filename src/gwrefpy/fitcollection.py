import logging
from collections.abc import Callable, Iterable, Iterator
from typing import overload

import pandas as pd

from .fitresults import (
    ChebyshevFitResult,
    FitResultData,
    LinRegResult,
    NPolyFitResult,
)
from .well import Well

logger = logging.getLogger(__name__)

# Maps the method names used in ``Model.fit(method=...)`` to their result classes
METHODS = {
    "linearregression": LinRegResult,
    "npolyfit": NPolyFitResult,
    "chebyshev": ChebyshevFitResult,
}

# Maps the keys accepted by ``by`` to the attribute of a fit they group on
GROUP_KEYS: dict[str, Callable[[FitResultData], str]] = {
    "obs": lambda fit: fit.obs_well.name,
    "ref": lambda fit: fit.ref_well.name,
    "method": lambda fit: _method_name(fit),
}

GroupBy = str | tuple[str, ...] | None
WellSelector = Well | str | Iterable[Well | str] | None
MethodSelector = str | Iterable[str] | None


def _method_name(fit: FitResultData) -> str:
    """The ``Model.fit(method=...)`` name of the method used for a fit."""
    for name, cls in METHODS.items():
        if isinstance(fit.fit_method, cls):
            return name
    raise ValueError(f"Unknown fit method {fit.fit_method.__class__.__name__}.")


def _as_names(value: WellSelector) -> set[str] | None:
    """Normalise a well/name selector to a set of names (None means any)."""
    if value is None:
        return None
    if isinstance(value, Well | str):
        value = [value]
    return {v.name if isinstance(v, Well) else v for v in value}


def _as_methods(value: MethodSelector) -> set[str] | None:
    """Normalise a method selector to a set of method names (None means any)."""
    if value is None:
        return None
    methods = {value} if isinstance(value, str) else set(value)
    unknown = methods - METHODS.keys()
    if unknown:
        msg = f"Unknown method(s) {sorted(unknown)}. Valid methods are {list(METHODS)}."
        logger.error(msg)
        raise ValueError(msg)
    return methods


def _group_key(by: GroupBy) -> Callable[[FitResultData], tuple]:
    """Build a function returning the group a fit belongs to."""
    if by is None:
        return lambda fit: ()
    keys = (by,) if isinstance(by, str) else tuple(by)
    unknown = [k for k in keys if k not in GROUP_KEYS]
    if unknown:
        msg = f"Unknown grouping {unknown}. Valid keys are {list(GROUP_KEYS)}."
        logger.error(msg)
        raise ValueError(msg)
    return lambda fit: tuple(GROUP_KEYS[k](fit) for k in keys)


class FitCollection:
    """
    A collection of fit results.

    The collection held by a model (``model.fits``) is the root collection and is the
    only one that can be modified. Collections returned by queries such as
    :meth:`filter` are detached snapshots: they can be queried further but not
    modified.

    Parameters
    ----------
    fits : Iterable[FitResultData] | None
        The fits to include in the collection.
    """

    def __init__(self, fits: Iterable[FitResultData] | None = None):
        self._fits: list[FitResultData] = list(fits) if fits is not None else []
        self._is_root = False

    @classmethod
    def _root(cls) -> "FitCollection":
        """Create an empty, mutable root collection (owned by a model)."""
        collection = cls()
        collection._is_root = True
        return collection

    def __repr__(self) -> str:
        obs = {fit.obs_well.name for fit in self._fits}
        ref = {fit.ref_well.name for fit in self._fits}
        return (
            f"FitCollection({len(self._fits)} fits, {len(obs)} obs wells, "
            f"{len(ref)} ref wells)"
        )

    def _repr_html_(self) -> str:
        return self.to_dataframe()._repr_html_()

    # --- list-like access ----------------------------------------------------

    def __len__(self) -> int:
        return len(self._fits)

    def __iter__(self) -> Iterator[FitResultData]:
        return iter(self._fits)

    def __bool__(self) -> bool:
        return bool(self._fits)

    def __contains__(self, item: object) -> bool:
        if isinstance(item, str):
            return any(fit.name == item for fit in self._fits)
        return any(fit is item for fit in self._fits)

    @overload
    def __getitem__(self, key: int | str) -> FitResultData: ...

    @overload
    def __getitem__(self, key: slice) -> "FitCollection": ...

    def __getitem__(self, key):
        if isinstance(key, str):
            for fit in self._fits:
                if fit.name == key:
                    return fit
            logger.error(f"No fit named '{key}'.")
            raise KeyError(f"No fit named '{key}'.")
        if isinstance(key, slice):
            return FitCollection(self._fits[key])
        return self._fits[key]

    # --- queries -------------------------------------------------------------

    def filter(
        self,
        predicate: Callable[[FitResultData], bool] | None = None,
        *,
        obs: WellSelector = None,
        ref: WellSelector = None,
        method: MethodSelector = None,
        name: str | Iterable[str] | None = None,
    ) -> "FitCollection":
        """
        Select the fits matching all the given criteria.

        Each keyword accepts a single value or a list of values; a fit matches a
        keyword if it matches any of its values. A fit is selected if it matches
        every keyword given.

        Parameters
        ----------
        predicate : Callable[[FitResultData], bool] | None
            Optional function returning True for the fits to keep, for criteria not
            covered by the keywords, e.g. ``lambda f: f.rmse < 0.05``.
        obs : Well | str | list[Well | str] | None
            The observation well(s).
        ref : Well | str | list[Well | str] | None
            The reference well(s).
        method : str | list[str] | None
            The fitting method(s): "linearregression", "npolyfit" or "chebyshev".
        name : str | list[str] | None
            The fit name(s).

        Returns
        -------
        FitCollection
            A detached collection with the matching fits.
        """
        result = self._select(predicate, obs=obs, ref=ref, method=method, name=name)
        if not result:
            logger.warning("No fits matched the filter.")
        return result

    def _select(
        self,
        predicate: Callable[[FitResultData], bool] | None = None,
        *,
        obs: WellSelector = None,
        ref: WellSelector = None,
        method: MethodSelector = None,
        name: str | Iterable[str] | None = None,
    ) -> "FitCollection":
        """Like :meth:`filter`, without warning when nothing matches."""
        obs_names = _as_names(obs)
        ref_names = _as_names(ref)
        methods = _as_methods(method)
        fit_names = _as_names(name)

        def matches(fit: FitResultData) -> bool:
            return (
                (obs_names is None or fit.obs_well.name in obs_names)
                and (ref_names is None or fit.ref_well.name in ref_names)
                and (methods is None or _method_name(fit) in methods)
                and (fit_names is None or fit.name in fit_names)
                and (predicate is None or predicate(fit))
            )

        return FitCollection(fit for fit in self._fits if matches(fit))

    def get(
        self,
        predicate: Callable[[FitResultData], bool] | None = None,
        *,
        obs: WellSelector = None,
        ref: WellSelector = None,
        method: MethodSelector = None,
        name: str | Iterable[str] | None = None,
    ) -> FitResultData:
        """
        Get the single fit matching the given criteria.

        Takes the same arguments as :meth:`filter`.

        Returns
        -------
        FitResultData
            The matching fit.

        Raises
        ------
        ValueError
            If no fit or more than one fit matches.
        """
        matches = self._select(predicate, obs=obs, ref=ref, method=method, name=name)
        if len(matches) != 1:
            msg = (
                "No fit matched."
                if not matches
                else f"Expected one fit but {len(matches)} fits matched: "
                f"{[fit.name for fit in matches]}."
            )
            logger.error(msg)
            raise ValueError(msg)
        return matches[0]

    def best(self) -> FitResultData:
        """
        Get the fit with the lowest RMSE.

        Returns
        -------
        FitResultData
            The best fit. On a tie, the fit added first wins.

        Raises
        ------
        ValueError
            If the collection is empty.
        """
        if not self._fits:
            logger.error("Cannot get the best fit of an empty collection.")
            raise ValueError("Cannot get the best fit of an empty collection.")
        return min(self._fits, key=lambda fit: fit.rmse)

    def top(self, n: int = 1, by: GroupBy = None) -> "FitCollection":
        """
        Get the ``n`` fits with the lowest RMSE, optionally per group.

        Parameters
        ----------
        n : int
            The number of fits to keep (per group if ``by`` is given).
        by : str | tuple[str, ...] | None
            Rank the fits within groups instead of across the whole collection.
            Valid keys are "obs", "ref" and "method", e.g. ``by="obs"`` gives the
            best ``n`` fits for each observation well, ``by=("obs", "method")`` the
            best ``n`` for each observation well and method.

        Returns
        -------
        FitCollection
            A detached collection, grouped in order of first appearance and sorted
            by RMSE within each group. On a tie, the fit added first wins.
        """
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            logger.error("Parameter 'n' must be a positive integer.")
            raise ValueError("Parameter 'n' must be a positive integer.")
        key = _group_key(by)
        groups: dict[tuple, list[FitResultData]] = {}
        for fit in self._fits:
            groups.setdefault(key(fit), []).append(fit)
        return FitCollection(
            fit
            for group in groups.values()
            for fit in sorted(group, key=lambda f: f.rmse)[:n]
        )

    def to_dataframe(self) -> pd.DataFrame:
        """
        Get a summary DataFrame of the fits in the collection.

        Returns
        -------
        pd.DataFrame
            DataFrame with common columns (ref_well_name, obs_well_name, method,
            rmse, etc.) and method-specific columns with appropriate prefixes
            (e.g., linreg_slope, linreg_intercept)
        """
        if not self._fits:
            return pd.DataFrame()

        data = []
        for fit in self._fits:
            # Common columns from FitResultData attributes
            row = {
                "ref_well_name": fit.ref_well.name,
                "obs_well_name": fit.obs_well.name,
                "method": fit.fit_method.__class__.__name__,
                "rmse": fit.rmse,
                "n_points": fit.n,
                "stderr": fit.stderr,
                "confidence_level": fit.p,
                "calibration_start": fit.tmin,
                "calibration_end": fit.tmax,
                "time_offset": str(fit.offset),
                "t_a": fit.t_a,
                "pred_const": fit.pred_const,
            }

            # Method-specific columns with prefixes
            if isinstance(fit.fit_method, LinRegResult):
                row.update(
                    {
                        "linreg_slope": fit.fit_method.slope,
                        "linreg_intercept": fit.fit_method.intercept,
                        "linreg_rvalue": fit.fit_method.rvalue,
                        "linreg_pvalue": fit.fit_method.pvalue,
                        "linreg_stderr": fit.fit_method.stderr,
                    }
                )
            # Future fitting methods would be added here as elif branches

            data.append(row)

        return pd.DataFrame(data)

    # --- mutation (root only) ------------------------------------------------

    def _check_root(self, action: str) -> None:
        """Raise if this is a detached collection, which can't be modified."""
        if not self._is_root:
            msg = (
                f"Cannot {action} a detached collection. Only the model's collection "
                "can be modified, e.g. model.fits.remove(...)."
            )
            logger.error(msg)
            raise TypeError(msg)

    def _add_or_replace(self, fit: FitResultData) -> None:
        """Add a fit, replacing any existing fit with the same name in place."""
        for i, existing in enumerate(self._fits):
            if existing.name == fit.name:
                self._fits[i] = fit
                logger.info(f"Replaced existing fit '{fit.name}'.")
                return
        self._fits.append(fit)

    def _add_renaming(self, fit: FitResultData) -> None:
        """Add a fit, renaming it with a ``#2``, ``#3``... suffix on a name clash.

        Used when loading saved models, where replacing would silently lose fits.
        """
        if fit.name in self:
            i = 2
            while f"{fit.name}#{i}" in self:
                i += 1
            new_name = f"{fit.name}#{i}"
            logger.warning(
                f"Duplicate fit name '{fit.name}' while loading; renamed to "
                f"'{new_name}'."
            )
            fit.name = new_name
        self._fits.append(fit)

    def _remove_fits(
        self, fits: Iterable[FitResultData], dry_run: bool
    ) -> "FitCollection":
        """Remove the given fits (unless dry_run) and return them as a collection."""
        removed = FitCollection(fits)
        if not dry_run:
            removed_ids = {id(fit) for fit in removed}
            self._fits = [fit for fit in self._fits if id(fit) not in removed_ids]
            if removed:
                logger.info(f"Removed {len(removed)} fit(s).")
        return removed

    def remove(
        self,
        target: "FitResultData | str | Iterable[FitResultData | str] | None" = None,
        /,
        *,
        obs: WellSelector = None,
        ref: WellSelector = None,
        method: MethodSelector = None,
        dry_run: bool = False,
    ) -> "FitCollection":
        """
        Remove fits from the model.

        Either pass the fits to remove as ``target``, or select them with the
        ``obs``, ``ref`` and ``method`` keywords (as in :meth:`filter`).

        Parameters
        ----------
        target : FitResultData | str | FitCollection | list[FitResultData | str]
            The fit(s) to remove, as fit objects or fit names.
        obs : Well | str | list[Well | str] | None
            Remove the fits of these observation well(s).
        ref : Well | str | list[Well | str] | None
            Remove the fits of these reference well(s).
        method : str | list[str] | None
            Remove the fits made with these method(s).
        dry_run : bool
            If True, nothing is removed; the fits that would be removed are
            returned.

        Returns
        -------
        FitCollection
            A detached collection with the removed fits.

        Raises
        ------
        TypeError
            If called on a detached collection, without arguments, or with both a
            target and keywords.
        KeyError
            If a target fit or name is not in the collection.
        """
        self._check_root("remove fits from")
        has_filter = any(v is not None for v in (obs, ref, method))
        if target is None and not has_filter:
            msg = (
                "Pass the fits to remove or filter keywords. Use clear() to remove all."
            )
            logger.error(msg)
            raise TypeError(msg)
        if target is not None and has_filter:
            msg = "Pass either the fits to remove or filter keywords, not both."
            logger.error(msg)
            raise TypeError(msg)

        if target is None:
            return self._remove_fits(
                self.filter(obs=obs, ref=ref, method=method), dry_run
            )

        if isinstance(target, FitResultData | str):
            target = [target]
        fits = []
        for item in target:
            if isinstance(item, str):
                fits.append(self[item])
            elif item in self:
                fits.append(item)
            else:
                msg = f"Fit '{item.name}' is not in the collection."
                logger.error(msg)
                raise KeyError(msg)
        return self._remove_fits(fits, dry_run)

    def clear(self) -> None:
        """Remove all fits from the model."""
        self._check_root("clear")
        self._fits = []
        logger.info("Removed all fits.")

    def keep_best(
        self, n: int, *, by: GroupBy, dry_run: bool = False
    ) -> "FitCollection":
        """
        Keep only the ``n`` best fits (lowest RMSE) of each group, remove the rest.

        Groups with ``n`` fits or fewer are left as they are.

        Parameters
        ----------
        n : int
            The number of fits to keep per group.
        by : str | tuple[str, ...] | None
            The grouping, see :meth:`top`. Required, to make the scope of the
            removal explicit; pass None to keep the best ``n`` fits overall.
        dry_run : bool
            If True, nothing is removed; the fits that would be removed are
            returned.

        Returns
        -------
        FitCollection
            A detached collection with the removed fits.
        """
        self._check_root("remove fits from")
        keep_ids = {id(fit) for fit in self.top(n, by=by)}
        return self._remove_fits(
            (fit for fit in self._fits if id(fit) not in keep_ids), dry_run
        )
