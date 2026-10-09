import logging
from collections.abc import Callable, Iterable, Iterator
from typing import TYPE_CHECKING, Literal, overload

import pandas as pd

from .well import Well

if TYPE_CHECKING:
    from .model import Model

logger = logging.getLogger(__name__)

Kind = Literal["obs", "ref"]

# Maps the values accepted by ``kind`` to the ``is_reference`` flag they select
KINDS: dict[str, bool] = {"obs": False, "ref": True}

# The columns of ``to_dataframe``, in order (fit columns depend on the well kind)
SUMMARY_COLUMNS = [
    "name",
    "well_type",
    "data_points",
    "start_date",
    "end_date",
    "mean_level",
    "latest_value",
    "latest_date",
    "latitude",
    "longitude",
    "elevation",
    "best_fit_ref_well",
    "best_rmse",
    "num_fits",
    "avg_rmse",
]


class WellCollection:
    """
    A collection of wells.

    The collection held by a model (``model.wells``) is the root collection and is
    the only one that can be modified. Collections returned by queries such as
    :meth:`filter` are detached snapshots: they can be queried further but not
    modified.

    Parameters
    ----------
    wells : Iterable[Well] | None
        The wells to include in the collection.
    """

    def __init__(self, wells: Iterable[Well] | None = None):
        self._wells: list[Well] = list(wells) if wells is not None else []
        self._is_root = False
        self._model: Model | None = None

    @classmethod
    def _root(cls, model: "Model") -> "WellCollection":
        """Create an empty, mutable root collection owned by ``model``."""
        collection = cls()
        collection._is_root = True
        collection._model = model
        return collection

    def _detached(self, wells: Iterable[Well]) -> "WellCollection":
        """Create a detached collection that keeps the link to this model."""
        collection = WellCollection(wells)
        collection._model = self._model
        return collection

    def __repr__(self) -> str:
        n_ref = sum(well.is_reference for well in self._wells)
        return (
            f"WellCollection({len(self._wells)} wells: "
            f"{len(self._wells) - n_ref} obs, {n_ref} ref)"
        )

    def _repr_html_(self) -> str:
        return self.to_dataframe()._repr_html_()

    # --- list-like access ----------------------------------------------------

    def __len__(self) -> int:
        return len(self._wells)

    def __iter__(self) -> Iterator[Well]:
        return iter(self._wells)

    def __bool__(self) -> bool:
        return bool(self._wells)

    def __contains__(self, item: object) -> bool:
        if isinstance(item, str):
            return any(well.name == item for well in self._wells)
        return any(well is item for well in self._wells)

    @overload
    def __getitem__(self, key: int | str) -> Well: ...

    @overload
    def __getitem__(self, key: slice) -> "WellCollection": ...

    def __getitem__(self, key):
        if isinstance(key, str):
            for well in self._wells:
                if well.name == key:
                    return well
            logger.error(f"No well named '{key}'.")
            raise KeyError(f"No well named '{key}'.")
        if isinstance(key, slice):
            return self._detached(self._wells[key])
        return self._wells[key]

    # --- queries -------------------------------------------------------------

    def filter(
        self,
        predicate: Callable[[Well], bool] | None = None,
        *,
        name: str | Iterable[str] | None = None,
        kind: Kind | None = None,
    ) -> "WellCollection":
        """
        Select the wells matching all the given criteria.

        Parameters
        ----------
        predicate : Callable[[Well], bool] | None
            Optional function returning True for the wells to keep, for criteria
            not covered by the keywords, e.g. ``lambda w: w.elevation > 50``.
        name : str | list[str] | None
            The well name(s). A well matches if its name is any of them.
        kind : str | None
            "obs" for observation wells or "ref" for reference wells.

        Returns
        -------
        WellCollection
            A detached collection with the matching wells.
        """
        result = self._select(predicate, name=name, kind=kind)
        if not result:
            logger.warning("No wells matched the filter.")
        return result

    def _select(
        self,
        predicate: Callable[[Well], bool] | None = None,
        *,
        name: str | Iterable[str] | None = None,
        kind: Kind | None = None,
    ) -> "WellCollection":
        """Like :meth:`filter`, without warning when nothing matches."""
        if kind is not None and kind not in KINDS:
            msg = f"Unknown kind '{kind}'. Valid kinds are {list(KINDS)}."
            logger.error(msg)
            raise ValueError(msg)
        wanted = {name} if isinstance(name, str) else name
        if wanted is not None:
            wanted = set(wanted)

        def matches(well: Well) -> bool:
            return (
                (wanted is None or well.name in wanted)
                and (kind is None or well.is_reference == KINDS[kind])
                and (predicate is None or predicate(well))
            )

        return self._detached(well for well in self._wells if matches(well))

    def get(
        self,
        predicate: Callable[[Well], bool] | None = None,
        *,
        name: str | Iterable[str] | None = None,
        kind: Kind | None = None,
    ) -> Well:
        """
        Get the single well matching the given criteria.

        Takes the same arguments as :meth:`filter`.

        Returns
        -------
        Well
            The matching well.

        Raises
        ------
        ValueError
            If no well or more than one well matches.
        """
        matches = self._select(predicate, name=name, kind=kind)
        if len(matches) != 1:
            msg = (
                "No well matched."
                if not matches
                else f"Expected one well but {len(matches)} wells matched: "
                f"{matches.names()}."
            )
            logger.error(msg)
            raise ValueError(msg)
        return matches[0]

    def obs(self) -> "WellCollection":
        """
        Get the observation wells.

        Returns
        -------
        WellCollection
            A detached collection with the observation wells.
        """
        return self._select(kind="obs")

    def ref(self) -> "WellCollection":
        """
        Get the reference wells.

        Returns
        -------
        WellCollection
            A detached collection with the reference wells.
        """
        return self._select(kind="ref")

    def to_dataframe(self) -> pd.DataFrame:
        """
        Get a summary DataFrame of the wells in the collection.

        Returns
        -------
        pd.DataFrame
            One row per well, in collection order, with columns: name, well_type,
            data_points, start_date, end_date, mean_level, latest_value,
            latest_date, latitude, longitude, elevation. Observation wells also
            get best_fit_ref_well and best_rmse, reference wells num_fits and
            avg_rmse. The fit columns are empty for a collection that doesn't
            belong to a model.
        """
        if not self._wells:
            return pd.DataFrame()

        df = pd.DataFrame([self._summary_row(well) for well in self._wells])
        return df.reindex(columns=[c for c in SUMMARY_COLUMNS if c in df.columns])

    def _summary_row(self, well: Well) -> dict:
        """The :meth:`to_dataframe` row of a single well."""
        ts = well.timeseries
        if ts is None or len(ts) == 0:
            ts = None
        row = {
            "name": well.name,
            "well_type": "reference" if well.is_reference else "observation",
            "data_points": len(ts) if ts is not None else 0,
            "start_date": ts.index.min() if ts is not None else None,
            "end_date": ts.index.max() if ts is not None else None,
            "mean_level": ts.mean() if ts is not None else None,
            "latest_value": ts.iloc[-1] if ts is not None else None,
            "latest_date": ts.index[-1] if ts is not None else None,
            "latitude": well.latitude,
            "longitude": well.longitude,
            "elevation": well.elevation,
        }

        if well.is_reference:
            fits = self._model.fits._select(ref=well) if self._model else None
            row["num_fits"] = len(fits) if fits is not None else None
            row["avg_rmse"] = (
                sum(fit.rmse for fit in fits) / len(fits) if fits else None
            )
        else:
            fits = self._model.fits._select(obs=well) if self._model else None
            best = fits.best() if fits else None
            row["best_fit_ref_well"] = best.ref_well.name if best else None
            row["best_rmse"] = best.rmse if best else None
        return row

    def names(self) -> list[str]:
        """
        Get the names of the wells in the collection.

        Returns
        -------
        list[str]
            The well names, in collection order.
        """
        return [well.name for well in self._wells]

    # --- mutation (root only) ------------------------------------------------

    def _check_root(self, action: str) -> None:
        """Raise if this is a detached collection, which can't be modified."""
        if not self._is_root:
            msg = (
                f"Cannot {action} a detached collection. Only the model's collection "
                "can be modified, e.g. model.wells.remove(...)."
            )
            logger.error(msg)
            raise TypeError(msg)

    def _add(self, well: Well) -> None:
        """Add a well (validation is done by ``Model.add_well``)."""
        self._wells.append(well)

    def remove(
        self,
        target: "Well | str | Iterable[Well | str]",
        /,
        *,
        dry_run: bool = False,
    ) -> "WellCollection":
        """
        Remove wells from the model, together with the fits that use them.

        Parameters
        ----------
        target : Well | str | WellCollection | list[Well | str]
            The well(s) to remove, as well objects or well names.
        dry_run : bool
            If True, nothing is removed; the wells that would be removed are
            returned and the number of fits that would be removed is logged.

        Returns
        -------
        WellCollection
            A detached collection with the removed wells.

        Raises
        ------
        TypeError
            If called on a detached collection.
        KeyError
            If a target well or name is not in the collection.
        """
        self._check_root("remove wells from")
        if isinstance(target, Well | str):
            target = [target]
        if not isinstance(target, Iterable):
            target = [target]
        wells: list[Well] = []
        for item in target:
            if isinstance(item, str):
                well = self[item]
            elif not isinstance(item, Well):
                msg = f"Pass Well objects or well names to remove, not {item!r}."
                logger.error(msg)
                raise TypeError(msg)
            elif item in self:
                well = item
            else:
                msg = f"Well '{item.name}' is not in the collection."
                logger.error(msg)
                raise KeyError(msg)
            if not any(w is well for w in wells):
                wells.append(well)
        removed = self._detached(wells)

        assert self._model is not None  # the root always belongs to a model
        removed_ids = {id(well) for well in removed}
        fits = self._model.fits
        well_fits = [
            fit
            for fit in fits
            if id(fit.obs_well) in removed_ids or id(fit.ref_well) in removed_ids
        ]
        if dry_run:
            logger.info(
                f"Dry run: would remove {len(removed)} well(s) and "
                f"{len(well_fits)} fit(s)."
            )
            return removed

        fits._remove_fits(well_fits, dry_run=False)
        self._wells = [well for well in self._wells if id(well) not in removed_ids]
        for well in removed:
            if self._model in well.model:
                well.model.remove(self._model)
        logger.info(f"Removed {len(removed)} well(s).")
        return removed
