import logging
import warnings

import pandas as pd

from . import __version__
from .fitbase import FitBase
from .fitcollection import FitCollection
from .fitresults import FitResultData, _unpack_dict_fit_method
from .io.io import load, save
from .plotter import Plotter
from .utils.conversions import float_to_datetime
from .well import Well
from .wellcollection import WellCollection

logger = logging.getLogger(__name__)


def _deprecated(old: str, new: str) -> None:
    """Warn that the model API ``old`` is deprecated in favour of ``new``."""
    warnings.warn(
        f"{old} is deprecated and will be removed in 2.0. Use {new} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


class Model(FitBase, Plotter):
    """
    A class representing a groundwater model that can contain multiple wells.

    Parameters
    ----------
    name : str
        The name of the model. If the name ends with ".gwref", it is treated as a
        filename and the model is loaded from that file.
    """

    def __init__(self, name: str):
        super(FitBase, self).__init__()
        super(Plotter, self).__init__()
        self.name = name

        # Well attributes
        self.wells: WellCollection = WellCollection._root(self)

        # Fit attributes
        self.fits: FitCollection = FitCollection._root()

        # Check if the name ends with the .gwref extension
        ext = name.split(".")[-1].lower()
        if ext == "gwref":
            self.open_project(name)
        elif ext != name.lower():
            name_ = name.split(".")[0]
            logger.warning(
                f"Model name '{name}' has an unrecognized extension '.{ext}'. "
                f"Proceeding with {name_} as name."
            )
            self.name = name_

    def __str__(self):
        """String representation of the Model object."""
        obs_count = len(self.wells.obs())
        ref_count = len(self.wells.ref())
        fits_count = len(self.fits)

        lines = [
            f"Model: {self.name}",
            "=" * (7 + len(self.name)),
            f"Wells: {len(self.wells)} total "
            f"({obs_count} observation, {ref_count} reference)",
            f"Fits: {fits_count}",
        ]

        if self.wells:
            lines.append("")
            lines.append("Wells:")
            for well in self.wells:
                well_type = "ref" if well.is_reference else "obs"
                data_points = len(well.timeseries) if hasattr(well, "timeseries") else 0
                lines.append(
                    f"  • {well.name} ({well_type}) - {data_points} data points"
                )

        if self.fits:
            lines.append("")
            lines.append("Recent fits:")
            for fit in self.fits[-3:]:  # Show last 3 fits
                lines.append(
                    f"  • {fit.obs_well.name} ~ {fit.ref_well.name}: "
                    f"RMSE={fit.rmse:.4f}"
                )
            if len(self.fits) > 3:
                lines.append(f"  ... and {len(self.fits) - 3} more")

        return "\n".join(lines)

    def __repr__(self):
        """Concise representation of the Model object for debugging."""
        return (
            f"Model(name='{self.name}', wells={len(self.wells)}, fits={len(self.fits)})"
        )

    # ======================== Well Management Methods ========================

    @property
    def obs_wells(self) -> list[Well]:
        """
        List of observation wells in the model.

        .. deprecated:: 1.1.0
            Use ``model.wells.obs()`` instead.
        """
        _deprecated("obs_wells", "model.wells.obs()")
        return list(self.wells.obs())

    @property
    def ref_wells(self) -> list[Well]:
        """
        List of reference wells in the model.

        .. deprecated:: 1.1.0
            Use ``model.wells.ref()`` instead.
        """
        _deprecated("ref_wells", "model.wells.ref()")
        return list(self.wells.ref())

    @property
    def well_names(self) -> list[str]:
        """
        List of all well names in the model.

        .. deprecated:: 1.1.0
            Use ``model.wells.names()`` instead.
        """
        _deprecated("well_names", "model.wells.names()")
        return self.wells.names()

    def obs_wells_summary(self) -> pd.DataFrame:
        """
        Get a summary DataFrame of observation wells in the model.

        .. deprecated:: 1.1.0
            Use ``model.wells.obs().to_dataframe()`` instead.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: name, data_points, start_date, end_date,
            mean_level, latest_value, latest_date, latitude, longitude, elevation,
            best_fit_ref_well, best_rmse
        """
        _deprecated("obs_wells_summary()", "model.wells.obs().to_dataframe()")
        return self.wells.obs().to_dataframe()

    def ref_wells_summary(self) -> pd.DataFrame:
        """
        Get a summary DataFrame of reference wells in the model.

        .. deprecated:: 1.1.0
            Use ``model.wells.ref().to_dataframe()`` instead.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: name, data_points, start_date, end_date,
            mean_level, latest_value, latest_date, latitude, longitude, elevation,
            num_fits, avg_rmse
        """
        _deprecated("ref_wells_summary()", "model.wells.ref().to_dataframe()")
        return self.wells.ref().to_dataframe()

    def wells_summary(self) -> pd.DataFrame:
        """
        Get a combined summary DataFrame of all wells in the model.

        .. deprecated:: 1.1.0
            Use ``model.wells.to_dataframe()`` instead, which keeps the wells in
            model order.

        Returns
        -------
        pd.DataFrame
            Combined DataFrame with the observation wells first, then the
            reference wells, including all columns from both summary types
        """
        _deprecated("wells_summary()", "model.wells.to_dataframe()")
        obs_df = self.wells.obs().to_dataframe()
        ref_df = self.wells.ref().to_dataframe()
        if obs_df.empty:
            return ref_df
        if ref_df.empty:
            return obs_df
        return pd.concat([obs_df, ref_df], ignore_index=True)

    def fits_summary(self) -> pd.DataFrame:
        """
        Get a summary DataFrame of all fit results in the model.

        .. deprecated:: 1.1.0
            Use ``model.fits.to_dataframe()`` instead.

        Returns
        -------
        pd.DataFrame
            DataFrame with common columns (ref_well_name, obs_well_name, method,
            rmse, etc.) and method-specific columns with appropriate prefixes
            (e.g., linreg_slope, linreg_intercept)
        """
        _deprecated("fits_summary()", "model.fits.to_dataframe()")
        return self.fits.to_dataframe()

    def add_well(self, well: Well | list[Well]):
        """
        Add a well or a list of wells to the model.

        Parameters
        ----------
        well : Well or list of Wells
            The well or list of wells to add to the model.

        Returns
        -------
        None
            This method modifies the model in place.
        """
        if isinstance(well, list):
            for w in well:
                self._add_well(w)
            logger.debug(f"Added {len(well)} wells to model '{self.name}'.")
        else:
            self._add_well(well)

    def _add_well(self, well):
        """
        The internal method to add a well to the model.

        Parameters
        ----------
        well : Well
            The well to add to the model.

        Raises
        ------
        TypeError
            If the well is not an instance of WellBase.
        ValueError
            If the well is already in the model.

        Returns
        -------
        None
            This method modifies the model in place.
        """

        # Check if the well is an instance of Well
        if not isinstance(well, Well):
            logger.error("Only Well instances can be added to the model.")
            raise TypeError("Only Well instances can be added to the model.")

        # Check if the well is already in the model
        if well in self.wells:
            logger.error(f"Well '{well.name}' is already in the model.")
            raise ValueError(f"Well '{well.name}' is already in the model.")

        # Check if the well name already exists in the model
        if well.name in self.wells:
            logger.error(f"Well name '{well.name}' already exists in the model.")
            raise ValueError(f"Well name '{well.name}' already exists in the model.")

        # Add the well to the model
        self.wells._add(well)
        well.model.append(self)
        logger.debug(f"Well '{well.name}' added to model '{self.name}'.")

    def delete_well(self, well: Well | list[Well]):
        """
        Delete a well or a list of wells from the model.

        .. deprecated:: 1.1.0
            Use ``model.wells.remove(...)`` instead.

        The fits that use the deleted wells are deleted too.

        Parameters
        ----------
        well : Well or list of Wells
            The well or list of wells to delete from the model.

        Returns
        -------
        None
            This method modifies the model in place.
        """
        _deprecated("delete_well()", "model.wells.remove(...)")
        wells = well if isinstance(well, list) else [well]
        for w in wells:
            if not isinstance(w, Well):
                logger.error("Only Well instances can be deleted from the model.")
                raise TypeError("Only Well instances can be deleted from the model.")
            if w not in self.wells:
                logger.error(f"Well '{w.name}' is not in the model.")
                raise ValueError(f"Well '{w.name}' is not in the model.")
        self.wells.remove(wells)

    def get_wells(self, names: list[str] | str) -> Well | list[Well]:
        """
        Get wells from the model by their names.

        .. deprecated:: 1.1.0
            Use ``model.wells["name"]`` or ``model.wells.filter(name=[...])``
            instead.

        Parameters
        ----------
        names : list of str or str
            The name or list of names of the wells to retrieve.

        Returns
        -------
        Well | list[Well]
            The well or list of wells with the specified names.

        Raises
        ------
        ValueError
            If any of the specified well names are not found in the model.
        """
        _deprecated("get_wells()", 'model.wells["name"]')
        if isinstance(names, str):
            names = [names]
        found_wells = [self._get_well(name) for name in names]
        return found_wells if len(found_wells) > 1 else found_wells[0]

    def _get_well(self, name: str) -> Well:
        """
        Get a well by name, raising ValueError if it is not in the model.

        Parameters
        ----------
        name : str
            The name of the well.

        Returns
        -------
        Well
            The well with the given name.

        Raises
        ------
        ValueError
            If no well with that name is in the model.
        """
        if name not in self.wells:
            logger.error(f"Well name '{name}' not found in the model.")
            raise ValueError(f"Well name '{name}' not found in the model.")
        return self.wells[name]

    # ============================== Fit methods ==============================

    def _display_result(self, result: FitResultData):
        """
        Display fit result appropriately based on environment.

        In Jupyter notebooks, rely on the automatic HTML display of the return value.
        In other environments, print the text representation.
        """
        try:
            # Check if we're in a Jupyter environment
            from IPython.core.getipython import get_ipython

            ipython = get_ipython()
            if (
                ipython is not None
                and ipython.__class__.__name__ == "ZMQInteractiveShell"
            ):
                # We're in Jupyter - don't print, let the return value display
                logger.debug(result)  # Log at debug level instead of printing
                return
        except ImportError:
            # IPython not available, definitely not in Jupyter
            pass

        # Not in Jupyter or IPython not available, use regular print
        logger.info(result)

    def _resolve_wells(self, wells: Well | list[Well] | str | list[str]) -> list[Well]:
        """
        Resolve well names to Well objects.

        Parameters
        ----------
        wells : Well | list[Well] | str | list[str]
            Well objects, well names, or lists of either.

        Returns
        -------
        list[Well]
            A list of Well objects.
        """
        if isinstance(wells, str | Well):
            # Single item - convert to list for consistent handling
            wells = [wells]

        resolved_wells = []
        for well in wells:
            if isinstance(well, str):
                # Convert string name to Well object
                resolved_wells.append(self._get_well(well))
            elif isinstance(well, Well):
                resolved_wells.append(well)
            else:
                raise TypeError(
                    f"Unsupported well type: {type(well)}. Expected Well or str."
                )

        return resolved_wells

    # ======================== Load and Save Methods ========================

    def _to_dict(self):
        """
        Convert the model to a dictionary representation.

        Returns
        -------
        dict
            A dictionary representation of the model.
        """
        # Create a dictionary representation of the model
        model_dict = {
            "name": self.name,
            "wells": [well.name for well in self.wells],
            "gwrefpy_version": __version__,
        }

        # Create a dictionary representation of each well
        wells_dict = {}
        for well in self.wells:
            wells_dict[well.name] = well._to_dict()
        model_dict["wells_dict"] = wells_dict

        # Add fits if they exist
        if self.fits:
            model_dict["fits"] = [fit._to_dict() for fit in self.fits]

        return model_dict

    def _unpack_dict(self, data):
        """
        Unpack a dictionary representation of the model and set the model's attributes.

        Parameters
        ----------
        data : dict
            A dictionary representation of the model.

        Returns
        -------
        None
            This method adds wells, fits, and properties to the model.
        """
        self.name = data.get("name", self.name)

        # Check gwrefpy version compatibility
        gwrefpy_version = data.get("gwrefpy_version", None)
        if gwrefpy_version is not None and gwrefpy_version != __version__:
            logger.warning(
                f"The model was created with gwrefpy version {gwrefpy_version}, "
                f"but the current version is {__version__}. "
                "There may be compatibility issues."
            )

        # Unpack wells
        wells_dict = data.get("wells_dict", {})
        for w in wells_dict.items():
            well_obj = w[1]
            well = Well(name=well_obj["name"], is_reference=well_obj["is_reference"])
            well._unpack_dict(well_obj)
            self.add_well(well)

        # Unpack fits
        fits_list = data.get("fits", [])
        for fit_data in fits_list:
            pred_const = fit_data.get("pred_const", None)
            if isinstance(pred_const, list):
                pred_const_index = fit_data.get("pred_const_index")
                if pred_const_index is not None:
                    pred_const_index = [
                        float_to_datetime(timestamp) for timestamp in pred_const_index
                    ]
                else:
                    ref_index = self.wells[fit_data["ref_well"]].timeseries.index
                    pred_const_index = ref_index[: len(pred_const)]
                pred_const = pd.Series(pred_const, index=pred_const_index)
            fit = FitResultData(
                ref_well=self.wells[fit_data["ref_well"]],
                obs_well=self.wells[fit_data["obs_well"]],
                rmse=fit_data.get("rmse", None),
                n=fit_data.get("n", None),
                fit_method=_unpack_dict_fit_method(fit_data),
                t_a=fit_data.get("t_a", None),
                stderr=fit_data.get("stderr", None),
                pred_const=pred_const,
                prediction_leverage_matrix=fit_data.get(
                    "prediction_leverage_matrix", None
                ),
                p=fit_data.get("p", None),
                offset=fit_data.get("offset", None),
                aggregation=fit_data.get("aggregation", "mean"),
                te_method=fit_data.get("te_method", "anchor"),
                tmin=float_to_datetime(fit_data.get("tmin", None)),
                tmax=float_to_datetime(fit_data.get("tmax", None)),
                shift=fit_data.get("shift", None),
                name=fit_data.get("name", None),
            )
            self.fits._add_renaming(fit)

    def save_project(self, filename=None, overwrite=False):
        """
        Save the model to a file.

        Parameters
        ----------
        filename : str or None, optional
            The name of the file where the model will be saved. The path can be
            included.
        overwrite : bool, optional
            Whether to overwrite the file if it already exists (default is False).

        Returns
        -------
        None
            This method saves the model to a file.
        """

        # Convert the model to a dictionary
        model_dict = self._to_dict()

        # Set default filename if not provided
        if filename is None:
            filename = f"{self.name}.gwref"

        # Save the model dictionary to a file
        saved = save(filename, model_dict, overwrite=overwrite)
        if saved:
            logger.info(f"Model '{self.name}' saved to '{filename}'.")

    def open_project(self, filepath):
        """
        Load the model from a file.

        Parameters
        ----------
        filepath : str
            The path to the file from which the model will be loaded.

        Returns
        -------
        None
            This method loads the model from a file.
        """
        # Placeholder for load logic
        data = load(filepath)
        self._unpack_dict(data)
        logger.info(f"Model '{self.name}' loaded from '{filepath}'.")
