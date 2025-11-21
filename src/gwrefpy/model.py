import logging

import pandas as pd

from . import __version__
from .fitbase import FitBase
from .fitresults import FitResultData, LinRegResult, _unpack_dict_fit_method
from .io.io import load, save
from .plotter import Plotter
from .utils.conversions import float_to_datetime
from .well import Well

logger = logging.getLogger(__name__)


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
        self.wells: list[Well] = []

        # Fit attributes
        self.fits: list[FitResultData] = []

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
        obs_count = len(self.obs_wells)
        ref_count = len(self.ref_wells)
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
        """List of observation wells in the model."""
        return [well for well in self.wells if not well.is_reference]

    def obs_wells_summary(self) -> pd.DataFrame:
        """
        Get a summary DataFrame of observation wells in the model.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: name, data_points, start_date, end_date,
            mean_level, latest_value, latest_date, latitude, longitude, elevation,
            best_fit_ref_well, best_rmse
        """
        obs_wells = self.obs_wells
        if not obs_wells:
            return pd.DataFrame()

        data = []
        for well in obs_wells:
            row = {
                "name": well.name,
                "well_type": "observation",
                "data_points": len(well.timeseries)
                if well.timeseries is not None
                else 0,
                "start_date": well.timeseries.index.min()
                if well.timeseries is not None and len(well.timeseries) > 0
                else None,
                "end_date": well.timeseries.index.max()
                if well.timeseries is not None and len(well.timeseries) > 0
                else None,
                "mean_level": well.timeseries.mean()
                if well.timeseries is not None and len(well.timeseries) > 0
                else None,
                "latest_value": well.timeseries.iloc[-1]
                if well.timeseries is not None and len(well.timeseries) > 0
                else None,
                "latest_date": well.timeseries.index[-1]
                if well.timeseries is not None and len(well.timeseries) > 0
                else None,
                "latitude": well.latitude,
                "longitude": well.longitude,
                "elevation": well.elevation,
            }

            # Find best fit for this observation well
            fits = self.get_fits(well)
            if fits:
                if isinstance(fits, list):
                    best_fit = min(fits, key=lambda x: x.rmse)
                else:
                    best_fit = fits
                row["best_fit_ref_well"] = best_fit.ref_well.name
                row["best_rmse"] = best_fit.rmse
            else:
                row["best_fit_ref_well"] = None
                row["best_rmse"] = None

            data.append(row)

        return pd.DataFrame(data)

    @property
    def ref_wells(self) -> list[Well]:
        """List of reference wells in the model."""
        return [well for well in self.wells if well.is_reference]

    def ref_wells_summary(self) -> pd.DataFrame:
        """
        Get a summary DataFrame of reference wells in the model.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: name, data_points, start_date, end_date,
            mean_level, latest_value, latest_date, latitude, longitude, elevation,
            num_fits, avg_rmse
        """
        ref_wells = self.ref_wells
        if not ref_wells:
            return pd.DataFrame()

        data = []
        for well in ref_wells:
            row = {
                "name": well.name,
                "well_type": "reference",
                "data_points": (
                    len(well.timeseries) if well.timeseries is not None else 0
                ),
                "start_date": (
                    well.timeseries.index.min()
                    if well.timeseries is not None and len(well.timeseries) > 0
                    else None
                ),
                "end_date": (
                    well.timeseries.index.max()
                    if well.timeseries is not None and len(well.timeseries) > 0
                    else None
                ),
                "mean_level": (
                    well.timeseries.mean()
                    if well.timeseries is not None and len(well.timeseries) > 0
                    else None
                ),
                "latest_value": (
                    well.timeseries.iloc[-1]
                    if well.timeseries is not None and len(well.timeseries) > 0
                    else None
                ),
                "latest_date": (
                    well.timeseries.index[-1]
                    if well.timeseries is not None and len(well.timeseries) > 0
                    else None
                ),
                "latitude": well.latitude,
                "longitude": well.longitude,
                "elevation": well.elevation,
            }

            # Find all fits using this reference well
            fits = self.get_fits(well)
            if fits:
                if isinstance(fits, list):
                    row["num_fits"] = len(fits)
                    row["avg_rmse"] = sum(fit.rmse for fit in fits) / len(fits)
                else:
                    row["num_fits"] = 1
                    row["avg_rmse"] = fits.rmse
            else:
                row["num_fits"] = 0
                row["avg_rmse"] = None

            data.append(row)

        return pd.DataFrame(data)

    def wells_summary(self) -> pd.DataFrame:
        """
        Get a combined summary DataFrame of all wells in the model.

        Returns
        -------
        pd.DataFrame
            Combined DataFrame with both observation and reference wells,
            including all columns from both summary types
        """
        obs_df = self.obs_wells_summary()
        ref_df = self.ref_wells_summary()

        if obs_df.empty and ref_df.empty:
            return pd.DataFrame()
        elif obs_df.empty:
            return ref_df
        elif ref_df.empty:
            return obs_df
        else:
            # Combine both DataFrames
            return pd.concat([obs_df, ref_df], ignore_index=True)

    def fits_summary(self) -> pd.DataFrame:
        """
        Get a summary DataFrame of all fit results in the model.

        Returns
        -------
        pd.DataFrame
            DataFrame with common columns (ref_well_name, obs_well_name, method,
            rmse, etc.) and method-specific columns with appropriate prefixes
            (e.g., linreg_slope, linreg_intercept)
        """
        if not self.fits:
            return pd.DataFrame()

        data = []
        for fit in self.fits:
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

    @property
    def well_names(self) -> list[str]:
        """List of all well names in the model."""
        return [well.name for well in self.wells]

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
        if well.name in self.well_names:
            logger.error(f"Well name '{well.name}' already exists in the model.")
            raise ValueError(f"Well name '{well.name}' already exists in the model.")

        # Add the well to the model
        self.wells.append(well)
        well.model.append(self)
        logger.debug(f"Well '{well.name}' added to model '{self.name}'.")

    def delete_well(self, well: Well | list[Well]):
        """
        Delete a well or a list of wells from the model.

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
                self._delete_well(w)
            logger.debug(f"Deleted {len(well)} wells from model '{self.name}'.")
        else:
            self._delete_well(well)

    def _delete_well(self, well):
        """
        The internal method to delete a well from the model.

        Parameters
        ----------
        well : Well
            The well to delete from the model.

        Raises
        ------
        TypeError
            If the well is not an instance of WellBase.
        ValueError
             If the well is not part of the model.

        Returns
        -------
        None
            This method modifies the model in place.
        """

        # Check if the well is an instance of Well
        if not isinstance(well, Well):
            logger.error("Only Well instances can be deleted from the model.")
            raise TypeError("Only Well instances can be deleted from the model.")

        # Check if the well is already in the model
        if well not in self.wells:
            logger.error(f"Well '{getattr(well, 'name', well)}' is not in the model.")
            raise ValueError(
                f"Well '{getattr(well, 'name', well)}' is not in the model."
            )

        self.wells.remove(well)
        if hasattr(well, "model") and self in well.model:
            well.model.remove(self)

        logger.debug(f"Well '{well.name}' deleted from model '{self.name}'.")

    def get_wells(self, names: list[str] | str) -> Well | list[Well]:
        """
        Get wells from the model by their names.

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
        if isinstance(names, str):
            names = [names]

        found_wells = []
        for name in names:
            if name in self.well_names:
                found_wells.append(self.wells[self.well_names.index(name)])
            else:
                logger.error(f"Well name '{name}' not found in the model.")
                raise ValueError(f"Well name '{name}' not found in the model.")

        return found_wells if len(found_wells) > 1 else found_wells[0]

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
        Resolve well names to Well objects using get_wells().

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
                resolved_well = self.get_wells(well)
                # get_wells returns single Well for single string input
                resolved_wells.append(resolved_well)
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
            fit = FitResultData(
                ref_well=self.wells[self.well_names.index(fit_data["ref_well"])],
                obs_well=self.wells[self.well_names.index(fit_data["obs_well"])],
                rmse=fit_data.get("rmse", None),
                n=fit_data.get("n", None),
                fit_method=_unpack_dict_fit_method(fit_data),
                t_a=fit_data.get("t_a", None),
                stderr=fit_data.get("stderr", None),
                pred_const=fit_data.get("pred_const", None),
                p=fit_data.get("p", None),
                offset=fit_data.get("offset", None),
                aggregation=fit_data.get("aggregation", "mean"),
                tmin=float_to_datetime(fit_data.get("tmin", None)),
                tmax=float_to_datetime(fit_data.get("tmax", None)),
                name=fit_data.get("name", None),
            )
            self.fits.append(fit)

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
