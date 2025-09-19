import logging

import pandas as pd

from .constants import DEFAULT_PLOT_ATTRIBUTES
from .utils.conversions import datetime_to_float, float_to_datetime

logger = logging.getLogger(__name__)


class Well:
    """
    Base class for a well in a groundwater model.

    Parameters
    ----------
    name : str
        The name of the well.
    is_reference : bool
        Whether the well is a reference well.
    timeseries : pd.Series, optional
        A pandas Series containing the time series data for the well. The index
        should be a pandas DatetimeIndex and the values should be floats.
        Default is None.
    """

    def __init__(
        self,
        name,
        is_reference,
        timeseries: pd.Series | None = None,
    ):
        """
        Initialize a WellBase object.
        """

        # Initialize attributes
        self._name = ""
        self.name = name  # This will call the setter
        self.is_reference = is_reference
        self.model = []

        # Time series data
        self.timeseries = None
        if timeseries is not None:
            self.add_timeseries(timeseries)

        # Plotting attributes
        self.color = DEFAULT_PLOT_ATTRIBUTES["color"]
        self.alpha = DEFAULT_PLOT_ATTRIBUTES["alpha"]
        self.linestyle = DEFAULT_PLOT_ATTRIBUTES["linestyle"]
        self.linewidth = DEFAULT_PLOT_ATTRIBUTES["linewidth"]
        self.marker = DEFAULT_PLOT_ATTRIBUTES["marker"]
        self.markersize = DEFAULT_PLOT_ATTRIBUTES["markersize"]
        self.marker_visible = DEFAULT_PLOT_ATTRIBUTES["marker_visible"]

        # Geographic attributes
        self.latitude = None
        self.longitude = None
        self.elevation = None

    @property
    def name(self) -> str:
        """The name of the well."""
        return self._name

    @name.setter
    def name(self, value):
        """Set the name of the well."""
        if not value:
            logger.error("Name cannot be an empty string.")
            raise ValueError("Name cannot be an empty string.")
        if not isinstance(value, str):
            logger.error("Name must be a string.")
            raise TypeError("Name must be a string.")
        self._name = value

    def set_kwargs(self, **kwargs):
        """
        Set attributes of the WellBase object using keyword arguments.

        Parameters
        ----------
        **kwargs : dict
            The attributes to set. The keys should be the names of the attributes and
            the values should be the new values.

        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"{self.__class__.__name__} has no attribute '{key}'")
                raise AttributeError(
                    f"{self.__class__.__name__} has no attribute '{key}'"
                )

    def __repr__(self):
        return f"Well(name={self.name})"

    def __str__(self):
        return f"Well: {self.name}"

    def add_timeseries(self, timeseries: pd.Series):
        """
        Add a timeseries to the well. This will be validated by `_validate_timeseries`.

        Parameters
        ----------
        timeseries : pd.Series
            A pandas Series containing the time series data to add.

        """
        self._validate_timeseries(timeseries)
        if self.timeseries is not None:
            logger.error(
                f"Well {self.name} already has a timeseries. Use "
                f"`append_timeseries` to add more data or overwrite it using "
                f"`replace_timeseries`."
            )
            raise ValueError(
                f"Well {self.name} already has a timeseries. Use "
                f"`append_timeseries` to add more data or overwrite it using "
                f"`replace_timeseries`."
            )
        self.timeseries = timeseries
        self.timeseries.name = self.name
        logger.debug(f"Added timeseries to well {self.name}")

    def append_timeseries(self, timeseries: pd.Series, remove_duplicates: bool = False):
        """
        Append a timeseries to the existing timeseries of the well. This will be
        validated by `_validate_timeseries`.

        Parameters
        ----------
        timeseries : pd.Series
            A pandas Series containing the time series data to append.
        remove_duplicates : bool, optional
            Whether to remove duplicate timestamps after appending. Default is False
            which will raise an error if duplicates are found.

        """
        self._validate_timeseries(timeseries)
        if self.timeseries is not None:
            new_timeseries = pd.concat([self.timeseries, timeseries]).sort_index()
            n0 = len(self.timeseries)
            new_timeseries = new_timeseries[
                ~new_timeseries.index.duplicated(keep="last")
            ]
            n1 = len(new_timeseries)
            if n1 < n0 + len(timeseries):
                if not remove_duplicates:
                    logger.error(
                        f"Appending timeseries to well {self.name} resulted in "
                        f"duplicate timestamps. Set `remove_duplicates=True` to remove "
                        f"them."
                    )
                    raise ValueError(
                        f"Appending timeseries to well {self.name} resulted in "
                        f"duplicate timestamps. Set `remove_duplicates=True` to remove "
                        f"them."
                    )
                logger.warning(
                    f"Appending timeseries to well {self.name} resulted in "
                    f"{n0 + len(timeseries) - n1} duplicate timestamps being removed."
                )
            self.timeseries = new_timeseries
            self.timeseries.name = self.name
        else:
            self.timeseries = timeseries
            self.timeseries.name = self.name
        logger.debug(f"Appended timeseries to well {self.name}")

    def replace_timeseries(self, timeseries: pd.Series):
        """
        Replace the existing timeseries of the well with a new timeseries. This will be
        validated by `_validate_timeseries`.

        Parameters
        ----------
        timeseries : pd.Series
            A pandas Series containing the new time series data.

        """
        self._validate_timeseries(timeseries)
        self.timeseries = timeseries
        self.timeseries.name = self.name
        logger.debug(f"Replaced timeseries of well {self.name}")

    def _validate_timeseries(self, timeseries: pd.Series):
        """
        Validate the timeseries data and data types.

        Parameters
        ----------
        timeseries : pd.Series
            The timeseries data to validate.

        Raises
        ------
        TypeError
            If the data types are invalid.
        ValueError
            If the timeseries is not valid.
        """
        # Check basic structure
        if not isinstance(timeseries, pd.Series):
            logger.error("Timeseries must be a pandas Series.")
            raise TypeError("Timeseries must be a pandas Series.")
        if timeseries.empty:
            logger.error("Timeseries cannot be empty.")
            raise ValueError("Timeseries cannot be empty.")

        # Check index is DatetimeIndex with pandas.Timestamps
        if not isinstance(timeseries.index, pd.DatetimeIndex):
            logger.error(
                f"Timeseries index must be pandas.DatetimeIndex, got "
                f"{type(timeseries.index)}"
            )
            raise TypeError(
                f"Timeseries index must be pandas.DatetimeIndex, "
                f"got {type(timeseries.index)}"
            )

        # Check values are float dtype
        if not pd.api.types.is_float_dtype(timeseries.values):
            logger.error(
                f"Timeseries values must be float dtype, got {timeseries.values.dtype}"
            )
            raise TypeError(
                f"Timeseries values must be float dtype, got {timeseries.values.dtype}"
            )

    def _time_series_to_dict(self):
        # Convert DatetimeIndex to float timestamps
        float_index = self.timeseries.index.map(lambda dt: datetime_to_float(dt))
        return pd.Series(self.timeseries.values, index=float_index).to_dict()

    def _to_dict(self):
        """
        Convert the Well object to a dictionary representation.

        Returns
        -------
        dict
            A dictionary containing the well's attributes.
        """
        return {
            "name": self.name,
            "is_reference": self.is_reference,
            "timeseries": self._time_series_to_dict()
            if self.timeseries is not None
            else None,
            "color": self.color,
            "alpha": self.alpha,
            "linestyle": self.linestyle,
            "linewidth": self.linewidth,
            "marker": self.marker,
            "markersize": self.markersize,
            "marker_visible": self.marker_visible,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "elevation": self.elevation,
        }

    def _unpack_dict(self, data):
        """
        Unpack a dictionary representation to set the Well object's attributes.

        Parameters
        ----------
        data : dict
            A dictionary containing the well's attributes.
        """

        timeseries_dict = data.get("timeseries", None)
        if timeseries_dict is not None:
            # Convert float timestamps back to DatetimeIndex
            datetime_index = pd.Index(
                [float_to_datetime(float(ts)) for ts in timeseries_dict.keys()]
            )
            values = list(timeseries_dict.values())
            self.timeseries = pd.Series(values, index=datetime_index)

        self.color = data.get("color", self.color)
        self.alpha = data.get("alpha", self.alpha)
        self.linestyle = data.get("linestyle", self.linestyle)
        self.linewidth = data.get("linewidth", self.linewidth)
        self.marker = data.get("marker", self.marker)
        self.markersize = data.get("markersize", self.markersize)
        self.marker_visible = data.get("marker_visible", self.marker_visible)
        self.latitude = data.get("latitude", self.latitude)
        self.longitude = data.get("longitude", self.longitude)
        self.elevation = data.get("elevation", self.elevation)
