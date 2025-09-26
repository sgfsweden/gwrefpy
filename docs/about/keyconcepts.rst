Key concepts
============

This section describes some of the key concepts behind ``gwrefpy`` and the Akvifär reference method.

.. tip::
    Many of the concepts described here are illustrated in the `User Guide <../user_guide/index.html>`_ and `Tutorials <../tutorials/index.html>`_.

.. card:: Observation well

    Observation wells are typically wells that the user is interested in to see if they are influenced by anthropogenic activities.

.. card:: Reference well

    Reference wells are wells that are used to represent natural conditions. These wells should not be influenced by anthropogenic activities.

.. card:: Model

    A model typically represents a site or an area of interest and contains one or more observation wells and one or more reference wells.

.. card:: Time offset

   Time offsets are a concept to handle time series with different timestamps. If two time series share the same timestamps, they can be compared directly. If not, we need to introduce a time offset to align them. The offset represents an interval for which all data points will be considered as having the same timestamp.

.. card:: Fitting

   To analyse deviations in groundwater level timeseries, the `gwrefpy` methodology relies on observation wells and reference wells. To check for deviations in an observation well, we fit its data to a reference well using regression. If later data from the observation well does not follow the fitted regression, a deviation has occurred.

.. card:: Time series

    A time series is a sequence of data points, typically consisting of successive measurements made over a time interval. In `gwrefpy`, time series are used to represent groundwater level measurements from observation and reference wells. The index of a time series is typically a datetime index, representing the timestamps of the measurements.

.. card:: Logging

    Logging is used to keep track of the operations performed on a model. The logging controls what messages are shown to the user and what messages are saved to a log file. The logging level can be set to control the verbosity of the output.

.. card:: Prediction constant

    The prediction constant is a value that is added to the predicted values of an observation well to account for systematic differences between the observation well and the reference well. This constant is determined during the fitting process.