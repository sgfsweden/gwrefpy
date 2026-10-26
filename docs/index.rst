gwrefpy
=======

A Python package for detection of deviations in groundwater time series.

.. grid::

    .. grid-item-card:: User Guide
        :link: user_guide/index
        :link-type: doc

        User guide on installation and the basic concepts.

    .. grid-item-card:: Tutorials
        :link: tutorials/index
        :link-type: doc

        Some tutorials on how to use the package.

    .. grid-item-card:: API Reference
        :link: apidocs
        :link-type: doc

        Application programming interface (API) reference.

.. grid::

    .. grid-item-card:: Exercises
        :link: tutorials/exercises/index
        :link-type: doc

        A collection of ``gwrefpy`` exercises.

    .. grid-item-card:: Key Concepts
        :link: about/keyconcepts
        :link-type: doc

        Key concepts behind ``gwrefpy``.

    .. grid-item-card:: GitHub Repository
        :link: https://github.com/andersretznerSGU/gwrefpy

        The source code for ``gwrefpy`` is hosted on GitHub.

Features
--------

- Programmatically fit observation wells to reference wells
- Visualize fits and deviations
- Save your work, share and pick up later with a custom ``.gwref`` file format
- Work with live data and update your analysis as new data comes in


Quick Example
-------------

.. tab-set::

    .. tab-item:: Python

        In this example an observation well and reference well are fitted and plotted.

        .. code-block:: python

            # Import the packages
            import gwrefpy as gr
            import pandas as pd

            # Load timeseries data from CSV files
            obs_data = pd.read_csv("obs.csv", index_col="date", parse_dates=["date"]).squeeze()
            ref_data = pd.read_csv("ref.csv", index_col="date", parse_dates=["date"]).squeeze()

            # Create Well objects and add timeseries data
            obs = gr.Well(name="12GIPGW", is_reference=False)
            obs.add_timeseries(obs_data)

            ref = gr.Well(name="45LOGW", is_reference=True)
            ref.add_timeseries(ref_data)

            # Create a Model object, add wells, and fit the model
            model = gr.Model(name="Small Example")
            model.add_well([obs, ref])

            fit = model.fit(obs_well=obs, ref_well=ref, offset="0D", tmin="2020-01-01", tmax="2020-03-23")

            # Plot the results
            fit.plot(outliers=True, calibration_period=True)

    .. tab-item:: Result

        .. figure:: _static/figures/quick_example_plot.png
            :alt: Quick example plot
            :align: center


.. toctree::
    :maxdepth: 1
    :titlesonly:
    :hidden:
    :caption: Contents:

    user_guide/index
    tutorials/index
    apidocs
    about/index
    index
