Plots
=====

Every fit and every well has a ``plots`` accessor, and ``plot()`` on the object
itself is a shortcut for ``plots.plot()``. Plots that combine several fits or
wells live in the ``gwrefpy.plots`` module.

.. code-block:: python

    import gwrefpy as gr

    fit.plot()                                         # obs, fitted ref, prediction interval
    fit.plot(outliers=True, calibration_period=True)
    fit.plot(tmin="2000", tmax="2002")                 # time slicing
    fit.plots.diagnostics()                            # cross-plot with regression and stats
    fit.plots.diagnostics(stats=False)

    well.plot(tmin="2000", tmax="2002")                # the well's own time series

    gr.plots.compare([fit1, fit2])                     # cross-plots overlaid
    gr.plots.compare(model.fits)
    gr.plots.compare([ref1, obs2])                     # time series overlaid
    gr.plots.compare(model.wells.ref())

Every plot takes ``ax=`` to draw on an existing Axes (otherwise it makes a new
figure of size ``figsize``) and returns the Axes. Titles, labels and saving are
done on the returned Axes, e.g. ``ax.set_title(...)`` or
``ax.figure.savefig(...)``.

In cross-plots each fit has its own colour and marker. Points inside the fit's
calibration period are filled and points outside it are hollow. ``tmin`` and
``tmax`` select which points are shown, and the regression curve spans them.

.. note::

    ``model.plot_wells()``, ``model.plot_fits()``, ``model.plot_fitmethod()``,
    ``model.show_plots()`` and the old ``fit.plot()`` arguments (``title``,
    ``mark_outliers``, ``plot_style`` and so on) are deprecated and will be
    removed in 2.0.


.. autofunction:: src.gwrefpy.plots.compare

.. autoclass:: src.gwrefpy.plots.FitPlots
   :members:

.. autoclass:: src.gwrefpy.plots.WellPlots
   :members:
