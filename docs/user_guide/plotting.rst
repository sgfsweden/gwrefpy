Plotting
========

The gwrefpy package has several plotting capabilities to visualize groundwater data. Below are some examples of how to use these plotting functions.

.. figure:: _static/figures/plot_example.png
   :alt: Example plot
   :align: center

   Example plot generated with gwrefpy.

Creating a plot 📊
-----------------------
In gwrefpy there are currently two main plotting functions: ``plot_wells()`` and ``plot_fits()``. The ``plot_wells()`` function is used to plot the timeseries data from well objects, while the ``plot_fits()`` function is used to plot the fitted models for the well data.



Setting plotting style
----------------------
You can set the individual plotting styles of each well object by assigning the plotting attributes to that object. The available attributes with their defaults are:

- ``color = None``
- ``alpha = 1.0``
- ``linestyle = None``
- ``linewidth = 1.0``
- ``marker = None``
- ``markersize = 6``
- ``marker_visible = False``

The attributes can be set when initializing the well object or by using the ``set_kwargs()`` method after the object has been created.

.. code-block:: python

    import gwrefpy as gr
    import matplotlib.pyplot as plt

    # Create the well object
    well1 = gr.Well(name='Well 1', is_reference=True, markersize=10)

    # Changing the plotting style after creation
    well1.set_kwargs(color='blue', linestyle='--')


Plotting kwargs
---------------
The plotting functions in gwrefpy accept additional keyword arguments (kwargs) that can be used to customize the appearance of the plots. These kwargs are passed directly to the underlying matplotlib functions. Currently kwargs are passed to the following functions: ``plt.subplot()`` and ``plt.savefig()``.

