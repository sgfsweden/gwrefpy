import matplotlib.pyplot as plt
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from gwrefpy.plotter import Plotter
from gwrefpy.well import Well


def test_plot_wells_returns_figure_and_axes(timeseries):
    """Test that plot_wells returns matplotlib Figure and Axes objects."""
    plotter = Plotter()
    well = Well("Test Well", is_reference=True, timeseries=timeseries)

    fig, ax = plotter.plot_wells(well)

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)  # Clean up


def test_plot_wells_with_list_of_wells(timeseries):
    """Test that plot_wells works with a list of wells."""
    plotter = Plotter()
    well1 = Well("Well 1", is_reference=True, timeseries=timeseries)
    well2 = Well("Well 2", is_reference=False, timeseries=timeseries)
    wells = [well1, well2]

    fig, ax = plotter.plot_wells(wells)

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_plot_fits(strandangers_model) -> None:
    """Test that plot_fits works."""
    fit = strandangers_model.best_fit(strandangers_model.obs_wells[0], offset="3.5D")

    fig, ax = strandangers_model.plot_fits(fit)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_plot_wells_separate_figures(timeseries):
    """Test that plot_separately=True returns lists of figures and axes."""
    plotter = Plotter()
    well1 = Well("Well 1", is_reference=True, timeseries=timeseries)
    well2 = Well("Well 2", is_reference=False, timeseries=timeseries)
    wells = [well1, well2]

    figs, axs = plotter.plot_wells(wells, plot_separately=True)

    assert isinstance(figs, list)
    assert isinstance(axs, list)
    assert len(figs) == 2
    assert len(axs) == 2
    assert all(isinstance(fig, Figure) for fig in figs)
    assert all(isinstance(ax, Axes) for ax in axs)

    # Clean up
    for fig in figs:
        plt.close(fig)


def test_plot_wells_invalid_input():
    """Test that plot_wells raises TypeError for invalid input."""
    plotter = Plotter()

    with pytest.raises(TypeError, match="fits must be a Well instance"):
        plotter.plot_wells("invalid_input")  # type: ignore


def test_plot_wells_plot_style_validation(timeseries):
    """Test that invalid plot_style raises ValueError."""
    plotter = Plotter()
    well = Well("Test Well", is_reference=True, timeseries=timeseries)

    with pytest.raises(
        ValueError, match="plot_style must be 'fancy', 'scientific', or None"
    ):
        plotter.plot_wells(well, plot_style="invalid_style")


def test_plot_wells_color_style_validation(timeseries):
    """Test that invalid color_style raises ValueError."""
    plotter = Plotter()
    well = Well("Test Well", is_reference=True, timeseries=timeseries)

    with pytest.raises(
        ValueError, match="color_style must be 'color', 'monochrome', or None"
    ):
        plotter.plot_wells(well, color_style="invalid_color")


def test_plot_wells_scientific_style(timeseries):
    """Test that scientific plot style works."""
    plotter = Plotter()
    well = Well("Test Well", is_reference=True, timeseries=timeseries)

    fig, ax = plotter.plot_wells(well, plot_style="scientific")

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_plot_wells_monochrome_style(timeseries):
    """Test that monochrome color style works."""
    plotter = Plotter()
    well = Well("Test Well", is_reference=True, timeseries=timeseries)

    fig, ax = plotter.plot_wells(well, color_style="monochrome")

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_plot_wells_none_style(timeseries):
    """Test that plot_style=None uses matplotlib defaults."""
    plotter = Plotter()
    well = Well("Test Well", is_reference=True, timeseries=timeseries)

    fig, ax = plotter.plot_wells(well, plot_style=None)

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_plot_wells_none_color_style(timeseries):
    """Test that color_style=None uses matplotlib defaults."""
    plotter = Plotter()
    well = Well("Test Well", is_reference=True, timeseries=timeseries)

    fig, ax = plotter.plot_wells(well, color_style=None)

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_plot_wells_custom_labels(timeseries):
    """Test that custom labels are applied."""
    plotter = Plotter()
    well = Well("Test Well", is_reference=True, timeseries=timeseries)

    title = "Custom Title"
    xlabel = "Custom X Label"
    ylabel = "Custom Y Label"

    fig, ax = plotter.plot_wells(well, title=title, xlabel=xlabel, ylabel=ylabel)
    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)

    assert ax.get_title() == title
    assert ax.get_xlabel() == xlabel
    assert ax.get_ylabel() == ylabel
    plt.close(fig)


def test_plot_wells_custom_figsize(timeseries):
    """Test that custom figsize is applied."""
    plotter = Plotter()
    well = Well("Test Well", is_reference=True, timeseries=timeseries)

    figsize = (12, 8)
    fig, _ = plotter.plot_wells(well, figsize=figsize)
    assert isinstance(fig, Figure)

    assert fig.get_size_inches()[0] == figsize[0]
    assert fig.get_size_inches()[1] == figsize[1]
    plt.close(fig)


def test_plot_wells_with_existing_axes(timeseries):
    """Test that plot_wells works with an existing matplotlib Axes object."""
    plotter = Plotter()
    well = Well("Test Well", is_reference=True, timeseries=timeseries)

    # Create figure and axes manually
    fig, ax = plt.subplots()

    # Plot on the existing axes
    returned_fig, returned_ax = plotter.plot_wells(well, ax=ax)

    # Should return the same figure and axes
    assert returned_fig is fig
    assert returned_ax is ax
    assert isinstance(returned_fig, Figure)
    assert isinstance(returned_ax, Axes)

    plt.close(fig)


def test_plot_wells_ax_with_plot_separately_raises_error(timeseries):
    """Test that using ax parameter with plot_separately=True raises ValueError."""
    plotter = Plotter()
    well = Well("Test Well", is_reference=True, timeseries=timeseries)

    fig, ax = plt.subplots()

    with pytest.raises(
        ValueError, match="ax parameter cannot be used with plot_separately=True"
    ):
        plotter.plot_wells(well, ax=ax, plot_separately=True)

    plt.close(fig)


def test_fit_result_data_plot(timeseries):
    """Test that FitResultData.plot() returns Figure and Axes."""
    from gwrefpy.fitresults import FitResultData, LinRegResult

    ref_well = Well("ref", is_reference=True, timeseries=timeseries)
    obs_well = Well("obs", is_reference=False, timeseries=timeseries)

    fit_method = LinRegResult(
        slope=1.0, intercept=0.0, rvalue=0.99, pvalue=0.001, stderr=0.01
    )
    fit = FitResultData(
        obs_well=obs_well,
        ref_well=ref_well,
        rmse=0.1,
        n=10,
        fit_method=fit_method,
        t_a=2.0,
        stderr=0.01,
        pred_const=0.05,
        p=0.95,
        offset="1D",
        aggregation="mean",
        shift=None,
        tmin=timeseries.index[0],
        tmax=timeseries.index[-1],
    )

    fig, ax = fit.plot()

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_fit_result_data_plot_with_tmin_tmax(timeseries):
    """Test that FitResultData.plot() works with tmin and tmax arguments."""
    from gwrefpy.fitresults import FitResultData, LinRegResult

    ref_well = Well("ref", is_reference=True, timeseries=timeseries)
    obs_well = Well("obs", is_reference=False, timeseries=timeseries)

    fit_method = LinRegResult(
        slope=1.0, intercept=0.0, rvalue=0.99, pvalue=0.001, stderr=0.01
    )
    fit = FitResultData(
        obs_well=obs_well,
        ref_well=ref_well,
        rmse=0.1,
        n=10,
        fit_method=fit_method,
        t_a=2.0,
        stderr=0.01,
        pred_const=0.05,
        p=0.95,
        offset="1D",
        aggregation="mean",
        shift=None,
        tmin=timeseries.index[0],
        tmax=timeseries.index[-1],
    )

    # Test with string tmin/tmax
    fig, ax = fit.plot(tmin="2024-11-01", tmax="2024-12-31")

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_fit_result_data_plot_with_tmin_only(timeseries):
    """Test that FitResultData.plot() works with only tmin argument."""
    from gwrefpy.fitresults import FitResultData, LinRegResult

    ref_well = Well("ref", is_reference=True, timeseries=timeseries)
    obs_well = Well("obs", is_reference=False, timeseries=timeseries)

    fit_method = LinRegResult(
        slope=1.0, intercept=0.0, rvalue=0.99, pvalue=0.001, stderr=0.01
    )
    fit = FitResultData(
        obs_well=obs_well,
        ref_well=ref_well,
        rmse=0.1,
        n=10,
        fit_method=fit_method,
        t_a=2.0,
        stderr=0.01,
        pred_const=0.05,
        p=0.95,
        offset="1D",
        aggregation="mean",
        shift=None,
        tmin=timeseries.index[0],
        tmax=timeseries.index[-1],
    )

    # Test with only tmin
    fig, ax = fit.plot(tmin="2024-12-01")

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)


def test_fit_result_data_plot_with_tmax_only(timeseries):
    """Test that FitResultData.plot() works with only tmax argument."""
    from gwrefpy.fitresults import FitResultData, LinRegResult

    ref_well = Well("ref", is_reference=True, timeseries=timeseries)
    obs_well = Well("obs", is_reference=False, timeseries=timeseries)

    fit_method = LinRegResult(
        slope=1.0, intercept=0.0, rvalue=0.99, pvalue=0.001, stderr=0.01
    )
    fit = FitResultData(
        obs_well=obs_well,
        ref_well=ref_well,
        rmse=0.1,
        n=10,
        fit_method=fit_method,
        t_a=2.0,
        stderr=0.01,
        pred_const=0.05,
        p=0.95,
        offset="1D",
        aggregation="mean",
        shift=None,
        tmin=timeseries.index[0],
        tmax=timeseries.index[-1],
    )

    # Test with only tmax
    fig, ax = fit.plot(tmax="2025-01-01")

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)
    plt.close(fig)
