import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

from gwrefpy import Chebyshev, Model, NPoly, Well


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture()
def model() -> Model:
    """A model with one obs well and two ref wells, a year of daily data."""
    dates = pd.date_range("2020-01-01", periods=366, freq="D")
    rng = np.random.default_rng(0)
    signal = np.sin(np.linspace(0, 4 * np.pi, 366))
    spike = np.where(dates == "2020-11-15", 3.0, 0.0)  # one outlier, after calibration
    model = Model("plots")
    model.add_well(
        [
            Well(
                "obs",
                is_reference=False,
                timeseries=pd.Series(
                    2 * signal + 10 + spike + rng.normal(0, 0.1, 366), index=dates
                ),
            ),
            Well(
                "ref1",
                is_reference=True,
                timeseries=pd.Series(signal + rng.normal(0, 0.05, 366), index=dates),
            ),
            Well(
                "ref2",
                is_reference=True,
                timeseries=pd.Series(signal + 5 + rng.normal(0, 0.2, 366), index=dates),
            ),
        ]
    )
    return model


@pytest.fixture()
def fit(model):
    """A linear fit calibrated on March to September 2020."""
    return model.fit(
        "obs", "ref1", offset="0D", tmin="2020-03-01", tmax="2020-09-30", report=False
    )


def legend_texts(ax: Axes) -> list[str]:
    legend = ax.get_legend()
    return [t.get_text() for t in legend.get_texts()] if legend else []


def test_fit_plot_draws_obs_fitted_ref_and_prediction_interval(fit):
    ax = fit.plots.plot()

    assert isinstance(ax, Axes)
    assert legend_texts(ax) == ["obs", "ref1 (fitted)", "95% prediction interval"]


def line_dates(ax: Axes) -> pd.DatetimeIndex:
    """All x values of the lines on a time axis, as dates."""
    import matplotlib.dates as mdates

    values = np.concatenate([mdates.date2num(line.get_xdata()) for line in ax.lines])
    return pd.DatetimeIndex(mdates.num2date(values)).tz_localize(None)


def test_fit_plot_tmin_tmax_slices_the_drawn_data(fit):
    ax = fit.plots.plot(tmin="2020-05-01", tmax="2020-06-30")

    dates = line_dates(ax)
    assert dates.min() == pd.Timestamp("2020-05-01")
    assert dates.max() == pd.Timestamp("2020-06-30")


def test_fit_plot_marks_outliers(fit):
    ax = fit.plots.plot(outliers=True)

    assert "outliers" in legend_texts(ax)
    (outlier_points,) = [c for c in ax.collections if c.get_label() == "outliers"]
    import matplotlib.dates as mdates

    dates = mdates.num2date(outlier_points.get_offsets()[:, 0])
    assert pd.Timestamp("2020-11-15", tz="UTC") in dates


def test_fit_plot_shades_calibration_period(fit):
    import matplotlib.dates as mdates

    ax = fit.plots.plot(calibration_period=True)

    assert "calibration period" in legend_texts(ax)
    (span,) = [p for p in ax.patches if p.get_label() == "calibration period"]
    x0, x1 = span.get_x(), span.get_x() + span.get_width()
    assert mdates.num2date(x0).date() == pd.Timestamp("2020-03-01").date()
    assert mdates.num2date(x1).date() == pd.Timestamp("2020-09-30").date()


def test_fit_plot_draws_raw_ref_series(fit):
    ax = fit.plots.plot(ref=True)

    assert "ref1" in legend_texts(ax)


def test_fit_plot_shortcut_is_the_new_plot_without_warnings(fit):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ax = fit.plot(outliers=True, tmin="2020-02-01")

    assert isinstance(ax, Axes)
    assert legend_texts(ax) == [
        "obs",
        "ref1 (fitted)",
        "95% prediction interval",
        "outliers",
    ]


@pytest.mark.parametrize(
    "args, kwargs",
    [
        ((), {"mark_outliers": False}),
        ((), {"plot_style": "fancy", "tmin": "2020-02-01"}),
        (("My title",), {}),
        ((), {"dpi": 80}),  # old fit.plot() passed extra kwargs to matplotlib
    ],
)
def test_fit_plot_with_old_arguments_warns_and_returns_fig_and_ax(fit, args, kwargs):
    from matplotlib.figure import Figure

    with pytest.deprecated_call():
        fig, ax = fit.plot(*args, **kwargs)

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_fit_plot_rejects_mixing_old_and_new_arguments(fit):
    with pytest.raises(TypeError, match="mix"):
        fit.plot(mark_outliers=True, outliers=True)


def test_fit_plot_with_a_misspelled_argument_still_fails(fit):
    # Unknown arguments go to the old API, which hands them to matplotlib.
    with pytest.raises((TypeError, AttributeError)), pytest.deprecated_call():
        fit.plot(outlier=True)


def test_diagnostics_legend_has_fit_stats(fit):
    ax = fit.plots.diagnostics()

    assert isinstance(ax, Axes)
    entry, other = legend_texts(ax)
    assert entry == (
        f"obs~ref1:Linear\nLinear, n=214, RMSE={fit.rmse:.3f}, R²={fit.rvalue**2:.3f}"
    )
    assert other == "outside calibration period"
    assert ax.get_xlabel() == "ref1"
    assert ax.get_ylabel() == "obs"


def test_diagnostics_without_stats_shows_only_fit_name(fit):
    ax = fit.plots.diagnostics(stats=False)

    assert legend_texts(ax) == ["obs~ref1:Linear", "outside calibration period"]


def scatters(ax: Axes):
    """The point sets that hold data, split into (filled, hollow)."""
    filled, hollow = [], []
    for c in ax.collections:
        if len(c.get_offsets()) == 0:
            continue
        is_hollow = len(c.get_facecolors()) == 0 or c.get_facecolors()[0][3] == 0
        (hollow if is_hollow else filled).append(c)
    return filled, hollow


def marker_of(collection) -> str:
    from matplotlib.markers import MarkerStyle

    path = collection.get_paths()[0]
    for m in "osD^vPX":
        style = MarkerStyle(m)
        ref = style.get_path().transformed(style.get_transform())
        if path.vertices.shape == ref.vertices.shape and np.allclose(
            path.vertices, ref.vertices
        ):
            return m
    raise AssertionError("unknown marker")


def test_diagnostics_calibration_points_filled_other_points_hollow(fit):
    from matplotlib.colors import to_rgba

    ax = fit.plots.diagnostics()

    (filled,), (hollow,) = scatters(ax)
    assert len(filled.get_offsets()) == 214  # March to September 2020
    assert len(hollow.get_offsets()) == 366 - 214
    assert tuple(filled.get_facecolors()[0]) == to_rgba("C0")
    assert tuple(hollow.get_edgecolors()[0]) == to_rgba("C0")
    assert marker_of(filled) == marker_of(hollow) == "o"


def test_diagnostics_tmin_tmax_selects_points_and_curve_spans_them(fit):
    ax = fit.plots.diagnostics(tmin="2020-02-01", tmax="2020-03-31")

    (filled,), (hollow,) = scatters(ax)
    assert len(filled.get_offsets()) == 31  # March
    assert len(hollow.get_offsets()) == 29  # February 2020
    shown_x = np.concatenate([filled.get_offsets()[:, 0], hollow.get_offsets()[:, 0]])
    curve = ax.lines[0]
    assert curve.get_xdata().min() == pytest.approx(shown_x.min())
    assert curve.get_xdata().max() == pytest.approx(shown_x.max())


def test_compare_fits_gives_each_fit_its_own_colour_and_marker(model, fit):
    from matplotlib.colors import to_rgba

    import gwrefpy as gr

    fit2 = model.fit("obs", "ref2", offset="0D", report=False)
    ax = gr.plots.compare([fit, fit2], stats=False)

    filled, _ = scatters(ax)
    assert [marker_of(c) for c in filled] == ["o", "s"]
    assert [tuple(c.get_facecolors()[0]) for c in filled] == [
        to_rgba("C0"),
        to_rgba("C1"),
    ]
    assert legend_texts(ax) == [
        "obs~ref1:Linear",
        "obs~ref2:Linear",
        "outside calibration period",
    ]
    assert ax.get_xlabel() == "reference well"
    assert ax.get_ylabel() == "obs"


def test_compare_accepts_a_fit_collection(model, fit):
    import gwrefpy as gr

    model.fit("obs", "ref2", offset="0D", report=False)
    ax = gr.plots.compare(model.fits, stats=False)

    assert legend_texts(ax)[:2] == ["obs~ref1:Linear", "obs~ref2:Linear"]


def test_compare_rejects_a_mix_of_fits_and_wells(model, fit):
    import gwrefpy as gr

    with pytest.raises(TypeError, match="mix"):
        gr.plots.compare([fit, model.wells["obs"]])


def test_compare_rejects_an_empty_list():
    import gwrefpy as gr

    with pytest.raises(ValueError, match="at least one"):
        gr.plots.compare([])


def test_gwrefpy_exposes_the_plots_module():
    import subprocess
    import sys

    code = "import gwrefpy as gr; print(gr.plots.compare.__name__)"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "compare"


def test_well_plot_draws_its_own_series_sliced(model):
    well = model.wells["obs"]

    ax = well.plot(tmin="2020-05-01", tmax="2020-06-30")

    assert isinstance(ax, Axes)
    assert legend_texts(ax) == ["obs"]
    dates = line_dates(ax)
    assert (dates.min(), dates.max()) == (
        pd.Timestamp("2020-05-01"),
        pd.Timestamp("2020-06-30"),
    )


def test_well_plots_plot_draws_on_given_axes(model):
    _, given = plt.subplots()

    ax = model.wells["ref1"].plots.plot(ax=given)

    assert ax is given
    assert len(ax.lines) == 1


def test_compare_wells_overlays_their_series(model):
    import gwrefpy as gr

    ax = gr.plots.compare(
        [model.wells["ref1"], model.wells["obs"]], tmin="2020-05-01", tmax="2020-06-30"
    )

    assert legend_texts(ax) == ["ref1", "obs"]
    assert len(ax.lines) == 2
    assert line_dates(ax).min() == pd.Timestamp("2020-05-01")


def test_compare_accepts_a_well_collection(model):
    import gwrefpy as gr

    ax = gr.plots.compare(model.wells.ref())

    assert legend_texts(ax) == ["ref1", "ref2"]


@pytest.mark.parametrize(
    "call",
    [
        lambda m, f: m.plot_wells(),
        lambda m, f: m.plot_fits(f),
        lambda m, f: m.plot_fits(),
        lambda m, f: m.plot_fitmethod(f),
    ],
    ids=["plot_wells", "plot_fits", "plot_fits_all", "plot_fitmethod"],
)
def test_model_plot_methods_are_deprecated_but_still_work(model, fit, call):
    from matplotlib.figure import Figure

    with pytest.deprecated_call(match="removed in 2.0"):
        fig, ax = call(model, fit)

    assert isinstance(fig, Figure)
    assert isinstance(ax, Axes)


def test_model_show_plots_is_deprecated(model, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)

    with pytest.deprecated_call(match="removed in 2.0"):
        model.show_plots()


@pytest.mark.parametrize(
    "fit_kwargs",
    [
        {"method": NPoly(degree=3)},
        {"method": Chebyshev(degree=3)},
        {"te_method": "consecutive"},
        {"shift": "2D"},
    ],
    ids=["npoly", "chebyshev", "consecutive", "shift"],
)
def test_diagnostics_calibration_points_are_the_ones_the_fit_used(model, fit_kwargs):
    fit = model.fit(
        "obs",
        "ref1",
        offset="0D",
        tmin="2020-03-01",
        tmax="2020-09-30",
        report=False,
        **fit_kwargs,
    )

    ax = fit.plots.diagnostics()
    fit.plot(outliers=True, calibration_period=True)

    (filled,), _ = scatters(ax)
    assert len(filled.get_offsets()) == fit.n


def test_diagnostics_consecutive_fit_without_calibration_bounds(model):
    # Such fits store group numbers as tmin/tmax (a fitbase quirk); every pair
    # was used, so every point counts as calibration data.
    fit = model.fit("obs", "ref1", offset="0D", te_method="consecutive", report=False)

    ax = fit.plots.diagnostics()

    filled, hollow = scatters(ax)
    assert len(filled[0].get_offsets()) == fit.n
    assert hollow == []
