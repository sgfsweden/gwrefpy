"""Plotting for fits and wells.

Each fit and well has a ``plots`` accessor (``fit.plots.plot()``,
``fit.plots.diagnostics()``, ``well.plots.plot()``), and ``plot()`` on the object
itself is a shortcut for ``plots.plot()``. Plots that combine several fits or wells
live in this module, e.g. ``gwrefpy.plots.compare([fit1, fit2])``.

Every plot draws on ``ax`` when given (otherwise on a new figure of size
``figsize``) and returns the Axes.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from .methods.linregressfit import Linear
from .methods.timeseries import groupby_time_equivalents

if TYPE_CHECKING:
    from .fitresults import FitResultData
    from .well import Well

TimeLike = str | pd.Timestamp | None

# Marker per fit in cross-plots; matplotlib has no default marker cycle.
MARKERS = ("o", "s", "^", "D", "v", "P", "X")


def _get_ax(ax: Axes | None, figsize: tuple[float, float] | None) -> Axes:
    """Return ``ax``, or the Axes of a new figure when it is None."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def _obs_series(fit: FitResultData) -> pd.Series:
    """The obs series as the fit saw it, i.e. shifted when the fit used a shift."""
    if fit.shift is not None:
        return fit.obs_well.shift_timeseries(fit.shift)
    return fit.obs_well.timeseries


def _calibration_window(fit: FitResultData) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Start and end of the calibration period, defaulting to the obs series ends.

    Fits grouped with ``te_method="consecutive"`` and no calibration bounds store
    group numbers as ``tmin``/``tmax``; those count as no bound.
    """
    obs = _obs_series(fit)

    def as_time(value, default):
        if value is None or isinstance(value, int | np.integer):
            return default
        return pd.Timestamp(value)

    return as_time(fit.tmin, obs.index[0]), as_time(fit.tmax, obs.index[-1])


def _r_squared(fit: FitResultData) -> float:
    """R² of a fit. Linear stores the correlation coefficient, the others R²
    (as a one-element array)."""
    rvalue = float(np.asarray(fit.fit_method.rvalue).squeeze())
    return rvalue**2 if isinstance(fit.fit_method, Linear) else rvalue


def _fit_label(fit: FitResultData, stats: bool) -> str:
    """Legend text for a fit in a cross-plot."""
    if not stats:
        return fit.name
    from .fitbase import _method_label

    return (
        f"{fit.name}\n{_method_label(fit.fit_method)}, n={fit.n}, "
        f"RMSE={fit.rmse:.3f}, R²={_r_squared(fit):.3f}"
    )


def _matched_pairs(fit: FitResultData, obs: pd.Series, ref: pd.Series):
    """Time-matched (ref, obs) value pairs, grouped the way the fit grouped them."""
    if obs.empty or ref.empty:
        return np.array([]), np.array([])
    ref_te, obs_te, _ = groupby_time_equivalents(
        obs, ref, fit.offset, fit.aggregation, fit.te_method
    )
    return ref_te.to_numpy(dtype=float), obs_te.to_numpy(dtype=float)


def _cross_plot_pairs(fit: FitResultData, tmin: TimeLike, tmax: TimeLike):
    """Pairs inside and outside the calibration period, within ``[tmin, tmax]``.

    Each part of the record is grouped on its own, so the calibration pairs are
    exactly the ones the fit was built on.
    """
    obs = _obs_series(fit).loc[tmin:tmax]
    ref = fit.ref_well.timeseries.loc[tmin:tmax]
    start, end = _calibration_window(fit)

    def part(keep):
        return _matched_pairs(fit, obs[keep(obs.index)], ref[keep(ref.index)])

    calibration = part(lambda idx: (idx >= start) & (idx <= end))
    before = part(lambda idx: idx < start)
    after = part(lambda idx: idx > end)
    other = tuple(np.concatenate(arrays) for arrays in zip(before, after, strict=True))
    return calibration, other


def _draw_fit_cross_plots(
    fits: list[FitResultData],
    ax: Axes,
    tmin: TimeLike,
    tmax: TimeLike,
    stats: bool,
) -> None:
    """Draw each fit's points, regression curve and prediction interval on ``ax``."""
    any_other = False
    for i, fit in enumerate(fits):
        color, marker = f"C{i % 10}", MARKERS[i % len(MARKERS)]
        (cal_ref, cal_obs), (other_ref, other_obs) = _cross_plot_pairs(fit, tmin, tmax)

        ax.scatter(cal_ref, cal_obs, marker=marker, color=color, s=14)
        if other_ref.size:
            any_other = True
            ax.scatter(
                other_ref,
                other_obs,
                marker=marker,
                facecolors="none",
                edgecolors=color,
                s=14,
            )

        # The curve only spans the points on display, so slicing zooms the axes.
        visible = np.concatenate([cal_ref, other_ref])
        if visible.size:
            x = pd.Series(np.linspace(visible.min(), visible.max(), 200))
            y = fit.fit_method.fit_timeseries(x).to_numpy()
            width = np.asarray(fit.fit_method.get_prediction_constant(x))
            ax.plot(x, y, color=color)
            ax.plot(x, y - width, color=color, linestyle="--", linewidth=0.8)
            ax.plot(x, y + width, color=color, linestyle="--", linewidth=0.8)

        # Legend-only handle that shows both the fit's colour and its marker.
        ax.plot([], [], color=color, marker=marker, label=_fit_label(fit, stats))

    if any_other:
        ax.scatter(
            [],
            [],
            marker="o",
            facecolors="none",
            edgecolors="0.4",
            label="outside calibration period",
        )

    ref_names = {fit.ref_well.name for fit in fits}
    obs_names = {fit.obs_well.name for fit in fits}
    ax.set_xlabel(ref_names.pop() if len(ref_names) == 1 else "reference well")
    ax.set_ylabel(obs_names.pop() if len(obs_names) == 1 else "observation well")
    ax.legend()


def compare(
    items: Iterable,
    *,
    tmin: TimeLike = None,
    tmax: TimeLike = None,
    stats: bool = True,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
) -> Axes:
    """Compare several fits, or several wells, on one Axes.

    Fits are drawn as a cross-plot (ref levels on x, obs levels on y), each fit in
    its own colour and marker. Points inside the fit's calibration period are
    filled and points outside it are hollow. Each fit's regression curve and its
    prediction interval (dashed) span the points on display.

    Wells are drawn as overlaid time series.

    Parameters
    ----------
    items : list[FitResultData] | FitCollection | list[Well] | WellCollection
        The fits or the wells to compare. Fits and wells cannot be mixed.
    tmin, tmax : str | pd.Timestamp | None
        Only use data inside this time window. For fits this selects the points
        shown; the fits themselves are unchanged.
    stats : bool
        For fits, show method, n, RMSE and R² in the legend. Ignored for wells.
    ax : matplotlib.axes.Axes | None
        Axes to draw on. A new figure is created when None.
    figsize : tuple[float, float] | None
        Size of the new figure, used only when ``ax`` is None.

    Returns
    -------
    matplotlib.axes.Axes
    """
    from .fitresults import FitResultData
    from .well import Well

    items = list(items)
    if not items:
        raise ValueError("compare() needs at least one fit or well.")
    if all(isinstance(item, FitResultData) for item in items):
        ax = _get_ax(ax, figsize)
        _draw_fit_cross_plots(items, ax, tmin, tmax, stats)
        return ax
    if all(isinstance(item, Well) for item in items):
        ax = _get_ax(ax, figsize)
        for well in items:
            _draw_well(well, ax, tmin, tmax)
        ax.legend()
        return ax
    raise TypeError("compare() takes only fits or only wells, not a mix.")


def _draw_well(well: Well, ax: Axes, tmin: TimeLike, tmax: TimeLike) -> None:
    """Draw a well's time series on ``ax``, in the next colour of the cycle."""
    if well.timeseries is None:
        raise ValueError(f"Well {well.name} has no timeseries to plot.")
    ts = well.timeseries.loc[tmin:tmax]
    ax.plot(ts.index, ts.to_numpy(), label=well.name)


class FitPlots:
    """Plots for one fit, available as ``fit.plots``."""

    def __init__(self, fit: FitResultData):
        self._fit = fit

    def plot(
        self,
        *,
        tmin: TimeLike = None,
        tmax: TimeLike = None,
        outliers: bool = False,
        calibration_period: bool = False,
        ref: bool = False,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> Axes:
        """Plot the obs series, the fitted ref series and the prediction interval.

        Parameters
        ----------
        tmin, tmax : str | pd.Timestamp | None
            Only draw data inside this time window. The fit itself is unchanged.
        outliers : bool
            Mark the obs points that fall outside the prediction interval.
        calibration_period : bool
            Shade the period the fit was calibrated on.
        ref : bool
            Also draw the raw (untransformed) ref series.
        ax : matplotlib.axes.Axes | None
            Axes to draw on. A new figure is created when None.
        figsize : tuple[float, float] | None
            Size of the new figure, used only when ``ax`` is None.

        Returns
        -------
        matplotlib.axes.Axes
        """
        fit = self._fit
        ax = _get_ax(ax, figsize)

        obs = _obs_series(fit).loc[tmin:tmax]
        ax.plot(obs.index, obs.to_numpy(), color="C0", label=fit.obs_well.name)

        fitted = fit.fit_timeseries().loc[tmin:tmax]
        width = fit.get_prediction_constant()
        if isinstance(width, pd.Series):
            width = width.loc[fitted.index].to_numpy()
        y = fitted.to_numpy()
        ax.plot(fitted.index, y, color="C1", label=f"{fit.ref_well.name} (fitted)")
        ax.fill_between(
            fitted.index,
            y - width,
            y + width,
            color="C1",
            alpha=0.3,
            linewidth=0,
            label=f"{fit.p:.0%} prediction interval",
        )

        if ref:
            raw = fit.ref_well.timeseries.loc[tmin:tmax]
            ax.plot(raw.index, raw.to_numpy(), color="C2", label=fit.ref_well.name)

        if outliers:
            is_outlier = fit.fit_outliers().reindex(obs.index, fill_value=False)
            points = obs[is_outlier.astype(bool)]
            ax.scatter(
                points.index,
                points.to_numpy(),
                facecolors="none",
                edgecolors="C3",
                s=50,
                zorder=3,
                label="outliers",
            )

        if calibration_period:
            start, end = _calibration_window(fit)
            start = max(start, pd.Timestamp(tmin)) if tmin is not None else start
            end = min(end, pd.Timestamp(tmax)) if tmax is not None else end
            if start <= end:
                ax.axvspan(
                    start,
                    end,
                    color="0.5",
                    alpha=0.15,
                    zorder=0,
                    label="calibration period",
                )

        ax.legend()
        return ax

    def diagnostics(
        self,
        *,
        tmin: TimeLike = None,
        tmax: TimeLike = None,
        stats: bool = True,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> Axes:
        """Cross-plot of ref levels against obs levels, with the regression.

        Points inside the calibration period are filled and points outside it
        are hollow. The regression curve and its prediction interval (dashed)
        span the points on display. This is the same plot as
        ``gwrefpy.plots.compare([fit])``.

        Parameters
        ----------
        tmin, tmax : str | pd.Timestamp | None
            Only show points inside this time window. The fit is unchanged.
        stats : bool
            Show method, n, RMSE and R² in the legend.
        ax : matplotlib.axes.Axes | None
            Axes to draw on. A new figure is created when None.
        figsize : tuple[float, float] | None
            Size of the new figure, used only when ``ax`` is None.

        Returns
        -------
        matplotlib.axes.Axes
        """
        return compare(
            [self._fit], tmin=tmin, tmax=tmax, stats=stats, ax=ax, figsize=figsize
        )


class WellPlots:
    """Plots for one well, available as ``well.plots``."""

    def __init__(self, well: Well):
        self._well = well

    def plot(
        self,
        *,
        tmin: TimeLike = None,
        tmax: TimeLike = None,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> Axes:
        """Plot the well's time series.

        Parameters
        ----------
        tmin, tmax : str | pd.Timestamp | None
            Only draw data inside this time window.
        ax : matplotlib.axes.Axes | None
            Axes to draw on. A new figure is created when None.
        figsize : tuple[float, float] | None
            Size of the new figure, used only when ``ax`` is None.

        Returns
        -------
        matplotlib.axes.Axes
        """
        ax = _get_ax(ax, figsize)
        _draw_well(self._well, ax, tmin, tmax)
        ax.legend()
        return ax
