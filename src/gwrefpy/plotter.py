import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.dates import date2num, num2date
from matplotlib.figure import Figure, SubFigure

from .constants import (
    DEFAULT_COLORS,
    DEFAULT_LINE_STYLES,
    DEFAULT_MARKER_STYLES,
    DEFAULT_MONOCHROME_COLORS,
    afont,
    lfont,
    tfont,
    tifont,
)
from .fitresults import FitResultData, LinRegResult
from .well import Well

logger = logging.getLogger(__name__)


class Plotter:
    """
    Base class for plotting wells and fit results.
    """

    def __init__(self):
        self.wells = None
        self._cnt_colors = 0
        self._cnt_linestyles = 0
        self._cnt_markers = 0
        self._plot_style = None
        self._color_style = None
        self._offset_text = None
        self._xmin = None
        self._xmax = None
        self._ymin = None
        self._ymax = None

        self.fits = []

    def plot_wells(
        self,
        wells: Well | list[Well] | None = None,
        title: str = "Wells Plot",
        xlabel: str = "Time",
        ylabel: str = "Measurement",
        plot_style: str | None = None,
        color_style: str | None = None,
        save_path: str | None = None,
        num: int = 6,
        plot_separately: bool = False,
        ax: Axes | None = None,
        offset_text: dict[str, float] | None = None,
        **kwargs,
    ) -> tuple[Figure | SubFigure, Axes] | tuple[list[Figure], list[Axes]]:
        """
        This method plots the time series data for all fits in the model.

        Parameters
        ----------
        wells : Well | list[Well]
            A Well instance or a list of Well instances containing the wells to be
            plotted. If None, all wells will be plotted.
        title : str
            The title of the plot.
        xlabel : str
            The label for the x-axis.
        ylabel : str
            The label for the y-axis.
        plot_style : str | None
            The style of the plot. Options are "fancy", "scientific", or None.
            If None, uses matplotlib defaults without custom styling.
        color_style : str | None
            The color style of the plot. Options are "color", "monochrome", or None.
            If None, uses matplotlib defaults without custom colors.
        save_path : str | None
            If provided, the plot will be saved to this path. If the plot_separately
            parameter is True, the well name will be appended to the file name.
        num : int
            Number of ticks on the x-axis (default is 6).
        plot_separately : bool
            If True, each well will be plotted in a separate figure. Default is False.
        ax : matplotlib.axes.Axes | None
            Optional matplotlib Axes object to plot on. If provided, the plot will be
            drawn on this axes instead of creating a new figure. Not compatible with
            plot_separately=True.
        offset_text : dict[str, float]
            A dictionary containing well names as keys and vertical offset values as
            values. This is used to offset the text labels for specific wells to
            avoid overlap. Default is None, which means no offset.
        **kwargs : dict
            Additional keyword arguments for customization. See the documentation of
            Matplotlib's `plt.subplots` and `plt.savefig` for more details.
            Common kwargs include:

            - figsize (tuple): Size of the figure (width, height) in inches.
            - dpi (int): Dots per inch for the saved figure.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        ax : matplotlib.axes.Axes
            The axes object of the plot.
        """
        if wells is not None and not (
            isinstance(wells, Well)
            or (isinstance(wells, list) and all(isinstance(w, Well) for w in wells))
        ):
            logger.error(
                "fits must be a Well instance or a list of FitResultData instances"
            )
            raise TypeError(
                "fits must be a Well instance or a list of FitResultData instances"
            )

        # Validate ax parameter compatibility
        if ax is not None and plot_separately:
            logger.error("ax parameter cannot be used with plot_separately=True")
            raise ValueError("ax parameter cannot be used with plot_separately=True")

        if wells is None:
            wells = self.wells
        elif isinstance(wells, Well):
            wells = [wells]

        if not wells:
            logger.error("No wells available to plot.")
            raise ValueError("No wells available to plot.")

        # Validate and store the plot styles
        self._validate_plot_styles(plot_style, color_style, offset_text)

        # Get the figsize
        figsize = kwargs.pop("figsize", (10, 6))

        # Create the plot/s
        if plot_separately:
            figs, axs = [], []
            for w in wells:
                fig, ax = plt.subplots(figsize=figsize, **kwargs)
                self._set_plot_labels(ax, title, xlabel, ylabel)
                logger.debug(f"Plotting well: {w.name}")
                self._set_plot_attributes(w)
                self._plot_well(w, ax)
                self._plot_settings(ax, num)

                if save_path is not None:
                    path_parts = save_path.rsplit(".", 1)
                    if len(path_parts) == 2:
                        save_path_well = f"{path_parts[0]}_{w.name}.{path_parts[1]}"
                    else:
                        save_path_well = f"{save_path}_{w.name}.png"
                    plt.savefig(save_path_well, **kwargs)
                    logger.info(f"Plot saved to {save_path_well}")

                figs.append(fig)
                axs.append(ax)
            return figs, axs

        # Plot all wells in a single figure
        if ax is not None:
            # Use provided axes
            fig = ax.figure
            self._set_plot_labels(ax, title, xlabel, ylabel)
            for w in wells:
                logger.info(f"Plotting well: {w.name}")
                self._set_plot_attributes(w)
                self._plot_well(w, ax)
            self._plot_settings(ax, num)

            if save_path is not None:
                # Handle both Figure and SubFigure cases
                if isinstance(fig, Figure):
                    fig.savefig(save_path, **kwargs)
                else:
                    # For SubFigure, use the parent figure's savefig
                    plt.savefig(save_path, **kwargs)
                logger.info(f"Plot saved to {save_path}")
        else:
            # Create new figure and axes
            fig, ax = plt.subplots(figsize=figsize, **kwargs)
            self._set_plot_labels(ax, title, xlabel, ylabel)
            for w in wells:
                logger.info(f"Plotting well: {w.name}")
                self._set_plot_attributes(w)
                self._plot_well(w, ax)
            self._plot_settings(ax, num)

            if save_path is not None:
                plt.savefig(save_path, **kwargs)
                logger.info(f"Plot saved to {save_path}")

        assert ax is not None  # Type narrowing for type checker
        return fig, ax

    def plot_fits(
        self,
        fits: FitResultData | list[FitResultData] = None,
        title: str = "Well Data Plot",
        xlabel: str = "Time",
        ylabel: str = "Measurement",
        mark_outliers: bool = True,
        show_initiation_period: bool = False,
        plot_style: str | None = None,
        color_style: str | None = None,
        save_path: str | None = None,
        num: int = 6,
        plot_separately: bool = False,
        ax: Axes | None = None,
        offset_text: dict[str, float] | None = None,
        **kwargs,
    ) -> tuple[Figure | SubFigure, Axes] | tuple[list[Figure], list[Axes]]:
        """
        This method plots the time series data for all fits in the model.

        Parameters
        ----------
        fits : FitResultData | list[FitResultData]
            A FitResultData instance or a list of FitResultData instances
            containing the fit results to be plotted. If None, all fits will be plotted.
        title : str
            The title of the plot.
        xlabel : str
            The label for the x-axis.
        ylabel : str
            The label for the y-axis.
        mark_outliers : bool
            If True, outliers will be marked on the plot.
        show_initiation_period : bool
            If True, the initiation period will be shaded on the plot. Default is False.
        plot_style : str | None
            The style of the plot. Options are "fancy", "scientific", or None.
            If None, uses matplotlib defaults without custom styling.
        color_style : str | None
            The color style of the plot. Options are "color", "monochrome", or None.
            If None, uses matplotlib defaults without custom colors.
        save_path : str | None
            If provided, the plot will be saved to this path. If the plot_separately
            parameter is True, the fit's observation well name will be appended to the
            file name.
        num : int
            Number of ticks on the x-axis (default is 6).
        plot_separately : bool
            If True, each fit will be plotted in a separate figure. Default is False.
        ax : matplotlib.axes.Axes | None
            Optional matplotlib Axes object to plot on. If provided, the plot will be
            drawn on this axes instead of creating a new figure. Not compatible with
            plot_separately=True.
        offset_text : dict[str, float]
            A dictionary containing well names as keys and vertical offset values as
            values. This is used to offset the text labels for specific wells to
            avoid overlap. Default is None, which means no offset.
        **kwargs : dict
            Additional keyword arguments for customization. See the documentation of
            Matplotlib's `plt.subplots` and `plt.savefig` for more details.
            Common kwargs include:

            - figsize (tuple): Size of the figure (width, height) in inches.
            - dpi (int): Dots per inch for the saved figure.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        ax : matplotlib.axes.Axes
            The axes object of the plot.
        """
        if fits is not None and not (
            isinstance(fits, FitResultData)
            or (
                isinstance(fits, list)
                and all(isinstance(f, FitResultData) for f in fits)
            )
        ):
            logger.error(
                "fits must be a FitResultData instance or a list of "
                "FitResultData instances"
            )
            raise TypeError(
                "fits must be a FitResultData instance or a list of "
                "FitResultData instances"
            )

        # Validate ax parameter compatibility
        if ax is not None and plot_separately:
            logger.error("ax parameter cannot be used with plot_separately=True")
            raise ValueError("ax parameter cannot be used with plot_separately=True")

        if fits is None:
            fits = self.fits
        elif isinstance(fits, FitResultData):
            fits = [fits]

        # Validate and store the plot styles
        self._validate_plot_styles(plot_style, color_style, offset_text)

        # Get the figsize
        figsize = kwargs.pop("figsize", (10, 6))

        # Create the plot/s
        if plot_separately:
            figs, axs = [], []
            for fit in fits:
                fig, ax = plt.subplots(figsize=figsize, **kwargs)
                self._set_plot_labels(ax, title, xlabel, ylabel)
                logger.debug(f"Plotting fit: {fit.obs_well.name} ~ {fit.ref_well.name}")
                self._set_plot_attributes(fit.obs_well)
                self._set_plot_attributes(fit.ref_well)
                self._plot_well(fit.obs_well, ax)
                self._plot_fit(fit.obs_well, ax)
                self._plot_well(fit.ref_well, ax)
                if mark_outliers:
                    self._plot_outliers(fit.obs_well, ax)
                if show_initiation_period:
                    self._plot_initiation_period(fit, ax)

                self._plot_settings(ax, num)

                if save_path is not None:
                    path_parts = save_path.rsplit(".", 1)
                    if len(path_parts) == 2:
                        save_path_fit = (
                            f"{path_parts[0]}_{fit.obs_well.name}.{path_parts[1]}"
                        )
                    else:
                        save_path_fit = f"{save_path}_{fit.obs_well.name}.png"
                    plt.savefig(save_path_fit, **kwargs)
                    logger.info(f"Plot saved to {save_path_fit}")

                figs.append(fig)
                axs.append(ax)
            return figs, axs

        # Plot all fits in a single figure
        if ax is not None:
            # Use provided axes
            fig = ax.figure
            self._set_plot_labels(ax, title, xlabel, ylabel)
            for fit in fits:
                logger.info(f"Plotting fit: {fit.obs_well.name} ~ {fit.ref_well.name}")
                self._set_plot_attributes(fit.obs_well)
                self._set_plot_attributes(fit.ref_well)
                self._plot_well(fit.obs_well, ax)
                self._plot_fit(fit.obs_well, ax)
                self._plot_well(fit.ref_well, ax)
                if mark_outliers:
                    self._plot_outliers(fit.obs_well, ax)
                if show_initiation_period:
                    self._plot_initiation_period(fit, ax)

            self._plot_settings(ax, num)

            if save_path is not None:
                # Handle both Figure and SubFigure cases
                if isinstance(fig, Figure):
                    fig.savefig(save_path, **kwargs)
                else:
                    # For SubFigure, use the parent figure's savefig
                    plt.savefig(save_path, **kwargs)
                logger.info(f"Plot saved to {save_path}")
        else:
            # Create new figure and axes
            fig, ax = plt.subplots(figsize=figsize, **kwargs)
            self._set_plot_labels(ax, title, xlabel, ylabel)
            for fit in fits:
                logger.info(f"Plotting fit: {fit.obs_well.name} ~ {fit.ref_well.name}")
                self._set_plot_attributes(fit.obs_well)
                self._set_plot_attributes(fit.ref_well)
                self._plot_well(fit.obs_well, ax)
                self._plot_fit(fit.obs_well, ax)
                self._plot_well(fit.ref_well, ax)
                if mark_outliers:
                    self._plot_outliers(fit.obs_well, ax)
                if show_initiation_period:
                    self._plot_initiation_period(fit, ax)

            self._plot_settings(ax, num)

            if save_path is not None:
                plt.savefig(save_path, **kwargs)
                logger.info(f"Plot saved to {save_path}")

        assert ax is not None  # Type narrowing for type checker
        return fig, ax

    def plot_fitmethod(
        self,
        fits: FitResultData | list[FitResultData] = None,
        title: str = "Fit Method Plot",
        xlabel: str = "Hydraulic Head Reference Well",
        ylabel: str = "Hydraulic Head Observation Well",
        plot_style: str | None = None,
        color_style: str | None = None,
        save_path: str | None = None,
        plot_separately: bool = False,
        ax: Axes | None = None,
        **kwargs,
    ) -> tuple[Figure | SubFigure, Axes] | tuple[list[Figure], list[Axes]]:
        """
        This method plots the time series data for all fits in the model.

        Parameters
        ----------
        fits : FitResultData | list[FitResultData]
            A FitResultData instance or a list of FitResultData instances
            containing the fit results to be plotted. If None, all fits will be plotted.
        title : str
            The title of the plot.
        xlabel : str
            The label for the x-axis.
        ylabel : str
            The label for the y-axis.
        plot_style : str | None
            The style of the plot. Options are "fancy", "scientific", or None.
            If None, uses matplotlib defaults without custom styling.
        color_style : str | None
            The color style of the plot. Options are "color", "monochrome", or None.
            If None, uses matplotlib defaults without custom colors.
        save_path : str | None
            If provided, the plot will be saved to this path. If the plot_separately
            parameter is True, the fit's observation well name will be appended to the
            file name.
        plot_separately : bool
            If True, each fit will be plotted in a separate figure. Default is False.
        ax : matplotlib.axes.Axes | None
            Optional matplotlib Axes object to plot on. If provided, the plot will be
            drawn on this axes instead of creating a new figure. Not compatible with
            plot_separately=True.
        **kwargs : dict
            Additional keyword arguments for customization. See the documentation of
            Matplotlib's `plt.subplots` and `plt.savefig` for more details.
            Common kwargs include:

            - figsize (tuple): Size of the figure (width, height) in inches.
            - dpi (int): Dots per inch for the saved figure.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        ax : matplotlib.axes.Axes
            The axes object of the plot.
        """
        if fits is not None and not (
            isinstance(fits, FitResultData)
            or (
                isinstance(fits, list)
                and all(isinstance(f, FitResultData) for f in fits)
            )
        ):
            logger.error(
                "fits must be a FitResultData instance or a list of "
                "FitResultData instances"
            )
            raise TypeError(
                "fits must be a FitResultData instance or a list of "
                "FitResultData instances"
            )

        # Validate ax parameter compatibility
        if ax is not None and plot_separately:
            logger.error("ax parameter cannot be used with plot_separately=True")
            raise ValueError("ax parameter cannot be used with plot_separately=True")

        if fits is None:
            fits = self.fits
        elif isinstance(fits, FitResultData):
            fits = [fits]

        # Validate and store the plot styles
        self._validate_plot_styles(plot_style, color_style, 0)

        # Get the figsize
        figsize = kwargs.pop("figsize", (5, 5))

        # Create the plot/s
        if plot_separately:
            figs, axs = [], []
            for fit in fits:
                fig, ax = plt.subplots(figsize=figsize, **kwargs)
                self._set_plot_labels(ax, title, xlabel, ylabel)
                logger.debug(f"Plotting fit: {fit.obs_well.name} ~ {fit.ref_well.name}")
                self._set_plot_attributes(fit.obs_well)
                self._set_plot_attributes(fit.ref_well)
                self._plot_well_scatter(fit.obs_well, fit.ref_well, ax)
                self._plot_fitmethod(fit, ax)

                if plot_style is not None:
                    ax.legend(prop=lfont)
                    # Update the ticks font if a plot style is set
                    for label in ax.get_xticklabels() + ax.get_yticklabels():
                        label.set_fontfamily(tifont["family"])
                        label.set_fontsize(tifont["size"])
                else:
                    ax.legend()

                fig.tight_layout()

                if save_path is not None:
                    path_parts = save_path.rsplit(".", 1)
                    if len(path_parts) == 2:
                        save_path_fit = (
                            f"{path_parts[0]}_{fit.obs_well.name}.{path_parts[1]}"
                        )
                    else:
                        save_path_fit = f"{save_path}_{fit.obs_well.name}.png"
                    plt.savefig(save_path_fit, **kwargs)
                    logger.info(f"Plot saved to {save_path_fit}")

                figs.append(fig)
                axs.append(ax)
            return figs, axs

        # Plot all fits in a single figure
        if ax is not None:
            # Use provided axes
            fig = ax.figure
            self._set_plot_labels(ax, title, xlabel, ylabel)
            for fit in fits:
                logger.info(f"Plotting fit: {fit.obs_well.name} ~ {fit.ref_well.name}")
                self._set_plot_attributes(fit.obs_well)
                self._set_plot_attributes(fit.ref_well)
                self._plot_well_scatter(fit.obs_well, fit.ref_well, ax)
                self._plot_fitmethod(fit, ax)
            if plot_style is not None:
                ax.legend(prop=lfont)
                # Update the ticks font if a plot style is set
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontfamily(tifont["family"])
                    label.set_fontsize(tifont["size"])
            else:
                ax.legend()

            fig.tight_layout()
            if save_path is not None:
                # Handle both Figure and SubFigure cases
                if isinstance(fig, Figure):
                    fig.savefig(save_path, **kwargs)
                else:
                    # For SubFigure, use the parent figure's savefig
                    plt.savefig(save_path, **kwargs)
                logger.info(f"Plot saved to {save_path}")
        else:
            # Create new figure and axes
            fig, ax = plt.subplots(figsize=figsize, **kwargs)
            self._set_plot_labels(ax, title, xlabel, ylabel)
            for fit in fits:
                logger.info(f"Plotting fit: {fit.obs_well.name} ~ {fit.ref_well.name}")
                self._set_plot_attributes(fit.obs_well)
                self._set_plot_attributes(fit.ref_well)
                self._plot_well_scatter(fit.obs_well, fit.ref_well, ax)
                self._plot_fitmethod(fit, ax)
            if plot_style is not None:
                ax.legend(prop=lfont)
                # Update the ticks font if a plot style is set
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontfamily(tifont["family"])
                    label.set_fontsize(tifont["size"])
            else:
                ax.legend()

            fig.tight_layout()
            if save_path is not None:
                plt.savefig(save_path, **kwargs)
                logger.info(f"Plot saved to {save_path}")

    def _set_plot_labels(self, ax, title, xlabel, ylabel):
        """Set plot labels with conditional font styling."""
        if self._plot_style is not None:
            ax.set_title(title, **tfont)
            ax.set_xlabel(xlabel, **afont)
            ax.set_ylabel(ylabel, **afont)
        else:
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

    def _validate_plot_styles(self, plot_style, color_style, offset_text):
        # Store the plot style
        if plot_style not in ["fancy", "scientific", None]:
            logger.error("Invalid plot_style. Must be 'fancy', 'scientific', or None.")
            raise ValueError("plot_style must be 'fancy', 'scientific', or None")
        self._plot_style = plot_style

        # Store the color style
        if color_style not in ["color", "monochrome", None]:
            logger.error("Invalid color_style. Must be 'color', 'monochrome', or None.")
            raise ValueError("color_style must be 'color', 'monochrome', or None")
        if self._color_style is not None and self._color_style != color_style:
            logger.warning(
                "Color style has changed from "
                f"'{self._color_style}' to '{color_style}'. "
                "Plot attributes will be reset and reassigned."
            )
            self._cnt_colors = 0
            self._cnt_linestyles = 0
            self._cnt_markers = 0
            for well in self.wells:
                well.color = None
                well.linestyle = None
                well.marker = None
        self._color_style = color_style

        self._offset_text = offset_text

    def _plot_well(self, well, ax):
        """Plot the time series data for a single well."""
        ax.plot(
            well.timeseries.index,
            well.timeseries.values,
            label=well.name,
            color=well.color,
            alpha=well.alpha,
            linestyle=well.linestyle,
            linewidth=well.linewidth,
            marker=well.marker if well.marker_visible else None,
            markersize=well.markersize,
        )
        self._update_axis_limits(well)
        if self._plot_style == "fancy":
            offset = (
                self._offset_text.get(well.name, 0.0)
                if (self._offset_text is not None)
                else 0.0
            )
            ax.text(
                well.timeseries.index[-1],
                well.timeseries.values[-1] + offset,
                f" {well.name}",
                color=well.color,
                horizontalalignment="left",
                verticalalignment="center",
                **lfont,
            )

    def _plot_well_scatter(self, obs_well, ref_well, ax):
        """Plot the time series data for a single well as a scatter plot."""
        ax.scatter(
            ref_well.timeseries.values,
            obs_well.timeseries.values,
            label=f"{obs_well.name} ~ {ref_well.name}",
            color=ref_well.color,
            alpha=1,
            marker="o",
            s=6,  # markersize is in points, s is in points^2
        )

    def _plot_fit(self, well, ax):
        """Plot the fitted model for a single well."""
        fits = self.get_fits(well)
        if isinstance(fits, list) is False:
            fits = [fits]
        for fit in fits:
            pred_const = fit.pred_const
            fit_timeseries = fit.fit_timeseries()
            x = fit_timeseries.index
            y = fit_timeseries.values
            ax.plot(x, y, linestyle="-", color=well.color, alpha=0.2, label=None)
            ax.fill_between(
                x,
                y - pred_const,
                y + pred_const,
                color=well.color,
                alpha=0.2,
                label=None,
            )
        logger.debug(f"Plotting fit for well: {well.name}")

    def _plot_fitmethod(self, fit, ax):
        """Plot the fit method for a single fit."""
        fm = fit.fit_method
        if isinstance(fm, LinRegResult):
            slope = fm.slope
            intercept = fm.intercept
            x = fit.ref_well.timeseries.values
            x = np.array([np.min(x), np.max(x)])
            y = intercept + slope * x
            ax.plot(x, y, color="black", linestyle="--", label="Linear regression")
            ax.plot(
                x[0],
                y[0],
                color="white",
                linestyle=None,
                marker=None,
                label=f"RMSE = {fit.rmse:.4f}\n"
                f"R$^2$ = {fm.rvalue**2:.4f}\n"
                f"y = {slope:.4f}x + {intercept:.4f}",
            )
        else:
            logger.error("Fit method not recognized for plotting.")
            raise TypeError("Fit method not recognized for plotting.")
        logger.debug(f"Plotting fit method for well: {fit.obs_well.name}")

    def _plot_outliers(self, well, ax):
        """Mark outliers on the plot for a single well."""
        fit = self.get_fits(well)
        if isinstance(fit, list):
            fit = fit[0]
        outliers = fit.fit_outliers()
        well_outliers = well.timeseries[outliers]
        if self._color_style is None:
            edgecolor = "red"  # Use matplotlib default
        else:
            edgecolor = "red" if self._color_style == "color" else "black"
        if well_outliers is not None and not well_outliers.empty:
            ax.scatter(
                well_outliers.index,
                well_outliers.values,
                edgecolor=edgecolor,
                facecolors="none",
                marker="o",
                s=50,
                label=None,
                zorder=500,
            )
            logger.debug(f"Marking outliers for well: {well.name}")

    @staticmethod
    def _plot_initiation_period(fit, ax):
        """Shade the initiation period on the plot for a single fit."""
        if fit.tmin is not None:
            ax.axvspan(
                fit.tmin,
                fit.tmax,
                color="#E0E0E0",
                alpha=0.3,
                label="Initiation Period",
                zorder=0,
                hatch="xx",
            )
            logger.debug(f"Shading initiation period for fit: {fit.obs_well.name}")

    def _set_plot_attributes(self, well):
        """Set default plot attributes for a well if not already set."""
        # Set default plot attributes if not already set
        if well.color is None and self._color_style is not None:
            cnt = self._cnt_colors
            if self._color_style == "monochrome":
                well.color = DEFAULT_MONOCHROME_COLORS[
                    cnt % len(DEFAULT_MONOCHROME_COLORS)
                ]
            else:
                well.color = DEFAULT_COLORS[cnt % len(DEFAULT_COLORS)]
            self._cnt_colors += 1
        if well.linestyle is None:
            cnt = self._cnt_linestyles
            well.linestyle = DEFAULT_LINE_STYLES[cnt % len(DEFAULT_LINE_STYLES)]
            self._cnt_linestyles += 1
        if well.marker is None:
            cnt = self._cnt_markers
            well.marker = DEFAULT_MARKER_STYLES[cnt % len(DEFAULT_MARKER_STYLES)]
            self._cnt_markers += 1
        if well.markersize is None:
            well.markersize = 6
        if well.alpha is None:
            well.alpha = 1.0

    def _update_axis_limits(self, well):
        """Update the axis limits based on the well's time series data."""
        if self._xmin is None or well.timeseries.index.min() < self._xmin:
            self._xmin = well.timeseries.index.min()
        if self._xmax is None or well.timeseries.index.max() > self._xmax:
            self._xmax = well.timeseries.index.max()
        if self._ymin is None or well.timeseries.min() < self._ymin:
            self._ymin = well.timeseries.min()
        if self._ymax is None or well.timeseries.max() > self._ymax:
            self._ymax = well.timeseries.max()

    def _plot_settings(self, ax, num):
        """Apply final plot settings based on the selected style."""
        if self._plot_style == "fancy":
            self._plot_settings_fancy(ax)
        elif self._plot_style == "scientific":
            self._plot_settings_scientific(ax)
        elif self._plot_style is None:
            # Skip custom styling, use matplotlib defaults
            pass

        # limit x axis to data range
        ax.set_xlim(left=self._xmin, right=self._xmax)

        # Apply custom formatting only if plot_style is not None
        if self._plot_style is not None:
            # Set ticks font
            xticks = np.linspace(date2num(self._xmin), date2num(self._xmax), num=num)
            xlabels = [f"{num2date(tick):%Y-%m-%d}" for tick in xticks]
            yticks = ax.get_yticks()
            ylabels = [item.get_text() for item in ax.get_yticklabels()]
            ax.set_xticks(xticks)
            ax.set_yticks(yticks)
            ax.set_xticklabels(xlabels, **tifont)
            ax.set_yticklabels(ylabels, **tifont)

            # Set font sizes and styles
            ax.title.set_fontsize(16)
            ax.xaxis.label.set_fontsize(14)
            ax.yaxis.label.set_fontsize(14)
            ax.tick_params(axis="both", which="major", labelsize=12)

        # Tight layout
        plt.tight_layout()

    def _plot_settings_fancy(self, ax):
        """Apply fancy plot settings."""
        # Hide the all but the bottom spines (axis lines)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")
        ax.spines["bottom"].set_bounds(date2num(self._xmin), date2num(self._xmax))

        # Add grid lines
        ax.grid(
            visible=True,
            which="major",
            color="#E8E8E8",
            linestyle="--",
            linewidth=0.5,
        )
        ax.grid(
            visible=True, which="minor", color="#E8E8E8", linestyle=":", linewidth=0.5
        )

    @staticmethod
    def _plot_settings_scientific(ax):
        """Apply scientific plot settings."""
        # Add grid lines
        ax.grid(
            visible=True,
            which="major",
            color="black",
            linestyle="--",
            linewidth=0.5,
        )
        ax.grid(
            visible=True, which="minor", color="black", linestyle=":", linewidth=0.5
        )

        ax.legend(prop=lfont)

    def get_fits(self, well):
        raise NotImplementedError("Subclasses should implement this method.")
