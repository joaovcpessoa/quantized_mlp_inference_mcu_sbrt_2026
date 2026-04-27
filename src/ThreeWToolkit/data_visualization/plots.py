from abc import ABC
from matplotlib.figure import Figure
import numpy as np
from matplotlib.axes import Axes
import pandas as pd

from .plot_utils import create_subplot_grid
from .plot_series import PlotSeries
from .plot_multiple_series import PlotMultipleSeries
from .correlation_heatmap import CorrelationHeatmap
from .plot_fft import PlotFFT
from .seasonal_decomposition import SeasonalDecompositionPlot
from .wavelet_spectrogram import WaveletSpectrogramPlot


class DataVisualization(ABC):
    """
    Facade class providing a backward-compatible static API for data visualization.

    - New style:
        vis = PlotSeries(...)
        fig, path = vis.plot()

    - Old style (still supported):
        fig, path = DataVisualization.plot_series(...)
    """

    @staticmethod
    def plot_series(
        series: pd.Series,
        title: str,
        xlabel: str,
        ylabel: str,
        overlay_events: bool = False,
        ax: Axes | None = None,
        **plot_kwargs,
    ) -> tuple[Figure, Axes]:
        """
        Plot a single time series.

        Args:
            series: Input pandas Series to be plotted.
            title: Title of the plot.
            xlabel: Label for the x-axis.
            ylabel: Label for the y-axis.
            overlay_events: Whether to overlay event markers at NaN positions.
            ax: Optional matplotlib Axes to draw on.
            **plot_kwargs: Additional keyword arguments forwarded to
                matplotlib Axes.plot.

        Returns:
            A tuple containing:
                - fig: The matplotlib Figure object.
                - ax: The matplotlib Axes with the rendered plot.

        Raises:
            ValueError: If the series is empty or contains only NaN values.
        """
        vis = PlotSeries(
            series=series,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            overlay_events=overlay_events,
            **plot_kwargs,
        )
        return vis.plot(ax=ax)

    @staticmethod
    def plot_multiple_series(
        series_list: list[pd.Series],
        labels: list[str],
        title: str,
        xlabel: str,
        ylabel: str,
        ax: Axes | None = None,
        **plot_kwargs,
    ) -> tuple[Figure, Axes]:
        """
        Plot multiple time series on the same axes.

        Args:
            series_list: List of pandas Series to be plotted.
            labels: List of labels corresponding to each series.
            title: Title of the plot.
            xlabel: Label for the x-axis.
            ylabel: Label for the y-axis.
            ax: Optional matplotlib Axes to draw on.
            **plot_kwargs: Additional keyword arguments forwarded to
                matplotlib Axes.plot.

        Returns:
            A tuple containing:
                - fig: The matplotlib Figure object.
                - ax: The matplotlib Axes with the rendered time series.

        Raises:
            ValueError: If series_list is empty or lengths do not match labels.
        """
        vis = PlotMultipleSeries(
            series_list=series_list,
            labels=labels,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            **plot_kwargs,
        )
        return vis.plot(ax=ax)

    @staticmethod
    def correlation_heatmap(
        df_of_series: pd.DataFrame,
        ax: Axes | None = None,
        **kwargs,
    ) -> tuple[Figure, Axes]:
        """
        Plot a correlation heatmap from a DataFrame of series.

        Args:
            df_of_series: DataFrame containing multiple series or variables.
            ax: Optional matplotlib Axes to draw on.
            **kwargs: Additional keyword arguments forwarded to seaborn.heatmap.

        Returns:
            A tuple containing:
                - fig: The matplotlib Figure object.
                - ax: The matplotlib Axes with the rendered heatmap.

        Raises:
            ValueError: If the DataFrame contains only NaN values.
        """
        vis = CorrelationHeatmap(df_of_series=df_of_series, **kwargs)
        return vis.plot(ax=ax)

    @staticmethod
    def plot_fft(
        series: pd.Series,
        title: str = "FFT Analysis",
        sample_rate: float | None = None,
        ax: Axes | None = None,
    ) -> tuple[Figure, Axes]:
        """
        Plot the Fast Fourier Transform (FFT) of a time series.

        Args:
            series: Input pandas Series used to compute the FFT.
            title: Title of the FFT plot.
            sample_rate: Optional sampling rate used for frequency scaling.
            ax: Optional matplotlib Axes to draw on.

        Returns:
            A tuple containing:
                - fig: The matplotlib Figure object.
                - ax: The matplotlib Axes with the FFT amplitude spectrum.

        Raises:
            ValueError: If the series is empty or contains only NaN values.
        """
        vis = PlotFFT(series=series, title=title, sample_rate=sample_rate)
        return vis.plot(ax=ax)

    @staticmethod
    def seasonal_decompose(
        series: pd.Series,
        model: str = "additive",
        period: int | None = None,
        ax: Axes | None = None,
    ) -> tuple[Figure, Axes]:
        """
        Perform and plot a seasonal decomposition of a time series.

        Args:
            series: Input pandas Series to decompose.
            model: Type of seasonal component. Typically 'additive' or 'multiplicative'.
            period: Period of the seasonal component.
            ax: Optional matplotlib Axes to draw on.

        Returns:
            A tuple containing:
                - fig: The matplotlib Figure object.
                - ax: The matplotlib Axes with the decomposition plot.

        Raises:
            ValueError: If the series is empty or invalid for decomposition.
        """
        vis = SeasonalDecompositionPlot(series=series, model=model, period=period)
        return vis.plot(ax=ax)

    @staticmethod
    def plot_wavelet_spectrogram(
        series: pd.Series,
        title: str = "Wavelet Spectrogram",
        ax: Axes | None = None,
    ) -> tuple[Figure, Axes]:
        """
        Plot a wavelet spectrogram of a time series.

        Args:
            series: Input pandas Series used to generate the spectrogram.
            title: Title of the spectrogram plot.
            ax: Optional matplotlib Axes to draw on.

        Returns:
            A tuple containing:
                - fig: The matplotlib Figure object.
                - ax: The matplotlib Axes with the spectrogram.

        Raises:
            ValueError: If the series is empty.
        """
        vis = WaveletSpectrogramPlot(series=series, title=title)
        return vis.plot(ax=ax)

    @staticmethod
    def create_subplot_grid(
        nrows: int,
        ncols: int,
        figsize: tuple[int, int] | None = None,
    ) -> tuple[Figure, np.ndarray]:
        """
        Create a grid of matplotlib subplots.

        This is a backward-compatible wrapper around
        plot_utils.create_subplot_grid().

        Args:
            nrows: Number of rows in the subplot grid.
            ncols: Number of columns in the subplot grid.
            figsize: Optional figure size as (width, height).

        Returns:
            A tuple containing:
                - fig: The matplotlib Figure object.
                - axes: A 2D NumPy array of Axes objects.

        Raises:
            ValueError: If nrows or ncols are not positive integers.
        """
        return create_subplot_grid(nrows=nrows, ncols=ncols, figsize=figsize)
