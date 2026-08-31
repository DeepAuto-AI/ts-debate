"""
Chart Generation Utilities for Time Series Visualization

Converts raw time series numbers into visual charts for multimodal LLM input.
Supports various chart types optimized for time series analysis.
"""

import base64
import io
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from .constants import DATETIME_FORMATS
from .sample_rate_inference import KNOWN_SAMPLE_RATES


class TimeSeriesChartGenerator:
    """Generate charts from time series data for LLM vision input"""

    def __init__(self, figsize: Tuple[int, int] = (12, 6), dpi: int = 100, style: str = "seaborn-v0_8-talk"):
        """
        Args:
            figsize: Figure size (width, height) in inches
            dpi: Dots per inch for resolution
            style: Matplotlib style
        """
        self.figsize = figsize
        self.dpi = dpi
        plt.style.use(style)

        # Set publication-quality defaults
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
        plt.rcParams["font.size"] = 11
        plt.rcParams["axes.labelsize"] = 12
        plt.rcParams["axes.titlesize"] = 14
        plt.rcParams["xtick.labelsize"] = 10
        plt.rcParams["ytick.labelsize"] = 10
        plt.rcParams["legend.fontsize"] = 10
        plt.rcParams["figure.titlesize"] = 16

    # Class constants
    DEFAULT_COLORS = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E", "#BC4B51"]

    # Visualization constants
    GRID_ALPHA = 0.3  # Grid transparency
    HIGHLIGHT_ALPHA = 0.2  # Highlight region transparency
    PREDICTION_ALPHA = 0.1  # Prediction horizon transparency
    STATS_BOX_ALPHA = 0.3  # Statistics box background transparency
    LINE_ALPHA_THICK = 0.8  # Line transparency for thick lines (small datasets)
    LINE_ALPHA_THIN = 0.9  # Line transparency for thin lines (large datasets)
    RECTANGLE_ALPHA = 0.8  # Rectangle/candlestick transparency
    VERTICAL_LINE_ALPHA = 0.5  # Vertical line transparency

    # Line width thresholds
    THICK_LINEWIDTH_THRESHOLD = 50  # Use thicker linewidth for datasets <= this size
    THICK_LINEWIDTH = 3.5
    THIN_LINEWIDTH = 2

    # Subplot sizing
    SUBPLOT_HEIGHT_MULTIPLIER = 3  # Height per channel in inches

    # Datetime formatting
    SECONDS_PER_HOUR = 3600  # Used for datetime format selection

    # Indicator subplot sizing
    INDICATOR_SUBPLOT_RATIO = 3  # Main chart height : indicator chart height ratio

    # Helper methods for code reuse
    def _is_multivariate(self, values: Union[List[float], List[List[float]]]) -> bool:
        """
        Detect if values represent multivariate time series.

        Args:
            values: Time series values (univariate or multivariate)

        Returns:
            True if multivariate (list of lists with >1 channel), False otherwise
        """
        return isinstance(values, list) and len(values) > 1 and isinstance(values[0], list)

    def _normalize_univariate_values(self, values: Union[List[float], List[List[float]], np.ndarray]) -> List[float]:
        """
        Normalize univariate values: handle wrapped lists like [[1,2,3]] -> [1,2,3]
        Also handles numpy arrays (1D or 2D with 1 row).

        Args:
            values: Time series values (can be wrapped list, flat list, or numpy array)

        Returns:
            Normalized flat list of floats
        """
        if isinstance(values, list):
            # Single-element list of lists: [[1,2,3]] -> [1,2,3]
            if len(values) == 1 and isinstance(values[0], list):
                return values[0]
            # Already flat list: [1,2,3]
            if all(isinstance(v, (int, float)) for v in values):
                return values
        if isinstance(values, np.ndarray):
            if values.ndim == 1:
                return values.tolist()
            if values.ndim == 2 and values.shape[0] == 1:
                return values[0].tolist()  # Single row -> flatten
        # Fallback: convert to list
        return list(np.array(values).flatten())

    def _format_statistics_text(self, values: Union[List[float], np.ndarray], handle_nan: bool = False) -> str:
        """
        Format statistics as text string for subtitle.
        Returns: "Mean: X.XX, Std: X.XX, Range: [X.XX, X.XX]"

        Args:
            values: Time series values
            handle_nan: Whether to handle NaN values (exclude from stats)

        Returns:
            Formatted statistics string
        """
        values_arr = np.array(values, dtype=float)
        if handle_nan:
            valid_mask = ~np.isnan(values_arr)
            if np.sum(valid_mask) == 0:
                return "No valid data"
            valid_values = values_arr[valid_mask]
            mean_val = np.mean(valid_values)
            std_val = np.std(valid_values)
            min_val = np.min(valid_values)
            max_val = np.max(valid_values)
        else:
            mean_val = np.mean(values_arr)
            std_val = np.std(values_arr)
            min_val = np.min(values_arr)
            max_val = np.max(values_arr)
        return f"Mean: {mean_val:.2f}, Std: {std_val:.2f}, Range: [{min_val:.2f}, {max_val:.2f}]"

    def _finalize_figure(self, fig, save_path: Optional[str] = None) -> str:
        """
        Finalize figure: save if path provided, convert to base64, and close.

        Args:
            fig: Matplotlib figure object
            save_path: Optional path to save figure

        Returns:
            Base64 encoded image string
        """
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=self.dpi)

        img_base64 = self._fig_to_base64(fig)
        plt.close(fig)
        return img_base64

    def _format_datetime_ticks(
        self, ax, parsed_timestamps: List[datetime], n_ticks: Optional[int] = None, n_values: Optional[int] = None
    ) -> None:
        """
        Format x-axis ticks for datetime timestamps with adaptive tick count.

        Args:
            ax: Matplotlib axes object
            parsed_timestamps: List of datetime objects
            n_ticks: Number of ticks to display (if None, adaptively calculated)
            n_values: Number of data points (if None, uses len(parsed_timestamps))
        """
        if n_values is None:
            n_values = len(parsed_timestamps)

        # Adaptive tick calculation based on data length
        if n_ticks is None:
            if n_values <= 10:
                n_ticks = min(5, n_values)  # Very small datasets: show all or up to 5
            elif n_values <= 50:
                n_ticks = 6  # Small datasets: 6 ticks
            elif n_values <= 200:
                n_ticks = 8  # Medium datasets: 8 ticks (default)
            elif n_values <= 1000:
                n_ticks = 10  # Large datasets: 10 ticks
            else:
                n_ticks = 12  # Very large datasets: 12 ticks

        n_ticks = min(n_ticks, n_values)
        tick_idx = np.linspace(0, n_values - 1, n_ticks, dtype=int)

        span_secs = (parsed_timestamps[-1] - parsed_timestamps[0]).total_seconds() if len(parsed_timestamps) > 1 else 0
        avg_interval = span_secs / max(n_values - 1, 1)
        fmt = "%Y-%m-%d %H:%M" if avg_interval < self.SECONDS_PER_HOUR else "%Y-%m-%d"

        ax.set_xticks(tick_idx)
        ax.set_xticklabels([parsed_timestamps[i].strftime(fmt) for i in tick_idx], rotation=45, ha="right")

    def _format_time_ticks(self, ax, time_values: np.ndarray, n_ticks: Optional[int] = None) -> None:
        """
        Format x-axis ticks for time values in seconds with adaptive tick count.

        Args:
            ax: Matplotlib axes object
            time_values: Array of time values in seconds
            n_ticks: Number of ticks to display (if None, adaptively calculated)
        """
        n_values = len(time_values)

        # Adaptive tick calculation based on data length
        if n_ticks is None:
            if n_values <= 10:
                n_ticks = min(5, n_values)  # Very small datasets: show all or up to 5
            elif n_values <= 50:
                n_ticks = 6  # Small datasets: 6 ticks
            elif n_values <= 200:
                n_ticks = 8  # Medium datasets: 8 ticks (default)
            elif n_values <= 1000:
                n_ticks = 10  # Large datasets: 10 ticks
            else:
                n_ticks = 12  # Very large datasets: 12 ticks

        n_ticks = min(n_ticks, n_values)
        tick_idx = np.linspace(0, n_values - 1, n_ticks, dtype=int)

        # Generate tick labels with adaptive precision to avoid duplicates
        tick_labels = []
        tick_positions = []
        prev_label = None

        for i in tick_idx:
            t = time_values[i]
            # Use adaptive precision based on magnitude
            if t < 0.1:
                label = f"{t:.3f}s"
            elif t < 1:
                label = f"{t:.2f}s"
            elif t < 10:
                label = f"{t:.1f}s"
            else:
                label = f"{t:.0f}s"

            # Deduplicate: only add if different from previous label
            if label != prev_label:
                tick_labels.append(label)
                tick_positions.append(t)
                prev_label = label

        # Set ticks and labels
        if tick_positions:
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45, ha="right")

    def _create_multivariate_subplots(
        self, n_channels: int, sharex: bool = True, sharey: bool = False
    ) -> Tuple[plt.Figure, List]:
        """
        Create subplots for multivariate charts.

        Args:
            n_channels: Number of channels/variables
            sharex: Whether to share x-axis across subplots
            sharey: Whether to share y-axis across subplots

        Returns:
            Tuple of (figure, axes_list)
        """
        fig, axes = plt.subplots(
            n_channels,
            1,
            figsize=(self.figsize[0], self.SUBPLOT_HEIGHT_MULTIPLIER * n_channels),
            dpi=self.dpi,
            sharex=sharex,
            sharey=sharey,
        )
        if n_channels == 1:
            axes = [axes]
        return fig, axes

    def create_chart(
        self,
        data: List[float],
        timestamps: Optional[List[Union[str, datetime]]] = None,
        chart_type: str = "line",
        title: str = "Time Series",
        ylabel: str = "Value",
        highlight_region: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None,
        sample_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generic chart creation method (wrapper for specific chart types)
        For forecasting tasks, only historical data should be shown.

        Args:
            data: List of numerical values
            timestamps: Optional list of timestamps
            chart_type: Type of chart ('line', 'candlestick', 'prediction', etc.)
            title: Chart title
            ylabel: Y-axis label
            highlight_region: Optional tuple (start_idx, end_idx) to highlight
            save_path: Optional path to save image
            sample_metadata: Optional dict with dataset_name, domain, source for time conversion

        Returns:
            Base64 encoded image string
        """
        # Dispatch to appropriate chart creation method
        if chart_type == "line":
            return self.create_line_chart(
                data, timestamps, title, ylabel, highlight_region, save_path, sample_metadata=sample_metadata
            )
        if chart_type == "prediction":
            # Caller passes only historical data (input values), NOT ground truth
            # prediction_horizon can be passed via highlight_region[1] to show shaded region
            pred_horizon = highlight_region[1] if highlight_region else 0
            return self.create_prediction_chart(
                historical_values=data,
                predicted_values=[],
                timestamps=timestamps,
                title=title,
                prediction_horizon=pred_horizon,
                save_path=save_path,
                sample_metadata=sample_metadata,
            )
        if chart_type == "imputation":
            # For imputation tasks, show data with gaps (NaN) clearly marked
            # NO ground truth values shown - model must impute the gaps
            return self.create_imputation_chart(
                data, timestamps, title, ylabel, save_path, sample_metadata=sample_metadata
            )
        # Default to line chart for unknown types
        return self.create_line_chart(
            data, timestamps, title, ylabel, highlight_region, save_path, sample_metadata=sample_metadata
        )

    def create_line_chart(
        self,
        values: Union[List[float], List[List[float]]],
        timestamps: Optional[List[Union[str, datetime]]] = None,
        title: str = "Time Series",
        ylabel: str = "Value",
        highlight_region: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None,
        show: bool = False,
        sample_metadata: Optional[Dict[str, Any]] = None,
        indicator_values: Optional[List[float]] = None,
        indicator_label: str = "Indicator",
    ) -> str:
        """
        Create a line chart from time series data (handles both univariate and multivariate)

        Args:
            values: List of numerical values (univariate) or List[List[float]] (multivariate)
            timestamps: Optional list of timestamps
            title: Chart title
            ylabel: Y-axis label
            highlight_region: Optional tuple (start_idx, end_idx) to highlight
            save_path: Optional path to save image
            show: Whether to display the chart (ignored in non-GUI mode)
            sample_metadata: Optional dict with dataset_name, domain, source for time conversion
            indicator_values: Optional pre-computed indicator values (e.g., MACD, BB) to show as subplot
            indicator_label: Label for the indicator subplot y-axis (e.g., "MACD", "Bollinger Band")

        Returns:
            Base64 encoded image string (for LLM vision API)
        """
        # Handle wrapped univariate data FIRST (e.g., [[1,2,3]] -> [1,2,3])
        # Check if it's a single-element list containing a list (wrapped univariate)
        if isinstance(values, list) and len(values) == 1 and isinstance(values[0], list):
            # Unwrap: [[1,2,3]] -> [1,2,3]
            values = values[0]

        # Detect multivariate data (list of lists with >1 channel)
        is_multivariate = self._is_multivariate(values)

        if is_multivariate:
            # Use multivariate chart for multi-dimensional data
            variable_names = [f"dim_{i}" for i in range(len(values))]
            return self.create_multivariate_chart(
                multivariate_data=values,
                variable_names=variable_names,
                timestamps=timestamps,
                title=title,
                ylabel=ylabel,
                save_path=save_path,
                sample_metadata=sample_metadata,
            )

        # Univariate case (original behavior)
        # Normalize univariate values (handle numpy arrays, etc.)
        values = self._normalize_univariate_values(values)
        values = np.array(values, dtype=float)  # Convert to numpy for consistency

        # Generate timestamps if None (for TimerBed sample rate conversion)
        if timestamps is None:
            timestamps = list(range(len(values)))

        # Normalize indicator values if provided
        has_indicator = indicator_values is not None and len(indicator_values) > 0
        if has_indicator:
            indicator_values = self._normalize_univariate_values(indicator_values)
            indicator_arr = np.array(indicator_values, dtype=float)
        else:
            indicator_arr = None

        # Create figure with optional indicator subplot
        if has_indicator:
            fig, (ax, ax_indicator) = plt.subplots(
                2,
                1,
                figsize=(self.figsize[0], self.figsize[1] * 1.4),
                dpi=self.dpi,
                sharex=True,
                gridspec_kw={"height_ratios": [self.INDICATOR_SUBPLOT_RATIO, 1]},
            )
        else:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            ax_indicator = None

        # Convert indices to time if applicable (TimerBed with known sample rates)
        time_values, xlabel = self._convert_indices_to_time(timestamps, sample_metadata, data_length=len(values))

        # Prepare x-axis
        if time_values is not None:
            # Use converted time values (in seconds)
            x = np.array(time_values)
            if len(x) != len(values):
                raise ValueError(f"Length mismatch: x has {len(x)} points, values has {len(values)} points")
            if len(values) <= self.THICK_LINEWIDTH_THRESHOLD:
                ax.plot(x, values, linewidth=self.THICK_LINEWIDTH, color="#2E86AB", alpha=self.LINE_ALPHA_THICK)
            else:
                ax.plot(x, values, linewidth=self.THIN_LINEWIDTH, color="#2E86AB", alpha=self.LINE_ALPHA_THIN)

            # Format time ticks (adaptive)
            self._format_time_ticks(ax, x)
            ax.set_xlabel(xlabel, fontsize=12)
        elif timestamps is not None and len(timestamps) > 0:
            # Try datetime parsing
            parsed_ts = self._parse_timestamps(timestamps)
            if parsed_ts is not None:
                # Datetime objects - use existing logic
                x = np.arange(len(values))
                if len(x) != len(values):
                    raise ValueError(f"Length mismatch: x has {len(x)} points, values has {len(values)} points")
                if len(values) <= 50:
                    ax.plot(x, values, linewidth=self.THICK_LINEWIDTH, color="#2E86AB", alpha=self.LINE_ALPHA_THICK)
                else:
                    ax.plot(x, values, linewidth=self.THIN_LINEWIDTH, color="#2E86AB", alpha=self.LINE_ALPHA_THIN)

                self._format_datetime_ticks(ax, parsed_ts, n_values=len(values))
                ax.set_xlabel("Time", fontsize=12)
            else:
                # Sequential indices - use index display
                x = np.arange(len(values))
                if len(x) != len(values):
                    raise ValueError(f"Length mismatch: x has {len(x)} points, values has {len(values)} points")
                if len(values) <= 50:
                    ax.plot(x, values, linewidth=self.THICK_LINEWIDTH, color="#2E86AB", alpha=self.LINE_ALPHA_THICK)
                else:
                    ax.plot(x, values, linewidth=self.THIN_LINEWIDTH, color="#2E86AB", alpha=self.LINE_ALPHA_THIN)
                ax.set_xlabel(xlabel, fontsize=12)
        else:
            # No timestamps - use index
            x = np.arange(len(values))
            if len(x) != len(values):
                raise ValueError(f"Length mismatch: x has {len(x)} points, values has {len(values)} points")
            if len(values) <= self.THICK_LINEWIDTH_THRESHOLD:
                ax.plot(x, values, linewidth=self.THICK_LINEWIDTH, color="#2E86AB", alpha=self.LINE_ALPHA_THICK)
            else:
                ax.plot(x, values, linewidth=self.THIN_LINEWIDTH, color="#2E86AB", alpha=self.LINE_ALPHA_THIN)
            ax.set_xlabel("Time Index", fontsize=12)

        # Highlight region if specified
        if highlight_region:
            start, end = highlight_region
            if timestamps is not None and len(timestamps) > 0:
                ax.axvspan(x[start], x[end], alpha=self.HIGHLIGHT_ALPHA, color="yellow", label="Prediction Period")
            else:
                ax.axvspan(start, end, alpha=self.HIGHLIGHT_ALPHA, color="yellow", label="Prediction Period")
            ax.legend()

        # Add statistics as subtitle
        stats_text = self._format_statistics_text(values, handle_nan=False)
        ax.set_title(f"{title}\n{stats_text}", fontsize=14, fontweight="bold", pad=15)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

        # Plot indicator subplot if provided
        if ax_indicator is not None and indicator_arr is not None:
            # Use same x-axis as main chart, but handle length mismatch
            indicator_len = len(indicator_arr)
            if indicator_len == len(x):
                x_indicator = x
            else:
                # Indicator might have different length - use indices
                x_indicator = np.arange(indicator_len)

            # Plot indicator line
            ax_indicator.plot(
                x_indicator, indicator_arr, linewidth=1.5, color="#F18F01", label=indicator_label
            )

            # Zero reference line (useful for MACD-like indicators)
            ax_indicator.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, alpha=0.7)

            # Styling
            ax_indicator.set_ylabel(indicator_label, fontsize=10)
            ax_indicator.legend(loc="upper left", fontsize=8)
            ax_indicator.grid(True, alpha=self.GRID_ALPHA, linestyle="--", linewidth=0.5)

            # X-axis formatting on bottom subplot (with rotation for readability)
            if time_values is not None:
                self._format_time_ticks(ax_indicator, np.array(time_values))
                ax_indicator.set_xlabel(xlabel, fontsize=12)
            elif timestamps is not None and len(timestamps) > 0:
                parsed_ts = self._parse_timestamps(timestamps)
                if parsed_ts is not None:
                    self._format_datetime_ticks(ax_indicator, parsed_ts, n_values=len(values))
                    ax_indicator.set_xlabel("Time", fontsize=12)
                else:
                    # Rotate even for index labels if there are many
                    plt.setp(ax_indicator.xaxis.get_majorticklabels(), rotation=45, ha="right")
                    ax_indicator.set_xlabel("Time Index", fontsize=12)
            else:
                plt.setp(ax_indicator.xaxis.get_majorticklabels(), rotation=45, ha="right")
                ax_indicator.set_xlabel("Time Index", fontsize=12)

            # Remove x-label from main chart (shared x-axis)
            ax.set_xlabel("")

        plt.tight_layout()

        # Finalize figure (save, convert to base64, close)
        return self._finalize_figure(fig, save_path)

    def create_candlestick_chart(
        self,
        open_prices: List[float],
        high_prices: List[float],
        low_prices: List[float],
        close_prices: List[float],
        timestamps: Optional[List[Union[str, datetime]]] = None,
        title: str = "Price Chart",
        save_path: Optional[str] = None,
    ) -> str:
        """
        Create a candlestick chart (finance-style)

        Args:
            open_prices: Opening prices
            high_prices: High prices
            low_prices: Low prices
            close_prices: Closing prices
            timestamps: Optional timestamps
            title: Chart title
            save_path: Optional save path

        Returns:
            Base64 encoded image
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        if timestamps is not None and len(timestamps) > 0:
            x = self._parse_timestamps(timestamps)
        else:
            x = np.arange(len(close_prices))

        # Plot candlesticks
        for i in range(len(close_prices)):
            color = "#26A69A" if close_prices[i] >= open_prices[i] else "#EF5350"

            # Vertical line (high-low)
            ax.plot([x[i], x[i]], [low_prices[i], high_prices[i]], color=color, linewidth=1)

            # Rectangle (open-close)
            height = abs(close_prices[i] - open_prices[i])
            bottom = min(open_prices[i], close_prices[i])
            ax.add_patch(
                plt.Rectangle(
                    (x[i] - 0.3, bottom), 0.6, height, facecolor=color, edgecolor=color, alpha=self.RECTANGLE_ALPHA
                )
            )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel("Price", fontsize=12)
        ax.grid(True, alpha=self.GRID_ALPHA, axis="y")

        if timestamps is not None and len(timestamps) > 0:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            plt.xticks(rotation=45, ha="right")

        plt.tight_layout()

        return self._finalize_figure(fig, save_path)

    def create_multi_series_chart(
        self,
        series_dict: dict,
        timestamps: Optional[List[Union[str, datetime]]] = None,
        title: str = "Multiple Time Series",
        ylabel: str = "Value",
        save_path: Optional[str] = None,
    ) -> str:
        """
        Create chart with multiple time series

        Args:
            series_dict: Dict of {label: values_list}
            timestamps: Optional timestamps
            title: Chart title
            ylabel: Y-axis label
            save_path: Optional save path

        Returns:
            Base64 encoded image
        """
        # Guard against empty series_dict
        if not series_dict:
            raise ValueError("series_dict cannot be empty. Provide at least one time series.")

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        if timestamps is not None and len(timestamps) > 0:
            x = self._parse_timestamps(timestamps)
        else:
            first_series = next(iter(series_dict.values()))
            x = np.arange(len(first_series))

        colors = self.DEFAULT_COLORS[:5]  # Use first 5 colors

        for idx, (label, values) in enumerate(series_dict.items()):
            color = colors[idx % len(colors)]
            ax.plot(x, values, linewidth=2, label=label, color=color)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(loc="best")
        ax.grid(True, alpha=self.GRID_ALPHA)

        if timestamps is not None and len(timestamps) > 0:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            plt.xticks(rotation=45, ha="right")

        plt.tight_layout()

        return self._finalize_figure(fig, save_path)

    def create_heatmap_chart(
        self,
        data: np.ndarray,
        x_labels: Optional[List[str]] = None,
        y_labels: Optional[List[str]] = None,
        title: str = "Time Series Heatmap",
        save_path: Optional[str] = None,
    ) -> str:
        """
        Create a heatmap (useful for multivariate time series)

        Args:
            data: 2D array of values
            x_labels: X-axis labels
            y_labels: Y-axis labels
            title: Chart title
            save_path: Optional save path

        Returns:
            Base64 encoded image
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        im = ax.imshow(data, cmap="RdYlBu_r", aspect="auto")

        if x_labels:
            ax.set_xticks(np.arange(len(x_labels)))
            ax.set_xticklabels(x_labels, rotation=45, ha="right")

        if y_labels:
            ax.set_yticks(np.arange(len(y_labels)))
            ax.set_yticklabels(y_labels)

        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()

        return self._finalize_figure(fig, save_path)

    def create_prediction_chart(
        self,
        historical_values: Union[List[float], List[List[float]]],
        predicted_values: Optional[Union[List[float], List[List[float]]]] = None,
        timestamps: Optional[List[Union[str, datetime]]] = None,
        output_timestamps: Optional[List[Union[str, datetime]]] = None,
        title: str = "Forecasting Task",
        prediction_horizon: int = 0,
        save_path: Optional[str] = None,
        sample_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create chart for forecasting tasks showing ONLY historical data.
        Handles BOTH univariate and multivariate (subplots per channel).

        Args:
            historical_values: Historical data (the input)
                - Univariate: List[float]
                - Multivariate: List[List[float]] (n_channels x n_timesteps)
            predicted_values: Model's predictions (optional, for post-hoc visualization only)
            timestamps: Optional timestamps
            output_timestamps: Optional timestamps for prediction period
            title: Chart title
            prediction_horizon: Number of steps to predict (shown as shaded region)
            save_path: Optional save path
            sample_metadata: Optional dict with dataset_name, domain, source for time conversion

        Returns:
            Base64 encoded image
        """
        # Detect multivariate data
        is_multivariate = self._is_multivariate(historical_values)

        if is_multivariate:
            return self._create_multivariate_prediction_chart(
                historical_values,
                predicted_values,
                timestamps,
                output_timestamps,
                title,
                prediction_horizon,
                save_path,
                sample_metadata,
            )

        # Univariate case - normalize values
        historical_values = self._normalize_univariate_values(historical_values)

        # Generate timestamps if None (for TimerBed sample rate conversion)
        if timestamps is None:
            timestamps = list(range(len(historical_values)))

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        self._plot_prediction_subplot(
            ax,
            historical_values,
            predicted_values,
            timestamps,
            output_timestamps,
            title,
            prediction_horizon,
            color="#2E86AB",
            sample_metadata=sample_metadata,
        )

        plt.tight_layout()

        return self._finalize_figure(fig, save_path)

    def _create_multivariate_prediction_chart(
        self,
        historical_values: List[List[float]],
        predicted_values: Optional[List[List[float]]],
        timestamps: Optional[List],
        output_timestamps: Optional[List],
        title: str,
        prediction_horizon: int,
        save_path: Optional[str],
        sample_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create multivariate prediction chart with subplots per channel."""
        n_channels = len(historical_values)
        colors = self.DEFAULT_COLORS

        # Generate timestamps if None (for TimerBed sample rate conversion)
        if timestamps is None:
            timestamps = list(range(len(historical_values[0])))  # Use first channel length

        fig, axes = self._create_multivariate_subplots(n_channels, sharex=True, sharey=False)

        for i, channel_data in enumerate(historical_values):
            channel_pred = predicted_values[i] if predicted_values and i < len(predicted_values) else None
            channel_title = f"Channel {i + 1}"
            self._plot_prediction_subplot(
                axes[i],
                channel_data,
                channel_pred,
                timestamps,
                output_timestamps,
                channel_title,
                prediction_horizon,
                colors[i % len(colors)],
                sample_metadata=sample_metadata,
            )

        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)
        plt.tight_layout()

        return self._finalize_figure(fig, save_path)

    def _plot_prediction_subplot(
        self,
        ax,
        historical_values: List[float],
        predicted_values: Optional[List[float]],
        timestamps: Optional[List],
        output_timestamps: Optional[List],
        title: str,
        prediction_horizon: int,
        color: str,
        sample_metadata: Optional[Dict[str, Any]] = None,
    ):
        """Plot a single prediction chart on the given axes."""
        hist_len = len(historical_values)

        # Generate timestamps if None (for TimerBed sample rate conversion)
        if timestamps is None:
            timestamps = list(range(hist_len))

        # Convert indices to time if applicable (TimerBed with known sample rates)
        time_values, xlabel = self._convert_indices_to_time(
            timestamps[:hist_len], sample_metadata, data_length=hist_len
        )

        # Prepare x-axis
        if time_values is not None:
            # Use converted time values (in seconds)
            x_hist = np.array(time_values)
            use_time = True
        else:
            # Use integer indices for plotting
            x_hist = np.arange(hist_len)
            use_time = False

        # Validate length before plotting
        if len(x_hist) != len(historical_values):
            raise ValueError(
                f"Length mismatch: x_hist has {len(x_hist)} points, historical_values has {len(historical_values)} points"
            )

        x_start_idx = hist_len - 0.5
        x_end_idx = hist_len + prediction_horizon - 0.5

        # Plot historical data
        ax.plot(x_hist, historical_values, linewidth=2, color=color, label="Historical")

        # Optionally plot model predictions (NOT ground truth!)
        if predicted_values and len(predicted_values) > 0:
            pred_len = len(predicted_values)
            x_pred = np.arange(hist_len, hist_len + pred_len)
            ax.plot(
                x_pred,
                predicted_values,
                linewidth=2,
                color="#F18F01",
                label="Predicted",
                marker="s",
                markersize=4,
                linestyle="--",
            )

        # Show prediction horizon as shaded region (without values!)
        if prediction_horizon > 0 and not predicted_values:
            ax.axvspan(
                x_start_idx,
                x_end_idx,
                alpha=self.PREDICTION_ALPHA,
                color="#F18F01",
                label=f"Predict {prediction_horizon} steps",
            )

        # Vertical line separating historical and prediction
        if use_time:
            ax.axvline(x=x_hist[-1], color="gray", linestyle=":", alpha=self.VERTICAL_LINE_ALPHA)
        else:
            ax.axvline(x=hist_len - 0.5, color="gray", linestyle=":", alpha=self.VERTICAL_LINE_ALPHA)

        # Format x-axis with timestamps if available (like line chart)
        if use_time:
            # Time in seconds - format ticks
            self._format_time_ticks(ax, x_hist)
            ax.set_xlabel(xlabel, fontsize=10)
        elif timestamps is not None and len(timestamps) > 0:
            parsed_ts = self._parse_timestamps(timestamps[:hist_len])
            if parsed_ts is not None:
                # Calculate total x-axis range including prediction horizon
                total_len = hist_len + prediction_horizon

                # Extend timestamps for prediction region if output_timestamps available
                if output_timestamps and len(output_timestamps) >= prediction_horizon:
                    pred_ts = self._parse_timestamps(output_timestamps[:prediction_horizon])
                    all_timestamps = parsed_ts + pred_ts
                elif output_timestamps and len(output_timestamps) > 0:
                    # Calculate interval from output_timestamps and extend
                    pred_ts_parsed = self._parse_timestamps(output_timestamps)
                    if len(pred_ts_parsed) > 1:
                        pred_delta = (pred_ts_parsed[-1] - pred_ts_parsed[0]) / (len(pred_ts_parsed) - 1)
                    else:
                        pred_delta = (parsed_ts[-1] - parsed_ts[-2]) if len(parsed_ts) > 1 else timedelta(days=1)
                    all_timestamps = parsed_ts + [
                        parsed_ts[-1] + pred_delta * (i + 1) for i in range(prediction_horizon)
                    ]
                else:
                    # Use historical delta to extend
                    delta = (parsed_ts[-1] - parsed_ts[-2]) if len(parsed_ts) > 1 else timedelta(days=1)
                    all_timestamps = parsed_ts + [parsed_ts[-1] + delta * (i + 1) for i in range(prediction_horizon)]

                # Format datetime ticks
                self._format_datetime_ticks(ax, all_timestamps, n_values=total_len)
                ax.set_xlabel("Time", fontsize=10)
            else:
                ax.set_xlabel("Time Index", fontsize=10)
        else:
            ax.set_xlabel("Time Index", fontsize=10)

        # Add statistics as subtitle
        stats_text = self._format_statistics_text(historical_values, handle_nan=False)
        ax.set_title(f"{title}\n{stats_text}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Value", fontsize=10)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=self.GRID_ALPHA)

    def create_imputation_chart(
        self,
        values: Union[List[float], List[List[float]]],
        timestamps: Optional[List[Union[str, datetime]]] = None,
        title: str = "Time Series (Imputation Task)",
        ylabel: str = "Value",
        save_path: Optional[str] = None,
        sample_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create chart for imputation tasks with missing values (NaN) clearly shown as gaps.
        Handles BOTH univariate and multivariate (subplots per channel).

        Args:
            values: Time series with NaN for missing values (model must impute these)
                - Univariate: List[float]
                - Multivariate: List[List[float]] (n_channels x n_timesteps)
            timestamps: Optional timestamps
            title: Chart title
            ylabel: Y-axis label
            save_path: Optional save path
            sample_metadata: Optional dict with dataset_name, domain, source for time conversion

        Returns:
            Base64 encoded image
        """
        # Detect multivariate data
        is_multivariate = self._is_multivariate(values)

        if is_multivariate:
            return self._create_multivariate_imputation_chart(
                values, timestamps, title, ylabel, save_path, sample_metadata
            )

        # Univariate case - normalize values
        values = self._normalize_univariate_values(values)

        # Generate timestamps if None (for TimerBed sample rate conversion)
        if timestamps is None:
            timestamps = list(range(len(values)))

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        n_missing = self._plot_imputation_subplot(
            ax, values, timestamps, title, ylabel, color="#2E86AB", sample_metadata=sample_metadata
        )

        plt.tight_layout()

        return self._finalize_figure(fig, save_path)

    def _create_multivariate_imputation_chart(
        self,
        values: List[List[float]],
        timestamps: Optional[List],
        title: str,
        ylabel: str,
        save_path: Optional[str],
        sample_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create multivariate imputation chart with subplots per channel."""
        n_channels = len(values)
        colors = self.DEFAULT_COLORS

        # Generate timestamps if None (for TimerBed sample rate conversion)
        if timestamps is None:
            timestamps = list(range(len(values[0])))  # Use first channel length

        fig, axes = self._create_multivariate_subplots(n_channels, sharex=True, sharey=False)

        total_missing = 0
        for i, channel_data in enumerate(values):
            channel_title = f"Channel {i + 1}"
            n_missing = self._plot_imputation_subplot(
                axes[i],
                channel_data,
                timestamps,
                channel_title,
                ylabel,
                colors[i % len(colors)],
                sample_metadata=sample_metadata,
            )
            total_missing += n_missing

        fig.suptitle(f"{title} (Total: {total_missing} missing values)", fontsize=14, fontweight="bold", y=0.995)
        plt.tight_layout()

        return self._finalize_figure(fig, save_path)

    def _plot_imputation_subplot(
        self,
        ax,
        values: List[float],
        timestamps: Optional[List],
        title: str,
        ylabel: str,
        color: str,
        sample_metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Plot a single imputation chart on the given axes. Returns number of missing values."""
        values_arr = np.array(values, dtype=float)
        n = len(values_arr)

        # Generate timestamps if None (for TimerBed sample rate conversion)
        if timestamps is None:
            timestamps = list(range(n))

        # Convert indices to time if applicable (TimerBed with known sample rates)
        time_values, xlabel = self._convert_indices_to_time(timestamps, sample_metadata, data_length=n)

        # Prepare x-axis
        if time_values is not None:
            # Use converted time values (in seconds)
            x = np.array(time_values)
            use_time = True
        else:
            # Use integer indices
            x = np.arange(n)
            use_time = False

        # Validate length before plotting
        if len(x) != len(values_arr):
            raise ValueError(f"Length mismatch: x has {len(x)} points, values_arr has {len(values_arr)} points")

        # Find missing value indices
        missing_mask = np.isnan(values_arr)
        n_missing = int(np.sum(missing_mask))

        # Plot existing values (NaN will create gaps automatically)
        ax.plot(x, values_arr, linewidth=2, color=color, label="Known Values")

        # Highlight missing regions with vertical spans
        if n_missing > 0:
            missing_idx = np.where(missing_mask)[0]
            if len(missing_idx) > 0:
                # Group into contiguous regions
                regions = []
                start = missing_idx[0]
                for i in range(1, len(missing_idx)):
                    if missing_idx[i] != missing_idx[i - 1] + 1:
                        regions.append((start, missing_idx[i - 1]))
                        start = missing_idx[i]
                regions.append((start, missing_idx[-1]))

                # Shade missing regions
                for i, (region_start, region_end) in enumerate(regions):
                    label = f"Missing ({n_missing})" if i == 0 else None
                    if use_time:
                        ax.axvspan(
                            x[region_start], x[region_end], alpha=self.HIGHLIGHT_ALPHA, color="#E74C3C", label=label
                        )
                    else:
                        ax.axvspan(
                            region_start - 0.5,
                            region_end + 0.5,
                            alpha=self.HIGHLIGHT_ALPHA,
                            color="#E74C3C",
                            label=label,
                        )

        # Format x-axis
        if use_time:
            # Time in seconds - format ticks
            self._format_time_ticks(ax, x)
            ax.set_xlabel(xlabel, fontsize=10)
        elif timestamps is not None and len(timestamps) > 0:
            parsed_ts = self._parse_timestamps(timestamps)
            if parsed_ts is not None:
                self._format_datetime_ticks(ax, parsed_ts, n_values=n)
                ax.set_xlabel("Time", fontsize=10)
            else:
                ax.set_xlabel("Time Index", fontsize=10)
        else:
            ax.set_xlabel("Time Index", fontsize=10)

        # Add statistics as subtitle
        stats_text = self._format_statistics_text(values_arr, handle_nan=True)
        ax.set_title(f"{title}\n{stats_text}", fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

        return n_missing

    def _convert_indices_to_time(
        self,
        timestamps: Optional[List[Union[int, float, str, datetime]]],
        sample_metadata: Optional[Dict[str, Any]] = None,
        data_length: Optional[int] = None,
    ) -> Tuple[Optional[List[float]], str]:
        """
        Convert integer indices to time values using known sample rates.

        Args:
            timestamps: List of timestamps (can be None, indices, datetime objects, or strings)
            sample_metadata: Optional dict with dataset_name, domain, source, etc.
            data_length: Optional data length (used to generate sequential indices if timestamps is None)

        Returns:
            Tuple of (time_values, xlabel):
            - time_values: List of time in seconds, or None if conversion not applicable
            - xlabel: "Time (s)" if converted, "Time Index" if indices, or "Time" if datetime
        """
        # Generate sequential indices if timestamps is None but we have sample_metadata
        if timestamps is None or len(timestamps) == 0:
            if data_length is not None and sample_metadata:
                # Try to generate sequential indices for TimerBed sample rate conversion
                dataset_name = sample_metadata.get("dataset_name")
                if dataset_name and dataset_name in KNOWN_SAMPLE_RATES:
                    rate = KNOWN_SAMPLE_RATES[dataset_name]
                    if isinstance(rate, dict):
                        rate = rate.get("default", 1.0)
                    if isinstance(rate, (int, float)) and rate > 0:
                        timestamps = list(range(data_length))
                    else:
                        return None, "Time Index"
                else:
                    return None, "Time Index"
            else:
                return None, "Time Index"

        # Check if timestamps are sequential integers starting from 0
        # Use tolerance for float comparison to avoid precision issues
        def is_close_to_int(ts, i):
            if isinstance(ts, int):
                return ts == i
            if isinstance(ts, float):
                return abs(ts - i) < 1e-9
            return False

        is_sequential_indices = all(is_close_to_int(ts, i) for i, ts in enumerate(timestamps))

        if not is_sequential_indices:
            # Not sequential indices - check if datetime objects or parseable strings
            if isinstance(timestamps[0], datetime) or (
                isinstance(timestamps[0], str) and any(c in str(timestamps[0]) for c in ["-", "/", ":"])
            ):
                return None, "Time"  # Will use datetime parsing
            return None, "Time Index"  # Unknown format, use indices

        # Sequential indices detected - check if we have known sample rate
        if sample_metadata:
            dataset_name = sample_metadata.get("dataset_name")
            domain = sample_metadata.get("domain")
            source = sample_metadata.get("source")

            # Priority 1: TimerBed known dataset rates
            if dataset_name and dataset_name in KNOWN_SAMPLE_RATES:
                rate = KNOWN_SAMPLE_RATES[dataset_name]
                if isinstance(rate, dict):
                    rate = rate.get("default", 1.0)
                if isinstance(rate, (int, float)) and rate > 0:
                    # Convert indices to time in seconds
                    time_values = [float(i) / rate for i in timestamps]
                    return time_values, "Time (s)"

            # Priority 2: MTBench domain-based lookup
            if domain and domain in KNOWN_SAMPLE_RATES:
                domain_rates = KNOWN_SAMPLE_RATES[domain]
                if isinstance(domain_rates, dict):
                    rate = None
                    if source:
                        rate = domain_rates.get(source)
                    if not rate:
                        rate = domain_rates.get("default", 1.0)
                    if isinstance(rate, (int, float)) and rate > 0:
                        time_values = [float(i) / rate for i in timestamps]
                        return time_values, "Time (s)"

        # No known sample rate - use indices
        return None, "Time Index"

    def _parse_timestamps(self, timestamps: List[Union[str, datetime, int, float]]) -> Optional[List[datetime]]:
        """
        Convert timestamps to datetime objects.
        Returns None if timestamps are sequential integer indices (should not be parsed as Unix timestamps).
        """
        if not timestamps or len(timestamps) == 0:
            return None

        # Check if timestamps are sequential integers starting from 0
        # If so, don't parse as Unix timestamps (prevents "1970-01-01" bug)
        # Use tolerance for float comparison to avoid precision issues
        def is_close_to_int(ts, i):
            if isinstance(ts, int):
                return ts == i
            if isinstance(ts, float):
                return abs(ts - i) < 1e-9
            return False

        is_sequential_indices = all(is_close_to_int(ts, i) for i, ts in enumerate(timestamps))
        if is_sequential_indices:
            return None  # Don't parse as Unix timestamps

        result = []
        for ts in timestamps:
            if isinstance(ts, datetime):
                # Already a datetime object
                result.append(ts)
            elif isinstance(ts, (int, float)):
                # Unix timestamp (seconds since epoch) - but only if not sequential indices
                result.append(datetime.fromtimestamp(ts))
            elif isinstance(ts, str):
                # Try multiple formats in order - fail fast if none match
                def try_parse(s: str, fmt: str) -> Optional[datetime]:
                    """Try parsing with format, return None if format doesn't match."""
                    try:
                        return datetime.strptime(s, fmt)
                    except ValueError:
                        return None

                parsed_dt = None
                # Try DATETIME_FORMATS plus ISO format
                formats_to_try = DATETIME_FORMATS + ["%Y-%m-%dT%H:%M:%S"]
                for fmt in formats_to_try:
                    parsed_dt = try_parse(ts, fmt)
                    if parsed_dt is not None:
                        break

                if parsed_dt is None:
                    # Try ISO format - fail fast if this also fails
                    try:
                        parsed_dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    except ValueError:
                        raise ValueError(
                            f"Unable to parse timestamp string: {ts}. Supported formats: '%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%Y-%m-%dT%H:%M:%S', or ISO format."
                        )

                result.append(parsed_dt)
            else:
                raise TypeError(f"Unsupported timestamp type: {type(ts)}. Expected datetime, int, float, or str.")
        return result

    def create_multivariate_chart(
        self,
        multivariate_data: Union[List[List[float]], np.ndarray],
        variable_names: Optional[List[str]] = None,
        timestamps: Optional[List[Union[str, datetime]]] = None,
        title: str = "Multi-variate Time Series",
        ylabel: str = "Value",
        save_path: Optional[str] = None,
        share_x: bool = True,
        share_y: bool = False,
        sample_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create TIME DOMAIN visualization with ONE SUBPLOT PER VARIABLE

        Avoids overlapping lines by giving each variable its own subplot.
        Perfect for multi-channel sensor data or multiple stocks.

        Args:
            multivariate_data: Multi-variate data (n_variables x n_timepoints)
            variable_names: Names for each variable
            timestamps: Optional timestamps for x-axis
            title: Overall figure title
            ylabel: Y-axis label (common for all subplots)
            save_path: Optional path to save image
            share_x: Whether to share x-axis across subplots
            share_y: Whether to share y-axis across subplots

        Returns:
            Base64 encoded image string
        """
        # Convert to numpy array
        data = np.array(multivariate_data)

        # Handle 1D data
        if len(data.shape) == 1:
            data = data.reshape(1, -1)

        n_variables = data.shape[0]
        n_timepoints = data.shape[1]

        # Generate timestamps if None (for TimerBed sample rate conversion)
        if timestamps is None:
            timestamps = list(range(n_timepoints))

        # Generate variable names if not provided
        if variable_names is None:
            variable_names = [f"Variable {i + 1}" for i in range(n_variables)]

        # Convert indices to time if applicable (TimerBed with known sample rates)
        time_values, xlabel = self._convert_indices_to_time(timestamps, sample_metadata, data_length=n_timepoints)

        # Prepare x-axis
        if time_values is not None:
            # Use converted time values (in seconds)
            x = np.array(time_values)
            if len(x) != n_timepoints:
                raise ValueError(f"Length mismatch: x has {len(x)} points, but data has {n_timepoints} timepoints")
            use_dates = False
            use_time = True
        elif timestamps is not None and len(timestamps) > 0:
            parsed_ts = self._parse_timestamps(timestamps)
            if parsed_ts is not None:
                x = parsed_ts
                if len(x) != n_timepoints:
                    raise ValueError(f"Length mismatch: x has {len(x)} points, but data has {n_timepoints} timepoints")
                use_dates = True
                use_time = False
            else:
                # Sequential indices
                x = np.arange(n_timepoints)
                use_dates = False
                use_time = False
        else:
            x = np.arange(n_timepoints)
            use_dates = False
            use_time = False

        # Create subplots - ONE PER VARIABLE
        fig, axes = self._create_multivariate_subplots(n_variables, sharex=share_x, sharey=share_y)

        # Define colors for variety
        colors = self.DEFAULT_COLORS

        # Plot each variable in its own subplot
        for i, (var_data, var_name) in enumerate(zip(data, variable_names)):
            color = colors[i % len(colors)]

            # Validate length before plotting
            if len(x) != len(var_data):
                raise ValueError(
                    f"Length mismatch for channel {i}: x has {len(x)} points, var_data has {len(var_data)} points"
                )

            if use_dates:
                axes[i].plot(x, var_data, linewidth=2, color=color)
                axes[i].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
                if i == n_variables - 1:  # Only rotate labels on bottom plot
                    plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=45, ha="right")
            elif use_time:
                # Time in seconds - format ticks
                axes[i].plot(x, var_data, linewidth=2, color=color)
                if i == n_variables - 1:  # Only format ticks on bottom plot
                    self._format_time_ticks(axes[i], np.array(x))
            else:
                axes[i].plot(x, var_data, linewidth=2, color=color)

            # Styling for each subplot
            axes[i].set_ylabel(ylabel, fontsize=10)
            # Add statistics as subtitle
            stats_text = self._format_statistics_text(var_data, handle_nan=False)
            axes[i].set_title(f"{var_name}\n{stats_text}", fontsize=11, fontweight="bold")
            axes[i].grid(True, alpha=self.GRID_ALPHA)

        # Set x-label only on bottom subplot
        if use_dates:
            axes[-1].set_xlabel("Time", fontsize=12)
        elif use_time:
            axes[-1].set_xlabel(xlabel, fontsize=12)
        else:
            axes[-1].set_xlabel(xlabel, fontsize=12)

        # Overall title
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.995)

        plt.tight_layout()

        # Finalize figure (save, convert to base64, close)
        return self._finalize_figure(fig, save_path)

    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string"""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=self.dpi)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        return img_base64

    def save_chart_to_file(self, img_base64: str, filepath: str):
        """Save base64 encoded image to file"""
        img_data = base64.b64decode(img_base64)
        with open(filepath, "wb") as f:
            f.write(img_data)


# Convenience functions
def quick_line_chart(values: List[float], save_path: Optional[str] = None) -> str:
    """Quick line chart generation"""
    gen = TimeSeriesChartGenerator()
    return gen.create_line_chart(values, save_path=save_path)


def quick_prediction_chart(
    historical: List[float],
    predicted: Optional[List[float]] = None,
    prediction_horizon: int = 0,
    save_path: Optional[str] = None,
) -> str:
    """
    Quick prediction chart generation.
    """
    gen = TimeSeriesChartGenerator()
    return gen.create_prediction_chart(
        historical, predicted_values=predicted, prediction_horizon=prediction_horizon, save_path=save_path
    )


def create_task_aware_chart(
    values: Union[List[float], List[List[float]]],
    task_type: str,
    timestamps: Optional[List] = None,
    output_timestamps: Optional[List] = None,
    prediction_length: int = 0,
    title: str = "Time Series",
    sample_metadata: Optional[Dict[str, Any]] = None,
    indicator_values: Optional[List[float]] = None,
    indicator_label: str = "Indicator",
) -> str:
    """
    Create appropriate chart based on task type. Handles BOTH univariate and multivariate time series.

    For multivariate data, creates SUBPLOTS with the SAME task-specific visualization
    (e.g., prediction horizon shading, missing value highlighting) per channel.

    Args:
        values: Time series values (historical input only!)
            - Univariate: List[float]
            - Multivariate: List[List[float]] (n_channels x n_timesteps)
        task_type: One of 'forecasting', 'imputation', 'trend', 'classification', etc.
        timestamps: Optional timestamps
        prediction_length: For forecasting tasks, how many steps to predict
        title: Chart title
        sample_metadata: Optional dict with dataset_name, domain, source for time conversion
        indicator_values: Optional pre-computed indicator values (e.g., MACD, BB) to show as subplot
        indicator_label: Label for the indicator subplot y-axis (e.g., "MACD", "Bollinger Band")

    Returns:
        Base64 encoded chart image
    """
    gen = TimeSeriesChartGenerator()
    task_lower = task_type.lower() if task_type else ""

    # Forecasting: show historical + prediction region
    if "forecast" in task_lower or "predict" in task_lower:
        return gen.create_prediction_chart(
            historical_values=values,
            predicted_values=None,
            timestamps=timestamps,
            output_timestamps=output_timestamps,
            title=title,
            prediction_horizon=prediction_length,
            sample_metadata=sample_metadata,
        )

    # Imputation: show data with gaps marked
    if "imputation" in task_lower or "impute" in task_lower:
        return gen.create_imputation_chart(
            values=values, timestamps=timestamps, title=title, sample_metadata=sample_metadata
        )

    # Default: line chart for trend, classification, anomaly, indicator, qa, etc.
    return gen.create_line_chart(
        values=values,
        timestamps=timestamps,
        title=title,
        sample_metadata=sample_metadata,
        indicator_values=indicator_values,
        indicator_label=indicator_label,
    )
