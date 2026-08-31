"""
Numerical Lookup Function for Cross-Modal Debate

Instead of sending ALL time series values to numerical agents (which can overflow
the context window), this provides:
1. **Summary View**: High-level statistics that always fit in context
2. **Key Indices**: Pre-computed significant points (peaks, troughs, anomalies, change points)
3. **Frequency Features**: Pre-computed frequency domain analysis (spectral centroid, dominant frequencies, etc.)
4. **Windowed Lookup**: On-demand access to specific index/timestamp ranges
5. **Cross-Modal Queries**: Allow other modalities to suggest focus regions

This enables numerical agents to "zoom in" on important regions rather than
getting overwhelmed by or losing track of thousands of values, while also providing
frequency-domain insights for periodic pattern analysis.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np

# Import scipy for signal processing
from scipy import signal

# Import FrequencyAnalyzer for frequency domain features
from .frequency_analyzer import FrequencyAnalyzer
from .sample_rate_inference import get_frequency_interpretation, infer_sample_rate


@dataclass
class LookupResult:
    """Result from a lookup query"""

    indices: List[int]
    values: List[float]
    timestamps: List[Any]
    context: str  # Human-readable description
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KeyPoint:
    """A significant point in the time series"""

    index: int
    value: float
    timestamp: Any
    point_type: str  # "peak", "trough", "anomaly", "change_point", "start", "end"
    significance: float  # 0-1, how important this point is
    metadata: Dict[str, Any] = field(default_factory=dict)


class NumericalLookupFunction:
    """
    Smart lookup function for numerical agents in cross-modal debate.

    Instead of showing ALL time series values (which can exceed context limits),
    this class provides:

    1. **Summary Statistics**: Always-available high-level view
    2. **Key Points**: Pre-computed significant indices (peaks, anomalies, etc.)
    3. **Windowed Access**: On-demand lookup of specific ranges
    4. **Cross-Modal Hints**: Accept focus suggestions from other modalities

    Architecture Role:
    - Numerical agents use this to intelligently access data
    - Prevents context overflow while maintaining analytical capability
    """

    # Maximum values to show in a single lookup (prevents context overflow)
    MAX_VALUES_PER_LOOKUP = 100
    # Default window size for point-focused lookups
    DEFAULT_WINDOW_SIZE = 10

    # Constants for trend detection
    TREND_SLOPE_THRESHOLD_POSITIVE = 0.01
    TREND_SLOPE_THRESHOLD_NEGATIVE = -0.01

    # Constants for peak detection
    PEAK_DETECTION_MIN_DISTANCE_BASE = 5
    PEAK_DETECTION_DISTANCE_DIVISOR = 50
    PEAK_PROMINENCE_RATIO = 0.1  # 10% of data range

    def __init__(
        self,
        values: Union[List[float], List[List[float]], np.ndarray],
        timestamps: Optional[List[Any]] = None,
        max_values_per_lookup: int = 100,
        sample_metadata: Optional[Dict[str, Any]] = None,
        indicator_values: Optional[List[float]] = None,
        indicator_label: str = "Indicator",
    ):
        """
        Initialize the lookup function with time series data.

        Args:
            values: Time series values - can be:
                - Univariate: List[float] or 1D array
                - Multivariate: List[List[float]] or 2D array (n_channels, n_timesteps)
            timestamps: Optional timestamps (defaults to indices 0, 1, 2, ...)
            max_values_per_lookup: Maximum values to return in single lookup
            sample_metadata: Optional metadata dict with dataset_name, domain, source, etc.
                           Used for sample rate inference in frequency analysis
            indicator_values: Optional pre-computed indicator values (e.g., MACD, BB)
            indicator_label: Label for the indicator (e.g., "MACD", "Bollinger Band")
        """
        raw_values = np.array(values, dtype=np.float64)

        # Handle multivariate data (shape: n_channels x n_timesteps)
        if raw_values.ndim == 2:
            self.is_multivariate = True
            self.n_channels = raw_values.shape[0]
            self.raw_channels = raw_values  # Keep original multivariate data
            # Use L2 norm across channels as summary for statistics/key points
            self.values = np.linalg.norm(raw_values, axis=0)
            self.n_points = raw_values.shape[1]
        else:
            self.is_multivariate = False
            self.n_channels = 1
            self.raw_channels = raw_values.reshape(1, -1)
            self.values = raw_values.flatten()
            self.n_points = len(self.values)

        self.timestamps = timestamps if timestamps is not None else list(range(self.n_points))
        self.max_values_per_lookup = max_values_per_lookup

        # Store sample metadata for sample rate inference
        self.sample_metadata = sample_metadata or {}
        self.dataset_name = self.sample_metadata.get("dataset_name")
        self.domain = self.sample_metadata.get("domain")
        self.source = self.sample_metadata.get("source")

        # Store indicator data if provided
        self.indicator_label = indicator_label
        if indicator_values is not None and len(indicator_values) > 0:
            self.indicator_values = np.array(indicator_values, dtype=np.float64)
            self.has_indicator = True
            self._indicator_stats = {
                "length": len(self.indicator_values),
                "min": float(np.min(self.indicator_values)),
                "max": float(np.max(self.indicator_values)),
                "mean": float(np.mean(self.indicator_values)),
                "std": float(np.std(self.indicator_values)),
            }
        else:
            self.indicator_values = None
            self.has_indicator = False
            self._indicator_stats = {}

        # Pre-compute key points and statistics
        self._statistics: Dict[str, Any] = {}
        self._key_points: List[KeyPoint] = []
        self._segments: List[Dict[str, Any]] = []
        self._frequency_features: Dict[str, Any] = {}

        self._compute_statistics()
        self._detect_key_points()
        self._segment_series()
        self._compute_frequency_features()

    def _compute_statistics(self):
        """Compute summary statistics (always fits in context)"""
        # Basic statistics on summary values (L2 norm for multivariate)
        self._statistics = {
            "length": self.n_points,
            "n_channels": self.n_channels,
            "is_multivariate": self.is_multivariate,
            "min": float(np.min(self.values)),
            "max": float(np.max(self.values)),
            "mean": float(np.mean(self.values)),
            "std": float(np.std(self.values)),
            "median": float(np.median(self.values)),
            "first_value": float(self.values[0]),
            "last_value": float(self.values[-1]),
            "total_change": float(self.values[-1] - self.values[0]),
            "pct_change": (
                float((self.values[-1] - self.values[0]) / self.values[0] * 100) if self.values[0] != 0 else 0.0
            ),
        }

        # Compute quartiles
        self._statistics["q1"] = float(np.percentile(self.values, 25))
        self._statistics["q3"] = float(np.percentile(self.values, 75))
        self._statistics["iqr"] = self._statistics["q3"] - self._statistics["q1"]

        # Trend direction (simple linear regression slope)
        if self.n_points > 1:
            x = np.arange(self.n_points)
            slope = np.polyfit(x, self.values, 1)[0]
            self._statistics["trend_slope"] = float(slope)
            self._statistics["trend_direction"] = (
                "UPWARD"
                if slope > self.TREND_SLOPE_THRESHOLD_POSITIVE
                else ("DOWNWARD" if slope < self.TREND_SLOPE_THRESHOLD_NEGATIVE else "STABLE")
            )
        else:
            self._statistics["trend_slope"] = 0.0
            self._statistics["trend_direction"] = "STABLE"

        # Per-channel statistics for multivariate data
        if self.is_multivariate:
            self._statistics["channel_stats"] = []
            for ch_idx in range(self.n_channels):
                ch_data = self.raw_channels[ch_idx]
                ch_stats = {
                    "channel": ch_idx,
                    "min": float(np.min(ch_data)),
                    "max": float(np.max(ch_data)),
                    "mean": float(np.mean(ch_data)),
                    "std": float(np.std(ch_data)),
                }
                self._statistics["channel_stats"].append(ch_stats)

    def _detect_key_points(self):
        """Detect significant points in the time series"""
        self._key_points = []

        # Always include start and end points
        self._key_points.append(
            KeyPoint(
                index=0,
                value=float(self.values[0]),
                timestamp=self.timestamps[0],
                point_type="start",
                significance=1.0,
            )
        )
        self._key_points.append(
            KeyPoint(
                index=self.n_points - 1,
                value=float(self.values[-1]),
                timestamp=self.timestamps[-1],
                point_type="end",
                significance=1.0,
            )
        )

        if self.n_points < 5:
            return

        # Detect peaks and troughs
        self._detect_peaks_troughs()

        # Detect anomalies (outliers)
        self._detect_anomalies()

        # Detect change points (trend shifts)
        self._detect_change_points()

        # Sort by significance
        self._key_points.sort(key=lambda p: p.significance, reverse=True)

    def _detect_peaks_troughs(self):
        """Detect local peaks and troughs"""
        if self.n_points < 10:
            # For very short series, find global max/min
            max_idx = int(np.argmax(self.values))
            min_idx = int(np.argmin(self.values))

            if max_idx not in [0, self.n_points - 1]:
                self._key_points.append(
                    KeyPoint(
                        index=max_idx,
                        value=float(self.values[max_idx]),
                        timestamp=self.timestamps[max_idx],
                        point_type="peak",
                        significance=0.9,
                        metadata={"type": "global_max"},
                    )
                )
            if min_idx not in [0, self.n_points - 1]:
                self._key_points.append(
                    KeyPoint(
                        index=min_idx,
                        value=float(self.values[min_idx]),
                        timestamp=self.timestamps[min_idx],
                        point_type="trough",
                        significance=0.9,
                        metadata={"type": "global_min"},
                    )
                )
            return

        # Use scipy for peak detection
        # Adaptive distance based on series length
        min_distance = max(
            self.PEAK_DETECTION_MIN_DISTANCE_BASE, self.n_points // self.PEAK_DETECTION_DISTANCE_DIVISOR
        )

        # Calculate prominence threshold based on data range
        data_range = np.max(self.values) - np.min(self.values)
        prominence_threshold = data_range * self.PEAK_PROMINENCE_RATIO

        # Detect peaks
        peaks, peak_props = signal.find_peaks(self.values, distance=min_distance, prominence=prominence_threshold)

        # Detect troughs (peaks in negative signal)
        troughs, trough_props = signal.find_peaks(-self.values, distance=min_distance, prominence=prominence_threshold)

        # Add top peaks (limit to avoid overwhelming)
        max_peaks = min(10, len(peaks))
        if len(peaks) > 0 and "prominences" in peak_props:
            # Sort by prominence
            peak_order = np.argsort(peak_props["prominences"])[::-1]
            for i in range(min(max_peaks, len(peak_order))):
                idx = int(peaks[peak_order[i]])
                prominence = float(peak_props["prominences"][peak_order[i]])
                self._key_points.append(
                    KeyPoint(
                        index=idx,
                        value=float(self.values[idx]),
                        timestamp=self.timestamps[idx],
                        point_type="peak",
                        significance=min(1.0, prominence / data_range),
                        metadata={"prominence": prominence},
                    )
                )

        # Add top troughs
        max_troughs = min(10, len(troughs))
        if len(troughs) > 0 and "prominences" in trough_props:
            trough_order = np.argsort(trough_props["prominences"])[::-1]
            for i in range(min(max_troughs, len(trough_order))):
                idx = int(troughs[trough_order[i]])
                prominence = float(trough_props["prominences"][trough_order[i]])
                self._key_points.append(
                    KeyPoint(
                        index=idx,
                        value=float(self.values[idx]),
                        timestamp=self.timestamps[idx],
                        point_type="trough",
                        significance=min(1.0, prominence / data_range),
                        metadata={"prominence": prominence},
                    )
                )

    def _detect_anomalies(self):
        """Detect anomalous points (outliers)"""
        # Use IQR method for robustness
        q1, q3 = np.percentile(self.values, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        anomaly_mask = (self.values < lower_bound) | (self.values > upper_bound)
        anomaly_indices = np.where(anomaly_mask)[0]

        # Process all anomalies sorted by deviation
        if len(anomaly_indices) > 0:
            # Sort by deviation from median
            median = np.median(self.values)
            deviations = np.abs(self.values[anomaly_indices] - median)
            sorted_anomalies = anomaly_indices[np.argsort(deviations)[::-1]]

            for idx in sorted_anomalies:
                idx = int(idx)
                deviation = float(np.abs(self.values[idx] - median))
                self._key_points.append(
                    KeyPoint(
                        index=idx,
                        value=float(self.values[idx]),
                        timestamp=self.timestamps[idx],
                        point_type="anomaly",
                        significance=min(1.0, deviation / (iqr + 1e-10)),
                        metadata={"deviation_from_median": deviation},
                    )
                )

    def _detect_change_points(self):
        """Detect trend change points"""
        if self.n_points < 20:
            return

        # Simple approach: detect significant slope changes
        window_size = max(5, self.n_points // 20)

        # Compute rolling slope
        slopes = []
        for i in range(window_size, self.n_points - window_size):
            left_slope = np.polyfit(range(window_size), self.values[i - window_size : i], 1)[0]
            right_slope = np.polyfit(range(window_size), self.values[i : i + window_size], 1)[0]
            slope_change = abs(right_slope - left_slope)
            slopes.append((i, slope_change, left_slope, right_slope))

        if not slopes:
            return

        # Find top change points
        slopes.sort(key=lambda x: x[1], reverse=True)
        max_change_points = min(5, len(slopes))

        for i in range(max_change_points):
            idx, change, left_slope, right_slope = slopes[i]
            idx = int(idx)

            # Determine change type
            if left_slope > 0 and right_slope < 0:
                change_type = "peak_reversal"
            elif left_slope < 0 and right_slope > 0:
                change_type = "trough_reversal"
            elif abs(right_slope) > abs(left_slope):
                change_type = "acceleration"
            else:
                change_type = "deceleration"

            self._key_points.append(
                KeyPoint(
                    index=idx,
                    value=float(self.values[idx]),
                    timestamp=self.timestamps[idx],
                    point_type="change_point",
                    significance=min(1.0, change / (np.std(self.values) + 1e-10)),
                    metadata={
                        "change_type": change_type,
                        "slope_before": float(left_slope),
                        "slope_after": float(right_slope),
                    },
                )
            )

    def _segment_series(self):
        """Segment series into meaningful chunks for summary"""
        # Simple equal segmentation with statistics
        n_segments = min(10, max(3, self.n_points // 100))
        segment_size = self.n_points // n_segments

        self._segments = []
        for i in range(n_segments):
            start_idx = i * segment_size
            end_idx = min((i + 1) * segment_size, self.n_points)
            segment_values = self.values[start_idx:end_idx]

            self._segments.append(
                {
                    "segment_id": i,
                    "start_index": start_idx,
                    "end_index": end_idx - 1,
                    "start_timestamp": self.timestamps[start_idx],
                    "end_timestamp": self.timestamps[end_idx - 1],
                    "mean": float(np.mean(segment_values)),
                    "std": float(np.std(segment_values)),
                    "min": float(np.min(segment_values)),
                    "max": float(np.max(segment_values)),
                    "trend": "up" if segment_values[-1] > segment_values[0] else "down",
                }
            )

    def _compute_frequency_features(self):
        """Compute frequency domain features using FrequencyAnalyzer with inferred sample rate."""
        # Infer sample rate using hybrid approach
        sample_rate, inference_method = infer_sample_rate(
            timestamps=self.timestamps,
            dataset_name=self.dataset_name,
            domain=self.domain,
            source=self.source,
            sample_metadata=self.sample_metadata,
        )

        # Create analyzer with inferred sample rate
        analyzer = FrequencyAnalyzer(sample_rate=sample_rate)

        # For multivariate: use raw_channels (list of lists format)
        if self.is_multivariate:
            # Convert to list of lists format expected by extract_frequency_features
            values_for_freq = [ch.tolist() for ch in self.raw_channels]
        else:
            # Univariate: use values as list
            values_for_freq = self.values.tolist()

        # Extract frequency features (handles both univariate and multivariate)
        self._frequency_features = analyzer.extract_frequency_features(values_for_freq)

        # Also get dominant frequencies for more detail
        if self.is_multivariate:
            # For multivariate, get dominant frequencies from first channel as summary
            # (extract_frequency_features already averages across channels for features)
            dominant = analyzer.detect_dominant_frequencies(self.raw_channels[0].tolist(), n_peaks=5)
        else:
            dominant = analyzer.detect_dominant_frequencies(values_for_freq, n_peaks=5)

        self._frequency_features["dominant_frequencies"] = [
            {
                "frequency": d["frequency"],
                "magnitude": d["magnitude"],
                "period": d["period"],
                "normalized_magnitude": d["normalized_magnitude"],
            }
            for d in dominant
        ]

        # Add sample rate metadata for transparency
        self._frequency_features["sample_rate"] = sample_rate
        self._frequency_features["sample_rate_inference_method"] = inference_method
        self._frequency_features["frequency_interpretation"] = get_frequency_interpretation(
            sample_rate, inference_method
        )

        # Add unit metadata for proper formatting (critical for normalized vs absolute)
        is_normalized = inference_method == "normalized"
        self._frequency_features["is_normalized"] = is_normalized
        self._frequency_features["frequency_unit"] = "cycles/sample" if is_normalized else "Hz"
        self._frequency_features["period_unit"] = "samples" if is_normalized else "seconds"

    # PUBLIC API - These methods are called by numerical agents
    def get_summary(self) -> str:
        """
        Get high-level summary of the time series.
        This ALWAYS fits in context and provides the initial view.

        Returns:
            Formatted string with summary statistics and key points
        """
        summary_parts = []

        # Basic statistics
        summary_parts.append("**Time Series Summary:**")
        summary_parts.append(f"- Length: {self._statistics['length']} data points")
        summary_parts.append(
            f"- Time Range: index 0 ({self.timestamps[0]}) to index {self.n_points - 1} ({self.timestamps[-1]})"
        )
        summary_parts.append(f"- Value Range: [{self._statistics['min']:.2f}, {self._statistics['max']:.2f}]")
        summary_parts.append(f"- Mean: {self._statistics['mean']:.2f} (±{self._statistics['std']:.2f})")
        summary_parts.append(f"- Median: {self._statistics['median']:.2f}")
        summary_parts.append(
            f"- Overall Trend: {self._statistics['trend_direction']} ({self._statistics['pct_change']:.2f}% change)"
        )

        # Key points summary
        summary_parts.append("\n**Key Points Detected:**")
        key_point_counts = {}
        for kp in self._key_points:
            key_point_counts[kp.point_type] = key_point_counts.get(kp.point_type, 0) + 1

        for pt, count in key_point_counts.items():
            summary_parts.append(f"- {pt.capitalize()}s: {count}")

        # All significant points
        summary_parts.append("\n**Significant Points (for focused analysis):**")
        for kp in self._key_points:
            summary_parts.append(
                f"- Index {kp.index} ({kp.timestamp}): {kp.value:.2f} [{kp.point_type}, significance={kp.significance:.2f}]"
            )

        # Segment overview
        summary_parts.append("\n**Segment Overview:**")
        for seg in self._segments:
            summary_parts.append(
                f"- Segment {seg['segment_id']}: idx [{seg['start_index']}-{seg['end_index']}], "
                f"mean={seg['mean']:.2f}, trend={seg['trend']}"
            )

        # Frequency domain features
        if self._frequency_features:
            summary_parts.append("\n**Frequency Domain Features:**")
            freq = self._frequency_features

            if freq.get("spectral_centroid") is not None:
                summary_parts.append(f"- Spectral Centroid: {freq['spectral_centroid']:.2f} Hz")
            if freq.get("spectral_spread") is not None:
                summary_parts.append(f"- Spectral Spread: {freq['spectral_spread']:.2f} Hz")
            if freq.get("dominant_frequency") is not None and freq.get("dominant_frequency", 0) > 0:
                summary_parts.append(f"- Dominant Frequency: {freq['dominant_frequency']:.2f} Hz")
                period = freq.get("dominant_period")
                if period is not None and period != float("inf"):
                    summary_parts.append(f"- Dominant Period: {period:.2f}")
            if freq.get("spectral_entropy") is not None:
                summary_parts.append(f"- Spectral Entropy: {freq['spectral_entropy']:.2f}")
            if freq.get("n_dominant_peaks") is not None:
                summary_parts.append(f"- Number of Dominant Peaks: {freq['n_dominant_peaks']}")

            # Show top dominant frequencies
            if freq.get("dominant_frequencies"):
                summary_parts.append("\n**Top Dominant Frequencies:**")
                for i, df in enumerate(freq["dominant_frequencies"][:3], 1):
                    period_str = f"{df['period']:.2f}" if df.get("period") != float("inf") else "N/A"
                    summary_parts.append(
                        f"  {i}. {df['frequency']:.2f} Hz (period: {period_str}, "
                        f"magnitude: {df['normalized_magnitude']:.3f})"
                    )

        # Instructions for focused lookup
        summary_parts.append("\n**Available Lookup Commands:**")
        summary_parts.append("- To examine specific indices: 'LOOKUP_RANGE: start_idx, end_idx'")
        summary_parts.append("- To focus on a key point: 'LOOKUP_POINT: index'")
        summary_parts.append("- To see all peaks: 'LOOKUP_TYPE: peaks'")
        summary_parts.append("- To see all anomalies: 'LOOKUP_TYPE: anomalies'")

        return "\n".join(summary_parts)

    def lookup_range(self, start_idx: int, end_idx: int) -> LookupResult:
        """
        Get values for a specific index range.

        Args:
            start_idx: Start index (inclusive)
            end_idx: End index (inclusive)

        Returns:
            LookupResult with values in the range
        """
        # Clamp indices
        start_idx = max(0, min(start_idx, self.n_points - 1))
        end_idx = max(0, min(end_idx, self.n_points - 1))

        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx

        # Limit range size
        range_size = end_idx - start_idx + 1
        if range_size > self.max_values_per_lookup:
            # Subsample to fit
            step = range_size // self.max_values_per_lookup
            indices = list(range(start_idx, end_idx + 1, step))
        else:
            indices = list(range(start_idx, end_idx + 1))

        values = [float(self.values[i]) for i in indices]
        timestamps = [self.timestamps[i] for i in indices]

        # Compute local statistics
        local_values = self.values[start_idx : end_idx + 1]
        metadata = {
            "requested_range": (start_idx, end_idx),
            "actual_points": len(indices),
            "local_mean": float(np.mean(local_values)),
            "local_std": float(np.std(local_values)),
            "local_min": float(np.min(local_values)),
            "local_max": float(np.max(local_values)),
            "local_trend": "up" if local_values[-1] > local_values[0] else "down",
        }

        context = (
            f"Values for index range [{start_idx}, {end_idx}] ({len(indices)} points shown):\n"
            f"Local stats: mean={metadata['local_mean']:.2f}, std={metadata['local_std']:.2f}, "
            f"range=[{metadata['local_min']:.2f}, {metadata['local_max']:.2f}], trend={metadata['local_trend']}"
        )

        return LookupResult(indices=indices, values=values, timestamps=timestamps, context=context, metadata=metadata)

    def lookup_around_point(self, center_idx: int, window_size: int = None) -> LookupResult:
        """
        Get values around a specific index (zoom in on a point).

        Args:
            center_idx: Center index to focus on
            window_size: Number of points on each side (default: 10)

        Returns:
            LookupResult with values around the center point
        """
        if window_size is None:
            window_size = self.DEFAULT_WINDOW_SIZE

        start_idx = max(0, center_idx - window_size)
        end_idx = min(self.n_points - 1, center_idx + window_size)

        result = self.lookup_range(start_idx, end_idx)
        result.metadata["center_index"] = center_idx
        result.metadata["center_value"] = float(self.values[center_idx])
        result.context = (
            f"Focused view around index {center_idx} (value={self.values[center_idx]:.2f}):\n" + result.context
        )

        return result

    def lookup_by_type(self, point_type: str, max_points: int = 10) -> LookupResult:
        """
        Get all key points of a specific type.

        Args:
            point_type: One of "peaks", "troughs", "anomalies", "change_points"
            max_points: Maximum number of points to return

        Returns:
            LookupResult with matching key points
        """
        # Normalize type name
        type_map = {
            "peaks": "peak",
            "peak": "peak",
            "troughs": "trough",
            "trough": "trough",
            "anomalies": "anomaly",
            "anomaly": "anomaly",
            "change_points": "change_point",
            "change_point": "change_point",
        }
        normalized_type = type_map.get(point_type.lower(), point_type.lower())

        matching_points = [kp for kp in self._key_points if kp.point_type == normalized_type]
        matching_points = matching_points[:max_points]

        if not matching_points:
            return LookupResult(
                indices=[],
                values=[],
                timestamps=[],
                context=f"No {point_type} found in the time series.",
                metadata={"point_type": point_type, "count": 0},
            )

        indices = [kp.index for kp in matching_points]
        values = [kp.value for kp in matching_points]
        timestamps = [kp.timestamp for kp in matching_points]

        context_lines = [f"Found {len(matching_points)} {point_type}:"]
        for kp in matching_points:
            context_lines.append(
                f"- Index {kp.index} ({kp.timestamp}): {kp.value:.2f} [significance={kp.significance:.2f}]"
            )

        return LookupResult(
            indices=indices,
            values=values,
            timestamps=timestamps,
            context="\n".join(context_lines),
            metadata={
                "point_type": point_type,
                "count": len(matching_points),
                "points": [
                    {"index": kp.index, "value": kp.value, "significance": kp.significance, "metadata": kp.metadata}
                    for kp in matching_points
                ],
            },
        )

    def lookup_by_timestamp_range(self, start_timestamp: Any, end_timestamp: Any) -> LookupResult:
        """
        Get values for a timestamp range (not index range).

        Args:
            start_timestamp: Start timestamp
            end_timestamp: End timestamp

        Returns:
            LookupResult with values in the timestamp range
        """
        # Find indices for timestamps
        try:
            start_idx = self.timestamps.index(start_timestamp)
        except ValueError:
            # Find closest timestamp
            start_idx = 0
            for i, ts in enumerate(self.timestamps):
                if ts >= start_timestamp:
                    start_idx = i
                    break

        try:
            end_idx = self.timestamps.index(end_timestamp)
        except ValueError:
            end_idx = self.n_points - 1
            for i in range(self.n_points - 1, -1, -1):
                if self.timestamps[i] <= end_timestamp:
                    end_idx = i
                    break

        result = self.lookup_range(start_idx, end_idx)
        result.metadata["timestamp_range"] = (start_timestamp, end_timestamp)
        return result

    def get_key_points(self, max_points: int = 20) -> List[KeyPoint]:
        """
        Get the most significant key points.

        Args:
            max_points: Maximum number of points to return

        Returns:
            List of KeyPoint objects sorted by significance
        """
        return self._key_points[:max_points]

    def get_segments(self) -> List[Dict[str, Any]]:
        """Get segment summaries"""
        return self._segments

    def get_indicator(self, start_idx: int = 0, end_idx: Optional[int] = None) -> LookupResult:
        """
        Get pre-computed technical indicator values (e.g., MACD, Bollinger Band).

        This indicator is computed from the main time series and provided by the dataset.
        Use this to analyze the indicator alongside the price/value data.

        Args:
            start_idx: Start index (default: 0)
            end_idx: End index (default: last index)

        Returns:
            LookupResult with indicator values and statistics
        """
        if not self.has_indicator:
            return LookupResult(
                indices=[],
                values=[],
                timestamps=[],
                context=f"No {self.indicator_label} indicator data available for this sample.",
                metadata={"has_indicator": False},
            )

        n_indicator = len(self.indicator_values)

        # Handle defaults
        if end_idx is None:
            end_idx = n_indicator - 1

        # Clamp indices
        start_idx = max(0, min(start_idx, n_indicator - 1))
        end_idx = max(0, min(end_idx, n_indicator - 1))

        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx

        # Limit range size
        range_size = end_idx - start_idx + 1
        if range_size > self.max_values_per_lookup:
            step = range_size // self.max_values_per_lookup
            indices = list(range(start_idx, end_idx + 1, step))
        else:
            indices = list(range(start_idx, end_idx + 1))

        values = [float(self.indicator_values[i]) for i in indices]
        # Use main series timestamps (indicator is aligned)
        timestamps = [self.timestamps[i] if i < len(self.timestamps) else i for i in indices]

        # Local statistics for the requested range
        local_values = self.indicator_values[start_idx : end_idx + 1]
        metadata = {
            "indicator_label": self.indicator_label,
            "has_indicator": True,
            "total_length": n_indicator,
            "requested_range": (start_idx, end_idx),
            "actual_points": len(indices),
            "global_stats": self._indicator_stats,
            "local_mean": float(np.mean(local_values)),
            "local_std": float(np.std(local_values)),
            "local_min": float(np.min(local_values)),
            "local_max": float(np.max(local_values)),
        }

        context_lines = [
            f"**{self.indicator_label} Indicator:**",
            f"  - Total Length: {n_indicator} points (aligned with main time series)",
            f"  - Global Range: [{self._indicator_stats['min']:.2f}, {self._indicator_stats['max']:.2f}]",
            f"  - Global Mean: {self._indicator_stats['mean']:.2f} (±{self._indicator_stats['std']:.2f})",
            "",
            f"Values for index range [{start_idx}, {end_idx}] ({len(indices)} points):",
            f"  Local stats: mean={metadata['local_mean']:.2f}, std={metadata['local_std']:.2f}, "
            f"range=[{metadata['local_min']:.2f}, {metadata['local_max']:.2f}]",
        ]

        return LookupResult(
            indices=indices,
            values=values,
            timestamps=timestamps,
            context="\n".join(context_lines),
            metadata=metadata,
        )

    def process_cross_modal_hint(self, hint: str) -> Optional[LookupResult]:
        """
        Process a hint from another modality agent.

        Other modalities (TEXT, VISUAL) can suggest regions to examine.
        For example:
        - TEXT: "News mentions earnings release on day 45"
        - VISUAL: "Chart shows spike around index 100-120"

        Args:
            hint: Natural language hint about where to focus

        Returns:
            LookupResult if hint can be parsed, None otherwise
        """
        import re

        # Try to extract index range from hint
        # Pattern: "index/indices X-Y" or "around index X" or "between X and Y"
        range_patterns = [
            r"ind(?:ex|ices)?\s*(\d+)\s*[-–to]+\s*(\d+)",
            r"between\s*(\d+)\s*and\s*(\d+)",
            r"range\s*\[?\s*(\d+)\s*[,\-–]\s*(\d+)\s*\]?",
        ]

        for pattern in range_patterns:
            match = re.search(pattern, hint, re.IGNORECASE)
            if match:
                start_idx = int(match.group(1))
                end_idx = int(match.group(2))
                result = self.lookup_range(start_idx, end_idx)
                result.metadata["triggered_by"] = hint
                return result

        # Try to extract single index
        single_patterns = [
            r"around\s*(?:index\s*)?(\d+)",
            r"at\s*(?:index\s*)?(\d+)",
            r"index\s*(\d+)",
            r"point\s*(\d+)",
        ]

        for pattern in single_patterns:
            match = re.search(pattern, hint, re.IGNORECASE)
            if match:
                center_idx = int(match.group(1))
                result = self.lookup_around_point(center_idx)
                result.metadata["triggered_by"] = hint
                return result

        # Try to extract point type
        type_patterns = [
            (r"peaks?|maximums?|highs?", "peaks"),
            (r"troughs?|minimums?|lows?", "troughs"),
            (r"anomal(?:y|ies)|outliers?|spikes?", "anomalies"),
            (r"change\s*points?|trend\s*change|reversal", "change_points"),
        ]

        for pattern, point_type in type_patterns:
            if re.search(pattern, hint, re.IGNORECASE):
                result = self.lookup_by_type(point_type)
                result.metadata["triggered_by"] = hint
                return result

        return None

    def to_prompt_string(self, include_values: bool = False, focus_indices: Optional[List[int]] = None) -> str:
        """
        Generate prompt string for numerical agents.

        Args:
            include_values: Whether to include raw values (for small series)
            focus_indices: Specific indices to focus on (from cross-modal hints)

        Returns:
            Formatted string for LLM prompt
        """
        parts = [self.get_summary()]

        # If specific focus indices are provided, show those values
        if focus_indices:
            parts.append("\n**Focused Values (from cross-modal hints):**")
            for idx in focus_indices:
                if 0 <= idx < self.n_points:
                    parts.append(f"- Index {idx} ({self.timestamps[idx]}): {self.values[idx]:.2f}")

        # Include all values if requested
        elif include_values:
            parts.append(f"\n**Complete Values ({self.n_points} points):**")
            formatted_values = [f"{v:.2f}" for v in self.values]
            parts.append(f"[{', '.join(formatted_values)}]")

        return "\n".join(parts)


# Convenience function for creating lookup from context
def create_lookup_function(context: Dict[str, Any]) -> NumericalLookupFunction:
    """
    Create a NumericalLookupFunction from debate context.

    Args:
        context: Debate context dictionary with 'values' and optional 'timestamps',
                'dataset_name', 'domain', 'source', 'input_indicator', 'indicator_label', etc.

    Returns:
        Configured NumericalLookupFunction with sample metadata for sample rate inference
    """
    values = context["values"]  # All loaders normalize to "values"
    timestamps = context.get("timestamps")

    # Extract sample metadata from context for sample rate inference
    sample_metadata = {
        "dataset_name": context.get("dataset_name"),
        "domain": context.get("domain"),
        "source": context.get("source"),
        # Include any other metadata that might be useful
    }

    # Extract indicator data if available
    indicator_values = context.get("input_indicator")
    indicator_label = context.get("indicator_label", "Indicator")

    return NumericalLookupFunction(
        values=values,
        timestamps=timestamps,
        sample_metadata=sample_metadata,
        indicator_values=indicator_values,
        indicator_label=indicator_label,
    )
