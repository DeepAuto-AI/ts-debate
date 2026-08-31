import io
import math
import statistics
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from langchain.tools import tool
from scipy import signal, stats

from utils.numerical_lookup import NumericalLookupFunction


# LOOKUP FUNCTION TOOLS (for NUMERICAL agents)
def create_lookup_tools(lookup_fn: NumericalLookupFunction, include_frequency_features: bool = True) -> List:
    """
    Create Lookup Function tools for NUMERICAL agents.

    - Tools return raw historical time series data
    - Agent decides what to compute from it
    """
    # Build schema description
    n = len(lookup_fn.values)
    ts_type = type(lookup_fn.timestamps[0]).__name__ if lookup_fn.timestamps else "unknown"
    val_min, val_max = float(lookup_fn.values.min()), float(lookup_fn.values.max())
    val_mean = float(lookup_fn.values.mean())

    @tool
    def get_info() -> str:
        """Get schema and statistics of the historical time series.

        Call this to understand the data before querying specific ranges.
        Returns: length, timestamp format, value range, channels (if multivariate), detected features, and frequency features summary.
        """
        try:
            info_lines = [
                "Time Series Schema:",
                f"  - Length: {n} points",
                f"  - Indices: 0 to {n - 1}",
                f"  - Timestamps: {ts_type}, first: {lookup_fn.timestamps[0] if lookup_fn.timestamps else 'N/A'}, last: {lookup_fn.timestamps[-1] if lookup_fn.timestamps else 'N/A'}",
            ]

            # Show multivariate info
            if lookup_fn.is_multivariate:
                info_lines.append(f"  - MULTIVARIATE: {lookup_fn.n_channels} channels")
                info_lines.append("  - Use get_channel_values(channel, start, end) to access specific channels")
                info_lines.append("")
                info_lines.append("Per-channel statistics:")
                for ch_stat in lookup_fn._statistics.get("channel_stats", []):
                    info_lines.append(
                        f"  Channel {ch_stat['channel']}: mean={ch_stat['mean']:.2f}, std={ch_stat['std']:.2f}, range=[{ch_stat['min']:.2f}, {ch_stat['max']:.2f}]"
                    )
            else:
                info_lines.append(f"  - Values: range=[{val_min:.2f}, {val_max:.2f}], mean={val_mean:.2f}")

            info_lines.append("")
            info_lines.append("Detected features (on aggregate signal):")
            # Count features by type
            feature_counts = {}
            for kp in lookup_fn._key_points:
                pt = kp.point_type
                feature_counts[pt] = feature_counts.get(pt, 0) + 1
            for pt, count in feature_counts.items():
                if pt not in ("start", "end"):
                    info_lines.append(f"  - {pt}s: {count}")

            # Add frequency domain features
            if lookup_fn._frequency_features:
                freq = lookup_fn._frequency_features
                freq_unit = freq.get("frequency_unit", "Hz")
                period_unit = freq.get("period_unit", "seconds")
                is_normalized = freq.get("is_normalized", False)

                info_lines.append("")
                info_lines.append("Frequency Domain Features:")
                if is_normalized:
                    info_lines.append("  ⚠️ Note: Frequencies are NORMALIZED (cycles/sample, not Hz)")
                if freq.get("spectral_centroid") is not None:
                    info_lines.append(f"  - Spectral Centroid: {freq['spectral_centroid']:.4f} {freq_unit}")
                if freq.get("dominant_frequency") is not None and freq.get("dominant_frequency", 0) > 0:
                    info_lines.append(f"  - Dominant Frequency: {freq['dominant_frequency']:.4f} {freq_unit}")
                    period = freq.get("dominant_period")
                    if period is not None and period != float("inf"):
                        info_lines.append(f"  - Dominant Period: {period:.2f} {period_unit}")
                if freq.get("n_dominant_peaks") is not None:
                    info_lines.append(f"  - Dominant Peaks: {freq['n_dominant_peaks']}")
                info_lines.append("  - Use get_frequency_features() for detailed frequency analysis")

            # Add indicator info if available
            if lookup_fn.has_indicator and lookup_fn.indicator_values is not None:
                ind_vals = lookup_fn.indicator_values
                info_lines.append("")
                info_lines.append(f"Technical Indicator: {lookup_fn.indicator_label}")
                info_lines.append(f"  - Length: {len(ind_vals)} points")
                info_lines.append(f"  - Range: [{float(ind_vals.min()):.4f}, {float(ind_vals.max()):.4f}]")
                info_lines.append(f"  - Mean: {float(ind_vals.mean()):.4f}")
                info_lines.append("  - Use get_indicator() to query indicator values")

            return "\n".join(info_lines)
        except Exception as e:
            return f"Error in get_info: {type(e).__name__}: {e!s}"

    MAX_VALUES_PER_CALL = 100  # Limit to prevent token explosion

    @tool
    def get_values(start: str, end: str) -> str:
        """Get historical time series values for a range. Accepts indices OR timestamps.
        
        ⚠️ IMPORTANT: Returns max 100 values. For larger ranges, use get_info() for summary stats
        or call get_features() to find key points first, then get_around() for specific locations.
        DO NOT call this tool repeatedly for overlapping ranges.

        Args:
            start: Start of range - can be index ("50") or timestamp ("2024-01-15", "14:30:00")
            end: End of range - can be index ("100") or timestamp ("2024-01-20", "15:00:00")

        Examples:
            get_values("0", "50")           # First 51 values by index
            get_values("100", "150")        # Values from index 100-150
            get_values("2024-01-15", "2024-01-20")  # By date range
        """
        try:
            n = len(lookup_fn.values)
            timestamps = lookup_fn.timestamps

            # Try to parse as indices first
            try:
                start_idx = int(start)
                end_idx = int(end)
            except ValueError:
                # Try to find timestamps
                start_idx = None
                end_idx = None

                # Convert timestamps to strings for comparison
                ts_strs = [str(t) for t in timestamps]

                # Find matching or closest timestamps
                for i, ts in enumerate(ts_strs):
                    if start in ts or ts in start:
                        start_idx = i
                        break
                for i, ts in enumerate(ts_strs):
                    if end in ts or ts in end:
                        end_idx = i

                if start_idx is None or end_idx is None:
                    return f"Could not find timestamps matching '{start}' to '{end}'. Use indices (0 to {n - 1}) instead."

            # Validate indices
            start_idx = max(0, start_idx)
            end_idx = min(n - 1, end_idx)

            if start_idx > end_idx:
                return f"Invalid range: start ({start_idx}) > end ({end_idx})"

            values = lookup_fn.values[start_idx : end_idx + 1].tolist()
            ts_range = timestamps[start_idx : end_idx + 1]
            
            # Limit values to prevent token explosion
            if len(values) > MAX_VALUES_PER_CALL:
                # Return summary + first/last samples
                first_n = MAX_VALUES_PER_CALL // 2
                last_n = MAX_VALUES_PER_CALL // 2
                first_vals = ", ".join([f"{v:.2f}" for v in values[:first_n]])
                last_vals = ", ".join([f"{v:.2f}" for v in values[-last_n:]])
                val_arr = np.array(values)
                return (
                    f"[{start_idx}:{end_idx}] ({len(values)} values, TRUNCATED to {MAX_VALUES_PER_CALL}):\n"
                    f"  Summary: min={val_arr.min():.2f}, max={val_arr.max():.2f}, mean={val_arr.mean():.2f}, std={val_arr.std():.2f}\n"
                    f"  First {first_n}: [{first_vals}]\n"
                    f"  Last {last_n}: [{last_vals}]\n"
                    f"  TIP: Use get_features() to find key points, then get_around() for details. If available, leverage location hints from other analysts to get more specific values."
                )
            
            values_str = ", ".join([f"{v:.2f}" for v in values])
            return f"[{start_idx}:{end_idx}] (timestamps: {ts_range[0]} to {ts_range[-1]}, {len(values)} values): [{values_str}]"
        except Exception as e:
            return f"Error in get_values: {type(e).__name__}: {e!s}"

    MAX_WINDOW = 50  # Max window size for get_around

    @tool
    def get_around(center: str, window: int) -> str:
        """Get historical time series values around a specific point. Useful for examining peaks, anomalies, etc.
        
        ⚠️ Max window: 50 points on each side. DO NOT call repeatedly for adjacent/overlapping regions.

        Args:
            center: Center point - index ("42") or timestamp ("2024-01-15 14:30")
            window: Number of points on each side (use 10 if unsure, max: 50)

        Example: get_around("42", 5) returns values from index 37 to 47
        """
        try:
            n = len(lookup_fn.values)
            timestamps = lookup_fn.timestamps
            
            # Enforce max window
            window = min(window, MAX_WINDOW)

            # Parse center
            try:
                center_idx = int(center)
            except ValueError:
                # Find matching timestamp
                center_idx = None
                ts_strs = [str(t) for t in timestamps]
                for i, ts in enumerate(ts_strs):
                    if center in ts or ts in center:
                        center_idx = i
                        break
                if center_idx is None:
                    return f"Could not find timestamp matching '{center}'. Use index (0 to {n - 1}) instead."

            # Validate center_idx is within bounds
            if center_idx < 0 or center_idx >= n:
                return f"Invalid center index {center_idx}. Valid range: 0 to {n - 1}."

            start_idx = max(0, center_idx - window)
            end_idx = min(n - 1, center_idx + window)

            values = lookup_fn.values[start_idx : end_idx + 1].tolist()
            values_str = ", ".join([f"{v:.2f}" for v in values])
            center_value = lookup_fn.values[center_idx]

            return f"Around index {center_idx} (value={center_value:.2f}), window [{start_idx}:{end_idx}]: [{values_str}]"
        except Exception as e:
            return f"Error in get_around: {type(e).__name__}: {e!s}"

    @tool
    def get_features(feature_type: str) -> str:
        """Get locations of significant features in the historical time series.

        Args:
            feature_type: One of:
                - 'peaks': Local maxima (detected via scipy)
                - 'troughs': Local minima
                - 'anomalies': Outliers (IQR method)
                - 'change_points': Trend reversals
                - 'all': All detected features

        Returns indices and values. Use get_around() or get_values() to examine these locations in detail.
        """
        try:
            if feature_type.lower() == "all":
                # Return all feature types
                all_features = []
                for ft in ["peaks", "troughs", "anomalies", "change_points"]:
                    result = lookup_fn.lookup_by_type(ft)
                    for idx, val in zip(result.indices, result.values):
                        all_features.append(f"{ft[:-1]}@{idx}: {val:.2f}")
                if not all_features:
                    return "No significant features detected in the time series."
                return f"ALL FEATURES ({len(all_features)} found): {', '.join(all_features)}"

            result = lookup_fn.lookup_by_type(feature_type)
            if not result.indices:
                return f"No {feature_type} found in the time series."

            features = []
            for idx, val in zip(result.indices, result.values):
                features.append(f"idx={idx}: {val:.2f}")

            return f"{feature_type.upper()} ({len(features)} found): {', '.join(features)}"
        except Exception as e:
            return f"Error in get_features: {type(e).__name__}: {e!s}"

    @tool
    def get_channel_values(channel: int, start: str, end: str) -> str:
        """Get historical time series values from a specific channel for multivariate time series.

        Args:
            channel: Channel index (0, 1, 2, ... for multivariate data)
            start: Start index as string ("0", "50", etc.)
            end: End index as string ("100", "150", etc.)

        Examples:
            get_channel_values(0, "0", "50")   # First 51 values from channel 0
            get_channel_values(1, "100", "150") # Values from channel 1, indices 100-150

        For HAR data: channel 0=X-axis, channel 1=Y-axis, channel 2=Z-axis
        """
        try:
            if not lookup_fn.is_multivariate:
                return "Data is univariate - use get_values() instead."

            if channel < 0 or channel >= lookup_fn.n_channels:
                return f"Invalid channel {channel}. Valid channels: 0 to {lookup_fn.n_channels - 1}"

            try:
                start_idx = int(start)
                end_idx = int(end)
            except ValueError:
                return f"Invalid indices. Use integers like get_channel_values({channel}, '0', '50')"

            # Validate indices
            start_idx = max(0, start_idx)
            end_idx = min(lookup_fn.n_points - 1, end_idx)

            if start_idx > end_idx:
                return f"Invalid range: start ({start_idx}) > end ({end_idx})"

            # Get values from specific channel
            channel_data = lookup_fn.raw_channels[channel]
            values = channel_data[start_idx : end_idx + 1].tolist()
            values_str = ", ".join([f"{v:.3f}" for v in values])

            return f"Channel {channel} [{start_idx}:{end_idx}] ({len(values)} values): [{values_str}]"
        except Exception as e:
            return f"Error in get_channel_values: {type(e).__name__}: {e!s}"

    @tool
    def get_all_channels(start: str, end: str) -> str:
        """Get historical time series values from all channels for a range of indices.

        Returns a table-like view comparing all channels at each timestamp.
        Useful for cross-channel correlation analysis.

        Args:
            start: Start index as string ("0", "50", etc.)
            end: End index as string ("100", "150", etc.)

        Example:
            get_all_channels("0", "10")  # Returns all channels for indices 0-10
        """
        try:
            if not lookup_fn.is_multivariate:
                return "Data is univariate - use get_values() instead."

            try:
                start_idx = int(start)
                end_idx = int(end)
            except ValueError:
                return "Invalid indices. Use integers like get_all_channels('0', '50')"

            # Validate indices
            start_idx = max(0, start_idx)
            end_idx = min(lookup_fn.n_points - 1, end_idx)

            if start_idx > end_idx:
                return f"Invalid range: start ({start_idx}) > end ({end_idx})"

            # Limit output to avoid context overflow
            max_rows = 50
            if end_idx - start_idx + 1 > max_rows:
                end_idx = start_idx + max_rows - 1

            # Build table output
            lines = [f"All {lookup_fn.n_channels} channels [{start_idx}:{end_idx}]:"]
            lines.append("idx | " + " | ".join([f"ch{i:d}" for i in range(lookup_fn.n_channels)]))
            lines.append("-" * (6 + lookup_fn.n_channels * 10))

            for idx in range(start_idx, end_idx + 1):
                row_values = [f"{lookup_fn.raw_channels[ch][idx]:.3f}" for ch in range(lookup_fn.n_channels)]
                lines.append(f"{idx:3d} | " + " | ".join(row_values))

            return "\n".join(lines)
        except Exception as e:
            return f"Error in get_all_channels: {type(e).__name__}: {e!s}"

    @tool
    def get_frequency_features() -> str:
        """Get detailed frequency domain analysis features.

        Returns spectral characteristics including:
        - Spectral centroid, spread, rolloff
        - Dominant frequencies and periods
        - Spectral entropy and flatness
        - Power spectral density characteristics

        Use this when you need frequency-domain insights for periodic pattern analysis,
        detecting cycles, or understanding the spectral composition of the time series.
        """
        try:
            freq = lookup_fn._frequency_features

            # Get unit metadata for proper formatting
            freq_unit = freq.get("frequency_unit", "Hz")
            period_unit = freq.get("period_unit", "seconds")
            is_normalized = freq.get("is_normalized", False)

            lines = ["**Frequency Domain Analysis:**"]

            # Add important note for normalized frequencies
            if is_normalized:
                lines.append("")
                lines.append("⚠️ **IMPORTANT: Frequencies are NORMALIZED (cycles per sample)**")
                lines.append("   - Frequency values are relative patterns, NOT absolute Hz")
                lines.append("   - Period values are in SAMPLES, not seconds")
                lines.append("   - To convert: actual_freq = normalized_freq × sampling_rate")
                lines.append("")

            # Core spectral features
            if freq.get("spectral_centroid") is not None:
                lines.append(f"Spectral Centroid: {freq['spectral_centroid']:.4f} {freq_unit}")
            if freq.get("spectral_spread") is not None:
                lines.append(f"Spectral Spread: {freq['spectral_spread']:.4f} {freq_unit}")
            if freq.get("spectral_rolloff") is not None:
                lines.append(f"Spectral Rolloff: {freq['spectral_rolloff']:.4f} {freq_unit}")

            # Dominant frequencies summary
            if freq.get("dominant_frequency") is not None and freq.get("dominant_frequency", 0) > 0:
                lines.append(f"\nDominant Frequency: {freq['dominant_frequency']:.4f} {freq_unit}")
                period = freq.get("dominant_period")
                if period is not None and period != float("inf"):
                    lines.append(f"Dominant Period: {period:.2f} {period_unit}")
                lines.append(f"Number of Dominant Peaks: {freq.get('n_dominant_peaks', 0)}")

            # All dominant frequencies
            if freq.get("dominant_frequencies"):
                lines.append("\n**All Dominant Frequencies:**")
                for i, df in enumerate(freq["dominant_frequencies"], 1):
                    period_str = f"{df['period']:.2f} {period_unit}" if df.get("period") != float("inf") else "N/A"
                    lines.append(
                        f"  {i}. Frequency: {df['frequency']:.4f} {freq_unit} | "
                        f"Period: {period_str} | "
                        f"Magnitude: {df['normalized_magnitude']:.3f}"
                    )

            # Spectral characteristics
            if freq.get("spectral_entropy") is not None:
                lines.append(f"\nSpectral Entropy: {freq['spectral_entropy']:.4f}")
            if freq.get("spectral_flatness") is not None:
                lines.append(f"Spectral Flatness: {freq['spectral_flatness']:.4f}")
            if freq.get("total_power") is not None:
                lines.append(f"Total Power: {freq['total_power']:.4f}")

            # Additional peak information if available
            for i in range(1, 4):  # Show up to 3 peaks
                peak_freq_key = f"peak_{i}_frequency"
                if peak_freq_key in freq and freq[peak_freq_key] is not None:
                    if i == 1:
                        lines.append("\n**Peak Frequencies:**")
                    peak_freq = freq[peak_freq_key]
                    peak_mag = freq.get(f"peak_{i}_magnitude", "N/A")
                    peak_period = freq.get(f"peak_{i}_period", "N/A")
                    if peak_period != float("inf"):
                        peak_period_str = f"{peak_period:.2f} {period_unit}"
                    else:
                        peak_period_str = "N/A"
                    lines.append(
                        f"  Peak {i}: {peak_freq:.4f} {freq_unit} (period: {peak_period_str}, magnitude: {peak_mag})"
                    )

            # Add interpretation note at the end
            if freq.get("frequency_interpretation"):
                lines.append(f"\n📊 Interpretation: {freq['frequency_interpretation']}")

            return "\n".join(lines)
        except Exception as e:
            return f"Error in get_frequency_features: {type(e).__name__}: {e!s}"

    @tool
    def get_indicator(start: str, end: str) -> str:
        """Get historical pre-computed technical indicator values (e.g., MACD, Bollinger Band).

        This indicator is computed from the main time series and provided by the dataset.
        Use this to analyze the indicator alongside the price/value data.

        Args:
            start: Start of range - index ("0") or timestamp ("2024-01-15"). Use "0" for beginning.
            end: End of range - index ("100"), timestamp or "" for 10 values.

        Examples:
            get_indicator("0", "")                   # Get indicator info and first 10 values
            get_indicator("0", "100")                # Get indices 0-100
            get_indicator("2024-01-15", "2024-01-20")  # By date range
        """
        try:
            if not lookup_fn.has_indicator or lookup_fn.indicator_values is None:
                return f"No {lookup_fn.indicator_label} indicator data available for this sample."

            n_indicator = len(lookup_fn.indicator_values)
            timestamps = lookup_fn.timestamps

            # Handle "all" for end
            if end.lower() == "all":
                start_idx = 0
                end_idx = n_indicator - 1
            elif end == "":
                # Default: first 10 values
                try:
                    start_idx = int(start)
                except ValueError:
                    start_idx = 0
                end_idx = min(start_idx + 10, n_indicator - 1)
            else:
                # Try to parse as indices first
                try:
                    start_idx = int(start)
                    end_idx = int(end)
                except ValueError:
                    # Try to find timestamps
                    start_idx = None
                    end_idx = None

                    # Convert timestamps to strings for comparison
                    ts_strs = [str(t) for t in timestamps]

                    # Find matching or closest timestamps
                    for i, ts in enumerate(ts_strs):
                        if i < n_indicator and (start in ts or ts in start):
                            start_idx = i
                            break
                    for i, ts in enumerate(ts_strs):
                        if i < n_indicator and (end in ts or ts in end):
                            end_idx = i

                    if start_idx is None or end_idx is None:
                        return f"Could not find timestamps matching '{start}' to '{end}'. Use indices (0 to {n_indicator - 1}) instead."

            # Validate indices
            start_idx = max(0, min(start_idx, n_indicator - 1))
            end_idx = max(0, min(end_idx, n_indicator - 1))

            if start_idx > end_idx:
                return f"Invalid range: start ({start_idx}) > end ({end_idx})"

            # Use the lookup function's method
            result = lookup_fn.get_indicator(start_idx, end_idx)

            # Format output
            output_lines = [result.context, ""]
            values_str = ", ".join([f"{v:.2f}" for v in result.values])
            ts_start = timestamps[result.indices[0]] if result.indices[0] < len(timestamps) else result.indices[0]
            ts_end = timestamps[result.indices[-1]] if result.indices[-1] < len(timestamps) else result.indices[-1]
            output_lines.append(
                f"[{result.indices[0]}:{result.indices[-1]}] (timestamps: {ts_start} to {ts_end}, {len(result.values)} values): [{values_str}]"
            )

            return "\n".join(output_lines)
        except Exception as e:
            return f"Error in get_indicator: {type(e).__name__}: {e!s}"

    # Return appropriate tools based on data type and frequency features flag
    base_tools = [get_info, get_values, get_around, get_features]
    if include_frequency_features:
        base_tools.append(get_frequency_features)
    if lookup_fn.is_multivariate:
        base_tools.extend([get_channel_values, get_all_channels])
    if lookup_fn.has_indicator:
        base_tools.append(get_indicator)
    return base_tools


# CODE EXECUTOR TOOL (for JUDGE agents)
def create_code_executor_tool(context: Dict[str, Any]) -> List:
    """
    Create Code Executor tool for JUDGE agents.
    Allows programmatic verification of claims from the debate.
    """

    # Build schema description of available data
    def _describe_var(key: str, value: Any) -> str:
        """Generate schema description for a variable."""
        if isinstance(value, np.ndarray):
            return f"{key}: ndarray, shape={value.shape}, dtype={value.dtype}, range=[{value.min():.2f}, {value.max():.2f}]"
        if isinstance(value, (list, tuple)) and len(value) > 0:
            elem_type = type(value[0]).__name__
            if isinstance(value[0], (int, float)):
                min_v, max_v = min(value), max(value)
                return f"{key}: list[{elem_type}], len={len(value)}, range=[{min_v:.2f}, {max_v:.2f}]"
            return f"{key}: list[{elem_type}], len={len(value)}"
        if isinstance(value, str) and len(value) < 200:
            return f"{key}: str, len={len(value)}"
        if isinstance(value, (int, float)):
            return f"{key}: {type(value).__name__} = {value}"
        return f"{key}: {type(value).__name__}"

    schema_lines = []
    for k, v in context.items():
        if isinstance(v, str) and len(v) >= 1000:
            continue  # Skip large strings (base64 images, etc.)
        schema_lines.append(_describe_var(k, v))

    schema_str = "\n    ".join(schema_lines) if schema_lines else "none"

    # Build dynamic docstring
    docstring = f"""Execute Python code to verify claims from the debate.

Args:
    code: Python code to execute.

To return values (choose one):
    result = {{"slope": 3.33, "trend": "UP"}}  # dict with named results
    result = 42.5                              # single value
    result_slope = 3.33                        # prefix with result_

Available data (schema):
    {schema_str}

Pre-loaded: np, pd, stats, signal, math, statistics
You can import additional libraries as needed.
"""

    @tool
    def execute_code(code: str) -> str:
        """Execute Python code to verify claims."""  # Real docstring set below
        try:
            # Full Python builtins - Judge can import anything it needs
            import builtins

            exec_globals: Dict[str, Any] = {
                # Pre-loaded for convenience
                "np": np,
                "pd": pd,
                "stats": stats,
                "signal": signal,
                "math": math,
                "statistics": statistics,
                # Full builtins including __import__ for autonomous imports
                "__builtins__": builtins,
            }

            # Add context data (skip large base64 strings)
            for key, value in context.items():
                if not isinstance(value, str) or len(value) < 1000:
                    exec_globals[key] = value

            # Capture stdout
            captured_output = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = captured_output

            exec_locals: Dict[str, Any] = {}
            exec(code, exec_globals, exec_locals)

            sys.stdout = old_stdout
            output = captured_output.getvalue()

            # Build result from multiple sources
            results = []

            # 1. Captured print() output
            if output:
                results.append(output.strip())

            # 2. Check for explicit 'result' variable (dict or single value)
            if "result" in exec_locals:
                r = exec_locals["result"]
                if isinstance(r, dict):
                    for k, v in r.items():
                        results.append(f"{k} = {v}")
                else:
                    results.append(f"result = {r}")

            # 3. Capture 'result_*' prefixed variables (explicit outputs)
            for k, v in exec_locals.items():
                if not k.startswith("result_"):
                    continue
                clean_name = k[7:]  # Remove 'result_' prefix
                if isinstance(v, (int, float, bool)):
                    results.append(f"{clean_name} = {v}")
                elif isinstance(v, str) and len(v) < 200:
                    results.append(f"{clean_name} = '{v}'")
                elif isinstance(v, (list, tuple, np.ndarray)) and len(v) <= 10:
                    results.append(f"{clean_name} = {v}")

            if results:
                return "Result:\n" + "\n".join(results)
            return "Code executed (no output - use 'result = ...' or 'result_varname = ...' to return values)"

        except (SyntaxError, NameError, TypeError, ValueError, KeyError, IndexError, ZeroDivisionError) as e:
            sys.stdout = sys.__stdout__
            return f"Error: {type(e).__name__}: {e!s}"
        except Exception as e:  # noqa: BLE001
            sys.stdout = sys.__stdout__
            return f"Error: {type(e).__name__}: {e!s}"

    # Update tool description with available variables
    execute_code.description = docstring
    return [execute_code]
