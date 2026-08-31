"""
Frequency Domain Analysis for Time Series

Provides frequency domain representations (both numerical and visual) to complement
time-domain analysis in multimodal time series reasoning.

Sample Rate Handling:
- The analyzer accepts a sample_rate parameter (default: 1.0 Hz)
- When sample_rate=1.0, frequencies are normalized (cycles per sample)
- For absolute frequencies, provide the actual sampling rate
- Use sample_rate_inference module to automatically infer sample_rate from:
  * Known dataset rates (TimerBed, MTBench)
  * Domain-based lookup (MTBench)
  * Timestamp inference (datetime timestamps)
  * Normalized fallback (unknown data)
"""

import base64
import io
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq

# Import sample rate inference for automatic inference
from .sample_rate_inference import infer_sample_rate


class FrequencyAnalyzer:
    """Analyze time series in frequency domain"""

    # Class constants for frequency analysis
    PEAK_HEIGHT_THRESHOLD = 0.01  # At least 1% of max magnitude for peak detection
    PEAK_PROMINENCE_THRESHOLD = 0.05  # Prominence threshold (5% of max) for peak detection
    SPECTRAL_ROLLOFF_THRESHOLD = 0.85  # 85% cumulative energy for spectral rolloff
    WELCH_NPERSEG_DEFAULT = 256  # Default nperseg for Welch PSD
    SPECTROGRAM_MIN_WINDOW = 4  # Minimum window size for spectrogram
    SPECTROGRAM_MAX_WINDOW = 32  # Maximum window size for spectrogram
    SPECTROGRAM_WINDOW_DIVISOR = 8  # Divide signal length by this for window size
    GRID_ALPHA = 0.3  # Grid transparency
    BAR_ALPHA = 0.7  # Bar chart transparency
    LINE_ALPHA = 0.7  # Line transparency for multivariate plots

    def __init__(self, sample_rate: float = 1.0):
        """
        Args:
            sample_rate: Sampling rate of the time series (Hz)

                        Note: When sample_rate=1.0, frequencies are normalized
                        (cycles per sample). For absolute frequencies, provide
                        the actual sampling rate. The sample_rate_inference module
                        can automatically infer this from dataset metadata or timestamps.
        """
        self.sample_rate = sample_rate

    def _is_multivariate(self, values: Union[List[float], List[List[float]]]) -> bool:
        """
        Detect if values represent multivariate time series.

        Args:
            values: Time series values (univariate or multivariate)

        Returns:
            True if multivariate (list of lists with >1 channel), False otherwise
        """
        return isinstance(values, list) and len(values) > 1 and isinstance(values[0], list)

    def _format_frequency(self, freq: float, show_period: bool = False) -> str:
        """
        Format frequency with appropriate precision based on magnitude.

        Args:
            freq: Frequency value in Hz
            show_period: If True, also show period (1/frequency) in samples or seconds

        Returns:
            Formatted frequency string
        """
        if freq == 0:
            result = "0 Hz"
        elif freq >= 1:
            result = f"{freq:.1f} Hz"
        elif freq >= 0.1:
            result = f"{freq:.2f} Hz"
        elif freq >= 0.01:
            result = f"{freq:.3f} Hz"
        elif freq >= 0.001:
            result = f"{freq:.4f} Hz"
        else:
            result = f"{freq:.2e} Hz"

        # Optionally add period information
        if show_period and freq > 0:
            period = 1.0 / freq
            if self.sample_rate == 1.0:
                # Normalized frequency - show period in samples
                result += f" (period: {period:.1f} samples)"
            else:
                # Absolute frequency - show period in seconds
                result += f" (period: {period:.2f} s)"

        return result

    def _format_frequency_with_units(self, freq: float, is_normalized: bool) -> str:
        """
        Format frequency with appropriate units based on whether normalized or absolute.

        Args:
            freq: Frequency value
            is_normalized: True if normalized (cycles/sample), False if absolute (Hz)

        Returns:
            Formatted frequency string with correct units
        """
        if freq == 0:
            return "0 cycles/sample" if is_normalized else "0 Hz"
        if is_normalized:
            # Normalized: show as cycles per sample with period in samples
            period = 1.0 / freq if freq > 0 else float("inf")
            if freq >= 0.1:
                return f"{freq:.3f} cyc/samp\n(period: {period:.1f})"
            if freq >= 0.01:
                return f"{freq:.4f} cyc/samp\n(period: {period:.1f})"
            return f"{freq:.4f} cyc/samp\n(period: {period:.0f})"
        # Absolute: show as Hz
        if freq >= 1:
            return f"{freq:.1f} Hz"
        if freq >= 0.1:
            return f"{freq:.2f} Hz"
        if freq >= 0.01:
            return f"{freq:.3f} Hz"
        if freq >= 0.001:
            return f"{freq:.4f} Hz"
        return f"{freq:.2e} Hz"

    def compute_fft(self, values: List[float], normalize: bool = True) -> Dict[str, np.ndarray]:
        """
        Compute Fast Fourier Transform

        Args:
            values: Time series values
            normalize: Whether to normalize the spectrum

        Returns:
            Dict with 'frequencies', 'magnitude', 'phase'
        """
        values = np.array(values, dtype=np.float64)

        # ENSURE 1-D: Flatten if it's a 2D array with 1 row (wrapped univariate)
        if values.ndim == 2 and values.shape[0] == 1:
            values = values.flatten()  # [[1,2,3]] -> [1,2,3]
        elif values.ndim > 1:
            raise ValueError(f"Expected 1-D array, got {values.ndim}-D array with shape {values.shape}")

        n = len(values)

        # Handle NaN values (from imputation tasks) - interpolate to fill gaps
        if np.any(np.isnan(values)):
            mask = ~np.isnan(values)
            n_valid = np.sum(mask)
            if n_valid < 4:
                raise ValueError(f"Insufficient non-NaN values ({n_valid}) for FFT. Need at least 4.")
            # Linear interpolation to fill NaN
            indices = np.arange(n)
            values = np.interp(indices, indices[mask], values[mask])

        # Handle constant/zero signals - return zero spectrum
        if np.std(values) == 0:
            n_positive = n // 2 + 1
            return {
                "frequencies": np.zeros(n_positive),
                "magnitude": np.zeros(n_positive),
                "phase": np.zeros(n_positive),
                "complex": np.zeros(n_positive, dtype=complex),
            }

        # Compute FFT
        fft_values = fft(values)
        frequencies = fftfreq(n, d=1 / self.sample_rate)

        # Take only positive frequencies
        positive_freq_idx = frequencies >= 0
        frequencies = frequencies[positive_freq_idx]
        fft_values = fft_values[positive_freq_idx]

        # Compute magnitude and phase
        magnitude = np.abs(fft_values)
        phase = np.angle(fft_values)

        if normalize:
            max_mag = np.max(magnitude)
            if max_mag > 0:
                magnitude = magnitude / max_mag

        return {"frequencies": frequencies, "magnitude": magnitude, "phase": phase, "complex": fft_values}

    def detect_dominant_frequencies(self, values: List[float], n_peaks: int = 5) -> List[Dict[str, float]]:
        """
        Detect dominant frequencies in the signal

        Args:
            values: Time series values
            n_peaks: Number of dominant frequencies to return

        Returns:
            List of dicts with 'frequency', 'magnitude', 'period'
        """
        fft_result = self.compute_fft(values, normalize=False)
        frequencies = fft_result["frequencies"]
        magnitude = fft_result["magnitude"]

        # Check for zero signal
        max_mag = np.max(magnitude)
        if max_mag == 0:
            return []  # No signal

        # Find peaks with relative threshold and prominence for better detection
        peaks, properties = signal.find_peaks(
            magnitude,
            height=max_mag * self.PEAK_HEIGHT_THRESHOLD,
            prominence=max_mag * self.PEAK_PROMINENCE_THRESHOLD,
        )

        if len(peaks) == 0:
            return []  # No peaks found

        # Sort by height
        sorted_idx = np.argsort(properties["peak_heights"])[::-1]
        peak_idx = peaks[sorted_idx][:n_peaks]

        # Extract dominant frequencies
        dominant = []
        for idx in peak_idx:
            if frequencies[idx] > 0:  # Exclude DC component
                normalized_mag = float(magnitude[idx] / max_mag) if max_mag > 0 else 0.0
                dominant.append(
                    {
                        "frequency": float(frequencies[idx]),
                        "magnitude": float(magnitude[idx]),
                        "period": float(1.0 / frequencies[idx]) if frequencies[idx] > 0 else float("inf"),
                        "normalized_magnitude": normalized_mag,
                    }
                )

        return dominant

    def compute_power_spectral_density(self, values: List[float], method: str = "welch") -> Dict[str, np.ndarray]:
        """
        Compute Power Spectral Density

        Args:
            values: Time series values
            method: 'welch' or 'periodogram'

        Returns:
            Dict with 'frequencies' and 'psd'
        """
        values = np.array(values)

        if method == "welch":
            # Adaptive nperseg: use default for long signals, scale down for short signals
            nperseg = min(self.WELCH_NPERSEG_DEFAULT, len(values))
            frequencies, psd = signal.welch(values, fs=self.sample_rate, nperseg=nperseg)
        else:  # periodogram
            frequencies, psd = signal.periodogram(values, fs=self.sample_rate)

        return {"frequencies": frequencies, "psd": psd}

    def compute_spectrogram(self, values: List[float], window_size: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Compute spectrogram (time-frequency representation)

        Args:
            values: Time series values
            window_size: Window size for STFT (default: len(values)//8)

        Returns:
            Dict with 'times', 'frequencies', 'spectrogram'
        """
        values = np.array(values)

        if len(values) < 8:
            raise ValueError(f"Signal too short ({len(values)} points). Need at least 8 points for spectrogram.")

        if window_size is None:
            # Adaptive window size: ensure valid bounds
            window_size = max(
                self.SPECTROGRAM_MIN_WINDOW,
                min(self.SPECTROGRAM_MAX_WINDOW, len(values) // self.SPECTROGRAM_WINDOW_DIVISOR),
            )

        frequencies, times, spec = signal.spectrogram(values, fs=self.sample_rate, nperseg=window_size)

        return {"times": times, "frequencies": frequencies, "spectrogram": np.abs(spec)}

    def extract_frequency_features(self, values: Union[List[float], List[List[float]]]) -> Dict[str, float]:
        """
        Extract numerical frequency domain features (handles both univariate and multivariate)

        Args:
            values: Time series values (univariate: List[float], multivariate: List[List[float]])

        Returns:
            Dict of frequency domain features
        """
        # NORMALIZE FIRST: Handle wrapped univariate data BEFORE checking multivariate
        # e.g., [[1,2,3]] -> [1,2,3]
        if isinstance(values, list) and len(values) == 1 and isinstance(values[0], list):
            values = values[0]  # Unwrap wrapped univariate data

        # NOW check if multivariate
        is_multivariate = self._is_multivariate(values)

        if is_multivariate:
            # Average signal across channels first, then extract features
            # (consistent with chart Panel 4 - single source of truth)
            avg_signal = np.mean(values, axis=0)
            avg_signal_demeaned = avg_signal - np.mean(avg_signal)
            return self._extract_single_channel_features(avg_signal_demeaned.tolist())

        # Univariate case
        return self._extract_single_channel_features(values)

    def _extract_single_channel_features(self, values: List[float]) -> Dict[str, float]:
        """Extract features for a single channel (internal helper)."""
        fft_result = self.compute_fft(values, normalize=False)
        psd_result = self.compute_power_spectral_density(values)
        dominant = self.detect_dominant_frequencies(values, n_peaks=3)

        frequencies = fft_result["frequencies"]
        magnitude = fft_result["magnitude"]
        psd = psd_result["psd"]

        # Guard against division by zero for zero-magnitude signals
        mag_sum = np.sum(magnitude)
        psd_sum = np.sum(psd)

        # Use small epsilon to prevent division by zero
        mag_sum_safe = mag_sum if mag_sum > 0 else 1e-10
        psd_sum_safe = psd_sum if psd_sum > 0 else 1e-10

        # Calculate spectral centroid first (needed for spectral spread)
        spectral_centroid = float(np.sum(frequencies * magnitude) / mag_sum_safe)

        # Calculate spectral spread with safe denominator
        spectral_spread = float(np.sqrt(np.sum(((frequencies - spectral_centroid) ** 2) * magnitude) / mag_sum_safe))

        # Calculate spectral rolloff with proper error handling
        spectral_rolloff = 0.0
        if len(magnitude) > 0 and mag_sum > 0:
            threshold = self.SPECTRAL_ROLLOFF_THRESHOLD * mag_sum
            rolloff_indices = np.where(np.cumsum(magnitude) >= threshold)[0]
            if len(rolloff_indices) > 0:
                spectral_rolloff = float(frequencies[rolloff_indices[0]])
            else:
                # Fallback: use the maximum frequency if threshold never reached
                # This can happen with very small magnitude values or precision issues
                spectral_rolloff = float(frequencies[-1]) if len(frequencies) > 0 else 0.0

        # Calculate spectral entropy with safe denominator
        psd_normalized = psd / psd_sum_safe
        spectral_entropy = float(-np.sum(psd_normalized * np.log2(psd_normalized + 1e-10)))

        features = {
            "spectral_centroid": spectral_centroid,
            "spectral_spread": spectral_spread,
            "spectral_rolloff": spectral_rolloff,
            "spectral_entropy": spectral_entropy,
            "spectral_flatness": float(np.exp(np.mean(np.log(magnitude + 1e-10))) / (np.mean(magnitude) + 1e-10)),
            "dominant_frequency": dominant[0]["frequency"] if dominant else 0.0,
            "dominant_period": dominant[0]["period"] if dominant else float("inf"),
            "n_dominant_peaks": len(dominant),
            "total_power": float(np.sum(magnitude**2)),
        }

        # Add all dominant frequencies
        for i, dom in enumerate(dominant):
            features[f"peak_{i + 1}_frequency"] = dom["frequency"]
            features[f"peak_{i + 1}_magnitude"] = dom["magnitude"]
            features[f"peak_{i + 1}_period"] = dom["period"]

        return features

    def extract_features(self, values: Union[List[float], List[List[float]]]) -> Dict[str, float]:
        """
        Alias for extract_frequency_features() for convenience.

        Args:
            values: Time series values (univariate or multivariate)

        Returns:
            Dict of frequency domain features
        """
        return self.extract_frequency_features(values)

    def create_frequency_chart(
        self,
        values: Union[List[float], List[List[float]]],
        timestamps: Optional[List[Any]],
        sample_metadata: Optional[Dict[str, Any]],
        chart_type: str = "default",
        title: str = "Frequency Analysis of Time Series",
        figsize: Tuple[int, int] = (12, 10),
        dpi: int = 100,
    ) -> str:
        """
        Create frequency domain visualization (handles both univariate and multivariate data).

        For multivariate data, automatically dispatches to create_multivariate_frequency_chart().
        For univariate data, creates 4-panel layout: FFT, PSD, Spectrogram, Dominant Frequencies.
        These are the SAME metrics used in multivariate panels 1-4 for fair comparison.

        Args:
            values: Time series values (univariate: List[float], multivariate: List[List[float]])
            timestamps: Timestamps for automatic sample rate inference
                        (at least one of timestamps or sample_metadata must be provided)
            sample_metadata: Metadata dict with dataset_name, domain, source, etc.
                            (at least one of timestamps or sample_metadata must be provided)
            chart_type: 'default'/'combined' (4 panels), 'spectrum', 'psd', 'spectrogram'
            title: Chart title (NO ground truth!)
            figsize: Figure size
            dpi: Resolution

        Returns:
            Base64 encoded image string
        """
        # Require at least one of timestamps or sample_metadata for proper inference
        if timestamps is None and sample_metadata is None:
            raise ValueError(
                "At least one of 'timestamps' or 'sample_metadata' must be provided for sample rate inference"
            )

        # Always infer sample rate (required for proper frequency analysis)
        inferred_rate, inference_method = infer_sample_rate(
            timestamps=timestamps,
            dataset_name=sample_metadata.get("dataset_name") if sample_metadata else None,
            domain=sample_metadata.get("domain") if sample_metadata else None,
            source=sample_metadata.get("source") if sample_metadata else None,
            sample_metadata=sample_metadata,
        )
        sample_rate = inferred_rate

        # Determine if frequencies are normalized (cycles per sample) vs absolute (Hz)
        is_normalized = inference_method == "normalized"

        # Get frequency label based on whether normalized or absolute
        freq_label = "Frequency (cycles/sample)" if is_normalized else "Frequency (Hz)"
        freq_unit_note = (
            "Note: Normalized frequencies (cycles per sample). Periods are in samples."
            if is_normalized
            else f"Note: Absolute frequencies in Hz (sampling rate: {sample_rate:.6f} Hz)"
        )

        # Use analyzer with correct sample rate (create new if different from instance)
        analyzer = self if sample_rate == self.sample_rate else FrequencyAnalyzer(sample_rate=sample_rate)

        # NORMALIZE FIRST: Handle wrapped univariate data BEFORE checking multivariate
        # e.g., [[1,2,3]] -> [1,2,3]
        if isinstance(values, list) and len(values) == 1 and isinstance(values[0], list):
            values = values[0]  # Unwrap wrapped univariate data

        # NOW check if multivariate
        is_multivariate = self._is_multivariate(values)

        if is_multivariate:
            # Automatically dispatch to multivariate chart (same pattern as chart_generator)
            if chart_type in ("default", "simple", "combined", "spectrogram"):
                multivariate_chart_type = "combined"
            elif chart_type in ("spectrum", "psd", "separate"):
                multivariate_chart_type = chart_type
            else:
                # Fallback to combined for unknown types
                multivariate_chart_type = "combined"
            return analyzer.create_multivariate_frequency_chart(
                multivariate_data=values,
                variable_names=None,  # Use default generic names
                chart_type=multivariate_chart_type,
                title=title,
                figsize=figsize,
                dpi=dpi,
                is_normalized=is_normalized,
                freq_label=freq_label,
                freq_unit_note=freq_unit_note,
            )

        # Univariate case
        values = np.array(values)
        # Flatten if it's a 2D array with 1 row (single channel)
        if len(values.shape) == 2 and values.shape[0] == 1:
            values = values.flatten()

        # Handle NaN values (for imputation tasks)
        if np.any(np.isnan(values)):
            mask = ~np.isnan(values)
            n_valid = np.sum(mask)
            if n_valid < 4:
                raise ValueError(f"Insufficient non-NaN values ({n_valid}) for frequency analysis. Need at least 4.")
            # Linear interpolation to fill NaN
            indices = np.arange(len(values))
            values = np.interp(indices, indices[mask], values[mask])

        # Check for constant/zero signals
        if len(values) < 4:
            raise ValueError(
                f"Signal too short ({len(values)} points). Need at least 4 points for frequency analysis."
            )
        if np.std(values) == 0 or np.all(values == 0):
            # Constant or zero signal - create chart with message
            fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
            for ax in axes.flat:
                ax.text(
                    0.5,
                    0.5,
                    "Constant/Zero signal: No frequency content",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title("Frequency Analysis")
            fig.suptitle(title, fontsize=14, fontweight="bold")
            plt.tight_layout()
            img_base64 = self._fig_to_base64(fig, dpi=dpi)
            plt.close(fig)
            return img_base64

        # Validate sample rate
        if analyzer.sample_rate <= 0:
            raise ValueError(f"Invalid sample rate: {analyzer.sample_rate}. Must be > 0.")

        if chart_type == "default" or chart_type == "simple" or chart_type == "combined":
            # 4-panel layout: SAME as multivariate panels 1-4 for fair comparison
            fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)

            # Panel 1: FFT Magnitude Spectrum (DC removed, log scale)
            values_demeaned = values - np.mean(values)
            fft_result = analyzer.compute_fft(values_demeaned.tolist())
            axes[0, 0].semilogy(fft_result["frequencies"], fft_result["magnitude"], "b-", linewidth=1.5)
            axes[0, 0].set_xlabel(freq_label)
            axes[0, 0].set_ylabel("Magnitude (log scale)")
            axes[0, 0].set_title("1. FFT Magnitude Spectrum (DC removed)")
            axes[0, 0].set_xlim(0, analyzer.sample_rate / 2)
            axes[0, 0].grid(True, alpha=self.GRID_ALPHA)

            # Panel 2: PSD (Welch)
            psd_result = analyzer.compute_power_spectral_density(values.tolist())
            axes[0, 1].semilogy(psd_result["frequencies"], psd_result["psd"], "r-", linewidth=1.5)
            axes[0, 1].set_xlabel(freq_label)
            axes[0, 1].set_ylabel("Power Spectral Density")
            axes[0, 1].set_title("2. PSD (Welch)")
            axes[0, 1].set_xlim(0, analyzer.sample_rate / 2)
            axes[0, 1].grid(True, alpha=self.GRID_ALPHA)

            # Panel 3: Spectrogram
            spec_result = analyzer.compute_spectrogram(values.tolist())
            im = axes[1, 0].pcolormesh(
                spec_result["times"],
                spec_result["frequencies"],
                10 * np.log10(spec_result["spectrogram"] + 1e-10),
                shading="gouraud",
                cmap="viridis",
            )
            # For normalized frequencies, time axis should be in samples, not seconds
            time_label = "Time (samples)" if is_normalized else "Time (s)"
            axes[1, 0].set_xlabel(time_label)
            axes[1, 0].set_ylabel(freq_label)
            axes[1, 0].set_title("3. Spectrogram")
            plt.colorbar(im, ax=axes[1, 0], label="Power (dB)")

            # Panel 4: Dominant Frequencies (bar chart)
            # Use demeaned values for consistency with Panel 1 (FFT)
            dominant = analyzer.detect_dominant_frequencies(values_demeaned.tolist(), n_peaks=5)
            if dominant:
                freqs = [d["frequency"] for d in dominant]
                mags = [d["normalized_magnitude"] for d in dominant]
                axes[1, 1].bar(range(len(freqs)), mags, color="green", alpha=self.BAR_ALPHA)
                axes[1, 1].set_xlabel(freq_label)
                axes[1, 1].set_ylabel("Normalized Magnitude")
                axes[1, 1].set_title("4. Dominant Frequencies")
                axes[1, 1].set_xticks(range(len(freqs)))
                # Format frequencies with proper units
                freq_labels = [self._format_frequency_with_units(f, is_normalized) for f in freqs]
                axes[1, 1].set_xticklabels(freq_labels, rotation=45)
                axes[1, 1].grid(True, alpha=self.GRID_ALPHA, axis="y")
            else:
                axes[1, 1].text(0.5, 0.5, "No dominant frequencies", ha="center", va="center")
                axes[1, 1].set_title("4. Dominant Frequencies")

            # Add title and subtitle with frequency interpretation note
            fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
            if freq_unit_note:
                fig.text(0.5, 0.99, freq_unit_note, ha="center", va="top", fontsize=12, style="italic")
            plt.tight_layout(rect=[0, 0, 1, 0.96])

        elif chart_type == "spectrum":
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            values_demeaned = values - np.mean(values)
            fft_result = analyzer.compute_fft(values_demeaned.tolist())
            ax.semilogy(fft_result["frequencies"], fft_result["magnitude"], "b-", linewidth=2)
            ax.set_xlabel(freq_label, fontsize=12)
            ax.set_ylabel("Magnitude (log scale)", fontsize=12)
            ax.set_title(f"{title} (DC removed)", fontsize=14, fontweight="bold")
            ax.text(
                0.5,
                0.98,
                freq_unit_note,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=9,
                style="italic",
                color="gray",
            )
            ax.grid(True, alpha=self.GRID_ALPHA)
            plt.tight_layout()

        elif chart_type == "psd":
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            psd_result = analyzer.compute_power_spectral_density(values.tolist())
            ax.semilogy(psd_result["frequencies"], psd_result["psd"], "r-", linewidth=2)
            ax.set_xlabel(freq_label, fontsize=12)
            ax.set_ylabel("Power Spectral Density", fontsize=12)
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.text(
                0.5,
                0.98,
                freq_unit_note,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=9,
                style="italic",
                color="gray",
            )
            ax.grid(True, alpha=self.GRID_ALPHA)
            plt.tight_layout()

        elif chart_type == "spectrogram":
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            spec_result = analyzer.compute_spectrogram(values.tolist())
            im = ax.pcolormesh(
                spec_result["times"],
                spec_result["frequencies"],
                10 * np.log10(spec_result["spectrogram"] + 1e-10),
                shading="gouraud",
                cmap="viridis",
            )
            time_label = "Time (samples)" if is_normalized else "Time (s)"
            ax.set_xlabel(time_label, fontsize=12)
            ax.set_ylabel(freq_label, fontsize=12)
            ax.set_title(title, fontsize=14, fontweight="bold")
            ax.text(
                0.5,
                0.98,
                freq_unit_note,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=9,
                style="italic",
                color="gray",
            )
            plt.colorbar(im, ax=ax, label="Power (dB)")
            plt.tight_layout()

        # Convert to base64
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        plt.close(fig)

        return img_base64

    def create_multivariate_frequency_chart(
        self,
        multivariate_data: Union[List[List[float]], np.ndarray],
        variable_names: Optional[List[str]] = None,
        chart_type: str = "combined",
        title: str = "Multivariate Frequency Analysis",
        figsize: Tuple[int, int] = (14, 10),
        dpi: int = 100,
        is_normalized: bool = False,
        freq_label: str = "Frequency (Hz)",
        freq_unit_note: str = "",
    ) -> str:
        """
        Create PROPER multivariate frequency analysis chart.

        IMPORTANT: Multivariate analysis is DIFFERENT from univariate!
        - Includes Cross-Spectral Density (frequency correlation BETWEEN channels)
        - Includes Coherence (correlation strength per frequency)
        - These metrics are IMPOSSIBLE to compute from univariate data!
        - Does NOT show ground truth labels

        Args:
            multivariate_data: Multi-variate time series data (n_variables x n_timepoints)
            variable_names: Names for each variable (use generic names like "Channel 1")
            chart_type: Type of chart ('combined', 'cross_spectral', 'separate')
            title: Overall figure title (NO ground truth!)
            figsize: Figure size (width, height)
            dpi: Resolution

        Returns:
            Base64 encoded image string
        """
        # Convert to numpy array - expect (n_channels, n_timepoints)
        data = np.array(multivariate_data)

        # Handle 1D data
        if len(data.shape) == 1:
            data = data.reshape(1, -1)

        n_channels = data.shape[0]
        n_timepoints = data.shape[1]

        # Generate GENERIC variable names if not provided (no domain hints!)
        if variable_names is None:
            variable_names = [f"Ch {i + 1}" for i in range(n_channels)]

        if chart_type == "combined":
            # 6-PANEL multivariate frequency analysis (OPTION B)
            # Panels 1-4: SAME metrics as univariate (for fair comparison)
            # Panels 5-6: MULTIVARIATE ONLY (added value)
            fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=dpi)

            colors = plt.cm.tab10(np.linspace(0, 1, n_channels))

            # Panel 1: FFT Magnitude per channel (DC removed, log scale)
            ax1 = axes[0, 0]
            for i, (var_data, var_name, color) in enumerate(zip(data, variable_names, colors)):
                var_data_demeaned = var_data - np.mean(var_data)
                fft_result = self.compute_fft(var_data_demeaned.tolist())
                ax1.semilogy(
                    fft_result["frequencies"],
                    fft_result["magnitude"],
                    color=color,
                    alpha=self.LINE_ALPHA,
                    label=var_name,
                )
            ax1.set_xlabel(freq_label, fontsize=10)
            ax1.set_ylabel("Magnitude (log scale)", fontsize=10)
            ax1.set_title("1. FFT Magnitude per Channel (DC removed)", fontsize=11)
            ax1.legend(loc="upper right", fontsize=9)
            ax1.grid(True, alpha=self.GRID_ALPHA)
            ax1.set_xlim(0, self.sample_rate / 2)

            # Panel 2: PSD per channel (SAME as univariate panel 2)
            ax2 = axes[0, 1]
            for i, (var_data, var_name, color) in enumerate(zip(data, variable_names, colors)):
                f, Pxx = signal.welch(var_data, fs=self.sample_rate, nperseg=min(64, n_timepoints))
                ax2.semilogy(f, Pxx, color=color, alpha=self.LINE_ALPHA, label=var_name)
            ax2.set_xlabel(freq_label, fontsize=10)
            ax2.set_ylabel("Power Spectral Density", fontsize=10)
            ax2.set_title("2. PSD per Channel (Welch)", fontsize=11)
            ax2.legend(loc="upper right", fontsize=9)
            ax2.grid(True, alpha=self.GRID_ALPHA)
            ax2.set_xlim(0, self.sample_rate / 2)

            # Panel 3: Spectrogram (averaged across channels) (SAME as univariate panel 3)
            ax3 = axes[0, 2]
            avg_signal = np.mean(data, axis=0)
            time_label = "Time (samples)" if is_normalized else "Time (s)"
            if len(avg_signal) < 8:
                # Too short for spectrogram - display message instead of empty plot
                ax3.text(
                    0.5,
                    0.5,
                    f"Signal too short\n({len(avg_signal)} points, need ≥8)",
                    ha="center",
                    va="center",
                    transform=ax3.transAxes,
                    fontsize=10,
                )
                ax3.set_xlabel(time_label, fontsize=10)
                ax3.set_ylabel(freq_label, fontsize=10)
                ax3.set_title("3. Spectrogram (Averaged)", fontsize=11)
            else:
                # Use same window size calculation as univariate for consistency
                window_size = max(
                    self.SPECTROGRAM_MIN_WINDOW,
                    min(self.SPECTROGRAM_MAX_WINDOW, len(avg_signal) // self.SPECTROGRAM_WINDOW_DIVISOR),
                )
                f_spec, t_spec, Sxx = signal.spectrogram(avg_signal, fs=self.sample_rate, nperseg=window_size)
                im = ax3.pcolormesh(t_spec, f_spec, 10 * np.log10(Sxx + 1e-10), shading="gouraud", cmap="viridis")
                ax3.set_xlabel(time_label, fontsize=10)
                ax3.set_ylabel(freq_label, fontsize=10)
                ax3.set_title("3. Spectrogram (Averaged)", fontsize=11)
                plt.colorbar(im, ax=ax3, label="Power (dB)")

            # Panel 4: Dominant Frequencies (Averaged) (SAME as univariate panel 4)
            ax4 = axes[1, 0]
            # Average channels first, then find dominant frequencies (consistent with Panel 3)
            avg_signal_demeaned = avg_signal - np.mean(avg_signal)
            dominant = self.detect_dominant_frequencies(avg_signal_demeaned.tolist(), n_peaks=5)
            
            if dominant:
                freqs = [d["frequency"] for d in dominant]
                mags = [d["normalized_magnitude"] for d in dominant]
                ax4.bar(range(len(freqs)), mags, color="green", alpha=self.BAR_ALPHA)
                ax4.set_xticks(range(len(freqs)))
                ax4.set_xticklabels(
                    [self._format_frequency_with_units(f, is_normalized) for f in freqs], rotation=45
                )
            else:
                ax4.text(0.5, 0.5, "No dominant frequencies", ha="center", va="center", transform=ax4.transAxes)
            ax4.set_xlabel(freq_label, fontsize=10)
            ax4.set_ylabel("Normalized Magnitude", fontsize=10)
            ax4.set_title("4. Dominant Frequencies (Averaged)", fontsize=11)
            ax4.grid(True, alpha=self.GRID_ALPHA, axis="y")

            # Panel 5: Cross-Spectral Density (MULTIVARIATE ONLY!)
            ax5 = axes[1, 1]
            if n_channels >= 2:
                pair_colors = ["purple", "brown", "pink", "cyan", "magenta", "olive"]
                pair_idx = 0
                for i in range(n_channels):
                    for j in range(i + 1, n_channels):
                        # Adaptive nperseg for cross-spectral density
                        nperseg = min(self.WELCH_NPERSEG_DEFAULT // 4, n_timepoints)  # Use 64 for multivariate
                        f, Cxy = signal.csd(data[i], data[j], fs=self.sample_rate, nperseg=nperseg)
                        color = pair_colors[pair_idx % len(pair_colors)]
                        ax5.plot(
                            f,
                            np.abs(Cxy),
                            color=color,
                            alpha=self.LINE_ALPHA,
                            label=f"{variable_names[i]}-{variable_names[j]}",
                        )
                        pair_idx += 1
                ax5.set_xlabel(freq_label, fontsize=10)
                ax5.set_ylabel("Cross-Spectral Density", fontsize=10)
                ax5.set_title("5. Cross-Spectral Density", fontsize=10)
                ax5.legend(loc="upper right", fontsize=8)
                ax5.grid(True, alpha=self.GRID_ALPHA)
                ax5.set_xlim(0, self.sample_rate / 2)
            else:
                ax5.text(0.5, 0.5, "Requires 2+ channels", ha="center", va="center", transform=ax5.transAxes)

            # Panel 6: Coherence (MULTIVARIATE ONLY!)
            ax6 = axes[1, 2]
            if n_channels >= 2:
                pair_colors = ["purple", "brown", "pink", "cyan", "magenta", "olive"]
                pair_idx = 0
                for i in range(n_channels):
                    for j in range(i + 1, n_channels):
                        # Adaptive nperseg for coherence
                        nperseg = min(self.WELCH_NPERSEG_DEFAULT // 4, n_timepoints)  # Use 64 for multivariate
                        f, Coh = signal.coherence(data[i], data[j], fs=self.sample_rate, nperseg=nperseg)
                        color = pair_colors[pair_idx % len(pair_colors)]
                        ax6.plot(
                            f,
                            Coh,
                            color=color,
                            alpha=self.LINE_ALPHA,
                            label=f"{variable_names[i]}-{variable_names[j]}",
                        )
                        pair_idx += 1
                ax6.set_xlabel(freq_label, fontsize=10)
                ax6.set_ylabel("Coherence", fontsize=10)
                ax6.set_title("6. Coherence", fontsize=10)
                ax6.legend(loc="upper right", fontsize=8)
                ax6.grid(True, alpha=self.GRID_ALPHA)
                ax6.set_xlim(0, self.sample_rate / 2)
                ax6.set_ylim(0, 1)
            else:
                ax6.text(0.5, 0.5, "Requires 2+ channels", ha="center", va="center", transform=ax6.transAxes)

            fig.suptitle(
                f"{title}\nPanels 1-4: Same as Univariate | Panels 5-6: Multivariate Only",
                fontsize=14,
                fontweight="bold",
                y=1.02,
            )
            if freq_unit_note:
                fig.text(0.5, 0.95, freq_unit_note, ha="center", va="top", fontsize=12, style="italic")
            plt.tight_layout(rect=[0, 0, 1, 0.95])

        elif chart_type == "separate":
            # Legacy: separate subplots per channel (for backward compatibility)
            fig, axes = plt.subplots(n_channels, 2, figsize=figsize, dpi=dpi)

            if n_channels == 1:
                axes = axes.reshape(1, -1)

            for i, (var_data, var_name) in enumerate(zip(data, variable_names)):
                axes[i, 0].plot(var_data, "b-", linewidth=1.5)
                axes[i, 0].set_ylabel("Amplitude", fontsize=10)
                axes[i, 0].set_title(f"{var_name} - Time Domain", fontsize=11)
                axes[i, 0].grid(True, alpha=self.GRID_ALPHA)

                if i == n_channels - 1:
                    axes[i, 0].set_xlabel("Time", fontsize=10)

                var_data_demeaned = var_data - np.mean(var_data)
                fft_result = self.compute_fft(var_data_demeaned.tolist())
                axes[i, 1].semilogy(fft_result["frequencies"], fft_result["magnitude"], "r-", linewidth=1.5)
                axes[i, 1].set_ylabel("Magnitude (log scale)", fontsize=10)
                axes[i, 1].set_title(f"{var_name} - Frequency Domain (DC removed)", fontsize=11)
                axes[i, 1].grid(True, alpha=self.GRID_ALPHA)

                if i == n_channels - 1:
                    axes[i, 1].set_xlabel(freq_label, fontsize=10)

            fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
            if freq_unit_note:
                fig.text(0.5, 0.99, freq_unit_note, ha="center", va="top", fontsize=12, style="italic")
            plt.tight_layout(rect=[0, 0, 1, 0.96])

        elif chart_type == "spectrum":
            # Create subplots for frequency spectrum only
            fig, axes = plt.subplots(n_channels, 1, figsize=figsize, dpi=dpi)

            if n_channels == 1:
                axes = [axes]

            for i, (var_data, var_name) in enumerate(zip(data, variable_names)):
                var_data_demeaned = var_data - np.mean(var_data)
                fft_result = self.compute_fft(var_data_demeaned.tolist())
                axes[i].semilogy(fft_result["frequencies"], fft_result["magnitude"], "b-", linewidth=2)
                axes[i].set_ylabel("Magnitude (log scale)", fontsize=10)
                axes[i].set_title(f"{var_name} (DC removed)", fontsize=11, fontweight="bold")
                axes[i].grid(True, alpha=self.GRID_ALPHA)

                if i == n_channels - 1:
                    axes[i].set_xlabel(freq_label, fontsize=10)

            fig.suptitle(f"{title} - Frequency Spectrum (DC Removed)", fontsize=14, fontweight="bold", y=1.02)
            if freq_unit_note:
                fig.text(0.5, 0.99, freq_unit_note, ha="center", va="top", fontsize=12, style="italic")
            plt.tight_layout(rect=[0, 0, 1, 0.96])

        elif chart_type == "psd":
            # Create subplots for PSD only
            fig, axes = plt.subplots(n_channels, 1, figsize=figsize, dpi=dpi)

            if n_channels == 1:
                axes = [axes]

            for i, (var_data, var_name) in enumerate(zip(data, variable_names)):
                psd_result = self.compute_power_spectral_density(var_data.tolist())
                axes[i].semilogy(psd_result["frequencies"], psd_result["psd"], "r-", linewidth=2)
                axes[i].set_ylabel("PSD", fontsize=10)
                axes[i].set_title(f"{var_name}", fontsize=11, fontweight="bold")
                axes[i].grid(True, alpha=self.GRID_ALPHA)

                if i == n_channels - 1:
                    axes[i].set_xlabel(freq_label, fontsize=10)

            fig.suptitle(f"{title} - Power Spectral Density", fontsize=14, fontweight="bold", y=1.02)
            if freq_unit_note:
                fig.text(0.5, 0.99, freq_unit_note, ha="center", va="top", fontsize=12, style="italic")
            plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Convert to base64
        img_base64 = self._fig_to_base64(fig, dpi=dpi)
        plt.close(fig)

        return img_base64

    def _fig_to_base64(self, fig, dpi: int = 100) -> str:
        """
        Convert matplotlib figure to base64 string (consistent with chart_generator).

        Args:
            fig: Matplotlib figure object
            dpi: Resolution for saving

        Returns:
            Base64 encoded image string
        """
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        return img_base64


def analyze_time_series_frequency(
    values: List[float],
    sample_rate: Optional[float] = None,
    timestamps: Optional[List[Any]] = None,
    sample_metadata: Optional[Dict[str, Any]] = None,
    include_visual: bool = True,
) -> Dict:
    """
    Comprehensive frequency domain analysis with automatic sample rate inference.

    Args:
        values: Time series values
        sample_rate: Sampling rate (Hz). If None, will attempt automatic inference.
        timestamps: Optional timestamps for sample rate inference
        sample_metadata: Optional metadata dict with dataset_name, domain, source, etc.
                        Used for automatic sample rate inference if sample_rate is None.
        include_visual: Whether to generate visual charts

    Returns:
        Dict with numerical features and optional visual charts
    """
    # Infer sample rate if not provided
    inference_method = None
    if sample_rate is None:
        inferred_rate, inference_method = infer_sample_rate(
            timestamps=timestamps,
            dataset_name=sample_metadata.get("dataset_name") if sample_metadata else None,
            domain=sample_metadata.get("domain") if sample_metadata else None,
            source=sample_metadata.get("source") if sample_metadata else None,
            sample_metadata=sample_metadata,
        )
        sample_rate = inferred_rate

    analyzer = FrequencyAnalyzer(sample_rate=sample_rate)

    result = {
        "numerical": analyzer.extract_frequency_features(values),
        "dominant_frequencies": analyzer.detect_dominant_frequencies(values, n_peaks=5),
    }

    # Add sample rate metadata if inference was used
    if inference_method is not None:
        result["sample_rate"] = sample_rate
        result["sample_rate_inference_method"] = inference_method

    if include_visual:
        result["visual"] = {
            "spectrum_chart": analyzer.create_frequency_chart(
                values, timestamps, sample_metadata, chart_type="spectrum"
            ),
            "psd_chart": analyzer.create_frequency_chart(values, timestamps, sample_metadata, chart_type="psd"),
            "spectrogram_chart": analyzer.create_frequency_chart(
                values, timestamps, sample_metadata, chart_type="spectrogram"
            ),
            "combined_chart": analyzer.create_frequency_chart(
                values, timestamps, sample_metadata, chart_type="combined"
            ),
        }

    return result


# Convenience function
def quick_frequency_analysis(
    values: List[float],
    timestamps: Optional[List[Any]] = None,
    sample_metadata: Optional[Dict[str, Any]] = None,
) -> Dict:
    """
    Quick frequency analysis with numerical features only and automatic sample rate inference.

    Args:
        values: Time series values
        timestamps: Optional timestamps for sample rate inference
        sample_metadata: Optional metadata dict with dataset_name, domain, source, etc.

    Returns:
        Dict of frequency domain features
    """
    # Infer sample rate
    inferred_rate, _ = infer_sample_rate(
        timestamps=timestamps,
        dataset_name=sample_metadata.get("dataset_name") if sample_metadata else None,
        domain=sample_metadata.get("domain") if sample_metadata else None,
        source=sample_metadata.get("source") if sample_metadata else None,
        sample_metadata=sample_metadata,
    )
    sample_rate = inferred_rate

    analyzer = FrequencyAnalyzer(sample_rate=sample_rate)
    return analyzer.extract_frequency_features(values)
