"""
Sample Rate Inference for Time Series Frequency Analysis

Provides hybrid approach to infer sample rates for frequency domain analysis:
1. Known dataset rates (TimerBed, MTBench)
2. Domain-based lookup (MTBench)
3. Timestamp inference (for real datetime timestamps)
4. Normalized frequency fallback (for unknown data)
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .constants import DATETIME_FORMATS

# Known sample rates lookup table
KNOWN_SAMPLE_RATES = {
    # TimerBed datasets (from DATA_DESCRIPTIONS in timerbed_loader.py)
    "CTU": 1.0 / 120,  # 1 sample per 2 minutes = 0.00833 Hz (24 hours, readings every 2 minutes)
    "ECG": 100.0,  # 100 Hz
    "EMG": 4000.0,  # 4K Hz = 4000 Hz
    "HAR": 50.0,  # 50 Hz
    "RCW": 2000.0,  # 2kHz = 2000 Hz
    "TEE": 50_000_000.0,  # 50 MHz = 50,000,000 Hz
    # MTBench domains
    "finance": {
        "aligned_in7days_out1days": 1.0 / 3600,  # hourly: 1 sample/hour
        "aligned_in30days_out7days": 1.0 / 300,  # 5-minute: 1 sample/5min
        "default": 1.0 / 3600,  # fallback: hourly
    },
    "weather": {
        "aligned_in7days_out1days": 1.0 / 3600,  # hourly
        "aligned_in14days_out3days": 1.0 / 3600,  # hourly
        "default": 1.0 / 3600,  # hourly
    },
}


def infer_sample_rate(
    timestamps: Optional[List[Any]] = None,
    dataset_name: Optional[str] = None,
    domain: Optional[str] = None,
    source: Optional[str] = None,
    sample_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[float, str]:
    """
    Infer sample rate using hybrid approach with multiple fallback strategies.

    Priority order:
    1. Known dataset rates (TimerBed: dataset_name)
    2. Domain-based lookup (MTBench: domain + source)
    3. Timestamp inference (for real datetime timestamps)
    4. Normalized frequency fallback (1.0 Hz, interpret as cycles per sample)

    Args:
        timestamps: List of timestamps (can be indices, datetime objects, or strings)
        dataset_name: Dataset name (e.g., "ECG", "HAR" for TimerBed)
        domain: Domain name (e.g., "finance", "weather" for MTBench)
        source: Source/setting name (e.g., "aligned_in7days_out1days" for MTBench)
        sample_metadata: Optional dict with additional metadata

    Returns:
        Tuple of (sample_rate, inference_method):
        - sample_rate: Inferred sample rate in Hz
        - inference_method: One of:
            - "known_dataset": From KNOWN_SAMPLE_RATES lookup
            - "domain_lookup": From domain-based lookup
            - "timestamp_inference": Inferred from datetime timestamps
            - "normalized": Fallback to normalized frequency (1.0 Hz)
    """
    # Extract metadata from sample_metadata if provided
    if sample_metadata:
        dataset_name = dataset_name or sample_metadata.get("dataset_name")
        domain = domain or sample_metadata.get("domain")
        source = source or sample_metadata.get("source")

    # Priority 1: Known dataset rates (TimerBed)
    if dataset_name and dataset_name in KNOWN_SAMPLE_RATES:
        rate = KNOWN_SAMPLE_RATES[dataset_name]
        # TimerBed rates are always floats, not dicts
        if isinstance(rate, (int, float)):
            return float(rate), "known_dataset"

    # Priority 2: Domain-based lookup (MTBench)
    if domain and domain in KNOWN_SAMPLE_RATES:
        domain_rates = KNOWN_SAMPLE_RATES[domain]
        if isinstance(domain_rates, dict):
            rate = None
            if source:
                # Try exact source match first
                rate = domain_rates.get(source)
            if not rate:
                # Fallback to default for domain
                rate = domain_rates.get("default", 1.0)
            return rate, "domain_lookup"

    # Priority 3: Timestamp inference (for real datetime timestamps)
    if timestamps and len(timestamps) > 1:
        inferred_rate = _infer_from_timestamps(timestamps)
        if inferred_rate and inferred_rate > 0:
            return inferred_rate, "timestamp_inference"

    # Priority 4: Normalized frequency (fallback)
    return 1.0, "normalized"


def _infer_from_timestamps(timestamps: List[Any]) -> Optional[float]:
    """
    Infer sample rate from datetime timestamps.

    Args:
        timestamps: List of timestamps (datetime objects, strings, or indices)

    Returns:
        Sample rate in Hz, or None if inference failed
    """
    if not timestamps or len(timestamps) < 2:
        return None

    # Check if timestamps are datetime objects or parseable strings
    first_ts_raw = timestamps[0]
    last_ts_raw = timestamps[-1]

    # Try to parse as datetime if strings
    try:
        first_ts: Optional[datetime] = None
        last_ts: Optional[datetime] = None

        if isinstance(first_ts_raw, datetime):
            first_ts = first_ts_raw
            last_ts = last_ts_raw if isinstance(last_ts_raw, datetime) else None
        elif isinstance(first_ts_raw, str) and isinstance(last_ts_raw, str):
            # Try common datetime formats
            for fmt in DATETIME_FORMATS:
                try:
                    first_ts = datetime.strptime(first_ts_raw, fmt)
                    last_ts = datetime.strptime(last_ts_raw, fmt)
                    break
                except ValueError:
                    continue

        # Check if we have datetime objects now
        if (
            first_ts is None
            or last_ts is None
            or not isinstance(first_ts, datetime)
            or not isinstance(last_ts, datetime)
        ):
            return None

        # Calculate time span
        time_span = (last_ts - first_ts).total_seconds()
        n_samples = len(timestamps)

        if time_span > 0 and n_samples > 1:
            # Average interval between samples
            avg_interval_seconds = time_span / (n_samples - 1)
            # Sample rate = 1 / interval
            return 1.0 / avg_interval_seconds

    except (ValueError, TypeError, AttributeError):
        # Timestamps are not datetime objects or parseable strings
        # Likely just indices (0, 1, 2, ...)
        return None

    return None


def get_frequency_interpretation(sample_rate: float, inference_method: str) -> str:
    """
    Get human-readable interpretation of frequency analysis.

    Args:
        sample_rate: Sample rate in Hz
        inference_method: How sample_rate was inferred

    Returns:
        Interpretation string
    """
    interpretation_map = {
        "normalized": (
            "normalized frequency (cycles per sample). "
            "Frequencies should be interpreted as relative patterns, "
            "and periods are in number of samples."
        ),
        "known_dataset": f"absolute frequency in Hz (sampling rate: {sample_rate} Hz from known dataset)",
        "domain_lookup": f"absolute frequency in Hz (sampling rate: {sample_rate} Hz from domain lookup)",
        "timestamp_inference": (
            f"absolute frequency in Hz (sampling rate: {sample_rate:.6f} Hz inferred from timestamps)"
        ),
    }
    return interpretation_map.get(inference_method, f"absolute frequency in Hz (sampling rate: {sample_rate} Hz)")
