"""
Task Configurations for All Benchmarks
This file defines the input/output formats for all tasks across all benchmarks.
"""

import re
from datetime import datetime
from typing import Any, Dict, Optional

from .chart_generator import create_task_aware_chart
from .constants import DATETIME_FORMATS, INPUT_CONTEXT_KEYWORD, TASK_CONFIGS
from .frequency_analyzer import FrequencyAnalyzer


def get_task_instruction(task_key: str, sample: dict) -> str:
    """
    Get task instruction for multimodal debate (data passed via context, not embedded).

    Args:
        task_key: Key from TASK_CONFIGS
        sample: Required sample dictionary from a data loader. Placeholder values
                will be automatically extracted from the sample.

    Returns:
        Short instruction string with placeholders filled

    Example:
        >>> sample = loader.load_task('forecasting')[0]
        >>> # Automatic extraction from sample
        >>> task = get_task_instruction('mtbench_finance_forecasting', sample=sample)
    """
    config = TASK_CONFIGS[task_key]
    instruction = config["task_instruction"]

    # Get all required placeholders from instruction
    required_placeholders = set(re.findall(r"\{(\w+)\}", instruction))

    # Extract placeholder values from sample
    extracted_values = {}
    timestamps = sample.get("timestamps", [])
    output_timestamps = sample.get("output_timestamps", [])
    output_values = sample.get("output_values", [])
    raw_data = sample.get("raw_data", {})

    missing = []
    for placeholder in required_placeholders:
        if placeholder in ("start_datetime", "end_datetime"):
            if not timestamps:
                missing.append(placeholder)
        elif placeholder == "pred_end_datetime":
            if not output_timestamps:
                missing.append(placeholder)
        elif placeholder == "prediction_length":
            if "output_values" not in sample:
                missing.append(placeholder)
        elif placeholder == "sticker":
            if not sample.get("filename"):
                missing.append(placeholder)
        elif placeholder == "news_timestamp":
            if not raw_data or not isinstance(raw_data, dict) or not raw_data.get("published_utc"):
                missing.append(placeholder)
        elif placeholder == "question":
            if not sample.get("question"):
                missing.append(placeholder)
        elif placeholder == "question_with_context":
            if not sample.get("question_with_context") and not sample.get("question"):
                missing.append(placeholder)

    if missing:
        raise ValueError(
            f"Missing placeholders for {task_key}: {sorted(missing)}. "
            f"Available sample keys: {list(sample.keys())}. "
            f"Required placeholders: {sorted(required_placeholders)}"
        )

    # Extract placeholder values from sample (all validations passed)
    extracted_values = {}

    # Extract common placeholders (no fallbacks - must fail if missing)
    for placeholder in required_placeholders:
        if placeholder == "start_datetime":
            try:
                _ = float(timestamps[0])
                start_datetime = datetime.fromtimestamp(timestamps[0])
                extracted_values[placeholder] = start_datetime.strftime(DATETIME_FORMATS[0])
            except Exception:
                extracted_values[placeholder] = timestamps[0]
        elif placeholder == "end_datetime":
            try:
                _ = float(timestamps[-1])
                end_datetime = datetime.fromtimestamp(timestamps[-1])
                extracted_values[placeholder] = end_datetime.strftime(DATETIME_FORMATS[0])
            except Exception:
                extracted_values[placeholder] = timestamps[-1]
        elif placeholder == "pred_end_datetime":
            try:
                _ = float(output_timestamps[-1])
                pred_end_datetime = datetime.fromtimestamp(output_timestamps[-1])
                extracted_values[placeholder] = pred_end_datetime.strftime(DATETIME_FORMATS[0])
            except Exception:
                extracted_values[placeholder] = output_timestamps[-1]
        elif placeholder == "prediction_length":
            extracted_values[placeholder] = len(output_values) if output_values else 0
        elif placeholder == "granularity":
            if timestamps and len(timestamps) > 1:
                if isinstance(timestamps[0], (int, float)):
                    delta = timestamps[1] - timestamps[0] if len(timestamps) > 1 else 3600
                    if delta < 300:
                        extracted_values[placeholder] = "5 minutes"
                    elif delta < 3600:
                        extracted_values[placeholder] = "1 hour"
                    else:
                        extracted_values[placeholder] = "1 hour"
                else:
                    extracted_values[placeholder] = "hourly"
            else:
                extracted_values[placeholder] = "hourly"
        elif placeholder == "next_days":
            prediction_length = len(output_values) if output_values else 0
            extracted_values[placeholder] = prediction_length // 24 if prediction_length >= 24 else 1
        elif placeholder == "past_days":
            values_list = sample.get("values", [])
            extracted_values[placeholder] = len(values_list) // 24 if values_list else 7
        elif placeholder == "sticker":
            filename = sample.get("filename", "")
            if "_" in filename:
                extracted_values[placeholder] = filename.split("_")[-1].replace(".json", "")
            else:
                extracted_values[placeholder] = filename.replace(".json", "")
        elif placeholder == "news_timestamp":
            news_ts = raw_data.get("published_utc", "")
            extracted_values[placeholder] = news_ts
        elif placeholder == "question":
            question = sample.get("question", "")
            extracted_values[placeholder] = question
        elif placeholder == "question_with_context":
            extracted_values[placeholder] = sample.get("question_with_context", sample.get("question", ""))
        elif placeholder == "description_tiny":
            extracted_values[placeholder] = sample.get("description_tiny", "")
        elif placeholder == "metadata":
            # metadata can be a dict or string
            meta = sample.get("metadata", "")
            if isinstance(meta, dict):
                # Format as key: value pairs
                extracted_values[placeholder] = ", ".join(f"{k}: {v}" for k, v in meta.items())
            else:
                extracted_values[placeholder] = str(meta)
        elif placeholder in ("option_a", "option_b", "option_c", "option_d"):
            # Extract from options list
            options = sample.get("options", [])
            option_idx = ord(placeholder[-1]) - ord("a")  # 'a'->0, 'b'->1, etc.
            if option_idx < len(options):
                extracted_values[placeholder] = options[option_idx]
            else:
                extracted_values[placeholder] = ""
        elif placeholder == "qa_format_instruction":
            # Format-specific instructions for TSQA QA task
            qa_format = sample.get("qa_format", "").lower()
            if qa_format == "multiple_choice":
                extracted_values[placeholder] = (
                    "For multiple-choice questions, provide your answer by selecting the correct option "
                    "(e.g., 'A', 'B', 'C', or 'D') and include a brief explanation."
                )
            elif qa_format == "true/false":
                extracted_values[placeholder] = (
                    "For true/false questions, start your answer with 'True' or 'False' followed by a brief explanation."
                )
            elif qa_format == "open_ended_question":
                extracted_values[placeholder] = (
                    "Provide a clear and concise answer in natural language with supporting reasoning."
                )
            else:
                # Default for unknown formats or when qa_format is not available
                extracted_values[placeholder] = "Provide a clear and concise answer in natural language."
        else:
            # Unknown placeholder - should not happen if instruction is correct
            raise ValueError(
                f"Unknown placeholder '{placeholder}' in task instruction for {task_key}. "
                f"Required placeholders: {sorted(required_placeholders)}"
            )
    # Fill in placeholders
    return instruction.format(**extracted_values)


# TASK KEY MAPPING
def get_task_key(benchmark: str, task: str) -> Optional[str]:
    """Map benchmark/task to task_key."""
    mapping = {
        # MTBench Finance
        ("MTBench", "finance/trend"): "mtbench_finance_trend",
        ("MTBench", "finance/forecasting"): "mtbench_finance_forecasting",
        ("MTBench", "finance/indicator_macd"): "mtbench_finance_indicator_macd",
        ("MTBench", "finance/indicator_bb"): "mtbench_finance_indicator_bb",
        ("MTBench", "finance/correlation"): "mtbench_finance_correlation",
        ("MTBench", "finance/mcqa"): "mtbench_finance_mcqa",
        # MTBench Weather
        ("MTBench", "weather/forecasting"): "mtbench_weather_forecasting",
        ("MTBench", "weather/trend"): "mtbench_weather_trend",
        ("MTBench", "weather/indicator_macd"): "mtbench_weather_indicator_macd",
        ("MTBench", "weather/mcqa"): "mtbench_weather_mcqa",
        # TimerBed (6 individual tasks)
        ("TimerBed", "HAR"): "timerbed_har",
        ("TimerBed", "ECG"): "timerbed_ecg",
        ("TimerBed", "CTU"): "timerbed_ctu",
        ("TimerBed", "EMG"): "timerbed_emg",
        ("TimerBed", "RCW"): "timerbed_rcw",
        ("TimerBed", "TEE"): "timerbed_tee",
        # TSQA
        ("TSQA", "classification"): "tsqa_classification",
        ("TSQA", "anomaly"): "tsqa_anomaly",
        ("TSQA", "forecasting"): "tsqa_forecasting",
        ("TSQA", "imputation"): "tsqa_imputation",
        ("TSQA", "qa"): "tsqa_qa",
    }
    return mapping.get((benchmark, task), None)  # All tasks now explicitly mapped


def get_task_type(task_key: str) -> str:
    """
    Get the task type for chart generation and evaluation.

    Returns one of: 'classification', 'regression', 'mcqa', 'qa', 'forecasting', 'imputation'
    """

    config = TASK_CONFIGS[task_key]
    task_type = config.get("task_type", "classification")

    # Map to chart-relevant types
    if "forecast" in task_key:
        return "forecasting"
    if "imputation" in task_key:
        return "imputation"

    return task_type


def fill_real_context(
    task_description: str, context: Dict[str, Any], allow_text: bool = True, allow_number: bool = True
) -> str:
    """
    Replace ###INPUT_CONTEXT### placeholder with actual context based on available modalities.

    Args:
        task_description: Task instruction string with ###INPUT_CONTEXT### placeholder
        context: Context dictionary containing available modalities:
            - text: Textual context (news, descriptions)
            - data_str: Pre-formatted numerical time series data string
            - question: Question text (for QA tasks)
            - options: List of choices/options
            - task_key: Task identifier (e.g., "mtbench_finance_trend") - extracted automatically if not provided
            - benchmark: Benchmark name (e.g., "MTBench", "TSQA", "TimerBed")
            - domain: Domain name (e.g., "finance", "weather", "HAR")
            - task_name: Task name (e.g., "trend", "forecasting", "HAR") - used to derive task_key if needed
            - task_type: Task type (e.g., "classification", "regression", "qa")

    Returns:
        Task description with ###INPUT_CONTEXT### replaced by formatted context
    """
    if INPUT_CONTEXT_KEYWORD not in task_description:
        return task_description

    # Extract task_key from context, or derive it from benchmark/task_name
    task_key = context.get("task_key")
    if not task_key:
        benchmark = context.get("benchmark", "")
        task_name = context.get("task_name", "")
        if benchmark and task_name:
            # Try to derive task_key using get_task_key()
            task_key = get_task_key(benchmark, task_name)
        # If still no task_key, try constructing from benchmark and task_type (for TSQA)
        if not task_key:
            benchmark_raw = context.get("benchmark", "").lower()
            task_type = context.get("task_type", "")
            if benchmark_raw == "tsqa" and task_type:
                task_key = f"tsqa_{task_type}"

    # Get task configuration if task_key found
    task_config = None
    if task_key:
        task_config = TASK_CONFIGS.get(task_key, {})

    # Get benchmark/domain from context or task_config
    benchmark = (context.get("benchmark") or (task_config.get("benchmark", "") if task_config else "")).lower()
    domain = (context.get("domain") or (task_config.get("domain", "") if task_config else "")).lower()
    task_type = context.get("task_type") or (task_config.get("task_type", "") if task_config else "")

    # Extract available modalities
    text_context = context.get("text", "")
    data_str = context.get("data_str", "")
    question = context.get("question", "")
    options = context.get("options", context.get("choices", []))

    # Build context string based on available modalities and task requirements
    context_parts = []

    # 1. NUMERICAL MODALITY: Add time series data
    if data_str and allow_number:
        # Use pre-formatted data_str if available (already formatted by data loader)
        if benchmark == "mtbench" and domain == "finance":
            context_parts.append(f"**Stock Price Data:**\n{data_str}")
        elif benchmark == "mtbench" and domain == "weather":
            context_parts.append(f"**Temperature Data:**\n{data_str}")
        else:
            context_parts.append(f"**Time Series Data:**\n{data_str}")

    # 2. TEXT MODALITY: Add text context (news, descriptions)
    if text_context and allow_text:
        # Format based on benchmark/domain
        if benchmark == "mtbench" and domain == "finance":
            # MTBench Finance: News articles are important
            context_parts.append(f"**News Article:**\n{text_context}")
        elif benchmark == "mtbench" and domain == "weather":
            # MTBench Weather: Weather reports
            context_parts.append(f"**Weather Report:**\n{text_context}")
        elif benchmark == "tsqa":
            if not question or text_context not in question:
                context_parts.append(f"**Context:**\n{text_context}")
        elif benchmark == "timerbed":
            if not question or text_context not in question:
                context_parts.append(f"**Data Description:**\n{text_context}")
        else:
            # Generic text context
            context_parts.append(f"**Text Context:**\n{text_context}")

    # Skip question/choices for benchmarks where they're embedded in the task_instruction:
    # - TimerBed: question in instruction
    if benchmark not in ("timerbed"):
        # 3. QUESTION MODALITY: Add question for QA/MCQA tasks
        if question:
            # TSQA: Question is self-contained, just include it
            context_parts.append(f"**Question:** {question}")

        # 4. OPTIONS: Add choices for classification/MCQA tasks
        if options and isinstance(options, list) and len(options) > 0:
            options_str = ", ".join([f"'{opt}'" for opt in options])
            context_parts.append(f"**Choices:** {options_str}")

    # Combine all context parts
    formatted_context = "\n\n".join(context_parts) if context_parts else ""

    # Replace placeholder
    return task_description.replace(INPUT_CONTEXT_KEYWORD, formatted_context)


# Helper: Prepare context with task-aware charts
freq_analyzer = FrequencyAnalyzer()


def prepare_context(sample, task_type, task_name, benchmark=None, domain=None, task_key=None, include_charts=True):
    """Prepare context with proper charts and metadata for the given task."""

    values = sample["values"]
    timestamps = sample.get("timestamps", [])
    output_timestamps = sample.get("output_timestamps", [])
    prediction_length = len(sample.get("output_values", []))

    # Extract metadata from sample if available
    sample_domain = sample.get("domain", domain)
    sample_task_type = sample.get("task_type", task_type)

    # Prepare sample_metadata for both charts (needed for TimerBed time conversion)
    sample_metadata = {
        "dataset_name": sample.get("dataset_name"),
        "domain": sample_domain or domain,
        "source": sample.get("source"),
    }

    # Extract pre-computed indicator values (e.g., MACD, BB from MTBench)
    indicator_values = sample.get("input_indicator")
    indicator_label = "Indicator"
    if sample_task_type:
        task_lower = sample_task_type.lower()
        if "macd" in task_lower:
            indicator_label = "MACD"
        elif "bb" in task_lower or "bollinger" in task_lower:
            indicator_label = "Bollinger Band"

    time_chart = None
    freq_chart = None

    if include_charts:
        # Task-aware time series chart
        time_chart = create_task_aware_chart(
            values=values,
            task_type=task_type,
            timestamps=timestamps,
            output_timestamps=output_timestamps,
            prediction_length=prediction_length,
            title=task_name,
            sample_metadata=sample_metadata,
            indicator_values=indicator_values,
            indicator_label=indicator_label,
        )

        # Frequency domain chart
        freq_chart = freq_analyzer.create_frequency_chart(values, timestamps, sample_metadata, chart_type="combined")

    return {
        "values": values,
        "timestamps": timestamps,
        # Use data_str from sample if available (properly formatted by data loader),
        # otherwise fallback to str(values) for backward compatibility
        "data_str": sample.get("data_str", str(values)),
        "options": sample.get("options", []),
        "time_series_chart": time_chart,
        "frequency_chart": freq_chart,
        "text": sample.get("text", ""),
        "question": sample.get("question", ""),
        "prediction_length": prediction_length,
        # Technical indicator values (e.g., MACD, BB from MTBench)
        "input_indicator": indicator_values,
        "indicator_label": indicator_label,
        # Metadata for fill_real_context and sample_rate inference
        "dataset_name": sample.get("dataset_name"),  # e.g., 'HAR', 'ECG' for TimerBed
        "source": sample.get("source"),  # e.g., 'aligned_in7days_out1days' for MTBench
        "benchmark": benchmark,  # e.g., 'MTBench', 'TSQA', 'TimerBed'
        "domain": sample_domain or domain,  # e.g., 'finance', 'weather', 'HAR'
        "task_type": sample_task_type,  # e.g., 'classification', 'regression', 'qa'
        "task_name": task_name,  # e.g., 'trend', 'forecasting', 'HAR'
        "task_key": task_key,  # e.g., 'mtbench_finance_trend' (optional, can be derived)
    }
