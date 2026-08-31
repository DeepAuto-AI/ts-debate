"""
Core Experiment Runner

1. Loop through tasks
2. Load N samples per task
3. Loop through each sample with method-specific prepare_context
4. Run run_debate on each sample
5. Collect raw responses + costs (tokens, time)

Results are saved per-sample in directory structure:
    results/{experiment}/run_{id}_{model}/
        └── {method}/{task}/sample_{idx}/
            - result.json (response, ground_truth, tokens, time)
            - context.json (input data without charts)
            - time_chart.png (if visual method)
            - freq_chart.png (if visual method)
            - trace.txt (full debate/reasoning trace)
"""

import base64
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm
from ..utils.cost_monitor import CostMonitor
from ..utils.task_config import prepare_context
from .configs import VISUAL_METHODS
from .methods import create_method, extract_result


def build_display_name(benchmark: str, task_name: str, domain: Optional[str] = None) -> str:
    """
    Build proper display name for prepare_context's task_name parameter.

    Examples:
        - MTBench + finance_trend + finance -> "MTBench Finance Trend"
        - TimerBed + HAR + None -> "TimerBed HAR"
        - TSQA + anomaly + None -> "TSQA Anomaly"
    """
    benchmark_upper = benchmark.upper() if benchmark.lower() in ("mtbench", "tsqa") else benchmark.capitalize()

    if benchmark.lower() == "mtbench":
        # MTBench: "MTBench {Domain} {TaskType}"
        # task_name like "finance_trend" -> extract just the task type "trend"
        parts = task_name.split("_")
        if len(parts) >= 2:
            task_part = "_".join(parts[1:])  # e.g., "trend", "forecasting", "indicator_macd"
        else:
            task_part = task_name
        task_display = task_part.replace("_", " ").title()
        domain_display = domain.capitalize() if domain else ""
        return f"MTBench {domain_display} {task_display}".strip()

    if benchmark.lower() == "timerbed":
        # TimerBed: "TimerBed {TaskName}"
        return f"TimerBed {task_name.upper()}"

    if benchmark.lower() == "tsqa":
        # TSQA: "TSQA {TaskName}"
        return f"TSQA {task_name.capitalize()}"

    # Fallback
    return f"{benchmark_upper} {task_name}"


def validate_sample(sample: Dict, idx: int) -> None:
    """
    Fail-fast validation of sample fields.

    Args:
        sample: Sample dictionary from data loader
        idx: Sample index for error messages

    Raises:
        ValueError: If required fields are missing
    """
    if "values" not in sample:
        raise ValueError(f"Sample {idx} missing required field 'values'. Available keys: {sorted(sample.keys())}")

    gt = sample.get("ground_truth") or sample.get("answer")
    if gt is None:
        raise ValueError(
            f"Sample {idx} missing required field 'ground_truth' or 'answer'. Available keys: {sorted(sample.keys())}"
        )


def save_sample(
    output_dir: Path,
    sample_idx: int,
    response: str,
    ground_truth: Any,
    tokens: Dict[str, int],
    time_seconds: float,
    context: Dict[str, Any],
    task_instruction: str,
    trace: str,
    task_key: str,
    method_name: str,
    run_id: int,
) -> None:
    """
    Save one sample's results to directory.

    Creates:
        - result.json: Lightweight results (response, ground_truth, tokens, time)
        - context.json: Input context without base64 charts
        - time_chart.png: Time-domain chart (if present)
        - freq_chart.png: Frequency chart (if present)
        - trace.txt: Full debate/reasoning trace

    Args:
        output_dir: Base directory for this task
        sample_idx: Sample index
        response: Model response
        ground_truth: Expected answer
        tokens: Token usage dict
        time_seconds: Elapsed time
        context: Full context dict
        task_instruction: Task instruction string
        trace: Debate/reasoning trace
        task_key: Task identifier
        method_name: Method identifier
        run_id: Run number
    """
    sample_dir = output_dir / f"sample_{sample_idx:03d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # 1. result.json (lightweight)
    result = {
        "sample_idx": sample_idx,
        "task_key": task_key,
        "method": method_name,
        "run_id": run_id,
        "response": response,
        "ground_truth": ground_truth,
        "tokens": tokens,
        "time_seconds": time_seconds,
    }
    (sample_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))

    # 2. context.json (without base64 charts)
    ctx_to_save = {k: v for k, v in context.items() if k not in ("time_series_chart", "frequency_chart")}
    ctx_to_save["task_instruction"] = task_instruction
    (sample_dir / "context.json").write_text(json.dumps(ctx_to_save, indent=2, default=str))

    # 3. Charts as PNG files
    if context.get("time_series_chart"):
        try:
            img_data = base64.b64decode(context["time_series_chart"])
            (sample_dir / "time_chart.png").write_bytes(img_data)
        except Exception:
            pass  # Skip if base64 decode fails

    if context.get("frequency_chart"):
        try:
            img_data = base64.b64decode(context["frequency_chart"])
            (sample_dir / "freq_chart.png").write_bytes(img_data)
        except Exception:
            pass  # Skip if base64 decode fails

    # 4. Debate/reasoning trace
    if trace:
        (sample_dir / "trace.txt").write_text(trace)


def prepare_context_for_ablation(
    sample: Dict,
    task_type: str,
    task_name: str,
    benchmark: str,
    domain: str,
    task_key: str,
    ablation_config: Dict,
) -> Dict[str, Any]:
    """
    Prepare context with modality ablation.

    Args:
        sample: Sample from data loader
        task_type: Task type (classification, regression, etc.)
        task_name: Task name
        benchmark: Benchmark name
        domain: Domain name
        task_key: Task key
        ablation_config: Ablation config with use_text, use_numerical, etc.

    Returns:
        Context dict with only enabled modalities
    """
    # First prepare full context
    full_context = prepare_context(
        sample=sample,
        task_type=task_type,
        task_name=task_name,
        benchmark=benchmark,
        domain=domain,
        task_key=task_key,
        include_charts=ablation_config.get("use_time_chart", True) or ablation_config.get("use_freq_chart", True),
    )

    # Filter based on ablation config
    context = {}

    # Always include metadata
    for key in (
        "benchmark",
        "domain",
        "task_type",
        "task_name",
        "task_key",
        "options",
        "question",
        "prediction_length",
    ):
        if key in full_context:
            context[key] = full_context[key]

    # Text modality
    if ablation_config.get("use_text", True):
        if "text" in full_context:
            context["text"] = full_context["text"]

    # Numerical modality
    if ablation_config.get("use_numerical", True):
        if "values" in full_context:
            context["values"] = full_context["values"]
        if "timestamps" in full_context:
            context["timestamps"] = full_context["timestamps"]
        if "data_str" in full_context:
            context["data_str"] = full_context["data_str"]

    # Visual modality - time chart
    if ablation_config.get("use_time_chart", True):
        if "time_series_chart" in full_context:
            context["time_series_chart"] = full_context["time_series_chart"]

    # Visual modality - frequency chart
    if ablation_config.get("use_freq_chart", True):
        if "frequency_chart" in full_context:
            context["frequency_chart"] = full_context["frequency_chart"]

    return context


def run_task(
    method_name: str,
    method_config: Dict[str, Any],
    task_key: str,
    benchmark: str,
    task_name: str,
    domain: str,
    task_type: str,
    samples: List[Dict],
    run_id: int,
    output_base: Path,
    ablation_config: Optional[Dict] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run one method on one task's samples.

    Args:
        method_name: Method identifier
        method_config: Method configuration
        task_key: Task identifier (e.g., mtbench_finance_trend)
        benchmark: Benchmark name (mtbench, timerbed, tsqa)
        task_name: Task name within benchmark
        domain: Domain (finance, weather, etc.)
        task_type: Task type (classification, regression, etc.)
        samples: List of sample dicts
        run_id: Run number (1, 2, 3)
        output_base: Base output directory
        ablation_config: Optional ablation configuration
        verbose: Print progress

    Returns:
        Summary dict with task_key, method, run_id, n_samples, total_tokens, total_time
    """
    # Create output directory (output_base already contains run_{id})
    output_dir = output_base / method_name / task_key
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create monitor and method
    monitor = CostMonitor()
    full_config = {**method_config, "monitor": monitor}

    # Apply ablation config to method config
    if ablation_config:
        full_config.update(ablation_config)

    method_instance = create_method(method_name, full_config, monitor)

    # Determine if method needs charts
    include_charts = method_name in VISUAL_METHODS

    # Build proper display name for prepare_context
    # e.g., "MTBench Finance Trend", "TimerBed HAR", "TSQA Anomaly"
    display_name = build_display_name(benchmark, task_name, domain)

    total_tokens = 0
    total_time = 0.0
    skipped_count = 0

    for idx, sample in tqdm(enumerate(samples), total=len(samples), desc=f"Running {method_name} on {task_key}"):
        # Resume capability: skip already completed samples
        sample_dir = output_dir / f"sample_{idx:03d}"
        result_file = sample_dir / "result.json"
        if result_file.exists():
            skipped_count += 1
            if verbose:
                print(f"  Sample {idx}: already completed, skipping")
            continue

        # Fail-fast: no try/except, let errors propagate immediately
        validate_sample(sample, idx)

        ground_truth = sample.get("ground_truth") or sample.get("answer")

        # Prepare context
        if ablation_config:
            context = prepare_context_for_ablation(
                sample=sample,
                task_type=task_type,
                task_name=display_name,
                benchmark=benchmark,
                domain=domain,
                task_key=task_key,
                ablation_config=ablation_config,
            )
        else:
            context = prepare_context(
                sample=sample,
                task_type=task_type,
                task_name=display_name,
                benchmark=benchmark,
                domain=domain,
                task_key=task_key,
                include_charts=include_charts,
            )

        # Task instruction is pre-filled by data loaders with sample-specific values
        task_instruction = sample["task_instruction"]

        context["qa_format"] = sample.get("qa_format", "")

        # Track tokens before
        tokens_before = monitor.total_tokens
        prompt_before = monitor.total_prompt_tokens
        completion_before = monitor.total_completion_tokens

        # Run method - returns (result, filled_instruction)
        time.sleep(1)
        start = time.time()
        raw_result = method_instance.run_debate(task_instruction, context)
        elapsed = time.time() - start
        time.sleep(1)

        # Extract result (includes filled_instruction from method)
        result = extract_result(method_instance, raw_result, method_name)

        # Compute token delta
        tokens = {
            "prompt": monitor.total_prompt_tokens - prompt_before,
            "completion": monitor.total_completion_tokens - completion_before,
            "total": monitor.total_tokens - tokens_before,
        }

        total_tokens += tokens["total"]
        total_time += elapsed

        # Save AFTER running - use filled_instruction returned by method
        if "all_samples" in context:
            del context["all_samples"]  # don't save all samples
        if "current_sample_idx" in context:
            del context["current_sample_idx"]  # don't save current sample index
        save_sample(
            output_dir=output_dir,
            sample_idx=idx,
            response=result.response,
            ground_truth=ground_truth,
            tokens=tokens,
            time_seconds=elapsed,
            context=context,
            task_instruction=result.filled_instruction,
            trace=result.trace,
            task_key=task_key,
            method_name=method_name,
            run_id=run_id,
        )

        if verbose:
            print(f"  Sample {idx + 1}/{len(samples)}: {elapsed:.2f}s, {tokens['total']} tokens")

    return {
        "task_key": task_key,
        "method": method_name,
        "run_id": run_id,
        "n_samples": len(samples),
        "skipped_samples": skipped_count,
        "processed_samples": len(samples) - skipped_count,
        "total_tokens": total_tokens,
        "total_time": total_time,
    }


def run_single_run(
    method: str,
    eval_sets: Dict[str, Dict],
    output_dir: Path,
    method_config: Dict[str, Any],
    run_id: int,
    ablation_configs: Optional[List[Dict]] = None,
    hyperparams_study: Optional[Dict[str, List]] = None,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run a single independent run for one method across tasks.
    Use different run_id (1, 2, 3) for each independent run.

    Args:
        method: Method name to evaluate
        eval_sets: Dict of {benchmark: {task: EvaluationSet}} from TaskMixer
        output_dir: Output directory for this run (e.g., results/main/run_1/)
        method_config: Base method configuration
        run_id: Run identifier (1, 2, 3, etc.)
        ablation_configs: List of ablation configs (for ablation study)
        hyperparams_study: Dict of param_name -> [values] (for hyperparameter study)
        verbose: Print progress

    Returns:
        List of summary dicts for each task combination
    """
    all_summaries = []

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"RUN {run_id}")
        print(f"{'=' * 60}")

    for benchmark, tasks in eval_sets.items():
        for task_name, eval_set in tasks.items():
            # Construct task_key
            if benchmark == "timerbed":
                task_key = f"timerbed_{task_name.lower()}"
            elif benchmark == "mtbench":
                task_key = f"mtbench_{task_name}"
            else:
                task_key = f"{benchmark}_{task_name}"

            # Get task metadata from first sample
            sample0 = eval_set.samples[0]
            domain = sample0["domain"]
            task_type = sample0["task_type"]

            if verbose:
                print(f"\n[Run {run_id}] {method} on {task_key} ({len(eval_set.samples)} samples)")
            else:
                # Main experiment
                summary = run_task(
                    method_name=method,
                    method_config=method_config,
                    task_key=task_key,
                    benchmark=benchmark,
                    task_name=task_name,
                    domain=domain,
                    task_type=task_type,
                    samples=eval_set.samples,
                    run_id=run_id,
                    output_base=output_dir,
                    verbose=verbose,
                )
                all_summaries.append(summary)

    # Save run summary
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(all_summaries, indent=2, default=str))

    return all_summaries
