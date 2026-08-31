"""
Evaluation Metrics for Time Series Benchmarks

- MTBench metrics: benchmarks/MTBench/evaluation/utils.py
  - Original repo: https://github.com/Graph-and-Geometric-Learning/MTBench
  - Paper: arXiv:2503.16858
  - We import and use their official evaluation functions directly.

- TimerBed/VL-Time: Standard sklearn accuracy + weighted F1
  - Original repo: https://github.com/AdityaLab/DeepTime/
  - Paper: "A Picture is Worth A Thousand Numbers" (NAACL 2025), arXiv:2411.06018
  - Official eval.py uses sklearn.metrics.accuracy_score and f1_score(average="weighted")

- TSQA: sklearn standard metrics (no official eval code exists)
  - Paper: "Time-MQA: Time Series Multi-Task Question Answering" (ACL 2025)
"""

import re
from typing import Dict, List, Optional, Union

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# MTBench
# Source: https://github.com/Graph-and-Geometric-Learning/MTBench/blob/main/evaluation/utils.py
from ..benchmarks.MTBench.evaluation.utils import calculate_acc as mtbench_calculate_acc
from ..benchmarks.MTBench.evaluation.utils import calculate_correlation_acc as mtbench_calculate_correlation_acc
from ..benchmarks.MTBench.evaluation.utils import calculate_mape as mtbench_calculate_mape
from ..benchmarks.MTBench.evaluation.utils import calculate_mcqa_acc as mtbench_calculate_mcqa_acc
from ..benchmarks.MTBench.evaluation.weather.meta_prompt import decode_temperature_indicator


def mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values

    Returns:
        MAPE as percentage (0-100)
    """
    y_true, y_pred = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


# NLI-based accuracy for open-ended QA
_NLI_MODEL = "sileod/deberta-v3-base-tasksource-nli"
_nli_pipeline = None


def _get_nli_pipeline():
    """Lazy load NLI pipeline."""
    global _nli_pipeline
    if _nli_pipeline is None:
        from transformers import pipeline

        _nli_pipeline = pipeline("text-classification", model=_NLI_MODEL)
    return _nli_pipeline


def tsqa_nli_accuracy(references: list[str], predictions: list[str]) -> float:
    """
    NLI-based accuracy for open-ended QA.

    Uses DeBERTa-v3 TaskSource NLI model to check if predictions
    entail the reference answers. Order: pred [SEP] ref (prediction as premise).

    Args:
        references: List of ground truth answers
        predictions: List of model predictions

    Returns:
        Proportion of predictions that entail the reference (0.0 to 1.0)
    """
    if len(references) == 0:
        return 0.0

    nli = _get_nli_pipeline()

    correct = 0
    for ref, pred in zip(references, predictions):
        # pred [SEP] ref: check if prediction entails the reference (correct order for QA)
        result = nli(f"{pred} [SEP] {ref}")
        if result[0]["label"].lower() == "entailment":
            correct += 1

    return correct / len(references)


# HELPER FUNCTIONS


def strip_markdown(text: str) -> str:
    """
    Strip markdown formatting from text.

    Handles:
    - Bold: **text** or __text__
    - Italic: *text* or _text_
    - Combined: ***text*** or ___text___

    Args:
        text: Input text potentially containing markdown

    Returns:
        Text with markdown formatting removed
    """
    # Remove bold/italic markers: *, **, ***, _, __, ___
    return re.sub(r"\*{1,3}|_{1,3}", "", text)


def extract_final_answer(response: str) -> str:
    """
    Extract text after 'FINAL ANSWER:' if present.

    This handles multi-agent debate outputs (e.g., ts_debate) where the response
    contains reasoning with multiple label mentions, but the actual answer
    is at the end after 'FINAL ANSWER:'.

    Args:
        response: Model output string

    Returns:
        Text after 'FINAL ANSWER:' if found, otherwise original response
    """
    match = re.search(r"FINAL ANSWER\s*[:\n](.+)", response, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return response


# OUTPUT PARSERS
# Parse LLM outputs according to task-specific formats


def parse_classification_output(response: str, labels: List[str], output_format: str) -> str:
    """
    Parse classification output according to task's output format.
    This follows official MTBench protocols.

    Note: All markdown formatting (**, *, etc.) is stripped from the response
    before parsing and from the final result.
    """
    response = response.strip()
    # Strip markdown from response for cleaner parsing
    response_clean = strip_markdown(response)
    labels = labels or []  # Handle None labels

    # Format: ^^^label^^^ (MTBench Finance Trend - OFFICIAL)
    if output_format == "^^^label^^^":
        # Official MTBench parser - handle spaces between carets (e.g., "^ ^ ^" or "^^^")
        # Pattern matches: ^^^...^^^, ^ ^ ^ ... ^ ^ ^, ^^ ... ^^, etc.
        match = re.search(r"\^[\s\^]*\^[\s\^]*\^(.*?)\^[\s\^]*\^[\s\^]*\^", response)
        if not match:
            # Fallback: try simpler ^+..^+ pattern
            match = re.search(r"\^+\s*(.*?)\s*\^+", response)

        if match:
            extracted = strip_markdown(match.group(1).strip())
            # Strip surrounding quotes and parentheses that models sometimes add
            # e.g., '"-2% ~ +2%"' -> '-2% ~ +2%', "'+2% ~ +4%'" -> '+2% ~ +4%'
            extracted = re.sub(r'^["\'\(\)]+|["\'\(\)]+$', "", extracted).strip()
            # Normalize whitespace: models sometimes add spaces around % and ~
            # e.g., "- 2 % ~ - 4 %" should match "-2% ~ -4%"
            normalized = re.sub(r"\s*%", "%", extracted)  # Remove space before %
            normalized = re.sub(r"([<>+-])\s+(\d)", r"\1\2", normalized)  # "+/- 2" -> "+/-2", "< 4" -> "<4"
            normalized = re.sub(r"\s+", " ", normalized).strip()  # Collapse multiple spaces

            # Try exact match with labels first
            if extracted in labels:
                return extracted
            # Try normalized match
            for label in labels:
                if normalized == label or normalized.replace(" ", "") == label.replace(" ", ""):
                    return label
            # Return normalized if no label match
            return normalized

    # Format: letter_only (A, B, C, D) - MTBench MCQA
    if output_format == "letter_only":
        # Look for standalone letter at end or after "Answer:"
        match = re.search(r"(?:Answer:?\s*)?([A-D])\s*$", response_clean.upper())
        if match:
            return match.group(1)
        # Fallback: any standalone letter
        match = re.search(r"\b([A-D])\b", response_clean.upper())
        if match:
            return match.group(1)

    # Format: single_word (MTBench Weather Trend - OFFICIAL: just one word)
    if output_format == "single_word":
        # Use cleaned response for matching
        response_lower = response_clean.lower().strip()
        for label in labels:
            if label.lower() == response_lower:
                return label
            # Also check if the word is at the end
            if response_lower.endswith(label.lower()):
                return label
            # Or if it's the only word-like thing
            words = re.findall(r"\b\w+\b", response_lower)
            if words and words[-1] == label.lower():
                return label

    # Format: answer_choice_confidence (OFFICIAL TimerBed format)
    # Parses: "Answer Choice: Walking\nConfidence Score: 0.85"
    if output_format == "answer_choice_confidence":
        match = re.search(r"Answer\s*Choice[:\s]*([^\n]+)", response_clean, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            # Try to match with known labels
            for label in labels:
                if label.lower() == answer.lower() or label.lower() in answer.lower():
                    return label
            return answer  # Return as-is if no label match

    # Format: tsqa_sentence (OFFICIAL TSQA format)
    # Parses: "Based on the given information, the activity is Walking."
    # Also handles: "Based on the given information, this time series includes Normal Point."
    if output_format == "tsqa_sentence":
        # Try to extract answer from sentence patterns (use cleaned response)
        # Find ALL matches for each pattern and prefer ones that match known labels
        patterns = [
            r"the (?:activity|answer|classification|result) is\s+([^.]+)",
            r"answer is\s+([^.]+)",
            r"classified as\s+([^.]+)",
            r"(?:time series|this) includes\s+([^.]+)",  # For tsqa_anomaly
        ]
        all_matches = []
        for pattern in patterns:
            # Find ALL matches, not just the first one
            for match in re.finditer(pattern, response_clean, re.IGNORECASE):
                answer = match.group(1).strip()
                all_matches.append((answer, match.start()))

        # First pass: find any match that matches a known label (prefer later matches)
        for answer, _ in reversed(all_matches):
            for label in labels:
                if label.lower() == answer.lower() or label.lower() in answer.lower():
                    return label

        # Second pass: return the last match (most likely the actual answer at end)
        if all_matches:
            return all_matches[-1][0]

    # Format: bymyeyes_answer (OFFICIAL ByMyEyes format)
    # Parses: "<answer>Walking</answer>"
    if output_format == "bymyeyes_answer":
        match = re.search(r"<answer>(.*?)</answer>", response, re.IGNORECASE)
        if match:
            answer = strip_markdown(match.group(1).strip())
            # Try to match with known labels
            for label in labels:
                if label.lower() == answer.lower() or label.lower() in answer.lower():
                    return label
            return answer

    # Format: label_only - find exact match
    if output_format == "label_only":
        response_upper = response_clean.upper()
        for label in labels:
            if label.upper() in response_upper:
                return label

    # Try <answer> tags as fallback (ByMyEyes format might appear in any response)
    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.IGNORECASE)
    if answer_match:
        answer = strip_markdown(answer_match.group(1).strip())
        for label in labels:
            if label.lower() == answer.lower() or label.lower() in answer.lower():
                return label
        return answer

    # Fallback: return first label found (use cleaned response)
    for label in labels:
        if label.lower() in response_clean.lower():
            return label

    # Final fallback: return cleaned response
    return response_clean


def parse_regression_output(response: str, prediction_length: int) -> Optional[List[float]]:
    """
    Parse regression output - extract list of numbers from response.

    Accepts prediction formats (LAST match wins, in order of priority):
    1. "Predicted Prices/Values/MACD/Temperatures:" - explicit prediction marker
    2. Bracket format: [num, num, ...]
    3. Lenient fallback: "any text:" followed by comma-separated numbers (>=2 numbers)

    If none found → return None → caller uses ALL ZEROS baseline.

    Args:
        response: Model output string
        prediction_length: Expected number of predicted values

    Returns:
        List of predicted float values, or None if parsing fails
    """
    if prediction_length <= 0 or not response or not response.strip():
        return None

    # Step 1: Strip markdown
    cleaned = strip_markdown(response)

    predictions = []

    # Step 2: Try "Predicted Prices/Values/MACD/Temperatures" marker (LAST match)
    #
    # IMPORTANT: Keep this marker reasonably strict so we don't accidentally match
    # prose like "predicted MACD values ... 2021-11-18 16:30:00" and start
    # extracting numbers from timestamps/years.
    #
    # Supported examples:
    #   - "Predicted Prices: 1.0, 2.0, ..."
    #   - "### Predicted Prices"  (markdown header; numbers on following lines)
    #   - "Predicted MACD values: 0.1, 0.2, ..." (optional "value(s)")
    #
    # Not supported by this marker (but handled by later fallbacks):
    #   - "Predicted MACD values for the next 78 hours are: ..."
    pred_pattern = (
        r"(?im)^[\s>#\-]*Predicted\s+(?:Prices|Temperatures|Values|MACD)"
        r"(?:\s+(?:value|values))?(?:\s*\([^\n:]*\))?\s*(?::\s*|$)"
    )
    pred_matches = list(re.finditer(pred_pattern, cleaned, re.IGNORECASE))

    for match in reversed(pred_matches):  # Start from LAST match
        # Get content after "Predicted X:"
        start_pos = match.end()
        remaining = cleaned[start_pos:]

        # Extract numbers until we hit a clear break (double newline, new section, etc.)
        numbers = []
        for line in remaining.split("\n"):
            line = line.strip()
            if not line:
                if numbers:  # Empty line after finding numbers = stop
                    break
                continue

            # Extract numbers from this line
            line_numbers = re.findall(r"-?\d+\.?\d*", line)
            if line_numbers:
                numbers.extend(line_numbers)
            elif numbers:  # Non-number line after finding numbers = stop
                break

        if numbers:
            predictions = [float(n) for n in numbers]
            break

    # Step 3: Try bracket format [num, num, ...] - LAST match
    if not predictions:
        bracket_matches = list(re.finditer(r"\[([^\]]+)\]", cleaned))
        for match in reversed(bracket_matches):  # Start from LAST match
            numbers = re.findall(r"-?\d+\.?\d*", match.group(1))
            if numbers:
                predictions = [float(n) for n in numbers]
                break

    # Step 4: Lenient fallback - any "text:" or "text are/is" followed by comma-separated numbers (LAST match)
    # Must have at least 2 numbers to distinguish from random text like "sampled at 256 Hz"
    if not predictions:
        # Pattern: colon OR "are/is" followed by whitespace/newline, then comma-separated numbers (at least 2)
        # This catches:
        #   - "imputed are: 0.37, 0.32, ..."  (colon)
        #   - "imputed are\n0.37, 0.32, ..."  (newline)
        #   - "imputed are 0.37, 0.32, ..."   (space only)
        lenient_pattern = r"(?::|(?:are|is))[\s:]*(-?\d+\.?\d*(?:\s*,\s*-?\d+\.?\d*)+)"
        lenient_matches = list(re.finditer(lenient_pattern, cleaned, re.IGNORECASE))
        for match in reversed(lenient_matches):  # Start from LAST match
            numbers = re.findall(r"-?\d+\.?\d*", match.group(1))
            if len(numbers) >= 2:  # Must have at least 2 numbers
                predictions = [float(n) for n in numbers]
                break

    # Step 5: Last resort - find longest comma-separated number sequence
    # This catches MAD judge output which strips prefix: "343.5, 342.8, 342.2, ..."
    # Safe because: requires comma-separation (not random numbers in prose),
    # takes longest sequence, and count is validated by interpolation.
    if not predictions:
        # Pattern: comma-separated numbers (at least 2 numbers)
        number_seq_pattern = r"-?\d+\.?\d*(?:\s*,\s*-?\d+\.?\d*)+"
        seq_matches = re.findall(number_seq_pattern, cleaned)
        if seq_matches:
            # Take the longest sequence (most likely to be the prediction)
            longest = max(seq_matches, key=len)
            numbers = re.findall(r"-?\d+\.?\d*", longest)
            if len(numbers) >= 2:
                predictions = [float(n) for n in numbers]

    # Step 6: Nothing found → return None (caller uses ALL ZEROS baseline)
    if not predictions:
        return None

    # Interpolate to expected length (official MTBench approach)
    if len(predictions) != prediction_length:
        pred_arr = np.array(predictions)
        predictions = np.interp(
            np.linspace(0, 1, prediction_length), np.linspace(0, 1, len(pred_arr)), pred_arr
        ).tolist()

    return predictions


def parse_weather_indicator_output(response: str) -> Dict[str, float]:
    """
    Parse MTBench weather indicator output.
    OFFICIAL FORMAT: "Highest temperature: X, Lowest temperature: Y, Temperature difference: Z"

    Uses official decode_temperature_indicator from benchmarks/MTBench/evaluation/weather/meta_prompt.py
    Returns dict with 'max', 'min', 'diff' keys.
    """
    # Strip markdown first for cleaner parsing (handles **Highest temperature:** etc.)
    response_clean = strip_markdown(response)

    # 1) Fast path: exact official format (no extra text)
    try:
        highest, lowest, diff = decode_temperature_indicator(response_clean)
        return {"max": float(highest), "min": float(lowest), "diff": float(diff)}
    except (ValueError, AttributeError):
        pass

    # 2) Robust path: find the LAST well-formed indicator triple inside longer CoT outputs.
    # This avoids accidentally reading unrelated numbers like "7*24=168 values" or years in timestamps.
    response_lower = response_clean.lower()
    indicator_pattern = (
        r"highest temperature:\s*(-?\d+\.?\d*)\s*,?\s*"
        r"lowest temperature:\s*(-?\d+\.?\d*)\s*,?\s*"
        r"temperature difference:\s*(-?\d+\.?\d*)"
    )

    matches = list(re.finditer(indicator_pattern, response_lower, re.IGNORECASE))
    for m in reversed(matches):
        candidate = m.group(0).strip()
        try:
            highest, lowest, diff = decode_temperature_indicator(candidate)
            return {"max": float(highest), "min": float(lowest), "diff": float(diff)}
        except ValueError:
            continue

    # 3) Last resort: take the last 3 numbers ONLY if they look like a valid (max, min, diff) triple.
    # Safer than "first 3 numbers anywhere" and avoids most CoT artifacts.
    numbers = [float(n) for n in re.findall(r"-?\d+\.?\d*", response_lower)]
    if len(numbers) >= 3:
        highest, lowest, diff = numbers[-3], numbers[-2], numbers[-1]
        # Basic sanity checks: plausible temperature range + consistent diff
        if all(abs(x) < 200 for x in (highest, lowest, diff)) and round(highest - lowest, 2) == round(diff, 2):
            return {"max": float(highest), "min": float(lowest), "diff": float(diff)}

    return {"max": 0.0, "min": 0.0, "diff": 0.0}


def parse_qa_output(response: str) -> str:
    """Parse QA output - returns cleaned text with markdown stripped."""
    return strip_markdown(response.strip())


# QA SUBTYPE EVALUATION FUNCTIONS
def _extract_true_false(text: str) -> str:
    """
    Extract true/false answer from text, handling markdown formatting.

    Handles formats like:
    - "True" / "False" (at start - official TSQA format)
    - "**True**" / "**False**" (markdown bold)
    - "*True*" / "*False*" (markdown italic)
    - "True." / "False." (with punctuation)
    - "Yes" / "No" (synonyms)
    - "Answer: True" / "Final Answer: False" (CoT methods)
    - True/False anywhere in text (fallback for step-by-step reasoning)
    """
    text = text.strip()
    if not text:
        return ""

    # Strip markdown formatting (*, **, _, __)
    cleaned = re.sub(r"[*_]+", "", text)

    # Get first word and clean punctuation
    words = cleaned.split()
    if not words:
        return ""

    first_word = words[0].lower().rstrip(".,!?:;")

    # Step 1: Check if answer starts with True/False (official TSQA format)
    if first_word in ("true", "yes", "correct", "t", "y"):
        return "true"
    if first_word in ("false", "no", "incorrect", "f", "n"):
        return "false"

    # Step 2: Look for answer markers (CoT methods put answer at end)
    # Patterns like "Answer: True", "Final Answer: False", "Conclusion: True"
    answer_patterns = [
        r"(?:final\s+)?answer[:\s]+\**(true|false)\**",
        r"conclusion[:\s]+\**(true|false)\**",
        r"(?:the\s+)?answer\s+is[:\s]+\**(true|false)\**",
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if match:
            return match.group(1).lower()

    # Step 3: Look for standalone True/False with word boundaries
    # Take the LAST occurrence (most likely to be the final answer in CoT)
    tf_matches = list(re.finditer(r"\b(true|false)\b", cleaned, re.IGNORECASE))
    if tf_matches:
        # Return the last match (final answer in step-by-step reasoning)
        return tf_matches[-1].group(1).lower()

    # Step 4: Check for Yes/No as fallback
    yn_matches = list(re.finditer(r"\b(yes|no)\b", cleaned, re.IGNORECASE))
    if yn_matches:
        last_yn = yn_matches[-1].group(1).lower()
        return "true" if last_yn == "yes" else "false"

    return ""


def compute_true_false_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    """
    Compute accuracy for true/false questions.

    Handles markdown formatting in predictions (e.g., **True**, *False*).

    Args:
        predictions: List of model predictions
        ground_truths: List of ground truth answers (starting with True/False)

    Returns:
        Accuracy as float (0.0 to 1.0)
    """
    if not predictions:
        return 0.0

    correct = 0
    for pred, gt in zip(predictions, ground_truths):
        pred_answer = _extract_true_false(pred)
        gt_answer = _extract_true_false(str(gt))

        if pred_answer and gt_answer and pred_answer == gt_answer:
            correct += 1

    return correct / len(predictions)


def compute_mcq_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    """
    Compute accuracy for multiple choice questions by extracting and comparing letters.

    Args:
        predictions: List of model predictions
        ground_truths: List of ground truth answers (containing A/B/C/D)

    Returns:
        Accuracy as float (0.0 to 1.0)
    """
    if not predictions:
        return 0.0

    correct = 0
    for pred, gt in zip(predictions, ground_truths):
        # Extract letter from prediction (look for A, B, C, D)
        pred_match = re.search(r"\b([A-D])\b", pred.upper())
        pred_letter = pred_match.group(1) if pred_match else None

        # Extract letter from ground truth
        gt_match = re.search(r"\b([A-D])\b", str(gt).upper())
        gt_letter = gt_match.group(1) if gt_match else None

        if pred_letter and gt_letter and pred_letter == gt_letter:
            correct += 1

    return correct / len(predictions)


def compute_aggregate_metrics(
    task_key: str,
    predictions: List[str],
    ground_truths: List[Union[str, List[float], Dict[str, float]]],
    labels: Optional[List[str]] = None,
    qa_formats: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute aggregate metrics over full dataset for paper reporting.

    OFFICIAL SOURCES:
    - MTBench: benchmarks/MTBench/evaluation/utils.py
      - Classification: calculate_acc (substring match), calculate_mcqa_acc, calculate_correlation_acc
      - Regression: MSE, MAE, MAPE (all three reported)
    - TimerBed: benchmarks/TimerBed/LLMs/Method/eval.py
      - Classification: sklearn accuracy_score + f1_score(average="weighted")
    - TSQA: sklearn (exact match for classification, MSE for regression)

    Args:
        task_key: Key from TASK_CONFIGS (e.g., "mtbench_finance_trend")
        predictions: List of model predictions (strings)
        ground_truths: List of ground truth values
        labels: Optional override for class labels
        qa_formats: Optional list of QA format types for TSQA QA task
                   (e.g., "true/false", "multiple_choice", "open_ended_question")

    Returns:
        Dict with all relevant metrics for the task type
    """
    from ..utils.task_config import TASK_CONFIGS

    if task_key not in TASK_CONFIGS:
        available_keys = ", ".join(sorted(TASK_CONFIGS.keys()))
        raise ValueError(
            f"Unknown task_key '{task_key}'. "
            f"Available keys include: {available_keys}... "
            f"(total: {len(TASK_CONFIGS)} keys)"
        )

    # Preprocess: extract FINAL ANSWER if present (for ts_debate and similar methods)
    predictions = [extract_final_answer(p) for p in predictions]

    config = TASK_CONFIGS[task_key]
    task_type = config["task_type"]
    output_format = config["output_format"]
    benchmark = config.get("benchmark", "")
    task_labels = labels or config.get("labels") or []

    result = {}

    # CLASSIFICATION TASKS
    if task_type in ["classification", "anomaly", "mcqa"]:
        # Parse all predictions
        parsed_preds = [parse_classification_output(p, task_labels, output_format) for p in predictions]

        # Use official MTBench functions for MTBench benchmark
        if benchmark == "MTBench":
            # Build result_list in official format
            result_list = [
                {"ground_truth": str(gt), "predict": str(pred)} for gt, pred in zip(ground_truths, parsed_preds)
            ]

            if task_key == "mtbench_finance_correlation":
                # Official: returns exact_accuracy and brief_accuracy
                metric_result = mtbench_calculate_correlation_acc(result_list)
                result["exact_accuracy"] = float(metric_result["exact_accuracy"].rstrip("%")) / 100.0
                result["brief_accuracy"] = float(metric_result["brief_accuracy"].rstrip("%")) / 100.0
                result["accuracy"] = result["exact_accuracy"]
            elif task_type == "mcqa":
                # Official: returns percentage (0-100)
                result["accuracy"] = mtbench_calculate_mcqa_acc(result_list) / 100.0
            else:
                # Official: calculate_acc uses substring match (GT in prediction)
                result["accuracy"] = mtbench_calculate_acc(result_list)

            # Add weighted F1 for MTBench classification tasks (not MCQA)
            if task_key in ["mtbench_finance_trend", "mtbench_finance_correlation", "mtbench_weather_trend"]:
                if task_labels:
                    label_map = {label.lower(): i for i, label in enumerate(task_labels)}
                    y_true = [label_map.get(str(gt).lower(), -1) for gt in ground_truths]
                    y_pred = [label_map.get(str(p).lower(), -1) for p in parsed_preds]
                    result["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted"))

        else:
            # TimerBed use sklearn with exact match
            if task_labels:
                # Labels provided: use sklearn with label mapping
                label_map = {label.lower(): i for i, label in enumerate(task_labels)}
                n_classes = len(task_labels)
                y_true = [label_map.get(str(gt).lower(), -1) for gt in ground_truths]
                y_pred_raw = [label_map.get(str(p).lower(), -1) for p in parsed_preds]

                # Unparseable predictions are counted as WRONG (model failure).
                n_unparseable = 0
                y_pred = []
                for t, p in zip(y_true, y_pred_raw):
                    if p < 0:
                        # Prediction unparseable - assign to a different class (guaranteed wrong)
                        y_pred.append((t + 1) % n_classes if n_classes > 1 else 0)
                        n_unparseable += 1
                    else:
                        y_pred.append(p)

                # Accuracy: sklearn accuracy_score (official TimerBed approach)
                result["accuracy"] = float(accuracy_score(y_true, y_pred))
                # Weighted F1: sklearn f1_score (official TimerBed approach)
                result["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted"))
                result["unparseable_predictions"] = n_unparseable
            else:
                # No labels provided (e.g., TSQA): use direct string comparison
                correct = sum(
                    1
                    for gt, pred in zip(ground_truths, parsed_preds)
                    if str(gt).lower().strip() == str(pred).lower().strip()
                )
                result["accuracy"] = correct / len(predictions) if predictions else 0.0
                # For F1, we need to build labels from data
                unique_labels = list({str(gt).lower().strip() for gt in ground_truths})
                label_map = {label: i for i, label in enumerate(unique_labels)}
                y_true = [label_map.get(str(gt).lower().strip(), 0) for gt in ground_truths]
                y_pred = [label_map.get(str(p).lower().strip(), 0) for p in parsed_preds]
                result["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division="warn"))

            # Report counts for transparency
            result["total_samples"] = len(predictions)

    # REGRESSION TASKS
    elif task_type == "regression":
        # Special case: Weather Indicator (dict with max, min, diff)
        if task_key == "mtbench_weather_indicator_macd":
            # OFFICIAL MTBench approach (indicator_prediction.py lines 137-142, 151-155):
            #   metric_dict["diff"]["mse"].append(mean_squared_error([gt["diff"]], [prediction["diff"]]))
            #   summary_dict[key]["mse"] = np.mean(metric_dict[key]["mse"])
            metric_dict = {
                "diff": {"mse": [], "mae": []},
                "max": {"mse": [], "mae": []},
                "min": {"mse": [], "mae": []},
            }

            for pred, gt in zip(predictions, ground_truths):
                parsed = parse_weather_indicator_output(pred)
                gt_dict = gt if isinstance(gt, dict) else parse_weather_indicator_output(str(gt))

                # Per-sample MSE/MAE for each component (official approach)
                for key in ["diff", "max", "min"]:
                    gt_val = gt_dict.get(key, 0.0)
                    pred_val = parsed.get(key, 0.0)
                    metric_dict[key]["mse"].append(float((gt_val - pred_val) ** 2))
                    metric_dict[key]["mae"].append(float(abs(gt_val - pred_val)))

            # Average across samples per component (official approach - lines 150-155)
            for key in ["diff", "max", "min"]:
                result[f"mse_{key}"] = float(np.mean(metric_dict[key]["mse"])) if metric_dict[key]["mse"] else 0.0
                result[f"mae_{key}"] = float(np.mean(metric_dict[key]["mae"])) if metric_dict[key]["mae"] else 0.0

        else:
            # Standard regression: compute per-sample metrics, then average
            # OFFICIAL MTBench approach (value_prediction.py lines 170-174, 221-230):
            #   mse = np.mean((np.array(output_ts) - np.array(predict_ts)) ** 2)
            #   mae = np.mean(np.abs(np.array(output_ts) - np.array(predict_ts)))
            #   mape = calculate_mape(output_ts, predict_ts)
            #   cumulative_mse.append(mse) ... result["mse"] = np.mean(cumulative_mse)
            sample_mses, sample_maes, sample_mapes = [], [], []
            n_parse_failures = 0

            for pred, gt in zip(predictions, ground_truths):
                gt_vals = np.array(gt if isinstance(gt, list) else [gt], dtype=float)
                pred_vals = parse_regression_output(str(pred), len(gt_vals))

                if pred_vals is not None:
                    pred_arr = np.array(pred_vals)
                else:
                    # Parse failed → ALL ZEROS baseline
                    n_parse_failures += 1
                    pred_arr = np.zeros_like(gt_vals)

                # Per-sample MSE (official: np.mean((output_ts - predict_ts) ** 2))
                sample_mses.append(float(np.mean((gt_vals - pred_arr) ** 2)))

                # Per-sample MAE (official: np.mean(np.abs(output_ts - predict_ts)))
                sample_maes.append(float(np.mean(np.abs(gt_vals - pred_arr))))

                # Per-sample MAPE (reuse official MTBench function)
                sample_mapes.append(float(mtbench_calculate_mape(gt_vals.tolist(), pred_arr.tolist())))

            # Report parse failures for transparency
            result["total_samples"] = len(predictions)
            result["unparseable_predictions"] = n_parse_failures

            # Average across samples (official: np.mean(cumulative_mse))
            result["mse"] = float(np.mean(sample_mses)) if sample_mses else 0.0
            result["mae"] = float(np.mean(sample_maes)) if sample_maes else 0.0
            result["mape"] = float(np.mean(sample_mapes)) if sample_mapes else 0.0

    # QA TASKS
    elif task_type == "qa":
        parsed_preds = [parse_qa_output(p) for p in predictions]
        gt_texts = [str(gt) for gt in ground_truths]

        # If qa_formats provided, evaluate by subtype
        if qa_formats and len(qa_formats) == len(predictions):
            # Group by qa_format subtype
            tf_preds, tf_gts = [], []  # true/false
            mc_preds, mc_gts = [], []  # multiple_choice
            oe_preds, oe_gts = [], []  # open_ended_question

            for pred, gt, fmt in zip(parsed_preds, gt_texts, qa_formats):
                if fmt == "true/false":
                    tf_preds.append(pred)
                    tf_gts.append(gt)
                elif fmt == "multiple_choice":
                    mc_preds.append(pred)
                    mc_gts.append(gt)
                else:  # open_ended_question or unknown
                    oe_preds.append(pred)
                    oe_gts.append(gt)

            # Evaluate each subtype with appropriate metric
            result["total_samples"] = len(predictions)

            if tf_preds:
                result["true_false_accuracy"] = compute_true_false_accuracy(tf_preds, tf_gts)
                result["true_false_n"] = len(tf_preds)

            if mc_preds:
                result["multiple_choice_accuracy"] = compute_mcq_accuracy(mc_preds, mc_gts)
                result["multiple_choice_n"] = len(mc_preds)

            if oe_preds:
                result["open_ended_nli_accuracy"] = tsqa_nli_accuracy(oe_gts, oe_preds)
                result["open_ended_n"] = len(oe_preds)

            # Compute weighted overall QA accuracy across all subtypes
            total_samples = len(tf_preds) + len(mc_preds) + len(oe_preds)
            if total_samples > 0:
                weighted_acc = (
                    result.get("true_false_accuracy", 0) * len(tf_preds)
                    + result.get("multiple_choice_accuracy", 0) * len(mc_preds)
                    + result.get("open_ended_nli_accuracy", 0) * len(oe_preds)
                ) / total_samples
                result["qa_accuracy"] = weighted_acc

        else:
            # No qa_formats provided - fallback to NLI for all
            result["nli_accuracy"] = tsqa_nli_accuracy(gt_texts, parsed_preds)
            result["total_samples"] = len(predictions)

    return result





__all__ = [
    "compute_aggregate_metrics",
    "compute_mcq_accuracy",
    "compute_true_false_accuracy",
    "mape",
    "mtbench_calculate_acc",
    "mtbench_calculate_correlation_acc",
    "mtbench_calculate_mape",
    "mtbench_calculate_mcqa_acc",
    "parse_classification_output",
    "parse_qa_output",
    "parse_regression_output",
    "parse_weather_indicator_output",
    "strip_markdown",
    "tsqa_nli_accuracy",
]
