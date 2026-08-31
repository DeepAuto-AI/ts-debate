"""
TSQA Loader - Time Series Multi-Task Question Answering Dataset

Paper: "Time-MQA: Time Series Multi-Task Question Answering with Context Enhancement"
Authors: Kong et al., ACL 2025
Dataset: https://huggingface.co/datasets/Time-MQA/TSQA
Paper: https://aclanthology.org/2025.acl-long.1437.pdf
"""

import contextlib
import re
import warnings
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Union

import pandas as pd

from utils.task_config import get_task_instruction
from .utils import format_values_as_data_str

# Root path for TSQA benchmark data
_TSQA_ROOT = Path(__file__).resolve().parent.parent / "benchmarks" / "TSQA"


class TSQALoader:
    """
    Load TSQA dataset

    The CSV files should be located at:
        benchmarks/TSQA/anomaly_detection.csv
        benchmarks/TSQA/classification.csv
        benchmarks/TSQA/forecasting_imputation1.csv
        benchmarks/TSQA/forecasting_imputation2.csv
        benchmarks/TSQA/open_ended_QA.csv
    """

    # Constants
    MIN_TIME_SERIES_LENGTH = 8  # Minimum number of values to consider valid time series

    # Task definitions matching paper statistics
    # Note: forecasting and imputation are MIXED across two files
    TASKS: ClassVar[Dict[str, Dict[str, Any]]] = {
        "anomaly": {
            "files": ["anomaly_detection.csv"],
            "eval_type": "classification",
            "description": "Detect anomalies in time series",
        },
        "classification": {
            "files": ["classification.csv"],
            "eval_type": "classification",
            "description": "Classify activity/state from time series",
        },
        "forecasting": {
            # Both files contain mixed forecasting/imputation - filter by task_type column
            "files": ["forecasting_imputation1.csv", "forecasting_imputation2.csv"],
            "eval_type": "regression",
            "description": "Predict future time series values",
        },
        "imputation": {
            # Both files contain mixed forecasting/imputation - filter by task_type column
            "files": ["forecasting_imputation1.csv", "forecasting_imputation2.csv"],
            "eval_type": "regression",
            "description": "Impute missing values (marked as 'X')",
        },
        "qa": {
            "files": ["open_ended_QA.csv"],
            "eval_type": "qa",
            "description": "Open-ended reasoning questions (MCQ, True/False, Open-ended)",
        },
    }

    # QA format types for open-ended QA task
    QA_FORMATS: ClassVar[List[str]] = ["multiple_choice", "true/false", "open_ended_question"]

    def __init__(self, data_root: Optional[Union[str, Path]] = None):
        """
        Initialize TSQA loader.

        Args:
            data_root: Optional custom path to TSQA data directory.
                       Defaults to benchmarks/TSQA relative to this file.
        """
        self.data_root = Path(data_root) if data_root else _TSQA_ROOT
        self._cache: Dict[str, pd.DataFrame] = {}

        if not self.data_root.exists():
            msg = f"TSQA data directory not found: {self.data_root}"
            raise FileNotFoundError(msg)

    def load_all(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load data for all tasks at once."""
        result = {}
        for task_name in self.TASKS:
            try:
                result[task_name] = self.load_task(task_name)
            except (FileNotFoundError, ValueError) as e:
                warnings.warn(f"Failed to load {task_name}: {e}")
        return result

    def load_task(
        self,
        task_type: str,
        domain_filter: Optional[List[str]] = None,
        qa_format: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Load all samples for a specific task.

        Args:
            task_type: One of 'anomaly', 'classification', 'forecasting', 'imputation', 'qa'
            domain_filter: Optional list of domains to filter by
            qa_format: For 'qa' task, filter by format ('multiple_choice', 'true/false', 'open_ended_question')

        Returns:
            List of sample dictionaries
        """
        if task_type not in self.TASKS:
            msg = f"Unknown task_type '{task_type}'. Must be one of {list(self.TASKS.keys())}"
            raise ValueError(msg)

        task_info = self.TASKS[task_type]

        # Load all CSV files for this task
        dfs = []
        for filename in task_info["files"]:
            if filename not in self._cache:
                csv_path = self.data_root / filename
                if not csv_path.exists():
                    msg = f"CSV file not found: {csv_path}"
                    raise FileNotFoundError(msg)

                # Use Python engine to handle complex CSV with multi-line cells
                df = pd.read_csv(csv_path, engine="python", on_bad_lines="warn")
                self._cache[filename] = df

            dfs.append(self._cache[filename])

        # Concatenate all dataframes
        df = pd.concat(dfs, ignore_index=True)

        # Filter by task_type column for forecasting/imputation (mixed in same files)
        if task_type in ["forecasting", "imputation"]:
            df = df[df["task_type"] == task_type]

        samples = []
        skipped = 0

        for idx, row in df.iterrows():
            try:
                sample = self._parse_row(row, task_type, task_info["eval_type"])

                # Filter by QA format if specified
                if task_type == "qa" and qa_format:
                    sample_format = sample.get("qa_format", "")
                    if sample_format.lower() != qa_format.lower():
                        continue

                # Filter by domain if specified
                if domain_filter:
                    sample_domain = sample.get("domain", "").lower()
                    if sample_domain not in [d.lower() for d in domain_filter]:
                        continue

                samples.append(sample)

            except (ValueError, KeyError) as e:
                skipped += 1
                if skipped <= 3:
                    warnings.warn(f"Skipping row {idx}: {e!s:.100}")

        if skipped > 0:
            warnings.warn(f"Skipped {skipped} rows out of {len(df)} total")

        return samples

    def _parse_row(self, row: pd.Series, task_type: str, eval_type: str) -> Dict[str, Any]:
        """
        Parse a single CSV row into a sample dictionary.

        The QA_list column contains a JSON-like string:
            '"question": "...", "answer": "..."'

        Time series data is embedded in the question text as bracketed arrays.
        """
        # Get domain
        domain = str(row.get("application_domain", "unknown"))

        # Get QA_list content
        qa_list = row.get("QA_list")
        if pd.isna(qa_list) or str(qa_list).strip() == "":
            raise ValueError("QA_list field is empty (dataset quality issue)")

        raw = str(qa_list)

        # Parse question and answer using regex
        # Format: "question": "...", "answer": "..."
        q_match = re.search(r'"question":\s*"(.+?)"\s*,\s*"answer"', raw, re.DOTALL)
        a_match = re.search(r'"answer":\s*"(.+?)"?\s*$', raw, re.DOTALL)

        if not q_match:
            # Try alternative parsing for edge cases
            q_match = re.search(r'"question":\s*"(.+?)(?:"\s*$|",\s*"answer")', raw, re.DOTALL)

        if not q_match:
            raise ValueError(f"Could not parse question from QA_list. Raw text: {raw}")

        question = q_match.group(1)
        answer = a_match.group(1) if a_match else ""

        # Clean up escaped quotes in question/answer
        question = question.replace('\\"', '"').replace("\\n", "\n")
        answer = answer.replace('\\"', '"').replace("\\n", "\n")

        # Extract time series from question text
        # If extraction fails, let ValueError propagate - outer loop will skip the sample
        time_series = self._extract_time_series(raw)

        # Get QA format for open-ended QA task
        qa_format_val = None
        if task_type == "qa":
            qa_format_val = str(row.get("question_format", "")) if "question_format" in row.index else None

        # Extract text context by removing embedded time series from question
        text_context = self._extract_text_context(question)

        sample_dict = {
            # Core fields
            "values": time_series,
            "timestamps": list(range(len(time_series))) if time_series else [],
            # Question: cleaned version (arrays already extracted to "values")
            "question": text_context,
            "answer": answer,
            "ground_truth": self._get_ground_truth(answer, task_type),  # Task-specific format
            # Text modality: same as question (cleaned version)
            "text": text_context,
            # Metadata
            "domain": domain,
            "task_type": task_type,
            "eval_type": eval_type,
            "qa_format": qa_format_val,
            # For compatibility: question IS the prompt in TSQA
            "prompt": question,
            # Raw data for debugging
            "raw_qa_list": raw,
            # Formatted data string
            "data_str": format_values_as_data_str(time_series) if time_series else "",
        }

        # Add filled task_instruction
        task_key = f"tsqa_{task_type}"
        sample_dict["task_instruction"] = get_task_instruction(task_key, sample=sample_dict)

        return sample_dict

    def _extract_time_series(self, raw_text: str) -> List[float]:
        """
        Extract time series data from the QA_list text.

        Time series are embedded as arrays in the question text:
            - Brackets: "The input Time Series are [1.0, 2.0, 3.0, ...]"
            - Parentheses: "data points (1.0, 2.0, 3.0, ...)"

        For imputation tasks, missing values are marked as 'X'.
        """
        time_series = None

        # Find arrays in both brackets [] and parentheses ()
        bracket_matches = list(re.finditer(r"\[([^\]]+)\]", raw_text))
        paren_matches = list(re.finditer(r"\(([^)]+)\)", raw_text))

        all_matches = bracket_matches + paren_matches

        for match in all_matches:
            content = match.group(1)

            # Skip non-numeric content
            if "Source:" in content or "www." in content or "http" in content:
                continue
            # Skip choice markers
            if content.strip() in ["A", "B", "C", "D", "True", "False"]:
                continue

            values = []
            for item in content.split(","):
                item = item.strip().strip("'\"")

                # Handle missing values in imputation task
                if item.upper() == "X":
                    values.append(float("nan"))
                    continue

                # Try to parse as number
                with contextlib.suppress(ValueError):
                    values.append(float(item))

            # Accept if we found at least minimum required values
            if len(values) >= self.MIN_TIME_SERIES_LENGTH:
                time_series = values
                break

        # If array extraction fails, raise error
        if time_series is None:
            raise ValueError(
                f"No valid time series array found in QA_list. "
                f"Expected bracketed array like [1.0, 2.0, ...]. "
                f"Text preview: {raw_text}..."
            )

        return time_series

    def _extract_text_context(self, question: str) -> str:
        """
        Extract text context by removing embedded time series from question.

        The question contains both contextual description AND numerical data.
        For TEXT modality, we want only the context (description, domain info).

        Note: 'X' markers in original data are converted to NaN in numerical values,
        so we update text references to match.
        """
        # Remove bracketed arrays [1.0, 2.0, ...] including 'X' for imputation
        text = re.sub(r"\[[^\]]*\]", "", question)

        # Remove parenthesized arrays (1.0, 2.0, ...) that look like numbers
        # But keep normal parentheses like "(EMG)" or "(20Hz)"
        text = re.sub(r"\([^)]*\d+\.?\d*[^)]*,[^)]*\)", "", text)

        # Clean up multiple spaces and normalize
        text = re.sub(r"\s+", " ", text).strip()

        # Remove dangling phrases like "The input Time Serie is" or "recorded Time Serie is"
        text = re.sub(r"The input Time Series? (is|are)\s*\.?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"The recorded Time Series? (is|are)\s*\.?\s*", "", text, flags=re.IGNORECASE)

        # Update 'X' references to 'NaN' to match numerical data (where 'X' -> float('nan'))
        text = re.sub(r"'X'", "NaN", text)
        text = re.sub(r'"X"', "NaN", text)

        return text.strip()

    def _get_ground_truth(self, answer: str, task_type: str) -> Union[List[float], str]:
        """
        Extract ground truth in the appropriate format for each task type.

        Returns:
            - List[float] for regression tasks (forecasting, imputation)
            - str (label) for classification/anomaly tasks
            - str (full answer) for QA tasks
        """
        if task_type in ["forecasting", "imputation"]:
            # Extract array from: "Based on... [1.0, 2.0, ...]"
            match = re.search(r"\[([^\]]+)\]", answer)
            if not match:
                raise ValueError(
                    f"Forecasting/imputation ground truth missing array. "
                    f"Expected format: [1.0, 2.0, ...]. Answer: {answer}"
                )
            try:
                values = [float(v.strip()) for v in match.group(1).split(",") if v.strip()]
                if not values:
                    raise ValueError(f"Empty array in forecasting/imputation ground truth. Answer: {answer}")
                return values
            except ValueError as e:
                raise ValueError(f"Failed to parse ground truth array: {e}. Answer: {answer}") from e

        if task_type in ["classification", "anomaly"]:
            # Extract label from: "Based on... the activity is Walking."
            # or "Based on... this time series includes Normal Point."
            patterns = [
                r"the (?:activity|answer) is\s+([^.]+)",
                r"includes\s+([^.]+)",
            ]
            for pattern in patterns:
                match = re.search(pattern, answer, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
            return answer  # Fallback to full answer

        # qa
        return answer  # Full answer for NLI comparison

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats: Dict[str, Any] = {
            "data_root": str(self.data_root),
            "tasks": {},
        }

        for name, info in self.TASKS.items():
            files_exist = all((self.data_root / f).exists() for f in info["files"])
            stats["tasks"][name] = {
                "files": info["files"],
                "eval_type": info["eval_type"],
                "description": info["description"],
                "files_exist": files_exist,
                "loaded": all(f in self._cache for f in info["files"]),
            }

        return stats

    def get_qa_format_counts(self) -> Dict[str, int]:
        """Get counts of each QA format in the open-ended QA task."""
        # Load qa task if not cached
        filename = "open_ended_QA.csv"
        if filename not in self._cache:
            _ = self.load_task("qa")

        df = self._cache[filename]
        if "question_format" not in df.columns:
            return {}

        return df["question_format"].value_counts().to_dict()


# Convenience function for quick loading
def load_tsqa(
    task: str,
    data_root: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """
    Convenience function to load TSQA task.

    Args:
        task: Task type ('anomaly', 'classification', 'forecasting', 'imputation', 'qa')
        data_root: Optional custom data directory
        **kwargs: Additional arguments passed to load_task()

    Returns:
        List of sample dictionaries
    """
    loader = TSQALoader(data_root=data_root)
    return loader.load_task(task, **kwargs)
