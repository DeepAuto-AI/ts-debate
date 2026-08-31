"""
MTBench Loader

- Paper: "MTBench: A Multimodal Time Series Benchmark for Temporal Reasoning and Question Answering"
- arXiv: https://arxiv.org/abs/2503.16858
- GitHub: https://github.com/Graph-and-Geometric-Learning/MTBench
- Prompt generators: benchmarks/MTBench/evaluation/finance/meta_prompt.py
                     benchmarks/MTBench/evaluation/weather/meta_prompt.py
- Evaluation utils: benchmarks/MTBench/evaluation/utils.py

Official Sample Counts:
    Finance:
        - aligned_in7days_out1days: 750 samples (trend, forecasting, indicator)
        - aligned_in30days_out7days: 525 samples
        - QAlong: 516 samples (correlation, MCQA)
        - QAshort: 491 samples
    Weather:
        - aligned_in7days_out1days: 1,959 samples
        - aligned_in14days_out3days: 1,959 samples
        - QAlong: 728 samples
        - QAshort: 645 samples
"""

import importlib.util
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from benchmarks.MTBench.evaluation.utils import compute_temperature_trend, get_temperature_diff_max_min
from utils.task_config import get_task_instruction
from .utils import format_values_as_data_str

_MTBENCH_ROOT = Path(__file__).resolve().parent.parent / "benchmarks" / "MTBench"

# Constants for granularity mapping
GRANULARITY_MAP = {
    "aligned_in7days_out1days": "1 hour",
    "aligned_in30days_out7days": "5 minutes",
    "default": "1 hour",
}


def _load_mtbench_module(subpath: str):
    """Load a module from MTBench by absolute path to avoid import conflicts."""
    module_path = _MTBENCH_ROOT / subpath
    spec = importlib.util.spec_from_file_location(subpath.replace("/", ".").replace(".py", ""), module_path)
    if spec is None:
        raise ImportError(f"Failed to create module spec for {module_path}")
    if spec.loader is None:
        raise ImportError(f"Module spec has no loader for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_finance_prompt = _load_mtbench_module("evaluation/finance/meta_prompt.py")
_weather_prompt = _load_mtbench_module("evaluation/weather/meta_prompt.py")

# Official finance prompt generators
finance_bb_metaprompt_generation = _finance_prompt.finance_bb_metaprompt_generation
finance_classification_metaprompt_generation = _finance_prompt.finance_classification_metaprompt_generation
finance_correlation_metaprompt_generation = _finance_prompt.finance_correlation_metaprompt_generation
finance_macd_metaprompt_generation = _finance_prompt.finance_macd_metaprompt_generation
finance_mcqa_metaprompt_generation = _finance_prompt.finance_mcqa_metaprompt_generation
finance_mse_metaprompt_generation = _finance_prompt.finance_mse_metaprompt_generation

# Official weather prompt generators
temperature_forecast_metaprompt_generation = _weather_prompt.temperature_forecast_metaprompt_generation
temperature_indicator_metaprompt_generation = _weather_prompt.temperature_indicator_metaprompt_generation
temperature_trend_metaprompt_generation = _weather_prompt.temperature_trend_metaprompt_generation
weather_mcqa_metaprompt_generation = _weather_prompt.weather_mcqa_metaprompt_generation


class MTBenchDataLoader:
    """Load MTBench dataset."""

    def _add_task_instruction(self, sample: Dict, task_type: str) -> Dict:
        """Add filled task_instruction to sample based on task_type."""
        # Map task_type to task_key
        task_key_map = {
            "trend": f"mtbench_{self.domain}_trend",
            "forecasting": f"mtbench_{self.domain}_forecasting",
            "indicator_macd": f"mtbench_{self.domain}_indicator_macd",
            "indicator_bb": f"mtbench_{self.domain}_indicator_bb",
            "correlation": f"mtbench_{self.domain}_correlation",
            "mcqa": f"mtbench_{self.domain}_mcqa",
        }

        task_key = task_key_map.get(task_type)
        if task_key:
            sample["task_instruction"] = get_task_instruction(task_key, sample=sample)
        return sample

    # Official class labels (from evaluation/finance/meta_prompt.py, evaluation/weather/meta_prompt.py)
    LABELS = {
        "finance": {
            "trend": ["<-4%", "-2% ~ -4%", "-2% ~ +2%", "+2% ~ +4%", ">+4%"],
            "correlation": [
                "Strong Positive Correlation",
                "Moderate Positive Correlation",
                "No Correlation",
                "Moderate Negative Correlation",
                "Strong Negative Correlation",
            ],
            "mcqa": ["A", "B", "C", "D"],
        },
        "weather": {
            "trend": ["increasing", "decreasing", "stable"],
            "mcqa": ["A", "B", "C", "D"],
        },
    }

    def __init__(self, mtbench_path: Optional[str] = None, domain: str = "finance"):
        if domain not in ["finance", "weather"]:
            raise ValueError(f"domain must be 'finance' or 'weather', got '{domain}'")

        self.domain = domain

        if mtbench_path is None:
            self.data_path = _MTBENCH_ROOT / "data" / "processed" / domain
        else:
            self.data_path = Path(mtbench_path)

        # Discover data folders
        self._aligned_folders: Dict[str, Path] = {}
        self._qa_folders: Dict[str, Path] = {}

        if self.data_path.exists():
            for folder in self.data_path.iterdir():
                if folder.is_dir():
                    if folder.name.startswith("aligned"):
                        self._aligned_folders[folder.name] = folder
                    elif folder.name.startswith("QA"):
                        self._qa_folders[folder.name] = folder

        self._cache: Dict[str, List[Dict]] = {}

    def load_all(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load data for all tasks at once.

        Returns:
            Dict mapping task names to list of samples:
            {
                'trend': [...],
                'forecasting': [...],
                'indicator_macd': [...],
                'indicator_bb': [...],  # finance only
                'correlation': [...],   # finance only
                'mcqa': [...],
            }
        """
        result = {}

        # Load aligned data (trend, forecasting, indicator)
        aligned_samples = self._load_all_aligned()
        if aligned_samples:
            result["trend"] = [self._add_trend_label(s) for s in aligned_samples]
            result["forecasting"] = [self._add_forecasting_label(s) for s in aligned_samples]
            result["indicator_macd"] = [self._add_indicator_label(s, "macd") for s in aligned_samples]
            if self.domain == "finance":
                result["indicator_bb"] = [self._add_indicator_label(s, "bb") for s in aligned_samples]

        # Load QA data (correlation, MCQA)
        qa_samples = self._load_all_qa()
        if qa_samples:
            result["mcqa"] = [self._add_mcqa_label(s) for s in qa_samples]
            if self.domain == "finance":
                result["correlation"] = [self._add_correlation_label(s) for s in qa_samples]

        # Filter out samples with mismatched lengths for each task
        for task_name, samples in result.items():
            valid_samples = []
            for s in samples:
                vals_len = len(s.get("values", []))
                ts_len = len(s.get("timestamps", []))
                ind_len = len(s.get("input_indicator", [])) if s.get("input_indicator") else vals_len
                if vals_len == ts_len and (not s.get("input_indicator") or vals_len == ind_len):
                    valid_samples.append(s)
            if len(valid_samples) < len(samples):
                print(f"  ⚠️ {task_name}: Filtered {len(samples) - len(valid_samples)} samples with mismatched lengths")
            result[task_name] = valid_samples

        return result

    def load_task(self, task_type: str, setting: str = "7days") -> List[Dict[str, Any]]:
        """
        Load ALL samples for a specific task.

        Args:
            task_type: 'trend', 'forecasting', 'indicator_macd', 'indicator_bb', 'correlation', 'mcqa'
            setting: '7days' or '30days' for aligned, 'long' or 'short' for QA
        """
        cache_key = f"{task_type}_{setting}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        if task_type in ["trend", "forecasting", "indicator_macd", "indicator_bb"]:
            folder_name = f"aligned_in{setting}_out1days" if "7" in setting else f"aligned_in{setting}_out7days"
            if self.domain == "weather" and "30" in setting:
                folder_name = "aligned_in14days_out3days"
            samples = self._load_aligned_folder(folder_name)

            if task_type == "trend":
                samples = [self._add_trend_label(s) for s in samples]
            elif task_type == "forecasting":
                samples = [self._add_forecasting_label(s) for s in samples]
            elif task_type == "indicator_macd":
                samples = [self._add_indicator_label(s, "macd") for s in samples]
            elif task_type == "indicator_bb":
                samples = [self._add_indicator_label(s, "bb") for s in samples]

        elif task_type in ["correlation", "mcqa"]:
            folder_name = f"QA{setting}"
            samples = self._load_qa_folder(folder_name)

            if task_type == "correlation":
                samples = [self._add_correlation_label(s) for s in samples]
            elif task_type == "mcqa":
                samples = [self._add_mcqa_label(s) for s in samples]
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        # Filter out samples with mismatched lengths
        valid_samples = []
        for s in samples:
            vals_len = len(s.get("values", []))
            ts_len = len(s.get("timestamps", []))
            ind_len = len(s.get("input_indicator", [])) if s.get("input_indicator") else vals_len
            if vals_len == ts_len and (not s.get("input_indicator") or vals_len == ind_len):
                valid_samples.append(s)
        if len(valid_samples) < len(samples):
            print(f"  ⚠️ Filtered {len(samples) - len(valid_samples)} samples with mismatched lengths")
        samples = valid_samples

        self._cache[cache_key] = samples
        return samples

    def _load_all_aligned(self) -> List[Dict[str, Any]]:
        """Load all aligned data from ALL available folders."""
        all_samples = []
        for name in ["aligned_in7days_out1days", "aligned_in30days_out7days", "aligned_in14days_out3days"]:
            if name in self._aligned_folders:
                samples = self._load_aligned_folder(name)
                for s in samples:
                    s['source'] = name
                all_samples.extend(samples)
        return all_samples

    def _load_all_qa(self) -> List[Dict[str, Any]]:
        """Load all QA data from ALL available folders."""
        all_samples = []
        for name in ["QAlong", "QAshort"]:
            if name in self._qa_folders:
                samples = self._load_qa_folder(name)
                for s in samples:
                    s['source'] = name
                all_samples.extend(samples)
        return all_samples

    def _load_aligned_folder(self, folder_name: str) -> List[Dict[str, Any]]:
        """Load all JSON files from an aligned folder."""
        if folder_name not in self._aligned_folders:
            raise FileNotFoundError(f"Folder {folder_name} not found in {self.data_path}")

        folder = self._aligned_folders[folder_name]
        samples = []

        for filepath in sorted(folder.glob("*.json")):
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                values = self._parse_array(data.get("input_window", []))
                timestamps = self._parse_array(data.get("input_timestamps", []))

                if not timestamps and values:
                    timestamps = list(range(len(values)))
                samples.append(
                    {
                        "filename": filepath.name,
                        "filepath": str(filepath),
                        "values": values,
                        "timestamps": timestamps,
                        "output_values": self._parse_array(data.get("output_window", [])),
                        "output_timestamps": self._parse_array(data.get("output_timestamps", [])),
                        "text": data.get("text", ""),
                        "raw_data": data,
                        "domain": self.domain,
                        "source": folder_name,
                        "data_str": format_values_as_data_str(values),
                    }
                )
            except Exception as e:
                warnings.warn(f"Failed to load {filepath}: {e}", UserWarning)

        return samples

    def _load_qa_folder(self, folder_name: str) -> List[Dict[str, Any]]:
        """Load all JSON files from a QA folder."""
        if folder_name not in self._qa_folders:
            raise FileNotFoundError(f"Folder {folder_name} not found in {self.data_path}")

        folder = self._qa_folders[folder_name]
        samples = []

        for filepath in sorted(folder.glob("*.json")):
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                values = self._parse_array(data.get("input_window", []))
                timestamps = self._parse_array(data.get("input_timestamps", []))
                # Fallback to indices if timestamps missing
                if not timestamps and values:
                    timestamps = list(range(len(values)))
                samples.append(
                    {
                        "filename": filepath.name,
                        "filepath": str(filepath),
                        "values": values,
                        "timestamps": timestamps,
                        "output_values": self._parse_array(data.get("output_window", [])),
                        "text": data.get("text", ""),
                        "raw_data": data,
                        "domain": self.domain,
                        "source": folder_name,
                        "data_str": format_values_as_data_str(values),
                    }
                )
            except Exception as e:
                warnings.warn(f"Failed to load {filepath}: {e}", UserWarning)

        return samples

    def _add_trend_label(self, sample: Dict) -> Dict:
        """Add trend classification label."""
        s = sample.copy()
        s["task_type"] = "trend"
        s["options"] = self.LABELS[self.domain]["trend"]

        raw = sample["raw_data"]
        in_values = sample.get("values", [])

        # Validate JSON structure for trend field
        trend = raw.get("trend", {})
        if isinstance(trend, str):
            try:
                trend = json.loads(trend)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in trend field: {e}") from e

        if self.domain == "weather":
            # Use official MTBench function for ground truth computation
            output = sample["output_values"]
            s["ground_truth"] = compute_temperature_trend(in_values, output)

        else:
            # Finance: from data file (trend already parsed above)
            s["ground_truth"] = trend.get("output_bin_label", "")
            if not s["ground_truth"]:
                raise ValueError(
                    f"MTBench Finance trend sample missing output_bin_label. Sample keys: {list(raw.keys())}"
                )

        s["answer"] = s["ground_truth"]
        return self._add_task_instruction(s, "trend")

    def _add_forecasting_label(self, sample: Dict) -> Dict:
        """Add forecasting regression label.

        Uses official prompt generators:
        - Finance: finance_mse_metaprompt_generation
        - Weather: temperature_forecast_metaprompt_generation
        """
        s = sample.copy()
        s["task_type"] = "forecasting"
        s["options"] = []  # Regression
        s["ground_truth"] = sample.get("output_values", [])

        s["answer"] = s["ground_truth"]
        return self._add_task_instruction(s, "forecasting")

    def _add_indicator_label(self, sample: Dict, indicator_type: str) -> Dict:
        """Add indicator prediction label.

        Uses official prompt generators:
        - Finance: finance_macd_metaprompt_generation / finance_bb_metaprompt_generation
        - Weather: temperature_indicator_metaprompt_generation
        """
        s = sample.copy()
        s["task_type"] = f"indicator_{indicator_type}"
        s["options"] = []  # Regression

        raw = sample["raw_data"]
        output = sample["output_values"]

        if self.domain == "finance":
            tech = raw.get("technical", {})
            if isinstance(tech, str):
                try:
                    tech = json.loads(tech)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in technical field: {e}") from e
            if indicator_type == "bb":
                s["ground_truth"] = tech.get("out_upper_bb", [])
                s["input_indicator"] = tech.get("in_upper_bb", [])
            else:  # macd
                s["ground_truth"] = tech.get("out_macd", [])
                s["input_indicator"] = tech.get("in_macd", [])

        else:  # weather
            # Use official MTBench function for ground truth computation
            s["ground_truth"] = get_temperature_diff_max_min(output)

        s["answer"] = s["ground_truth"]

        if indicator_type == "bb":
            return self._add_task_instruction(s, "indicator_bb")
        else:
            return self._add_task_instruction(s, "indicator_macd")

    def _add_correlation_label(self, sample: Dict) -> Dict:
        """Add correlation classification label (finance only).

        Uses official prompt generator: finance_correlation_metaprompt_generation
        """
        s = sample.copy()
        s["task_type"] = "correlation"
        s["options"] = self.LABELS["finance"]["correlation"]
        s["ground_truth"] = sample["raw_data"].get("news_price_correlation", "")
        s["answer"] = s["ground_truth"]

        return self._add_task_instruction(s, "correlation")

    def _add_mcqa_label(self, sample: Dict) -> Dict:
        """Add MCQA label.

        Uses official prompt generators:
        - Finance: finance_mcqa_metaprompt_generation
        - Weather: weather_mcqa_metaprompt_generation
        """
        s = sample.copy()
        s["task_type"] = "mcqa"
        s["options"] = self.LABELS[self.domain]["mcqa"]

        raw = sample["raw_data"]

        if self.domain == "finance":
            mcqa = raw.get("MCQA", {})
            s["ground_truth"] = mcqa.get("answer", "")
            s["question"] = mcqa.get("question", "")

        else:  # weather
            s["ground_truth"] = raw.get("answer", "")
            s["question"] = raw.get("question", "")

        s["answer"] = s["ground_truth"]
        return self._add_task_instruction(s, "mcqa")

    def _parse_array(self, value) -> List[float]:
        """Parse string or list to list of floats."""
        if isinstance(value, str):
            return np.fromstring(value.strip("[]"), sep=" ").tolist()
        return list(value) if value else []

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {"domain": self.domain, "data_path": str(self.data_path), "folders": {}}
        for name, path in {**self._aligned_folders, **self._qa_folders}.items():
            count = len(list(path.glob("*.json")))
            stats["folders"][name] = {"count": count}
        return stats
