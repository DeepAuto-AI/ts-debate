"""
Task Mixer: Sampling Strategies for Time Series Benchmark Evaluation

This module provides sampling strategies for creating evaluation subsets
from the three time series benchmarks: MTBench, TimerBed, and TSQA.

Sampling Strategies:
    1. Stratified: Class/domain-balanced sampling (for QA tasks)
    2. Difficulty-Stratified: Uniform across difficulty levels (for regression)
    3. Hybrid: Class-balanced + difficulty-diverse (for classification)

Default Strategy per Task Type:
    - Classification (TimerBed, TSQA classification): HYBRID
    - Regression (forecasting, imputation): DIFFICULTY_STRATIFIED
    - QA tasks (TSQA qa, MTBench mcqa): STRATIFIED
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class SamplingStrategy(Enum):
    """Available sampling strategies."""

    RANDOM = "random"
    STRATIFIED = "stratified"
    DIFFICULTY_STRATIFIED = "difficulty_stratified"
    HYBRID = "hybrid"


@dataclass
class SamplingConfig:
    """Configuration for a sampling task."""

    n_samples: int = 100
    strategy: SamplingStrategy = SamplingStrategy.STRATIFIED
    stratify_by: List[str] = field(default_factory=list)
    difficulty_key: Optional[str] = None
    n_difficulty_buckets: int = 5
    min_per_stratum: int = 3


@dataclass
class EvaluationSet:
    """Container for evaluation samples with metadata."""

    samples: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    config: SamplingConfig
    benchmark: str
    task: str

    def __len__(self) -> int:
        return len(self.samples)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "task": self.task,
            "n_samples": len(self.samples),
            "strategy": self.config.strategy.value,
            "metadata": self.metadata,
        }


class Sampler:
    """Base class for sampling strategies."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def sample(
        self, samples: List[Dict[str, Any]], config: SamplingConfig
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Sample from the given samples according to config."""
        raise NotImplementedError


class RandomSampler(Sampler):
    """Simple random sampling."""

    def sample(
        self, samples: List[Dict[str, Any]], config: SamplingConfig
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        n = min(config.n_samples, len(samples))
        indices = self.rng.choice(len(samples), n, replace=False)
        sampled = [samples[i] for i in indices]
        metadata = {
            "strategy": "random",
            "n_requested": config.n_samples,
            "n_sampled": len(sampled),
            "n_total": len(samples),
        }
        return sampled, metadata


class StratifiedSampler(Sampler):
    """Stratified random sampling with class balance."""

    def sample(
        self, samples: List[Dict[str, Any]], config: SamplingConfig
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not config.stratify_by or len(samples) <= config.n_samples:
            return RandomSampler(self.seed).sample(samples, config)

        df = pd.DataFrame(samples)

        # Check if stratify columns exist
        valid_cols = [c for c in config.stratify_by if c in df.columns]
        if not valid_cols:
            warnings.warn(f"Stratify columns {config.stratify_by} not found. Falling back to random.")
            return RandomSampler(self.seed).sample(samples, config)

        # Create stratification key
        df["_stratum"] = df[valid_cols].astype(str).agg("|".join, axis=1)

        # Count per stratum
        stratum_counts = df["_stratum"].value_counts()
        n_strata = len(stratum_counts)

        # Just use random sampling instead
        if config.n_samples < n_strata:
            return RandomSampler(self.seed).sample(samples, config)

        # Calculate samples per stratum (proportional with minimum guarantee)
        # Adjust min_per_stratum if n_samples is small
        effective_min = min(config.min_per_stratum, config.n_samples // n_strata)
        samples_per_stratum = {}
        remaining = config.n_samples

        for stratum, count in stratum_counts.items():
            prop_n = max(effective_min, int(config.n_samples * count / len(df)))
            samples_per_stratum[stratum] = min(prop_n, count, remaining)
            remaining -= samples_per_stratum[stratum]
            if remaining <= 0:
                break

        # Sample from each stratum
        sampled_indices = []
        stratum_details = {}

        for stratum, stratum_n in samples_per_stratum.items():
            stratum_indices = df[df["_stratum"] == stratum].index.tolist()
            selected = self.rng.choice(stratum_indices, stratum_n, replace=False)
            sampled_indices.extend(selected)
            stratum_details[stratum] = {"n_sampled": stratum_n, "n_total": len(stratum_indices)}

        sampled = [samples[i] for i in sampled_indices[: config.n_samples]]

        metadata = {
            "strategy": "stratified",
            "stratify_by": valid_cols,
            "n_strata": n_strata,
            "n_requested": config.n_samples,
            "n_sampled": len(sampled),
            "n_total": len(samples),
            "stratum_details": stratum_details,
        }

        return sampled, metadata


class DifficultyStratifiedSampler(Sampler):
    """Sample uniformly across difficulty levels."""

    def sample(
        self, samples: List[Dict[str, Any]], config: SamplingConfig
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not config.difficulty_key or len(samples) <= config.n_samples:
            return RandomSampler(self.seed).sample(samples, config)

        df = pd.DataFrame(samples)

        if config.difficulty_key not in df.columns:
            warnings.warn(f"Difficulty key '{config.difficulty_key}' not found. Falling back to random.")
            return RandomSampler(self.seed).sample(samples, config)

        # Handle non-numeric difficulty values
        diff_values = df[config.difficulty_key]
        if not np.issubdtype(diff_values.dtype, np.number):
            # Try to compute a numeric proxy (e.g., length of lists)
            try:
                diff_values = diff_values.apply(lambda x: len(x) if hasattr(x, "__len__") else hash(str(x)) % 1000)
            except Exception:
                return StratifiedSampler(self.seed).sample(samples, config)

        df["_difficulty_value"] = diff_values

        # Create difficulty buckets (quantiles)
        try:
            df["_difficulty_bucket"] = pd.qcut(
                df["_difficulty_value"],
                q=config.n_difficulty_buckets,
                labels=False,
                duplicates="drop",
            )
        except ValueError:
            # Not enough unique values for requested buckets
            df["_difficulty_bucket"] = pd.cut(
                df["_difficulty_value"],
                bins=min(config.n_difficulty_buckets, df["_difficulty_value"].nunique()),
                labels=False,
            )

        actual_buckets = df["_difficulty_bucket"].dropna().nunique()
        samples_per_bucket = config.n_samples // max(actual_buckets, 1)

        # Otherwise samples_per_bucket = 0 and we'd return 0 samples!
        if samples_per_bucket == 0:
            return RandomSampler(self.seed).sample(samples, config)

        sampled_indices = []
        bucket_details = {}

        for bucket in df["_difficulty_bucket"].dropna().unique():
            bucket_df = df[df["_difficulty_bucket"] == bucket]
            k = min(samples_per_bucket, len(bucket_df))
            selected = self.rng.choice(bucket_df.index.tolist(), k, replace=False)
            sampled_indices.extend(selected)
            bucket_details[int(bucket)] = {
                "n_sampled": k,
                "n_total": len(bucket_df),
                "difficulty_range": [
                    float(bucket_df["_difficulty_value"].min()),
                    float(bucket_df["_difficulty_value"].max()),
                ],
            }

        sampled = [samples[i] for i in sampled_indices[: config.n_samples]]

        metadata = {
            "strategy": "difficulty_stratified",
            "difficulty_key": config.difficulty_key,
            "n_buckets": actual_buckets,
            "n_requested": config.n_samples,
            "n_sampled": len(sampled),
            "n_total": len(samples),
            "bucket_details": bucket_details,
        }

        return sampled, metadata


class HybridSampler(Sampler):
    """Class-balanced + difficulty-diverse sampling."""

    def sample(
        self, samples: List[Dict[str, Any]], config: SamplingConfig
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if len(samples) <= config.n_samples:
            return RandomSampler(self.seed).sample(samples, config)

        df = pd.DataFrame(samples)

        # Check for class column
        class_col = None
        for col in config.stratify_by or []:
            if col in df.columns:
                class_col = col
                break

        if not class_col:
            # Fall back to difficulty stratified
            return DifficultyStratifiedSampler(self.seed).sample(samples, config)

        classes = df[class_col].unique()
        n_per_class = config.n_samples // len(classes)

        # Otherwise n_per_class = 0 and we'd return 0 samples!
        if n_per_class == 0:
            return RandomSampler(self.seed).sample(samples, config)

        # Check for difficulty key
        diff_key = config.difficulty_key
        if diff_key and diff_key in df.columns:
            diff_values = df[diff_key]
            if not np.issubdtype(diff_values.dtype, np.number):
                try:
                    diff_values = diff_values.apply(lambda x: len(x) if hasattr(x, "__len__") else 0)
                except Exception:
                    diff_key = None
            df["_difficulty"] = diff_values
        else:
            diff_key = None

        sampled_indices = []
        class_details = {}

        for cls in classes:
            class_df = df[df[class_col] == cls]

            if len(class_df) <= n_per_class:
                sampled_indices.extend(class_df.index.tolist())
                class_details[str(cls)] = {
                    "n_sampled": len(class_df),
                    "n_total": len(class_df),
                    "method": "full",
                }
            elif diff_key:
                # Sample across difficulty within this class
                try:
                    class_df = class_df.copy()
                    class_df["_diff_bucket"] = pd.qcut(
                        class_df["_difficulty"],
                        q=min(config.n_difficulty_buckets, len(class_df)),
                        labels=False,
                        duplicates="drop",
                    )
                except ValueError:
                    class_df["_diff_bucket"] = 0

                n_buckets = class_df["_diff_bucket"].nunique()
                if n_buckets == 0:
                    # No buckets available, use simple random
                    selected = self.rng.choice(class_df.index.tolist(), n_per_class, replace=False)
                    sampled_indices.extend(selected)
                    class_details[str(cls)] = {
                        "n_sampled": n_per_class,
                        "n_total": len(class_df),
                        "method": "random",
                    }
                    continue

                per_bucket = n_per_class // n_buckets
                if per_bucket == 0:
                    # Not enough samples per bucket, use simple random
                    selected = self.rng.choice(class_df.index.tolist(), n_per_class, replace=False)
                    sampled_indices.extend(selected)
                    class_details[str(cls)] = {
                        "n_sampled": n_per_class,
                        "n_total": len(class_df),
                        "method": "random",
                    }
                    continue

                class_sampled = []
                for bucket in class_df["_diff_bucket"].unique():
                    bucket_df = class_df[class_df["_diff_bucket"] == bucket]
                    k = min(per_bucket, len(bucket_df))
                    selected = self.rng.choice(bucket_df.index.tolist(), k, replace=False)
                    class_sampled.extend(selected)

                sampled_indices.extend(class_sampled[:n_per_class])
                class_details[str(cls)] = {
                    "n_sampled": min(len(class_sampled), n_per_class),
                    "n_total": len(class_df),
                    "method": "difficulty_stratified",
                    "n_buckets": n_buckets,
                }
            else:
                # Simple random within class
                selected = self.rng.choice(class_df.index.tolist(), n_per_class, replace=False)
                sampled_indices.extend(selected)
                class_details[str(cls)] = {
                    "n_sampled": n_per_class,
                    "n_total": len(class_df),
                    "method": "random",
                }

        sampled = [samples[i] for i in sampled_indices[: config.n_samples]]

        metadata = {
            "strategy": "hybrid",
            "class_column": class_col,
            "difficulty_key": diff_key,
            "n_classes": len(classes),
            "n_requested": config.n_samples,
            "n_sampled": len(sampled),
            "n_total": len(samples),
            "class_details": class_details,
        }

        return sampled, metadata


# Sampler registry
SAMPLERS: Dict[SamplingStrategy, type] = {
    SamplingStrategy.RANDOM: RandomSampler,
    SamplingStrategy.STRATIFIED: StratifiedSampler,
    SamplingStrategy.DIFFICULTY_STRATIFIED: DifficultyStratifiedSampler,
    SamplingStrategy.HYBRID: HybridSampler,
}


class TaskMixer:
    """
    Main class for creating evaluation sets from benchmarks.

    Example:
        >>> mixer = TaskMixer(seed=42)
        >>>
        >>> # Quick start: Create evaluation set with defaults
        >>> eval_sets = mixer.create_evaluation_set(n_per_task=100)
        >>>
        >>> # Custom configuration per benchmark
        >>> eval_sets = mixer.create_evaluation_set(
        ...     n_per_task=100,
        ...     mtbench_strategy=SamplingStrategy.STRATIFIED,
        ...     timerbed_strategy=SamplingStrategy.HYBRID,
        ...     tsqa_strategy=SamplingStrategy.DIFFICULTY_STRATIFIED,
        ... )
    """

    # Default configurations per benchmark/task
    DEFAULT_CONFIGS = {
        "mtbench": {
            "finance": {
                "trend": SamplingConfig(
                    strategy=SamplingStrategy.STRATIFIED,
                    stratify_by=["source"],
                ),
                "forecasting": SamplingConfig(
                    strategy=SamplingStrategy.STRATIFIED,
                    stratify_by=["source"],
                ),
                "indicator_macd": SamplingConfig(
                    strategy=SamplingStrategy.STRATIFIED,
                    stratify_by=["source"],
                ),
                "indicator_bb": SamplingConfig(
                    strategy=SamplingStrategy.STRATIFIED,
                    stratify_by=["source"],
                ),
                "correlation": SamplingConfig(
                    strategy=SamplingStrategy.STRATIFIED,
                    stratify_by=["source"],
                ),
                "mcqa": SamplingConfig(
                    strategy=SamplingStrategy.RANDOM,
                ),
            },
            "weather": {
                "trend": SamplingConfig(
                    strategy=SamplingStrategy.STRATIFIED,
                    stratify_by=["source"],
                ),
                "forecasting": SamplingConfig(
                    strategy=SamplingStrategy.STRATIFIED,
                    stratify_by=["source"],
                ),
                "indicator_macd": SamplingConfig(
                    strategy=SamplingStrategy.STRATIFIED,
                    stratify_by=["source"],
                ),
                "mcqa": SamplingConfig(
                    strategy=SamplingStrategy.RANDOM,
                ),
            },
        },
        "timerbed": {
            "ECG": SamplingConfig(
                strategy=SamplingStrategy.HYBRID,
                stratify_by=["label"],
                difficulty_key="ts_length",
            ),
            "RCW": SamplingConfig(
                strategy=SamplingStrategy.HYBRID,
                stratify_by=["label"],
                difficulty_key="ts_length",
            ),
            "CTU": SamplingConfig(
                strategy=SamplingStrategy.HYBRID,
                stratify_by=["label"],
                difficulty_key="ts_length",
            ),
            "EMG": SamplingConfig(
                strategy=SamplingStrategy.HYBRID,
                stratify_by=["label"],
                difficulty_key="ts_length",
            ),
            "TEE": SamplingConfig(
                strategy=SamplingStrategy.HYBRID,
                stratify_by=["label"],
                difficulty_key="ts_length",
            ),
            "HAR": SamplingConfig(
                strategy=SamplingStrategy.HYBRID,
                stratify_by=["label"],
                difficulty_key="ts_length",
            ),
        },
        "tsqa": {
            "anomaly": SamplingConfig(
                strategy=SamplingStrategy.STRATIFIED,
                stratify_by=["domain"],
            ),
            "classification": SamplingConfig(
                strategy=SamplingStrategy.STRATIFIED,
                stratify_by=["domain"],
            ),
            "forecasting": SamplingConfig(
                strategy=SamplingStrategy.DIFFICULTY_STRATIFIED,
                difficulty_key="input_length",
            ),
            "imputation": SamplingConfig(
                strategy=SamplingStrategy.DIFFICULTY_STRATIFIED,
                difficulty_key="input_length",
            ),
            "qa": SamplingConfig(
                strategy=SamplingStrategy.STRATIFIED,
                stratify_by=["domain", "question_type"],
            ),
        },
    }

    def __init__(self, seed: int = 42, verbose: bool = True):
        """
        Initialize TaskMixer.

        Args:
            seed: Random seed for reproducibility
            verbose: Print progress during data loading
        """
        self.seed = seed
        self.verbose = verbose
        self._loaders_cache: Dict[str, Any] = {}
        self._data_cache: Dict[str, Any] = {}  # Cache loaded data to avoid re-parsing

    def _get_sampler(self, strategy: SamplingStrategy) -> Sampler:
        """Get sampler instance for given strategy."""
        sampler_class = SAMPLERS.get(strategy, RandomSampler)
        return sampler_class(self.seed)

    def _load_benchmark(self, benchmark: str) -> Dict[str, Any]:
        """Lazy load benchmark data."""
        if benchmark in self._loaders_cache:
            return self._loaders_cache[benchmark]

        if benchmark == "mtbench":
            from ..data_loaders.mtbench_loader import MTBenchDataLoader

            data = {
                "finance": MTBenchDataLoader(domain="finance"),
                "weather": MTBenchDataLoader(domain="weather"),
            }
        elif benchmark == "timerbed":
            from ..data_loaders.timerbed_loader import TimerBedLoader

            data = {"loader": TimerBedLoader()}
        elif benchmark == "tsqa":
            from ..data_loaders.tsqa_loader import TSQALoader

            data = {"loader": TSQALoader()}
        else:
            raise ValueError(f"Unknown benchmark: {benchmark}")

        self._loaders_cache[benchmark] = data
        return data

    def sample_task(
        self,
        benchmark: str,
        task: str,
        samples: List[Dict[str, Any]],
        config: Optional[SamplingConfig] = None,
        n_samples: Optional[int] = None,
    ) -> EvaluationSet:
        """
        Sample from a single task.

        Args:
            benchmark: Benchmark name ('mtbench', 'timerbed', 'tsqa')
            task: Task name
            samples: List of sample dictionaries
            config: Sampling configuration (uses defaults if None)
            n_samples: Override number of samples

        Returns:
            EvaluationSet with sampled data and metadata
        """
        # Get default config if not provided
        if config is None:
            if benchmark == "mtbench":
                # Try to find domain from samples
                domain = samples[0].get("domain", "finance") if samples else "finance"
                config = self.DEFAULT_CONFIGS.get(benchmark, {}).get(domain, {}).get(task, SamplingConfig())
            else:
                config = self.DEFAULT_CONFIGS.get(benchmark, {}).get(task, SamplingConfig())

        # Override n_samples if provided
        if n_samples is not None:
            config = SamplingConfig(
                n_samples=n_samples,
                strategy=config.strategy,
                stratify_by=config.stratify_by,
                difficulty_key=config.difficulty_key,
                n_difficulty_buckets=config.n_difficulty_buckets,
                min_per_stratum=config.min_per_stratum,
            )

        # Sample (automatically uses all if dataset < n_samples)
        sampler = self._get_sampler(config.strategy)
        sampled, metadata = sampler.sample(samples, config)

        # Ensure exactly min(n_samples, len(samples))
        target = min(config.n_samples, len(samples))
        if len(sampled) < target:
            sampled_ids = {id(s) for s in sampled}
            remaining = [s for s in samples if id(s) not in sampled_ids]
            np.random.RandomState(self.seed).shuffle(remaining)
            sampled.extend(remaining[: target - len(sampled)])
            metadata["n_sampled"] = len(sampled)

        return EvaluationSet(
            samples=sampled,
            metadata=metadata,
            config=config,
            benchmark=benchmark,
            task=task,
        )

    def create_evaluation_set(
        self,
        n_per_task: int = 100,
        benchmarks: Optional[List[str]] = None,
        mtbench_strategy: Optional[SamplingStrategy] = None,
        timerbed_strategy: Optional[SamplingStrategy] = None,
        tsqa_strategy: Optional[SamplingStrategy] = None,
        include_tasks: Optional[Dict[str, List[str]]] = None,
        exclude_tasks: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Dict[str, EvaluationSet]]:
        """
        Create complete evaluation set from all benchmarks.

        Args:
            n_per_task: Number of samples per task
            benchmarks: List of benchmarks to include (default: all)
            mtbench_strategy: Override strategy for MTBench
            timerbed_strategy: Override strategy for TimerBed
            tsqa_strategy: Override strategy for TSQA
            include_tasks: Dict mapping benchmark to list of tasks to include
            exclude_tasks: Dict mapping benchmark to list of tasks to exclude

        Returns:
            Nested dict: {benchmark: {task: EvaluationSet}}
        """
        benchmarks = benchmarks or ["mtbench", "timerbed", "tsqa"]
        include_tasks = include_tasks or {}
        exclude_tasks = exclude_tasks or {}

        # Check if this is first load (data not cached yet)
        needs_loading = not self._data_cache
        if needs_loading and self.verbose:
            print("📥 First-time data loading (will be cached for subsequent calls)...")

        result: Dict[str, Dict[str, EvaluationSet]] = {}

        for benchmark in benchmarks:
            result[benchmark] = {}

            if benchmark == "mtbench":
                data = self._load_benchmark("mtbench")
                for domain in ["finance", "weather"]:
                    cache_key = f"mtbench_{domain}"
                    if cache_key in self._data_cache:
                        all_tasks = self._data_cache[cache_key]
                    else:
                        if self.verbose:
                            print(f"  Loading MTBench {domain}...")
                        loader = data[domain]
                        all_tasks = loader.load_all()
                        self._data_cache[cache_key] = all_tasks

                    for task, samples in all_tasks.items():
                        full_task = f"{domain}_{task}"

                        # Check include/exclude
                        if include_tasks.get("mtbench") and full_task not in include_tasks["mtbench"]:
                            continue
                        if exclude_tasks.get("mtbench") and full_task in exclude_tasks["mtbench"]:
                            continue

                        # Get config
                        config = self.DEFAULT_CONFIGS["mtbench"][domain].get(task, SamplingConfig())
                        if mtbench_strategy:
                            config.strategy = mtbench_strategy

                        eval_set = self.sample_task(
                            benchmark="mtbench",
                            task=full_task,
                            samples=samples,
                            config=config,
                            n_samples=n_per_task,
                        )
                        result[benchmark][full_task] = eval_set

            elif benchmark == "timerbed":
                data = self._load_benchmark("timerbed")
                loader = data["loader"]

                # Use dynamic dataset list from TimerBedLoader
                from ..data_loaders.timerbed_loader import TimerBedLoader

                timerbed_datasets = list(TimerBedLoader.DATASETS.keys())
                for dataset in timerbed_datasets:
                    # Check include/exclude
                    if include_tasks.get("timerbed") and dataset not in include_tasks["timerbed"]:
                        continue
                    if exclude_tasks.get("timerbed") and dataset in exclude_tasks["timerbed"]:
                        continue

                    cache_key = f"timerbed_{dataset}"
                    if cache_key in self._data_cache:
                        samples = self._data_cache[cache_key]
                    else:
                        if self.verbose:
                            print(f"  Loading TimerBed {dataset}...")
                        splits = loader.load_dataset(dataset)
                        # Use test split only
                        samples = splits.get("test", [])
                        self._data_cache[cache_key] = samples

                    # Skip datasets with no test data
                    if not samples:
                        warnings.warn(f"TimerBed {dataset}: no test samples found, skipping")
                        continue

                    # Add label and ts_length for stratification
                    for s in samples:
                        if "label" not in s:
                            # Try multiple possible keys for label
                            # Use explicit None checks to handle falsy values like "" correctly
                            label = None
                            for key in ["ground_truth", "class", "label"]:
                                if key in s and s[key] is not None:
                                    label = s[key]
                                    break
                            if label is None:
                                raise ValueError(
                                    f"TimerBed {dataset}: sample missing label/ground_truth/class field. "
                                    f"Sample keys: {list(s.keys())}"
                                )
                            s["label"] = label
                        if "ts_length" not in s:
                            ts = s.get("time_series", s.get("values", []))
                            s["ts_length"] = len(ts) if isinstance(ts, (list, np.ndarray)) else 0

                    config = self.DEFAULT_CONFIGS["timerbed"].get(dataset, SamplingConfig())
                    if timerbed_strategy:
                        config.strategy = timerbed_strategy

                    eval_set = self.sample_task(
                        benchmark="timerbed",
                        task=dataset,
                        samples=samples,
                        config=config,
                        n_samples=n_per_task,
                    )

                    # Only add if we got samples
                    if len(eval_set.samples) > 0:
                        result[benchmark][dataset] = eval_set
                    else:
                        warnings.warn(f"TimerBed {dataset}: sampler returned 0 samples, skipping")

            elif benchmark == "tsqa":
                data = self._load_benchmark("tsqa")
                loader = data["loader"]

                for task in ["anomaly", "classification", "forecasting", "imputation", "qa"]:
                    # Check include/exclude
                    if include_tasks.get("tsqa") and task not in include_tasks["tsqa"]:
                        continue
                    if exclude_tasks.get("tsqa") and task in exclude_tasks["tsqa"]:
                        continue

                    cache_key = f"tsqa_{task}"
                    if cache_key in self._data_cache:
                        samples = self._data_cache[cache_key]
                    else:
                        if self.verbose:
                            print(f"  Loading TSQA {task}...")
                        samples = loader.load_task(task)
                        self._data_cache[cache_key] = samples

                    # Skip tasks with no data
                    if not samples:
                        warnings.warn(f"TSQA {task}: no samples found, skipping")
                        continue

                    # Add derived fields for stratification
                    for s in samples:
                        if "input_length" not in s:
                            ts = s.get("time_series", s.get("values", s.get("input", [])))
                            s["input_length"] = len(ts) if isinstance(ts, (list, np.ndarray)) else 0

                    config = self.DEFAULT_CONFIGS["tsqa"].get(task, SamplingConfig())
                    if tsqa_strategy:
                        config.strategy = tsqa_strategy

                    eval_set = self.sample_task(
                        benchmark="tsqa",
                        task=task,
                        samples=samples,
                        config=config,
                        n_samples=n_per_task,
                    )

                    # Only add if we got samples
                    if len(eval_set.samples) > 0:
                        result[benchmark][task] = eval_set
                    else:
                        warnings.warn(f"TSQA {task}: sampler returned 0 samples, skipping")

        if needs_loading and self.verbose:
            print("✅ Data loading complete (cached for future calls)")

        return result

    def get_summary(self, eval_sets: Dict[str, Dict[str, EvaluationSet]]) -> Dict[str, Any]:
        """Get summary statistics for evaluation sets."""
        summary = {
            "total_samples": 0,
            "total_tasks": 0,
            "benchmarks": {},
        }

        for benchmark, tasks in eval_sets.items():
            benchmark_summary = {
                "n_tasks": len(tasks),
                "n_samples": sum(len(es) for es in tasks.values()),
                "tasks": {},
            }

            for task, eval_set in tasks.items():
                benchmark_summary["tasks"][task] = eval_set.to_dict()
                summary["total_samples"] += len(eval_set)
                summary["total_tasks"] += 1

            summary["benchmarks"][benchmark] = benchmark_summary

        return summary

    def save_evaluation_set(
        self,
        eval_sets: Dict[str, Dict[str, EvaluationSet]],
        path: Union[str, Path],
        save_samples: bool = True,
    ) -> None:
        """
        Save evaluation set to JSON file.

        Args:
            eval_sets: Evaluation sets to save
            path: Output file path
            save_samples: Whether to include full samples (can be large)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        output = {
            "seed": self.seed,
            "summary": self.get_summary(eval_sets),
            "benchmarks": {},
        }

        for benchmark, tasks in eval_sets.items():
            output["benchmarks"][benchmark] = {}
            for task, eval_set in tasks.items():
                task_data = {
                    "metadata": eval_set.metadata,
                    "config": {
                        "n_samples": eval_set.config.n_samples,
                        "strategy": eval_set.config.strategy.value,
                        "stratify_by": eval_set.config.stratify_by,
                        "difficulty_key": eval_set.config.difficulty_key,
                    },
                }
                if save_samples:
                    # Save sample indices or identifiers instead of full data
                    task_data["sample_ids"] = [
                        s.get("id", s.get("filename", i)) for i, s in enumerate(eval_set.samples)
                    ]
                output["benchmarks"][benchmark][task] = task_data

        with open(path, "w") as f:
            json.dump(output, f, indent=2, default=str)

    @staticmethod
    def compute_config_hash(eval_sets: Dict[str, Dict[str, EvaluationSet]]) -> str:
        """Compute a hash of the evaluation configuration for reproducibility tracking."""
        config_str = json.dumps(
            {
                benchmark: {
                    task: {
                        "n": len(es),
                        "strategy": es.config.strategy.value,
                    }
                    for task, es in tasks.items()
                }
                for benchmark, tasks in eval_sets.items()
            },
            sort_keys=True,
        )
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


# Convenience functions
def create_default_evaluation_set(
    n_per_task: int = 100,
    seed: int = 42,
) -> Dict[str, Dict[str, EvaluationSet]]:
    """
    Create evaluation set with recommended defaults.

    This follows the evaluation protocols from the original papers:
    - MTBench: Stratified by source (7d vs 30d input)
    - TimerBed: Hybrid (class-balanced + difficulty-diverse), TEST split only
    - TSQA: Stratified by domain, difficulty-stratified for regression tasks

    Args:
        n_per_task: Number of samples per task (default: 100, paper used 50)
        seed: Random seed

    Returns:
        Evaluation sets ready for benchmarking
    """
    mixer = TaskMixer(seed=seed)
    return mixer.create_evaluation_set(n_per_task=n_per_task)


def create_minimal_evaluation_set(seed: int = 42) -> Dict[str, Dict[str, EvaluationSet]]:
    """
    Create minimal evaluation set for quick testing (~500 samples total).

    Uses 25 samples per task, following TSQA paper's 50-sample protocol
    but halved for faster iteration.
    """
    mixer = TaskMixer(seed=seed)
    return mixer.create_evaluation_set(n_per_task=25)


def create_full_evaluation_set(seed: int = 42) -> Dict[str, Dict[str, EvaluationSet]]:
    """
    Create full evaluation set using all available test data.

    Warning: This can be slow and expensive for LLM evaluation.
    """
    mixer = TaskMixer(seed=seed)
    return mixer.create_evaluation_set(n_per_task=10000)  # Large number to get all


__all__ = [
    "EvaluationSet",
    "SamplingConfig",
    "SamplingStrategy",
    "TaskMixer",
    "create_default_evaluation_set",
    "create_full_evaluation_set",
    "create_minimal_evaluation_set",
]
