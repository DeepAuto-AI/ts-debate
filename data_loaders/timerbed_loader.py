"""
TimerBed Loader

- Paper: "A Picture is Worth A Thousand Numbers" (NAACL 2025), arXiv:2411.06018
- GitHub: https://github.com/AdityaLab/DeepTime/
- Prompt format: benchmarks/TimerBed/LLMs/Method/prompt.py (lines 394-398, 52-77)
- Evaluation: benchmarks/TimerBed/LLMs/Method/eval.py
- Class labels: VL-Time paper Appendix F (https://aclanthology.org/2025.naacl-long.383.pdf)
- Dataset descriptions: https://timeseriesclassification.com/
"""

import random
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from utils.task_config import get_task_instruction
from .utils import format_values_as_data_str

_DATASETS_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "TimerBed" / "datasets"


def timerbed_prompt_generation(
    time_series: Union[List[float], List[List[float]], np.ndarray],
    question: str,
    choices: List[str],
    modal: str = "L",
    decimal_places: int = 3,
) -> str:
    """
    Official TimerBed prompt generation.
    From: LLMs/Method/prompt.py lines 394-398 (L modal) and df2str function (lines 52-77)

    Args:
        time_series: The numerical time series values
            - Univariate: List[float] or 1D array
            - Multivariate: 2D array with shape (n_channels, n_timesteps) or List[List[float]]
        question: The question to ask (e.g., "What activity is being performed?")
        choices: List of class choices
        modal: "L" for language only, "V" for vision only, "LV" for both
        decimal_places: Decimal places for formatting numbers

    Returns:
        Official formatted prompt string
    """
    # Convert to numpy array for consistent handling
    if isinstance(time_series, list):
        # Check if multivariate (list of lists)
        if len(time_series) > 0 and isinstance(time_series[0], list):
            time_series = np.array(time_series)
        else:
            time_series = np.array(time_series)

    # Format time series data (official: df2str function)
    if isinstance(time_series, np.ndarray):
        if time_series.ndim > 1:
            # Multivariate: format each dimension separately (official format)
            formatted_dims = []
            for i, dim in enumerate(time_series):
                dim_str = ",".join(f"{x:.{decimal_places}f}" for x in dim)
                formatted_dims.append(f"dim_{i}: {dim_str}")
            data_str = ";".join(formatted_dims)
        else:
            # Univariate
            data_str = ",".join(f"{x:.{decimal_places}f}" for x in time_series.flatten())
    else:
        data_str = str(time_series)

    # Official prompt format from prompt.py line 394-398
    if modal == "L":
        prompt = f"""Given the corresponding specific numerical series are as follows: {data_str}. Answer the following question using the specified format. 
Question: {question}
Choices: {choices!s}
"""
    elif modal == "V":
        prompt = f"""<<IMG>>Given the image above, answer the following question using the specified format. 
Question: {question}
Choices: {choices!s}
"""
    elif modal == "LV":
        prompt = f"""<<IMG>>Given the image above, and the corresponding specific values are as follows: {data_str}. Answer the following question using the specified format. 
Question: {question}
Choices: {choices!s}
"""
    else:
        raise ValueError(f"Invalid modal: {modal}. Must be 'L', 'V', or 'LV'")

    return prompt


class TimerBedLoader:
    """Load TimerBed benchmark - returns ALL samples for full evaluation."""

    # Constants
    DEFAULT_DECIMAL_PLACES = 3

    # Official dataset info from VL-Time paper Table 2 (arXiv:2411.06018)
    # Note: Total samples from paper; train/test splits are from local .ts files
    DATASETS = {
        "CTU": {"domain": "Energy/Usage", "n_classes": 2, "dim": 1, "length": 720, "total": 500},
        "ECG": {"domain": "Healthcare", "n_classes": 4, "dim": 1, "length": 1500, "total": 43673},
        "EMG": {"domain": "Healthcare", "n_classes": 3, "dim": 1, "length": 1500, "total": 205},
        "HAR": {"domain": "Sports Monitoring", "n_classes": 6, "dim": 3, "length": 206, "total": 10299},
        "RCW": {"domain": "Bioacoustics", "n_classes": 2, "dim": 1, "length": 4000, "total": 30000},
        "TEE": {"domain": "Geophysics", "n_classes": 7, "dim": 1, "length": 319, "total": 143},
    }

    # Class descriptions from VL-Time paper Section F.1 + demo.csv
    # Source: https://aclanthology.org/2025.naacl-long.383.pdf (Appendix F)
    CLASSES = {
        "CTU": ["desktop", "laptop"],
        "ECG": ["normal sinus rhythm", "fibrillation", "alternative rhythm", "too noisy to be classified"],
        "EMG": ["healthy", "suffering from neuropathy", "suffering from myopathy"],
        "HAR": ["walking", "walking upstairs", "walking downstairs", "sitting", "standing", "laying down"],
        "RCW": ["There is no right whale call in the image.", "There is a right whale call in the image."],
        "TEE": [
            "CG Positive Initial Return Stroke",
            "IR Negative Initial Return Stroke",
            "SR Subsequent Negative Return Stroke",
            "I Impulsive Event",
            "I2 Impulsive Event Pair",
            "KM Gradual Intra-Cloud Stroke",
            "O Off-record",
        ],
    }

    # Dataset descriptions from official UCR/UEA Time Series Classification Archive
    # Source: https://timeseriesclassification.com/
    DATA_DESCRIPTIONS = {
        # Source: https://timeseriesclassification.com/description.php?Dataset=Computers
        "CTU": (
            "The Computers dataset measures the power consumption of computers to distinguish between "
            "desktops and laptops. Each series is a univariate time series of length 720 (24 hours of readings taken every 2 minutes) representing "
            "power consumption readings. The goal is to classify whether the computer is a desktop or laptop "
            "based on power usage patterns."
        ),
        # Source: https://timeseriesclassification.com/description.php?Dataset=CardiacArrhythmia
        "ECG": (
            "The CardiacArrhythmia dataset contains single-lead ECG recordings collected from 82 subjects, "
            "sampled at 100 Hz. The original recordings were split into 1,500 observation windows (5 seconds). "
            "Classes are: Normal sinus rhythm, Atrial fibrillation, Alternative rhythm, and Others (too noisy). "
            "The data is from PhysioNet Computing in Cardiology Challenge 2017."
        ),
        # Source: https://timeseriesclassification.com/description.php?Dataset=NerveDamage
        "EMG": (
            "The NerveDamage dataset consists of single-channel EMG recordings from the tibialis anterior muscle "
            "of three volunteers: one healthy, one suffering from neuropathy, and one suffering from myopathy. "
            "Electromyograms (EMG) measure muscle responses as electrical activity to neural stimulation, "
            "used to diagnose muscular dystrophies and neuropathies. Sampled at 4K Hz, split into 1,500 observation windows. "
            "Data from PhysioNet."
        ),
        # Source: https://timeseriesclassification.com/description.php?Dataset=WalkingSittingStanding
        "HAR": (
            "The WalkingSittingStanding dataset (Human Activity Recognition) contains recordings from 30 volunteers "
            "aged 19-48 years. The wearable sensors on a smartphone measure triaxial linear acceleration at 50 Hz. "
            "The six classes are: walking, walking upstairs, walking downstairs, sitting, standing, laying down. "
            "Each sample has 3 channels (X, Y, Z acceleration) with 206 timesteps. Data from UCI HAR dataset."
        ),
        # Source: https://timeseriesclassification.com/description.php?Dataset=RightWhaleCalls
        "RCW": (
            "The RightWhaleCalls dataset classifies audio signals as containing a right whale up-call or not. "
            "Up-calls are the most commonly documented right whale vocalisation with acoustic signature of "
            "approximately 60Hz-250Hz, typically lasting 1 second. Each case is a two second audio segment "
            "sampled at 2kHz (series length 4000). Right whale calls can be difficult to hear due to "
            "anthropogenic sounds such as ship noise, drilling, or naval operations. "
            "Data from Marinexplore and Cornell University Whale Detection Challenge."
        ),
        # Source: https://timeseriesclassification.com/description.php?Dataset=Lightning7
        "TEE": (
            "The Lightning7 dataset contains transient electromagnetic events detected by the FORTE satellite "
            "using optical and radio-frequency instruments. Data is collected at 50 MHz for 800 microseconds. "
            "A Fourier transform produces a spectrogram, collapsed in frequency to create power density time series. "
            "The seven classes are lightning event types: CG Positive Initial Return Stroke (sharp radiation turn-on), "
            "IR Negative Initial Return Stroke (ramp up to attachment point then exponential decline), "
            "SR Subsequent Negative Return Stroke, I Impulsive Event (sudden peak), "
            "I2 Impulsive Event Pair (TIPPs), KM Gradual Intra-Cloud Stroke, O Off-record."
        ),
    }

    # Task descriptions from VL-Time paper (NAACL 2025) Section E.1
    # Source: https://aclanthology.org/2025.naacl-long.383.pdf
    QUESTIONS = {
        "CTU": (
            "Play as a computer energy consumption analysis expert, please correctly "
            "determine whether this computer is a desktop or a laptop based on the "
            "24-hour power consumption data."
        ),
        "ECG": (
            "As a cardiologist, you are tasked with classifying a patient's heart "
            "condition based on single-lead ECG recordings."
        ),
        "EMG": (
            "As an Electromyograms (EMG) analysis expert, you are tasked with "
            "determining the type of the subject based on the EMG record."
        ),
        "HAR": (
            "As a human activity recognition expert, you are tasked with determining "
            "the type of activity performed by the subject based on the accelerometer "
            "record series along the x, y, and z axes over time."
        ),
        "RCW": ("Play the role of a marine biology expert: is there a right whale call in the record?"),
        "TEE": (
            "Based on the power density time series data and select the transient "
            "electromagnetic event that best matches. The FORTE satellite detects "
            "transient electromagnetic events associated with lightning using a suite "
            "of optical and radio-frequency (RF) instruments. There are 7 event types. "
            "CG Positive Initial Return Stroke: A positive charge is lowered from a cloud "
            "to the ground. The characteristic feature of this type of event in the power "
            "density time series is a sharp turn-on of radiation, followed by a few hundreds "
            "of microseconds of noise; IR Negative Initial Return Stroke: A negative charge "
            "is lowered from a cloud to ground. The power waveform slowly ramps up to a level "
            "known as an attachment point, where a large surge current causes the VHF power to "
            "'spike'. This attachment is followed by an exponentially shaped decline in the "
            "waveform.; SR Subsequent Negative Return Stroke: A negative charge is lowered "
            "from a cloud to ground. As the name implies, subsequent return strokes come after "
            "initial return strokes. Note that subsequent positive return strokes don't exist. "
            "I Impulsive Event: Typically an intra-cloud event characterized by a sudden peak "
            "in the waveform. I2 Impulsive Event Pair: Another intra-cloud event characterized "
            "by sudden peaks in the waveform that come in closely separated pairs. These are "
            "also called TIPPs (Trans-Ionospheric Pulse Pairs). KM Gradual Intra-Cloud Stroke: "
            "An intra-cloud event which increases in power more gradually than an impulsive "
            "event. O Off-record: 800 microseconds was not enough to fully capture the "
            "lightning event."
        ),
    }

    def __init__(self, dataset_path: Optional[str] = None, modal: str = "LV"):
        """
        Initialize TimerBed loader.

        Args:
            dataset_path: Path to datasets folder (default: datasets/TimerBed/datasets)
            modal: Prompt modality - "L" (language only), "V" (vision), "LV" (both)
        """
        self.dataset_path = Path(dataset_path) if dataset_path else _DATASETS_DIR
        self.modal = modal
        self._cache: Dict[str, Dict] = {}

    def load_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Load ALL data for ALL datasets at once.

        Returns:
            Dict mapping dataset names to their data:
            {
                'CTU': {'train': [...], 'test': [...], 'info': {...}},
                'ECG': {'train': [...], 'test': [...], 'info': {...}},
                ...
            }
        """
        result = {}
        for name in self.DATASETS:
            try:
                result[name] = self.load_dataset(name)
            except Exception as e:
                warnings.warn(f"Failed to load {name}: {e}", UserWarning)
        return result

    def load_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """
        Load all samples for a specific dataset.

        Returns:
            {
                'train': List of sample dicts,
                'test': List of sample dicts,
                'info': Dataset metadata,
            }
        """
        if dataset_name in self._cache:
            return self._cache[dataset_name]

        if dataset_name not in self.DATASETS:
            raise ValueError(f"Unknown dataset '{dataset_name}'. Must be one of {list(self.DATASETS.keys())}")

        dataset_dir = self.dataset_path / dataset_name
        if not dataset_dir.exists():
            raise FileNotFoundError(f"Dataset {dataset_name} not found at {dataset_dir}")

        info = self.DATASETS[dataset_name].copy()
        info["classes"] = self.CLASSES.get(dataset_name, [])

        result = {
            "train": self._load_split(dataset_dir, dataset_name, "TRAIN", modal=self.modal),
            "test": self._load_split(dataset_dir, dataset_name, "TEST", modal=self.modal),
            "info": info,
        }

        self._cache[dataset_name] = result
        return result

    def _load_split(
        self,
        dataset_dir: Path,
        dataset_name: str,
        split: str,
        modal: str = "L",
        question: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Load all samples from a split with official prompts."""
        split_files = list(dataset_dir.glob(f"*_{split}.ts"))
        if not split_files:
            raise FileNotFoundError(
                f"TimerBed {dataset_name}: No {split} split files found in {dataset_dir}. "
                f"Expected pattern: *_{split}.ts"
            )

        data, labels = self._parse_ts_file(split_files[0])
        classes = self.CLASSES.get(dataset_name, [])
        info = self.DATASETS[dataset_name]

        # Default question from official run.py (can be customized per dataset)
        if question is None:
            question = self.QUESTIONS.get(dataset_name, "What is the classification of this time series?")

        samples = []
        for i, (ts_data, label) in enumerate(zip(data, labels)):
            # Get semantic label
            try:
                label_idx = int(float(label))  # Handle "5.0" format
                # CTU uses 1-indexed labels (1, 2) instead of 0-indexed (0, 1)
                if dataset_name == "CTU":
                    label_idx -= 1
                semantic_label = classes[label_idx] if 0 <= label_idx < len(classes) else f"Class_{label}"
            except (ValueError, IndexError):
                semantic_label = f"Class_{label}"

            if ts_data.ndim > 1:
                # Multivariate: values is list of lists (one list per dimension/channel)
                values = [dim.tolist() for dim in ts_data]  # All dimensions, not averaged
                raw_channels = ts_data.tolist()
                n_channels = ts_data.shape[0]
                n_timesteps = ts_data.shape[1]
            else:
                # Univariate: values is a single list
                values = ts_data.tolist()
                raw_channels = [ts_data.tolist()]
                n_channels = 1
                n_timesteps = len(values)

            # Generate official prompt (uses full multivariate data)
            prompt = timerbed_prompt_generation(
                time_series=ts_data if ts_data.ndim > 1 else values,
                question=question,
                choices=classes,
                modal=modal,
            )

            sample_dict = {
                "idx": i,
                "values": values,
                "timestamps": list(range(n_timesteps)),
                "raw_channels": raw_channels,
                "n_channels": n_channels,
                "numeric_label": str(label),
                "ground_truth": semantic_label,
                "answer": semantic_label,
                "options": classes,
                "dataset_name": dataset_name,
                "domain": info["domain"],
                "split": split,
                "task_type": "classification",
                "question": question,
                "prompt": prompt,
                # Text modality: dataset description (what the data IS)
                "text": self.DATA_DESCRIPTIONS.get(dataset_name, ""),
                # Formatted data string (multivariate-aware)
                "data_str": format_values_as_data_str(values),
            }

            # Add filled task_instruction
            task_key = f"timerbed_{dataset_name.lower()}"
            sample_dict["task_instruction"] = get_task_instruction(task_key, sample=sample_dict)

            samples.append(sample_dict)

        return samples

    def _parse_ts_file(self, filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Parse .ts file format (UCR/UEA standard)."""
        with filepath.open("r") as f:
            lines = f.readlines()

        # Filter out metadata lines
        data_lines = [
            line.strip() for line in lines if line.strip() and not line.startswith("@") and not line.startswith("#")
        ]

        all_series = []
        all_labels = []

        for line in data_lines:
            parts = line.split(":")
            if len(parts) < 2:
                continue

            label = parts[-1].strip()
            dimensions = []

            for dim_str in parts[:-1]:
                dim_str = dim_str.strip().strip("()")
                values = [float(v) for v in dim_str.replace(",", " ").split() if v]
                if values:
                    dimensions.append(np.array(values))

            if dimensions:
                # Pad dimensions to same length
                max_len = max(len(d) for d in dimensions)
                padded = [np.pad(d, (0, max_len - len(d)), mode="edge") for d in dimensions]
                all_series.append(np.array(padded))
                all_labels.append(label)

        if not all_series:
            return np.array([]), np.array([])

        # Pad all series to same length
        max_timepoints = max(s.shape[1] for s in all_series)
        padded_series = []
        for series in all_series:
            if series.shape[1] < max_timepoints:
                pad_width = ((0, 0), (0, max_timepoints - series.shape[1]))
                series = np.pad(series, pad_width, mode="constant")
            padded_series.append(series)

        return np.array(padded_series), np.array(all_labels)

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {"path": str(self.dataset_path), "datasets": {}}

        for name, info in self.DATASETS.items():
            dataset_dir = self.dataset_path / name
            if dataset_dir.exists():
                train_file = list(dataset_dir.glob("*_TRAIN.ts"))
                test_file = list(dataset_dir.glob("*_TEST.ts"))
                stats["datasets"][name] = {
                    "exists": True,
                    "train_file": str(train_file[0]) if train_file else None,
                    "test_file": str(test_file[0]) if test_file else None,
                    "dim": info["dim"],
                    "length": info["length"],
                    "domain": info["domain"],
                    "n_classes": info["n_classes"],
                }
            else:
                stats["datasets"][name] = {"exists": False}

        return stats

    def add_demo_examples(
        self,
        samples: List[Dict[str, Any]],
        train_samples: List[Dict[str, Any]],
        num_shots: int = 1,
        seed: int = 42,
    ) -> List[Dict[str, Any]]:
        """
        Add demo examples to samples for VL-Time ICL support.

        Matches official prompt.py behavior - provides labeled demo examples
        for few-shot in-context learning.

        Args:
            samples: List of test samples to add demos to
            train_samples: List of training samples to draw demos from
            num_shots: Number of demo examples per class
            seed: Random seed for reproducibility

        Returns:
            samples with 'demo_examples' field added
        """
        if num_shots <= 0:
            # Zero-shot: no demos needed
            for sample in samples:
                sample["demo_examples"] = []
            return samples

        rng = random.Random(seed)

        # Group train samples by label
        train_by_label: Dict[str, List[Dict[str, Any]]] = {}
        for train_sample in train_samples:
            label = train_sample.get("ground_truth", train_sample.get("answer", ""))
            if label not in train_by_label:
                train_by_label[label] = []
            train_by_label[label].append(train_sample)

        # Get all unique labels
        all_labels = list(train_by_label.keys())

        for sample in samples:
            target_idx = sample.get("idx", -1)
            demo_examples = []

            # Sample num_shots examples per class (matching official prompt.py)
            for label in all_labels:
                class_samples = train_by_label.get(label, [])
                # Exclude the target sample itself if it's in train
                class_samples = [s for s in class_samples if s.get("idx") != target_idx]

                if class_samples:
                    n_to_sample = min(num_shots, len(class_samples))
                    selected = rng.sample(class_samples, n_to_sample)
                    demo_examples.extend(selected)

            # Shuffle demos (matching official prompt.py random.shuffle behavior)
            rng.shuffle(demo_examples)

            sample["demo_examples"] = demo_examples

        return samples

    def load_dataset_with_demos(
        self,
        dataset_name: str,
        num_shots: int = 1,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Load dataset with demo examples already attached to test samples.

        Convenience method for VL-Time ICL evaluation.

        Note: Demo samples may not have charts generated. If using VL-Time with
        visual modality, charts should be generated separately (VL-Time baseline
        can generate them on-demand if a chart_generator is provided).

        Args:
            dataset_name: Dataset to load
            num_shots: Number of demo examples per class
            seed: Random seed for reproducibility

        Returns:
            Dataset dict with test samples containing 'demo_examples' field
        """
        data = self.load_dataset(dataset_name)

        # Add demos to test samples using train samples
        data["test"] = self.add_demo_examples(
            samples=data["test"],
            train_samples=data["train"],
            num_shots=num_shots,
            seed=seed,
        )

        return data
