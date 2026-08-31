"""
Cost Monitoring and Analysis for LLM API Usage

Tracks token usage and calculates costs based on model-specific pricing.
"""

import json
import threading
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

# Pricing data (USD per 1M tokens)
# Source: https://openrouter.ai/api/v1/models (queried live)
MODEL_PRICING = {
    # OpenAI models via OpenRouter
    "openai/gpt-4o": {"prompt": 2.50, "completion": 10.00},
    "openai/gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "openai/gpt-4.1": {"prompt": 2.00, "completion": 8.00},
    "openai/gpt-4.1-mini": {"prompt": 0.40, "completion": 1.60},
    "openai/gpt-5": {"prompt": 1.25, "completion": 10.00},
    "openai/gpt-5-mini": {"prompt": 0.25, "completion": 2.00},
    "openai/gpt-5.1": {"prompt": 1.25, "completion": 10.00},
    "openai/gpt-5.2": {"prompt": 1.75, "completion": 14.00},
    # Google models via OpenRouter
    "google/gemini-2.5-flash-preview-09-2025": {"prompt": 0.30, "completion": 2.50},
    "google/gemini-2.5-pro": {"prompt": 1.25, "completion": 10.00},
    "google/gemini-3-pro-preview": {"prompt": 2.00, "completion": 12.00},
    # Qwen models via OpenRouter
    "qwen/qwen3-vl-235b-a22b-thinking": {"prompt": 0.30, "completion": 1.20},
    # xAI models via OpenRouter (2M context, reasoning optional)
    "x-ai/grok-4.1-fast": {"prompt": 0.20, "completion": 0.50},
    # Zhipu AI (GLM) models via OpenRouter
    "z-ai/glm-4.6v": {"prompt": 0.30, "completion": 0.90},
    # Default fallback
    "default": {"prompt": 1.00, "completion": 2.00},
}


@dataclass
class APICall:
    """Record of a single API call"""

    timestamp: float
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_cost: float
    completion_cost: float
    total_cost: float
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "prompt_cost": self.prompt_cost,
            "completion_cost": self.completion_cost,
            "total_cost": self.total_cost,
            "metadata": self.metadata,
        }


class CostMonitor:
    """
    Monitor and track LLM API costs across experiments.

    Features:
    - Accurate token counting and cost calculation
    - Per-model cost tracking
    - Per-framework aggregation
    - Export to CSV/JSON for analysis
    - Budget warnings

    Example:
        monitor = CostMonitor(budget_limit=100.0)
        monitor.log_call("openai/gpt-4o", prompt_tokens=100, completion_tokens=50)
        print(monitor.summary())
        monitor.export_csv("costs.csv")
    """

    def __init__(self, budget_limit: Optional[float] = None, warn_threshold: float = 0.8):
        """
        Initialize cost monitor

        Args:
            budget_limit: Optional budget limit in USD (None = no limit)
            warn_threshold: Fraction of budget to trigger warning (0.8 = 80%)
        """
        self.calls: List[APICall] = []
        self.budget_limit = budget_limit
        self.warn_threshold = warn_threshold
        self._warned = False

        # Aggregated stats
        self.total_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_cost = 0.0

        # Per-model stats
        self.model_stats: Dict[str, Dict] = {}

        # Per-framework stats (for comparison)
        self.framework_stats: Dict[str, Dict] = {}

        # Per-stage stats (for call-wise cost tracking)
        self.stage_stats: Dict[str, Dict] = {}
        self._context = threading.local()

    def set_stage(self, stage: str, agent_id: str = "") -> None:
        """
        Set current stage context for subsequent log_call() invocations.

        This enables call-wise cost tracking by stage (e.g., "ke", "debate", "reviewer").
        Thread-safe via thread-local storage.

        Args:
            stage: Stage name (e.g., "ke", "debate_r1", "reviewer", "synthesizer")
            agent_id: Optional agent identifier for finer granularity
        """
        self._context.stage = stage
        self._context.agent_id = agent_id
        self._context.start_time = time.time()

    def record_stage_time(self, stage: str, elapsed: float) -> None:
        """
        Record wall-clock time for a stage.

        Args:
            stage: Stage name (must match the stage used in set_stage)
            elapsed: Elapsed time in seconds
        """
        if stage not in self.stage_stats:
            self.stage_stats[stage] = {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost": 0.0,
                "time": 0.0,
            }
        self.stage_stats[stage]["time"] += elapsed

    def log_call(
        self, model: str, prompt_tokens: int, completion_tokens: int, framework: Optional[str] = None, **metadata
    ) -> float:
        """
        Log an API call and calculate cost

        Args:
            model: Model identifier (e.g., "openai/gpt-4o")
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            framework: Optional framework name (e.g., "TS-Debate", "MAD")
            **metadata: Additional metadata to store

        Returns:
            Cost of this call in USD
        """
        # Get pricing for this model (lazy import to avoid circular dependency)
        from .llm_providers import AVAILABLE_MODELS

        model = AVAILABLE_MODELS[model]["model"] if model in list(AVAILABLE_MODELS.keys()) else model
        pricing = MODEL_PRICING.get(model, MODEL_PRICING["default"])
        if model not in MODEL_PRICING:
            warnings.warn(f"Unknown model '{model}', using default pricing", UserWarning)

        # Calculate costs (price per 1M tokens)
        prompt_cost = (prompt_tokens / 1_000_000) * pricing["prompt"]
        completion_cost = (completion_tokens / 1_000_000) * pricing["completion"]
        total_cost = prompt_cost + completion_cost
        total_tokens = prompt_tokens + completion_tokens

        # Create call record
        call = APICall(
            timestamp=time.time(),
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            prompt_cost=prompt_cost,
            completion_cost=completion_cost,
            total_cost=total_cost,
            metadata={"framework": framework, **metadata},
        )

        self.calls.append(call)

        # Update aggregated stats
        self.total_calls += 1
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += total_tokens
        self.total_cost += total_cost

        # Update per-model stats
        if model not in self.model_stats:
            self.model_stats[model] = {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost": 0.0,
            }
        self.model_stats[model]["calls"] += 1
        self.model_stats[model]["prompt_tokens"] += prompt_tokens
        self.model_stats[model]["completion_tokens"] += completion_tokens
        self.model_stats[model]["total_tokens"] += total_tokens
        self.model_stats[model]["cost"] += total_cost

        # Update per-framework stats
        if framework:
            if framework not in self.framework_stats:
                self.framework_stats[framework] = {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cost": 0.0,
                }
            self.framework_stats[framework]["calls"] += 1
            self.framework_stats[framework]["prompt_tokens"] += prompt_tokens
            self.framework_stats[framework]["completion_tokens"] += completion_tokens
            self.framework_stats[framework]["total_tokens"] += total_tokens
            self.framework_stats[framework]["cost"] += total_cost

        # Update per-stage stats (from thread-local context)
        stage = getattr(self._context, "stage", None)
        if stage:
            if stage not in self.stage_stats:
                self.stage_stats[stage] = {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cost": 0.0,
                    "time": 0.0,
                }
            self.stage_stats[stage]["calls"] += 1
            self.stage_stats[stage]["prompt_tokens"] += prompt_tokens
            self.stage_stats[stage]["completion_tokens"] += completion_tokens
            self.stage_stats[stage]["total_tokens"] += total_tokens
            self.stage_stats[stage]["cost"] += total_cost

        # Check budget
        if self.budget_limit and not self._warned:
            if self.total_cost >= self.budget_limit * self.warn_threshold:
                print("\n⚠️  WARNING: Budget threshold reached!")
                print(f"   Current: ${self.total_cost:.2f} / ${self.budget_limit:.2f}")
                print(f"   Usage: {(self.total_cost / self.budget_limit * 100):.1f}%\n")
                self._warned = True

        return total_cost

    def summary(self, detailed: bool = False) -> str:
        """
        Generate summary report

        Args:
            detailed: Include per-call details

        Returns:
            Formatted summary string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("COST ANALYSIS SUMMARY")
        lines.append("=" * 80)

        # Overall stats
        lines.append("\n📊 Overall Statistics:")
        lines.append(f"  Total API Calls: {self.total_calls:,}")
        lines.append(f"  Total Tokens: {self.total_tokens:,}")
        lines.append(f"    - Prompt: {self.total_prompt_tokens:,}")
        lines.append(f"    - Completion: {self.total_completion_tokens:,}")
        lines.append(f"  Total Cost: ${self.total_cost:.4f}")

        if self.budget_limit:
            pct = self.total_cost / self.budget_limit * 100
            lines.append(f"  Budget: ${self.total_cost:.2f} / ${self.budget_limit:.2f} ({pct:.1f}%)")

        # Per-model breakdown
        if self.model_stats:
            lines.append("\n📈 By Model:")
            for model, stats in sorted(self.model_stats.items(), key=lambda x: x[1]["cost"], reverse=True):
                pct = (stats["cost"] / self.total_cost * 100) if self.total_cost > 0 else 0
                lines.append(f"  • {model}:")
                lines.append(
                    f"    Calls: {stats['calls']:,} | Tokens: {stats['total_tokens']:,} | Cost: ${stats['cost']:.4f} ({pct:.1f}%)"
                )

        # Per-framework breakdown
        if self.framework_stats:
            lines.append("\n🔬 By Framework:")
            for framework, stats in sorted(self.framework_stats.items(), key=lambda x: x[1]["cost"], reverse=True):
                pct = (stats["cost"] / self.total_cost * 100) if self.total_cost > 0 else 0
                lines.append(f"  • {framework}:")
                lines.append(
                    f"    Calls: {stats['calls']:,} | Tokens: {stats['total_tokens']:,} | Cost: ${stats['cost']:.4f} ({pct:.1f}%)"
                )

        # Per-stage breakdown (call-wise cost and time tracking)
        if self.stage_stats:
            lines.append("\n⏱️ By Stage:")
            for stage, stats in sorted(self.stage_stats.items(), key=lambda x: x[1]["cost"], reverse=True):
                pct = (stats["cost"] / self.total_cost * 100) if self.total_cost > 0 else 0
                time_str = f"{stats.get('time', 0.0):.2f}s"
                lines.append(f"  • {stage}:")
                lines.append(
                    f"    Calls: {stats['calls']:,} | Tokens: {stats['total_tokens']:,} | Cost: ${stats['cost']:.4f} ({pct:.1f}%) | Time: {time_str}"
                )

        # Cost efficiency (if framework stats available)
        if self.framework_stats and len(self.framework_stats) > 1:
            lines.append("\n💡 Cost Efficiency:")
            for framework, stats in sorted(self.framework_stats.items(), key=lambda x: x[1]["cost"]):
                cost_per_call = stats["cost"] / stats["calls"] if stats["calls"] > 0 else 0
                lines.append(f"  • {framework}: ${cost_per_call:.4f} per call")

        lines.append("\n" + "=" * 80)

        return "\n".join(lines)

    def export_json(self, filepath: str):
        """Export all calls to JSON"""
        data = {
            "summary": {
                "total_calls": self.total_calls,
                "total_tokens": self.total_tokens,
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "total_cost": self.total_cost,
                "budget_limit": self.budget_limit,
            },
            "model_stats": self.model_stats,
            "framework_stats": self.framework_stats,
            "stage_stats": self.stage_stats,
            "calls": [call.to_dict() for call in self.calls],
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        print(f"✅ Exported {len(self.calls)} calls to {filepath}")

    def export_csv(self, filepath: str):
        """Export calls to CSV for analysis"""
        import csv

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "datetime",
                    "model",
                    "framework",
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "prompt_cost",
                    "completion_cost",
                    "total_cost",
                ]
            )

            for call in self.calls:
                writer.writerow(
                    [
                        call.timestamp,
                        datetime.fromtimestamp(call.timestamp).isoformat(),
                        call.model,
                        call.metadata.get("framework", ""),
                        call.prompt_tokens,
                        call.completion_tokens,
                        call.total_tokens,
                        f"{call.prompt_cost:.6f}",
                        f"{call.completion_cost:.6f}",
                        f"{call.total_cost:.6f}",
                    ]
                )

        print(f"✅ Exported {len(self.calls)} calls to {filepath}")

    def reset(self):
        """Reset all tracking data"""
        self.calls.clear()
        self.total_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        self.model_stats.clear()
        self.framework_stats.clear()
        self.stage_stats.clear()
        self._warned = False


def compare_frameworks(monitors: Dict[str, CostMonitor], accuracy_scores: Optional[Dict[str, float]] = None) -> str:
    """
    Compare costs across multiple frameworks

    Args:
        monitors: Dict of {framework_name: CostMonitor}
        accuracy_scores: Optional dict of {framework_name: accuracy}

    Returns:
        Formatted comparison report
    """
    lines = []
    lines.append("=" * 80)
    lines.append("FRAMEWORK COST COMPARISON")
    lines.append("=" * 80)

    # Collect stats
    framework_data = []
    for name, monitor in monitors.items():
        acc = accuracy_scores.get(name) if accuracy_scores else None
        framework_data.append(
            {
                "name": name,
                "calls": monitor.total_calls,
                "tokens": monitor.total_tokens,
                "cost": monitor.total_cost,
                "accuracy": acc,
                "cost_per_call": monitor.total_cost / monitor.total_calls if monitor.total_calls > 0 else 0,
            }
        )

    # Sort by cost
    framework_data.sort(key=lambda x: x["cost"], reverse=True)

    lines.append(f"\n{'Framework':<25} {'Calls':<10} {'Tokens':<15} {'Cost':<12} {'$/Call':<12}")
    lines.append("-" * 80)

    for data in framework_data:
        lines.append(
            f"{data['name']:<25} "
            f"{data['calls']:<10,} "
            f"{data['tokens']:<15,} "
            f"${data['cost']:<11.4f} "
            f"${data['cost_per_call']:<11.4f}"
        )

    # If accuracy provided, show cost-effectiveness
    if accuracy_scores:
        lines.append(f"\n{'Framework':<25} {'Accuracy':<12} {'Cost':<12} {'Acc/$':<12}")
        lines.append("-" * 80)

        for data in framework_data:
            if data["accuracy"] is not None:
                acc_per_dollar = data["accuracy"] / data["cost"] if data["cost"] > 0 else 0
                lines.append(
                    f"{data['name']:<25} {data['accuracy']:<12.2%} ${data['cost']:<11.4f} {acc_per_dollar:<12.4f}"
                )

    lines.append("\n" + "=" * 80)

    return "\n".join(lines)


# Convenience function for quick cost calculation
def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> Dict[str, float]:
    """
    Quick cost calculation without creating a monitor

    Args:
        model: Model identifier
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens

    Returns:
        Dict with prompt_cost, completion_cost, total_cost
    """
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["default"])

    prompt_cost = (prompt_tokens / 1_000_000) * pricing["prompt"]
    completion_cost = (completion_tokens / 1_000_000) * pricing["completion"]

    return {
        "prompt_cost": prompt_cost,
        "completion_cost": completion_cost,
        "total_cost": prompt_cost + completion_cost,
        "total_tokens": prompt_tokens + completion_tokens,
    }
