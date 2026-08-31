#!/usr/bin/env python3
"""
Entry Point for Experiments

Each run can be executed separately.
Use --run-id to specify which run (1, 2, 3).
Samples are selected using seed = base_seed + run_id for reproducibility.

Usage:
    # Run 1 of main experiments (100 samples)
    uv run python -m projects.agent_builder.scripts.ts_debate.experiments.cli \
        --experiment main --run-id 1 --n-samples 100

Output structure:
    results/{experiment}/
    ├── run_1_gpt-4.1-mini/
    │   └── {method}/{task}/sample_000/...
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()

from ..evaluation.task_mixer import TaskMixer
from ..utils.llm_providers import AVAILABLE_MODELS
from .configs import (
    ADDITIONAL_N_SAMPLES,
    AGENT_CONFIGS_ABLATION,
    COMPONENT_ABLATION,
    DEFAULT_HYPERPARAMS,
    HYPERPARAMS,
    JUDGE_TOOLS_ABLATION,
    MAIN_METHODS,
    MAIN_N_SAMPLES,
    MODALITY_ABLATION,
)
from .runner import run_single_run


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CLI for Running Experiments", formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__
    )

    # Experiment type
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=[
            "main",
            "ablation_modality",
            "ablation_judge_tools",
            "ablation_agent_configs",
            "ablation_components",
            "hyperparameter_judges",
            "hyperparameter_rounds",
        ],
        help="Experiment type to run",
    )

    # Run ID (for independent runs)
    parser.add_argument(
        "--run-id",
        type=int,
        required=True,
        help="Run identifier (1, 2, 3, etc.). Each run uses seed = base_seed + run_id",
    )

    # Filtering
    parser.add_argument("--method", type=str, required=True, choices=MAIN_METHODS, help="Method to run")
    parser.add_argument(
        "--benchmark", type=str, required=True, choices=["mtbench", "timerbed", "tsqa"], help="Benchmark"
    )
    parser.add_argument("--task", type=str, required=True, help="Task (e.g., finance_trend)")

    # Sampling
    parser.add_argument("--n-samples", type=int, default=10, help="Samples per task")
    parser.add_argument(
        "--seed", type=int, default=2025, help="Base random seed (default: 2025). Actual seed = base + run_id"
    )

    # Model
    parser.add_argument(
        "--model", type=str, choices=AVAILABLE_MODELS.keys(), default="gemini-2.5-flash", help="LLM model to use"
    )
    parser.add_argument("--provider", type=str, default="openrouter", help="LLM provider")

    # Output
    default_output = Path(__file__).parent.parent / "results"
    parser.add_argument("--output", type=str, default=str(default_output), help="Output directory")

    # Verbosity
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    run_id = args.run_id

    # Determine n_samples
    if args.n_samples:
        n_samples = args.n_samples
    elif args.experiment == "main":
        n_samples = MAIN_N_SAMPLES
    else:
        n_samples = ADDITIONAL_N_SAMPLES

    # Determine method (single value, not list)
    method = args.method

    # Create output directory: results/{experiment}/run_{id}_{model}/
    output_dir = Path(args.output) / args.experiment / f"run_{run_id}_{args.model}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute actual seed for this run
    actual_seed = args.seed + run_id

    # Save run config
    config_path = output_dir / "run_config.json"
    if not config_path.exists():
        config_path.write_text(
            json.dumps(
                {
                    "experiment": args.experiment,
                    "run_id": run_id,
                    "n_samples": n_samples,
                    "base_seed": args.seed,
                    "actual_seed": actual_seed,
                    "model": args.model,
                    "provider": args.provider,
                }
            )
        )

    verbose = args.verbose
    if verbose:
        print("Experiment Runner CLI")
        print("=" * 60)
        print(f"Experiment: {args.experiment}")
        print(f"Run ID: {run_id}")
        print(f"Method: {method}")
        print(f"Samples per task: {n_samples}")
        print(f"Seed: {actual_seed} (base {args.seed} + run_id {run_id})")
        print(f"Model: {args.model}")
        print(f"Output: {output_dir}")
        print("=" * 60)

    # Load samples using TaskMixer with run-specific seed
    if verbose:
        print(f"\nLoading data with seed {actual_seed}...")

    # Build filters for TaskMixer
    benchmarks = [args.benchmark]
    task_name = args.task

    # Strip benchmark prefix if present
    if task_name.startswith(f"{args.benchmark}_"):
        task_name = task_name[len(f"{args.benchmark}_") :]

    # TimerBed uses uppercase task names
    if args.benchmark == "timerbed":
        task_name = task_name.upper()

    include_tasks = {args.benchmark: [task_name]}

    mixer = TaskMixer(seed=actual_seed, verbose=verbose)
    eval_sets = mixer.create_evaluation_set(
        n_per_task=n_samples,
        benchmarks=benchmarks,
        include_tasks=include_tasks,
    )

    if not eval_sets:
        print("ERROR: No tasks matched the filter criteria.")
        sys.exit(1)

    # Method config - use full model ID from AVAILABLE_MODELS
    full_model_id = AVAILABLE_MODELS[args.model]["model"]
    method_config = {
        "provider": args.provider,
        "model": full_model_id,
        "verbose": verbose,
        **DEFAULT_HYPERPARAMS,
    }

    # Determine ablation/hyperparameter configs
    ablation_configs = None
    hyperparams_study = None

    if args.experiment == "ablation_modality":
        ablation_configs = MODALITY_ABLATION
    elif args.experiment == "ablation_judge_tools":
        ablation_configs = JUDGE_TOOLS_ABLATION
    elif args.experiment == "ablation_agent_configs":
        ablation_configs = AGENT_CONFIGS_ABLATION
    elif args.experiment == "ablation_components":
        ablation_configs = COMPONENT_ABLATION
    elif args.experiment == "hyperparameter_judges":
        hyperparams_study = {"num_judges": HYPERPARAMS["num_judges"]}
    elif args.experiment == "hyperparameter_rounds":
        hyperparams_study = {
            "max_rounds": HYPERPARAMS["max_rounds"],
            "max_judge_rounds": HYPERPARAMS["max_judge_rounds"],
        }

    # Run single independent run
    summaries = run_single_run(
        method=method,
        eval_sets=eval_sets,
        output_dir=output_dir,
        method_config=method_config,
        run_id=run_id,
        ablation_configs=ablation_configs,
        hyperparams_study=hyperparams_study,
        verbose=verbose,
    )

    # Print summary
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"RUN {run_id} COMPLETE")
        print(f"{'=' * 60}")
        print(f"Tasks completed: {len(summaries)}")
        total_tokens = sum(s.get("total_tokens", 0) for s in summaries)
        total_time = sum(s.get("total_time", 0) for s in summaries)

        print(f"Total tokens: {total_tokens:,}")
        print(f"Total time: {total_time:.1f}s ({total_time / 60:.1f}m)")

        print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
