"""
Provides unified interface for creating method instances and extracting results.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..utils.cost_monitor import CostMonitor


@dataclass
class MethodResult:
    """Unified result format for all methods."""

    response: str
    trace: str  # Full reasoning/debate trace
    filled_instruction: str  # The actual task instruction sent to LLM


def create_method(method_name: str, config: Dict[str, Any], monitor: Optional[CostMonitor] = None) -> Any:
    """
    Create method instance with config.

    Args:
        method_name: One of ts_debate, single_agent, single_agent_cot,
                     single_agent_mm, mad, mad_multimodal, vltime, bymyeyes
        config: Method configuration dict
        monitor: Optional CostMonitor for tracking tokens

    Returns:
        Method instance with run_debate() method

    Raises:
        ValueError: If method_name is unknown
    """
    from ..ts_debate import TSDebate

    common = {
        "provider": config.get("provider", "openrouter"),
        "model": config.get("model", "openai/gpt-4.1-mini"),
        "verbose": config.get("verbose", False),
        "monitor": monitor,
    }

    if method_name == "ts_debate":
        return TSDebate(
            **common,
            num_judges=config.get("num_judges", 3),
            max_rounds=config.get("max_rounds", 2),
            max_judge_rounds=config.get("max_judge_rounds", 1),
            use_lookup=config.get("use_lookup", True),
            judge_use_code_executor=config.get("judge_use_code_executor", True),
            judge_use_lookup=config.get("judge_use_lookup", True),
            judge_only=config.get("judge_only", False),
            use_text_agents=config.get("use_text_agents", True),
            use_visual_agents=config.get("use_visual_agents", True),
            use_numerical_agents=config.get("use_numerical_agents", True),
            use_frequency_features=config.get("use_frequency_features", True),
            judge_use_time_chart=config.get("judge_use_time_chart", True),
            judge_use_freq_chart=config.get("judge_use_freq_chart", True),
            use_knowledge_elicitation=config.get("use_knowledge_elicitation", True),
        )
    raise ValueError(f"Unknown method: {method_name}")


def extract_result(method_instance: Any, raw_result: tuple, method_name: str) -> MethodResult:
    """
    Extract response, trace, and filled_instruction from method result.

    All methods return (result, filled_instruction) tuple.

    Args:
        method_instance: The method instance that produced the result
        raw_result: Tuple of (result, filled_instruction) from run_debate()
        method_name: Method identifier for type-specific extraction

    Returns:
        MethodResult with response, trace, and filled_instruction

    Raises:
        TypeError: If result type doesn't match expected
        ValueError: If method_name is unknown
    """
    # All methods return (result, filled_instruction) tuple
    if not isinstance(raw_result, tuple) or len(raw_result) != 2:
        raise TypeError(f"Expected (result, filled_instruction) tuple, got {type(raw_result)}")

    result, filled_instruction = raw_result

    if method_name == "ts_debate":
        # TSDebate returns (DebateState, filled_instruction)
        if not hasattr(result, "final_decision"):
            raise TypeError(f"TSDebate expected DebateState with final_decision, got {type(result)}")

        response = result.final_decision or ""

        # Combine agent debate messages + judge/synthesizer traces
        trace_parts = []

        # Agent debate messages
        if hasattr(result, "messages") and result.messages:
            trace_parts.append("=== AGENT DEBATE ===")
            for m in result.messages:
                trace_parts.append(f"[{m.agent_id}] (Round {m.round_number}, {m.modality.value}):\n{m.content}")

        # Judge and synthesizer traces
        if hasattr(method_instance, "orchestrator") and hasattr(method_instance.orchestrator, "judge_traces"):
            if method_instance.orchestrator.judge_traces:
                trace_parts.append("\n=== JUDGE & SYNTHESIZER ===")
                trace_parts.extend(method_instance.orchestrator.judge_traces)

        trace = "\n\n".join(trace_parts)

        return MethodResult(response=response, trace=trace, filled_instruction=filled_instruction)

    raise ValueError(f"Unknown method: {method_name}")
