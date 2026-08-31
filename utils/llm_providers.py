"""
LLM Providers for TS-Debate Framework

Reproducibility defaults:
- DEFAULT_TEMPERATURE = 0.0  (deterministic/greedy decoding)
- DEFAULT_SEED = 42          (fixed random seed)

Note: OpenAI's seed parameter provides "best effort" reproducibility.
Results should be highly consistent but not guaranteed 100% identical
across different infrastructure or API versions.
"""

import os
from typing import Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from typing_extensions import override

from .cost_monitor import CostMonitor

# Reproducibility constants
DEFAULT_TEMPERATURE = 0.0  # Deterministic outputs
DEFAULT_SEED = 42  # Fixed seed for reproducibility

AVAILABLE_MODELS = {
    # OpenRouter models - OpenAI
    "gpt-4.1": {"provider": "openrouter", "model": "openai/gpt-4.1"},
    "gpt-4.1-mini": {"provider": "openrouter", "model": "openai/gpt-4.1-mini"},
    "gpt-4o": {"provider": "openrouter", "model": "openai/gpt-4o"},
    "gpt-5": {"provider": "openrouter", "model": "openai/gpt-5"},
    "gpt-5-mini": {"provider": "openrouter", "model": "openai/gpt-5-mini"},
    "gpt-5.1": {"provider": "openrouter", "model": "openai/gpt-5.1"},
    "gpt-5.2": {"provider": "openrouter", "model": "openai/gpt-5.2"},
    # OpenRouter models - Google
    "gemini-2.5-flash": {"provider": "openrouter", "model": "google/gemini-2.5-flash-preview-09-2025"},
    "gemini-2.5-pro": {"provider": "openrouter", "model": "google/gemini-2.5-pro"},
    "gemini-3-pro": {"provider": "openrouter", "model": "google/gemini-3-pro-preview"},
    # OpenRouter models - Qwen
    "qwen3-vl": {"provider": "openrouter", "model": "qwen/qwen3-vl-235b-a22b-thinking"},
    # OpenRouter models - xAI (2M context, reasoning optional)
    "grok-4.1-fast": {"provider": "openrouter", "model": "x-ai/grok-4.1-fast"},
    # OpenRouter models - Zhipu AI (GLM)
    "glm-4.6v": {"provider": "openrouter", "model": "z-ai/glm-4.6v"}
}


class CostTrackingCallback(BaseCallbackHandler):
    """
    LangChain callback that integrates with CostMonitor.

    Use this when creating ChatOpenAI models to track costs.
    """

    def __init__(self, monitor: CostMonitor, model: str, framework: str = "TS-Debate"):
        self.monitor = monitor
        self.model = model
        self.framework = framework

    @override
    def on_llm_end(self, response, **kwargs) -> None:
        """Track token usage after each LLM call."""
        del kwargs  # Unused but required by interface
        if hasattr(response, "llm_output") and response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            if usage:
                _ = self.monitor.log_call(
                    model=self.model,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    framework=self.framework,
                )


def create_chat_model(
    provider: str = "openrouter",
    model: str = "openai/gpt-4.1-mini",
    api_key: Optional[str] = None,
    monitor: Optional[CostMonitor] = None,
    framework: str = "TS-Debate",
    temperature: float = DEFAULT_TEMPERATURE,
    seed: int = DEFAULT_SEED,
    max_tokens: int = 2048,
    *,  # Force keyword-only arguments after this
    disable_reasoning: bool = True,  # Disable reasoning tokens by default for speed
    **kwargs,
) -> ChatOpenAI:
    """
    Create a LangChain ChatOpenAI model compatible with OpenRouter.

    This is the LLM interface for all frameworks (TS-Debate, MAD, VL-Time, etc.)

    Reproducibility:
        - temperature=0.0: Deterministic (greedy) decoding
        - seed=42: Fixed random seed (OpenAI API feature)

        Note: Even with seed, OpenAI does not guarantee 100% identical outputs
        due to infrastructure changes, but results should be highly consistent.

    Args:
        provider: "openrouter", or "openai"
        model: Model identifier (e.g., "openai/gpt-4.1-mini" for openrouter, "gpt-4.1-mini" for openai)
        api_key: API key (or uses env var)
        monitor: Optional CostMonitor for tracking
        framework: Framework name for cost aggregation
        temperature: Model temperature (default: 0.0 for reproducibility)
        seed: Random seed for reproducibility (default: 42)
        disable_reasoning: Disable reasoning/thinking tokens (default: True)
            - True: Disable reasoning tokens entirely (fastest, for OpenRouter)
            - False: Allow reasoning tokens (for thinking models)
        **kwargs: Additional model parameters

    Returns:
        ChatOpenAI instance
    """
    callbacks = []
    if monitor:
        callbacks.append(CostTrackingCallback(monitor, model, framework))

    # Extract model_kwargs from kwargs (excluding seed which is passed directly)
    model_kwargs = kwargs.get("model_kwargs", {}).copy()
    # Remove seed from model_kwargs if present (it should be passed directly)
    model_kwargs.pop("seed", None)

    # Using extra_body instead of native 'reasoning' to avoid Responses API conflicts
    # See: https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
    extra_body = kwargs.get("extra_body", {}).copy()
    if disable_reasoning and "gpt-5" not in model:
        extra_body["reasoning"] = {"enabled": False, "effort": "none"}
    else:
        extra_body["reasoning"] = {"effort": "none"}

    # Exclude Azure to avoid strict function schema validation issues
    # extra_body["provider"] = {"ignore": ["Azure"]}

    if provider == "openrouter":
        api_key_str = api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key_str:
            msg = "OpenRouter API key required. Set OPENROUTER_API_KEY env var."
            raise ValueError(msg)
        return ChatOpenAI(
            model=model,
            base_url="https://openrouter.ai/api/v1",
            api_key=SecretStr(api_key_str),
            temperature=temperature,
            seed=seed,
            max_completion_tokens=max_tokens,
            timeout=300,
            max_retries=5,
            model_kwargs=model_kwargs,
            extra_body=extra_body if extra_body else None,  # Pass reasoning config via extra_body
            callbacks=callbacks if callbacks else None,
            **{k: v for k, v in kwargs.items() if k not in ("model_kwargs", "extra_body")},
        )

    if provider == "openai":
        api_key_str = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key_str:
            msg = "OpenAI API key required. Set OPENAI_API_KEY env var."
            raise ValueError(msg)
        return ChatOpenAI(
            model=model.split("/")[1],
            api_key=SecretStr(api_key_str),
            temperature=temperature,
            seed=seed,
            max_completion_tokens=max_tokens,
            timeout=300,
            max_retries=5,
            model_kwargs=model_kwargs,
            callbacks=callbacks if callbacks else None,
            **{k: v for k, v in kwargs.items() if k not in ("model_kwargs", "extra_body")},
        )

    msg = f"Unsupported provider: {provider}. Use 'openrouter' or 'openai'."
    raise ValueError(msg)


def get_chat_model(
    model_name: str,
    api_key: Optional[str] = None,
    monitor: Optional[CostMonitor] = None,
    framework: str = "TS-Debate",
) -> ChatOpenAI:
    """
    Get ChatOpenAI for an available model with optional cost monitoring.

    Args:
        model_name: Model name from AVAILABLE_MODELS
        api_key: API key if needed
        monitor: Optional CostMonitor instance for cost tracking
        framework: Framework name for cost aggregation

    Returns:
        ChatOpenAI instance
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(AVAILABLE_MODELS.keys())}")

    config = AVAILABLE_MODELS[model_name]
    return create_chat_model(
        provider=config["provider"],
        model=config["model"],
        api_key=api_key,
        monitor=monitor,
        framework=framework,
    )


__all__ = [
    # Reproducibility constants
    "DEFAULT_TEMPERATURE",
    "DEFAULT_SEED",
    # Model registry
    "AVAILABLE_MODELS",
    # LLM interface (unified)
    "create_chat_model",
    "get_chat_model",
    # Monitoring
    "CostTrackingCallback",
    "CostMonitor",
]
