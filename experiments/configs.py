from typing import Dict, List, Set

# All methods to evaluate
MAIN_METHODS: List[str] = ["ts_debate"]

# All 20 tasks organized by benchmark
TASKS: Dict[str, List[str]] = {
    # MTBench (9 tasks)
    "mtbench": [
        # Finance (5)
        "finance_trend",
        "finance_forecasting",
        "finance_indicator_macd",
        "finance_correlation",
        "finance_mcqa",
        # Weather (4)
        "weather_trend",
        "weather_forecasting",
        "weather_indicator_macd",
        "weather_mcqa",
    ],
    # TimerBed (6 tasks)
    "timerbed": [
        "ECG",
        "HAR",
        "EMG",
        "CTU",
        "RCW",
        "TEE",
    ],
    # TSQA (5 tasks)
    "tsqa": [
        "anomaly",
        "classification",
        "forecasting",
        "imputation",
        "qa",
    ]
}

# Task types for evaluation metrics
TASK_TYPES: Dict[str, str] = {
    # MTBench Finance
    "mtbench_finance_trend": "classification",
    "mtbench_finance_forecasting": "regression",
    "mtbench_finance_indicator_macd": "regression",
    "mtbench_finance_correlation": "classification",
    "mtbench_finance_mcqa": "mcqa",
    # MTBench Weather
    "mtbench_weather_trend": "classification",
    "mtbench_weather_forecasting": "regression",
    "mtbench_weather_indicator_macd": "regression",
    "mtbench_weather_mcqa": "mcqa",
    # TimerBed (all classification)
    "timerbed_ecg": "classification",
    "timerbed_har": "classification",
    "timerbed_emg": "classification",
    "timerbed_ctu": "classification",
    "timerbed_rcw": "classification",
    "timerbed_tee": "classification",
    # TSQA
    "tsqa_anomaly": "classification",
    "tsqa_classification": "classification",
    "tsqa_forecasting": "regression",
    "tsqa_imputation": "regression",
    "tsqa_qa": "qa"
}


# Default hyperparameters (baseline for comparison)
DEFAULT_HYPERPARAMS = {
    "num_judges": 3,
    # 2 rounds: Evidence presentation + Refinement with cross-modal insights
    "max_rounds": 2,
    "max_judge_rounds": 1,
}

LLM_MODELS = ["gpt-4.1-mini", "gemini-2.5-flash", "grok-4.1-fast"]

# Main experiments: 100 samples per task
MAIN_N_SAMPLES: int = 100
