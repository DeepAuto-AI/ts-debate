# TS-Debate

This is the official implementation of _Multimodal Collaborative Debate for Zero-Shot Time Series Reasoning_ (EMNLP 2026, Main Conference)
> [[Paper](https://openreview.net/forum?id=8PVTky3Tiz)] [[arXiv](https://arxiv.org/abs/2601.19151)][[Poster](static/pdfs/poster.pdf)][[Slides](static/pdfs/slides.pdf)]



### Citation
```bibtex
@inproceedings{trirat2026tsdebate,
    title={Multimodal Collaborative Debate for Zero-Shot Time Series Reasoning},
    author={Patara Trirat and Jin Myung Kwak and Jay Heo and Heejun Lee and Sung Ju Hwang},
    booktitle={The 2026 Conference on Empirical Methods in Natural Language Processing},
    year={2026},
    url={https://openreview.net/forum?id=8PVTky3Tiz}
}
```

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Installation](#2-installation)
3. [Repository Structure](#3-repository-structure)
4. [Method Overview](#4-method-overview)
5. [Key Components](#5-key-components)
6. [Benchmarks & Tasks](#6-benchmarks--tasks)
7. [Running Experiments](#7-running-experiments)

---

## 1. Quick Start

```bash
# Install dependencies
uv sync

# Run TS-Debate on a single sample
uv run python -m projects.agent_builder.scripts.ts_debate.experiments.cli \
    --experiment main --run-id 1 --n-samples 1 \
    --method ts_debate --benchmark mtbench --task finance_trend \
    --model gpt-4.1-mini --verbose
```

For interactive exploration, see the Jupyter notebooks in `Quick_Start_TS_Debate.ipynb`.

---

## 2. Installation

### Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

```bash
# Clone repository (if not already done)
cd /path/to/ts-debate

# Install all dependencies via uv
uv sync

# Set API keys
export OPENROUTER_API_KEY="your-openrouter-key"
```

### Verify Installation

```bash
# Test import
uv run python -c "from ts_debate.ts_debate import TSDebate; print('✓ TS-Debate imported')"
```

---

## 3. Repository Structure

```
.
├── ts_debate/
├── ts_debate.py                 # Main orchestrator (CrossModalDebateOrchestrator, TSDebate)
├── agent_specs.py               # Agent implementations (Text, Visual, Numerical, JudgeAgent, KnowledgeElicitor)
├── agent_prompts.py             # All prompts, protocols, and evaluation criteria
├── agent_tools.py               # Lookup tools and code executor for agents/reviewers
│
├── experiments/                 # Experiment infrastructure
│   ├── cli.py                   # Command-line entry point
│   ├── configs.py               # Experiment configurations (methods, tasks, ablations)
│   ├── methods.py               # Method factory and result extraction
│   └── runner.py                # Core experiment runner
│
├── data_loaders/                # Benchmark data loaders
│   ├── mtbench_loader.py        # MTBench loader (Finance, Weather)
│   ├── timerbed_loader.py       # TimerBed loader (HAR, ECG, EMG, etc.)
│   ├── tsqa_loader.py           # TSQA loader (Anomaly, Classification, etc.)
│   └── utils.py                 # Data loading utilities
│
├── evaluation/                  # Evaluation infrastructure
│   ├── metrics.py               # Official metrics for all benchmarks
│   ├── task_mixer.py            # Multi-benchmark sample management
│   ├── ablation_study.py        # Ablation analysis utilities
│   └── hyperparameter_study.py  # Hyperparameter sweep analysis
│
├── utils/                       # Utility modules
│   ├── llm_providers.py         # LLM creation with cost tracking
│   ├── task_config.py           # Task configurations and context preparation
│   ├── chart_generator.py       # Task-aware time series visualization
│   ├── frequency_analyzer.py    # Spectral analysis utilities
│   ├── numerical_lookup.py      # Numerical lookup function implementation
│   ├── cost_monitor.py          # Token and cost tracking
│   ├── constants.py             # Task configs, datetime formats
│   └── sample_rate_inference.py # Sample rate detection utilities
│
├── benchmarks/                  # Benchmark datasets
│   ├── MTBench/                 # Finance + Weather tasks
│   ├── TimerBed/                # Time series classification (HAR, ECG, etc.)
│   └── TSQA/                    # Time Series QA tasks
```

---

## 4. Method Overview

### 4.1 What is Collaborative Debate?

TS-Debate uses **collaborative debate** rather than adversarial debate:

- Participants are **teammates exploring an issue together**, not opponents trying to win
- The goal is to **find the best solution** by synthesizing diverse perspectives
- Each modality contributes its unique strengths while acknowledging limitations
- Conflicts are resolved through **verification and domain knowledge**, not by majority vote

This approach is particularly suited for time series reasoning where Text (context), Visual (patterns), and Numerical (precision) modalities each capture different aspects of the data.

### 4.2 Collaborative Debate Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT                                          │
│            Task Description + Time Series + Context                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 0: KNOWLEDGE ELICITATION                           │
│                                                                             │
│  Activate domain knowledge BEFORE data analysis:                            │
│  • DOMAIN: What type of task is this?                                       │
│  • KNOWLEDGE: What domain constraints and patterns apply?                   │
│  • KEY SIGNALS: What features are typically relevant?                       │
│  • SUGGESTED APPROACH: How to analyze data correctly for this task?         │
│  • PITFALLS: Common mistakes to avoid?                                      │
│  • MODALITY: Which modalities to focus on?                                  │
│                                                                             │
│  Output: DOMAIN_KNOWLEDGE (shared with all subsequent stages)               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌───────────────┐           ┌───────────────┐           ┌───────────────┐
│  TEXT ANALYST │           │VISUAL ANALYST │           │  NUM ANALYST  │
│ +DOMAIN_KNOW  │           │ +DOMAIN_KNOW  │           │ +DOMAIN_KNOW  │
│               │           │               │           │               │
│ Input:        │           │ Input:        │           │ Input:        │
│ • Reports     │           │ • Time chart  │           │ • Lookup tools│
│ • Context     │           │ • Freq chart  │           │ • Statistics  │
│               │           │               │           │               │
│ Tools: None   │           │ Tools: None   │           │ Tools: Lookup │
└───────────────┘           └───────────────┘           └───────────────┘
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────┐           ┌───────────────┐           ┌───────────────┐
│  EVIDENCE     │           │  EVIDENCE     │           │  EVIDENCE     │
│ • Understanding│          │ • Understanding│          │ • Understanding│
│ • Observations│           │ • Observations│           │ • Observations│
│ • Inferences  │           │ • Inferences  │           │ • Inferences  │
│ • Limits      │           │ • Limits      │           │ • Limits      │
└───────────────┘           └───────────────┘           └───────────────┘
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   REVIEWERS WITH VCC (M reviewers, parallel)                │
│                                                                             │
│  Score each analyst's evidence (100 points max):                            │
│  • INFERENCE QUALITY (0-50): Logic? Uses domain knowledge? Calibrated?      │
│  • OBSERVATION QUALITY (0-30): Specific? Verifiable? Relevant?              │
│  • HONESTY (0-20): Acknowledged limits? Avoided overclaiming?               │
│                                                                             │
│  VCC Framework (Verification-Conflict-Calibration):                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ V - VERIFY: Check claims against data sources + domain knowledge    │   │
│  │     Format: [VERIFIED/UNVERIFIED/CONTRADICTED] + [DOMAIN: M/V/N-A]  │   │
│  │ C - CONFLICT: Detect conflicts [NO_CONFLICT / DETECTED / RESOLVED]  │   │
│  │ C - CALIBRATE: Match answer to conflict + domain status             │   │
│  │     • NO_CONFLICT + VERIFIED → confident answer                     │   │
│  │     • CONFLICT UNRESOLVED → conservative answer                     │   │
│  │     • DOMAIN_VIOLATION → reject that claim                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Output: CALIBRATED ANSWER                                                  │
│  Tools: Lookup Function + Code Executor + Charts (based on config)          │
│  Tool Limits: Max 5 calls per reviewer                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      FINAL SYNTHESIZER WITH VCC                             │
│                                                                             │
│  NOT a vote counter. Evaluates reviewer REASONING quality.                  │
│                                                                             │
│  STEP 1: Check APPROACH - Did reviewers follow SUGGESTED APPROACH?          │
│  STEP 2: Score reviewers (5 criteria × 20 pts each = 100)                   │
│          Task Understanding, Evidence Usage, Verification,                  │
│          Conflict Handling, Calibration                                     │
│  STEP 3: Check TASK TYPE (FUTURE vs PAST-PRESENT)                           │
│  STEP 4: Evaluate CONFLICT STATUS:                                          │
│          Reviewer Agreement: [UNANIMOUS / SPLIT / ALL_DIFFERENT]            │
│          Resolution: [VERIFIED_RESOLUTION / UNRESOLVED / APPROACH_ERROR]    │
│  STEP 5: Derive CALIBRATED final answer                                     │
│                                                                             │
│  Note: "Perfect score is IMPOSSIBLE" - always find reviewer flaws           │
│                                                                             │
│  Output: FINAL ANSWER                                                       │
│  Tools: Lookup + Code Executor + Charts (based on config)                   │
│  Tool Limits: Max 3 calls                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                        ┌───────────────────┐
                        │  FINAL ANSWER     │
                        └───────────────────┘
```

---

## 5. Key Components

### 5.1 Analysts

| Agent | Input | Output | Tools |
|-------|-------|--------|-------|
| **KnowledgeElicitor** | Task description (text only) | Domain knowledge | None |
| **TextAgent** | Reports, descriptions, text context | Evidence (observations + inferences) | None |
| **VisualAgent** | Time series chart, frequency chart | Evidence (observations + inferences) | None |
| **NumericalAgent** | Lookup function access | Evidence (observations + inferences) | `get_info`, `get_values`, `get_around`, `get_features`, `get_frequency_features`, + conditional tools (max 5 calls) |

### 5.2 Reviewers & Synthesizer

| Component | Role | Output Field | Tools |
|-----------|------|--------------|-------|
| **Reviewer** (JudgeAgent) | Evaluate evidence, verify claims, detect conflicts, synthesize answer | `CALIBRATED ANSWER` | Lookup + Code Executor + Charts (max 5 calls) |
| **Synthesizer** (JudgeAgent, is_synthesizer=True) | Compare reviewer answers, verify disputed answers, derive final decision | `FINAL ANSWER` | Lookup + Code Executor + Charts (max 3 calls) |

### 5.3 Tools

| Tool | Used By | Description |
|------|---------|-------------|
| `get_info()` | Numerical Analyst, Reviewers | Time series schema, statistics, detected features, indicator info |
| `get_values(start, end)` | Numerical Analyst, Reviewers | Query values by index or timestamp (max 100 values) |
| `get_around(center, window)` | Numerical Analyst, Reviewers | Values around a specific point (max window: 50) |
| `get_features(type)` | Numerical Analyst, Reviewers | Peaks, troughs, anomalies, change points |
| `get_frequency_features()` | Numerical Analyst, Reviewers | Spectral analysis features |
| `get_channel_values(ch, start, end)` | Numerical Analyst, Reviewers | Specific channel (multivariate data only) |
| `get_all_channels(start, end)` | Numerical Analyst, Reviewers | All channels at once (multivariate data only) |
| `get_indicator(start, end)` | Numerical Analyst, Reviewers | Technical indicator values (MACD, BB) |
| `execute_code(code)` | Reviewers, Synthesizer | Python code execution for verification |

---

## 6. Benchmarks & Tasks

### 6.1 Overview

| Benchmark | Domain | Tasks | Task Types | Paper |
|-----------|--------|-------|------------|-------|
| **MTBench** | Finance, Weather | 9 | Classification, Regression, MCQA | arXiv:2503.16858 |
| **TimerBed** | Sensors | 6 | Classification | NAACL 2025 (VL-Time) |
| **TSQA** | General | 5 | Classification, Regression, QA | ACL 2025 (Time-MQA) |
| **Total** | - | **20** | - | - |

### 6.2 Task Details

#### MTBench (9 tasks)

| Task Key | Domain | Type | Description |
|----------|--------|------|-------------|
| `finance_trend` | Finance | Classification | Predict stock price trend category |
| `finance_forecasting` | Finance | Regression | Forecast future stock prices |
| `finance_indicator_macd` | Finance | Regression | Predict MACD indicator values |
| `finance_correlation` | Finance | Classification | Identify correlated stocks |
| `finance_mcqa` | Finance | MCQA | Multiple-choice finance questions |
| `weather_trend` | Weather | Classification | Predict temperature trend |
| `weather_forecasting` | Weather | Regression | Forecast temperatures |
| `weather_indicator_macd` | Weather | Regression | Predict temperature indicators |
| `weather_mcqa` | Weather | MCQA | Multiple-choice weather questions |

#### TimerBed (6 tasks)

| Task Key | Dataset | Classes | Labels (Full) |
|----------|---------|---------|---------------|
| `ECG` | ECG | 4 | normal sinus rhythm, fibrillation, alternative rhythm, too noisy to be classified |
| `HAR` | HAR | 6 | walking, walking upstairs, walking downstairs, sitting, standing, laying down |
| `EMG` | EMG | 3 | healthy, suffering from neuropathy, suffering from myopathy |
| `CTU` | CTU | 2 | desktop, laptop |
| `RCW` | RCW | 2 | "There is no right whale call in the image.", "There is a right whale call in the image." |
| `TEE` | TEE | 7 | CG Positive Initial Return Stroke, IR Negative Initial Return Stroke, SR Subsequent Negative Return Stroke, I Impulsive Event, I2 Impulsive Event Pair, KM Gradual Intra-Cloud Stroke, O Off-record |

#### TSQA (5 tasks)

| Task Key | Type | Description |
|----------|------|-------------|
| `anomaly` | Classification | Anomaly detection (Normal/Anomaly) |
| `classification` | Classification | General time series classification |
| `forecasting` | Regression | Forecast future values |
| `imputation` | Regression | Impute missing values |
| `qa` | QA | True/False, MCQ, Open-ended questions |

---

## 7. Running Experiments

```bash
uv run python -m projects.agent_builder.scripts.ts_debate.experiments.cli \
    --experiment main --run-id 1 --n-samples 100 \
    --method ts_debate --benchmark mtbench --task finance_trend \
    --model gpt-4.1-mini
```

## License

MIT License. See `LICENSE` file.
