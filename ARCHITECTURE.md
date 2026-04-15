# FraudRecon-Bench Architecture

## Scope

This repository implements the evaluation side of the FraudRecon benchmark.

The paper defines five tasks:

- `T1` Fraud Detection
- `T2` Category Classification
- `T3` Phase Identification
- `T4` Interaction Signal Recognition
- `T5` Evidence-Chain Reasoning

The framework executes them in three logical steps:

1. `Round 1`: one model call for `T1 + T2 + T4`
2. `Round 2`: prefix-based phase prediction for `T3`
3. `Judge`: reasoning quality scoring for `T5`

## Key Modules

- `app/core/evaluator.py`
  Orchestrates dataset loading, model calls, retries, persistence, and summary aggregation.
- `app/data/dataset.py`
  Loads `index / labels / samples`, builds dialogue text, and normalizes phase aliases to the paper schema.
- `app/data/prompt_builder.py`
  Builds prompts for text, audio, and multimodal evaluation.
- `app/adapters/openai_compatible.py`
  Sends OpenAI-compatible requests and now omits the `Authorization` header when no key is provided.
- `app/scoring/metrics.py`
  Implements accuracy, weighted F1, macro F1, and multilabel F1.
- `app/scoring/cot.py`
  Aggregates judge votes into the final `T5` reasoning score.
- `app/api/server.py`
  Exposes the local FastAPI endpoints for programmatic evaluation.

## Data Contract

Each case requires:

- `test_index.jsonl`: `{ "case_id": "..." }`
- `test_labels.jsonl`: `{ "case_id": "...", "labels": {...} }`
- `samples/<case_id>.json`

The important label fields are:

- `is_fraud`
- `level_1_category`
- `level_2_phases`
- `level_3_signals`

The canonical phase labels are:

- Fraud: `initial_contact`, `trust_building`, `inducement`, `operation_guidance`, `fund_transfer`, `sustained_pressure_discovery`
- Benign: `opening`, `core_service`, `closure`

Backward-compatible aliases still supported by the loader:

- `sustained_pressure` -> `sustained_pressure_discovery`
- `discovery` -> `sustained_pressure_discovery`
- `initial_contact` (benign) -> `opening`
- `service_handling` -> `core_service`

## Output Metrics

The paper-aligned batch summary keys are:

- `T1_detection_accuracy`
- `T2_category_weighted_f1`
- `T3_phase_macro_f1`
- `T4_signal_macro_sample_f1`
- `T5_reasoning_score_5_avg`

Legacy aliases are kept in the JSON output for compatibility with older scripts.

## Local Workflow

1. Put benchmark files under `data/FraudRecon-Bench`
2. Configure models in `configs/models.yaml`
3. Run `python -m app.cli ...` or start `scripts/start_api.ps1`
4. Outputs are written to `runs/<run_id>/`
