# FraudRecon Architecture

## Scope

The repository contains FraudRecon-Bench evaluation and NDRR dialogue construction.

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

- `bench/core/evaluator.py`
  Orchestrates dataset loading, model calls, retries, persistence, and summary aggregation.
- `bench/data/dataset.py`
  Loads `index / labels / samples`, builds dialogue text, and normalizes phase aliases to the paper schema.
- `bench/data/prompt_builder.py`
  Builds prompts for text, audio, and multimodal evaluation.
- `bench/adapters/openai_compatible.py`
  Sends OpenAI-compatible requests using the configured endpoint and credentials.
- `bench/scoring/metrics.py`
  Implements accuracy, weighted F1, macro F1, and multilabel F1.
- `bench/scoring/reasoning.py`
  Aggregates judge votes into the final `T5` reasoning score.
- `bench/api/server.py`
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

T3 uses text prefixes in every input mode. The T5 judge receives text. Aggregate metrics include successfully completed cases; failure counts and errors are reported separately.

## Local Workflow

1. Extract benchmark data to a directory and pass it with `--dataset-root`; `bench/examples/` supplies the included demo
2. Configure models in `bench/configs/models.yaml`
3. Run `python -m bench.cli ...` or start `bench/scripts/start_api.ps1`
4. Outputs are written to `outputs/bench/<run_id>/`


## NDRR construction component

`ndrr/cli.py` connects FPE, FADR, post-dialogue signal annotation, evidence-chain annotation and export. `ndrr/fadr` coordinates the Scammer, Victim and Director agents with a sliding context window. Signal definitions are in `ndrr/annotation/signal_rules.txt`; evidence-chain prompts are in `ndrr/annotation/evidence_chain_prompts.py`. `ndrr/export.py` packages completed dialogues and annotations into the sample/index/label layout accepted by the benchmark loader.

Intermediate samples are written under `annotations/input/`. Final dataset files are updated after annotation succeeds and validates. Signal annotation checkpoints completed cases; reconstruction export resolves appended retries to the latest record.

Benchmark task modes are defined in `bench/modes.py`. The loader reads the requested split's index, labels and samples. Run reuse checks model/task settings and case inventory before loading completed results.
