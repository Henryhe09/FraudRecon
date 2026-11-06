# FraudRecon-Bench

Evaluation code for the five FraudRecon tasks. Run commands from the **repository root**.

## Tasks and execution

| Task | Evaluation target | Summary metric |
| --- | --- | --- |
| T1: Fraud Detection | Scam or benign | `T1_detection_accuracy` |
| T2: Category Classification | Fraud/service category | `T2_category_weighted_f1` |
| T3: Phase Identification | Procedural phase | `T3_phase_macro_f1` |
| T4: Signal Recognition | Applicable interaction signals | `T4_signal_macro_sample_f1` |
| T5: Evidence-Chain Reasoning | Entities, phases, signals and verdict | `T5_reasoning_score_5_avg` |

T1/T2/T4 use the complete dialogue. T3 uses prefixes ending at each annotated phase boundary. T5 uses a separate reasoning judge. Prompts are in [`prompts/`](prompts/); metrics are in [`scoring/`](scoring/).

The selected input mode applies to full-dialogue generation. T3 always receives text prefixes; the T5 judge receives the transcript, reference labels and generated reasoning as text.

## Run the included demo

Install the project from the repository root:

```bash
python -m pip install -e .
```

Run the two fictional examples:

```bash
python -m bench.cli \
  --model-id mock-main \
  --judge-model-id mock-judge \
  --split test \
  --input-mode text
```

The mock demo reads [`examples/`](examples/) and writes results to `outputs/bench/` without API calls. Omitting `--dataset-root` selects this demo directory (`bench/examples/`).

## Configure your models

In [`configs/models.yaml`](configs/models.yaml), configure the evaluated model and the reasoning judge:

```yaml
models:
  evaluated-model:
    provider: openai_compatible
    base_url: ""       # Your API base URL
    model_name: ""     # Model name accepted by the endpoint
    api_key_env: OPENAI_API_KEY
    supports_audio: false

  reasoning-judge:
    provider: openai_compatible
    base_url: ""       # Your judge API base URL
    model_name: ""     # Judge model name
    api_key_env: JUDGE_API_KEY
    supports_audio: false
```

`--model-id` and `--judge-model-id` refer to these profile names. Credentials are read from the named environment variables:

```bash
export OPENAI_API_KEY="your-model-api-key"
export JUDGE_API_KEY="your-judge-api-key"
```

The model and judge may share a deployment. For audio inputs, set `supports_audio: true` on the evaluated model profile.

## Prepare benchmark data

Pass the extracted benchmark directory with `--dataset-root`. Its layout should be:

```text
<dataset-root>/
  test_index.jsonl
  test_labels.jsonl
  samples/
    <case_id>.json
  audio/
    ...
```

The loader also accepts this layout inside `<dataset-root>/test/`.

| File | Content |
| --- | --- |
| `test_index.jsonl` | Case list; each row contains `case_id` |
| `test_labels.jsonl` | Each row contains `case_id` and `labels` |
| `samples/<case_id>.json` | Dialogue in `reconstructed_dialogue.flat_turns`; `audio_path` for audio inputs |

Evaluation labels come from `<split>_labels.jsonl`, taking precedence over any sample-level `labels`.

Required label fields:

| Field | Format |
| --- | --- |
| `is_fraud` | JSON boolean |
| `level_1_category` | Category string |
| `level_2_phases` | Phase instances, each with `phase_id`, `type` and inclusive `[start_turn, end_turn]` in `turn_range` |
| `level_3_signals` | List of signal identifier strings; `[]` is valid |

Each sample provides `case_id` and `reconstructed_dialogue.flat_turns`; each turn has integer `turn_id`, `speaker`, and `utterance`. Audio inputs also require `audio_path`.

`phase_id` identifies a phase instance; `type` identifies its phase category. The same phase type may occur in multiple instances. See [`examples/`](examples/) for complete sample and label files.

For another split, use the matching `<split>_index.jsonl` and `<split>_labels.jsonl` filenames.

Phase types are mapped as follows during evaluation:

- Scam: `sustained_pressure` and `discovery` → `sustained_pressure_discovery`.
- Benign: `initial_contact` → `opening`; `service_handling` → `core_service`.

## Run evaluation

```bash
python -m bench.cli \
  --model-id evaluated-model \
  --judge-model-id reasoning-judge \
  --dataset-root /path/to/benchmark \
  --split test \
  --input-mode text
```

For an audio-capable model, set `--input-mode audio` or `--input-mode multimodal`, and ensure the samples' audio paths resolve under the dataset directory.

Useful options:

- `--max-workers`: concurrent case evaluations.
- `--run-id`: resume a named run with the same model/task settings and case inventory; mismatches require a new run ID. Concurrency and raw-response saving may change. Keep input files unchanged when resuming.
- `--dump-raw`: save model responses with the run artifacts.
- `--eval-mode`: select the tasks below.

| Mode | Tasks |
| --- | --- |
| `full` | T1–T5 |
| `without_phases` | T1/T2/T4/T5 |
| `dialogue_only` | T1/T2/T4 |
| `detection_only` | T1 |
| `category_only` | T2 |
| `signals_only` | T4 |
| `phase_only` | T3 |
| `reasoning_only` | Generate an evidence-chain response and apply the T5 judge |

## Results and configuration

Each run writes:

```text
outputs/bench/<run_id>/
  summary.json      Aggregate metrics and success/failure counts
  cases.json        Per-case predictions and metrics
  errors.json       Failed cases and error messages
  raw_responses/    Present when --dump-raw is enabled
```

Run state is stored in `outputs/bench/eval_runs.db`.

Task metrics are computed over successfully completed cases. `num_cases_total`, `num_cases_success`, `num_cases_failed` and `errors.json` report coverage and failures. A case that fails a later task is excluded from all aggregate task metrics for that run; compare scores together with these counts.

- [`configs/models.yaml`](configs/models.yaml): model profiles and connection settings.
- [`configs/runtime.yaml`](configs/runtime.yaml): dataset/output paths, concurrency and retries.
- [`configs/eval_tasks.yaml`](configs/eval_tasks.yaml): task stages and reasoning-judge settings.

## Code entry points

| Path | Purpose |
| --- | --- |
| [`cli.py`](cli.py) | Batch evaluation command |
| [`core/evaluator.py`](core/evaluator.py) | Evaluation workflow |
| [`data/`](data/) | Dataset loading, phase prefixes and prompt construction |
| [`adapters/`](adapters/) | Mock and OpenAI-compatible model interfaces |
| [`prompts/`](prompts/) | Evaluation and judge prompts |
| [`scoring/metrics.py`](scoring/metrics.py) | Classification, phase and signal metrics |
| [`scoring/reasoning.py`](scoring/reasoning.py) | Evidence-chain judge aggregation |
| [`api/server.py`](api/server.py) | Optional local evaluation API |

To start the optional API from the repository root:

```bash
python -m uvicorn bench.api.server:app --host 127.0.0.1 --port 8010
```

PowerShell launchers are available in [`scripts/`](scripts/). Return to the [project README](../README.md) for the paper, dataset links and citation.

### Prompts

| Files | Purpose |
| --- | --- |
| `dialogue_tasks.txt` | Combined full-dialogue tasks; also used by `reasoning_only` for generation |
| `fraud_detection.txt`, `category_classification.txt`, `signal_recognition.txt` | Single-task evaluation |
| `phase_identification.txt` | T3 phase prediction |
| `evidence_chain_judge.txt` | T5 reasoning judge |
| `dialogue_tasks_think_tags.txt` | Think-tag response format |
