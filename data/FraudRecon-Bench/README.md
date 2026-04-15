# FraudRecon-Bench Data Layout

This directory is a public-facing placeholder for the benchmark release.

## Supported layout

The loader accepts either of the following layouts:

```text
data/FraudRecon-Bench/
  test_index.jsonl
  test_labels.jsonl
  samples/
    <case_id>.json
  audio/
    ...
```

or

```text
data/FraudRecon-Bench/
  test/
    test_index.jsonl
    test_labels.jsonl
    samples/
      <case_id>.json
    audio/
      ...
```

## Required files

- `test_index.jsonl`: one JSON object per case with at least `case_id`
- `test_labels.jsonl`: one JSON object per case with `case_id` and `labels`
- `samples/<case_id>.json`: dialogue sample containing `reconstructed_dialogue.flat_turns`

## Required label fields

The evaluation framework expects each `labels` object to include:

- `is_fraud`
- `level_1_category`
- `level_2_phases`
- `level_3_signals`

The canonical phase schema follows the paper:

- Fraud: `initial_contact`, `trust_building`, `inducement`, `operation_guidance`, `fund_transfer`, `sustained_pressure_discovery`
- Benign: `opening`, `core_service`, `closure`

For backward compatibility, the loader also normalizes older aliases such as `discovery`, `sustained_pressure`, and `service_handling`.

## Included demo subset

This repository ships with two tiny mock cases for smoke tests only. They are not part of the paper dataset and should be replaced by the actual release files once available.
