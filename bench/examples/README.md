# Benchmark examples

Two fictional text-only samples, one scam and one benign:

```text
examples/
  test_index.jsonl
  test_labels.jsonl
  samples/
    demo_fraud_0001.json
    demo_normal_0001.json
```

The index lists case IDs; the label file supplies category, phase and signal annotations. Sample JSONs contain the dialogue turns. Audio paths are empty.

See the [benchmark guide](../README.md) for the demo command and input format.
