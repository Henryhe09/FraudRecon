from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench.core.evaluator import EvaluationEngine
from bench.modes import CLI_MODES


def main():
    parser = argparse.ArgumentParser(description="FraudRecon-Bench evaluation runner")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--judge-model-id", required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--input-mode", default="text", choices=["text", "audio", "multimodal"])
    parser.add_argument(
        "--eval-mode",
        default=None,
        choices=CLI_MODES,
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--dump-raw", action="store_true", help="Dump raw model responses to outputs/bench/<run_id>/raw_responses")
    parser.add_argument("--strip-tone-hints", action="store_true", help="Strip tone hints inside () / （） from utterances")
    parser.add_argument(
        "--light-json-repair",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable light JSON repair fallback for full/dialogue_only dialogue parsing (default: runtime config / enabled)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    engine = EvaluationEngine(project_root)
    result = engine.run_batch(
        model_id=args.model_id,
        judge_model_id=args.judge_model_id,
        dataset_root=args.dataset_root,
        split=args.split,
        input_mode=args.input_mode,
        eval_mode=args.eval_mode,
        run_id=args.run_id,
        max_workers=args.max_workers,
        dump_raw=args.dump_raw,
        strip_tone_hints=args.strip_tone_hints,
        enable_light_json_repair=args.light_json_repair,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
