from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from bench.adapters.base import BaseAdapter


class MockAdapter(BaseAdapter):
    def __init__(self, mode: str = "main"):
        self.mode = mode

    def chat(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        modalities: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        _ = (model_name, messages, temperature, max_tokens, modalities)
        ctx = context or {}
        task = ctx.get("task")
        gold = ctx.get("gold_labels", {})

        if self.mode == "judge" or task == "cot_judge":
            return json.dumps(
                {
                    "evidence_alignment": 5,
                    "logical_coherence": 5,
                    "clarity": 5,
                    "cot_score_15": 15,
                    "cot_score_5": 5.0,
                    "feedback": "mock judge output",
                },
                ensure_ascii=False,
            )

        if task == "round1":
            is_fraud = bool(gold.get("is_fraud", False))
            level_1_category = gold.get("level_1_category", "unknown")
            level_3_signals = gold.get("level_3_signals", [])
            signal_type = "fraud_tactics" if is_fraud else "service_behaviors"
            return json.dumps(
                {
                    "think": "This is a mock reasoning trace used for local smoke tests.",
                    "output": {
                        "is_fraud": is_fraud,
                        "level_1_category": level_1_category,
                        "signal_type": signal_type,
                        "level_3_signals": level_3_signals,
                        "confidence": 0.99,
                    },
                },
                ensure_ascii=False,
            )

        if task == "round2":
            return json.dumps(
                {"phase_type": ctx.get("gold_phase_type", "initial_contact")},
                ensure_ascii=False,
            )

        return json.dumps({"message": "mock"}, ensure_ascii=False)
