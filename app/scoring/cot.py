from __future__ import annotations

from statistics import median
from typing import Any, Dict, List


def clamp_1_5(value: Any) -> int:
    try:
        n = int(round(float(value)))
    except Exception:
        n = 1
    return max(1, min(5, n))


def normalize_cot_result(raw: Dict[str, Any]) -> Dict[str, Any]:
    e = clamp_1_5(raw.get("evidence_alignment", 1))
    l = clamp_1_5(raw.get("logical_coherence", 1))
    c = clamp_1_5(raw.get("clarity", 1))
    cot15 = e + l + c
    cot5 = round(cot15 / 3.0, 2)
    return {
        "evidence_alignment": e,
        "logical_coherence": l,
        "clarity": c,
        "cot_score_15": cot15,
        "cot_score_5": cot5,
        "t5_reasoning_score_15": cot15,
        "t5_reasoning_score_5": cot5,
        "feedback": str(raw.get("feedback", "")).strip(),
    }


def aggregate_votes(votes: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not votes:
        return {
            "evidence_alignment": 1,
            "logical_coherence": 1,
            "clarity": 1,
            "cot_score_15": 3,
            "cot_score_5": 1.0,
            "t5_reasoning_score_15": 3,
            "t5_reasoning_score_5": 1.0,
            "feedback": "",
        }
    norm = [normalize_cot_result(v) for v in votes]
    e = int(median([v["evidence_alignment"] for v in norm]))
    l = int(median([v["logical_coherence"] for v in norm]))
    c = int(median([v["clarity"] for v in norm]))
    cot15 = e + l + c
    cot5 = round(cot15 / 3.0, 2)
    return {
        "evidence_alignment": e,
        "logical_coherence": l,
        "clarity": c,
        "cot_score_15": cot15,
        "cot_score_5": cot5,
        "t5_reasoning_score_15": cot15,
        "t5_reasoning_score_5": cot5,
        "feedback": " | ".join(v["feedback"] for v in norm if v.get("feedback")),
        "votes": norm,
    }
