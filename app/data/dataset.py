from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


FRAUD_PHASE_ALIASES = {
    "initial_contact": "initial_contact",
    "trust_building": "trust_building",
    "inducement": "inducement",
    "operation_guidance": "operation_guidance",
    "fund_transfer": "fund_transfer",
    "sustained_pressure": "sustained_pressure_discovery",
    "discovery": "sustained_pressure_discovery",
    "sustained_pressure_discovery": "sustained_pressure_discovery",
    "sustained_pressure/discovery": "sustained_pressure_discovery",
    "sustained pressure / discovery": "sustained_pressure_discovery",
}

BENIGN_PHASE_ALIASES = {
    "opening": "opening",
    "initial_contact": "opening",
    "core_service": "core_service",
    "service_handling": "core_service",
    "closure": "closure",
}


@dataclass
class CaseRecord:
    case_id: str
    split: str
    sample: Dict[str, Any]
    labels: Dict[str, Any]

    @property
    def turns(self) -> List[Dict[str, Any]]:
        return (
            self.sample.get("reconstructed_dialogue", {})
            .get("flat_turns", [])
        )

    @property
    def is_fraud(self) -> bool:
        labels = self.labels or {}
        return bool(labels.get("is_fraud", False))

    @property
    def audio_relpath(self) -> str:
        return self.sample.get("audio_path", "")


def normalize_phase_type(phase_type: Any, is_fraud: bool) -> str:
    token = str(phase_type or "").strip().lower().replace("-", "_")
    if not token:
        return ""
    aliases = FRAUD_PHASE_ALIASES if is_fraud else BENIGN_PHASE_ALIASES
    return aliases.get(token, token)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _first_existing(base_dir: Path, names: List[str]) -> Optional[Path]:
    for name in names:
        p = base_dir / name
        if p.exists():
            return p
    return None


def _detect_stage_variant(split: str) -> Optional[str]:
    s = str(split or "").strip().lower()
    if not s:
        return None
    # Accept generic stage suffixes like early/mid/late/p50 from split names.
    for sep in ("_", "-"):
        if sep in s:
            tail = s.rsplit(sep, 1)[-1].strip()
            if tail:
                return tail
    if s.isalnum():
        return s
    return None


def _detect_stage_variants(split: str) -> List[str]:
    s = str(split or "").strip().lower()
    if not s:
        return []

    tokens = [t for t in re.split(r"[_-]+", s) if t]
    variants: List[str] = []

    # Prefer longer suffixes first, e.g. p30_noshuadan before noshuadan.
    for n in range(len(tokens), 0, -1):
        variants.append("_".join(tokens[-n:]))

    tail = _detect_stage_variant(split)
    if tail:
        variants.append(tail)

    seen = set()
    deduped: List[str] = []
    for v in variants:
        if v and v not in seen:
            deduped.append(v)
            seen.add(v)
    return deduped


def load_cases(dataset_root: Path, split: str = "test") -> List[CaseRecord]:
    variants = _detect_stage_variants(split)
    dir_candidates = [dataset_root / split, dataset_root]
    index_names: List[str] = [f"{split}_index.jsonl"]
    for variant in variants:
        index_names.append(f"test_index_{variant}.jsonl")
    index_names.extend(["test_index.jsonl", "train_index.jsonl"])

    split_dir: Optional[Path] = None
    index_path: Optional[Path] = None
    for d in dir_candidates:
        index_path = _first_existing(d, index_names)
        if index_path:
            split_dir = d
            break
    if not index_path:
        raise FileNotFoundError(f"Index file not found under: {dataset_root} (split={split})")
    assert split_dir is not None
    index_items = _read_jsonl(index_path)

    labels_map: Dict[str, Dict[str, Any]] = {}
    labels_names: List[str] = [f"{split}_labels.jsonl"]
    for variant in variants:
        labels_names.append(f"test_labels_{variant}.jsonl")
    labels_names.append("test_labels.jsonl")
    labels_path = _first_existing(
        split_dir,
        labels_names,
    )
    if labels_path:
        for row in _read_jsonl(labels_path):
            labels_map[row["case_id"]] = row["labels"]

    sample_dir_candidates: List[Path] = []
    for variant in variants:
        sample_dir_candidates.append(split_dir / f"samples_{variant}")
    sample_dir_candidates.extend([split_dir / "samples", split_dir / f"samples_{split}"])
    sample_dir = next((d for d in sample_dir_candidates if d.exists()), None)
    if not sample_dir:
        raise FileNotFoundError(f"Sample dir not found under: {split_dir}")

    cases: List[CaseRecord] = []
    for row in index_items:
        case_id = row["case_id"]
        sample_names: List[str] = [f"{case_id}.json"]
        for variant in variants:
            sample_names.append(f"{case_id}_{variant}.json")
        # Keep order while deduping.
        sample_names = list(dict.fromkeys(sample_names))

        sample_path = _first_existing(sample_dir, sample_names)
        if not sample_path:
            raise FileNotFoundError(
                f"Sample file not found for case_id={case_id} under {sample_dir}; "
                f"tried={sample_names}"
            )
        with sample_path.open("r", encoding="utf-8") as f:
            sample = json.load(f)

        labels = labels_map.get(case_id, sample.get("labels", {}))
        cases.append(CaseRecord(case_id=case_id, split=split, sample=sample, labels=labels))

    return cases


def turns_to_text(turns: List[Dict[str, Any]]) -> str:
    return turns_to_text_with_options(turns, strip_tone_hints=False)


def _strip_tone_hints(text: str) -> str:
    s = str(text or "")
    # Remove short parenthetical hints like "(语气急促)" or "（停顿）".
    s = re.sub(r"\s*[（(][^（）()]{1,40}[）)]\s*", "", s)
    return s.strip()


def turns_to_text_with_options(turns: List[Dict[str, Any]], strip_tone_hints: bool = False) -> str:
    lines: List[str] = []
    for t in turns:
        turn_id = t.get("turn_id", "")
        speaker = t.get("speaker", "")
        utt_raw = str(t.get("utterance", "")).strip()
        utt = _strip_tone_hints(utt_raw) if strip_tone_hints else utt_raw
        lines.append(f"[Turn {turn_id}][{speaker}] {utt}")
    return "\n".join(lines)


def build_t2_prefixes(case: CaseRecord, strip_tone_hints: bool = False) -> List[Dict[str, Any]]:
    phases = case.labels.get("level_2_phases", [])
    turns = case.turns
    prefixes: List[Dict[str, Any]] = []
    for idx, p in enumerate(phases, start=1):
        turn_range = p.get("turn_range", [1, 1])
        end_turn = int(turn_range[1])
        prefix_turns = [t for t in turns if int(t.get("turn_id", 0)) <= end_turn]
        prefixes.append(
            {
                "phase_idx": idx,
                "phase_id": p.get("phase_id"),
                "gold_phase_type": normalize_phase_type(p.get("type"), is_fraud=case.is_fraud),
                "dialogue_prefix": turns_to_text_with_options(prefix_turns, strip_tone_hints=strip_tone_hints),
                "end_turn": end_turn,
            }
        )
    return prefixes
