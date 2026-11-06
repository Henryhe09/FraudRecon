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


def _validate_case(case_id: str, sample: Dict[str, Any], labels: Dict[str, Any]) -> None:
    if sample.get("case_id") != case_id:
        raise ValueError(f"Sample case_id does not match index: {case_id}")
    if type(labels.get("is_fraud")) is not bool:
        raise ValueError(f"{case_id}: labels.is_fraud must be a boolean")
    if not isinstance(labels.get("level_1_category"), str):
        raise ValueError(f"{case_id}: missing level_1_category")
    if not isinstance(labels.get("level_3_signals"), list) or any(not isinstance(x,str) for x in labels['level_3_signals']):
        raise ValueError(f"{case_id}: level_3_signals must be a list of signal IDs")
    turns = sample.get("reconstructed_dialogue", {}).get("flat_turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError(f"{case_id}: missing reconstructed_dialogue.flat_turns")
    ids = [t.get("turn_id") for t in turns]
    if any(type(i) is not int or i < 1 for i in ids) or ids != sorted(set(ids)):
        raise ValueError(f"{case_id}: turn_id must be unique positive integers in order")
    if any(not isinstance(t.get("speaker"), str) or not isinstance(t.get("utterance"), str) for t in turns):
        raise ValueError(f"{case_id}: each turn needs speaker and utterance strings")
    phases = labels.get("level_2_phases")
    if not isinstance(phases, list) or not phases:
        raise ValueError(f"{case_id}: missing level_2_phases")
    for phase in phases:
        span = phase.get("turn_range")
        if not isinstance(phase.get("type"), str) or not isinstance(phase.get("phase_id"), (str,int)) or isinstance(phase.get('phase_id'),bool):
            raise ValueError(f"{case_id}: phase needs phase_id and type")
        if (not isinstance(span,list) or len(span)!=2 or any(type(i) is not int or i not in ids for i in span) or span[0]>span[1]):
            raise ValueError(f"{case_id}: invalid phase turn_range")


def load_cases(dataset_root: Path, split: str = "test") -> List[CaseRecord]:
    """Read <split>_index/labels.jsonl and samples/; never guess another split."""
    dataset_root = Path(dataset_root)
    if not split or Path(split).name != split or split in ('.','..'):
        raise ValueError("split must be a plain split name")
    candidates = [dataset_root / split, dataset_root]
    split_dir = next((p for p in candidates if (p/f"{split}_index.jsonl").is_file()), None)
    if split_dir is None:
        raise FileNotFoundError(f"Missing {split}_index.jsonl under {dataset_root} or {dataset_root/split}")
    index_items = _read_jsonl(split_dir/f"{split}_index.jsonl")
    labels_path = split_dir/f"{split}_labels.jsonl"
    if not labels_path.is_file():
        raise FileNotFoundError(f"Missing label file: {labels_path}")
    labels_map = {}
    for row in _read_jsonl(labels_path):
        cid = row['case_id']
        if cid in labels_map: raise ValueError(f"Duplicate label case_id: {cid}")
        if not isinstance(row.get('labels'),dict): raise ValueError(f"{cid}: labels must be an object")
        labels_map[cid] = row['labels']
    cases = []; seen = set()
    for row in index_items:
        cid = row.get('case_id')
        if not isinstance(cid,str) or not cid or Path(cid).name!=cid or cid in ('.','..'):
            raise ValueError("Index case_id must be a nonempty filename-safe string")
        if cid in seen: raise ValueError(f"Duplicate index case_id: {cid}")
        seen.add(cid)
        if cid not in labels_map: raise ValueError(f"Missing labels for case_id: {cid}")
        path = split_dir/'samples'/f"{cid}.json"
        if not path.is_file(): raise FileNotFoundError(f"Missing sample: {path}")
        sample = json.loads(path.read_text(encoding='utf-8'))
        labels = labels_map[cid]  # The external label file is authoritative.
        _validate_case(cid,sample,labels)
        cases.append(CaseRecord(case_id=cid,split=split,sample=sample,labels=labels))
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


def build_phase_prefixes(case: CaseRecord, strip_tone_hints: bool = False) -> List[Dict[str, Any]]:
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
