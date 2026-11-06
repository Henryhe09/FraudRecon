#!/usr/bin/env python3
import argparse
import os
import json
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency fallback
    tqdm = None

from .evidence_chain_prompts import SYSTEM_PROMPT, build_repair_prompt, build_user_prompt

# 连接信息仅从环境变量或显式命令行参数读取。
ENV_API_KEY = os.getenv("OPENAI_API_KEY", "")
ENV_BASE_URL = os.getenv("OPENAI_BASE_URL", "")


FRAUD_PHASE_MAP = {
    "initial_contact": "P1",
    "trust_building": "P2",
    "inducement": "P3",
    "operation_guidance": "P4",
    "fund_transfer": "P5",
    "sustained_pressure": "P6",
    "discovery": "P6",
}

SIM_PHASE_MAP = {
    "initial_contact": "Ph1",
    "service_handling": "Ph2",
    "closure": "Ph3",
}

# Allowed taxonomy (hard whitelist)
FRAUD_ALLOWED_PHASE_TYPES = {
    "initial_contact",
    "trust_building",
    "inducement",
    "operation_guidance",
    "fund_transfer",
    "sustained_pressure",
    "discovery",
}

NORMAL_ALLOWED_PHASE_TYPES = {
    "initial_contact",
    "service_handling",
    "closure",
}

FRAUD_ALLOWED_SIGNAL_IDS = {
    "impersonate_ecommerce_cs",
    "impersonate_bank_finance",
    "impersonate_gov_law",
    "impersonate_leader",
    "impersonate_acquaintance",
    "impersonate_military",
    "impersonate_expert",
    "urgency_creation",
    "credit_threat",
    "account_freeze_threat",
    "legal_threat",
    "profit_lure",
    "free_gift_lure",
    "small_return_trust",
    "sunk_cost_pressure",
    "group_pressure",
    "emotional_manipulation",
    "app_download_lure",
    "link_click_lure",
    "screen_share_lure",
    "group_chat_lure",
    "deposit_excuse",
    "unfreeze_fee_excuse",
    "operation_error_excuse",
    "upgrade_fee_excuse",
    "tax_fee_excuse",
}

NORMAL_ALLOWED_SIGNAL_IDS = {
    "identity_verification",
    "privacy_reassurance",
    "order_confirmation",
    "user_intent_confirmation",
    "rule_explanation",
    "policy_disclosure",
    "status_inquiry",
    "process_guidance",
    "solution_proposal",
    "escalation_handover",
    "complaint_handling",
    "empathy_expression",
    "expectation_alignment",
    "proactive_closure",
    "user_initiated_closure",
    "polite_farewell",
}

# Compatibility aliases from historical data.
SIGNAL_ID_ALIASES = {
    "impersonate_finance_platform": "impersonate_bank_finance",
}


def format_exception_message(exc: Exception) -> str:
    parts = [f"{exc.__class__.__name__}: {exc}"]
    cur = exc.__cause__ or exc.__context__
    depth = 0
    while cur is not None and depth < 3:
        parts.append(f"caused by {cur.__class__.__name__}: {cur}")
        cur = cur.__cause__ or cur.__context__
        depth += 1
    return " | ".join(parts)


def normalize_signal_id(raw: Any) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    # Remove full-width/half-width parenthetical notes, e.g. "profit_lure（...）"
    text = re.sub(r"[（(].*?[）)]", "", text).strip()
    return SIGNAL_ID_ALIASES.get(text, text)


def normalize_phase_type(raw: Any) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    # Handle noisy merged labels like "trust_building/inducement"
    if "/" in text:
        text = text.split("/", 1)[0].strip()
    return text


def sanitize_label_hints(
    hints: Dict[str, Any], expected_is_fraud: bool
) -> Dict[str, Any]:
    """Drop invalid label-hint items so noisy input labels do not create impossible constraints."""
    allowed_phase_types = (
        FRAUD_ALLOWED_PHASE_TYPES if expected_is_fraud else NORMAL_ALLOWED_PHASE_TYPES
    )
    allowed_signal_ids = (
        FRAUD_ALLOWED_SIGNAL_IDS if expected_is_fraud else NORMAL_ALLOWED_SIGNAL_IDS
    )
    out = dict(hints)

    phases = out.get("phases")
    if isinstance(phases, list):
        cleaned_phases: List[Dict[str, Any]] = []
        for item in phases:
            if not isinstance(item, dict):
                continue
            t = item.get("type")
            if not isinstance(t, str):
                continue
            nt = normalize_phase_type(t)
            if nt not in allowed_phase_types:
                continue
            copied = dict(item)
            copied["type"] = nt
            cleaned_phases.append(copied)
        if cleaned_phases:
            out["phases"] = cleaned_phases
        else:
            out.pop("phases", None)

    signals = out.get("signals")
    if isinstance(signals, list):
        cleaned_signals: List[str] = []
        seen = set()
        for item in signals:
            sid = normalize_signal_id(item)
            if sid and sid in allowed_signal_ids and sid not in seen:
                seen.add(sid)
                cleaned_signals.append(sid)
        if cleaned_signals:
            out["signals"] = cleaned_signals
        else:
            out.pop("signals", None)

    pairs = out.get("signals_with_phase")
    if isinstance(pairs, list):
        cleaned_pairs: List[Dict[str, str]] = []
        seen = set()
        for item in pairs:
            if not isinstance(item, dict):
                continue
            sid_raw = item.get("signal_id")
            ph = item.get("phase")
            if not isinstance(ph, (str, int)):
                continue
            sid = normalize_signal_id(sid_raw)
            if not sid or sid not in allowed_signal_ids:
                continue
            copied = dict(item)
            copied["signal_id"] = sid
            if isinstance(copied.get("phase_type"), str):
                pt = normalize_phase_type(copied["phase_type"])
                if pt not in allowed_phase_types:
                    continue
                copied["phase_type"] = pt
            copied["phase"] = str(ph)
            k = (copied["signal_id"], copied["phase"])
            if k in seen:
                continue
            seen.add(k)
            cleaned_pairs.append(copied)
        if cleaned_pairs:
            out["signals_with_phase"] = cleaned_pairs
        else:
            out.pop("signals_with_phase", None)

    return out

class ThreadLocalLLMClient:
    def __init__(self, api_key: str, base_url: str, timeout_seconds: int):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self._local = threading.local()

    def _get_client(self) -> OpenAI:
        client = getattr(self._local, "client", None)
        if client is None:
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout_seconds,
            )
            self._local.client = client
        return client

    def chat(self, model: str, messages: List[Dict[str, str]], temperature: float) -> str:
        rsp = self._get_client().chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=False,
        )
        content = rsp.choices[0].message.content
        return (content or "").strip()


class FraudSourceChunkResolver:
    """从 source_chunk 文件按 index 回查策略标签（tactics）。"""

    def __init__(self, source_chunk_dir: Optional[Path]):
        self.source_chunk_dir = source_chunk_dir
        self._chunk_cache: Dict[str, Dict[int, Dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def _resolve_chunk_path(self, sample: Dict[str, Any]) -> Optional[Path]:
        source_chunk = sample.get("source_chunk")
        if not isinstance(source_chunk, str) or not source_chunk.strip():
            return None
        source_chunk = source_chunk.strip()
        raw_path = Path(source_chunk)
        if raw_path.is_absolute() and raw_path.exists():
            return raw_path

        candidates: List[Path] = []
        if self.source_chunk_dir is not None:
            candidates.append(self.source_chunk_dir / source_chunk)
        candidates.append(Path.cwd() / source_chunk)

        for path in candidates:
            if path.exists():
                return path
        return None

    def _load_chunk_index_map(self, chunk_path: Path) -> Dict[int, Dict[str, Any]]:
        with chunk_path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and isinstance(obj.get("results"), list):
            rows = obj["results"]
        elif isinstance(obj, list):
            rows = obj
        else:
            rows = []

        index_map: Dict[int, Dict[str, Any]] = {}
        for item in rows:
            if not isinstance(item, dict):
                continue
            idx = item.get("index")
            if isinstance(idx, int):
                index_map[idx] = item
        return index_map

    def _get_chunk_map(self, chunk_path: Path) -> Dict[int, Dict[str, Any]]:
        key = str(chunk_path.resolve())
        with self._lock:
            cached = self._chunk_cache.get(key)
        if cached is not None:
            return cached

        loaded = self._load_chunk_index_map(chunk_path)
        with self._lock:
            self._chunk_cache[key] = loaded
        return loaded

    def get_tactics_with_phase(self, sample: Dict[str, Any]) -> List[Dict[str, str]]:
        chunk_path = self._resolve_chunk_path(sample)
        idx = sample.get("index")
        if chunk_path is None or not isinstance(idx, int):
            return []

        chunk_map = self._get_chunk_map(chunk_path)
        row = chunk_map.get(idx)
        if not isinstance(row, dict):
            return []
        result = row.get("result")
        if not isinstance(result, dict):
            return []
        fraud_phases = result.get("fraud_phases")
        if not isinstance(fraud_phases, list):
            return []

        signal_pairs: List[Dict[str, str]] = []
        for i, phase in enumerate(fraud_phases, start=1):
            if not isinstance(phase, dict):
                continue
            phase_type = normalize_phase_type(phase.get("phase_type"))
            phase_raw = phase.get("phase_id")
            phase_id = str(phase_raw) if isinstance(phase_raw, (str, int)) else str(i)
            tags = phase.get("tactics")
            if not isinstance(tags, list):
                continue
            for tag in tags:
                if isinstance(tag, (str, int, float)):
                    t = normalize_signal_id(tag)
                    if t:
                        signal_pairs.append(
                            {"signal_id": t, "phase": phase_id, "phase_type": phase_type}
                        )

        seen = set()
        dedup_pairs: List[Dict[str, str]] = []
        for item in signal_pairs:
            key = (item["signal_id"], item["phase"])
            if key not in seen:
                seen.add(key)
                dedup_pairs.append(item)
        return dedup_pairs

    def get_tactics(self, sample: Dict[str, Any]) -> List[str]:
        pairs = self.get_tactics_with_phase(sample)
        seen = set()
        dedup: List[str] = []
        for item in pairs:
            sig = item["signal_id"]
            if sig not in seen:
                seen.add(sig)
                dedup.append(sig)
        return dedup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 FraudRecon 的 CoT 标注 evidence_chain")
    parser.add_argument(
        "--input",
        required=True,
        help="输入路径（.json/.jsonl，或包含 *.json 的目录）",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="输出 JSONL 文件路径（断点续跑）",
    )
    parser.add_argument(
        "--output-json-dir",
        default="",
        help="可选：逐条 JSON 输出目录（会与 JSONL 同时写入）",
    )
    parser.add_argument("--model", default=os.getenv("OPENAI_ANNOTATION_MODEL", ""), help="模型名")
    parser.add_argument(
        "--base-url",
        default=ENV_BASE_URL,
        help="OpenAI 兼容 API Base URL",
    )
    parser.add_argument(
        "--api-key",
        default=ENV_API_KEY,
        help="API key（默认读取 OPENAI_API_KEY）",
    )
    parser.add_argument(
        "--label-mode",
        choices=["blind", "constrained"],
        default="constrained",
        help="blind: 仅输入对话；constrained: 加入已有标签约束",
    )
    parser.add_argument("--max-workers", type=int, default=50, help="并发请求数")
    parser.add_argument("--max-retries", type=int, default=3, help="API 重试次数")
    parser.add_argument("--repair-attempts", type=int, default=1, help="格式修复次数")
    parser.add_argument("--temperature", type=float, default=0.0, help="采样温度（稳定格式建议 0.0）")
    parser.add_argument("--timeout", type=int, default=600, help="单请求超时（秒）")
    parser.add_argument("--start-index", type=int, default=0, help="起始 index（含）")
    parser.add_argument("--end-index", type=int, default=None, help="结束 index（含）")
    parser.add_argument("--limit", type=int, default=None, help="最多处理多少条")
    parser.add_argument(
        "--use-existing-dialogue-text",
        action="store_true",
        help="优先使用样本内已有 dialogue_text（否则由 reconstructed_dialogue 重新拼接）",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="断点续跑时，是否重跑历史 error 记录",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="测试模式：在输出中附带 prompt 全文（system/user/repair）",
    )
    parser.add_argument(
        "--source-chunk-dir",
        default="",
        help="可选：显式指定历史 source_chunk 目录；默认不读取",
    )
    parser.add_argument(
        "--labels-jsonl",
        default="",
        help="可选：外部标签 JSONL（按 case_id 对齐到输入样本）",
    )
    parser.add_argument(
        "--auto-validate",
        action="store_true",
        help="生成后自动对照（label↔cot）并对不通过样本自动重跑",
    )
    parser.add_argument(
        "--auto-max-rounds",
        type=int,
        default=1,
        help="自动对照+重跑最大轮次（仅 --auto-validate 时生效）",
    )
    parser.add_argument(
        "--auto-signal-mode",
        choices=["set", "dedupe_order", "raw"],
        default="set",
        help="自动对照时 signal 比较口径：set（推荐）/dedupe_order/raw",
    )
    return parser.parse_args()


def load_records(input_path: Path) -> List[Dict[str, Any]]:
    if input_path.is_dir():
        records: List[Dict[str, Any]] = []
        json_files = sorted(input_path.glob("*.json"))
        for p in json_files:
            with p.open("r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict) and isinstance(obj.get("results"), list):
                records.extend(_assert_records(obj["results"]))
            elif isinstance(obj, dict):
                records.append(obj)
            elif isinstance(obj, list):
                records.extend(_assert_records(obj))
            else:
                raise ValueError(f"目录内文件格式不支持: {p}")
        return records

    if input_path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with input_path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"JSONL 解析失败: line={lineno}, error={exc}") from exc
                if not isinstance(item, dict):
                    raise ValueError(f"JSONL 每行必须是对象: line={lineno}")
                rows.append(item)
        return rows

    with input_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return _assert_records(obj)
    if isinstance(obj, dict) and isinstance(obj.get("results"), list):
        return _assert_records(obj["results"])
    if isinstance(obj, dict):
        return [obj]
    raise ValueError("输入 JSON 需为数组，或包含 results 数组的对象")


def load_labels_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
    label_map: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"labels JSONL 解析失败: line={lineno}, error={exc}") from exc
            if not isinstance(row, dict):
                continue
            case_id = row.get("case_id")
            labels = row.get("labels")
            if isinstance(case_id, str) and case_id.strip() and isinstance(labels, dict):
                label_map[case_id.strip()] = labels
    return label_map


def attach_labels_by_case_id(
    records: List[Dict[str, Any]], label_map: Dict[str, Dict[str, Any]]
) -> Tuple[int, int]:
    matched = 0
    missing = 0
    for row in records:
        if not isinstance(row, dict):
            continue
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            missing += 1
            continue
        labels = label_map.get(case_id.strip())
        if isinstance(labels, dict):
            row["labels"] = labels
            matched += 1
        else:
            missing += 1
    return matched, missing


def _assert_records(rows: List[Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for i, item in enumerate(rows):
        if not isinstance(item, dict):
            raise ValueError(f"样本必须为对象: index={i}")
        records.append(item)
    return records


def load_done_status(output_path: Path) -> Dict[int, str]:
    done: Dict[int, str] = {}
    if not output_path.exists():
        return done
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            idx = row.get("index")
            status = row.get("status")
            if isinstance(idx, int) and isinstance(status, str):
                done[idx] = status
    return done


def append_jsonl(output_path: Path, row: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _safe_filename(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_.")
    return cleaned or "sample"


def write_single_json(output_dir: Path, row: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    case_id = str(row.get("case_id") or "unknown")
    filename = f"{_safe_filename(case_id)}.json"
    path = output_dir / filename
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def detect_data_type(sample: Dict[str, Any]) -> str:
    data_type = sample.get("data_type")
    if isinstance(data_type, str):
        dt = data_type.strip().lower()
        if dt == "fraud":
            return "fraud"
        if dt in {"normal", "sim"}:
            return "normal"
    is_fraud = sample.get("is_fraud")
    if is_fraud is True:
        return "fraud"
    if is_fraud is False:
        return "normal"
    if "案件类别" in sample or "fraud_meta" in sample:
        return "fraud"
    if "call_category" in sample or "normal_meta" in sample or "simulated_id" in sample:
        return "normal"
    return "fraud"


def infer_is_fraud(sample: Dict[str, Any], data_type: str) -> bool:
    is_fraud = sample.get("is_fraud")
    if isinstance(is_fraud, bool):
        return is_fraud
    return data_type == "fraud"


def infer_case_id(sample: Dict[str, Any], index: int, data_type: str) -> str:
    for key in ("case_id", "simulated_id"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    case_no = sample.get("案件编号")
    if isinstance(case_no, int):
        prefix = "fraud" if data_type == "fraud" else "normal"
        return f"{prefix}_{case_no:06d}"
    return f"{data_type}_{index:07d}"


def map_speaker(raw_speaker: Any, data_type: str) -> str:
    s = str(raw_speaker or "").strip().upper()
    if data_type == "fraud":
        if s in {"S", "LEFT", "L"}:
            return "诈骗者"
        if s in {"V", "RIGHT", "R"}:
            return "受害者"
    else:
        if s in {"S", "LEFT", "L"}:
            return "客服"
        if s in {"V", "RIGHT", "R"}:
            return "用户"
    return str(raw_speaker or "未知角色")


def build_dialogue_text(sample: Dict[str, Any], data_type: str) -> str:
    reconstructed = sample.get("reconstructed_dialogue")
    if not isinstance(reconstructed, dict):
        raise ValueError("缺少 reconstructed_dialogue")

    phase_dialogues = reconstructed.get("phase_dialogues")
    phase_blocks: List[str] = []

    if isinstance(phase_dialogues, list) and phase_dialogues:
        turn_cursor = 1
        for phase in phase_dialogues:
            if not isinstance(phase, dict):
                continue
            phase_type = str(phase.get("phase_type") or "unknown")
            turns = phase.get("dialogue_turns")
            if not isinstance(turns, list):
                continue
            start = turn_cursor
            lines: List[str] = []
            for turn in turns:
                if not isinstance(turn, dict):
                    continue
                utterance = str(turn.get("utterance") or "").strip()
                if not utterance:
                    continue
                speaker = map_speaker(turn.get("speaker"), data_type)
                lines.append(f"{speaker}：{utterance}")
                turn_cursor += 1
            if not lines:
                continue
            end = turn_cursor - 1
            header = f"--- Phase {phase_type} (turns {start}-{end}) ---"
            phase_blocks.append(header + "\n" + "\n".join(lines))
    else:
        flat_turns = reconstructed.get("flat_turns")
        if not isinstance(flat_turns, list) or not flat_turns:
            raise ValueError("reconstructed_dialogue.phase_dialogues/flat_turns 均为空")

        sorted_turns = sorted(
            [t for t in flat_turns if isinstance(t, dict)],
            key=lambda x: (x.get("turn_id") if isinstance(x.get("turn_id"), int) else 10**9),
        )
        turn_cursor = 1
        current_key: Optional[Tuple[str, str]] = None
        block_start = 1
        block_lines: List[str] = []

        def flush_block(end_turn: int) -> None:
            if not block_lines or current_key is None:
                return
            phase_type = current_key[1]
            header = f"--- Phase {phase_type} (turns {block_start}-{end_turn}) ---"
            phase_blocks.append(header + "\n" + "\n".join(block_lines))

        for turn in sorted_turns:
            utterance = str(turn.get("utterance") or "").strip()
            if not utterance:
                continue
            phase_id = str(turn.get("phase_id") or "")
            phase_type = str(turn.get("phase_type") or "unknown")
            key = (phase_id, phase_type)
            if current_key is None:
                current_key = key
                block_start = turn_cursor
            elif key != current_key:
                flush_block(turn_cursor - 1)
                block_lines = []
                current_key = key
                block_start = turn_cursor

            speaker = map_speaker(turn.get("speaker"), data_type)
            block_lines.append(f"{speaker}：{utterance}")
            turn_cursor += 1
        flush_block(turn_cursor - 1)

    if not phase_blocks:
        raise ValueError("无法从 reconstructed_dialogue 生成 dialogue_text")
    return "\n\n".join(phase_blocks)


def infer_phase_code(phase_type: str, phase_num: int, data_type: str) -> str:
    phase_type = phase_type.strip()
    if data_type == "fraud":
        return FRAUD_PHASE_MAP.get(phase_type, f"P{phase_num}")
    return SIM_PHASE_MAP.get(phase_type, f"Ph{phase_num}")


def infer_phase_hints(sample: Dict[str, Any], data_type: str) -> List[Dict[str, Any]]:
    labels = sample.get("labels")
    if isinstance(labels, dict):
        keys = ["level_2_phases"] if data_type == "fraud" else ["service_phases", "level_2_phases"]
        phases = None
        for key in keys:
            value = labels.get(key)
            if isinstance(value, list):
                phases = value
                break
        if isinstance(phases, list):
            normalized: List[Dict[str, Any]] = []
            for p in phases:
                if not isinstance(p, dict):
                    continue
                item = dict(p)
                if "id" not in item and isinstance(item.get("phase_id"), (str, int)):
                    item["id"] = str(item["phase_id"])
                if "id" in item and isinstance(item["id"], int):
                    item["id"] = str(item["id"])
                if isinstance(item.get("type"), str):
                    item["type"] = normalize_phase_type(item["type"]) or item["type"].strip()
                normalized.append(item)
            return normalized

    reconstructed = sample.get("reconstructed_dialogue")
    if not isinstance(reconstructed, dict):
        return []
    phase_dialogues = reconstructed.get("phase_dialogues")
    if not isinstance(phase_dialogues, list):
        return []

    hints: List[Dict[str, Any]] = []
    turn_cursor = 1
    for i, phase in enumerate(phase_dialogues, start=1):
        if not isinstance(phase, dict):
            continue
        turns = phase.get("dialogue_turns")
        if not isinstance(turns, list):
            continue
        valid_turn_count = sum(1 for t in turns if isinstance(t, dict) and t.get("utterance"))
        if valid_turn_count <= 0:
            continue
        start = turn_cursor
        end = turn_cursor + valid_turn_count - 1
        turn_cursor = end + 1
        phase_type = normalize_phase_type(phase.get("phase_type")) or "unknown"
        phase_raw = phase.get("phase_id")
        phase_id = str(phase_raw) if isinstance(phase_raw, (str, int)) else str(i)
        hints.append(
            {
                "id": phase_id,
                "type": phase_type,
                "turn_range": [start, end],
            }
        )
    return hints


def extract_signal_hints(
    sample: Dict[str, Any],
    data_type: str,
    source_chunk_resolver: Optional[FraudSourceChunkResolver],
) -> List[str]:
    labels = sample.get("labels")
    if isinstance(labels, dict):
        keys = ["level_3_tactics", "level_3_signals"] if data_type == "fraud" else [
            "service_behaviors",
            "level_3_signals",
        ]
        for key in keys:
            value = labels.get(key)
            if isinstance(value, list):
                out: List[str] = []
                seen = set()
                for x in value:
                    if not isinstance(x, (str, int, float)):
                        continue
                    sid = normalize_signal_id(x)
                    if sid and sid not in seen:
                        seen.add(sid)
                        out.append(sid)
                return out

    if data_type == "fraud":
        if source_chunk_resolver is not None:
            from_source = source_chunk_resolver.get_tactics(sample)
            if from_source:
                return from_source
        proc = sample.get("result")
        if isinstance(proc, dict):
            fraud_phases = proc.get("fraud_phases")
            if isinstance(fraud_phases, list):
                signals: List[str] = []
                for p in fraud_phases:
                    if not isinstance(p, dict):
                        continue
                    tactics = p.get("tactics")
                    if isinstance(tactics, list):
                        for t in tactics:
                            if isinstance(t, (str, int, float)):
                                sid = normalize_signal_id(t)
                                if sid:
                                    signals.append(sid)
                if signals:
                    return sorted(set(signals))
    return []


def extract_signal_hints_with_phase(
    sample: Dict[str, Any],
    data_type: str,
    source_chunk_resolver: Optional[FraudSourceChunkResolver],
) -> List[Dict[str, str]]:
    if data_type != "fraud" or source_chunk_resolver is None:
        return []
    return source_chunk_resolver.get_tactics_with_phase(sample)


def extract_category_hint(sample: Dict[str, Any], data_type: str) -> Optional[str]:
    labels = sample.get("labels")
    if isinstance(labels, dict):
        keys = ["level_1_fraud_type", "level_1_category"] if data_type == "fraud" else [
            "call_category",
            "level_1_category",
        ]
        for key in keys:
            value = labels.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    if data_type == "fraud":
        for key in ("案件类别",):
            value = sample.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        fraud_meta = sample.get("fraud_meta")
        if isinstance(fraud_meta, dict):
            value = fraud_meta.get("fraud_category")
            if isinstance(value, str) and value.strip():
                return value.strip()
    else:
        for key in ("call_category",):
            value = sample.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        normal_meta = sample.get("normal_meta")
        if isinstance(normal_meta, dict):
            value = normal_meta.get("call_category")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def collect_label_hints(
    sample: Dict[str, Any],
    data_type: str,
    is_fraud: bool,
    source_chunk_resolver: Optional[FraudSourceChunkResolver],
) -> Dict[str, Any]:
    hints: Dict[str, Any] = {"is_fraud": is_fraud}
    category = extract_category_hint(sample, data_type)
    if category:
        key = "fraud_type" if is_fraud else "call_category"
        hints[key] = category

    phase_system = "fraud_6phase" if is_fraud else "normal_3phase"
    phase_hints = infer_phase_hints(sample, data_type)
    if phase_hints:
        hints["phase_system"] = phase_system
        hints["phases"] = phase_hints

    signal_hints = extract_signal_hints(sample, data_type, source_chunk_resolver)
    if signal_hints:
        hints["signal_type"] = "fraud_tactics" if is_fraud else "service_behaviors"
        hints["signals"] = signal_hints
        signal_phase_hints = extract_signal_hints_with_phase(
            sample, data_type, source_chunk_resolver
        )
        if signal_phase_hints:
            hints["signals_with_phase"] = signal_phase_hints
    return hints


def parse_json_object(text: str) -> Dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.startswith("json"):
            raw = raw[4:].strip()
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        left = raw.find("{")
        right = raw.rfind("}")
        if left < 0 or right <= left:
            raise
        obj = json.loads(raw[left : right + 1])
    if not isinstance(obj, dict):
        raise ValueError("输出不是 JSON 对象")
    return obj


def split_cn_sentences(text: str) -> List[str]:
    parts = re.split(r"[。！？!?；;]+", text)
    return [p.strip() for p in parts if p and p.strip()]


EVIDENCE_ANCHOR_RE = re.compile(
    r"(turns?\s*\d+\s*-\s*\d+|turn_range|第\s*\d+\s*轮|“[^”]{2,30}”|\"[^\"]{2,30}\")",
    re.IGNORECASE,
)


def validate_think_quality(step_key: str, think_text: str, errors: List[str], warnings: List[str]) -> None:
    text = (think_text or "").strip()
    if not text:
        return
    sent_count = len(split_cn_sentences(text))
    if sent_count < 3:
        errors.append(f"{step_key}.think 句子数不足（当前 {sent_count}，至少 3 句）")
    if len(text) < 60:
        errors.append(f"{step_key}.think 内容过短（当前 {len(text)} 字，至少 60 字）")
    if step_key in {
        "step_2_phase_identification",
        "step_3_signal_identification",
        "step_4_judgment",
    }:
        if not EVIDENCE_ANCHOR_RE.search(text):
            warnings.append(f"{step_key}.think 未检测到明显对话证据锚点（turn/原话）")


def _require_step_obj(
    payload: Dict[str, Any], step_key: str, errors: List[str]
) -> Optional[Dict[str, Any]]:
    step = payload.get(step_key)
    if not isinstance(step, dict):
        errors.append(f"{step_key} 缺失或非对象")
        return None
    think = step.get("think")
    output = step.get("output")
    if not isinstance(think, str) or not think.strip():
        errors.append(f"{step_key}.think 缺失或非字符串")
    if not isinstance(output, dict):
        errors.append(f"{step_key}.output 缺失或非对象")
        return None
    return output


def validate_evidence_chain(
    payload: Dict[str, Any],
    expected_is_fraud: bool,
    label_mode: str,
    label_hints: Optional[Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    for key in (
        "step_1_entity_extraction",
        "step_2_phase_identification",
        "step_3_signal_identification",
        "step_4_judgment",
    ):
        _require_step_obj(payload, key, errors)

    out2 = _require_step_obj(payload, "step_2_phase_identification", errors)
    out3 = _require_step_obj(payload, "step_3_signal_identification", errors)
    out4 = _require_step_obj(payload, "step_4_judgment", errors)
    out1 = _require_step_obj(payload, "step_1_entity_extraction", errors)

    expected_phase_system = "fraud_6phase" if expected_is_fraud else "normal_3phase"
    expected_signal_type = "fraud_tactics" if expected_is_fraud else "service_behaviors"
    allowed_phase_types = (
        FRAUD_ALLOWED_PHASE_TYPES if expected_is_fraud else NORMAL_ALLOWED_PHASE_TYPES
    )
    allowed_signal_ids = (
        FRAUD_ALLOWED_SIGNAL_IDS if expected_is_fraud else NORMAL_ALLOWED_SIGNAL_IDS
    )

    if out2 is not None:
        phase_system = out2.get("phase_system")
        if not isinstance(phase_system, str) or not phase_system.strip():
            errors.append("step_2.output.phase_system 缺失或非字符串")
        elif phase_system != expected_phase_system:
            message = f"phase_system={phase_system} 与期望 {expected_phase_system} 不一致"
            (errors if label_mode == "constrained" else warnings).append(message)
        phases = out2.get("phases")
        if not isinstance(phases, list) or not phases:
            errors.append("step_2.output.phases 缺失或为空列表")
        else:
            for i, phase in enumerate(phases, start=1):
                if not isinstance(phase, dict):
                    errors.append(f"step_2.output.phases[{i}] 不是对象")
                    continue
                if not isinstance(phase.get("id"), (str, int)):
                    errors.append(f"step_2.output.phases[{i}].id 缺失或非字符串/整数")
                phase_type_raw = phase.get("type")
                if not isinstance(phase_type_raw, str):
                    errors.append(f"step_2.output.phases[{i}].type 缺失或非字符串")
                else:
                    phase_type = phase_type_raw.strip()
                    if phase_type not in allowed_phase_types:
                        errors.append(
                            f"step_2.output.phases[{i}].type 不在白名单: {phase_type}"
                        )
                tr = phase.get("turn_range")
                if (
                    not isinstance(tr, list)
                    or len(tr) != 2
                    or not isinstance(tr[0], int)
                    or not isinstance(tr[1], int)
                ):
                    errors.append(f"step_2.output.phases[{i}].turn_range 必须是两个整数")
                if not isinstance(phase.get("description"), str):
                    errors.append(f"step_2.output.phases[{i}].description 缺失或非字符串")

    if out3 is not None:
        signal_type = out3.get("signal_type")
        if not isinstance(signal_type, str) or not signal_type.strip():
            errors.append("step_3.output.signal_type 缺失或非字符串")
        elif signal_type != expected_signal_type:
            message = f"signal_type={signal_type} 与期望 {expected_signal_type} 不一致"
            (errors if label_mode == "constrained" else warnings).append(message)
        signals = out3.get("signals")
        if not isinstance(signals, list):
            errors.append("step_3.output.signals 缺失或非列表")
        else:
            for i, sig in enumerate(signals, start=1):
                if not isinstance(sig, dict):
                    errors.append(f"step_3.output.signals[{i}] 不是对象")
                    continue
                signal_id_raw = sig.get("signal_id")
                if not isinstance(signal_id_raw, str):
                    errors.append(f"step_3.output.signals[{i}].signal_id 缺失或非字符串")
                else:
                    signal_id = normalize_signal_id(signal_id_raw)
                    if signal_id not in allowed_signal_ids:
                        errors.append(
                            f"step_3.output.signals[{i}].signal_id 不在白名单: {signal_id_raw}"
                        )
                if not isinstance(sig.get("phase"), (str, int)):
                    errors.append(f"step_3.output.signals[{i}].phase 缺失或非字符串/整数")
                if not isinstance(sig.get("description"), str):
                    errors.append(f"step_3.output.signals[{i}].description 缺失或非字符串")

    if out1 is not None:
        entities = out1.get("entities")
        if not isinstance(entities, dict):
            errors.append("step_1.output.entities 缺失或非对象")

    for step_key in (
        "step_1_entity_extraction",
        "step_2_phase_identification",
        "step_3_signal_identification",
        "step_4_judgment",
    ):
        step = payload.get(step_key)
        if isinstance(step, dict):
            think = step.get("think")
            if isinstance(think, str):
                validate_think_quality(step_key, think, errors, warnings)

    if out4 is not None:
        is_fraud = out4.get("is_fraud")
        if not isinstance(is_fraud, bool):
            errors.append("step_4.output.is_fraud 缺失或非布尔值")
        elif is_fraud != expected_is_fraud:
            message = f"step_4.output.is_fraud={is_fraud} 与样本期望 {expected_is_fraud} 不一致"
            (errors if label_mode == "constrained" else warnings).append(message)
        confidence = out4.get("confidence")
        if not isinstance(confidence, (int, float)):
            errors.append("step_4.output.confidence 缺失或非数值")
        elif float(confidence) < 0 or float(confidence) > 1:
            errors.append("step_4.output.confidence 必须在 [0,1] 区间")

        if is_fraud is True:
            if not isinstance(out4.get("fraud_type"), str):
                errors.append("step_4.output.fraud_type 缺失或非字符串")
            if not isinstance(out4.get("disambiguation"), str):
                errors.append("step_4.output.disambiguation 缺失或非字符串")
            if not isinstance(out4.get("reasoning_summary"), str):
                errors.append("step_4.output.reasoning_summary 缺失或非字符串")
        if is_fraud is False:
            if not isinstance(out4.get("call_category"), str):
                errors.append("step_4.output.call_category 缺失或非字符串")
            basis = out4.get("normal_judgment_basis")
            if not isinstance(basis, list) or not basis:
                errors.append("step_4.output.normal_judgment_basis 缺失或为空列表")

    if label_mode == "constrained" and label_hints:
        expected_category = label_hints.get("fraud_type") or label_hints.get("call_category")
        if isinstance(expected_category, str) and out4 is not None:
            model_category = out4.get("fraud_type") if expected_is_fraud else out4.get("call_category")
            if isinstance(model_category, str) and model_category != expected_category:
                errors.append(
                    f"Step4 类别不一致: output={model_category}, expected={expected_category}"
                )

        if expected_is_fraud and out3 is not None:
            expected_pairs_raw = label_hints.get("signals_with_phase")
            if isinstance(expected_pairs_raw, list) and expected_pairs_raw:
                expected_pairs = set()
                expected_ids = set()
                for item in expected_pairs_raw:
                    if not isinstance(item, dict):
                        continue
                    sid = item.get("signal_id")
                    ph = item.get("phase")
                    if isinstance(sid, str) and isinstance(ph, (str, int)):
                        sid2 = normalize_signal_id(sid)
                        ph2 = str(ph)
                        if sid2:
                            expected_pairs.add((sid2, ph2))
                            expected_ids.add(sid2)

                model_signals = out3.get("signals")
                if isinstance(model_signals, list):
                    got_pairs = set()
                    for item in model_signals:
                        if not isinstance(item, dict):
                            continue
                        sid = item.get("signal_id")
                        ph = item.get("phase")
                        if isinstance(sid, str) and isinstance(ph, (str, int)):
                            sid2 = normalize_signal_id(sid)
                            if sid2:
                                got_pairs.add((sid2, str(ph)))

                    missing = expected_pairs - got_pairs
                    for sid, ph in sorted(missing):
                        errors.append(f"Step3 信号缺失或阶段不一致: signal_id={sid}, expected_phase={ph}")

                    wrong_phase = [
                        (sid, ph) for (sid, ph) in got_pairs if sid in expected_ids and (sid, ph) not in expected_pairs
                    ]
                    for sid, ph in sorted(wrong_phase):
                        errors.append(f"Step3 信号阶段错位: signal_id={sid}, got_phase={ph}")

    return errors, warnings


def run_single_generation(
    llm: ThreadLocalLLMClient,
    model: str,
    temperature: float,
    user_prompt: str,
) -> str:
    return llm.chat(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )


def run_repair_generation(
    llm: ThreadLocalLLMClient,
    model: str,
    temperature: float,
    repair_prompt: str,
) -> str:
    return llm.chat(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": repair_prompt},
        ],
    )


def process_record(
    index: int,
    sample: Dict[str, Any],
    args: argparse.Namespace,
    llm: ThreadLocalLLMClient,
    source_chunk_resolver: Optional[FraudSourceChunkResolver],
) -> Dict[str, Any]:
    data_type = detect_data_type(sample)
    is_fraud = infer_is_fraud(sample, data_type)
    case_id = infer_case_id(sample, index, data_type)

    if args.use_existing_dialogue_text and isinstance(sample.get("dialogue_text"), str):
        dialogue_text = sample["dialogue_text"]
    else:
        dialogue_text = build_dialogue_text(sample, data_type)

    label_hints: Optional[Dict[str, Any]] = None
    if args.label_mode == "constrained":
        label_hints = collect_label_hints(
            sample=sample,
            data_type=data_type,
            is_fraud=is_fraud,
            source_chunk_resolver=source_chunk_resolver,
        )
        label_hints = sanitize_label_hints(label_hints, expected_is_fraud=is_fraud)

    base_user_prompt = build_user_prompt(dialogue_text=dialogue_text, label_hints=label_hints)
    prompt_trace: Dict[str, Any] = {
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": base_user_prompt,
        "repair_prompts": [],
    }

    last_error = ""
    last_output = ""
    for _ in range(args.max_retries):
        try:
            raw = run_single_generation(
                llm=llm,
                model=args.model,
                temperature=args.temperature,
                user_prompt=base_user_prompt,
            )
            obj = parse_json_object(raw)
            errors, warnings = validate_evidence_chain(
                payload=obj,
                expected_is_fraud=is_fraud,
                label_mode=args.label_mode,
                label_hints=label_hints,
            )

            repaired_raw = raw
            repaired_obj = obj
            for _ in range(args.repair_attempts):
                if not errors:
                    break
                repair_prompt = build_repair_prompt(
                    dialogue_text=dialogue_text,
                    invalid_json_text=repaired_raw,
                    validation_errors=errors,
                    label_hints=label_hints,
                )
                prompt_trace["repair_prompts"].append(repair_prompt)
                repaired_raw = run_repair_generation(
                    llm=llm,
                    model=args.model,
                    temperature=args.temperature,
                    repair_prompt=repair_prompt,
                )
                repaired_obj = parse_json_object(repaired_raw)
                errors, warnings = validate_evidence_chain(
                    payload=repaired_obj,
                    expected_is_fraud=is_fraud,
                    label_mode=args.label_mode,
                    label_hints=label_hints,
                )

            if errors:
                raise ValueError("; ".join(errors))

            out: Dict[str, Any] = {
                "index": index,
                "case_id": case_id,
                "status": "ok",
                "data_type": data_type,
                "is_fraud": is_fraud,
                "evidence_chain": repaired_obj,
                "warnings": warnings,
            }
            if args.test_mode:
                out["prompt_trace"] = prompt_trace
                out["prompt"] = base_user_prompt
            return out
        except Exception as exc:
            last_error = format_exception_message(exc)
            last_output = locals().get("raw", "") or locals().get("repaired_raw", "")
            time.sleep(1.0)

    out = {
        "index": index,
        "case_id": case_id,
        "status": "error",
        "data_type": data_type,
        "is_fraud": is_fraud,
        "error": last_error or "unknown error",
        "raw_output": last_output[:4000],
    }
    if args.test_mode:
        out["prompt_trace"] = prompt_trace
        out["prompt"] = base_user_prompt
    return out


def iter_target_indices(
    total: int,
    start: int,
    end: Optional[int],
    limit: Optional[int],
    done_status: Dict[int, str],
    retry_errors: bool,
) -> Iterable[int]:
    real_end = total - 1 if end is None else min(end, total - 1)
    count = 0
    for idx in range(max(0, start), real_end + 1):
        old = done_status.get(idx)
        if old is not None:
            if old == "ok":
                continue
            if old == "error" and not retry_errors:
                continue
        yield idx
        count += 1
        if limit is not None and count >= limit:
            return


def render_progress(
    finished: int,
    total: int,
    ok_count: int,
    err_count: int,
    start_ts: float,
    width: int = 26,
) -> str:
    if total <= 0:
        return "Progress [--------------------------] 0/0 ok=0 err=0"
    ratio = min(1.0, max(0.0, finished / total))
    filled = int(ratio * width)
    bar = "#" * filled + "-" * (width - filled)
    elapsed = max(0.001, time.time() - start_ts)
    speed = finished / elapsed
    eta = (total - finished) / speed if speed > 1e-9 else float("inf")
    eta_text = "--" if eta == float("inf") else f"{int(eta)}s"
    return (
        f"Progress [{bar}] {finished}/{total} "
        f"ok={ok_count} err={err_count} {speed:.2f}/s eta={eta_text}"
    )


def _canon_turn_range(value: Any) -> Tuple[Optional[int], Optional[int]]:
    if isinstance(value, list) and len(value) == 2:
        try:
            return int(value[0]), int(value[1])
        except Exception:
            return None, None
    return None, None


def canonicalize_label_phases(phases: Any) -> List[Tuple[str, str, Tuple[Optional[int], Optional[int]]]]:
    out: List[Tuple[str, str, Tuple[Optional[int], Optional[int]]]] = []
    if not isinstance(phases, list):
        return out
    for i, phase in enumerate(phases, start=1):
        if not isinstance(phase, dict):
            continue
        phase_id = str(phase.get("phase_id", phase.get("id", i)))
        phase_type = normalize_phase_type(phase.get("type", phase.get("phase_type", "unknown"))) or "unknown"
        out.append((phase_id, phase_type, _canon_turn_range(phase.get("turn_range"))))
    return out


def canonicalize_cot_phases(evidence_chain: Dict[str, Any]) -> List[Tuple[str, str, Tuple[Optional[int], Optional[int]]]]:
    out: List[Tuple[str, str, Tuple[Optional[int], Optional[int]]]] = []
    out2 = (
        ((evidence_chain.get("step_2_phase_identification") or {}).get("output") or {})
        if isinstance(evidence_chain, dict)
        else {}
    )
    phases = out2.get("phases")
    if not isinstance(phases, list):
        return out
    for i, phase in enumerate(phases, start=1):
        if not isinstance(phase, dict):
            continue
        phase_id = str(phase.get("id", phase.get("phase_id", i)))
        phase_type = normalize_phase_type(phase.get("type", phase.get("phase_type", "unknown"))) or "unknown"
        out.append((phase_id, phase_type, _canon_turn_range(phase.get("turn_range"))))
    return out


def dedupe_keep_order(values: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for v in values:
        if isinstance(v, str) and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def label_signal_list(sample: Dict[str, Any], data_type: str) -> List[str]:
    labels = sample.get("labels")
    if not isinstance(labels, dict):
        return []
    keys = ["level_3_signals", "level_3_tactics"] if data_type == "fraud" else [
        "level_3_signals",
        "service_behaviors",
    ]
    for key in keys:
        value = labels.get(key)
        if isinstance(value, list):
            out: List[str] = []
            for x in value:
                sid = normalize_signal_id(x)
                if sid:
                    out.append(sid)
            return out
    return []


def cot_signal_list(evidence_chain: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    out3 = (
        ((evidence_chain.get("step_3_signal_identification") or {}).get("output") or {})
        if isinstance(evidence_chain, dict)
        else {}
    )
    signals = out3.get("signals")
    if not isinstance(signals, list):
        return out
    for item in signals:
        if not isinstance(item, dict):
            continue
        sid = normalize_signal_id(item.get("signal_id"))
        if sid:
            out.append(sid)
    return out


def label_phase_list(sample: Dict[str, Any]) -> List[Tuple[str, str, Tuple[Optional[int], Optional[int]]]]:
    labels = sample.get("labels")
    if not isinstance(labels, dict):
        return []
    phases = labels.get("level_2_phases")
    return canonicalize_label_phases(phases)


def expected_category_from_sample(sample: Dict[str, Any], data_type: str) -> Optional[str]:
    return extract_category_hint(sample, data_type)


def actual_category_from_chain(evidence_chain: Dict[str, Any], is_fraud: bool) -> Optional[str]:
    out4 = (
        ((evidence_chain.get("step_4_judgment") or {}).get("output") or {})
        if isinstance(evidence_chain, dict)
        else {}
    )
    value = out4.get("fraud_type") if is_fraud else out4.get("call_category")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def signal_compare_ok(expected: List[str], actual: List[str], mode: str) -> bool:
    if mode == "raw":
        return expected == actual
    if mode == "dedupe_order":
        return dedupe_keep_order(expected) == dedupe_keep_order(actual)
    return set(expected) == set(actual)


def validate_row_against_label(
    sample: Dict[str, Any],
    row: Dict[str, Any],
    signal_mode: str,
) -> Tuple[bool, Dict[str, Any]]:
    data_type = detect_data_type(sample)
    is_fraud = infer_is_fraud(sample, data_type)
    evidence_chain = row.get("evidence_chain")
    if row.get("status") != "ok" or not isinstance(evidence_chain, dict):
        return False, {"reason": "row_not_ok", "status": row.get("status"), "error": row.get("error")}

    expected_category = expected_category_from_sample(sample, data_type)
    actual_category = actual_category_from_chain(evidence_chain, is_fraud)
    category_ok = expected_category is None or expected_category == actual_category

    expected_phases = label_phase_list(sample)
    actual_phases = canonicalize_cot_phases(evidence_chain)
    phase_ok = (not expected_phases) or (expected_phases == actual_phases)

    expected_signals = label_signal_list(sample, data_type)
    actual_signals = cot_signal_list(evidence_chain)
    signal_ok = (not expected_signals) or signal_compare_ok(expected_signals, actual_signals, signal_mode)

    ok = category_ok and phase_ok and signal_ok
    return ok, {
        "category_ok": category_ok,
        "phase_ok": phase_ok,
        "signal_ok": signal_ok,
        "expected_category": expected_category,
        "actual_category": actual_category,
        "expected_phases": expected_phases,
        "actual_phases": actual_phases,
        "expected_signals": expected_signals,
        "actual_signals": actual_signals,
    }


def run_generation_pass(
    records: List[Dict[str, Any]],
    target_indices: List[int],
    args: argparse.Namespace,
    output_path: Path,
    output_json_dir: Optional[Path],
    llm: ThreadLocalLLMClient,
    source_chunk_resolver: Optional[FraudSourceChunkResolver],
) -> Dict[int, Dict[str, Any]]:
    row_by_index: Dict[int, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        in_flight: Dict[Any, int] = {}
        next_ptr = 0
        start_ts = time.time()
        finished = 0
        ok_count = 0
        err_count = 0

        while next_ptr < len(target_indices) and len(in_flight) < args.max_workers:
            idx = target_indices[next_ptr]
            future = executor.submit(
                process_record,
                idx,
                records[idx],
                args,
                llm,
                source_chunk_resolver,
            )
            in_flight[future] = idx
            next_ptr += 1

        use_tqdm = tqdm is not None
        pbar = None
        if use_tqdm:
            pbar = tqdm(
                total=len(target_indices),
                desc="CoT",
                unit="sample",
                dynamic_ncols=True,
            )
            pbar.set_postfix(ok=0, err=0)
        else:
            print(render_progress(0, len(target_indices), 0, 0, start_ts), end="", flush=True)

        while in_flight:
            done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                idx = in_flight.pop(future)
                try:
                    row = future.result()
                except Exception as exc:  # defensive guard
                    row = {
                        "index": idx,
                        "case_id": f"unknown_{idx}",
                        "status": "error",
                        "error": f"worker exception: {exc}",
                    }
                row_by_index[idx] = row
                append_jsonl(output_path, row)
                if output_json_dir is not None:
                    write_single_json(output_json_dir, row)
                finished += 1
                if row.get("status") == "ok":
                    ok_count += 1
                else:
                    err_count += 1
                    err_msg = str(row.get("error") or "unknown error")
                    brief = err_msg if len(err_msg) <= 280 else err_msg[:280] + "..."
                    line = f"[index={idx}] {brief}"
                    if pbar is not None:
                        pbar.write(line)
                    else:
                        print("\n" + line, flush=True)
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(ok=ok_count, err=err_count)
                else:
                    print(
                        "\r"
                        + render_progress(
                            finished, len(target_indices), ok_count, err_count, start_ts
                        ),
                        end="",
                        flush=True,
                    )

                if next_ptr < len(target_indices):
                    new_idx = target_indices[next_ptr]
                    new_future = executor.submit(
                        process_record,
                        new_idx,
                        records[new_idx],
                        args,
                        llm,
                        source_chunk_resolver,
                    )
                    in_flight[new_future] = new_idx
                    next_ptr += 1

    if pbar is not None:
        pbar.close()
    else:
        print()
    return row_by_index


def main() -> None:
    args = parse_args()
    if not args.api_key or not args.base_url or not args.model:
        raise SystemExit("请显式配置 API key、base URL 和标注模型（OPENAI_ANNOTATION_MODEL / --model）。")

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_json_dir: Optional[Path] = (
        Path(args.output_json_dir) if args.output_json_dir.strip() else None
    )

    records = load_records(input_path)
    if args.labels_jsonl.strip():
        labels_path = Path(args.labels_jsonl)
        label_map = load_labels_jsonl(labels_path)
        matched, missing = attach_labels_by_case_id(records, label_map)
        print(f"已加载外部标签: {labels_path}，按 case_id 匹配 {matched} 条，未匹配 {missing} 条")

    done_status = load_done_status(output_path)
    target_indices = list(
        iter_target_indices(
            total=len(records),
            start=args.start_index,
            end=args.end_index,
            limit=args.limit,
            done_status=done_status,
            retry_errors=args.retry_errors,
        )
    )

    print(f"输入样本总数: {len(records)}")
    print(f"历史已处理条数: {len(done_status)}")
    print(f"本次待处理条数: {len(target_indices)}")
    print(f"输出文件: {output_path}")
    if output_json_dir is not None:
        print(f"逐条 JSON 输出目录: {output_json_dir}")

    if not target_indices:
        print("没有待处理样本。")
        return

    llm = ThreadLocalLLMClient(
        api_key=args.api_key,
        base_url=args.base_url,
        timeout_seconds=args.timeout,
    )

    source_chunk_dir: Optional[Path] = None
    if args.source_chunk_dir.strip():
        source_chunk_dir = Path(args.source_chunk_dir)
    source_chunk_resolver = FraudSourceChunkResolver(source_chunk_dir=source_chunk_dir)
    if not args.auto_validate:
        run_generation_pass(
            records=records,
            target_indices=target_indices,
            args=args,
            output_path=output_path,
            output_json_dir=output_json_dir,
            llm=llm,
            source_chunk_resolver=source_chunk_resolver,
        )
        print("全部完成。")
        return

    if args.auto_max_rounds < 1:
        raise SystemExit("--auto-max-rounds 必须 >= 1")

    pending_indices = list(target_indices)
    for round_id in range(1, args.auto_max_rounds + 1):
        if not pending_indices:
            print(f"[auto] 第 {round_id} 轮前已全部通过。")
            break

        print(f"[auto] 第 {round_id} 轮开始，待处理 {len(pending_indices)} 条。")
        rows = run_generation_pass(
            records=records,
            target_indices=pending_indices,
            args=args,
            output_path=output_path,
            output_json_dir=output_json_dir,
            llm=llm,
            source_chunk_resolver=source_chunk_resolver,
        )

        next_pending: List[int] = []
        validate_ok = 0
        validate_bad = 0
        category_bad = 0
        phase_bad = 0
        signal_bad = 0

        for idx in pending_indices:
            sample = records[idx]
            row = rows.get(idx)
            if row is None:
                next_pending.append(idx)
                validate_bad += 1
                continue
            ok, detail = validate_row_against_label(
                sample=sample,
                row=row,
                signal_mode=args.auto_signal_mode,
            )
            if ok:
                validate_ok += 1
                continue
            validate_bad += 1
            if detail.get("category_ok") is False:
                category_bad += 1
            if detail.get("phase_ok") is False:
                phase_bad += 1
            if detail.get("signal_ok") is False:
                signal_bad += 1
            next_pending.append(idx)

        print(
            f"[auto] 第 {round_id} 轮校验: pass={validate_ok}, fail={validate_bad}, "
            f"category_fail={category_bad}, phase_fail={phase_bad}, signal_fail={signal_bad}"
        )

        pending_indices = next_pending
        if not pending_indices:
            print(f"[auto] 第 {round_id} 轮后全部通过。")
            break

    if pending_indices:
        print(f"[auto] 完成最大轮次后仍未通过: {len(pending_indices)} 条。")
    else:
        print("[auto] 全部通过。")
    print("全部完成。")


if __name__ == "__main__":
    main()
