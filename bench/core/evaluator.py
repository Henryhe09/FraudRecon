from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import random
import re
import sys
import time
from threading import Lock
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from bench.adapters.base import AdapterError, TransientAdapterError
from bench.adapters.factory import build_adapter
from bench.config import Config
from bench.modes import EVAL_MODE_ALIASES
from bench.data.dataset import (
    CaseRecord,
    build_phase_prefixes,
    load_cases,
    normalize_phase_type,
    turns_to_text_with_options,
)
from bench.data.prompt_builder import PromptBuilder
from bench.db.store import EvalStore
from bench.scoring.reasoning import aggregate_votes
from bench.scoring.metrics import accuracy, macro_f1, multilabel_f1, weighted_f1
from bench.utils import extract_json_obj, json_dumps


logger = logging.getLogger(__name__)


class CaseEvalError(RuntimeError):
    def __init__(self, message: str, raw_trace: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.raw_trace = raw_trace or {}


class EvaluationEngine:
    _FRAUD_CATEGORY_HINTS = {
        "刷单返利类",
        "虚假网络投资理财类",
        "冒充电商物流客服类",
        "贷款、代办信用卡类",
        "网络游戏产品虚假交易类",
        "虚假购物、服务类",
        "虚假征信类",
        "冒充公检法及政府机关类",
        "冒充领导、熟人类",
        "网络婚恋、交友类",
        "网黑案件",
        "冒充军警购物类诈骗",
    }
    _NORMAL_CATEGORY_HINTS = {
        "平台活动与返利服务",
        "投资理财咨询服务",
        "贷款与信用卡服务",
        "快递外卖与电商售后服务",
        "征信与信用管理服务",
        "政务公共服务",
        "购物消费与生活服务",
        "企业协作与熟人事务沟通",
        "游戏交易与账号服务",
        "婚恋交友平台服务",
        "账号隐私与反勒索支持",
        "政企采购与单位团购服务",
    }

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.cfg = Config(project_root)
        self.runtime = self.cfg.runtime
        self.prompt_builder = PromptBuilder(project_root / "bench" / "prompts")
        db_path = (project_root / self.runtime.get("db_path", "./outputs/bench/eval_runs.db")).resolve()
        self.store = EvalStore(db_path)
        self._adapter_cache: Dict[str, Any] = {}
        self._adapter_lock = Lock()

    def _build_model_adapter(self, model_id: str):
        model_cfg = self.cfg.models.get(model_id)
        if not model_cfg:
            raise ValueError(f"Unknown model_id: {model_id}")
        with self._adapter_lock:
            adapter = self._adapter_cache.get(model_id)
            if adapter is None:
                adapter = build_adapter(model_cfg)
                self._adapter_cache[model_id] = adapter
        return adapter, model_cfg

    @staticmethod
    def _safe_token(value: str, default: str = "unknown") -> str:
        token = re.sub(r"[^0-9A-Za-z._-]+", "-", str(value or "").strip())
        token = token.strip("-._")
        return token or default

    @staticmethod
    def _strict_bool(value: Any) -> Optional[bool]:
        return value if isinstance(value, bool) else None

    @staticmethod
    def _dedupe_keep_order(values: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for value in values:
            token = str(value or "").strip()
            if not token or token in seen:
                continue
            seen.add(token)
            out.append(token)
        return out

    @classmethod
    def _normalize_signal_ids(cls, raw_signals: Any) -> List[str]:
        if isinstance(raw_signals, str):
            return cls._dedupe_keep_order([raw_signals])
        if not isinstance(raw_signals, list):
            return []
        normalized: List[str] = []
        for item in raw_signals:
            if isinstance(item, str):
                normalized.append(item)
                continue
            if isinstance(item, dict):
                signal_id = item.get("signal_id") or item.get("id") or item.get("name")
                if isinstance(signal_id, str):
                    normalized.append(signal_id)
        return cls._dedupe_keep_order(normalized)

    @classmethod
    def _extract_signal_ids_from_text(cls, text: str) -> List[str]:
        signals: List[str] = []
        blacklist = {
            "signal_id",
            "confidence",
            "turns",
            "turn",
            "text",
            "description",
            "phase",
            "id",
        }

        for m in re.finditer(r"[\"']signal_id[\"']\s*:\s*[\"']([^\"']+)[\"']", text or "", flags=re.IGNORECASE):
            signals.append(m.group(1))

        for m in re.finditer(
            r"[\"'](?:level_3_signals|signals)[\"']\s*:\s*\[([\s\S]*?)\]",
            text or "",
            flags=re.IGNORECASE,
        ):
            block = m.group(1)
            for sm in re.finditer(r"[\"']([A-Za-z0-9_.-]+)[\"']", block):
                token = sm.group(1).strip()
                if not token:
                    continue
                token_low = token.lower()
                if token_low in blacklist:
                    continue
                if token_low != token:
                    continue
                if "_" not in token:
                    continue
                signals.append(token)

        return cls._dedupe_keep_order(signals)
    @classmethod
    def _recover_round1_json(cls, text: str, eval_mode: str = "full") -> Optional[Dict[str, Any]]:
        s = str(text or "").strip()
        if not s:
            return None

        if s.startswith("```"):
            s = re.sub(r"^```[A-Za-z0-9_-]*\n?", "", s)
            s = re.sub(r"\n?```$", "", s).strip()

        output: Dict[str, Any] = {}
        m_is_fraud = re.search(r"[\"']is_fraud[\"']\s*[:：]\s*(true|false)", s, flags=re.IGNORECASE)
        if m_is_fraud:
            output["is_fraud"] = m_is_fraud.group(1).lower() == "true"

        m_l1 = re.search(r"[\"']level_1_category[\"']\s*[:：]\s*[\"']([^\"']+)[\"']", s)
        if m_l1:
            output["level_1_category"] = m_l1.group(1).strip()
        m_call = re.search(r"[\"']call_category[\"']\s*[:：]\s*[\"']([^\"']+)[\"']", s)
        if m_call:
            output["call_category"] = m_call.group(1).strip()

        m_signal_type = re.search(r"[\"']signal_type[\"']\s*[:：]\s*[\"']([^\"']+)[\"']", s)
        if m_signal_type:
            output["signal_type"] = m_signal_type.group(1).strip()

        signal_ids = cls._extract_signal_ids_from_text(s)
        if signal_ids:
            output["level_3_signals"] = signal_ids

        m_conf = re.search(r"[\"']confidence[\"']\s*[:：]\s*(-?\d+(?:\.\d+)?)", s)
        if m_conf:
            try:
                conf = float(m_conf.group(1))
                output["confidence"] = max(0.0, min(1.0, conf))
            except Exception:
                pass

        if "signal_type" not in output and isinstance(output.get("is_fraud"), bool):
            output["signal_type"] = "fraud_tactics" if output["is_fraud"] else "service_behaviors"
        if "is_fraud" not in output:
            if output.get("signal_type") == "fraud_tactics":
                output["is_fraud"] = True
            elif output.get("signal_type") == "service_behaviors":
                output["is_fraud"] = False
        if "is_fraud" not in output:
            category = str(output.get("level_1_category", "")).strip()
            if category in cls._FRAUD_CATEGORY_HINTS:
                output["is_fraud"] = True
            elif category in cls._NORMAL_CATEGORY_HINTS:
                output["is_fraud"] = False

        if eval_mode == "acc_only" and "is_fraud" not in output:
            return None
        if not output:
            return None
        return {"think": "", "output": output}

    @staticmethod
    def _escape_control_chars_in_json_strings(text: str) -> str:
        out: List[str] = []
        in_string = False
        escaped = False

        for ch in text:
            if in_string:
                if escaped:
                    out.append(ch)
                    escaped = False
                    continue
                if ch == "\\":
                    out.append(ch)
                    escaped = True
                    continue
                if ch == '"':
                    out.append(ch)
                    in_string = False
                    continue
                if ch == "\n":
                    out.append("\\n")
                    continue
                if ch == "\r":
                    out.append("\\r")
                    continue
                if ch == "\t":
                    out.append("\\t")
                    continue
                out.append(ch)
                continue

            out.append(ch)
            if ch == '"':
                in_string = True

        return "".join(out)

    @classmethod
    def _parse_json_with_light_repair(cls, text: str) -> Optional[Dict[str, Any]]:
        s = str(text or "").strip()
        if not s:
            return None

        if s.startswith("```"):
            s = re.sub(r"^```[A-Za-z0-9_-]*\n?", "", s)
            s = re.sub(r"\n?```$", "", s).strip()

        candidates: List[str] = []

        def _add_candidate(value: str) -> None:
            token = str(value or "").strip()
            if token:
                candidates.append(token)

        _add_candidate(s)
        first = s.find("{")
        last = s.rfind("}")
        if first >= 0 and last > first:
            _add_candidate(s[first : last + 1])

        repaired_candidates: List[str] = []
        for base in candidates:
            fixed = cls._escape_control_chars_in_json_strings(base)
            fixed = re.sub(r"\[\s*(\d+)\s*-\s*(\d+)\s*\]", r"[\1,\2]", fixed)
            fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
            repaired_candidates.append(fixed)

        seen = set()
        ordered: List[str] = []
        for candidate in candidates + repaired_candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            ordered.append(candidate)

        for candidate in ordered:
            try:
                parsed = json.loads(candidate)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed

        return None

    def _resolve_eval_mode(self, eval_mode: Optional[str] = None) -> str:
        mode = str(eval_mode or self.runtime.get("eval_mode", "full")).strip().lower()
        mode = EVAL_MODE_ALIASES.get(mode, mode)
        if mode not in {
            "full",
            "full_no_t2",
            "round1_only",
            "acc_only",
            "category_only",
            "signals_only",
            "cot_only",
            "t2_only",
        }:
            raise ValueError(
                "Unsupported eval_mode: "
                f"{mode}. Expected one of: full, full_no_t2, round1_only, acc_only, category_only, signals_only, cot_only, t2_only"
            )
        return mode

    def _resolve_light_json_repair(self, enable_light_json_repair: Optional[bool] = None) -> bool:
        if enable_light_json_repair is None:
            return bool(self.runtime.get("enable_light_json_repair", True))
        return bool(enable_light_json_repair)

    @staticmethod
    def _uses_full_dialogue_prompt(eval_mode: str) -> bool:
        return eval_mode in {"full", "full_no_t2", "round1_only", "cot_only"}

    @staticmethod
    def _is_qwen3_8b_model(model_id: str, model_name: str) -> bool:
        token = f"{model_id} {model_name}".lower()
        return "qwen3-8b" in token

    @staticmethod
    def _is_glm_47_siliconflow_model(model_id: str, model_name: str) -> bool:
        _ = model_name
        return str(model_id or "").strip().lower() == "glm-4.7-siliconflow"

    @staticmethod
    def _extract_think_tag_content(text: str) -> str:
        s = str(text or "")
        blocks = re.findall(r"<think>([\s\S]*?)</think>", s, flags=re.IGNORECASE)
        merged = "\n\n".join([b.strip() for b in blocks if str(b).strip()]).strip()
        if merged:
            return merged

        open_match = re.search(r"<think>([\s\S]*)$", s, flags=re.IGNORECASE)
        if not open_match:
            return ""
        tail = open_match.group(1)
        json_start = tail.rfind("{")
        if json_start > 0:
            tail = tail[:json_start]
        return tail.strip()

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        s = str(text or "")
        s = re.sub(r"<think>[\s\S]*?</think>", "", s, flags=re.IGNORECASE)
        s = re.sub(r"</?think>", "", s, flags=re.IGNORECASE)
        return s.strip()

    @staticmethod
    def _extract_last_json_obj(text: str) -> Dict[str, Any]:
        raw = str(text or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-zA-Z]*\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        decoder = json.JSONDecoder()
        last_obj: Optional[Dict[str, Any]] = None
        search_from = 0
        while True:
            brace_idx = raw.find("{", search_from)
            if brace_idx < 0:
                break
            try:
                candidate, _ = decoder.raw_decode(raw[brace_idx:])
                if isinstance(candidate, dict):
                    last_obj = candidate
            except json.JSONDecodeError:
                pass
            search_from = brace_idx + 1

        if last_obj is not None:
            return last_obj
        return extract_json_obj(raw)

    @staticmethod
    def _merge_think_into_round1(raw: Dict[str, Any], think_text: str) -> Dict[str, Any]:
        if not isinstance(raw, dict):
            return raw
        merged = dict(raw)
        if think_text and not str(merged.get("think", "")).strip():
            merged["think"] = think_text
        return merged

    @staticmethod
    def _should_run_phases(eval_mode: str) -> bool:
        return eval_mode in {"full", "t2_only"}

    @staticmethod
    def _should_judge_evidence_chain(eval_mode: str) -> bool:
        return eval_mode in {"full", "full_no_t2", "cot_only"}

    @staticmethod
    def _should_run_dialogue_tasks(eval_mode: str) -> bool:
        return eval_mode != "t2_only"

    def make_run_id(self, model_id: str, split: str, input_mode: str) -> str:
        model_token = self._safe_token(model_id, default="model")
        split_token = self._safe_token(split, default="split")
        input_token = self._safe_token(input_mode, default="input")
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"run_{model_token}_{split_token}_{input_token}_{ts}"

    def resolve_dataset_root(self, dataset_root: Optional[str] = None, split: Optional[str] = None) -> Path:
        if dataset_root:
            root_spec = dataset_root
        else:
            split_roots = self.runtime.get("split_dataset_roots", {}) or {}
            root_spec = split_roots.get(split or "") or self.runtime.get("default_dataset_root")
        if not root_spec:
            raise ValueError("Dataset root is not configured")
        return (self.project_root / root_spec).resolve()

    @staticmethod
    def _append_raw_trace(raw_trace: Optional[Dict[str, Any]], key: str, payload: Dict[str, Any]) -> None:
        if raw_trace is None:
            return
        raw_trace.setdefault(key, []).append(payload)

    @staticmethod
    def _stringify_prompt_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for part in content:
                if not isinstance(part, dict):
                    parts.append(str(part))
                    continue
                p_type = part.get("type")
                if p_type == "text":
                    parts.append(str(part.get("text", "")))
                    continue
                if p_type == "input_audio":
                    audio = part.get("input_audio") or {}
                    fmt = str(audio.get("format", "unknown"))
                    data = str(audio.get("data", ""))
                    parts.append(f"[input_audio format={fmt} base64_chars={len(data)}]")
                    continue
                parts.append(str(part))
            return "\n".join([p for p in parts if p]).strip()
        return str(content)

    @staticmethod
    def _recover_acc_only_is_fraud(text: str) -> Optional[Dict[str, Any]]:
        s = str(text or "").strip()
        if not s:
            return None

        if s.startswith("```"):
            s = re.sub(r"^```[A-Za-z0-9_-]*\n?", "", s)
            s = re.sub(r"\n?```$", "", s).strip()

        match = re.search(r"[\"']?is_fraud[\"']?\s*[:：]\s*(true|false)", s, flags=re.IGNORECASE)
        if match:
            value = match.group(1).lower() == "true"
            return {"is_fraud": value, "output": {"is_fraud": value}}

        token = s.lower()
        if token in {"true", "false"}:
            value = token == "true"
            return {"is_fraud": value, "output": {"is_fraud": value}}

        return None

    @staticmethod
    def _recover_category_only_level_1(text: str) -> Optional[Dict[str, Any]]:
        s = str(text or "").strip()
        if not s:
            return None

        if s.startswith("```"):
            s = re.sub(r"^```[A-Za-z0-9_-]*\n?", "", s)
            s = re.sub(r"\n?```$", "", s).strip()

        match = re.search(r"[\"']?(?:level_1_category|call_category)[\"']?\s*[:：]\s*[\"']([^\"']+)[\"']", s)
        if match:
            category = match.group(1).strip()
            return {"level_1_category": category, "output": {"level_1_category": category}}

        compact = re.sub(r"\s+", "", s)
        candidates = sorted(EvaluationEngine._FRAUD_CATEGORY_HINTS | {"其他"}, key=len, reverse=True)
        for category in candidates:
            if category in compact:
                return {"level_1_category": category, "output": {"level_1_category": category}}

        return None

    @classmethod
    def _recover_signals_only_level_3(cls, text: str) -> Optional[Dict[str, Any]]:
        signal_ids = cls._extract_signal_ids_from_text(str(text or ""))
        if not signal_ids:
            return None
        return {"level_3_signals": signal_ids, "output": {"level_3_signals": signal_ids}}

    @classmethod
    def _build_input_prompt(
        cls,
        messages: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        sections: List[str] = []
        for idx, message in enumerate(messages, start=1):
            role = str(message.get("role", "unknown"))
            content = cls._stringify_prompt_content(message.get("content", ""))
            sections.append(f"[{idx}][{role}]\n{content}")
        return {"input_prompt": "\n\n".join(sections).strip()}

    def _dump_case_raw(
        self,
        raw_dir: Optional[Path],
        case_id: str,
        raw_trace: Optional[Dict[str, Any]],
        error: str = "",
    ) -> None:
        if not raw_dir:
            return
        raw_dir.mkdir(parents=True, exist_ok=True)
        case_token = self._safe_token(case_id, default="case")
        payload = {
            "case_id": case_id,
            "error": error,
            "raw_trace": raw_trace or {},
        }
        (raw_dir / f"{case_token}.json").write_text(json_dumps(payload), encoding="utf-8")

    @staticmethod
    def _render_progress(done: int, total: int, success: int, failed: int, width: int = 24) -> str:
        done = max(0, done)
        total = max(0, total)
        success = max(0, success)
        failed = max(0, failed)
        ratio = 1.0 if total == 0 else min(1.0, done / total)
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        percent = ratio * 100.0
        return (
            f"Progress [{bar}] {done}/{total} ({percent:5.1f}%) "
            f"success={success} failed={failed}"
        )

    def _print_progress(self, done: int, total: int, success: int, failed: int, final: bool = False) -> None:
        line = self._render_progress(done=done, total=total, success=success, failed=failed)
        end = "\n" if final else "\r"
        print(line, end=end, file=sys.stderr, flush=True)

    @staticmethod
    def _is_rate_limited_error(exc: Exception) -> bool:
        msg = str(exc or "")
        return "429" in msg or "rate limit" in msg.lower()

    def _call_with_retry(self, fn, max_retries: int, base_backoff: float, max_backoff: float, jitter_ratio: float):
        last_exc: Optional[Exception] = None
        rl_base = max(base_backoff, float(self.runtime.get("rate_limit_backoff_seconds", 8)))
        rl_max = max(max_backoff, float(self.runtime.get("rate_limit_max_backoff_seconds", 60)))
        for attempt in range(max_retries + 1):
            try:
                return fn()
            except TransientAdapterError as e:
                last_exc = e
                if attempt >= max_retries:
                    break
                if self._is_rate_limited_error(e):
                    wait = min(rl_max, rl_base * (2 ** attempt))
                else:
                    wait = min(max_backoff, base_backoff * (2 ** attempt))
                jitter = wait * jitter_ratio * random.random()
                time.sleep(wait + jitter)
            except AdapterError:
                raise
        raise last_exc if last_exc else RuntimeError("Model call failed")

    def _evaluate_dialogue_tasks(
        self,
        case: CaseRecord,
        dataset_root: Path,
        input_mode: str,
        model_id: str,
        eval_mode: str = "full",
        strip_tone_hints: bool = False,
        raw_trace: Optional[Dict[str, Any]] = None,
        enable_light_json_repair: Optional[bool] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        adapter, model_cfg = self._build_model_adapter(model_id)
        model_name = model_cfg.get("model_name", model_id)
        qwen3_8b_think_tag_mode = (
            self._is_qwen3_8b_model(model_id=model_id, model_name=model_name)
            and self._uses_full_dialogue_prompt(eval_mode)
        )
        glm_47_siliconflow_mode = self._is_glm_47_siliconflow_model(
            model_id=model_id, model_name=model_name
        )
        messages = self.prompt_builder.build_dialogue_messages(
            case,
            dataset_root,
            input_mode=input_mode,
            strip_tone_hints=strip_tone_hints,
            eval_mode=eval_mode,
            model_name=model_name,
            qwen3_8b_think_tag_mode=qwen3_8b_think_tag_mode,
            audio_url_prefix=str(model_cfg.get("audio_url_prefix", "")).strip(),
            prefer_audio_url=bool(model_cfg.get("prefer_audio_url", True)),
            omit_input_audio_type=bool(model_cfg.get("omit_input_audio_type", False)),
            use_qwen3_omni_audio_payload=bool(model_cfg.get("use_qwen3_omni_audio_payload", False)),
        )
        max_retries = int(model_cfg.get("max_retries", self.runtime.get("max_retries", 5)))
        light_json_repair_enabled = self._resolve_light_json_repair(enable_light_json_repair)
        round1_max_tokens = int(self.runtime.get("round1_max_tokens", 4096))

        def _invoke():
            trace_entry: Dict[str, Any] = {
                "case_id": case.case_id,
                "qwen3_8b_think_tag_mode": qwen3_8b_think_tag_mode,
                **self._build_input_prompt(messages),
            }
            self._append_raw_trace(raw_trace, "round1", trace_entry)
            request_context: Dict[str, Any] = {"task": "round1", "gold_labels": case.labels}
            if qwen3_8b_think_tag_mode:
                request_context["extra_payload"] = {
                    "chat_template_kwargs": {"enable_thinking": True}
                }
            text = adapter.chat(
                model_name=model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=round1_max_tokens,
                modalities=["text"],
                context=request_context,
            )
            trace_entry["response_text"] = text
            think_from_tag = ""
            parse_text = text
            if qwen3_8b_think_tag_mode or glm_47_siliconflow_mode:
                think_from_tag = self._extract_think_tag_content(text)
                parse_text = self._strip_think_tags(text)
                if think_from_tag:
                    trace_entry["think_from_tag_chars"] = len(think_from_tag)
            try:
                parsed = (
                    self._extract_last_json_obj(parse_text)
                    if glm_47_siliconflow_mode
                    else extract_json_obj(parse_text)
                )
                return self._merge_think_into_round1(parsed, think_from_tag)
            except Exception as e:
                if eval_mode == "acc_only":
                    recovered = self._recover_acc_only_is_fraud(parse_text)
                    recovered_from = "is_fraud_regex"
                elif eval_mode == "category_only":
                    recovered = self._recover_category_only_level_1(parse_text)
                    recovered_from = "category_regex"
                elif eval_mode == "signals_only":
                    recovered = self._recover_signals_only_level_3(parse_text)
                    recovered_from = "signals_regex"
                else:
                    recovered = self._recover_round1_json(parse_text, eval_mode=eval_mode)
                    recovered_from = "round1_field_regex"
                if recovered is not None:
                    trace_entry["parse_recovered"] = True
                    trace_entry["parse_error"] = str(e)
                    trace_entry["parse_recovered_from"] = recovered_from
                    return self._merge_think_into_round1(recovered, think_from_tag)

                if self._uses_full_dialogue_prompt(eval_mode):
                    if light_json_repair_enabled:
                        repaired = self._parse_json_with_light_repair(parse_text)
                        if repaired is not None:
                            trace_entry["parse_recovered"] = True
                            trace_entry["parse_error"] = str(e)
                            trace_entry["parse_recovered_from"] = "light_json_repair"
                            return self._merge_think_into_round1(repaired, think_from_tag)

                        # Last-resort fallback for full-like round1 mode: do not fail the whole case on pure format glitches.
                        trace_entry["parse_recovered"] = True
                        trace_entry["parse_error"] = str(e)
                        trace_entry["parse_recovered_from"] = "empty_output_fallback"
                        fallback = {"think": "", "output": {}}
                        return self._merge_think_into_round1(fallback, think_from_tag)

                    raise

                snippet = re.sub(r"\s+", " ", str(parse_text or "")).strip()[:240]
                raise TransientAdapterError(
                    f"Failed to parse {eval_mode} JSON: {e}; response_snippet={snippet}"
                ) from e

        raw = self._call_with_retry(
            _invoke,
            max_retries=max_retries,
            base_backoff=float(self.runtime.get("backoff_seconds", 2)),
            max_backoff=float(self.runtime.get("max_backoff_seconds", 45)),
            jitter_ratio=float(self.runtime.get("jitter_ratio", 0.25)),
        )
        output = raw.get("output", {})
        if not isinstance(output, dict):
            output = {}
        if "is_fraud" in raw and "is_fraud" not in output:
            output["is_fraud"] = raw.get("is_fraud")
        for key in ("level_1_category", "call_category", "signal_type", "level_3_signals", "confidence"):
            if key in raw and key not in output:
                output[key] = raw.get(key)
        normalized_raw = dict(raw)
        normalized_raw["output"] = output

        pred_is_fraud = self._strict_bool(output.get("is_fraud"))
        pred_category = output.get("level_1_category") or output.get("call_category") or ""
        pred_signals = self._normalize_signal_ids(output.get("level_3_signals", []))
        output["level_3_signals"] = pred_signals

        gold_is_fraud = self._strict_bool(case.labels.get("is_fraud"))
        gold_category = case.labels.get("level_1_category", "")
        gold_signals = self._normalize_signal_ids(case.labels.get("level_3_signals", []))

        if eval_mode == "acc_only":
            metrics = {
                "t1_is_fraud_correct": int(
                    pred_is_fraud is not None and gold_is_fraud is not None and pred_is_fraud == gold_is_fraud
                ),
                "t1_detection_correct": int(
                    pred_is_fraud is not None and gold_is_fraud is not None and pred_is_fraud == gold_is_fraud
                ),
            }
            return normalized_raw, metrics

        if eval_mode == "category_only":
            metrics = {
                "t1_category_correct": int(pred_category == gold_category),
                "t2_category_correct": int(pred_category == gold_category),
            }
            return normalized_raw, metrics

        if eval_mode == "signals_only":
            metrics = {
                "t3_sample_f1": round(multilabel_f1(gold_signals, pred_signals), 4),
                "t4_signal_sample_f1": round(multilabel_f1(gold_signals, pred_signals), 4),
            }
            return normalized_raw, metrics

        metrics = {
            "t1_is_fraud_correct": int(
                pred_is_fraud is not None and gold_is_fraud is not None and pred_is_fraud == gold_is_fraud
            ),
            "t1_category_correct": int(pred_category == gold_category),
            "t3_sample_f1": round(multilabel_f1(gold_signals, pred_signals), 4),
            "t1_detection_correct": int(
                pred_is_fraud is not None and gold_is_fraud is not None and pred_is_fraud == gold_is_fraud
            ),
            "t2_category_correct": int(pred_category == gold_category),
            "t4_signal_sample_f1": round(multilabel_f1(gold_signals, pred_signals), 4),
        }
        return normalized_raw, metrics

    def _evaluate_phases(
        self,
        case: CaseRecord,
        model_id: str,
        strip_tone_hints: bool = False,
        raw_trace: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        adapter, model_cfg = self._build_model_adapter(model_id)
        model_name = model_cfg.get("model_name", model_id)
        glm_47_siliconflow_mode = self._is_glm_47_siliconflow_model(
            model_id=model_id, model_name=model_name
        )
        prefixes = build_phase_prefixes(case, strip_tone_hints=strip_tone_hints)
        preds: List[Dict[str, Any]] = []
        y_true: List[str] = []
        y_pred: List[str] = []

        for p in prefixes:
            messages = self.prompt_builder.build_phase_messages(
                case_id=case.case_id,
                prefix_dialogue=p["dialogue_prefix"],
                is_fraud=case.is_fraud,
            )

            def _invoke():
                trace_entry: Dict[str, Any] = {
                    "case_id": case.case_id,
                    "phase_idx": p["phase_idx"],
                    "phase_id": p["phase_id"],
                    "end_turn": p["end_turn"],
                    **self._build_input_prompt(messages),
                }
                self._append_raw_trace(raw_trace, "round2", trace_entry)
                text = adapter.chat(
                    model_name=model_name,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=256,
                    modalities=["text"],
                    context={"task": "round2", "gold_phase_type": p["gold_phase_type"]},
                )
                trace_entry["response_text"] = text
                if glm_47_siliconflow_mode:
                    parse_text = self._strip_think_tags(text)
                    return self._extract_last_json_obj(parse_text)
                return extract_json_obj(text)

            raw = self._call_with_retry(
                _invoke,
                max_retries=int(model_cfg.get("max_retries", self.runtime.get("max_retries", 5))),
                base_backoff=float(self.runtime.get("backoff_seconds", 2)),
                max_backoff=float(self.runtime.get("max_backoff_seconds", 45)),
                jitter_ratio=float(self.runtime.get("jitter_ratio", 0.25)),
            )
            phase_pred = normalize_phase_type(raw.get("phase_type", ""), is_fraud=case.is_fraud)
            preds.append(
                {
                    "phase_idx": p["phase_idx"],
                    "phase_id": p["phase_id"],
                    "gold_phase_type": p["gold_phase_type"],
                    "pred_phase_type": phase_pred,
                    "end_turn": p["end_turn"],
                }
            )
            y_true.append(p["gold_phase_type"])
            y_pred.append(phase_pred)

        metrics = {
            "t2_phase_accuracy": round(accuracy(y_true, y_pred), 4),
            "t2_phase_macro_f1": round(macro_f1(y_true, y_pred), 4),
            "t3_phase_accuracy": round(accuracy(y_true, y_pred), 4),
            "t3_phase_macro_f1": round(macro_f1(y_true, y_pred), 4),
        }
        return {"prefix_predictions": preds}, metrics

    def _evaluate_evidence_chain(
        self,
        case: CaseRecord,
        round1_output: Dict[str, Any],
        judge_model_id: str,
        strip_tone_hints: bool = False,
        raw_trace: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        votes_n = int(self.cfg.eval_tasks.get("cot_eval", {}).get("judge_votes", 3))
        var_threshold = int(self.cfg.eval_tasks.get("cot_eval", {}).get("variance_retry_threshold", 2))
        cooldown = int(self.cfg.eval_tasks.get("cot_eval", {}).get("cooldown_seconds", 20))

        dialogue = turns_to_text_with_options(case.turns, strip_tone_hints=strip_tone_hints)
        adapter, model_cfg = self._build_model_adapter(judge_model_id)
        model_name = model_cfg.get("model_name", judge_model_id)
        glm_47_siliconflow_mode = self._is_glm_47_siliconflow_model(
            model_id=judge_model_id, model_name=model_name
        )
        messages = self.prompt_builder.build_evidence_chain_judge_messages(
            case_id=case.case_id,
            dialogue=dialogue,
            gold_labels=case.labels,
            model_output=round1_output,
        )

        def _invoke_once():
            trace_entry: Dict[str, Any] = {
                "case_id": case.case_id,
                **self._build_input_prompt(messages),
            }
            self._append_raw_trace(raw_trace, "cot_judge", trace_entry)
            text = adapter.chat(
                model_name=model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=512,
                modalities=["text"],
                context={"task": "cot_judge", "gold_labels": case.labels},
            )
            trace_entry["response_text"] = text
            if glm_47_siliconflow_mode:
                parse_text = self._strip_think_tags(text)
                return self._extract_last_json_obj(parse_text)
            return extract_json_obj(text)

        votes: List[Dict[str, Any]] = []
        for _ in range(votes_n):
            votes.append(
                self._call_with_retry(
                    _invoke_once,
                    max_retries=int(model_cfg.get("max_retries", self.runtime.get("max_retries", 5))),
                    base_backoff=float(self.runtime.get("backoff_seconds", 2)),
                    max_backoff=float(self.runtime.get("max_backoff_seconds", 45)),
                    jitter_ratio=float(self.runtime.get("jitter_ratio", 0.25)),
                )
            )

        first_scores = [int(v.get("cot_score_15", 3)) for v in votes]
        if votes_n >= 3 and max(first_scores) - min(first_scores) > var_threshold:
            time.sleep(cooldown)
            votes.append(
                self._call_with_retry(
                    _invoke_once,
                    max_retries=int(model_cfg.get("max_retries", self.runtime.get("max_retries", 5))),
                    base_backoff=float(self.runtime.get("backoff_seconds", 2)),
                    max_backoff=float(self.runtime.get("max_backoff_seconds", 45)),
                    jitter_ratio=float(self.runtime.get("jitter_ratio", 0.25)),
                )
            )

        return aggregate_votes(votes)

    def evaluate_single_case(
        self,
        case: CaseRecord,
        dataset_root: Path,
        model_id: str,
        judge_model_id: str,
        input_mode: str,
        eval_mode: Optional[str] = None,
        strip_tone_hints: bool = False,
        enable_light_json_repair: Optional[bool] = None,
    ) -> Dict[str, Any]:
        resolved_eval_mode = self._resolve_eval_mode(eval_mode)
        if self._should_run_dialogue_tasks(resolved_eval_mode):
            round1_out, round1_metrics = self._evaluate_dialogue_tasks(
                case,
                dataset_root,
                input_mode,
                model_id,
                eval_mode=resolved_eval_mode,
                strip_tone_hints=strip_tone_hints,
                enable_light_json_repair=enable_light_json_repair,
            )
            if self._should_judge_evidence_chain(resolved_eval_mode):
                cot = self._evaluate_evidence_chain(case, round1_out, judge_model_id, strip_tone_hints=strip_tone_hints)
            else:
                cot = {}
        else:
            round1_out = {}
            round1_metrics = {}
            cot = {}

        if self._should_run_phases(resolved_eval_mode):
            t2_out, t2_metrics = self._evaluate_phases(case, model_id, strip_tone_hints=strip_tone_hints)
        else:
            t2_out = {}
            t2_metrics = {}
        return {
            "case_id": case.case_id,
            "round1_output": round1_out,
            "round1_metrics": round1_metrics,
            "t2_output": t2_out,
            "t2_metrics": t2_metrics,
            "cot": cot,
        }

    def _evaluate_case_payload(
        self,
        case: CaseRecord,
        dataset_root: Path,
        input_mode: str,
        model_id: str,
        judge_model_id: str,
        strip_tone_hints: bool = False,
        dump_raw: bool = False,
        eval_mode: Optional[str] = None,
        enable_light_json_repair: Optional[bool] = None,
    ) -> Dict[str, Any]:
        resolved_eval_mode = self._resolve_eval_mode(eval_mode)
        raw_trace: Optional[Dict[str, Any]] = {} if dump_raw else None
        try:
            if self._should_run_dialogue_tasks(resolved_eval_mode):
                round1_out, round1_metrics = self._evaluate_dialogue_tasks(
                    case,
                    dataset_root,
                    input_mode,
                    model_id,
                    eval_mode=resolved_eval_mode,
                    strip_tone_hints=strip_tone_hints,
                    raw_trace=raw_trace,
                    enable_light_json_repair=enable_light_json_repair,
                )
                if self._should_judge_evidence_chain(resolved_eval_mode):
                    cot = self._evaluate_evidence_chain(
                        case,
                        round1_out,
                        judge_model_id,
                        strip_tone_hints=strip_tone_hints,
                        raw_trace=raw_trace,
                    )
                else:
                    cot = {}
            else:
                round1_out = {}
                round1_metrics = {}
                cot = {}

            if self._should_run_phases(resolved_eval_mode):
                t2_out, t2_metrics = self._evaluate_phases(
                    case,
                    model_id,
                    strip_tone_hints=strip_tone_hints,
                    raw_trace=raw_trace,
                )
            else:
                t2_out = {}
                t2_metrics = {}
            payload: Dict[str, Any] = {
                "case_id": case.case_id,
                "gold_labels": case.labels,
                "round1_output": round1_out,
                "round1_metrics": round1_metrics,
                "t2_output": t2_out,
                "t2_metrics": t2_metrics,
                "cot": cot,
            }
            if raw_trace is not None:
                payload["_raw_trace"] = raw_trace
            return payload
        except Exception as e:
            raise CaseEvalError(str(e), raw_trace=raw_trace) from e

    def _aggregate_batch(self, results: List[Dict[str, Any]], eval_mode: str = "full") -> Dict[str, Any]:
        y_true_t1: List[str] = []
        y_pred_t1: List[str] = []
        t1_is_fraud_accs: List[int] = []
        t4_f1s: List[float] = []
        t3_true: List[str] = []
        t3_pred: List[str] = []
        cot5: List[float] = []

        for r in results:
            gold = r["gold_labels"]
            pred = r["round1_output"].get("output", {})
            if not isinstance(pred, dict):
                pred = {}
            r1_metrics = r.get("round1_metrics", {}) or {}

            if eval_mode in {"full", "full_no_t2", "round1_only", "category_only"}:
                y_true_t1.append(str(gold.get("level_1_category", "")))
                y_pred_t1.append(str(pred.get("level_1_category", pred.get("call_category", ""))))

            if eval_mode in {"full", "full_no_t2", "round1_only", "acc_only"}:
                metric_flag = r1_metrics.get("t1_is_fraud_correct")
                if isinstance(metric_flag, (int, float)):
                    t1_is_fraud_accs.append(1 if int(metric_flag) == 1 else 0)
                else:
                    gold_bool = self._strict_bool(gold.get("is_fraud"))
                    pred_bool = self._strict_bool(pred.get("is_fraud"))
                    t1_is_fraud_accs.append(
                        1 if gold_bool is not None and pred_bool is not None and gold_bool == pred_bool else 0
                    )

            if eval_mode in {"full", "full_no_t2", "round1_only", "signals_only"}:
                t4_f1s.append(float(r1_metrics.get("t3_sample_f1", 0.0)))

            if self._should_judge_evidence_chain(eval_mode):
                cot5.append(float(r.get("cot", {}).get("cot_score_5", 0.0)))
            if self._should_run_phases(eval_mode):
                for p in r.get("t2_output", {}).get("prefix_predictions", []):
                    t3_true.append(str(p.get("gold_phase_type", "")))
                    t3_pred.append(str(p.get("pred_phase_type", "")))

        summary: Dict[str, Any] = {
            "num_cases": len(results),
        }

        if eval_mode in {"full", "full_no_t2", "round1_only", "acc_only"}:
            detection_acc = round(sum(t1_is_fraud_accs) / len(t1_is_fraud_accs), 4) if t1_is_fraud_accs else 0.0
            summary["T1_detection_accuracy"] = detection_acc
            summary["T1_is_fraud_accuracy"] = detection_acc
        if eval_mode in {"full", "full_no_t2", "round1_only", "category_only"}:
            category_weighted_f1 = round(weighted_f1(y_true_t1, y_pred_t1), 4) if y_true_t1 else 0.0
            summary["T2_category_weighted_f1"] = category_weighted_f1
            summary["T1_weighted_f1"] = category_weighted_f1
        if eval_mode in {"full", "full_no_t2", "round1_only", "signals_only"}:
            signal_f1 = round(sum(t4_f1s) / len(t4_f1s), 4) if t4_f1s else 0.0
            summary["T4_signal_macro_sample_f1"] = signal_f1
            summary["T3_macro_sample_f1"] = signal_f1

        if self._should_run_phases(eval_mode):
            phase_accuracy = round(accuracy(t3_true, t3_pred), 4) if t3_true else 0.0
            phase_macro_f1 = round(macro_f1(t3_true, t3_pred), 4) if t3_true else 0.0
            summary["T3_phase_accuracy"] = phase_accuracy
            summary["T3_phase_macro_f1"] = phase_macro_f1
            summary["T2_phase_accuracy"] = phase_accuracy
            summary["T2_phase_macro_f1"] = phase_macro_f1
        if self._should_judge_evidence_chain(eval_mode):
            reasoning_score = round(sum(cot5) / len(cot5), 4) if cot5 else 0.0
            summary["T5_reasoning_score_5_avg"] = reasoning_score
            summary["T4_cot_score_5_avg"] = reasoning_score

        return summary

    def run_batch(
        self,
        model_id: str,
        judge_model_id: str,
        dataset_root: Optional[str] = None,
        split: Optional[str] = None,
        input_mode: Optional[str] = None,
        run_id: Optional[str] = None,
        max_workers: Optional[int] = None,
        dump_raw: Optional[bool] = None,
        strip_tone_hints: Optional[bool] = None,
        eval_mode: Optional[str] = None,
        enable_light_json_repair: Optional[bool] = None,
    ) -> Dict[str, Any]:
        resolved_eval_mode = self._resolve_eval_mode(eval_mode)
        split = split or self.runtime.get("default_split", "test")
        ds_root = self.resolve_dataset_root(dataset_root=dataset_root, split=split)
        input_mode = input_mode or self.runtime.get("default_input_mode", "text")
        cases = load_cases(ds_root, split=split)
        run_id = run_id or self.make_run_id(model_id=model_id, split=split, input_mode=input_mode)
        resolved_workers = max(1, int(max_workers or self.runtime.get("api_max_workers", 4)))
        model_worker_limits: List[int] = []
        for cfg_id in (model_id, judge_model_id):
            cfg = self.cfg.models.get(cfg_id, {}) or {}
            cap = cfg.get("max_workers")
            if cap is None:
                continue
            try:
                model_worker_limits.append(max(1, int(cap)))
            except (TypeError, ValueError):
                logger.warning("Ignore invalid max_workers for model %s: %r", cfg_id, cap)
        if model_worker_limits:
            resolved_workers = min(resolved_workers, min(model_worker_limits))
        dump_raw_enabled = bool(
            self.runtime.get("dump_raw_responses", False) if dump_raw is None else dump_raw
        )
        strip_tone_hints_enabled = bool(
            self.runtime.get("strip_tone_hints", False) if strip_tone_hints is None else strip_tone_hints
        )
        light_json_repair_enabled = self._resolve_light_json_repair(enable_light_json_repair)
        case_map = {c.case_id: c for c in cases}
        artifact_dir = (self.project_root / self.runtime.get("artifacts_dir", "./outputs/bench") / run_id).resolve()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        raw_dir = artifact_dir / "raw_responses" if dump_raw_enabled else None

        run_cfg = {
            "model_id": model_id,
            "judge_model_id": judge_model_id,
            "dataset_root": str(ds_root),
            "split": split,
            "input_mode": input_mode,
            "max_workers": resolved_workers,
            "dump_raw": dump_raw_enabled,
            "strip_tone_hints": strip_tone_hints_enabled,
            "eval_mode": resolved_eval_mode,
            "light_json_repair": light_json_repair_enabled,
        }
        profile_fields = ("provider", "mode", "model_name", "base_url", "supports_audio",
                          "extra_payload", "chat_template_kwargs", "audio_url_prefix",
                          "prefer_audio_url", "omit_input_audio_type", "use_qwen3_omni_audio_payload")
        for name, profile_id in (("model_profile", model_id), ("judge_profile", judge_model_id)):
            profile = self.cfg.models.get(profile_id, {})
            run_cfg[name] = {key: profile[key] for key in profile_fields if key in profile}
        run_cfg["task_settings"] = self.cfg.eval_tasks
        run_cfg["round1_max_tokens"] = self.runtime.get("round1_max_tokens", 4096)
        self.store.create_or_resume_run(run_id, run_cfg, list(case_map.keys()))

        completed_results: List[Dict[str, Any]] = []
        pending_case_ids = self.store.get_pending_case_ids(run_id)
        total_cases = len(case_map)
        done_count = total_cases - len(pending_case_ids)
        success_count = done_count
        failed_count = 0
        self._print_progress(done=done_count, total=total_cases, success=success_count, failed=failed_count)
        for case_id in pending_case_ids:
            self.store.update_case(run_id, case_id, status="running", error="")

        if pending_case_ids:
            worker_n = min(resolved_workers, len(pending_case_ids))
            logger.info("Run %s starts with %d pending cases and %d workers", run_id, len(pending_case_ids), worker_n)
            with ThreadPoolExecutor(max_workers=worker_n, thread_name_prefix="eval-worker") as executor:
                future_to_case = {
                    executor.submit(
                        self._evaluate_case_payload,
                        case_map[case_id],
                        ds_root,
                        input_mode,
                        model_id,
                        judge_model_id,
                        strip_tone_hints_enabled,
                        dump_raw_enabled,
                        resolved_eval_mode,
                        light_json_repair_enabled,
                    ): case_id
                    for case_id in pending_case_ids
                }
                for future in as_completed(future_to_case):
                    case_id = future_to_case[future]
                    try:
                        result = future.result()
                        raw_trace = result.pop("_raw_trace", None) if isinstance(result, dict) else None
                        if dump_raw_enabled:
                            self._dump_case_raw(raw_dir, case_id, raw_trace, error="")
                        self.store.update_case(
                            run_id,
                            case_id,
                            status="success",
                            round1_json=result["round1_output"],
                            round1_metrics=result["round1_metrics"],
                            t2_json=result["t2_output"],
                            t2_metrics=result["t2_metrics"],
                            cot_json=result["cot"],
                        )
                        completed_results.append(result)
                        success_count += 1
                    except Exception as e:
                        raw_trace = getattr(e, "raw_trace", None)
                        if dump_raw_enabled:
                            self._dump_case_raw(raw_dir, case_id, raw_trace, error=str(e))
                        self.store.update_case(run_id, case_id, status="failed", error=str(e))
                        logger.error("Case failed: run_id=%s case_id=%s error=%s", run_id, case_id, str(e))
                        failed_count += 1
                    done_count += 1
                    self._print_progress(done=done_count, total=total_cases, success=success_count, failed=failed_count)

        # Include already completed cases for resume runs.
        run_rows = self.store.get_run_cases(run_id)
        for row in run_rows:
            if row.get("status") != "success":
                continue
            if any(x["case_id"] == row["case_id"] for x in completed_results):
                continue
            case = case_map[row["case_id"]]
            completed_results.append(
                {
                    "case_id": row["case_id"],
                    "gold_labels": case.labels,
                    "round1_output": extract_json_obj(row.get("round1_json") or "{}"),
                    "round1_metrics": extract_json_obj(row.get("round1_metrics_json") or "{}"),
                    "t2_output": extract_json_obj(row.get("t2_json") or "{}"),
                    "t2_metrics": extract_json_obj(row.get("t2_metrics_json") or "{}"),
                    "cot": extract_json_obj(row.get("cot_json") or "{}"),
                }
            )

        failed_cases = [
            {
                "case_id": row.get("case_id", ""),
                "error": row.get("error", "") or "",
            }
            for row in run_rows
            if row.get("status") == "failed"
        ]
        completed_results = sorted(completed_results, key=lambda x: x.get("case_id", ""))
        failed_cases = sorted(failed_cases, key=lambda x: x.get("case_id", ""))
        self._print_progress(
            done=min(total_cases, len(completed_results) + len(failed_cases)),
            total=total_cases,
            success=len(completed_results),
            failed=len(failed_cases),
            final=True,
        )

        summary = self._aggregate_batch(completed_results, eval_mode=resolved_eval_mode)
        summary["num_cases_total"] = len(case_map)
        summary["num_cases_success"] = len(completed_results)
        summary["num_cases_failed"] = len(failed_cases)

        if failed_cases:
            summary["failed_case_ids"] = [x["case_id"] for x in failed_cases]

        if failed_cases and completed_results:
            run_status = "partial_failed"
        elif failed_cases:
            run_status = "failed"
        else:
            run_status = "completed"

        self.store.set_run_status(run_id, run_status, summary=summary)

        (artifact_dir / "summary.json").write_text(json_dumps(summary), encoding="utf-8")
        (artifact_dir / "cases.json").write_text(json_dumps(completed_results), encoding="utf-8")
        (artifact_dir / "errors.json").write_text(json_dumps(failed_cases), encoding="utf-8")

        return {
            "run_id": run_id,
            "status": run_status,
            "summary": summary,
            "failed_cases": failed_cases,
        }
