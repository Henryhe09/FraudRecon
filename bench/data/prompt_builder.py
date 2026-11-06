from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from bench.data.dataset import CaseRecord, turns_to_text_with_options


def _load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _is_http_url(value: str) -> bool:
    s = str(value or "").strip().lower()
    return s.startswith("http://") or s.startswith("https://")


def _audio_format_from_ref(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "mp3"
    ref_path = urlparse(raw).path if _is_http_url(raw) else raw
    fmt = Path(ref_path).suffix.lstrip(".").lower()
    return fmt or "mp3"


class PromptBuilder:
    def __init__(self, prompts_dir: Path):
        self.dialogue_system = _load_prompt(prompts_dir / "dialogue_tasks.txt")
        self.dialogue_think_tag_system = _load_prompt(prompts_dir / "dialogue_tasks_think_tags.txt")
        self.detection_system = _load_prompt(prompts_dir / "fraud_detection.txt")
        self.category_system = _load_prompt(prompts_dir / "category_classification.txt")
        self.signals_system = _load_prompt(prompts_dir / "signal_recognition.txt")
        self.phase_system = _load_prompt(prompts_dir / "phase_identification.txt")
        self.evidence_chain_judge_system = _load_prompt(prompts_dir / "evidence_chain_judge.txt")

    @staticmethod
    def _join_url(prefix: str, relpath: str) -> str:
        p = str(prefix or "").strip()
        r = str(relpath or "").strip()
        if not p or not r:
            return ""
        return f"{p.rstrip('/')}/{r.lstrip('/')}"

    def _resolve_audio_url(
        self,
        case: CaseRecord,
        audio_relpath: str,
        audio_url_prefix: str,
        prefer_audio_url: bool,
    ) -> str:
        if not prefer_audio_url:
            return ""

        sample = case.sample or {}
        metadata = sample.get("metadata") if isinstance(sample, dict) else {}
        normal_meta = sample.get("normal_meta") if isinstance(sample, dict) else {}

        for container in (sample, metadata, normal_meta):
            if not isinstance(container, dict):
                continue
            for key in (
                "audio_url",
                "audio_uri",
                "audio_link",
                "audio_http_url",
                "media_url",
                "url",
            ):
                value = container.get(key)
                if isinstance(value, str) and _is_http_url(value):
                    return value.strip()

        for container in (sample.get("audio"), metadata.get("audio") if isinstance(metadata, dict) else None):
            if not isinstance(container, dict):
                continue
            for key in ("url", "audio_url", "uri"):
                value = container.get(key)
                if isinstance(value, str) and _is_http_url(value):
                    return value.strip()

        built = self._join_url(audio_url_prefix, audio_relpath)
        if _is_http_url(built):
            return built
        return ""

    def _audio_content(
        self,
        abs_audio_path: Optional[Path],
        audio_relpath: str,
        audio_url: str = "",
        model_name: str = "",
        omit_input_audio_type: bool = False,
        use_qwen3_omni_audio_payload: bool = False,
    ) -> Dict[str, Any]:
        model_token = str(model_name or "").strip().lower()
        is_qwen3_omni = model_token.startswith("qwen3-omni") or bool(use_qwen3_omni_audio_payload)
        is_type_less_audio = is_qwen3_omni or bool(omit_input_audio_type)

        if audio_url:
            fmt = _audio_format_from_ref(audio_url) or _audio_format_from_ref(audio_relpath)
            if is_type_less_audio:
                return {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_url,
                        "format": fmt,
                    },
                }
            return {
                "type": "input_audio",
                "input_audio": {
                    "type": "url",
                    "data": audio_url,
                    "format": fmt,
                },
            }

        if abs_audio_path is None:
            raise FileNotFoundError(f"Audio source missing for case audio_relpath={audio_relpath}")

        fmt = abs_audio_path.suffix.lstrip(".").lower() or _audio_format_from_ref(audio_relpath)
        audio_b64 = base64.b64encode(abs_audio_path.read_bytes()).decode("utf-8")
        if is_qwen3_omni:
            return {
                "type": "input_audio",
                "input_audio": {
                    "data": f"data:audio/{fmt};base64,{audio_b64}",
                    "format": fmt,
                },
            }
        if omit_input_audio_type:
            return {
                "type": "input_audio",
                "input_audio": {
                    "data": audio_b64,
                    "format": fmt,
                },
            }
        return {
            "type": "input_audio",
            "input_audio": {
                "type": "base64",
                "data": audio_b64,
                "format": fmt,
            },
        }

    @staticmethod
    def _normalize_longcat_omni_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized_messages: List[Dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role", "user"))
            content = message.get("content", "")

            normalized_content: List[Dict[str, Any]] = []
            if isinstance(content, str):
                normalized_content = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        normalized_content.append({"type": "text", "text": str(part)})
                        continue
                    copied = dict(part)
                    if copied.get("type") == "input_audio":
                        input_audio = copied.get("input_audio")
                        if isinstance(input_audio, dict):
                            fixed_input_audio = dict(input_audio)
                            source_type = str(fixed_input_audio.get("type", "")).strip().lower()
                            if source_type not in {"url", "base64"}:
                                fixed_input_audio["type"] = "base64"
                            copied["input_audio"] = fixed_input_audio
                    normalized_content.append(copied)
            else:
                normalized_content = [{"type": "text", "text": str(content)}]

            normalized_messages.append({"role": role, "content": normalized_content})

        return normalized_messages

    def build_dialogue_messages(
        self,
        case: CaseRecord,
        dataset_root: Path,
        input_mode: str,
        strip_tone_hints: bool = False,
        eval_mode: str = "full",
        model_name: str = "",
        qwen3_8b_think_tag_mode: bool = False,
        audio_url_prefix: str = "",
        prefer_audio_url: bool = True,
        omit_input_audio_type: bool = False,
        use_qwen3_omni_audio_payload: bool = False,
    ) -> List[Dict[str, Any]]:
        dialogue = turns_to_text_with_options(case.turns, strip_tone_hints=strip_tone_hints)
        base_text = (
            f"请分析以下通话对话并输出指定 JSON。\n\n"
            f"[Dialogue]\n{dialogue}\n"
        )

        if input_mode == "text":
            user_content: Any = base_text
        else:
            audio_relpath = case.audio_relpath
            audio_url = self._resolve_audio_url(
                case=case,
                audio_relpath=audio_relpath,
                audio_url_prefix=audio_url_prefix,
                prefer_audio_url=prefer_audio_url,
            )

            audio_abs: Optional[Path] = None
            if not audio_url:
                audio_abs = (dataset_root / case.split / audio_relpath).resolve()
                if not audio_abs.exists():
                    audio_abs = (dataset_root / audio_relpath).resolve()
                if not audio_abs.exists():
                    raise FileNotFoundError(
                        f"Audio file not found for case_id={case.case_id}, relpath={audio_relpath}"
                    )

            parts: List[Dict[str, Any]] = []
            if input_mode == "multimodal":
                parts.append({"type": "text", "text": base_text})
            parts.append(
                self._audio_content(
                    abs_audio_path=audio_abs,
                    audio_relpath=audio_relpath,
                    audio_url=audio_url,
                    model_name=model_name,
                    omit_input_audio_type=omit_input_audio_type,
                    use_qwen3_omni_audio_payload=use_qwen3_omni_audio_payload,
                )
            )
            user_content = parts

        if eval_mode == "acc_only":
            dialogue_system = self.detection_system
        elif eval_mode == "category_only":
            dialogue_system = self.category_system
        elif eval_mode == "signals_only":
            dialogue_system = self.signals_system
        elif qwen3_8b_think_tag_mode:
            dialogue_system = self.dialogue_think_tag_system
        else:
            dialogue_system = self.dialogue_system
        messages = [
            {"role": "system", "content": dialogue_system},
            {"role": "user", "content": user_content},
        ]

        if str(model_name or "").strip().lower().startswith("longcat-flash-omni"):
            return self._normalize_longcat_omni_messages(messages)
        return messages

    def build_phase_messages(self, case_id: str, prefix_dialogue: str, is_fraud: bool) -> List[Dict[str, Any]]:
        mode = "fraud_6phase" if is_fraud else "normal_3phase"
        user_text = (
            f"phase_system={mode}\n"
            f"下面是前缀对话（仅上文，无下文），请输出当前阶段标签。\n\n"
            f"[Prefix Dialogue]\n{prefix_dialogue}\n"
        )
        return [
            {"role": "system", "content": self.phase_system},
            {"role": "user", "content": user_text},
        ]

    def build_evidence_chain_judge_messages(
        self,
        case_id: str,
        dialogue: str,
        gold_labels: Dict[str, Any],
        model_output: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        user_text = (
            f"[Dialogue]\n{dialogue}\n\n"
            f"[Gold Labels]\n{gold_labels}\n\n"
            f"[Model Output]\n{model_output}\n"
        )
        return [
            {"role": "system", "content": self.evidence_chain_judge_system},
            {"role": "user", "content": user_text},
        ]
