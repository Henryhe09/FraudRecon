import json
from typing import Any, Dict, List

from agents.base_agent import BaseAgent
from agents.prompts.reconstruction_prompts import (
    build_director_eval_prompt,
    build_director_system_prompt,
)
from utils.openai_client import OpenAIClient


class DirectorAgent(BaseAgent):
    """Manager agent that decides whether current phase should end."""

    def __init__(
        self,
        client: OpenAIClient,
        model: str,
        global_context: Dict[str, Any],
        phase: Dict[str, Any],
        **kwargs,
    ):
        super().__init__(
            client=client,
            model=model,
            system_prompt=build_director_system_prompt(global_context, phase),
            **kwargs,
        )

    def evaluate_phase(self, phase_dialogue_history: List[Dict[str, str]], round_idx: int) -> Dict[str, Any]:
        prompt = build_director_eval_prompt(phase_dialogue_history, round_index=round_idx)
        # Manager does stateless evaluation; avoid carrying previous eval history.
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        reply = self._call_with_retry(messages=messages, temperature=self.temperature)
        try:
            parsed = self._parse_json_response(reply)
            return {
                "should_end_phase": bool(parsed.get("should_end_phase", False)),
                "reason": str(parsed.get("reason", "") or "manager 判断阶段结束"),
                "coverage": str(parsed.get("coverage", "") or "unknown"),
            }
        except Exception:
            return {
                "should_end_phase": False,
                "reason": "manager 解析失败，继续本阶段",
                "coverage": "unknown",
            }

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        raw = (text or "").strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

        for candidate in self._extract_json_candidates(raw):
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
        raise ValueError("无法解析 manager JSON")

    @staticmethod
    def _extract_json_candidates(text: str) -> List[str]:
        candidates: List[str] = []
        stack: List[int] = []
        for idx, ch in enumerate(text):
            if ch == "{":
                stack.append(idx)
            elif ch == "}":
                if not stack:
                    continue
                start = stack.pop()
                if not stack:
                    candidates.append(text[start : idx + 1])
        candidates.sort(key=len, reverse=True)
        return candidates

