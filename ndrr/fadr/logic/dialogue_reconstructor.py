from typing import Any, Dict, List, Optional, Tuple

from agents.scammer_agent import ScammerAgent
from agents.director_agent import DirectorAgent
from agents.prompts.reconstruction_prompts import build_global_context
from agents.victim_agent import VictimAgent
from utils.openai_client import OpenAIClient


class FactAnchoredDialogueReconstructor:
    """根据 Stage1 结构化标注，使用 left/right/manager 三角色逐阶段还原对话。"""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        temperature: float = 0.7,
        manager_temperature: float = 0.2,
        max_tokens: int = 900,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        tail_size: int = 4,
        max_turns_per_phase: int = 6,
    ):
        self.model = model
        self.temperature = temperature
        self.manager_temperature = manager_temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.tail_size = tail_size
        self.max_turns_per_phase = max_turns_per_phase
        self.client = OpenAIClient(api_key=api_key, base_url=base_url)

    def reconstruct_case(
        self,
        structured_case: Dict[str, Any],
        case_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        case_meta = case_meta or {}
        fraud_type = structured_case.get("fraud_type") or case_meta.get("案件类别") or "未提及"
        victim_profile = structured_case.get("victim_profile") or {}
        initial_contact = structured_case.get("initial_contact") or {}
        turning_point = structured_case.get("turning_point") or "未提及"
        total_loss = structured_case.get("total_loss") or "未提及"
        phases = structured_case.get("fraud_phases") or []

        if not isinstance(phases, list) or not phases:
            raise ValueError("fraud_phases 为空或格式错误")

        global_context = build_global_context(
            fraud_type=fraud_type,
            victim_profile=victim_profile,
            initial_contact=initial_contact,
            turning_point=turning_point,
            total_loss=total_loss,
        )
        sorted_phases = sorted(
            phases,
            key=lambda x: x.get("phase_id", 10**9) if isinstance(x, dict) else 10**9,
        )

        phase_dialogues: List[Dict[str, Any]] = []
        flat_turns: List[Dict[str, Any]] = []
        previous_tail_turns: List[Dict[str, Any]] = []
        turn_id = 1

        for phase in sorted_phases:
            if not isinstance(phase, dict):
                continue

            phase_payload = self._run_one_phase(
                global_context=global_context,
                phase=phase,
                previous_tail_turns=previous_tail_turns,
            )
            turns = phase_payload["dialogue_turns"]
            phase_id = phase_payload["phase_id"]
            phase_type = phase_payload["phase_type"]

            for t in turns:
                flat_turns.append(
                    {
                        "turn_id": turn_id,
                        "phase_id": phase_id,
                        "phase_type": phase_type,
                        "speaker": t["speaker"],
                        "utterance": t["utterance"],
                    }
                )
                turn_id += 1

            phase_dialogues.append(phase_payload)
            previous_tail_turns = [
                {"speaker": t["speaker"], "utterance": t["utterance"]}
                for t in flat_turns[-self.tail_size :]
            ]

        return {
            "status": "ok",
            "case_meta": case_meta,
            "reconstructed_dialogue": {
                "phase_dialogues": phase_dialogues,
                "flat_turns": flat_turns,
                "turning_point": turning_point,
                "total_loss": total_loss,
            },
        }

    def _run_one_phase(
        self,
        global_context: Dict[str, Any],
        phase: Dict[str, Any],
        previous_tail_turns: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        left_seed, right_seed = self._seed_histories(previous_tail_turns)

        scammer_agent = ScammerAgent(
            client=self.client,
            model=self.model,
            global_context=global_context,
            phase=phase,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
        )
        victim_agent = VictimAgent(
            client=self.client,
            model=self.model,
            global_context=global_context,
            phase=phase,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
        )
        director_agent = DirectorAgent(
            client=self.client,
            model=self.model,
            global_context=global_context,
            phase=phase,
            temperature=self.manager_temperature,
            max_tokens=self.max_tokens,
            max_retries=self.max_retries,
            retry_delay=self.retry_delay,
        )
        scammer_agent.seed_history(left_seed)
        victim_agent.seed_history(right_seed)

        phase_dialogue_history: List[Dict[str, str]] = []

        left_message = scammer_agent.generate_opening(
            phase=phase,
            previous_tail_turns=previous_tail_turns,
        )
        self._append_scammer_turn(left_message, phase_dialogue_history)

        ended = False
        manager_reason = ""
        for round_idx in range(1, self.max_turns_per_phase + 1):
            right_message = victim_agent.generate_response(left_message)
            self._append_victim_turn(right_message, phase_dialogue_history)

            decision = director_agent.evaluate_phase(
                phase_dialogue_history=phase_dialogue_history,
                round_idx=round_idx,
            )
            if decision["should_end_phase"]:
                ended = True
                manager_reason = decision["reason"]
                break

            left_message = scammer_agent.generate_response(right_message)
            self._append_scammer_turn(left_message, phase_dialogue_history)

            decision = director_agent.evaluate_phase(
                phase_dialogue_history=phase_dialogue_history,
                round_idx=round_idx,
            )
            if decision["should_end_phase"]:
                ended = True
                manager_reason = decision["reason"]
                break

        if not ended:
            manager_reason = "达到阶段最大轮次，自动切换到下一阶段。"

        raw_turns = [
            {
                "speaker": "S" if msg["role"] == "left" else "V",
                "utterance": msg["content"],
            }
            for msg in phase_dialogue_history
        ]
        normalized_turns = self._normalize_turns(raw_turns)
        if not normalized_turns:
            normalized_turns = self._fallback_turns(phase)

        return {
            "phase_id": phase.get("phase_id"),
            "phase_type": phase.get("phase_type", "unknown"),
            "dialogue_turns": normalized_turns,
            "phase_notes": manager_reason,
        }

    @staticmethod
    def _append_scammer_turn(left_message: str, phase_dialogue_history: List[Dict[str, str]]) -> None:
        left_message = left_message.strip()
        if not left_message:
            return
        phase_dialogue_history.append({"role": "left", "content": left_message})

    @staticmethod
    def _append_victim_turn(right_message: str, phase_dialogue_history: List[Dict[str, str]]) -> None:
        right_message = right_message.strip()
        if not right_message:
            return
        phase_dialogue_history.append({"role": "right", "content": right_message})

    @staticmethod
    def _seed_histories(
        previous_tail_turns: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        left_history: List[Dict[str, str]] = []
        right_history: List[Dict[str, str]] = []
        for turn in previous_tail_turns:
            speaker = str(turn.get("speaker", "")).strip().upper()
            utterance = str(turn.get("utterance", "")).strip()
            if not utterance:
                continue
            if speaker == "S":
                left_history.append({"role": "assistant", "content": utterance})
                right_history.append({"role": "user", "content": utterance})
            elif speaker == "V":
                left_history.append({"role": "user", "content": utterance})
                right_history.append({"role": "assistant", "content": utterance})
        return left_history, right_history

    @staticmethod
    def _normalize_speaker(raw_speaker: str) -> Optional[str]:
        speaker = (raw_speaker or "").strip().lower()
        if speaker in {"s", "scammer", "left", "诈骗者"}:
            return "S"
        if speaker in {"v", "victim", "right", "受害人", "用户"}:
            return "V"
        return None

    def _normalize_turns(self, turns: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        normalized: List[Dict[str, str]] = []
        for t in turns:
            if not isinstance(t, dict):
                continue
            speaker = self._normalize_speaker(str(t.get("speaker", "")))
            utterance = str(t.get("utterance", "")).strip()
            if not speaker or not utterance:
                continue
            normalized.append({"speaker": speaker, "utterance": utterance})

        if not normalized:
            return []

        alternating: List[Dict[str, str]] = []
        prev = ""
        for t in normalized:
            current = t["speaker"]
            if current == prev:
                current = "V" if prev == "S" else "S"
            alternating.append({"speaker": current, "utterance": t["utterance"]})
            prev = current
        return alternating

    @staticmethod
    def _fallback_turns(phase: Dict[str, Any]) -> List[Dict[str, str]]:
        scammer_text = (
            str(phase.get("scammer_speech_summary") or "").strip()
            or str(phase.get("scammer_action") or "").strip()
            or "我这边按流程给您处理，您先配合一下。"
        )
        victim_text = (
            str(phase.get("victim_action") or "").strip()
            or str(phase.get("victim_psychology") or "").strip()
            or "好的，我先按你说的看看。"
        )
        return [
            {"speaker": "S", "utterance": scammer_text},
            {"speaker": "V", "utterance": victim_text},
        ]
