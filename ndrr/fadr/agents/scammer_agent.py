from typing import Any, Dict, List

from agents.base_agent import BaseAgent
from agents.prompts.reconstruction_prompts import (
    build_scammer_opening_prompt,
    build_scammer_system_prompt,
)
from utils.openai_client import OpenAIClient


class ScammerAgent(BaseAgent):
    """Scammer-side agent for phase reconstruction."""

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
            system_prompt=build_scammer_system_prompt(global_context, phase),
            **kwargs,
        )

    def generate_opening(self, phase: Dict[str, Any], previous_tail_turns: List[Dict[str, Any]]) -> str:
        prompt = build_scammer_opening_prompt(phase, previous_tail_turns)
        return self.chat(prompt)

    def generate_response(self, right_message: str) -> str:
        return self.chat(right_message)

