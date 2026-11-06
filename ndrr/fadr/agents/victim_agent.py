from typing import Any, Dict

from agents.base_agent import BaseAgent
from agents.prompts.reconstruction_prompts import build_victim_system_prompt
from utils.openai_client import OpenAIClient


class VictimAgent(BaseAgent):
    """Victim-side agent for phase reconstruction."""

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
            system_prompt=build_victim_system_prompt(global_context, phase),
            **kwargs,
        )

    def generate_response(self, left_message: str) -> str:
        return self.chat(left_message)

