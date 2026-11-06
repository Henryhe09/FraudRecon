import time
from typing import Dict, List, Optional

from utils.openai_client import OpenAIClient


class BaseAgent:
    """Shared retry/history logic for role agents."""

    def __init__(
        self,
        client: OpenAIClient,
        model: str,
        system_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 900,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.history: List[Dict[str, str]] = []

    def seed_history(self, history: List[Dict[str, str]]) -> None:
        self.history = list(history)

    def chat(self, user_message: str, temperature: Optional[float] = None) -> str:
        messages = [{"role": "system", "content": self.system_prompt}] + self.history + [
            {"role": "user", "content": user_message}
        ]
        content = self._call_with_retry(messages, temperature=temperature or self.temperature)
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": content})
        return content

    def _call_with_retry(self, messages: List[Dict[str, str]], temperature: float) -> str:
        last_err = ""
        for _ in range(self.max_retries):
            try:
                reply = self.client.chat_completion(
                    messages=messages,
                    model=self.model,
                    temperature=temperature,
                    max_tokens=self.max_tokens,
                )
                content = (reply or "").strip()
                if content:
                    return content
                last_err = "模型返回空内容"
            except Exception as e:
                last_err = str(e)
            time.sleep(self.retry_delay)
        raise RuntimeError(f"agent 调用失败: {last_err}")

