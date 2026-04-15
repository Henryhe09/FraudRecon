from __future__ import annotations

from typing import Any, Dict

from app.adapters.base import BaseAdapter
from app.adapters.mock import MockAdapter
from app.adapters.openai_compatible import OpenAICompatibleAdapter


def build_adapter(model_cfg: Dict[str, Any]) -> BaseAdapter:
    provider = model_cfg.get("provider")
    if provider == "mock":
        return MockAdapter(mode=model_cfg.get("mode", "main"))
    if provider == "openai_compatible":
        extra_payload: Dict[str, Any] = {}
        configured_extra = model_cfg.get("extra_payload", {})
        if isinstance(configured_extra, dict):
            extra_payload.update(configured_extra)
        chat_template_kwargs = model_cfg.get("chat_template_kwargs")
        if isinstance(chat_template_kwargs, dict):
            extra_payload["chat_template_kwargs"] = chat_template_kwargs
        return OpenAICompatibleAdapter(
            base_url=model_cfg["base_url"],
            api_key_env=model_cfg.get("api_key_env", "OPENAI_API_KEY"),
            api_key=str(model_cfg.get("api_key", "")),
            timeout_seconds=int(model_cfg.get("timeout_seconds", 120)),
            request_interval_seconds=float(model_cfg.get("request_interval_seconds", 0)),
            extra_payload=extra_payload,
        )
    raise ValueError(f"Unsupported provider: {provider}")
