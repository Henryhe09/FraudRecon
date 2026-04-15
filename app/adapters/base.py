from __future__ import annotations

from typing import Any, Dict, List, Optional


class AdapterError(Exception):
    pass


class TransientAdapterError(AdapterError):
    pass


class BaseAdapter:
    def chat(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        modalities: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        raise NotImplementedError

