from __future__ import annotations

import os
import time
from threading import Lock
from typing import Any, Dict, List, Optional

import httpx

from bench.adapters.base import BaseAdapter, AdapterError, TransientAdapterError


class OpenAICompatibleAdapter(BaseAdapter):
    def __init__(
        self,
        base_url: str,
        api_key_env: str,
        api_key: str = "",
        timeout_seconds: int = 120,
        request_interval_seconds: float = 0.0,
        extra_payload: Optional[Dict[str, Any]] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.api_key = str(api_key or "").strip()
        self.timeout_seconds = timeout_seconds
        self.request_interval_seconds = max(0.0, float(request_interval_seconds or 0.0))
        self.extra_payload = dict(extra_payload or {})
        self._client: Optional[httpx.Client] = None
        self._client_lock = Lock()
        self._request_interval_lock = Lock()
        self._next_request_time = 0.0

    def _get_client(self) -> httpx.Client:
        client = self._client
        if client is not None:
            return client
        with self._client_lock:
            client = self._client
            if client is None:
                client = httpx.Client(timeout=self.timeout_seconds)
                self._client = client
        return client

    def close(self) -> None:
        with self._client_lock:
            client = self._client
            self._client = None
        if client is not None:
            client.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _wait_for_request_slot(self) -> None:
        if self.request_interval_seconds <= 0:
            return
        while True:
            with self._request_interval_lock:
                now = time.monotonic()
                if now >= self._next_request_time:
                    self._next_request_time = now + self.request_interval_seconds
                    return
                wait_seconds = self._next_request_time - now
            time.sleep(wait_seconds)

    @staticmethod
    def _collect_text(value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            parts: List[str] = []
            for item in value:
                if isinstance(item, dict):
                    if item.get("type") in ("text", "output_text", "reasoning_text"):
                        parts.append(str(item.get("text", "")))
                    elif "content" in item:
                        parts.append(OpenAICompatibleAdapter._collect_text(item.get("content")))
                elif item is not None:
                    parts.append(str(item))
            return "\n".join([p for p in parts if str(p).strip()]).strip()
        if isinstance(value, dict):
            for key in ("text", "content", "reasoning_content"):
                if key in value:
                    txt = OpenAICompatibleAdapter._collect_text(value.get(key))
                    if txt:
                        return txt
        if value is None:
            return ""
        return str(value).strip()

    def chat(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        modalities: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        api_key = self.api_key or os.getenv(self.api_key_env, "")
        headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if modalities:
            payload["modalities"] = modalities
        if self.extra_payload:
            payload.update(self.extra_payload)
        if context:
            runtime_extra_payload = context.get("extra_payload")
            if isinstance(runtime_extra_payload, dict):
                runtime_chat_kwargs = runtime_extra_payload.get("chat_template_kwargs")
                base_chat_kwargs = payload.get("chat_template_kwargs")
                if isinstance(base_chat_kwargs, dict) and isinstance(runtime_chat_kwargs, dict):
                    merged_chat_kwargs = dict(base_chat_kwargs)
                    merged_chat_kwargs.update(runtime_chat_kwargs)
                    runtime_extra_payload = dict(runtime_extra_payload)
                    runtime_extra_payload["chat_template_kwargs"] = merged_chat_kwargs
                payload.update(runtime_extra_payload)

        try:
            client = self._get_client()
            self._wait_for_request_slot()
            resp = client.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
            status = resp.status_code
            body = (resp.text or "").strip()

            # LongCat multimodal may intermittently fail to fetch remote audio URLs.
            # Treat this as transient so engine-level retry can recover.
            if status == 400 and "getAudioData error" in body:
                raise TransientAdapterError(f"Transient HTTP {status}: {body}")

            if status in (408, 409, 425, 429) or status >= 500:
                raise TransientAdapterError(f"Transient HTTP {status}: {body}")

            if status >= 400:
                snippet = body[:2000] if body else "<empty body>"
                raise AdapterError(f"HTTP {status}: {snippet}")

            data = resp.json()
        except (httpx.TimeoutException, httpx.NetworkError) as e:
            raise TransientAdapterError(str(e)) from e
        except ValueError as e:
            raise AdapterError(f"Invalid JSON response: {e}") from e
        except httpx.HTTPError as e:
            raise AdapterError(str(e)) from e

        choices = data.get("choices", [])
        if not choices:
            raise AdapterError("No choices in model response")
        first_choice = choices[0] if isinstance(choices[0], dict) else {}
        msg = first_choice.get("message", {}) if isinstance(first_choice, dict) else {}

        content_text = self._collect_text(msg.get("content", ""))

        reasoning_text = ""
        for key in ("reasoning_content", "reasoning"):
            if key in msg:
                reasoning_text = self._collect_text(msg.get(key))
                if reasoning_text:
                    break
        if not reasoning_text:
            for key in ("reasoning_content", "reasoning"):
                if key in first_choice:
                    reasoning_text = self._collect_text(first_choice.get(key))
                    if reasoning_text:
                        break

        if reasoning_text and content_text:
            return f"<think>{reasoning_text}</think>\n{content_text}".strip()
        if reasoning_text:
            return f"<think>{reasoning_text}</think>"
        return content_text.strip()
