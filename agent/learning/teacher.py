"""Teacher model client: DeepSeek chat completions (OpenAI-compatible), streamed, thinking off for voice.

The API key is read from an environment variable (default DEEPSEEK_API_KEY), which the orchestrator
service loads from /userdata/secrets/deepseek.env - the key never lives in the repo or the config.
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable

import requests

LOG = logging.getLogger(__name__)


class TeacherUnavailable(RuntimeError):
    """No key configured, network down, or the API refused the request."""


class DeepSeekTeacher:
    def __init__(self, *, base_url: str = "https://api.deepseek.com", model: str = "deepseek-flash",
                 api_key_env: str = "DEEPSEEK_API_KEY", timeout_s: float = 20.0, max_tokens: int = 400) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self._session = requests.Session()  # keep the TLS connection warm between turns

    @property
    def available(self) -> bool:
        return bool(os.environ.get(self.api_key_env, "").strip())

    def stream(self, system: str, user: str, on_token: Callable[[str], None] | None = None) -> str:
        """Blocking; call from a worker thread. Returns the full reply text."""
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise TeacherUnavailable(f"{self.api_key_env} is not set")
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": True,
            "max_tokens": self.max_tokens,
            "thinking": {"type": "disabled"},  # voice: answer right away
        }
        try:
            r = self._session.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                stream=True,
                timeout=(5, self.timeout_s),
            )
        except requests.RequestException as exc:
            raise TeacherUnavailable(f"request failed: {exc}") from exc
        if r.status_code != 200:
            raise TeacherUnavailable(f"HTTP {r.status_code}: {r.text[:200]}")
        text: list[str] = []
        try:
            for raw in r.iter_lines(decode_unicode=False):
                if not raw or not raw.startswith(b"data:"):
                    continue
                data = raw[5:].strip()
                if data == b"[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0].get("delta") or {}
                except (ValueError, KeyError, IndexError):
                    continue
                piece = delta.get("content") or ""
                if piece:
                    text.append(piece)
                    if on_token:
                        on_token(piece)
        except requests.RequestException as exc:
            if not text:
                raise TeacherUnavailable(f"stream failed: {exc}") from exc
            LOG.warning("[teacher] stream cut off: %s", exc)
        finally:
            r.close()
        return "".join(text)
