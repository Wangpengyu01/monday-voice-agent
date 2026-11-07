"""Teacher model client: DeepSeek chat completions (OpenAI-compatible), streamed, thinking off for voice.

The API key is read from an environment variable (default DEEPSEEK_API_KEY), which the orchestrator
service loads from /userdata/secrets/deepseek.env - the key never lives in the repo or the config.
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable

import time

import requests

from learning import usage

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

    def complete(self, system: str, user: str, *, json_mode: bool = False, max_tokens: int | None = None,
                 timeout_s: float | None = None) -> str:
        """Blocking, non-streamed (offline jobs: grading, quizzes, the diary). json_mode asks for a JSON object."""
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise TeacherUnavailable(f"{self.api_key_env} is not set")
        if usage.peak_blocked():
            raise TeacherUnavailable(f"peak hours: {usage.job_tag()} may not call the teacher now")
        body: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens or self.max_tokens,
            "thinking": {"type": "disabled"},
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        for attempt in range(3):   # 429 (the account allows only a few calls at once) / 5xx: wait and retry
            try:
                r = self._session.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    data=payload,
                    timeout=(5, timeout_s or max(self.timeout_s, 60.0)),
                )
            except requests.RequestException as exc:
                raise TeacherUnavailable(f"request failed: {exc}") from exc
            if r.status_code in (429, 500, 502, 503) and attempt < 2:
                LOG.info("[teacher] HTTP %s, retry in %ds", r.status_code, 3 * (attempt + 1) ** 2)
                time.sleep(3 * (attempt + 1) ** 2)
                continue
            break
        if r.status_code != 200:
            raise TeacherUnavailable(f"HTTP {r.status_code}: {r.text[:200]}")
        try:
            data = r.json()
            if not isinstance(data, dict):   # seen 2026-11-07: HTTP 200 with a body of plain `null`
                raise ValueError(f"not a JSON object: {r.text[:80]!r}")
            usage.record(data.get("usage"), self.model)
            return data["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise TeacherUnavailable(f"bad response: {exc}") from exc

    def stream(self, system: str, user: str, on_token: Callable[[str], None] | None = None) -> str:
        """Blocking; call from a worker thread. Returns the full reply text."""
        key = os.environ.get(self.api_key_env, "").strip()
        if not key:
            raise TeacherUnavailable(f"{self.api_key_env} is not set")
        if usage.peak_blocked():
            raise TeacherUnavailable(f"peak hours: {usage.job_tag()} may not call the teacher now")
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": True,
            "stream_options": {"include_usage": True},   # the last chunk carries the token counts
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
                    chunk = json.loads(data)
                    if chunk.get("usage"):
                        usage.record(chunk["usage"], self.model)
                    if not chunk.get("choices"):
                        continue
                    delta = chunk["choices"][0].get("delta") or {}
                except (ValueError, KeyError, IndexError, AttributeError):
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
