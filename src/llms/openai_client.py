"""
Client for any OpenAI-compatible chat/completions endpoint — Together, DeepInfra, Fireworks,
OpenRouter, a self-hosted vLLM, etc. Lets the harness drive strong CLOUD-OPEN models
(Qwen3-235B, DeepSeek-V3, Llama-405B) whose licenses permit training on their outputs.

Config (per-agent or via env):
    base_url      OPENAI_BASE_URL   e.g. https://api.together.xyz/v1
    api_key       OPENAI_API_KEY    (or set OPENAI_API_KEY_ENV to point at another var,
                                     e.g. TOGETHER_API_KEY)
    model         the provider's model id, e.g. Qwen/Qwen3-235B-A22B-Instruct-2507

Uses plain requests (no extra dep) against POST {base_url}/chat/completions. Reports token
usage to llms.cost for the budget guard.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any

import requests

from .base import BaseLLMClient

logger = logging.getLogger("filthy_clanker")

_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "3"))
_READ_TIMEOUT = float(os.getenv("OPENAI_READ_TIMEOUT", "300"))


class OpenAIClient(BaseLLMClient):
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None):
        self.model = model
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")

    @staticmethod
    def format_tools(mcp_tools: list[dict]) -> list[dict]:
        return [{
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("inputSchema", {"type": "object", "properties": {}}),
            },
        } for t in mcp_tools]

    async def generate_response(self, messages: list[dict], tools: list[dict],
                                system_prompt: str) -> dict[str, Any]:
        oai_tools = self.format_tools(tools)
        payload: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}] + list(messages),
            "temperature": 0.7,
            "max_tokens": 8000,
        }
        if oai_tools:
            payload["tools"] = oai_tools
        logger.info("[OpenAI] REQUEST model=%s msgs=%d tools=%d base=%s",
                    self.model, len(messages), len(oai_tools), self.base_url)

        def _call() -> dict:
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            last: Exception | None = None
            for attempt in range(1, _MAX_RETRIES + 1):
                try:
                    r = requests.post(f"{self.base_url}/chat/completions", json=payload,
                                      headers=headers, timeout=(10, _READ_TIMEOUT))
                    if r.status_code >= 400:
                        body = r.text[:600].replace("\n", " ")
                        logger.error("[OpenAI] HTTP %d (attempt %d/%d) model=%s body=%.600s",
                                     r.status_code, attempt, _MAX_RETRIES, self.model, body)
                        if 500 <= r.status_code < 600 and attempt < _MAX_RETRIES:
                            time.sleep(min(8.0, 0.5 * 2 ** (attempt - 1)))
                            last = requests.HTTPError(f"{r.status_code}")
                            continue
                        r.raise_for_status()
                    return r.json()
                except (requests.Timeout, requests.ConnectionError) as exc:
                    last = exc
                    logger.error("[OpenAI] %s (attempt %d/%d)", type(exc).__name__, attempt, _MAX_RETRIES)
                    if attempt < _MAX_RETRIES:
                        time.sleep(min(8.0, 0.5 * 2 ** (attempt - 1)))
                        continue
                    raise
            if last:
                raise last
            raise RuntimeError("OpenAI call failed")

        data = await asyncio.get_event_loop().run_in_executor(None, _call)

        msg = ((data.get("choices") or [{}])[0]).get("message", {}) or {}
        text = msg.get("content") or None
        tool_calls = []
        for tc in (msg.get("tool_calls") or []):
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            tool_calls.append({"id": tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                               "name": fn.get("name", ""), "arguments": args})

        usage_raw = data.get("usage", {}) or {}
        usage = {"input_tokens": usage_raw.get("prompt_tokens", 0),
                 "output_tokens": usage_raw.get("completion_tokens", 0)}
        from .cost import add_usage as _add_cost
        _add_cost(self.model, usage)
        logger.info("[OpenAI] RESPONSE tool_calls=%s in=%s out=%s text=%.160s",
                    [t["name"] for t in tool_calls], usage["input_tokens"], usage["output_tokens"],
                    (text or "").replace("\n", " "))
        return {"text": text, "tool_calls": tool_calls, "raw": data, "usage": usage}

    def parse_tool_calls(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        return response.get("tool_calls", [])

    @staticmethod
    def make_assistant_message(response: dict[str, Any]) -> dict:
        content = response.get("text") or ""
        msg: dict[str, Any] = {"role": "assistant", "content": content}
        tcs = response.get("tool_calls", [])
        if tcs:
            msg["tool_calls"] = [{
                "id": tc["id"], "type": "function",
                "function": {"name": tc["name"],
                             "arguments": tc["arguments"] if isinstance(tc["arguments"], str)
                             else json.dumps(tc["arguments"])},
            } for tc in tcs]
        return msg

    @staticmethod
    def make_tool_result_message(tool_call_id: str, result: str) -> dict:
        """OpenAI tool role message — MUST carry the originating tool_call_id."""
        return {"role": "tool", "tool_call_id": tool_call_id, "content": result}
