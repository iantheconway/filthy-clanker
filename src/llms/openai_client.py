"""
Client for any OpenAI-compatible chat/completions endpoint — Together, DeepInfra, Fireworks,
OpenRouter, a self-hosted vLLM, etc. Lets the harness drive strong CLOUD-OPEN models
(gpt-oss-120b, Qwen3, DeepSeek, Llama) whose licenses permit training on their outputs.

Config (per-agent or via env):
    base_url      OPENAI_BASE_URL   e.g. https://api.together.xyz/v1
    api_key       OPENAI_API_KEY    (or set OPENAI_API_KEY_ENV to point at another var,
                                     e.g. TOGETHER_API_KEY)
    model         the provider's model id, e.g. openai/gpt-oss-120b

Tuning (env):
    OPENAI_REASONING_EFFORT   low|medium|high — sent to reasoning models (gpt-oss); unset = omit
    OPENAI_MAX_TOKENS         completion cap (default 8000; raise for long chains of thought)
    OPENAI_READ_TIMEOUT       max SECONDS BETWEEN streamed chunks (default 180) — NOT total
                              generation time, so a long "thinking" response never trips it
    OPENAI_STREAM             1 (default) stream via SSE; 0 = single blocking response
    OPENAI_MAX_RETRIES        retries on transport error / 5xx (default 3)

Streams via ``requests`` (stream=True, iterated in an executor) — the proven path through the
eval sandbox's tinyproxy CONNECT tunnel (httpx 0.28's async proxy+stream hangs there). Streaming
keeps bytes flowing so a slow high-effort reasoning turn never trips the proxy's inactivity
timeout or the client read timeout. Captures the model's chain-of-thought (``delta.reasoning``)
onto the assistant message as a ``reasoning`` field — kept for SFT, stripped before the next
upstream request. Reports token usage to llms.cost for the budget guard.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
import uuid
from typing import Any

import requests

from .base import BaseLLMClient

logger = logging.getLogger("filthy_clanker")

_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "5"))
# Read timeout = max gap BETWEEN streamed chunks, not total duration. A long reasoning
# response streams tokens the whole time, so this never trips; only a genuine stall does.
_READ_TIMEOUT = float(os.getenv("OPENAI_READ_TIMEOUT", "180"))
_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "8000"))
_REASONING_EFFORT = os.getenv("OPENAI_REASONING_EFFORT") or None  # e.g. "high" for gpt-oss
_STREAM = os.getenv("OPENAI_STREAM", "1") != "0"

# Keys an OpenAI-compatible endpoint accepts on an input message. Anything else we attach for
# our own bookkeeping (notably captured ``reasoning``) is stripped before the request.
_MSG_KEYS = ("role", "content", "tool_calls", "tool_call_id", "name")


def _backoff(attempt: int) -> float:
    """Patient jittered backoff — rides out transient 429/503 capacity spikes on serverless
    endpoints (gpt-oss on Together can 503 for a minute+ under load) rather than failing the turn."""
    return min(30.0, 2.0 * 2 ** (attempt - 1)) + random.uniform(0, 2.0)


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

    @staticmethod
    def _sanitize_messages(messages: list[dict]) -> list[dict]:
        """Keep only OpenAI-accepted keys — drops our captured ``reasoning`` (and any other
        bookkeeping) so it never gets echoed back upstream."""
        out: list[dict] = []
        for m in messages:
            if isinstance(m, dict):
                out.append({k: v for k, v in m.items() if k in _MSG_KEYS})
            else:
                out.append(m)
        return out

    async def generate_response(self, messages: list[dict], tools: list[dict],
                                system_prompt: str) -> dict[str, Any]:
        oai_tools = self.format_tools(tools)
        payload: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}] + self._sanitize_messages(messages),
            "temperature": 0.7,
            "max_tokens": _MAX_TOKENS,
        }
        if _STREAM:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        if _REASONING_EFFORT:
            payload["reasoning_effort"] = _REASONING_EFFORT
        if oai_tools:
            payload["tools"] = oai_tools
        logger.info("[OpenAI] REQUEST model=%s msgs=%d tools=%d stream=%d effort=%s",
                    self.model, len(messages), len(oai_tools), int(_STREAM), _REASONING_EFFORT or "-")

        content, reasoning, tool_calls, usage = await asyncio.get_event_loop().run_in_executor(
            None, self._call, payload)

        usage_norm = {"input_tokens": int(usage.get("prompt_tokens", 0) or 0),
                      "output_tokens": int(usage.get("completion_tokens", 0) or 0)}
        from .cost import add_usage as _add_cost
        _add_cost(self.model, usage_norm)
        logger.info("[OpenAI] RESPONSE tool_calls=%s in=%s out=%s reasoning=%dch text=%.140s",
                    [t["name"] for t in tool_calls], usage_norm["input_tokens"],
                    usage_norm["output_tokens"], len(reasoning or ""),
                    (content or "").replace("\n", " "))
        return {"text": content or None, "reasoning": reasoning or None,
                "tool_calls": tool_calls, "raw": {"usage": usage}, "usage": usage_norm}

    def _call(self, payload: dict) -> tuple[str, str, list[dict], dict]:
        """Blocking request (run in an executor). Streams SSE when payload['stream'], else one JSON."""
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        streaming = bool(payload.get("stream"))
        last: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                r = requests.post(url, json=payload, headers=headers, stream=streaming,
                                  timeout=(15, _READ_TIMEOUT))
                if r.status_code >= 400:
                    body = r.text[:400].replace("\n", " ")
                    if r.status_code in (429, 500, 502, 503, 504) and attempt < _MAX_RETRIES:
                        back = _backoff(attempt)
                        logger.warning("[OpenAI] HTTP %d (transient) — retry %d/%d after %.1fs",
                                       r.status_code, attempt, _MAX_RETRIES, back)
                        time.sleep(back)
                        last = requests.HTTPError(f"{r.status_code}")
                        continue
                    logger.error("[OpenAI] HTTP %d (final, attempt %d/%d) model=%s body=%.400s",
                                 r.status_code, attempt, _MAX_RETRIES, self.model, body)
                    r.raise_for_status()
                return self._parse_stream(r) if streaming else self._parse_json(r.json())
            except (requests.Timeout, requests.ConnectionError) as exc:
                last = exc
                logger.error("[OpenAI] %s (attempt %d/%d)", type(exc).__name__, attempt, _MAX_RETRIES)
                if attempt < _MAX_RETRIES:
                    time.sleep(_backoff(attempt))
                    continue
                raise
        if last:
            raise last
        raise RuntimeError("OpenAI call failed")

    @staticmethod
    def _accumulate_tool_frag(frags: dict, tc: dict) -> None:
        idx = tc.get("index", 0)
        slot = frags.setdefault(idx, {"id": "", "name": "", "args": ""})
        if tc.get("id"):
            slot["id"] = tc["id"]
        fn = tc.get("function") or {}
        if fn.get("name"):
            slot["name"] = fn["name"]
        if fn.get("arguments"):
            slot["args"] += fn["arguments"]

    @staticmethod
    def _assemble_tool_calls(frags: dict) -> list[dict]:
        out: list[dict] = []
        for idx in sorted(frags):
            slot = frags[idx]
            if not slot["name"]:
                continue
            try:
                args = json.loads(slot["args"]) if slot["args"].strip() else {}
            except json.JSONDecodeError:
                args = {}
            out.append({"id": slot["id"] or f"call_{uuid.uuid4().hex[:8]}",
                        "name": slot["name"], "arguments": args})
        return out

    def _parse_stream(self, r: requests.Response) -> tuple[str, str, list[dict], dict]:
        content, reasoning = "", ""
        frags: dict[int, dict] = {}
        usage: dict = {}
        for line in r.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = (choices[0] or {}).get("delta", {}) or {}
            if delta.get("content"):
                content += delta["content"]
            rc = delta.get("reasoning") or delta.get("reasoning_content")
            if rc:
                reasoning += rc
            for tc in (delta.get("tool_calls") or []):
                self._accumulate_tool_frag(frags, tc)
        return content, reasoning, self._assemble_tool_calls(frags), usage

    def _parse_json(self, data: dict) -> tuple[str, str, list[dict], dict]:
        msg = ((data.get("choices") or [{}])[0]).get("message", {}) or {}
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
        frags: dict[int, dict] = {}
        for i, tc in enumerate(msg.get("tool_calls") or []):
            self._accumulate_tool_frag(frags, {**tc, "index": tc.get("index", i)})
        return content, reasoning, self._assemble_tool_calls(frags), (data.get("usage") or {})

    def parse_tool_calls(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        return response.get("tool_calls", [])

    @staticmethod
    def make_assistant_message(response: dict[str, Any]) -> dict:
        content = response.get("text") or ""
        msg: dict[str, Any] = {"role": "assistant", "content": content}
        reasoning = response.get("reasoning")
        if reasoning:
            # Captured for SFT (build_sft_dataset can fold it into the target); stripped
            # before the next upstream request by _sanitize_messages.
            msg["reasoning"] = reasoning
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
