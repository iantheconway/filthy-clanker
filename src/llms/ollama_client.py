import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any

import requests
from langsmith import traceable
import langsmith

from .base import BaseLLMClient

logger = logging.getLogger("filthy_clanker")

_DEFAULT_HOST = "http://host.docker.internal:11434"

# Hardening knobs (env-overridable).
#   OLLAMA_MAX_RETRIES  — attempts per call before giving up (transient + one 4xx retry)
#   OLLAMA_NUM_CTX      — if set, passed as options.num_ctx so long conversations
#                         don't overflow the model's default window. NOT set by
#                         default (a large num_ctx grows KV-cache VRAM and can OOM).
#   OLLAMA_READ_TIMEOUT — seconds to wait for a generation before treating the
#                         call as hung. Without this a stalled /api/chat blocks
#                         forever (observed: a single hung call ate a whole 1800s
#                         challenge). Generous by default so slow-but-real
#                         generations aren't killed; a true hang is retried.
_MAX_RETRIES = int(os.getenv("OLLAMA_MAX_RETRIES", "3"))
_READ_TIMEOUT = float(os.getenv("OLLAMA_READ_TIMEOUT", "300"))


def _num_ctx_option() -> dict:
    raw = os.getenv("OLLAMA_NUM_CTX", "").strip()
    if not raw:
        return {}
    try:
        return {"num_ctx": int(raw)}
    except ValueError:
        return {}


def _backoff_seconds(attempt: int) -> float:
    """Exponential backoff capped at 8s: 0.5, 1, 2, 4, 8…"""
    return min(8.0, 0.5 * (2 ** (attempt - 1)))


def _sanitize_messages(messages: list[dict]) -> list[dict]:
    """Keep the conversation well-formed for Ollama's /api/chat.

    Ollama (and the underlying chat templates) reject two or more assistant
    messages in a row — "Cannot have 2 or more assistant messages at the end of
    the list" (HTTP 400). This happens in the multi-agent graph whenever agents
    return back-to-back tool-less "report" turns (the shared message history then
    holds consecutive assistant messages). Insert a minimal user turn between any
    two adjacent assistant messages so the history stays valid without losing
    either agent's output.
    """
    out: list[dict] = []
    for m in messages:
        if out and out[-1].get("role") == "assistant" and m.get("role") == "assistant":
            out.append({"role": "user", "content": "Continue."})
        out.append(m)
    return out


class OllamaClient(BaseLLMClient):
    def __init__(self, model: str, host: str = _DEFAULT_HOST):
        self.model = model
        self.host = host.rstrip("/")

    @staticmethod
    def format_tools(mcp_tools: list[dict]) -> list[dict]:
        """Convert MCP tool schemas to Ollama/OpenAI function-calling format."""
        tools = []
        for tool in mcp_tools:
            tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("inputSchema", {
                        "type": "object",
                        "properties": {},
                    }),
                },
            })
        return tools

    @traceable(run_type="llm", name="Ollama")
    async def generate_response(
        self,
        messages: list[dict],
        tools: list[dict],
        system_prompt: str,
    ) -> dict[str, Any]:
        rt = langsmith.get_current_run_tree()
        if rt:
            rt.name = f"Ollama / {self.model}"

        ollama_tools = self.format_tools(tools)

        last_content = ""
        if messages:
            raw = messages[-1].get("content", "")
            last_content = (raw if isinstance(raw, str) else json.dumps(raw))[:300]

        logger.info("[Ollama] REQUEST  model=%s  msgs=%d  tools=%d  system=%.120s",
                    self.model, len(messages), len(ollama_tools),
                    system_prompt.replace("\n", " "))
        logger.info("[Ollama] LAST_MSG %.300s", last_content.replace("\n", " "))

        options = _num_ctx_option()
        system_msg = {"role": "system", "content": system_prompt}

        def _payload(msgs: list[dict]) -> dict:
            p = {
                "model": self.model,
                # Sanitize so trimming/accumulation never yields consecutive
                # assistant messages (Ollama 400s on those).
                "messages": _sanitize_messages([system_msg] + list(msgs)),
                "tools": ollama_tools,
                "stream": False,
            }
            if options:
                p["options"] = options
            return p

        def _call():
            """POST /api/chat with retries.

            - Transient failures (connect timeout, connection error, HTTP 5xx) are
              retried with exponential backoff.
            - HTTP 4xx (e.g. Ollama's 400 on an over-long or malformed context) is
              logged WITH the response body, then retried once against a trimmed
              conversation (oldest non-system turns dropped) in case the window
              overflowed. Persistent failure re-raises for the caller to handle.
            """
            msgs = list(messages)
            last_exc: Exception | None = None
            for attempt in range(1, _MAX_RETRIES + 1):
                try:
                    resp = requests.post(
                        f"{self.host}/api/chat",
                        json=_payload(msgs),
                        timeout=(10, _READ_TIMEOUT),  # 10s connect, bounded read
                    )
                    if resp.status_code >= 400:
                        body = resp.text[:600].replace("\n", " ")
                        logger.error(
                            "[Ollama] HTTP %d (attempt %d/%d) model=%s body=%.600s",
                            resp.status_code, attempt, _MAX_RETRIES, self.model, body,
                        )
                        # 4xx is usually deterministic; the one thing worth retrying
                        # is an over-long context, so trim and try again.
                        if 400 <= resp.status_code < 500 and len(msgs) > 2 and attempt < _MAX_RETRIES:
                            keep = max(2, len(msgs) // 2)
                            logger.warning("[Ollama] Trimming history %d→%d and retrying after 4xx",
                                           len(msgs), keep)
                            msgs = msgs[-keep:]
                            last_exc = requests.HTTPError(f"{resp.status_code} for /api/chat")
                            time.sleep(_backoff_seconds(attempt))
                            continue
                        resp.raise_for_status()
                    return resp.json()
                except (requests.Timeout, requests.ConnectionError) as exc:
                    last_exc = exc
                    logger.error("[Ollama] %s (attempt %d/%d) model=%s host=%s",
                                 type(exc).__name__, attempt, _MAX_RETRIES, self.model, self.host)
                    if attempt < _MAX_RETRIES:
                        time.sleep(_backoff_seconds(attempt))
                        continue
                    raise
                except requests.HTTPError as exc:
                    # 5xx (raised above) — retry; final attempt re-raises.
                    last_exc = exc
                    logger.error("[Ollama] HTTP ERROR %s (attempt %d/%d) model=%s",
                                 exc, attempt, _MAX_RETRIES, self.model)
                    if attempt < _MAX_RETRIES:
                        time.sleep(_backoff_seconds(attempt))
                        continue
                    raise
            # Exhausted retries without returning.
            if last_exc:
                raise last_exc
            raise RuntimeError("Ollama call failed with no captured exception")

        data = await asyncio.get_event_loop().run_in_executor(None, _call)

        message = data.get("message", {})
        text = message.get("content") or None

        tool_calls = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            tool_calls.append({
                "id": str(uuid.uuid4()),
                "name": fn.get("name", ""),
                "arguments": args,
            })

        tc_names = [tc["name"] for tc in tool_calls]
        logger.info("[Ollama] RESPONSE  tool_calls=%s  text=%.200s",
                    tc_names, (text or "").replace("\n", " "))

        return {
            "text": text,
            "tool_calls": tool_calls,
            "raw": data,
            "usage": {
                "input_tokens": data.get("prompt_eval_count", 0),
                "output_tokens": data.get("eval_count", 0),
            },
        }

    def parse_tool_calls(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        return response.get("tool_calls", [])

    @staticmethod
    def make_assistant_message(response: dict[str, Any]) -> dict:
        """Build an assistant message for Ollama conversation history."""
        content = response.get("text") or ""
        tool_calls = response.get("tool_calls", [])
        msg: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = [
                {"function": {"name": tc["name"], "arguments": tc["arguments"]}}
                for tc in tool_calls
            ]
        return msg

    @staticmethod
    def make_tool_result_message(tool_name: str, result: str) -> dict:
        """Build a tool result message for Ollama conversation history."""
        return {"role": "tool", "content": result}
