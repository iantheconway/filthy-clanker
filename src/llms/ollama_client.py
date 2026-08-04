import asyncio
import json
import logging
import uuid
from typing import Any

import requests
from langsmith import traceable
import langsmith

from .base import BaseLLMClient

logger = logging.getLogger("filthy_clanker")

_DEFAULT_HOST = "http://host.docker.internal:11434"


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
        ollama_messages = [{"role": "system", "content": system_prompt}] + list(messages)

        last_content = ""
        if messages:
            raw = messages[-1].get("content", "")
            last_content = (raw if isinstance(raw, str) else json.dumps(raw))[:300]

        logger.info("[Ollama] REQUEST  model=%s  msgs=%d  tools=%d  system=%.120s",
                    self.model, len(messages), len(ollama_tools),
                    system_prompt.replace("\n", " "))
        logger.info("[Ollama] LAST_MSG %.300s", last_content.replace("\n", " "))

        payload = {
            "model": self.model,
            "messages": ollama_messages,
            "tools": ollama_tools,
            "stream": False,
        }

        def _call():
            try:
                resp = requests.post(
                    f"{self.host}/api/chat",
                    json=payload,
                    timeout=(10, None),  # 10s connect, no read timeout
                )
                resp.raise_for_status()
                return resp.json()
            except requests.Timeout:
                logger.error("[Ollama] CONNECT TIMEOUT (>10s)  model=%s  host=%s",
                             self.model, self.host)
                raise
            except requests.HTTPError as exc:
                logger.error("[Ollama] HTTP ERROR %s  model=%s", exc, self.model)
                raise
            except requests.RequestException as exc:
                logger.error("[Ollama] REQUEST ERROR %s: %s", type(exc).__name__, exc)
                raise

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
