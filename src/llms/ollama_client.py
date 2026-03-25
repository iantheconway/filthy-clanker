import asyncio
import json
import uuid
from typing import Any

import requests

from .base import BaseLLMClient

_DEFAULT_HOST = "http://10.0.2.2:11434"


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

    async def generate_response(
        self,
        messages: list[dict],
        tools: list[dict],
        system_prompt: str,
    ) -> dict[str, Any]:
        ollama_tools = self.format_tools(tools)
        ollama_messages = [{"role": "system", "content": system_prompt}] + list(messages)

        payload = {
            "model": self.model,
            "messages": ollama_messages,
            "tools": ollama_tools,
            "stream": False,
        }

        def _call():
            resp = requests.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=300,
            )
            resp.raise_for_status()
            return resp.json()

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

        return {
            "text": text,
            "tool_calls": tool_calls,
            "raw": data,
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
