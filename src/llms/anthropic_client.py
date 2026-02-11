import json
from typing import Any

import anthropic

from .base import BaseLLMClient


class AnthropicClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model

    @staticmethod
    def format_tools(mcp_tools: list[dict]) -> list[dict]:
        """Convert MCP tool schemas to Anthropic tool format."""
        formatted = []
        for tool in mcp_tools:
            formatted.append({
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool.get("inputSchema", {
                    "type": "object",
                    "properties": {},
                }),
            })
        return formatted

    async def generate_response(
        self,
        messages: list[dict],
        tools: list[dict],
        system_prompt: str,
    ) -> dict[str, Any]:
        formatted_tools = self.format_tools(tools)

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_prompt,
            messages=messages,
            tools=formatted_tools,
        )

        text_parts = []
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,
                })

        return {
            "text": "\n".join(text_parts) if text_parts else None,
            "tool_calls": tool_calls,
            "raw": response,
            "stop_reason": response.stop_reason,
        }

    def parse_tool_calls(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        return response.get("tool_calls", [])

    @staticmethod
    def make_tool_result_message(tool_call_id: str, result: str) -> dict:
        """Build the Anthropic tool_result content block."""
        return {
            "type": "tool_result",
            "tool_use_id": tool_call_id,
            "content": result,
        }

    @staticmethod
    def make_assistant_message(response: dict[str, Any]) -> dict:
        """Convert the raw response into an assistant message for the conversation."""
        content = []
        raw = response["raw"]
        for block in raw.content:
            if block.type == "text":
                content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
        return {"role": "assistant", "content": content}
