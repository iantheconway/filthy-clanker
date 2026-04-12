import json
import logging
from typing import Any

import anthropic

from .base import BaseLLMClient

logger = logging.getLogger("filthy_clanker")


class AnthropicClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str = "claude-opus-4-6"):
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

        last_content = ""
        if messages:
            raw = messages[-1].get("content", "")
            last_content = (raw if isinstance(raw, str) else json.dumps(raw))[:300]

        logger.info("[Anthropic] REQUEST  model=%s  msgs=%d  tools=%d  system=%.120s",
                    self.model, len(messages), len(formatted_tools),
                    system_prompt.replace("\n", " "))
        logger.info("[Anthropic] LAST_MSG %.300s", last_content.replace("\n", " "))

        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=16000,
                system=system_prompt,
                messages=messages,
                tools=formatted_tools,
                thinking={
                    "type": "adaptive",
                },
            )
        except anthropic.APITimeoutError as exc:
            logger.error("[Anthropic] TIMEOUT after %s", exc)
            raise
        except anthropic.APIError as exc:
            logger.error("[Anthropic] API ERROR %s: %s", type(exc).__name__, exc)
            raise

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

        text = "\n".join(text_parts) if text_parts else None
        tc_names = [tc["name"] for tc in tool_calls]
        logger.info("[Anthropic] RESPONSE  stop=%s  tool_calls=%s  text=%.200s",
                    response.stop_reason, tc_names,
                    (text or "").replace("\n", " "))

        return {
            "text": text,
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
            if block.type == "thinking":
                content.append({
                    "type": "thinking",
                    "thinking": block.thinking,
                    "signature": block.signature,
                })
            elif block.type == "text":
                content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
        return {"role": "assistant", "content": content}
