import json
import logging
import os
from typing import Any

from google import genai
from google.genai import types
from langsmith import traceable
import langsmith

from .base import BaseLLMClient

logger = logging.getLogger("filthy_clanker")


class GeminiClient(BaseLLMClient):
    def __init__(self, api_key: str | None = None, model: str = "gemini-2.5-flash"):
        # api_key falls back to the GEMINI_API_KEY environment variable.
        self.client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY"))
        self.model = model

    @staticmethod
    def _convert_schema(input_schema: dict) -> dict:
        """Convert an MCP JSON-Schema to a Gemini-compatible schema dict.

        Strips unsupported keys like '$schema' and 'additionalProperties'
        and recursively cleans nested property schemas.
        """
        UNSUPPORTED = {"$schema", "additionalProperties"}
        cleaned = {k: v for k, v in input_schema.items() if k not in UNSUPPORTED}

        if "properties" in cleaned:
            cleaned["properties"] = {
                name: GeminiClient._convert_schema(prop)
                for name, prop in cleaned["properties"].items()
            }
        if "items" in cleaned and isinstance(cleaned["items"], dict):
            cleaned["items"] = GeminiClient._convert_schema(cleaned["items"])

        return cleaned

    @staticmethod
    def format_tools(mcp_tools: list[dict]) -> list[types.Tool]:
        """Convert MCP tool schemas to Gemini FunctionDeclarations."""
        declarations = []
        for tool in mcp_tools:
            schema = tool.get("inputSchema", {"type": "object", "properties": {}})
            cleaned = GeminiClient._convert_schema(schema)
            declarations.append(types.FunctionDeclaration(
                name=tool["name"],
                description=tool.get("description", ""),
                parameters=cleaned,
            ))
        return [types.Tool(function_declarations=declarations)]

    @traceable(run_type="llm", name="Gemini")
    async def generate_response(
        self,
        messages: list[dict],
        tools: list[dict],
        system_prompt: str,
    ) -> dict[str, Any]:
        rt = langsmith.get_current_run_tree()
        if rt:
            rt.name = f"Gemini / {self.model}"

        gemini_tools = self.format_tools(tools)

        last_content = ""
        if messages:
            raw = messages[-1].get("content", "")
            last_content = (raw if isinstance(raw, str) else json.dumps(raw))[:300]

        logger.info("[Gemini] REQUEST  model=%s  msgs=%d  system=%.120s",
                    self.model, len(messages), system_prompt.replace("\n", " "))
        logger.info("[Gemini] LAST_MSG %.300s", last_content.replace("\n", " "))

        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            if isinstance(msg.get("content"), str):
                contents.append(types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=msg["content"])],
                ))
            elif isinstance(msg.get("content"), list):
                parts = []
                for part in msg["content"]:
                    if isinstance(part, str):
                        parts.append(types.Part.from_text(text=part))
                    elif isinstance(part, dict):
                        if "text" in part:
                            parts.append(types.Part.from_text(text=part["text"]))
                        elif "function_call" in part:
                            parts.append(types.Part.from_function_call(
                                name=part["function_call"]["name"],
                                args=part["function_call"]["args"],
                            ))
                        elif "function_response" in part:
                            parts.append(types.Part.from_function_response(
                                name=part["function_response"]["name"],
                                response=part["function_response"]["response"],
                            ))
                if parts:
                    contents.append(types.Content(role=role, parts=parts))

        try:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(
                    tools=gemini_tools,
                    system_instruction=system_prompt,
                ),
            )
        except Exception as exc:
            logger.error("[Gemini] ERROR %s: %s", type(exc).__name__, exc)
            raise

        text_parts = []
        tool_calls = []

        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if part.text:
                    text_parts.append(part.text)
                elif part.function_call:
                    fc = part.function_call
                    tool_calls.append({
                        "id": fc.name,
                        "name": fc.name,
                        "arguments": dict(fc.args) if fc.args else {},
                    })

        text = "\n".join(text_parts) if text_parts else None
        tc_names = [tc["name"] for tc in tool_calls]
        logger.info("[Gemini] RESPONSE  tool_calls=%s  text=%.200s",
                    tc_names, (text or "").replace("\n", " "))

        usage = getattr(response, "usage_metadata", None)
        return {
            "text": text,
            "tool_calls": tool_calls,
            "raw": response,
            "usage": {
                "input_tokens": getattr(usage, "prompt_token_count", 0),
                "output_tokens": getattr(usage, "candidates_token_count", 0),
            } if usage else {},
        }

    def parse_tool_calls(self, response: dict[str, Any]) -> list[dict[str, Any]]:
        return response.get("tool_calls", [])

    @staticmethod
    def make_assistant_message(response: dict[str, Any]) -> dict:
        """Build a conversation-history message from the model response."""
        content = []
        if response["text"]:
            content.append({"text": response["text"]})
        for tc in response["tool_calls"]:
            content.append({
                "function_call": {"name": tc["name"], "args": tc["arguments"]},
            })
        return {"role": "assistant", "content": content}

    @staticmethod
    def make_tool_result_message(tool_name: str, result: str) -> dict:
        """Build a tool-response message for Gemini conversation history."""
        return {
            "role": "user",
            "content": [{
                "function_response": {
                    "name": tool_name,
                    "response": {"result": result},
                },
            }],
        }
