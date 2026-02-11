import json
from typing import Any

from google import genai
from google.genai import types

from .base import BaseLLMClient


class GeminiClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self.client = genai.Client(api_key=api_key)
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

    async def generate_response(
        self,
        messages: list[dict],
        tools: list[dict],
        system_prompt: str,
    ) -> dict[str, Any]:
        gemini_tools = self.format_tools(tools)

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

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                tools=gemini_tools,
                system_instruction=system_prompt,
            ),
        )

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

        return {
            "text": "\n".join(text_parts) if text_parts else None,
            "tool_calls": tool_calls,
            "raw": response,
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
