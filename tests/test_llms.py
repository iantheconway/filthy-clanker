"""Tests for LLM client static/pure methods (no API calls required)."""
import pytest

from llms.anthropic_client import AnthropicClient
from llms.gemini_client import GeminiClient


# ---------------------------------------------------------------------------
# AnthropicClient
# ---------------------------------------------------------------------------

class TestAnthropicFormatTools:
    def test_basic_tool(self):
        mcp_tools = [{
            "name": "nmap_scan",
            "description": "Run nmap",
            "inputSchema": {
                "type": "object",
                "properties": {"target": {"type": "string"}},
                "required": ["target"],
            },
        }]
        result = AnthropicClient.format_tools(mcp_tools)
        assert len(result) == 1
        assert result[0]["name"] == "nmap_scan"
        assert result[0]["description"] == "Run nmap"
        assert result[0]["input_schema"]["type"] == "object"

    def test_missing_input_schema_gets_default(self):
        mcp_tools = [{"name": "tool_no_schema", "description": "desc"}]
        result = AnthropicClient.format_tools(mcp_tools)
        assert result[0]["input_schema"] == {"type": "object", "properties": {}}

    def test_empty_tools_list(self):
        assert AnthropicClient.format_tools([]) == []

    def test_multiple_tools(self):
        mcp_tools = [
            {"name": "tool_a", "description": "a"},
            {"name": "tool_b", "description": "b"},
        ]
        result = AnthropicClient.format_tools(mcp_tools)
        assert len(result) == 2
        assert result[0]["name"] == "tool_a"
        assert result[1]["name"] == "tool_b"


class TestAnthropicMakeToolResultMessage:
    def test_structure(self):
        msg = AnthropicClient.make_tool_result_message("call_123", "scan output")
        assert msg["type"] == "tool_result"
        assert msg["tool_use_id"] == "call_123"
        assert msg["content"] == "scan output"


class TestAnthropicMakeAssistantMessage:
    def _make_response(self, blocks):
        """Build a minimal response dict with a fake raw object."""
        class FakeBlock:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class FakeRaw:
            def __init__(self, blocks):
                self.content = [FakeBlock(**b) for b in blocks]

        return {"text": None, "tool_calls": [], "raw": FakeRaw(blocks)}

    def test_text_block(self):
        response = self._make_response([{"type": "text", "text": "hello"}])
        msg = AnthropicClient.make_assistant_message(response)
        assert msg["role"] == "assistant"
        assert {"type": "text", "text": "hello"} in msg["content"]

    def test_tool_use_block(self):
        response = self._make_response([{
            "type": "tool_use",
            "id": "tu_1",
            "name": "nmap",
            "input": {"target": "10.0.0.1"},
        }])
        msg = AnthropicClient.make_assistant_message(response)
        assert msg["role"] == "assistant"
        block = msg["content"][0]
        assert block["type"] == "tool_use"
        assert block["id"] == "tu_1"
        assert block["name"] == "nmap"
        assert block["input"] == {"target": "10.0.0.1"}

    def test_thinking_block(self):
        response = self._make_response([{
            "type": "thinking",
            "thinking": "I need to scan",
            "signature": "sig_abc",
        }])
        msg = AnthropicClient.make_assistant_message(response)
        block = msg["content"][0]
        assert block["type"] == "thinking"
        assert block["thinking"] == "I need to scan"
        assert block["signature"] == "sig_abc"

    def test_mixed_blocks(self):
        response = self._make_response([
            {"type": "text", "text": "Running scan"},
            {"type": "tool_use", "id": "tu_2", "name": "nmap", "input": {}},
        ])
        msg = AnthropicClient.make_assistant_message(response)
        assert len(msg["content"]) == 2


class TestAnthropicParseToolCalls:
    def test_returns_tool_calls(self):
        client = AnthropicClient.__new__(AnthropicClient)
        response = {"tool_calls": [{"id": "1", "name": "tool", "arguments": {}}]}
        assert client.parse_tool_calls(response) == response["tool_calls"]

    def test_empty_when_no_key(self):
        client = AnthropicClient.__new__(AnthropicClient)
        assert client.parse_tool_calls({}) == []


# ---------------------------------------------------------------------------
# GeminiClient
# ---------------------------------------------------------------------------

class TestGeminiConvertSchema:
    def test_strips_unsupported_keys(self):
        schema = {
            "type": "object",
            "$schema": "http://json-schema.org/draft-07/schema",
            "additionalProperties": False,
            "properties": {"x": {"type": "string"}},
        }
        result = GeminiClient._convert_schema(schema)
        assert "$schema" not in result
        assert "additionalProperties" not in result
        assert result["type"] == "object"

    def test_recursive_properties(self):
        schema = {
            "type": "object",
            "properties": {
                "nested": {
                    "type": "object",
                    "$schema": "should-be-removed",
                    "properties": {},
                }
            },
        }
        result = GeminiClient._convert_schema(schema)
        assert "$schema" not in result["properties"]["nested"]

    def test_recursive_items(self):
        schema = {
            "type": "array",
            "items": {"type": "string", "additionalProperties": True},
        }
        result = GeminiClient._convert_schema(schema)
        assert "additionalProperties" not in result["items"]

    def test_items_not_dict_is_unchanged(self):
        schema = {"type": "array", "items": "string"}
        result = GeminiClient._convert_schema(schema)
        assert result["items"] == "string"

    def test_passthrough_with_no_unsupported_keys(self):
        schema = {"type": "string", "description": "A value"}
        result = GeminiClient._convert_schema(schema)
        assert result == schema


class TestGeminiMakeToolResultMessage:
    def test_structure(self):
        msg = GeminiClient.make_tool_result_message("nmap_scan", "scan results")
        assert msg["role"] == "user"
        assert len(msg["content"]) == 1
        block = msg["content"][0]
        assert "function_response" in block
        assert block["function_response"]["name"] == "nmap_scan"
        assert block["function_response"]["response"] == {"result": "scan results"}


class TestGeminiMakeAssistantMessage:
    def test_text_only(self):
        response = {"text": "Hello", "tool_calls": []}
        msg = GeminiClient.make_assistant_message(response)
        assert msg["role"] == "assistant"
        assert {"text": "Hello"} in msg["content"]

    def test_tool_calls_only(self):
        response = {
            "text": None,
            "tool_calls": [{"name": "nmap", "id": "nmap", "arguments": {"target": "10.0.0.1"}}],
        }
        msg = GeminiClient.make_assistant_message(response)
        assert msg["role"] == "assistant"
        block = msg["content"][0]
        assert "function_call" in block
        assert block["function_call"]["name"] == "nmap"
        assert block["function_call"]["args"] == {"target": "10.0.0.1"}

    def test_text_and_tool_calls(self):
        response = {
            "text": "Running",
            "tool_calls": [{"name": "gobuster", "id": "gobuster", "arguments": {}}],
        }
        msg = GeminiClient.make_assistant_message(response)
        assert len(msg["content"]) == 2

    def test_empty_response(self):
        response = {"text": None, "tool_calls": []}
        msg = GeminiClient.make_assistant_message(response)
        assert msg["role"] == "assistant"
        assert msg["content"] == []


class TestGeminiParseToolCalls:
    def test_returns_tool_calls(self):
        client = GeminiClient.__new__(GeminiClient)
        response = {"tool_calls": [{"id": "x", "name": "tool", "arguments": {}}]}
        assert client.parse_tool_calls(response) == response["tool_calls"]

    def test_empty_when_missing(self):
        client = GeminiClient.__new__(GeminiClient)
        assert client.parse_tool_calls({}) == []
