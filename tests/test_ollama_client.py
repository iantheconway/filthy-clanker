"""Tests for OllamaClient static/pure methods (no API calls required)."""
import pytest

from llms.ollama_client import OllamaClient


class TestOllamaFormatTools:
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
        result = OllamaClient.format_tools(mcp_tools)
        assert len(result) == 1
        tool = result[0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "nmap_scan"
        assert tool["function"]["description"] == "Run nmap"
        assert tool["function"]["parameters"]["type"] == "object"

    def test_missing_input_schema_gets_default(self):
        mcp_tools = [{"name": "tool", "description": "d"}]
        result = OllamaClient.format_tools(mcp_tools)
        assert result[0]["function"]["parameters"] == {"type": "object", "properties": {}}

    def test_empty_tools_list(self):
        assert OllamaClient.format_tools([]) == []


class TestOllamaMakeToolResultMessage:
    def test_structure(self):
        msg = OllamaClient.make_tool_result_message("nmap_scan", "scan output")
        assert msg["role"] == "tool"
        assert msg["content"] == "scan output"


class TestOllamaMakeAssistantMessage:
    def test_text_only(self):
        response = {"text": "Here are my findings", "tool_calls": []}
        msg = OllamaClient.make_assistant_message(response)
        assert msg["role"] == "assistant"
        assert msg["content"] == "Here are my findings"
        assert "tool_calls" not in msg

    def test_tool_calls(self):
        response = {
            "text": "",
            "tool_calls": [{"id": "x", "name": "nmap", "arguments": {"target": "10.0.0.1"}}],
        }
        msg = OllamaClient.make_assistant_message(response)
        assert msg["role"] == "assistant"
        assert "tool_calls" in msg
        assert msg["tool_calls"][0]["function"]["name"] == "nmap"
        assert msg["tool_calls"][0]["function"]["arguments"] == {"target": "10.0.0.1"}

    def test_none_text_becomes_empty_string(self):
        response = {"text": None, "tool_calls": []}
        msg = OllamaClient.make_assistant_message(response)
        assert msg["content"] == ""

    def test_mixed_text_and_tools(self):
        response = {
            "text": "Running",
            "tool_calls": [{"id": "1", "name": "gobuster", "arguments": {}}],
        }
        msg = OllamaClient.make_assistant_message(response)
        assert msg["content"] == "Running"
        assert len(msg["tool_calls"]) == 1


class TestOllamaParseToolCalls:
    def test_returns_tool_calls(self):
        client = OllamaClient.__new__(OllamaClient)
        response = {"tool_calls": [{"id": "1", "name": "t", "arguments": {}}]}
        assert client.parse_tool_calls(response) == response["tool_calls"]

    def test_empty_when_missing(self):
        client = OllamaClient.__new__(OllamaClient)
        assert client.parse_tool_calls({}) == []
