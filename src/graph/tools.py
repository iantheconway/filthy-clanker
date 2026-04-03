"""
LangChain-compatible tool wrappers for the HexstrikeMCPClient.

At graph initialization time, call `build_langchain_tools(mcp_client)` to get a
list of LangChain BaseTool instances, one per tool exposed by the MCP server.
These can be passed to any LangChain-compatible agent or used directly.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, create_model


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def _json_type_to_python(json_type: str, field_schema: dict) -> type:
    """Map a JSON Schema type string to a Python type."""
    if json_type == "array":
        return list
    if json_type == "object":
        return dict
    if json_type == "integer":
        return int
    if json_type == "number":
        return float
    if json_type == "boolean":
        return bool
    return str  # default / "string"


def _build_pydantic_model(tool_name: str, input_schema: dict) -> Type[BaseModel]:
    """
    Dynamically build a Pydantic v2 model from a JSON Schema dict.
    Used to give each LangChain tool a proper args_schema.
    """
    properties: dict = input_schema.get("properties", {})
    required: list = input_schema.get("required", [])

    fields: dict = {}
    for name, prop in properties.items():
        json_type = prop.get("type", "string")
        py_type = _json_type_to_python(json_type, prop)
        description = prop.get("description", "")

        if name in required:
            fields[name] = (py_type, Field(description=description))
        else:
            fields[name] = (Optional[py_type], Field(default=None, description=description))

    # If no properties defined, accept a single free-form dict
    if not fields:
        fields["kwargs"] = (Optional[Dict[str, Any]], Field(default=None, description="Tool arguments"))

    model_name = f"{tool_name.replace('-', '_').title()}Args"
    return create_model(model_name, **fields)


# ---------------------------------------------------------------------------
# Dynamic MCP tool wrapper
# ---------------------------------------------------------------------------

class MCPTool(BaseTool):
    """
    A LangChain BaseTool that delegates execution to the HexstrikeMCPClient.
    One instance is created per tool discovered from the MCP server.
    """
    name: str
    description: str
    args_schema: Type[BaseModel]
    mcp_tool_name: str
    _mcp_client: Any  # HexstrikeMCPClient — stored as private attribute

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, mcp_client: Any, tool_name: str, tool_description: str,
                 input_schema: dict, **kwargs):
        args_schema = _build_pydantic_model(tool_name, input_schema)
        super().__init__(
            name=tool_name,
            description=tool_description,
            args_schema=args_schema,
            mcp_tool_name=tool_name,
            **kwargs,
        )
        # Bypass pydantic to store the non-serializable client
        object.__setattr__(self, "_mcp_client", mcp_client)

    def _run(self, **kwargs: Any) -> str:
        """Synchronous wrapper — runs the async call in an event loop."""
        # Filter out None values (optional args not provided)
        clean_args = {k: v for k, v in kwargs.items() if v is not None and k != "kwargs"}
        # Handle the free-form "kwargs" field
        if "kwargs" in kwargs and kwargs["kwargs"]:
            clean_args.update(kwargs["kwargs"])

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're already in an async context; schedule as coroutine
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self._arun(**clean_args))
                    return future.result()
            else:
                return loop.run_until_complete(self._arun(**clean_args))
        except Exception as e:
            return f"Tool error: {e}"

    async def _arun(self, **kwargs: Any) -> str:
        """Async execution via the MCP client."""
        clean_args = {k: v for k, v in kwargs.items() if v is not None}
        client = object.__getattribute__(self, "_mcp_client")
        return await client.call_tool(self.mcp_tool_name, clean_args)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

async def build_langchain_tools(mcp_client: Any) -> List[MCPTool]:
    """
    Connect to the MCP server, enumerate all available tools, and return a
    list of MCPTool instances ready for use in LangChain/LangGraph nodes.

    Args:
        mcp_client: A connected HexstrikeMCPClient instance.

    Returns:
        List of MCPTool objects, one per Hexstrike tool.
    """
    raw_tools = await mcp_client.list_tools()
    langchain_tools: List[MCPTool] = []

    for tool in raw_tools:
        name = tool.name
        description = tool.description or f"Run the {name} security tool."
        schema = tool.inputSchema if hasattr(tool, "inputSchema") else {}

        lc_tool = MCPTool(
            mcp_client=mcp_client,
            tool_name=name,
            tool_description=description,
            input_schema=schema,
        )
        langchain_tools.append(lc_tool)

    return langchain_tools


def get_tool_by_name(tools: List[MCPTool], name: str) -> Optional[MCPTool]:
    """Lookup a tool by its MCP tool name."""
    for t in tools:
        if t.name == name:
            return t
    return None
