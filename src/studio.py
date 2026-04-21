"""
LangGraph Studio entry point.

Builds the graph with:
  - MemorySaver checkpointer (no SQLite file needed)
  - Stub MCP client (tool schemas unavailable, but graph structure is intact)
  - Real agents.yaml config (so node names / routing logic are accurate)

Studio uses this module-level `graph` variable for visualization and
interactive debugging. Live tool calls will return a placeholder message.
"""
from __future__ import annotations

import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(__file__))
from graph.graph import build_graph  # noqa: E402


# ---------------------------------------------------------------------------
# Stub MCP client — provides empty tool list so graph compiles without a
# running Hexstrike server.
# ---------------------------------------------------------------------------

class _StubMCPClient:
    async def list_tools(self) -> list:
        return []

    async def call_tool(self, name: str, arguments: dict) -> str:
        return f"[Studio stub — tool '{name}' is not available in Studio mode]"

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Load config and build graph.
# Pass checkpointer=None — LangGraph Studio / API injects its own persistence.
# ---------------------------------------------------------------------------

_config_path = os.path.join(os.path.dirname(__file__), "..", "agents.yaml")
with open(_config_path) as _f:
    _config = yaml.safe_load(_f)

graph = build_graph(_StubMCPClient(), _config, checkpointer=None)
