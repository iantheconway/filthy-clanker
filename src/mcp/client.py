import json
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class HexstrikeMCPClient:
    def __init__(self, command: str, args: list[str] | None = None):
        self.server_params = StdioServerParameters(
            command=command,
            args=args or [],
        )
        self.session: ClientSession | None = None
        self._exit_stack = AsyncExitStack()
        self._tools_cache: list[dict] | None = None

    async def connect(self) -> None:
        """Start the MCP server process and initialise the session."""
        transport = await self._exit_stack.enter_async_context(
            stdio_client(self.server_params)
        )
        read_stream, write_stream = transport
        self.session = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await self.session.initialize()

    async def list_tools(self) -> list[dict]:
        """Return the list of tools exposed by the MCP server as plain dicts."""
        if self._tools_cache is not None:
            return self._tools_cache

        if not self.session:
            raise RuntimeError("Not connected — call connect() first")

        result = await self.session.list_tools()
        self._tools_cache = [
            {
                "name": tool.name,
                "description": tool.description or "",
                "inputSchema": tool.inputSchema if tool.inputSchema else {
                    "type": "object",
                    "properties": {},
                },
            }
            for tool in result.tools
        ]
        return self._tools_cache

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool call and return the textual result."""
        if not self.session:
            raise RuntimeError("Not connected — call connect() first")

        result = await self.session.call_tool(name, arguments)

        parts = []
        for block in result.content:
            if hasattr(block, "text"):
                parts.append(block.text)
            else:
                parts.append(json.dumps(block.model_dump()))
        return "\n".join(parts)

    async def disconnect(self) -> None:
        """Shut down the session and server process."""
        await self._exit_stack.aclose()
        self.session = None
        self._tools_cache = None
