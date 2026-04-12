import json
import logging
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger("filthy_clanker")


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

        args_snippet = json.dumps(arguments)[:200]
        logger.info("[MCP] CALL  %s(%s)", name, args_snippet)

        try:
            result = await self.session.call_tool(name, arguments)
        except Exception as exc:
            logger.error("[MCP] ERROR %s — %s: %s", name, type(exc).__name__, exc)
            raise

        parts = []
        for block in result.content:
            if hasattr(block, "text"):
                parts.append(block.text)
            else:
                parts.append(json.dumps(block.model_dump()))
        output = "\n".join(parts)

        logger.info("[MCP] RESULT %s — %d chars: %s%s",
                    name, len(output),
                    output[:300].replace("\n", " "),
                    " …" if len(output) > 300 else "")
        return output

    async def disconnect(self) -> None:
        """Shut down the session and server process."""
        await self._exit_stack.aclose()
        self.session = None
        self._tools_cache = None
