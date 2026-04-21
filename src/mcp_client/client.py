import json
import logging
from contextlib import AsyncExitStack
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langsmith import traceable
import langsmith

logger = logging.getLogger("filthy_clanker")


class HexstrikeMCPClient:
    """Single-server MCP client — connects to one stdio MCP server."""

    def __init__(self, command: str, args: list[str] | None = None,
                 env: dict[str, str] | None = None):
        self.server_params = StdioServerParameters(
            command=command,
            args=args or [],
            env=env,
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


class MCPClientPool:
    """
    Aggregates multiple MCP server clients behind a single list_tools / call_tool
    interface.  Agents and graph nodes call this exactly like HexstrikeMCPClient.

    Tool routing:
      - On list_tools(), tools from all servers are merged.
      - If two servers expose a tool with the same name the first-registered server
        wins; a warning is logged.
      - call_tool() dispatches to whichever server registered the tool.
    """

    def __init__(self) -> None:
        self._servers: list[tuple[str, HexstrikeMCPClient]] = []
        # Maps tool name → client that owns it (populated on first list_tools call)
        self._tool_router: dict[str, HexstrikeMCPClient] = {}
        self._tools_cache: list[dict] | None = None

    def add_server(self, name: str, client: HexstrikeMCPClient) -> None:
        """Register a server.  Call before connect()."""
        self._servers.append((name, client))

    async def connect(self) -> None:
        """Connect to all registered servers.  Servers that fail are skipped."""
        for name, client in self._servers:
            try:
                await client.connect()
                logger.info("[MCPPool] Connected → %s", name)
            except Exception as exc:
                logger.error("[MCPPool] Failed to connect to %s: %s", name, exc)

    async def list_tools(self) -> list[dict]:
        """Return merged tool list from all connected servers."""
        if self._tools_cache is not None:
            return self._tools_cache

        all_tools: list[dict] = []
        self._tool_router = {}

        for server_name, client in self._servers:
            try:
                tools = await client.list_tools()
            except Exception as exc:
                logger.error("[MCPPool] list_tools failed for %s: %s", server_name, exc)
                continue

            registered = 0
            for tool in tools:
                tname = tool["name"]
                if tname in self._tool_router:
                    logger.warning(
                        "[MCPPool] Tool name collision: '%s' already registered by another "
                        "server — skipping duplicate from %s", tname, server_name
                    )
                    continue
                self._tool_router[tname] = client
                all_tools.append(tool)
                registered += 1

            logger.info("[MCPPool] %s: %d/%d tools registered", server_name, registered, len(tools))

        self._tools_cache = all_tools
        logger.info("[MCPPool] Total tools available: %d", len(all_tools))
        return self._tools_cache

    @traceable(run_type="tool", name="mcp_tool")
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Dispatch a tool call to the server that owns it."""
        rt = langsmith.get_current_run_tree()
        if rt:
            rt.name = f"tool / {name}"

        # Populate router if needed
        if not self._tool_router:
            await self.list_tools()

        client = self._tool_router.get(name)
        if client is None:
            error_msg = f"Unknown tool: {name}"
            logger.error("[MCPPool] %s", error_msg)
            return error_msg

        return await client.call_tool(name, arguments)

    async def disconnect(self) -> None:
        """Disconnect all servers."""
        for name, client in self._servers:
            try:
                await client.disconnect()
                logger.info("[MCPPool] Disconnected from %s", name)
            except BaseException as exc:
                # CancelledError is BaseException (not Exception) in Python 3.8+;
                # swallow all disconnect errors — the process is exiting anyway.
                logger.debug("[MCPPool] Ignored error disconnecting from %s: %s", name, exc)
        self._tool_router = {}
        self._tools_cache = None
