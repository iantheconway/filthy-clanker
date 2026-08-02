"""Tests for the LangGraph multi-agent layer (graph/agents.py, supervisor.py).

These exercise the wiring between the graph nodes and the LLM clients — the
exact seams that were broken in the first multi-agent pass (tool arguments
dropped, tool schemas double-formatted, tool_result messages malformed, the
sync checkpointer used under async streaming).
"""
import copy

import pytest

from graph import agents, supervisor
from graph.agents import (
    _extract_kb_updates,
    _run_agent_loop,
)
from graph.graph import build_graph, create_checkpointer, initial_state
from llms.anthropic_client import AnthropicClient


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _Block:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Raw:
    def __init__(self, blocks):
        self.content = [_Block(**b) for b in blocks]


def _anthropic_response(text=None, tool_calls=None):
    """Build a response dict in the exact shape AnthropicClient.generate_response returns."""
    tool_calls = tool_calls or []
    blocks = []
    if text:
        blocks.append({"type": "text", "text": text})
    for tc in tool_calls:
        blocks.append({"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["arguments"]})
    return {
        "text": text,
        "tool_calls": tool_calls,
        "raw": _Raw(blocks),
        "stop_reason": "tool_use" if tool_calls else "end_turn",
    }


class ScriptedAnthropic(AnthropicClient):
    """AnthropicClient whose generate_response replays a canned script.

    Everything else (format_tools, parse_tool_calls, make_assistant_message,
    make_tool_result_message) is the real implementation, so the test exercises
    the true provider message-shaping code.
    """

    def __init__(self, script):
        self.model = "test-model"
        self._script = list(script)
        self._i = 0
        self.tools_seen = []

    async def generate_response(self, messages, tools, system_prompt):
        self.tools_seen.append(tools)
        resp = self._script[self._i]
        self._i += 1
        return resp


class FakeMCP:
    def __init__(self, tools, result="scan complete"):
        self._tools = tools
        self._result = result
        self.calls = []

    async def list_tools(self):
        return self._tools

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        return self._result


NMAP_TOOLS = [{
    "name": "nmap_scan",
    "description": "Run nmap",
    "inputSchema": {
        "type": "object",
        "properties": {"target": {"type": "string"}},
        "required": ["target"],
    },
}]

NMAP_OUTPUT = (
    "Starting Nmap...\n"
    "22/tcp open ssh OpenSSH 8.2\n"
    "80/tcp open http nginx\n"
    "HTB{demo_flag_123}\n"
)


def _config_for(agent_name, provider="anthropic"):
    return {
        "agents": {agent_name: {"provider": provider, "model": "t", "system_prompt": agent_name}},
        # High threshold so maybe_summarize never reaches out to Ollama in tests.
        "settings": {"tool_output_threshold": 1_000_000},
    }


# ---------------------------------------------------------------------------
# _extract_kb_updates
# ---------------------------------------------------------------------------

def test_extract_kb_ports_services_flags():
    out = _extract_kb_updates("nmap_scan", NMAP_OUTPUT, {"ips": ["10.10.11.5"]})
    assert out["open_ports"]["10.10.11.5"] == [22, 80]
    assert out["services"]["10.10.11.5:22"] == "ssh"
    assert out["services"]["10.10.11.5:80"] == "http"
    assert any(f.upper().startswith("HTB{") for f in out["flags"])


def test_extract_kb_credentials():
    out = _extract_kb_updates("hydra", "username: admin\npassword: hunter2", {})
    assert {"user": "admin", "pass": "hunter2"} in out["credentials"]


def test_extract_kb_does_not_mutate_input():
    """The input KB may be checkpointed shared state — it must not be mutated."""
    kb = {"ips": ["10.10.11.5"], "open_ports": {}, "services": {}, "credentials": []}
    snapshot = copy.deepcopy(kb)
    _extract_kb_updates("nmap_scan", NMAP_OUTPUT, kb)
    assert kb == snapshot


# ---------------------------------------------------------------------------
# Failure / progress heuristics
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _run_agent_loop — the integration seam
# ---------------------------------------------------------------------------

async def test_run_agent_loop_wiring(monkeypatch):
    """Tool args reach the MCP client, raw schemas reach the LLM, results are shaped right."""
    cfg = _config_for("recon")
    state = initial_state("hack 10.10.11.5", "anthropic", "sess-1", cfg, target_ip="10.10.11.5")

    fake_llm = ScriptedAnthropic([
        _anthropic_response(text="scanning", tool_calls=[
            {"id": "tu1", "name": "nmap_scan", "arguments": {"target": "10.10.11.5"}},
        ]),
        _anthropic_response(text="found ssh and http"),
    ])
    monkeypatch.setattr(agents, "_resolve_llm", lambda state, agent_cfg: ("anthropic", "t", fake_llm))
    mcp = FakeMCP(NMAP_TOOLS, NMAP_OUTPUT)

    out = await _run_agent_loop("recon", state, [], mcp)

    # 1. Tool called with the correct, non-empty arguments (previously always {}).
    assert mcp.calls == [("nmap_scan", {"target": "10.10.11.5"})]
    # 2. Raw MCP schemas passed to the LLM, not pre-formatted (previously double-formatted).
    assert fake_llm.tools_seen[0] == NMAP_TOOLS
    # 3. A well-formed Anthropic tool_result user message was produced.
    user_msgs = [m for m in out["messages"] if m.get("role") == "user"]
    block = user_msgs[0]["content"][0]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "tu1"
    assert "22/tcp open ssh" in block["content"]
    # 4. KB updated from the tool output.
    assert out["knowledge_base"]["open_ports"]["10.10.11.5"] == [22, 80]
    assert out["current_agent"] == "recon"


async def test_exploit_attempts_increment_on_failure(monkeypatch):
    cfg = _config_for("exploit")
    state = initial_state("x", "anthropic", "s", cfg, target_ip="10.10.11.5")
    state["exploit_attempts"] = 2

    fake_llm = ScriptedAnthropic([
        _anthropic_response(text="trying", tool_calls=[{"id": "e1", "name": "run_exploit", "arguments": {}}]),
        _anthropic_response(text="no luck"),
    ])
    monkeypatch.setattr(agents, "_resolve_llm", lambda s, c: ("anthropic", "t", fake_llm))
    tools = [{"name": "run_exploit", "description": "x", "inputSchema": {"type": "object", "properties": {}}}]
    mcp = FakeMCP(tools, "Exploit failed: Connection refused")

    out = await _run_agent_loop("exploit", state, [], mcp)
    assert out["exploit_attempts"] == 3


async def test_exploit_attempts_not_incremented_on_success(monkeypatch):
    """A successful exploit (data/flag returned, no hard failure) must not add to
    the failure streak — only hard fails (non-zero exit, stderr, connection
    errors) increment it. The HITL threshold handles genuine exhaustion."""
    cfg = _config_for("exploit")
    state = initial_state("x", "anthropic", "s", cfg, target_ip="10.10.11.5")
    state["exploit_attempts"] = 4  # already deep into a failing streak

    fake_llm = ScriptedAnthropic([
        _anthropic_response(text="try", tool_calls=[{"id": "e1", "name": "run_exploit", "arguments": {}}]),
        _anthropic_response(text="got a shell"),
    ])
    monkeypatch.setattr(agents, "_resolve_llm", lambda s, c: ("anthropic", "t", fake_llm))
    tools = [{"name": "run_exploit", "description": "x", "inputSchema": {"type": "object", "properties": {}}}]
    mcp = FakeMCP(tools, "shell obtained\nHTB{root_flag}")

    out = await _run_agent_loop("exploit", state, [], mcp)
    assert out["knowledge_base"]["flags"]
    assert out["exploit_attempts"] == 4  # success is not a failure — streak unchanged


# ---------------------------------------------------------------------------
# Supervisor routing
# ---------------------------------------------------------------------------

async def test_supervisor_routes_from_json(monkeypatch):
    cfg = {"agents": {"supervisor": {"provider": "anthropic", "model": "t", "system_prompt": "route"}}, "settings": {}}
    state = initial_state("x", "anthropic", "s", cfg)
    fake_llm = ScriptedAnthropic([
        _anthropic_response(text='{"next": "exploit", "reasoning": "foothold known", "hitl_reason": null}'),
    ])
    monkeypatch.setattr(supervisor, "_resolve_llm", lambda s, c: ("anthropic", "t", fake_llm))

    out = await supervisor.supervisor_node(state)
    assert out["next"] == "exploit"


async def test_supervisor_finishes_on_flag():
    cfg = {"agents": {}, "settings": {}}
    state = initial_state("x", "anthropic", "s", cfg)
    state["knowledge_base"] = {"flags": ["HTB{done}"]}
    out = await supervisor.supervisor_node(state)
    assert out["next"] == "__end__"


def test_route_from_supervisor():
    assert supervisor.route_from_supervisor({"next": "privesc"}) == "privesc"
    assert supervisor.route_from_supervisor({}) == "recon"


# ---------------------------------------------------------------------------
# Checkpointer + graph assembly
# ---------------------------------------------------------------------------

async def test_create_checkpointer_is_async_saver(tmp_path):
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    # create_checkpointer returns an async context manager (from_conn_string),
    # not an awaitable — the graph is driven with astream/aget_state, so a sync
    # SqliteSaver would raise NotImplementedError under async streaming.
    async with create_checkpointer(str(tmp_path / "cp.db")) as cp:
        assert isinstance(cp, AsyncSqliteSaver)


async def test_build_graph_compiles(tmp_path):
    async with create_checkpointer(str(tmp_path / "cp.db")) as cp:
        graph = build_graph(FakeMCP([]), {"agents": {}, "settings": {}}, cp)
        assert graph is not None
        # Nodes are wired: supervisor plus the three specialists and compaction.
        assert "supervisor" in graph.nodes
        assert {"recon", "exploit", "privesc", "compaction"} <= set(graph.nodes)


async def test_hitl_exploit_loop_interrupt_and_resume(monkeypatch, tmp_path):
    """When the exploit failure streak hits the threshold, the supervisor pauses
    via interrupt(); a Command(resume=...) injects operator input and resets the
    streak. Exercises interrupt()/Command through the compiled graph."""
    from langgraph.types import Command

    # Exploit specialist that immediately captures a flag once resumed.
    class ExploitFake(AnthropicClient):
        def __init__(self):
            self.model = "t"
            self.i = 0

        async def generate_response(self, messages, tools, system_prompt):
            i, self.i = self.i, self.i + 1
            if i == 0:
                return _anthropic_response(text="retry", tool_calls=[
                    {"id": "e1", "name": "run_exploit", "arguments": {"target": "10.10.11.5"}},
                ])
            return _anthropic_response(text="shell obtained")

    monkeypatch.setattr(agents, "_resolve_llm", lambda s, c: ("anthropic", "t", ExploitFake()))
    monkeypatch.setattr(supervisor, "_resolve_llm", lambda s, c: ("anthropic", "t", ExploitFake()))

    class FlagMCP:
        async def list_tools(self):
            return [{"name": "run_exploit", "description": "e", "inputSchema": {"type": "object", "properties": {}}}]

        async def call_tool(self, name, args):
            return "meterpreter session 1 opened\nHTB{after_hint}\n"

    cfg = {
        "agents": {a: {"provider": "anthropic", "model": "t", "system_prompt": a}
                   for a in ("supervisor", "recon", "exploit", "privesc")},
        "settings": {"tool_output_threshold": 1_000_000, "max_exploit_attempts": 5,
                     "context_limit_threshold": 80000},
    }
    async with create_checkpointer(str(tmp_path / "cp.db")) as cp:
        graph = build_graph(FlagMCP(), cfg, cp)
        state = initial_state("x", "anthropic", "hitl-1", cfg, target_ip="10.10.11.5")
        state["exploit_attempts"] = 5  # already at the HITL threshold
        thread = {"configurable": {"thread_id": "hitl-1"}}

        interrupted = False
        async for ev in graph.astream(state, thread, stream_mode="updates"):
            if "__interrupt__" in ev:
                interrupted = True
                payload = ev["__interrupt__"][0].value
                assert payload["reason"] == "exploit_loop"
        assert interrupted, "supervisor should have paused for human input"

        # Operator supplies a hint; graph resumes, exploit runs, flag is captured.
        async for ev in graph.astream(Command(resume="try SQLi on /login"), thread, stream_mode="updates"):
            pass
        final = await graph.aget_state(thread)
        assert final.values["knowledge_base"].get("flags") == ["HTB{after_hint}"]
        assert final.next == ()  # ran to completion
