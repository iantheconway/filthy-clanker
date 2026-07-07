"""
StateGraph assembly for the Filthy-Clanker multi-agent workflow.

Graph topology:
                      ┌─────────────────────────────────────┐
                      │                                     │
    START ──► supervisor ──► recon ────────────────────────►│
                   ▲         exploit ─────────────────────►│
                   │         privesc ────────────────────►  │
                   │         compaction ─────────────────►  │
                   └─────────────────────────────────────────┘
                              (loop until __end__)

Checkpointing: SqliteSaver persists graph state after every node execution,
enabling crash recovery and session resume via thread_id.
"""
from __future__ import annotations

import os
from typing import Any, Dict

import aiosqlite
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from .state import TeamState
from .supervisor import supervisor_node, route_from_supervisor
from .summarizer import compaction_node
from .agents import make_recon_node, make_exploit_node, make_privesc_node


async def create_checkpointer(db_path: str) -> AsyncSqliteSaver:
    """
    Create and return an AsyncSqliteSaver checkpointer.

    The graph is driven with ``astream``/``aget_state`` (async), so an async
    checkpointer is required — the synchronous ``SqliteSaver`` raises
    ``NotImplementedError`` on the async methods LangGraph calls during
    ``astream``. The database file (and its schema) are created automatically.

    The caller owns the returned saver's ``conn`` and should close it at
    shutdown (``await checkpointer.conn.close()``).
    """
    os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
    conn = await aiosqlite.connect(db_path)
    saver = AsyncSqliteSaver(conn)
    await saver.setup()
    return saver


def build_graph(mcp_client: Any, config: Dict[str, Any], checkpointer: AsyncSqliteSaver):
    """
    Assemble and compile the LangGraph StateGraph.

    Args:
        mcp_client:   Connected HexstrikeMCPClient instance.
        config:       Parsed agents.yaml dict.
        checkpointer: SqliteSaver for persistent state.

    Returns:
        A compiled LangGraph CompiledGraph ready for invocation.
    """
    # Build agent nodes (closures that capture mcp_client)
    tools: list = []  # LangChain tool objects; built lazily — agents use mcp_client directly
    recon_node = make_recon_node(mcp_client, tools)
    exploit_node = make_exploit_node(mcp_client, tools)
    privesc_node = make_privesc_node(mcp_client, tools)

    # -----------------------------------------------------------------------
    # Construct the StateGraph
    # -----------------------------------------------------------------------
    graph = StateGraph(TeamState)

    # Add nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("recon", recon_node)
    graph.add_node("exploit", exploit_node)
    graph.add_node("privesc", privesc_node)
    graph.add_node("compaction", compaction_node)

    # Entry: always start at supervisor
    graph.add_edge(START, "supervisor")

    # Conditional routing from supervisor
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "recon": "recon",
            "exploit": "exploit",
            "privesc": "privesc",
            "compaction": "compaction",
            "__end__": END,
        },
    )

    # All agent nodes report back to supervisor after completing their sub-task
    graph.add_edge("recon", "supervisor")
    graph.add_edge("exploit", "supervisor")
    graph.add_edge("privesc", "supervisor")
    graph.add_edge("compaction", "supervisor")

    # Compile with checkpointer for persistence
    compiled = graph.compile(checkpointer=checkpointer)
    return compiled


def initial_state(
    task: str,
    provider: str,
    session_id: str,
    config: Dict[str, Any],
    target_ip: str = "",
    machine_name: str = "",
) -> TeamState:
    """
    Build the initial TeamState for a new hacking session.

    Args:
        task:         High-level objective (e.g., "Hack the machine at 10.10.11.1").
        provider:     Primary LLM provider for the session.
        session_id:   Unique session/thread ID.
        config:       Parsed agents.yaml.
        target_ip:    Optional pre-known target IP.
        machine_name: Optional machine name from HTB.

    Returns:
        A complete TeamState dict ready for graph invocation.
    """
    kb = {}
    if target_ip:
        kb["ips"] = [target_ip]
    if machine_name:
        kb["notes"] = [f"Target machine name: {machine_name}"]

    initial_message = {
        "role": "user",
        "content": task,
    }

    return TeamState(
        messages=[initial_message],
        knowledge_base=kb,
        current_agent="none",
        task=task,
        next="recon",
        exploit_attempts=0,
        provider=provider,
        context_token_estimate=len(task) // 4,
        hitl_reason=None,
        session_id=session_id,
        config=config,
    )
