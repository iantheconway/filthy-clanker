"""
StateGraph assembly for the Filthy-Clanker multi-agent workflow.

Graph topology:
                                                    ┌──────────────────────────┐
                                                    │                          │
    START ──► supervisor ──► recon ──────► evaluator ──►│ supervisor               │
                   ▲         webexplorer ── evaluator ──►│                          │
                   │         vulnsearch ─── evaluator ──►│                          │
                   │         exploit ────── evaluator ──►│                          │
                   │         privesc ────── evaluator ──►│                          │
                   │                        │        │                          │
                   │              refusal   │        └──────────────────────────┘
                   │           ┌───────────►│
                   └───────────┤ refusal_specialist
                               │ (llama3-abliterated)
                   compaction ─┘
                              (loop until __end__)

Checkpointing: SqliteSaver persists graph state after every node execution,
enabling crash recovery and session resume via thread_id.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from .state import TeamState
from .supervisor import supervisor_node, route_from_supervisor
from .summarizer import compaction_node
from .file_sweep import sweep_files
from .agents import (
    make_recon_node, make_exploit_node, make_privesc_node, make_webexplorer_node,
    make_vulnsearch_node, make_reversing_node, make_refusal_specialist_node,
    lightweight_evaluator,
)

logger = logging.getLogger(__name__)


def create_checkpointer(db_path: str):
    """
    Return an AsyncSqliteSaver async context manager.

    Usage in main:
        async with create_checkpointer(db_path) as checkpointer:
            graph = build_graph(mcp_client, config, checkpointer)
            ...
    """
    os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
    return AsyncSqliteSaver.from_conn_string(db_path)


def build_graph(mcp_client: Any, config: Dict[str, Any], checkpointer):
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
    webexplorer_node = make_webexplorer_node(mcp_client, tools)
    vulnsearch_node = make_vulnsearch_node(mcp_client, tools)
    reversing_node = make_reversing_node(mcp_client, tools)
    refusal_specialist_node = make_refusal_specialist_node(mcp_client)

    # -----------------------------------------------------------------------
    # Construct the StateGraph
    # -----------------------------------------------------------------------
    graph = StateGraph(TeamState)

    # Add nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("recon", recon_node)
    graph.add_node("exploit", exploit_node)
    graph.add_node("privesc", privesc_node)
    graph.add_node("webexplorer", webexplorer_node)
    graph.add_node("vulnsearch", vulnsearch_node)
    graph.add_node("reversing", reversing_node)
    graph.add_node("refusal_specialist", refusal_specialist_node)
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
            "webexplorer": "webexplorer",
            "vulnsearch": "vulnsearch",
            "reversing": "reversing",
            "refusal_specialist": "refusal_specialist",
            "compaction": "compaction",
            "__end__": END,
        },
    )

    # Each agent node passes through the lightweight evaluator before returning
    # to the supervisor — the evaluator redirects to refusal_specialist if needed.
    _evaluator_map = {"supervisor": "supervisor", "refusal_specialist": "refusal_specialist"}
    for _agent in ("recon", "exploit", "privesc", "webexplorer", "vulnsearch", "reversing"):
        graph.add_conditional_edges(_agent, lightweight_evaluator, _evaluator_map)

    # refusal_specialist always hands back to supervisor after fixing a response
    graph.add_edge("refusal_specialist", "supervisor")
    graph.add_edge("compaction", "supervisor")

    # Compile with checkpointer for persistence
    compiled = graph.compile(checkpointer=checkpointer)
    return compiled


def initial_state(
    task: str,
    provider: Optional[str],
    session_id: str,
    config: Dict[str, Any],
    target_ip: str = "",
    machine_name: str = "",
    flag_format: str = "",
    challenge_category: str = "",
    has_files: bool = False,
    provided_files: Optional[List[str]] = None,
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
    provided_files = provided_files or []

    kb = {}
    if target_ip:
        kb["ips"] = [target_ip]
    if machine_name:
        kb["notes"] = [f"Target machine name: {machine_name}"]

    # Deterministic pre-solve file sweep: if the challenge ships files, grep the
    # flag format across all of them (unpacking archives) BEFORE any LLM routing.
    # A flag literally sitting in a source/config/Dockerfile is captured here so it
    # can never be missed by a suboptimal routing decision (the failure that
    # regressed historypeats/MFW). Text-file flags seed kb["flags"] and end the run
    # via the supervisor's existing flag-end gate; decoy-prone binary hits are
    # recorded as leads for the agents to weigh, not auto-ended on.
    if provided_files:
        try:
            swept = sweep_files(provided_files, flag_format)
        except Exception as exc:  # a sweep failure must never block the run
            logger.warning("File sweep failed: %s", exc)
            swept = {"flags": [], "leads": []}
        # Only auto-end on a UNIQUE text-file candidate — multiple candidates mean
        # ambiguity (possible decoy), so surface them as leads instead of guessing.
        if len(swept["flags"]) == 1:
            kb["flags"] = swept["flags"]
            kb["grounded_flags"] = list(swept["flags"])  # from a handout file → grounded (may end)
            logger.info("[file_sweep] Flag found in provided files: %s", swept["flags"])
        elif swept["flags"]:
            swept["leads"] = swept["flags"] + swept["leads"]
            logger.info("[file_sweep] %d ambiguous flag candidates in files → leads",
                        len(swept["flags"]))
        if swept["leads"]:
            kb["notes"] = kb.get("notes", []) + [
                f"file_sweep flag-format candidates (verify — may be decoys): "
                f"{swept['leads'][:20]}"
            ]

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
        unproductive_streak=0,
        flag_format=flag_format,
        challenge_category=challenge_category,
        has_files=has_files,
        provided_files=provided_files,
        completed_agents=[],
        provider=provider,
        context_token_estimate=len(task) // 4,
        hitl_reason=None,
        session_id=session_id,
        config=config,
    )
