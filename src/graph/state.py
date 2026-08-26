"""
TeamState and supporting types for the LangGraph multi-agent workflow.
"""
from typing import TypedDict, Annotated, List, Dict, Optional, Any


def _append_messages(left: list, right) -> list:
    """
    Append reducer for the messages field.

    Normal use: right is a list → append to left.
    Compaction use: right is {"__replace__": [...]}} → replace left entirely.
    This lets compaction_node swap out the full history without changing the
    state schema or the reducer annotation.
    """
    if not right:
        return left
    if isinstance(right, dict) and "__replace__" in right:
        return list(right["__replace__"])
    return left + list(right)


def _append_tool_log(left: list, right) -> list:
    """Append reducer for raw_tool_log, capped so it can't bloat the checkpoint.
    Holds the RAW (pre-summarisation) tool output the frontier-hint payload needs."""
    if not right:
        return left or []
    return (list(left or []) + list(right))[-60:]


class KnowledgeBase(TypedDict, total=False):
    """Structured environmental facts shared across all agents."""
    ips: List[str]
    open_ports: Dict[str, List[int]]       # ip -> [port, ...]
    services: Dict[str, str]               # "ip:port" -> service banner
    tech_stack: Dict[str, List[str]]       # "ip:port" -> ["Software/version", ...]
    response_headers: Dict[str, Dict[str, str]]  # "ip:port" -> {header: value}
    credentials: List[Dict[str, str]]      # [{"user": ..., "pass": ..., "service": ...}]
    flags: List[str]                       # All flag candidates (tool-grounded AND model-typed)
    grounded_flags: List[str]              # Flags extracted from TOOL OUTPUT only (never model
                                           # prose). Only these may END a challenge — a typed
                                           # guess must not, or the run dies the moment the model
                                           # guesses instead of continuing to solve.
    attack_surface: List[str]              # Discovered paths, endpoints, CVEs, vulns
    notes: List[str]                       # Freeform analyst notes
    # Work-history fields — survive compaction, prevent agents re-doing completed work
    scan_history: Dict[str, List[str]]     # ip -> ["[agent] tool_name: key_args", ...]
    visited_urls: List[str]                # URLs already fetched/analysed by webexplorer
    exploit_history: List[str]             # ["tool(args): outcome", ...] — failed attempts
    shells: List[Dict[str, str]]           # [{"user": "www-data", "host": "...", "via": "CVE-..."}]


class TeamState(TypedDict):
    """
    Shared state threaded through every node in the graph.

    messages              — Full conversation/task history (auto-appended).
    knowledge_base        — Structured facts about the target environment.
    current_agent         — Name of the agent that last ran ("recon", "exploit", etc.).
    task                  — Current high-level objective.
    next                  — Routing decision set by the supervisor.
    exploit_attempts      — Counter for consecutive failed exploit attempts.
    unproductive_streak   — Consecutive agent turns that executed ZERO real tool
                            calls. Reset to 0 the moment any agent runs a tool.
                            The supervisor ends a challenge once it exceeds
                            settings.max_unproductive_turns (the team is stuck —
                            emitting text without acting — so more time won't help).
    completed_agents      — Set of agent names that have signalled TASK COMPLETE
                            with substantive findings. Reset when KB changes significantly.
    provider              — Global provider override ("anthropic", "gemini", "ollama"),
                            or None to use each agent's own config from agents.yaml.
    context_token_estimate— Rough token count for context-limit tracking.
    hitl_reason           — If set, supervisor will interrupt for human input.
    session_id            — Unique ID for this hacking session (maps to checkpoint thread).
    config                — Runtime config loaded from agents.yaml.
    """
    messages: Annotated[list, _append_messages]
    knowledge_base: KnowledgeBase
    current_agent: str
    task: str
    next: str
    exploit_attempts: int
    unproductive_streak: int
    flag_format: str
    challenge_category: str   # CTF category (rev, pwn, crypto, …) — drives RE routing
    has_files: bool           # challenge ships local files (vs. only a live service)
    provided_files: List[str] # absolute paths of the challenge's handout files, if any
    completed_agents: List[str]
    provider: Optional[str]
    context_token_estimate: int
    hitl_reason: Optional[str]
    session_id: str
    config: Dict[str, Any]
    # Frontier-hint / Phase 3 guidance (see src/graph/guidance.py). Cooldown clock is
    # the message count, so no per-return counter is needed.
    hints_used: int            # frontier hints requested this session (capped)
    last_hint_step: int        # message count at the last hint (cooldown)
    hint_reason: Optional[str] # why the supervisor routed to guidance (stall symptom)
    hint_log: List[dict]       # captured (stall → payload → hint) triples for the flywheel
    raw_tool_log: Annotated[list, _append_tool_log]  # RAW tool cmd+output per call (guidance payload)
