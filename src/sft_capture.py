"""
SFT trajectory capture — full-fidelity (system, messages, tools) → assistant records.

WHY THIS EXISTS
───────────────
The older ``data_capture.TrajectoryLogger`` (still used for analytics) records a
*derived* view of each step: the parsed ``{tool_name, arguments}`` + a truncated
result snippet + KB diffs. That is fine for the success-score heuristic, but it is
NOT what supervised fine-tuning needs. SFT of a tool-calling agent needs the exact
prompt the model saw (system prompt + full message history + the tool JSON schema)
paired with the exact assistant completion (its text + tool_calls) — token-for-token
what the deployed model will produce. The eval's ``_log_trajectories`` also passed an
EMPTY ``state_before`` (see run_eval.py), so its scores are unreliable and its records
carry no prompt/completion. None of the 306 legacy files are usable as SFT data.

WHAT THIS CAPTURES
──────────────────
One record per *agent invocation* (one pass through ``_run_agent_loop``). Each record
holds the system prompt, the tool schema actually presented (post-trim), and the full
message list for that invocation — i.e. every assistant turn (with tool_calls) and the
tool results it received, in order. A dataset builder can later split each record into
one training example per assistant turn (context → assistant completion), or keep whole
trajectories. Capturing once per invocation (not once per ReAct iteration) avoids the
O(n^2) context duplication you'd get by dumping the growing history every step.

This is OPT-IN and side-effect-isolated: it never raises into the agent loop, and does
nothing unless capture is enabled. It writes append-only JSONL to ``data/sft/``.

ENABLING
────────
Set ``CLANKER_CAPTURE_SFT=1`` (run_eval exposes ``--capture-sft``). Optional:
  • ``CLANKER_SFT_DIR``   — output dir (default ``<project>/data/sft``).
  • ``CLANKER_SFT_TOOLS`` — ``full`` (default: store the whole tool schema) or
                            ``names`` (store only tool names, much smaller files).

RELATION TO LANGSMITH
─────────────────────
``OllamaClient.generate_response`` is ``@traceable`` — so when LANGSMITH_TRACING=true
LangSmith already stores the same (messages, tools, system) inputs and the response
output for every LLM call. This module is the local, self-contained equivalent that
does not depend on the tracing backend and is trivially exportable to a training set.
See docs/SFT_PLAN.md for how the two compare and which to use.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Resolved lazily so tests / imports don't touch the filesystem.
_OUT_DIR: Optional[Path] = None


def capture_enabled() -> bool:
    """True when SFT capture is switched on via the environment."""
    return os.getenv("CLANKER_CAPTURE_SFT", "").strip().lower() in ("1", "true", "yes", "on")


def _project_root() -> Path:
    # src/sft_capture.py → project root is one level above src/.
    return Path(__file__).resolve().parent.parent


def _out_dir() -> Path:
    global _OUT_DIR
    if _OUT_DIR is None:
        raw = os.getenv("CLANKER_SFT_DIR", "").strip()
        _OUT_DIR = Path(raw) if raw else (_project_root() / "data" / "sft")
        _OUT_DIR.mkdir(parents=True, exist_ok=True)
    return _OUT_DIR


def _tools_repr(tools: list) -> Any:
    """How to persist the tool schema, per CLANKER_SFT_TOOLS.

    ``full`` keeps the exact JSON schema the model was shown (needed to reproduce the
    prompt for tool-calling SFT). ``names`` keeps only the ordered tool names — much
    smaller, useful when the schema is constant and reconstructed at build time.
    """
    mode = os.getenv("CLANKER_SFT_TOOLS", "full").strip().lower()
    if mode == "names":
        return [t.get("name", "") for t in tools if isinstance(t, dict)]
    return tools


def record_turn(
    *,
    session_id: str,
    agent: str,
    provider: str,
    model: str,
    system_prompt: str,
    tools: list,
    messages: list,
    task: str = "",
    knowledge_base: Optional[dict] = None,
) -> None:
    """Append one agent-invocation trajectory to ``data/sft/<session_id>.jsonl``.

    ``messages`` is the full message list for this invocation (incoming shared history
    plus every assistant/tool turn produced in the loop). ``system_prompt`` and
    ``tools`` are exactly what was passed to the LLM client, so (system_prompt, messages
    minus the trailing assistant turns, tools) reproduces the prompt and each assistant
    turn is a training target.

    Never raises: capture must not be able to break an eval run.
    """
    if not capture_enabled():
        return
    try:
        rec = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id or "unknown",
            "agent": agent,
            "provider": provider,
            "model": model,
            "task": task,
            "system_prompt": system_prompt,
            "tools": _tools_repr(tools),
            "tool_count": len(tools) if isinstance(tools, list) else 0,
            "messages": messages,
            "knowledge_base": knowledge_base or {},
            # Convenience: how many assistant turns this trajectory yields as targets.
            "assistant_turns": sum(
                1 for m in (messages or [])
                if isinstance(m, dict) and m.get("role") == "assistant"
            ),
        }
        safe_session = (session_id or "unknown").replace("/", "_").replace("\\", "_")
        path = _out_dir() / f"{safe_session}.jsonl"
        line = json.dumps(rec, ensure_ascii=False, default=str) + "\n"
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line)
    except Exception:
        # Deliberately swallow — a capture failure must never abort the agent loop.
        # (If you're debugging capture, set CLANKER_SFT_DEBUG=1 to surface it.)
        if os.getenv("CLANKER_SFT_DEBUG", "").strip() in ("1", "true", "yes"):
            import traceback
            traceback.print_exc()


# ── Timeout-safe capture ──────────────────────────────────────────────────────
# record_turn() only fires when an agent invocation RETURNS normally. A run killed
# by the challenge timeout is usually cancelled MID-ReAct-loop, so the (often most
# valuable, longest) in-progress trajectory would be lost — this is why HTB runs that
# time out captured nothing. The stash fixes that: the agent loop registers its
# in-progress invocation once (by REFERENCE to the live ``messages``/``new_messages``
# lists — no per-iteration copy, so no O(n^2)); on cancellation the run harness calls
# flush_stash() to write whatever was accumulated. The graph runs one invocation at a
# time, so a single global slot is safe.
_STASH: Optional[dict] = None


def stash_invocation(**kwargs) -> None:
    """Register the in-progress invocation so a timeout can still capture it. No-op if disabled."""
    global _STASH
    if not capture_enabled():
        return
    _STASH = kwargs


def clear_stash() -> None:
    """Drop the stash (call after a normal record_turn so a later flush can't duplicate it)."""
    global _STASH
    _STASH = None


def flush_stash() -> bool:
    """Write the stashed in-progress invocation, if any (call from the timeout handler).

    Reads the live message lists that were stashed by reference. Returns True if a
    record was written. Never raises.
    """
    global _STASH
    kw, _STASH = _STASH, None
    if not kw:
        return False
    try:
        msgs = list(kw.get("messages") or []) + list(kw.get("new_messages") or [])
        record_turn(
            session_id=kw.get("session_id", ""), agent=kw.get("agent", ""),
            provider=kw.get("provider", ""), model=kw.get("model", ""),
            system_prompt=kw.get("system_prompt", ""), tools=kw.get("tools") or [],
            messages=msgs, task=kw.get("task", ""), knowledge_base=kw.get("knowledge_base") or {},
        )
        return True
    except Exception:
        return False
