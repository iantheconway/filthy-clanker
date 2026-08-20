"""
Load + normalize captured trajectories into the shape signals/detector expect.

Stdlib-only. Reads `data/training/*.jsonl` from the main tree by default.
"""
from __future__ import annotations

import glob
import json
import os
import re

from signals import normalize_cmd

def _find_data_dir() -> str:
    """Locate data/training. Honors $FC_DATA_DIR; else walks up from this file
    looking for a `data/training` dir. Works whether run from the main tree or a
    nested git worktree (worktrees share the repo but sit deeper in the path)."""
    env = os.environ.get("FC_DATA_DIR")
    if env:
        return env
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        cand = os.path.join(d, "data", "training")
        # require actual captured data (a worktree carries an empty data/training
        # from .gitkeep — the real jsonl live only in the main clone)
        if os.path.isdir(cand) and (
            os.path.exists(os.path.join(cand, "all_trajectories.jsonl"))
            or glob.glob(os.path.join(cand, "eval-*.jsonl"))
        ):
            return cand
        d = os.path.dirname(d)
    # last resort: main clone next to a .claude worktree root
    return os.path.join(d, "data", "training")


# Default corpus lives in the MAIN working tree (read-only), not this worktree —
# so we backtest against the real captured data without copying it.
DEFAULT_DATA_DIR = _find_data_dir()

_SID_CAT = re.compile(r"^eval-\d{4}[a-z]-([a-z]+)-")


def category_of(session_id: str) -> str:
    """Extract the challenge category from a session id, e.g.
    'eval-2013q-cry-slurp-...' -> 'cry'. Returns 'default' if unknown.
    """
    m = _SID_CAT.match(session_id or "")
    return m.group(1) if m else "default"


def _arg_string(action: dict) -> str:
    """Best-effort single string for a tool call's arguments (for repeat/ledger)."""
    args = action.get("arguments") or {}
    if isinstance(args, dict):
        if "command" in args:
            return str(args["command"])
        return json.dumps(args, sort_keys=True)
    return str(args)


def normalize_actions(records: list[dict]) -> list[dict]:
    """Turn raw action records (type absent, has 'action') into normalized dicts."""
    out = []
    for d in records:
        if d.get("type") is not None or "action" not in d:
            continue
        action = d.get("action") or {}
        raw_cmd = _arg_string(action)
        out.append({
            "idx": len(out),
            "agent": d.get("agent"),
            "tool": action.get("tool_name"),
            "raw_cmd": raw_cmd,
            "cmd": normalize_cmd(f"{action.get('tool_name','')} {raw_cmd}"),
            "score": d.get("success_score"),
            "result_snippet": d.get("result_snippet") or "",
            "result_length": d.get("result_length"),
            "exploit_attempts": d.get("exploit_attempts", 0),
            "session_id": d.get("session_id"),
            "task": d.get("task", ""),
            "kb_before": d.get("knowledge_base_before") or {},
            "kb_after": d.get("knowledge_base_after") or {},
        })
    return out


def load_session_files(data_dir: str = DEFAULT_DATA_DIR) -> list[dict]:
    """Load every per-session action file into a list of session dicts.

    Each: {session_id, category, task, actions:[...], n_steps}.
    """
    sessions = []
    for f in sorted(glob.glob(os.path.join(data_dir, "eval-*.jsonl"))):
        try:
            records = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        except (OSError, json.JSONDecodeError):
            continue
        actions = normalize_actions(records)
        if not actions:
            continue
        sid = actions[0]["session_id"]
        sessions.append({
            "session_id": sid,
            "category": category_of(sid),
            "task": actions[0]["task"],
            "actions": actions,
            "n_steps": len(actions),
            "file": os.path.basename(f),
        })
    return sessions


def load_outcomes(data_dir: str = DEFAULT_DATA_DIR) -> dict[str, dict]:
    """Map session_id -> outcome from session_end records in all_trajectories.jsonl."""
    outcomes: dict[str, dict] = {}
    path = os.path.join(data_dir, "all_trajectories.jsonl")
    if not os.path.exists(path):
        return outcomes
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("type") == "session_end":
            outcomes[d["session_id"]] = {
                "success": bool(d.get("session_success")),
                "flags": list(d.get("flags_captured") or []),
            }
    return outcomes


def solved(session: dict, outcomes: dict[str, dict]) -> bool:
    """Ground-truth: did this run capture a flag?

    Prefer an explicit session_end label; fall back to a 1.0 success_score in the
    trajectory (flag captured) so runs with no summary record still get labeled.
    """
    o = outcomes.get(session["session_id"])
    if o is not None:
        return o["success"] or bool(o["flags"])
    return any((a["score"] or 0) >= 1.0 for a in session["actions"])
