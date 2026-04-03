"""
Trajectory capture for fine-tuning dataset generation.

Records (State, Action, Result) triplets as JSONL to data/training/.
Each entry includes a `success_score` (0.0–1.0) reflecting whether the
action moved the agent closer to the objective.

Scoring heuristic:
  1.0  — Flag captured in this result
  0.85 — New credentials discovered
  0.70 — New attack surface (exploitable path/service found)
  0.55 — New open port or service discovered
  0.40 — New IP or host discovered
  0.20 — New directory/file found (gobuster/ferox style)
  0.05 — Tool ran successfully but no new findings
  0.0  — Tool error or explicit failure

Usage:
    from src.data_capture import TrajectoryLogger
    logger = TrajectoryLogger("data/training")
    logger.record(state_before, action, result, state_after)
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def compute_success_score(
    result: str,
    state_before: Dict[str, Any],
    state_after: Dict[str, Any],
) -> float:
    """
    Compute a 0.0–1.0 success score by diffing knowledge_base before/after
    and checking for error patterns in the tool result.

    Higher scores = more valuable training signal.
    """
    result_lower = result.lower()

    # Hard failure patterns
    failure_patterns = [
        "connection refused", "connection timed out", "no route to host",
        "error:", "failed:", "exception:", "traceback", "command not found",
        "permission denied", "access denied",
    ]
    for pat in failure_patterns:
        if pat in result_lower[:500]:
            return 0.0

    kb_before = state_before.get("knowledge_base", {})
    kb_after = state_after.get("knowledge_base", {})

    # 1.0: New flag
    flags_before = set(kb_before.get("flags", []))
    flags_after = set(kb_after.get("flags", []))
    if flags_after - flags_before:
        return 1.0

    # 0.85: New credentials
    creds_before = {json.dumps(c, sort_keys=True) for c in kb_before.get("credentials", [])}
    creds_after = {json.dumps(c, sort_keys=True) for c in kb_after.get("credentials", [])}
    if creds_after - creds_before:
        return 0.85

    # 0.70: New attack surface (exploitable)
    surface_before = set(kb_before.get("attack_surface", []))
    surface_after = set(kb_after.get("attack_surface", []))
    if surface_after - surface_before:
        return 0.70

    # 0.55: New ports
    ports_before: set = set()
    for ports in kb_before.get("open_ports", {}).values():
        ports_before.update(ports)
    ports_after: set = set()
    for ports in kb_after.get("open_ports", {}).values():
        ports_after.update(ports)
    if ports_after - ports_before:
        return 0.55

    # 0.40: New IPs
    ips_before = set(kb_before.get("ips", []))
    ips_after = set(kb_after.get("ips", []))
    if ips_after - ips_before:
        return 0.40

    # 0.20: Successful directory/file discovery (heuristic on result text)
    if re.search(r'status: 200|found:|discovered:', result_lower):
        return 0.20

    # 0.05: Tool ran but found nothing new
    return 0.05


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class TrajectoryLogger:
    """
    Appends trajectory records to a JSONL file in the training data directory.

    Each session gets its own file: <session_id>.jsonl
    A cumulative all_trajectories.jsonl is also maintained for convenience.
    """

    def __init__(self, output_dir: str = "data/training"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._session_file: Optional[Path] = None

    def set_session(self, session_id: str) -> None:
        """Set the active session so records are written to the correct file."""
        self._session_file = self.output_dir / f"{session_id}.jsonl"

    def record(
        self,
        state_before: Dict[str, Any],
        action: Dict[str, Any],
        result: str,
        state_after: Dict[str, Any],
        agent_name: str = "",
    ) -> float:
        """
        Record a (State, Action, Result) trajectory and compute success_score.

        Args:
            state_before: TeamState snapshot before the tool call.
            action:       Dict with keys: tool_name, arguments.
            result:       Raw (or summarized) tool output string.
            state_after:  TeamState snapshot after the tool call and KB update.
            agent_name:   Which agent executed this action.

        Returns:
            The computed success_score for this trajectory.
        """
        score = compute_success_score(result, state_before, state_after)

        record: Dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": state_before.get("session_id", "unknown"),
            "agent": agent_name,
            "task": state_before.get("task", ""),
            "action": {
                "tool_name": action.get("tool_name", ""),
                "arguments": action.get("arguments", {}),
            },
            "result_snippet": result[:2000],  # cap for file size
            "result_length": len(result),
            "knowledge_base_before": state_before.get("knowledge_base", {}),
            "knowledge_base_after": state_after.get("knowledge_base", {}),
            "success_score": score,
            "exploit_attempts": state_after.get("exploit_attempts", 0),
        }

        self._write(record)
        return score

    def record_session_end(self, final_state: Dict[str, Any]) -> None:
        """Write a final summary record at session end."""
        kb = final_state.get("knowledge_base", {})
        flags = kb.get("flags", [])
        session_id = final_state.get("session_id", "unknown")

        record: Dict[str, Any] = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "type": "session_end",
            "task": final_state.get("task", ""),
            "flags_captured": flags,
            "final_knowledge_base": kb,
            "exploit_attempts_total": final_state.get("exploit_attempts", 0),
            "context_tokens_final": final_state.get("context_token_estimate", 0),
            "session_success": bool(flags),
        }

        self._write(record)
        print(f"\n[DataCapture] Session end recorded. Flags: {flags or 'none'}")

    def _write(self, record: Dict[str, Any]) -> None:
        """Write a record to both the session file and the cumulative log."""
        line = json.dumps(record, ensure_ascii=False) + "\n"

        # Session-specific file
        if self._session_file:
            with open(self._session_file, "a", encoding="utf-8") as f:
                f.write(line)

        # Cumulative file
        cumulative = self.output_dir / "all_trajectories.jsonl"
        with open(cumulative, "a", encoding="utf-8") as f:
            f.write(line)

    def get_stats(self) -> Dict[str, Any]:
        """Return summary statistics for the current session."""
        if not self._session_file or not self._session_file.exists():
            return {"records": 0}

        records = []
        with open(self._session_file, encoding="utf-8") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

        trajectory_records = [r for r in records if r.get("type") != "session_end"]
        if not trajectory_records:
            return {"records": 0}

        scores = [r.get("success_score", 0) for r in trajectory_records]
        return {
            "records": len(trajectory_records),
            "avg_score": round(sum(scores) / len(scores), 3),
            "max_score": max(scores),
            "high_value": sum(1 for s in scores if s >= 0.7),
            "failures": sum(1 for s in scores if s == 0.0),
        }
