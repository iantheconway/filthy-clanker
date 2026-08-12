#!/usr/bin/env python3
"""
Ingest externally-released CTF trajectories (CTF-Dojo / Cyber-Zero, or any SWE-agent /
ShareGPT-style dump) into our SFT format.

⚠ FORMAT + LICENSE NOTES
  • CTF-Dojo's framework is **CC-BY-NC-4.0** (non-commercial research OK, NOT commercial).
    Confirm the license on the specific trajectory DATA you download, and attribute.
    Repos: amazon-science/CTF-Dojo (2508.18370), amazon-science/Cyber-Zero (2508.00910).
  • The exact on-disk schema isn't documented on the repo landing page (trajectories are
    EnIGMA+/SWE-agent lineage). So this ingester is **format-detecting** and has an
    `--inspect` mode: run it first to see the shape, then it maps the common cases:
        (a) already OpenAI chat:   {"messages":[{role,content,tool_calls?}, ...]}
        (b) ShareGPT:              {"conversations":[{"from":"human/gpt/system","value":...}]}
        (c) SWE-agent .traj:       {"history":[{role,content, ...}]} or {"trajectory":[...]}
    If your download doesn't match, tell me the keys from --inspect and I'll add a mapper.

Usage:
    python training/ingest_ctfdojo.py --inspect --in path/to/trajectories.jsonl
    python training/ingest_ctfdojo.py --in path/to/trajectories.jsonl --out training/data/ctfdojo.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sft_common import write_jsonl  # type: ignore

_ROLE_MAP = {"human": "user", "gpt": "assistant", "system": "system", "tool": "tool",
             "observation": "tool", "user": "user", "assistant": "assistant"}


def _load_any(path: Path) -> list:
    """Load JSONL (one obj/line) or a single JSON array/object."""
    text = path.read_text(encoding="utf-8")
    rows = []
    stripped = text.lstrip()
    if stripped.startswith("["):
        return json.loads(text)
    if stripped.startswith("{") and "\n" not in stripped.rstrip():
        return [json.loads(text)]
    for line in text.splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def _to_messages(rec: dict) -> list[dict] | None:
    """Map one external record to an OpenAI-style messages list, or None if unrecognised."""
    # (a) already messages
    if isinstance(rec.get("messages"), list):
        return rec["messages"]
    # (b) ShareGPT conversations
    if isinstance(rec.get("conversations"), list):
        out = []
        for turn in rec["conversations"]:
            role = _ROLE_MAP.get(str(turn.get("from", "")).lower(), "user")
            out.append({"role": role, "content": turn.get("value", "")})
        return out
    # (c) SWE-agent history/trajectory
    for key in ("history", "trajectory", "traj"):
        seq = rec.get(key)
        if isinstance(seq, list):
            out = []
            for turn in seq:
                if not isinstance(turn, dict):
                    continue
                role = _ROLE_MAP.get(str(turn.get("role", "")).lower(), turn.get("role", "user"))
                out.append({"role": role, "content": turn.get("content", turn.get("value", ""))})
            return out
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest CTF-Dojo/Cyber-Zero/SWE-agent trajectories → SFT.")
    ap.add_argument("--in", dest="inp", required=True, help="Path to a trajectory JSON/JSONL file.")
    ap.add_argument("--out", default="training/data/ctfdojo.jsonl")
    ap.add_argument("--inspect", action="store_true",
                    help="Print top-level keys + the first record's shape and exit (no conversion).")
    ap.add_argument("--min-assistant-turns", type=int, default=1)
    args = ap.parse_args()

    rows = _load_any(Path(args.inp))
    print(f"[ingest] loaded {len(rows)} record(s) from {args.inp}")
    if not rows:
        sys.exit("No records loaded.")

    if args.inspect:
        first = rows[0]
        print("[inspect] first record top-level keys:", list(first.keys()) if isinstance(first, dict) else type(first))
        msgs = _to_messages(first) if isinstance(first, dict) else None
        print("[inspect] detected messages:", "yes" if msgs else "NO — schema not recognised")
        if msgs:
            print("[inspect] first 3 turns:", json.dumps(msgs[:3], ensure_ascii=False)[:800])
        else:
            print("[inspect] paste these keys to have a mapper added:",
                  list(first.keys()) if isinstance(first, dict) else first)
        return

    examples, skipped = [], 0
    for rec in rows:
        msgs = _to_messages(rec) if isinstance(rec, dict) else None
        if not msgs:
            skipped += 1
            continue
        if sum(1 for m in msgs if isinstance(m, dict) and m.get("role") == "assistant") < args.min_assistant_turns:
            skipped += 1
            continue
        ex = {"messages": msgs}
        if isinstance(rec.get("tools"), list):
            ex["tools"] = rec["tools"]
        examples.append(ex)

    n = write_jsonl(args.out, examples)
    print(f"[ingest] wrote {n} example(s) → {args.out}  (skipped {skipped} unrecognised)")
    if skipped and not examples:
        print("[ingest] Nothing converted — run with --inspect and share the keys.")


if __name__ == "__main__":
    main()
