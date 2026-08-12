#!/usr/bin/env python3
r"""
Build an SFT dataset from captured trajectories (data/sft/) — the shared backend for the
HTB and frontier bootstrap paths (both write data/sft/ via --capture-sft).

    python training/build_sft_dataset.py \
        --sft-dir data/sft \
        --report evals/nyu_bench/results/run-XXage.json \   # optional: keep only SOLVED sessions
        --only-solved \
        --strip-htb-hint \                                   # remove injected walkthrough hints
        --out training/data/sft_from_capture.jsonl

Output: one OpenAI chat-with-tools JSON object per line (see training/sft_common.py). Point
your trainer (axolotl / unsloth / trl) at it with a tool-aware chat template and assistant-
only loss masking.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sft_common import (  # type: ignore
    iter_captured_records, record_to_example, solved_sessions, strip_hint, write_jsonl,
)

# Sentinel wrapping an injected HTB walkthrough hint (see generate_htb_trajectories.py).
HINT_START = "=== WALKTHROUGH HINT (capture-only; do NOT learn to depend on this) ==="
HINT_END = "=== END WALKTHROUGH HINT ==="


def main() -> None:
    ap = argparse.ArgumentParser(description="Build an SFT dataset from captured trajectories.")
    ap.add_argument("--sft-dir", default="data/sft", help="Directory of capture JSONL (data/sft).")
    ap.add_argument("--report", default=None,
                    help="Run-report JSON; with --only-solved, keep only its solved sessions.")
    ap.add_argument("--only-solved", action="store_true",
                    help="Keep only trajectories from sessions the report marked solved. "
                         "STRONGLY recommended — negatives teach the model to fail.")
    ap.add_argument("--agents", default=None,
                    help="Comma-separated agent names to keep (e.g. 'exploit,reversing').")
    ap.add_argument("--min-assistant-turns", type=int, default=1,
                    help="Drop trajectories with fewer than N assistant turns.")
    ap.add_argument("--strip-htb-hint", action="store_true",
                    help="Remove the injected walkthrough hint block from prompts (HTB path).")
    ap.add_argument("--no-tools", action="store_true",
                    help="Omit the tool schema from examples (smaller; only if training text-only).")
    ap.add_argument("--out", default="training/data/sft_from_capture.jsonl")
    args = ap.parse_args()

    sessions = None
    if args.only_solved:
        if not args.report:
            ap.error("--only-solved requires --report.")
        sessions = solved_sessions(args.report)
        print(f"[build] {len(sessions)} solved session(s) from {args.report}")
        if not sessions:
            print("[build] WARNING: no solved sessions — output will be empty. "
                  "Bootstrap positives first (see docs/SFT_PLAN.md).")

    agent_filter = {a.strip() for a in args.agents.split(",")} if args.agents else None

    examples, kept, skipped = [], 0, 0
    for rec in iter_captured_records(args.sft_dir, sessions=sessions):
        if agent_filter and rec.get("agent") not in agent_filter:
            continue
        if args.strip_htb_hint:
            rec = dict(rec)
            rec["system_prompt"] = strip_hint(rec.get("system_prompt", ""), HINT_START, HINT_END)
            rec["messages"] = [
                {**m, "content": strip_hint(m["content"], HINT_START, HINT_END)}
                if isinstance(m, dict) and isinstance(m.get("content"), str) else m
                for m in rec.get("messages", [])
            ]
        ex = record_to_example(rec, include_tools=not args.no_tools)
        if ex is None:
            skipped += 1
            continue
        n_asst = sum(1 for m in ex["messages"] if m.get("role") == "assistant")
        if n_asst < args.min_assistant_turns:
            skipped += 1
            continue
        examples.append(ex)
        kept += 1

    n = write_jsonl(args.out, examples)
    print(f"[build] wrote {n} example(s) → {args.out}  (kept={kept}, skipped={skipped})")


if __name__ == "__main__":
    main()
