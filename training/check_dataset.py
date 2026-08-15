#!/usr/bin/env python3
"""
Inspect captured SFT data for training-readiness.

Reports, over everything in data/sft/: how many trajectories converted cleanly, how many
ASSISTANT turns (the training targets), rough token sizes (and how much is the tool schema),
and — critically — how many come from SOLVED sessions (the positives you actually want to
train on). Cross-references the eval run reports to label solved vs failed.

    python training/check_dataset.py
    python training/check_dataset.py --sft-dir data/sft --results-glob "evals/**/results/**/*.json"
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sft_common import iter_captured_records, record_to_example  # type: ignore


def _solved_sessions(results_glob: str) -> tuple[set, set]:
    """(solved_session_ids, all_session_ids_seen_in_reports) from eval run reports."""
    solved, seen = set(), set()
    for jf in glob.glob(results_glob, recursive=True):
        try:
            data = json.loads(Path(jf).read_text(encoding="utf-8"))
        except Exception:
            continue
        for r in data.get("results", []):
            sid = r.get("session_id")
            if not sid:
                continue
            seen.add(sid)
            if r.get("solved"):
                solved.add(sid)
    return solved, seen


def _example_tokens(ex: dict) -> tuple[int, int]:
    """Rough (total, tools_only) token estimate (chars/4) for a built example."""
    msg_chars = sum(len(json.dumps(m, default=str)) for m in ex.get("messages", []))
    tool_chars = len(json.dumps(ex.get("tools", []), default=str))
    return (msg_chars + tool_chars) // 4, tool_chars // 4


def main() -> None:
    ap = argparse.ArgumentParser(description="Check captured SFT data for training-readiness.")
    ap.add_argument("--sft-dir", default="data/sft")
    ap.add_argument("--results-glob", default="evals/**/results/**/*.json")
    args = ap.parse_args()

    solved, seen = _solved_sessions(args.results_glob)
    files = sorted(Path(args.sft_dir).glob("*.jsonl"))

    n_records = n_examples = asst_turns = 0
    sessions: set = set()
    toks: list[int] = []
    tool_toks: list[int] = []
    for rec in iter_captured_records(args.sft_dir):
        n_records += 1
        sessions.add(rec.get("session_id"))
        ex = record_to_example(rec)
        if not ex:
            continue
        n_examples += 1
        asst_turns += sum(1 for m in ex["messages"]
                          if m.get("role") == "assistant" and (m.get("content") or m.get("tool_calls")))
        t, tt = _example_tokens(ex)
        toks.append(t)
        tool_toks.append(tt)

    solved_here = {s for s in sessions if s in solved}

    print(f"session files             : {len(files)}")
    print(f"capture records           : {n_records}")
    print(f"convertible examples      : {n_examples}")
    print(f"total assistant turns     : {asst_turns}   (SFT training targets)")
    print(f"distinct sessions in data : {len(sessions)}")
    print(f"  SOLVED (positives)      : {len(solved_here)}")
    print(f"reports scanned           : {len(seen)} sessions ({len(solved)} solved overall)")
    if toks:
        print(f"example tokens (~chars/4) : min={min(toks)} median={int(statistics.median(toks))} max={max(toks)}")
        print(f"  of which tool schema    : median={int(statistics.median(tool_toks))} "
              f"({int(100*statistics.median(tool_toks)/max(1,statistics.median(toks)))}% of the example)")
    print()
    if n_examples == 0:
        print("VERDICT: no convertible examples — run an eval with --capture-sft first.")
    elif not solved_here:
        print("VERDICT: format is CORRECT and converts cleanly, but ZERO solved sessions --")
        print("  every captured trajectory is a FAILURE (weak-model flailing / guessing).")
        print("  Training on these would teach failure patterns. You need bootstrapped POSITIVES")
        print("  (training/ingest_ctfdojo.py, generate_htb_trajectories.py, or frontier) before SFT.")
        print("  The QLoRA pipeline is still worth building now -- it just needs a positive dataset.")
    else:
        print(f"VERDICT: {len(solved_here)} solved session(s) usable as positives "
              f"(build with: build_sft_dataset.py --report <run.json> --only-solved).")
        print("  Likely still too few for a full fine-tune; supplement with bootstrapped positives.")


if __name__ == "__main__":
    main()
