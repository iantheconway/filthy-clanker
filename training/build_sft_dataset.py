#!/usr/bin/env python3
r"""
Build an SFT dataset from captured trajectories (data/sft/).

Positives only (recommended — auto-unions the solved sessions from every eval run report):
    python training/build_sft_dataset.py --only-solved --out training/data/sft_positives.jsonl

Balanced set (positives + an equal share of sampled negatives, for contrastive signal):
    python training/build_sft_dataset.py --only-solved --neg-ratio 1.0 \
        --out training/data/sft_balanced.jsonl

Include frontier (Claude/GPT/Gemini) trajectories too — ToS-gated:
    ... --include-frontier --yes-i-accept-tos-risk

Output: one OpenAI chat-with-tools JSON object per line (see training/sft_common.py). Feed it to
train_qlora.py (local/rented GPU) or a managed fine-tuner (Together) with a tool-aware chat
template and assistant-only loss masking. Run from the repo root so --reports-glob resolves.
"""
from __future__ import annotations

import argparse
import glob as _glob
import random as _random
import re as _re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sft_common import (  # type: ignore
    iter_captured_records, record_to_example, solved_sessions, strip_hint, write_jsonl,
)

# Sentinel wrapping an injected HTB walkthrough hint (see generate_htb_trajectories.py).
HINT_START = "=== WALKTHROUGH HINT (capture-only; do NOT learn to depend on this) ==="
HINT_END = "=== END WALKTHROUGH HINT ==="

# Proprietary FRONTIER models whose provider ToS restricts using their OUTPUTS to train competing
# models (Anthropic / OpenAI / Gemini). Excluded by default (ToS-clean); --include-frontier (with
# --yes-i-accept-tos-risk) keeps them. Open-WEIGHT models are always clean: Qwen / DeepSeek / Llama
# via Together, local Ollama, and OpenAI's *open* gpt-oss-* (Apache-2.0). Matches "gpt-<digit>"
# (gpt-4 / gpt-4o / gpt-3.5) but deliberately NOT "gpt-oss". See docs/CLOUD_TRAINING.md.
_FRONTIER_MODEL_RE = _re.compile(r"\bclaude|gpt-[0-9]|gemini|\bo1\b|\bo3\b|davinci", _re.IGNORECASE)


def _is_frontier(model: str) -> bool:
    return bool(_FRONTIER_MODEL_RE.search(model or ""))


def _discover_solved(reports: list[str] | None, glob_pat: str | None) -> tuple[set, list[str]]:
    """Union the solved session ids from explicit --report files and/or a glob of run reports."""
    paths: list[str] = []
    for p in (reports or []):
        if p not in paths:
            paths.append(p)
    if glob_pat:
        for pat in glob_pat.split(","):
            pat = pat.strip()
            if not pat:
                continue
            for p in sorted(_glob.glob(pat, recursive=True)):
                if p not in paths:
                    paths.append(p)
    solved: set = set()
    used: list[str] = []
    for p in paths:
        try:
            solved |= solved_sessions(p)
            used.append(p)
        except Exception as exc:
            print(f"[build] WARN: could not read report {p}: {exc}")
    return solved, used


def _prep_record(rec: dict, agent_filter, include_frontier: bool, strip_htb: bool):
    """Apply agent + frontier + hint filters. Returns (record_or_None, is_frontier)."""
    if agent_filter and rec.get("agent") not in agent_filter:
        return None, False
    is_front = _is_frontier(rec.get("model", ""))
    if is_front and not include_frontier:
        return None, True
    if strip_htb:
        rec = dict(rec)
        rec["system_prompt"] = strip_hint(rec.get("system_prompt", ""), HINT_START, HINT_END)
        rec["messages"] = [
            {**m, "content": strip_hint(m["content"], HINT_START, HINT_END)}
            if isinstance(m, dict) and isinstance(m.get("content"), str) else m
            for m in rec.get("messages", [])
        ]
    return rec, is_front


def main() -> None:
    ap = argparse.ArgumentParser(description="Build an SFT dataset from captured trajectories.")
    ap.add_argument("--sft-dir", default="data/sft", help="Directory of capture JSONL (data/sft).")
    ap.add_argument("--report", action="append", default=None,
                    help="Run-report JSON marking solved sessions (repeatable). A session solved "
                         "in ANY report counts as a positive.")
    ap.add_argument("--reports-glob", default="evals/**/run-*.json,evals/**/ic-*.json",
                    help="Comma-separated globs to auto-union run reports (default: NYU run-*.json "
                         "and InterCode ic-*.json). Pass '' to disable auto-discovery.")
    ap.add_argument("--only-solved", action="store_true",
                    help="Keep only trajectories from SOLVED sessions (positives). Strongly "
                         "recommended — negatives teach the model to fail.")
    ap.add_argument("--neg-ratio", type=float, default=0.0,
                    help="Balanced set: also include up to N negatives per positive (0 = positives "
                         "only; 1.0 ≈ 50/50). Negatives are sampled from non-solved sessions.")
    ap.add_argument("--seed", type=int, default=3407, help="Shuffle seed for negative sampling/order.")
    ap.add_argument("--exclude-challenges", default=None,
                    help="File of canonical challenge names (one per line) to EXCLUDE — e.g. a "
                         "held-out eval set, so we never train on what we evaluate. Matched as a "
                         "substring of each record's session_id.")
    ap.add_argument("--agents", default=None,
                    help="Comma-separated agent names to keep (e.g. 'exploit,reversing').")
    ap.add_argument("--min-assistant-turns", type=int, default=1,
                    help="Drop trajectories with fewer than N assistant turns.")
    ap.add_argument("--max-tool-chars", type=int, default=0,
                    help="Condense TOOL-result messages longer than N chars (head+tail, marker in "
                         "between). 0 = off. Tool output is ~76%% of tokens and the harness summarizes "
                         "it at inference anyway, so capping shrinks examples AND matches inference.")
    ap.add_argument("--strip-htb-hint", action="store_true",
                    help="Remove the injected walkthrough hint block from prompts (HTB path).")
    ap.add_argument("--no-tools", action="store_true",
                    help="Omit the tool schema from examples (smaller; only if training text-only).")
    ap.add_argument("--include-frontier", action="store_true",
                    help="Include PROPRIETARY frontier models (claude/gpt/gemini). OFF by default "
                         "(ToS-clean). Requires --yes-i-accept-tos-risk (docs/CLOUD_TRAINING.md).")
    ap.add_argument("--yes-i-accept-tos-risk", action="store_true",
                    help="Acknowledge the ToS risk of including frontier-model trajectories.")
    ap.add_argument("--out", default="training/data/sft_from_capture.jsonl")
    args = ap.parse_args()

    if args.include_frontier and not args.yes_i_accept_tos_risk:
        ap.error("--include-frontier requires --yes-i-accept-tos-risk (Anthropic/OpenAI/Gemini "
                 "restrict training competing models on their outputs; see docs/CLOUD_TRAINING.md).")

    need_solved = args.only_solved or args.neg_ratio > 0
    solved = None
    if need_solved:
        solved, used = _discover_solved(args.report, args.reports_glob or None)
        print(f"[build] {len(solved)} solved session(s) unioned from {len(used)} report(s)")
        if not solved:
            print("[build] WARNING: no solved sessions found — check --reports-glob and that cwd "
                  "is the repo root. See docs/SFT_PLAN.md.")

    agent_filter = {a.strip() for a in args.agents.split(",")} if args.agents else None
    exclude = set()
    if args.exclude_challenges:
        exclude = {l.strip() for l in open(args.exclude_challenges, encoding="utf-8")
                   if l.strip() and not l.startswith("#")}
        print(f"[build] excluding {len(exclude)} held-out challenge(s) from training")

    pos, neg = [], []
    skipped = skipped_frontier = incl_frontier = 0
    for rec in iter_captured_records(args.sft_dir, sessions=None):
        sid = rec.get("session_id", "")
        if exclude and any(x in sid for x in exclude):
            continue
        rec2, is_front = _prep_record(rec, agent_filter, args.include_frontier, args.strip_htb_hint)
        if rec2 is None:
            if is_front:
                skipped_frontier += 1
            continue
        is_pos = (solved is None) or (rec.get("session_id") in solved)
        if not is_pos and args.neg_ratio <= 0:
            continue  # positives-only: don't bother converting negatives
        ex = record_to_example(rec2, include_tools=not args.no_tools)
        if ex is None:
            skipped += 1
            continue
        if args.max_tool_chars > 0:
            cap, head = args.max_tool_chars, int(args.max_tool_chars * 0.7)
            for m in ex["messages"]:
                c = m.get("content")
                if m.get("role") == "tool" and isinstance(c, str) and len(c) > cap:
                    m["content"] = (c[:head] + f"\n…[{len(c) - cap} chars truncated]…\n" + c[-(cap - head):])
        if sum(1 for m in ex["messages"] if m.get("role") == "assistant") < args.min_assistant_turns:
            skipped += 1
            continue
        if is_front:
            incl_frontier += 1
        (pos if is_pos else neg).append(ex)

    examples = list(pos)
    if solved is not None and args.neg_ratio > 0 and neg:
        k = min(len(neg), int(round(args.neg_ratio * len(pos))))
        _random.Random(args.seed).shuffle(neg)
        examples += neg[:k]
    _random.Random(args.seed).shuffle(examples)

    n = write_jsonl(args.out, examples)
    _ftxt = (f"INCLUDED {incl_frontier} frontier example(s)" if args.include_frontier
             else f"excluded {skipped_frontier} frontier example(s) as ToS-restricted")
    _mix = (f"positives={len(pos)} negatives_added={len(examples) - len(pos)}"
            if solved is not None else f"all-records={len(examples)} (no solved filter)")
    print(f"[build] wrote {n} -> {args.out}  ({_mix}, dropped={skipped}); {_ftxt}")


if __name__ == "__main__":
    main()
