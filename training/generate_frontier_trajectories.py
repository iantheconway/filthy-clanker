#!/usr/bin/env python3
"""
Frontier / strong-model distillation — generate positive trajectories by running a STRONGER
model through the exact same harness, capturing, and filtering to solves.

    python training/generate_frontier_trajectories.py --model deepseek-v3 --provider ollama --subset ctftiny
    python training/generate_frontier_trajectories.py --provider anthropic --model claude-opus-5 --subset ctftiny

It orchestrates: `run_eval.py … --capture-sft` (with the chosen model) → then
`build_sft_dataset.py --only-solved` on the run's report. Same MCP tools ⇒ the captured
tool-call format matches deployment, so the small model can imitate it.

════════════════════════════════════════════════════════════════════════════════════════
⚠ ToS — READ BEFORE POINTING THIS AT A FRONTIER API
   Anthropic, OpenAI, and Google Gemini all restrict using their **Outputs to train models
   that compete with them** (Anthropic permits narrow "specialized tools" but prohibits
   general/open-ended-generation models; the others are broadly similar). Fine-tuning a
   general open-weight LLM (e.g. Qwen3-32B) on their outputs sits in / near the prohibited
   zone and needs their written permission if it does. Gemini's UNPAID tier additionally
   uses+reviews your inputs/outputs. See docs/SFT_PLAN.md.
   ToS-CLEAN options this script also supports (recommended):
     • --provider ollama --model <strong OPEN-WEIGHT> (deepseek-v3, qwen3:235b, llama-405b …)
       — permissive licenses generally allow training on outputs.
     • Or skip generation entirely and reuse released CTF-Dojo/Cyber-Zero data
       (training/ingest_ctfdojo.py; that data is CC-BY-NC — non-commercial).
   Choosing a frontier API here is YOUR call and YOUR account's ToS to honor.
════════════════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_EVAL = PROJECT_ROOT / "evals" / "nyu_bench" / "run_eval.py"
BUILDER = Path(__file__).resolve().parent / "build_sft_dataset.py"

_FRONTIER_API = {"anthropic", "gemini"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate + collect frontier/strong-model SFT trajectories.")
    ap.add_argument("--provider", default="ollama", choices=["anthropic", "gemini", "ollama"])
    ap.add_argument("--model", default=None,
                    help="For ollama: a strong OPEN-WEIGHT tag (--worker-model). For anthropic/"
                         "gemini: the API model id (--provider).")
    ap.add_argument("--subset", default="ctftiny", choices=["ctftiny"])
    ap.add_argument("--split", default=None, choices=["development", "test"],
                    help="Use a whole split instead of --subset (mutually exclusive).")
    ap.add_argument("--max-chals", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--profile", default="nyu-ctf")
    ap.add_argument("--out", default="training/data/frontier.jsonl")
    ap.add_argument("--yes-i-accept-tos-risk", action="store_true",
                    help="Required to run against a frontier API (anthropic/gemini).")
    ap.add_argument("--dry-run", action="store_true", help="Print the commands without running.")
    args = ap.parse_args()

    if args.provider in _FRONTIER_API and not args.yes_i_accept_tos_risk:
        sys.exit("Refusing to distill from a frontier API without --yes-i-accept-tos-risk "
                 "(see the ToS block in this file / docs/SFT_PLAN.md). Prefer a strong "
                 "open-weight model: --provider ollama --model deepseek-v3.")

    # Build the eval command.
    cmd = [sys.executable, str(RUN_EVAL), "--capture-sft", "--timeout", str(args.timeout),
           "--profile", args.profile]
    if args.split:
        cmd += ["--split", args.split]
    else:
        cmd += ["--subset", args.subset]
    if args.max_chals:
        cmd += ["--max-chals", str(args.max_chals)]
    if args.provider in _FRONTIER_API:
        cmd += ["--provider", args.provider]
        if args.model:
            cmd += ["--model-override", f"supervisor={args.model}"]  # provider default handles the rest
    elif args.model:
        cmd += ["--worker-model", args.model]

    out_dir = PROJECT_ROOT / "evals" / "nyu_bench" / "results"
    print("[frontier] eval command:\n  " + " ".join(cmd))
    if args.dry_run:
        print("[frontier] --dry-run: not executing.")
        return

    before = set(glob.glob(str(out_dir / "*.json")))
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
    after = set(glob.glob(str(out_dir / "*.json")))
    new_reports = sorted(after - before, key=os.path.getmtime)
    if not new_reports:
        sys.exit("[frontier] No new run report found in results/ — cannot filter to solves.")
    report = new_reports[-1]
    print(f"[frontier] run report: {report}")

    build_cmd = [sys.executable, str(BUILDER), "--sft-dir", "data/sft",
                 "--report", report, "--only-solved", "--out", args.out]
    print("[frontier] build command:\n  " + " ".join(build_cmd))
    subprocess.run(build_cmd, check=True, cwd=str(PROJECT_ROOT))
    print(f"[frontier] done → {args.out}")


if __name__ == "__main__":
    main()
