#!/usr/bin/env python3
"""
InterCode-CTF evaluation harness for Filthy-Clanker.

Runs the 100 picoCTF tasks (princeton-nlp/intercode, data/ctf/) through the SAME multi-agent
graph as the NYU eval, reusing its tested core (FlagCheckingMCPPool, run_challenge_headless,
MCP pool + Hexstrike bring-up). InterCode-CTF is the EASIEST benchmark we run — picked so the
local models actually land some solves and the metrics have resolution (NYU/CTFTiny are 0/6,
all noise). Most tasks are file-based, so no per-task Docker is needed.

Setup:
    git clone https://github.com/princeton-nlp/intercode /path/to/intercode
    export INTERCODE_DIR=/path/to/intercode
    # (optional) fetch the large `setup`-wget assets ON THE HOST first — the eval sandbox has
    # no internet: python evals/intercode/fetch_assets.py

Run (file-based subset — the default; ~75 tasks with local assets, nc-service tasks skipped):
    python evals/intercode/run_intercode.py --max-tasks 10 --timeout 600 --capture-sft
    python evals/intercode/run_intercode.py --tag crypto --worker-model gemma4:26b

Same --worker-model / --model-override / --capture-sft levers as run_eval.py, so model A/B
runs and SFT capture work identically.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent.parent
for p in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "evals" / "nyu_bench"), str(EVAL_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from dotenv import load_dotenv

from run_eval import (  # type: ignore
    FlagCheckingMCPPool, run_challenge_headless, build_mcp_pool,
    ensure_hexstrike_running, load_config, _apply_model_overrides, flags_match,
)
from graph import build_graph, create_checkpointer  # type: ignore
from data_capture import TrajectoryLogger  # type: ignore

from intercode_dataset import load_tasks, build_intercode_prompt, InterCodeTask

log = logging.getLogger("intercode-eval")
logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

CATEGORIES = ["crypto", "rev", "pwn", "web", "forensics", "misc"]


async def evaluate_task(task: InterCodeTask, graph, config, tlogger, *,
                        timeout_sec: int, provider, flag_pool) -> dict:
    session_id = f"ic-{int(task.task_id):03d}-{str(uuid.uuid4())[:8]}"
    log.info("=" * 70)
    log.info("Task %s: %s  [%s]  session=%s", task.task_id, task.canonical_name,
             task.category, session_id)
    flag_pool.set_challenge(task.flag)

    staged_dir = Path(tempfile.mkdtemp(prefix=f"ic_{int(task.task_id):03d}_"))
    try:
        staged = task.stage_files(staged_dir)
        prompt = build_intercode_prompt(task, staged, submit_flag_mode=flag_pool.mode)
        run_result = await run_challenge_headless(
            graph=graph, session_id=session_id, task=prompt, provider=provider, config=config,
            trajectory_logger=tlogger, target_ip="", timeout_sec=timeout_sec, flag_pool=flag_pool,
            flag_format=task.flag_format, challenge_category=task.category, has_files=bool(staged),
            provided_files=[str(p) for p in staged] or None,
        )
        submitted = run_result["flags"]
        solved = run_result.get("solved_via_tool") or any(flags_match(f, task.flag) for f in submitted)
        result = {
            "task_id": task.task_id, "challenge": task.canonical_name, "category": task.category,
            "tags": task.tags, "correct_flag": task.flag, "submitted_flags": submitted,
            "solved": solved, "solved_via_tool": run_result.get("solved_via_tool", False),
            "timed_out": run_result["timed_out"], "error": run_result["error"],
            "refusal_fired": run_result.get("refusal_fired", False),
            "n_files": len(staged), "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        log.info("Result: %s | submitted=%s | correct=%s",
                 "SOLVED" if solved else ("TIMEOUT" if run_result["timed_out"] else "FAILED"),
                 submitted, task.flag)
        return result
    finally:
        try:
            import shutil
            shutil.rmtree(staged_dir, ignore_errors=True)
        except Exception:
            pass


def _summarize(results: list[dict]) -> dict:
    total = len(results)
    solved = sum(1 for r in results if r.get("solved"))
    by_cat: dict = {}
    for r in results:
        c = r.get("category", "misc")
        by_cat.setdefault(c, {"total": 0, "solved": 0})
        by_cat[c]["total"] += 1
        by_cat[c]["solved"] += 1 if r.get("solved") else 0
    return {
        "total": total, "solved": solved,
        "solve_rate": round(solved / total, 4) if total else 0.0,
        "by_category": by_cat,
    }


async def main() -> None:
    load_dotenv(str(PROJECT_ROOT / ".env"), override=False)
    ap = argparse.ArgumentParser(description="Evaluate Filthy-Clanker against InterCode-CTF (picoCTF).")
    ap.add_argument("--intercode-dir", default=os.getenv("INTERCODE_DIR", ""),
                    help="Path to a clone of princeton-nlp/intercode (or set $INTERCODE_DIR).")
    ap.add_argument("--max-tasks", type=int, default=None)
    ap.add_argument("--tag", default=None, choices=CATEGORIES, help="Only this category.")
    ap.add_argument("--only", default=None, help="Comma-separated task_ids to run (e.g. '0,2,5').")
    ap.add_argument("--include-service", action="store_true",
                    help="Also run tasks that need a live `nc` service (will fail under the egress "
                         "sandbox — off by default).")
    ap.add_argument("--allow-missing-assets", action="store_true",
                    help="Run tasks even when their local assets are missing (default: skip them).")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--provider", default=None, choices=["anthropic", "gemini", "ollama"])
    ap.add_argument("--worker-model", default=None)
    ap.add_argument("--summarizer-model", default=None)
    ap.add_argument("--model-override", action="append", default=[], metavar="AGENT=TAG")
    ap.add_argument("--profile", default="nyu-ctf")
    ap.add_argument("--submit-flag-mode", default="gated", choices=["always", "gated", "off"])
    ap.add_argument("--gate-after", type=int, default=3)
    ap.add_argument("--capture-sft", action="store_true")
    ap.add_argument("--output-dir", default=str(EVAL_DIR / "results"))
    ap.add_argument("--db", default=str(EVAL_DIR / "intercode_checkpoints.db"))
    args = ap.parse_args()

    if args.capture_sft:
        os.environ["CLANKER_CAPTURE_SFT"] = "1"
        log.info("SFT capture ENABLED (CLANKER_CAPTURE_SFT=1).")
    if not args.intercode_dir:
        sys.exit("No InterCode dir — clone princeton-nlp/intercode and pass --intercode-dir or set $INTERCODE_DIR.")

    tasks = load_tasks(args.intercode_dir)
    log.info("Loaded %d InterCode-CTF tasks.", len(tasks))

    # Filter: optional task-id list, tag, service exclusion, missing-asset exclusion.
    if args.only:
        wanted = {int(s) for s in args.only.split(",") if s.strip().isdigit()}
        tasks = [t for t in tasks if t.task_id in wanted]
    if args.tag:
        tasks = [t for t in tasks if t.category == args.tag]
    if not args.include_service:
        svc = [t for t in tasks if t.needs_service]
        if svc:
            log.info("Skipping %d nc-service task(s) (need internet; use --include-service to force).",
                     len(svc))
        tasks = [t for t in tasks if not t.needs_service]
    if not args.allow_missing_assets:
        missing = [t for t in tasks if not t.has_assets()]
        if missing:
            log.warning("Skipping %d task(s) with NO local assets: %s (run fetch_assets.py on the "
                        "host, or --allow-missing-assets).", len(missing),
                        ", ".join(str(t.task_id) for t in missing[:20]))
        tasks = [t for t in tasks if t.has_assets()]
    if args.max_tasks:
        tasks = tasks[: args.max_tasks]
    log.info("Running %d task(s).", len(tasks))
    if not tasks:
        sys.exit("No runnable tasks after filtering.")

    config = load_config(PROJECT_ROOT / "agents.yaml", profile=args.profile)
    _apply_model_overrides(config, args)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"ic-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{str(uuid.uuid4())[:6]}"
    tlogger = TrajectoryLogger(str(PROJECT_ROOT / "data" / "training"))

    ensure_hexstrike_running()
    mcp_pool = await build_mcp_pool()
    flag_pool = FlagCheckingMCPPool(mcp_pool, mode=args.submit_flag_mode, gate_after=args.gate_after)

    results: list[dict] = []
    try:
        async with create_checkpointer(args.db) as checkpointer:
            graph = build_graph(flag_pool, config, checkpointer)
            for i, task in enumerate(tasks, 1):
                log.info("Task %d/%d", i, len(tasks))
                try:
                    results.append(await evaluate_task(
                        task, graph, config, tlogger, timeout_sec=args.timeout,
                        provider=args.provider, flag_pool=flag_pool))
                except Exception as exc:
                    log.error("Task %s crashed: %s", task.task_id, exc, exc_info=True)
                    results.append({"task_id": task.task_id, "solved": False, "error": str(exc)})
                (out_dir / f"{run_id}.jsonl").open("a", encoding="utf-8").write(
                    json.dumps(results[-1], ensure_ascii=False) + "\n")
    finally:
        await mcp_pool.disconnect()

    summary = _summarize(results)
    report = {"run_id": run_id, "timestamp": datetime.now(timezone.utc).isoformat(),
              "config": {"tag": args.tag, "worker_model": args.worker_model,
                         "models": getattr(args, "_effective_models", None),
                         "profile": args.profile, "timeout_sec": args.timeout},
              "summary": summary, "results": results}
    (out_dir / f"{run_id}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                            encoding="utf-8")
    log.info("=" * 70)
    log.info("INTERCODE COMPLETE: %d / %d solved (%.1f%%)  by-cat=%s",
             summary["solved"], summary["total"], 100 * summary["solve_rate"],
             {c: f"{v['solved']}/{v['total']}" for c, v in summary["by_category"].items()})
    log.info("Report: %s", out_dir / f"{run_id}.json")


if __name__ == "__main__":
    asyncio.run(main())
