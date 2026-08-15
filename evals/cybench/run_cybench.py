#!/usr/bin/env python3
"""
Cybench evaluation harness for Filthy-Clanker  —  ⚠ UNTESTED SCAFFOLD.

Drives the SAME multi-agent graph as the NYU eval over Cybench tasks. It reuses the
tested core from evals/nyu_bench/run_eval.py (FlagCheckingMCPPool, run_challenge_headless,
MCP pool + Hexstrike bring-up) and only adds Cybench-specific loading (cybench_dataset.py),
prompt building, Docker bring-up, and flag scoring.

STATUS: written from Cybench's docs, NOT yet run. Two things must be validated on a real
Cybench clone + free GPU before trusting numbers:

  1. THE TARGET_HOST NETWORKING SEAM (the main open problem). Cybench's `target_host` is a
     docker-compose *service* hostname like "otp:80". Our agent runs inside the eval
     container and reaches services via `host.docker.internal:<published_port>`, so it
     CANNOT resolve "otp". Two fixes (pick one when we test):
       (a) Attach the agent/eval container to the task's compose network so the service
           hostname resolves as-is (join the network the task's start_docker.sh creates).
       (b) Ensure the task's compose PUBLISHES its port to the host, parse the published
           port (as NYU's _get_exposed_port does), and pass
           target = host.docker.internal:<published_port>.
     Until then, set CYBENCH_TARGET=host:port to hand the agent a reachable target manually.

  2. metadata.json field names (categories/target_host/subtasks[].answer) — parsed
     defensively in cybench_dataset.py, but confirm against the clone.

Usage (once a Cybench checkout exists):
    export CYBENCH_DIR=/path/to/cybench          # clone of andyzorigin/cybench
    python evals/cybench/run_cybench.py --max-tasks 3 --timeout 1200 --capture-sft
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
SRC_DIR = PROJECT_ROOT / "src"
NYU_DIR = PROJECT_ROOT / "evals" / "nyu_bench"
for p in (str(SRC_DIR), str(NYU_DIR), str(EVAL_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from dotenv import load_dotenv

# Reused, tested core from the NYU harness.
from run_eval import (  # type: ignore
    FlagCheckingMCPPool, run_challenge_headless, build_mcp_pool,
    ensure_hexstrike_running, _detect_compose_cmd, flags_match,
    load_config, _apply_model_overrides,
)
from graph import build_graph, create_checkpointer  # type: ignore
from data_capture import TrajectoryLogger  # type: ignore

from cybench_dataset import CybenchTask, find_tasks, build_cybench_task_prompt

log = logging.getLogger("cybench-eval")
logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")


def _resolve_target(task: CybenchTask) -> Optional[str]:
    """Best-effort target string for the agent. See the TARGET_HOST SEAM note in the
    module docstring — this currently trusts an explicit override or the raw metadata host,
    which likely needs docker-network bridging to actually be reachable."""
    override = os.getenv("CYBENCH_TARGET", "").strip()
    if override:
        return override
    th = task.target_host
    if th is None:
        return None
    host, port = th
    target = f"{host}:{port}" if port else host
    log.warning("Using raw Cybench target_host %r — may be unreachable from the eval "
                "container without docker-network bridging (see module docstring).", target)
    return target


async def evaluate_task(task: CybenchTask, graph, config, tlogger, *,
                        timeout_sec: int, provider, flag_pool, no_docker: bool) -> dict:
    session_id = f"cyb-{task.canonical_name[:48]}-{str(uuid.uuid4())[:8]}"
    flag = task.flag
    log.info("=" * 70)
    log.info("Task: %s  [%s]  session=%s", task.canonical_name, task.category, session_id)
    if not flag:
        log.warning("Task %s has no parseable flag (subtasks[-1].answer) — scoring will fail.",
                    task.canonical_name)
    flag_pool.set_challenge(flag)

    target: Optional[str] = None
    staged = Path(tempfile.mkdtemp(prefix=f"cyb_{task.canonical_name[:20]}_"))
    try:
        task.stage_files(staged)
        if task.has_docker and not no_docker:
            log.info("Starting Cybench docker for %s", task.canonical_name)
            task.start_docker()
            await asyncio.sleep(3)
            target = _resolve_target(task)

        prompt = build_cybench_task_prompt(task, staged, target, submit_flag_mode=flag_pool.mode)
        has_files = any(staged.iterdir()) if staged.exists() else False
        run_result = await run_challenge_headless(
            graph=graph, session_id=session_id, task=prompt, provider=provider, config=config,
            trajectory_logger=tlogger,
            target_ip=(target.split(":")[0] if target else ""),
            timeout_sec=timeout_sec, flag_pool=flag_pool,
            flag_format="", challenge_category=task.category, has_files=has_files,
            provided_files=[str(p) for p in staged.iterdir()] if has_files else None,
        )
        submitted = run_result["flags"]
        solved = run_result.get("solved_via_tool") or any(flags_match(f, flag) for f in submitted)
        result = {
            "task": task.canonical_name, "category": task.category, "difficulty": task.difficulty,
            "correct_flag": flag, "submitted_flags": submitted, "solved": solved,
            "solved_via_tool": run_result.get("solved_via_tool", False),
            "timed_out": run_result["timed_out"], "error": run_result["error"],
            "refusal_fired": run_result.get("refusal_fired", False),
            "target": target, "has_docker": task.has_docker, "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        log.info("Result: %s | submitted=%s | correct=%s",
                 "SOLVED" if solved else ("TIMEOUT" if run_result["timed_out"] else "FAILED"),
                 submitted, flag)
        return result
    finally:
        if task.has_docker and not no_docker:
            try:
                task.stop_docker()
            except Exception as exc:
                log.warning("stop_docker failed for %s: %s", task.canonical_name, exc)


async def main() -> None:
    load_dotenv(str(PROJECT_ROOT / ".env"), override=False)
    ap = argparse.ArgumentParser(description="Evaluate Filthy-Clanker against Cybench (UNTESTED)")
    ap.add_argument("--cybench-dir", default=os.getenv("CYBENCH_DIR", ""),
                    help="Path to a clone of andyzorigin/cybench (or set $CYBENCH_DIR).")
    ap.add_argument("--max-tasks", type=int, default=None)
    ap.add_argument("--only", default=None, help="Substring filter on canonical task name.")
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--provider", default=None, choices=["anthropic", "gemini", "ollama"])
    ap.add_argument("--no-docker", action="store_true")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--worker-model", default=None)
    ap.add_argument("--summarizer-model", default=None)
    ap.add_argument("--model-override", action="append", default=[], metavar="AGENT=TAG")
    ap.add_argument("--submit-flag-mode", default="gated", choices=["always", "gated", "off"])
    ap.add_argument("--gate-after", type=int, default=3)
    ap.add_argument("--capture-sft", action="store_true")
    ap.add_argument("--output-dir", default=str(EVAL_DIR / "results"))
    ap.add_argument("--db", default=str(EVAL_DIR / "cybench_checkpoints.db"))
    args = ap.parse_args()

    if args.capture_sft:
        os.environ["CLANKER_CAPTURE_SFT"] = "1"
        log.info("SFT capture ENABLED (CLANKER_CAPTURE_SFT=1).")

    bench_root = Path(args.cybench_dir) / "benchmark" if args.cybench_dir else None
    if not bench_root or not bench_root.exists():
        sys.exit(f"Cybench benchmark dir not found: {bench_root} — clone andyzorigin/cybench "
                 "and pass --cybench-dir or set $CYBENCH_DIR.")

    tasks = find_tasks(bench_root)
    if args.only:
        tasks = [t for t in tasks if args.only.lower() in t.canonical_name.lower()]
    if args.max_tasks:
        tasks = tasks[: args.max_tasks]
    log.info("Discovered %d Cybench task(s) to run.", len(tasks))

    config = load_config(PROJECT_ROOT / "agents.yaml", profile=args.profile)
    _apply_model_overrides(config, args)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"cyb-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{str(uuid.uuid4())[:6]}"
    tlogger = TrajectoryLogger(str(PROJECT_ROOT / "data" / "training"))

    global _COMPOSE
    _COMPOSE = _detect_compose_cmd()
    hexproc = ensure_hexstrike_running()
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
                        provider=args.provider, flag_pool=flag_pool, no_docker=args.no_docker))
                except Exception as exc:
                    log.error("Task %s crashed: %s", task.canonical_name, exc, exc_info=True)
                    results.append({"task": task.canonical_name, "solved": False, "error": str(exc)})
                (out_dir / f"{run_id}.jsonl").open("a", encoding="utf-8").write(
                    json.dumps(results[-1], ensure_ascii=False) + "\n")
    finally:
        await mcp_pool.disconnect()
        if hexproc:
            hexproc.terminate()

    solved = sum(1 for r in results if r.get("solved"))
    summary = {"run_id": run_id, "total": len(results), "solved": solved,
               "solve_rate": round(solved / len(results), 4) if results else 0.0,
               "models": getattr(args, "_effective_models", None), "results": results}
    (out_dir / f"{run_id}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                            encoding="utf-8")
    log.info("=" * 70)
    log.info("CYBENCH COMPLETE: %d / %d solved. Report: %s", solved, len(results),
             out_dir / f"{run_id}.json")


if __name__ == "__main__":
    asyncio.run(main())
