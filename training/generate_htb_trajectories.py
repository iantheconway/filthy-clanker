#!/usr/bin/env python3
"""
HTB walkthrough-guided self-distillation  —  ⚠ UNTESTED (needs HTB VPN + token + GPU).

Bootstrap POSITIVE trajectories the base model can't produce alone: spawn an HTB machine,
inject its official walkthrough as a capture-only HINT, let the (local) model drive the real
MCP tools to the flag, and capture the run in our SFT format. At dataset-build time the hint
is stripped (`build_sft_dataset.py --strip-htb-hint`) so the model learns to produce the
trajectory WITHOUT the crutch (STaR / rejection-sampling style).

Why this path: it generates positives in our EXACT harness + tool space (no distribution
shift), and targets the recon→exploit→privesc / multi-step / web-service skills that are the
30B's documented weak spot. It uses only your own local model — no frontier API, so no API
ToS question (HTB walkthrough content is HTB's IP: use it as a private solving aid, keep only
the generated tool-call trajectories, don't redistribute the walkthrough text).

Prereqs: HTB VPN connected; HTB app token in env (HTB_TOKEN or HTB_APP_TOKEN); Ollama up;
Hexstrike present. Domain note: HTB is pentest/host-compromise, so this trains web/pwn/priv-esc,
not CTF crypto/rev/forensics.

Usage:
    export HTB_TOKEN=...            # HTB app token
    python training/generate_htb_trajectories.py --machine Lame --walkthrough walkthroughs/lame.md
    # then: python training/build_sft_dataset.py --sft-dir data/sft --strip-htb-hint --out ...
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "evals" / "nyu_bench"),
          str(Path(__file__).resolve().parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

from dotenv import load_dotenv

from run_eval import (  # type: ignore
    FlagCheckingMCPPool, run_challenge_headless, build_mcp_pool,
    ensure_hexstrike_running, load_config, _apply_model_overrides,
)
from graph import build_graph, create_checkpointer  # type: ignore
from data_capture import TrajectoryLogger  # type: ignore
from htb_client import HTB  # type: ignore
from build_sft_dataset import HINT_START, HINT_END  # type: ignore


def _build_task(machine: str, ip: str, walkthrough: str | None = None,
                vhost: str | None = None) -> str:
    lines = [
        f"You are solving the HackTheBox machine '{machine}' at {ip}. Capture user.txt and root.txt.",
        "Work autonomously: scan, enumerate, exploit for a foothold, then escalate to root. Read",
        "both flag files and record their contents.",
    ]
    if vhost:
        lines += [
            "",
            f"HTB WEB ROUTING: this box uses virtual-host routing and '{vhost}' -> {ip} is already in "
            f"/etc/hosts. Target the app by HOSTNAME (http://{vhost}/ and https://{vhost}/), NOT the raw "
            f"IP — the IP usually serves an empty/default vhost, so IP-based dir fuzzing finds nothing. "
            f"Also fuzz for additional vhosts (e.g. ffuf -H 'Host: FUZZ.{vhost}' -u http://{ip}/ and "
            f"filter by response size) and add any you find to /etc/hosts.",
        ]
    if walkthrough and walkthrough.strip():
        lines += [
            "",
            HINT_START,
            "A walkthrough for THIS machine is provided ONLY to help you reach the flag quickly during",
            "data capture. Follow its technical steps using the real tools:",
            "",
            walkthrough.strip(),
            HINT_END,
        ]
    lines += ["", f"Target IP: {ip}. Begin."]
    return "\n".join(lines)


async def main() -> None:
    load_dotenv(str(PROJECT_ROOT / ".env"), override=False)
    ap = argparse.ArgumentParser(description="HTB walkthrough-guided trajectory capture (UNTESTED).")
    ap.add_argument("--machine", required=True, help="HTB machine name or id.")
    ap.add_argument("--walkthrough", default=None,
                    help="Path to a walkthrough (OPTIONAL). Omit for an UNHINTED genuine "
                         "solve/eval; provide it for hinted data capture (STaR).")
    ap.add_argument("--htb-token", default=os.getenv("HTB_TOKEN") or os.getenv("HTB_APP_TOKEN", ""))
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--provider", default=None, choices=["anthropic", "gemini", "ollama"])
    ap.add_argument("--worker-model", default=None)
    ap.add_argument("--summarizer-model", default=None)
    ap.add_argument("--model-override", action="append", default=[], metavar="AGENT=TAG")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--db", default=str(PROJECT_ROOT / "evals" / "nyu_bench" / "htb_checkpoints.db"))
    ap.add_argument("--no-stop", action="store_true", help="Leave the machine running afterward.")
    ap.add_argument("--verify-flags", action="store_true", help="Submit captured flags to HTB to label success.")
    ap.add_argument("--max-cost", type=float, default=0.0,
                    help="Hard USD cap for PAID (API) runs — ends the run mid-flight when the "
                         "cumulative API cost crosses this. 0 = no cap.")
    args = ap.parse_args()

    if not args.htb_token:
        sys.exit("No HTB token — set HTB_TOKEN (or pass --htb-token).")
    walkthrough = Path(args.walkthrough).read_text(encoding="utf-8") if args.walkthrough else None
    print(f"[htb] mode: {'HINTED capture' if walkthrough else 'UNHINTED genuine attempt'}")

    os.environ["CLANKER_CAPTURE_SFT"] = "1"  # this whole script exists to capture
    print("[htb] SFT capture ENABLED → data/sft/")

    htb = HTB(app_token=args.htb_token)
    print(f"[htb] spawning {args.machine} …")
    ip = htb.spawn_machine(args.machine)
    if ip.startswith("Error"):
        sys.exit(f"[htb] {ip}")
    print(f"[htb] {args.machine} at {ip}")

    # HTB web apps route by virtual host (<name>.htb). Resolve the machine name and add the
    # conventional vhost to /etc/hosts so the agent's tools reach the real app by hostname —
    # scanning the raw IP usually hits an empty default vhost (this is why the first Cohort
    # attempt fuzzed IP paths and found nothing).
    vhost = None
    _info = htb.get_machine_info(args.machine)
    if isinstance(_info, dict) and _info.get("name"):
        vhost = f"{_info['name'].lower()}.htb"
        try:
            with open("/etc/hosts", "a", encoding="utf-8") as _hf:
                _hf.write(f"\n{ip} {vhost}\n")
            print(f"[htb] added vhost to /etc/hosts: {ip} {vhost}")
        except Exception as _e:
            print(f"[htb] WARN: could not write /etc/hosts ({_e}); the agent must add {vhost} itself")

    config = load_config(PROJECT_ROOT / "agents.yaml", profile=args.profile)
    _apply_model_overrides(config, args)
    tlogger = TrajectoryLogger(str(PROJECT_ROOT / "data" / "training"))
    session_id = f"htb-{args.machine.lower()}-{str(uuid.uuid4())[:8]}"
    task = _build_task(args.machine, ip, walkthrough, vhost)

    ensure_hexstrike_running()
    mcp_pool = await build_mcp_pool()
    flag_pool = FlagCheckingMCPPool(mcp_pool, mode="off")  # no ground-truth flag to gate on
    try:
        async with create_checkpointer(args.db) as checkpointer:
            graph = build_graph(flag_pool, config, checkpointer)
            result = await run_challenge_headless(
                graph=graph, session_id=session_id, task=task, provider=args.provider,
                config=config, trajectory_logger=tlogger, target_ip=ip,
                timeout_sec=args.timeout, flag_pool=flag_pool, challenge_category="pwn",
                has_files=False, max_cost=args.max_cost,
            )
        flags = result.get("flags", [])
        print(f"[htb] captured flags: {flags or 'none'}  (trajectory → data/sft/{session_id}.jsonl)")
        if args.verify_flags and flags:
            for f in flags:
                print(f"[htb] submit {f!r}: {htb.submit_flag(f)}")
    finally:
        await mcp_pool.disconnect()
        if not args.no_stop:
            print(f"[htb] {htb.stop_machine()}")

    print("[htb] Next: python training/build_sft_dataset.py --sft-dir data/sft "
          "--strip-htb-hint --out training/data/htb.jsonl "
          "(add --report/--only-solved once you label solves).")


if __name__ == "__main__":
    asyncio.run(main())
