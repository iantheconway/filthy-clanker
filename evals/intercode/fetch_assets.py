#!/usr/bin/env python3
"""
Host-side asset prefetch for InterCode-CTF — run BEFORE the sandboxed eval.

The eval sandbox blocks public internet, but some tasks carry a `setup` wget that downloads a
(often large) asset from picoCTF. Run this ON THE HOST (which has internet) once to populate
data/ctf/task_assets/<task_id>/ so those tasks can then run offline.

    export INTERCODE_DIR=/path/to/intercode
    python evals/intercode/fetch_assets.py            # fetch every task with a `setup`
    python evals/intercode/fetch_assets.py --ids 1,7  # only these task_ids

Note: `setup` values come from the InterCode-CTF dataset (wget of picoCTF assets); this runs
them as-is in each task's asset dir, the same thing InterCode's own provisioning does.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Prefetch InterCode-CTF `setup` assets on the host.")
    ap.add_argument("--intercode-dir", default=os.getenv("INTERCODE_DIR", ""))
    ap.add_argument("--ids", default=None, help="Comma-separated task_ids (default: all with a setup).")
    ap.add_argument("--force", action="store_true", help="Re-run setup even if the asset dir is non-empty.")
    args = ap.parse_args()
    if not args.intercode_dir:
        sys.exit("Set --intercode-dir or $INTERCODE_DIR.")

    ctf = Path(args.intercode_dir) / "data" / "ctf"
    data = json.loads((ctf / "ic_ctf.json").read_text(encoding="utf-8"))
    assets = ctf / "task_assets"
    ids = {int(x) for x in args.ids.split(",") if x.strip().isdigit()} if args.ids else None

    fetched = skipped = failed = 0
    for e in data:
        setup = (e.get("setup") or "").strip()
        tid = e.get("task_id")
        if not setup or (ids is not None and tid not in ids):
            continue
        d = assets / str(tid)
        d.mkdir(parents=True, exist_ok=True)
        if not args.force and any(d.iterdir()):
            skipped += 1
            continue
        print(f"[fetch] task {tid}: {setup}")
        try:
            subprocess.run(setup, shell=True, cwd=str(d), check=True, timeout=900)
            fetched += 1
        except Exception as exc:
            print(f"[fetch] task {tid} FAILED: {exc}")
            failed += 1
    print(f"[fetch] done: fetched={fetched} skipped(already present)={skipped} failed={failed}")


if __name__ == "__main__":
    main()
