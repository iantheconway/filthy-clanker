"""Diagnostic: why does/doesn't the detector fire? Per-session peak stuckness and
per-signal magnitudes, so weights/threshold can be tuned against evidence."""
from __future__ import annotations

import sys

import argparse
import statistics as st

from detector import load_config, step_stuckness
from loader import load_outcomes, load_session_files, solved


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # avoid cp1252 mojibake on Windows consoles
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--min-steps", type=int, default=5)
    args = ap.parse_args()
    cfg = load_config()
    kw = {} if args.data_dir is None else {"data_dir": args.data_dir}
    sessions = [s for s in load_session_files(**kw) if s["n_steps"] >= args.min_steps]
    outcomes = load_outcomes(**kw)

    # global score distribution
    all_scores = [a["score"] for s in sessions for a in s["actions"] if a["score"] is not None]
    from collections import Counter
    print("score histogram:", dict(sorted(Counter(all_scores).items())))
    print()

    signames = list(cfg["weights"].keys())
    peak_raw = {k: [] for k in signames}

    print(f"{'session':40s} {'solved':6s} {'peakStuck':9s}  top raw signals at peak")
    for s in sorted(sessions, key=lambda x: -x["n_steps"]):
        best = (-1, None, None)
        for i in range(len(s["actions"])):
            if i + 1 < cfg["min_steps"]:
                continue
            stuck, contrib, raw = step_stuckness(s["actions"], i, cfg, s["category"])
            if stuck > best[0]:
                best = (stuck, raw, i)
        stuck, raw, i = best
        for k in signames:
            peak_raw[k].append(raw[k])
        top = sorted(raw.items(), key=lambda kv: -kv[1])[:3]
        topstr = ", ".join(f"{k}={v:.2f}" for k, v in top if v > 0) or "—"
        print(f"{s['session_id'][:40]:40s} {str(solved(s,outcomes)):6s} "
              f"{stuck:6.3f}    {topstr}")

    print("\nmean RAW magnitude per signal at each session's peak step:")
    for k in signames:
        vals = peak_raw[k]
        print(f"  {k:16s} mean={st.mean(vals):.3f}  max={max(vals):.3f}  "
              f"nonzero={sum(1 for v in vals if v>0)}/{len(vals)}")


if __name__ == "__main__":
    main()
