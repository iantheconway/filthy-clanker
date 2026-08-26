"""Threshold + weight-profile sweep. Reports, per operating point, stuck-recall vs
solved false-alarm rate, so we pick the fire_threshold/weights from a Pareto view
instead of guessing. Peak-stuckness per session is profile-dependent, so we
recompute it for each weight profile."""
from __future__ import annotations

import sys

import argparse

from detector import load_config, step_stuckness
from loader import load_outcomes, load_session_files, solved

# Weight profiles to compare. 'default' = config.yaml. 'grind' leans on the signals
# that are actually alive in the file-challenge corpus (repeat/step_budget/kb_stall)
# and down-weights the near-dead ones (error_rate/oscillation) — those stay useful
# for multi-agent network runs, this profile just isn't betting on them here.
PROFILES = {
    "default": None,  # use config as-is
    "grind": {"kb_stall": 0.18, "repeat": 0.35, "error_rate": 0.20,
              "stale_progress": 0.15, "oscillation": 0.15, "step_budget": 0.30},
}


def peak_per_session(sessions, cfg):
    out = []
    for s in sessions:
        best = 0.0
        for i in range(len(s["actions"])):
            if i + 1 < cfg["min_steps"]:
                continue
            stuck, _, _ = step_stuckness(s["actions"], i, cfg, s["category"])
            best = max(best, stuck)
        out.append(best)
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # avoid cp1252 mojibake on Windows consoles
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--min-steps", type=int, default=5)
    args = ap.parse_args()
    base = load_config()
    kw = {} if args.data_dir is None else {"data_dir": args.data_dir}
    sessions = [s for s in load_session_files(**kw) if s["n_steps"] >= args.min_steps]
    outcomes = load_outcomes(**kw)
    labels = [solved(s, outcomes) for s in sessions]
    n_stuck = labels.count(False)
    n_solved = labels.count(True)
    print(f"corpus: {len(sessions)} sessions  ({n_stuck} stuck / {n_solved} solved)\n")

    thresholds = [round(0.25 + 0.05 * k, 2) for k in range(9)]  # 0.25..0.65
    for pname, pw in PROFILES.items():
        cfg = dict(base)
        cfg["weights"] = dict(base["weights"]) if pw is None else pw
        peaks = peak_per_session(sessions, cfg)
        print(f"=== profile: {pname} ===")
        print(f"{'thresh':7s} {'recall(stuck)':14s} {'false-alarm(solved)':20s} {'youdenJ':8s}")
        for t in thresholds:
            tp = sum(1 for p, l in zip(peaks, labels) if not l and p >= t)  # stuck & fired
            fp = sum(1 for p, l in zip(peaks, labels) if l and p >= t)      # solved & fired
            recall = tp / n_stuck if n_stuck else 0
            far = fp / n_solved if n_solved else 0
            youden = recall - far  # higher = better separation at this threshold
            print(f"{t:<7.2f} {tp}/{n_stuck} = {recall:0.2f}    "
                  f"{fp}/{n_solved} = {far:0.2f}           {youden:+.2f}")
        print()


if __name__ == "__main__":
    main()
