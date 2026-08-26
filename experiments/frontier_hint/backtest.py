"""
Backtest the stuck-detector + compactor over captured trajectories.

Offline, GPU-free, no API spend. For every multi-step session it:
  * runs the detector, labels the run solved vs stuck (flag captured?),
  * records where/whether it fires and how many steps remained,
  * builds + dumps the frontier payload for each fire (out/payloads/),
  * projects frontier cost at Sonnet/Opus prices,
and writes out/backtest_report.md + prints a summary.

Usage:
  python backtest.py [--data-dir DIR] [--min-steps N] [--config config.yaml]
"""
from __future__ import annotations

import sys

import argparse
import json
import os

import compactor
import frontier_hint
from detector import StuckDetector, load_config
from loader import load_outcomes, load_session_files, solved

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "out")


def first_finding_after(actions, idx, thresh=0.55):
    """First step > idx with a real finding (score >= thresh); None if none."""
    for a in actions[idx + 1:]:
        if (a["score"] or 0) >= thresh:
            return a["idx"]
    return None


def flag_step(actions):
    for a in actions:
        if (a["score"] or 0) >= 1.0:
            return a["idx"]
    return None


def run(data_dir=None, min_steps=5, config_path=None):
    cfg = load_config(config_path)
    kwargs = {} if data_dir is None else {"data_dir": data_dir}
    sessions = load_session_files(**kwargs)
    outcomes = load_outcomes(**kwargs)
    sessions = [s for s in sessions if s["n_steps"] >= min_steps]

    os.makedirs(os.path.join(OUT, "payloads"), exist_ok=True)

    rows = []
    total_fires = 0
    cost_sonnet = cost_opus = 0.0

    for s in sessions:
        det = StuckDetector(cfg, category=s["category"])
        fires = det.evaluate(s["actions"])
        is_solved = solved(s, outcomes)
        fstep = flag_step(s["actions"]) if is_solved else None
        first_fire = fires[0].idx if fires else None

        # dump payloads + accumulate cost
        payload_files = []
        for fe in fires:
            payload = compactor.build_payload(s, fe)
            fname = f"{s['session_id']}__step{fe.idx + 1:02d}.json"
            with open(os.path.join(OUT, "payloads", fname), "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            payload_files.append(fname)
            cost_sonnet += frontier_hint.project_cost(payload, "claude-sonnet-5")
            cost_opus += frontier_hint.project_cost(payload, "claude-opus-5")
        total_fires += len(fires)

        # did a real finding follow the first fire? (timing sanity / recovery)
        recovered = None
        if first_fire is not None:
            recovered = first_finding_after(s["actions"], first_fire) is not None

        rows.append({
            "session_id": s["session_id"],
            "category": s["category"],
            "steps": s["n_steps"],
            "solved": is_solved,
            "flag_step": (fstep + 1) if fstep is not None else None,
            "n_fires": len(fires),
            "first_fire_step": (first_fire + 1) if first_fire is not None else None,
            "steps_left_at_first_fire": (s["n_steps"] - (first_fire + 1)) if first_fire is not None else None,
            "recovered_after_fire": recovered,
            "top_signals": (dict(sorted(
                ((k, round(v, 2)) for k, v in fires[0].contributions.items() if v > 0),
                key=lambda kv: -kv[1])) if fires else {}),
            "payloads": payload_files,
        })

    return cfg, rows, {"total_fires": total_fires,
                       "cost_sonnet": cost_sonnet, "cost_opus": cost_opus}


def summarize(rows, totals):
    stuck = [r for r in rows if not r["solved"]]
    solv = [r for r in rows if r["solved"]]
    stuck_fired = [r for r in stuck if r["n_fires"] > 0]
    solved_fired = [r for r in solv if r["n_fires"] > 0]

    def avg(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 1) if xs else None

    return {
        "n_sessions": len(rows),
        "n_stuck": len(stuck),
        "n_solved": len(solv),
        "stuck_detection_recall": f"{len(stuck_fired)}/{len(stuck)}" if stuck else "0/0",
        "avg_steps_left_at_fire_on_stuck": avg([r["steps_left_at_first_fire"] for r in stuck_fired]),
        "solved_runs_that_fired": f"{len(solved_fired)}/{len(solv)}" if solv else "0/0",
        "solved_fired_that_recovered": f"{sum(1 for r in solved_fired if r['recovered_after_fire'])}/{len(solved_fired)}"
                                       if solved_fired else "0/0",
        "total_fires": totals["total_fires"],
        "avg_fires_per_session": round(totals["total_fires"] / len(rows), 2) if rows else 0,
        "projected_cost_all_fires_sonnet": round(totals["cost_sonnet"], 4),
        "projected_cost_all_fires_opus": round(totals["cost_opus"], 4),
        "projected_cost_per_session_sonnet": round(totals["cost_sonnet"] / len(rows), 5) if rows else 0,
    }


def write_report(cfg, rows, totals, summary, path):
    L = []
    L.append("# Frontier-Hint Backtest Report\n")
    L.append("Offline detection + compaction study over captured trajectories. "
             "No GPU, no API spend.\n")
    L.append("## Summary\n")
    for k, v in summary.items():
        L.append(f"- **{k}**: {v}")
    L.append("\n### How to read it\n")
    L.append("- `stuck_detection_recall` — of runs that captured NO flag, how many the "
             "detector flagged as stuck (higher = catches more salvageable runs).")
    L.append("- `avg_steps_left_at_fire_on_stuck` — headroom a hint would have had "
             "(higher = more chance to change the outcome).")
    L.append("- `solved_runs_that_fired` — potential false alarms; "
             "`solved_fired_that_recovered` shows how many of those were followed by a "
             "real finding anyway (a fire-then-recover is benign, a frequent one erodes economics).")
    L.append("- cost lines project the frontier spend if EVERY fire made one hint call.\n")

    L.append("## Config used\n```yaml")
    L.append(f"window: {cfg['window']}   min_steps: {cfg['min_steps']}   "
             f"cooldown: {cfg['cooldown']}   fire_threshold: {cfg['fire_threshold']}")
    L.append(f"weights: {json.dumps(cfg['weights'])}")
    L.append("```\n")

    L.append("## Per-session\n")
    L.append("| session | cat | steps | solved | flag@ | fires | 1st fire@ | steps left | recovered | top signals |")
    L.append("|---|---|--:|:--:|--:|--:|--:|--:|:--:|---|")
    for r in sorted(rows, key=lambda x: (x["solved"], -x["steps"])):
        sig = ", ".join(f"{k}:{v}" for k, v in r["top_signals"].items()) or "—"
        L.append("| {sid} | {cat} | {st} | {sv} | {fl} | {nf} | {ff} | {sl} | {rc} | {sig} |".format(
            sid=r["session_id"].replace("eval-", ""), cat=r["category"], st=r["steps"],
            sv="✅" if r["solved"] else "❌", fl=r["flag_step"] or "—", nf=r["n_fires"],
            ff=r["first_fire_step"] or "—", sl=r["steps_left_at_first_fire"] if r["steps_left_at_first_fire"] is not None else "—",
            rc=("—" if r["recovered_after_fire"] is None else ("yes" if r["recovered_after_fire"] else "no")),
            sig=sig))
    L.append("\n_Payloads for every fire are in `out/payloads/`. Inspect one with "
             "`frontier_hint.py --payload <f>` (dry-run) to see exactly what the frontier would receive._\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # avoid cp1252 mojibake on Windows consoles
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--min-steps", type=int, default=5)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg, rows, totals = run(args.data_dir, args.min_steps, args.config)
    summary = summarize(rows, totals)
    report_path = os.path.join(OUT, "backtest_report.md")
    write_report(cfg, rows, totals, summary, report_path)

    print("=== FRONTIER-HINT BACKTEST ===")
    for k, v in summary.items():
        print(f"{k:38s}: {v}")
    print(f"\nreport : {report_path}")
    print(f"payloads: {os.path.join(OUT, 'payloads')}/  ({totals['total_fires']} files)")


if __name__ == "__main__":
    main()
