"""
Build the payload sent to the frontier model when the detector fires.

Design principle (the load-bearing one): send the frontier **raw ground-truth
evidence**, not the local model's paraphrase. If we forward the local model's
summary of the situation, we launder its blind spot into the frontier — and the
reason it is stuck is usually that it misread a banner or skipped a detail. So
the payload carries:

  1. objective           — the challenge task (trimmed)
  2. knowledge_base       — structured state snapshot at fire time
  3. tried_ledger         — deduped commands run so far + their outcome tier,
                            so the frontier does not re-suggest dead ends
  4. recent_evidence      — the last K raw tool outputs (truncated), verbatim
  5. stuck_symptom        — which signals fired + stuckness

Renders three things: a `prompt` string (ready for the frontier), a structured
`payload` dict (persisted for inspection + later training data), and a rough
`token_estimate`.
"""
from __future__ import annotations

import json

# outcome tier labels for the tried-ledger (mirrors the success_score heuristic)
_TIER = [
    (1.0, "FLAG"), (0.85, "creds"), (0.70, "attack-surface"), (0.55, "new-port/find"),
    (0.40, "new-ip"), (0.20, "dir/file-find"), (0.05, "no-new-finding"), (0.0, "ERROR"),
]


def _tier(score) -> str:
    if score is None:
        return "?"
    for thresh, label in _TIER:
        if score >= thresh:
            return label
    return "ERROR"


def _trim(text: str, n: int) -> str:
    text = text or ""
    if len(text) <= n:
        return text
    return text[:n] + f"\n…[+{len(text) - n} chars truncated]"


def _trim_task(task: str, n: int = 900) -> str:
    """Keep the informative head of the task (category/name/description/files)."""
    # drop the long boilerplate divider lines that some tasks carry
    lines = [l for l in (task or "").splitlines() if set(l.strip()) - set("-—_ ")]
    return _trim("\n".join(lines), n)


def build_tried_ledger(actions, upto, max_entries=24):
    """Deduped list of commands attempted through step `upto`, newest-relevant last.

    Collapses near-identical repeats into a count so the frontier sees the loop.
    """
    ledger = []
    seen = {}
    for a in actions[:upto + 1]:
        key = a["cmd"]
        cmd = a["raw_cmd"].strip().splitlines()[0][:160] if a["raw_cmd"] else a["tool"]
        if key in seen:
            ledger[seen[key]]["count"] += 1
            # keep the best (highest-score) outcome seen for this command
            if (a["score"] or 0) > (ledger[seen[key]]["best_score"] or 0):
                ledger[seen[key]]["best_score"] = a["score"]
            continue
        seen[key] = len(ledger)
        ledger.append({"cmd": cmd, "tool": a["tool"], "count": 1, "best_score": a["score"]})
    # if too long, keep the most-repeated / most-recent
    if len(ledger) > max_entries:
        ledger = ledger[-max_entries:]
    for e in ledger:
        e["outcome"] = _tier(e.pop("best_score"))
    return ledger


def build_payload(session, fire, actions=None, *, evidence_k=4,
                  evidence_chars=1400, task_chars=900):
    """Assemble the structured payload dict for a FireEvent on a session."""
    actions = actions if actions is not None else session["actions"]
    i = fire.idx
    fired = sorted(((k, round(v, 3)) for k, v in fire.contributions.items() if v > 0),
                   key=lambda kv: -kv[1])
    recent = actions[max(0, i - evidence_k + 1):i + 1]
    payload = {
        "session_id": session["session_id"],
        "category": session.get("category"),
        "fired_at_step": i + 1,
        "total_steps_in_run": len(actions),
        "stuckness": fire.stuckness,
        "stuck_symptom": {
            "fired_signals": fired,
            "raw_magnitudes": {k: round(v, 3) for k, v in fire.raw.items()},
        },
        "objective": _trim_task(session["task"], task_chars),
        "knowledge_base": actions[i].get("kb_before") or {},
        "tried_ledger": build_tried_ledger(actions, i),
        "recent_evidence": [
            {
                "step": a["idx"] + 1,
                "agent": a["agent"],
                "tool": a["tool"],
                "command": _trim(a["raw_cmd"], 300),
                "outcome": _tier(a["score"]),
                "raw_output": _trim(a["result_snippet"], evidence_chars),
            }
            for a in recent
        ],
    }
    return payload


SYSTEM_PROMPT = (
    "You are a senior CTF operator. A much weaker autonomous agent has STALLED and CANNOT finish "
    "this on its own — it will not reliably follow multi-step advice, so THIS IS YOUR ONE SHOT to "
    "solve the challenge. Do not plan a long back-and-forth; give the COMPLETE solution now. You are "
    "given the objective, the knowledge base, a ledger of what was already tried (with outcomes), and "
    "the RAW output of the last few tool calls — the raw output is ground truth; the stalled agent "
    "likely MISREAD or OVERLOOKED something there, so look there first.\n\n"
    "Your FIRST ranked_action must be the single command that makes the most decisive progress toward "
    "the flag — ideally one that SOLVES it outright. For crypto/rev/pwn that means WRITE THE FULL "
    "SOLVER: emit a complete self-contained script (python3 with pwntools / pycryptodome / z3 / gmpy2 "
    "as needed) via `execute_command` and a heredoc, e.g.\n"
    "  cat > /tmp/solve.py <<'PYEOF'\\n<the entire solver>\\nPYEOF\\npython3 /tmp/solve.py\n"
    "That ONE script must do the WHOLE solve — decode, extract, AND at the end PRINT the recovered "
    "flag on its own line as exactly `FLAG: <the flag>` (the exact value: no wrapper added or changed, "
    "every character preserved). Whatever text follows `FLAG:` is AUTO-SUBMITTED, so finish the solve "
    "in this one script — do NOT split decode / extract / submit across separate steps.\n"
    "Do NOT emit another cat/xxd/ls/strings inspection step unless you genuinely still lack the bytes "
    "needed to write the solver. Never repeat anything in the tried-ledger. Assume you will NOT be "
    "called again — hold nothing back.\n\n"
    "SUBMITTING THE FLAG: the moment you know it — your solver reveals it, or it is already visible in "
    "the raw evidence — make your ranked_action a submit call: "
    '{"tool":"submit_flag","command":"<the EXACT flag>"}. Use the exact value: do NOT add or change any '
    "flag{}/KEY{}/CSAW{} wrapper, and preserve EVERY character (underscores, case, punctuation). If the "
    "flag is a raw plaintext secret with no wrapper, submit that raw string verbatim.\n\n"
    "Return STRICT JSON with keys:\n"
    '  "observation":  the key thing the agent missed in the raw output (or null)\n'
    '  "hypothesis":   the exact path to the flag, one sentence\n'
    '  "ranked_actions": 1-3 concrete tool calls, MOST DECISIVE FIRST, each {"tool","command","why"}; '
    "the first should solve or near-solve the challenge\n"
    '  "rule_out":     up to 2 dead ends to abandon, with why\n'
    "Be concrete and executable. No prose outside the JSON."
)


def render_prompt(payload) -> str:
    """The user-message string for the frontier call."""
    kb = payload["knowledge_base"]
    kb_str = json.dumps(kb, indent=2)[:1500] if kb else "(empty)"
    parts = [
        f"## Objective\n{payload['objective']}",
        f"\n## Why the agent looks stuck (step {payload['fired_at_step']}, "
        f"stuckness={payload['stuckness']})\n"
        + ", ".join(f"{k}={v}" for k, v in payload["stuck_symptom"]["fired_signals"]),
        f"\n## Knowledge base (structured state)\n{kb_str}",
        "\n## Already tried (do NOT repeat)\n" + "\n".join(
            f"- [{e['outcome']}] x{e['count']} {e['tool']}: {e['cmd']}"
            for e in payload["tried_ledger"]
        ),
        "\n## Raw output of the last few tool calls (GROUND TRUTH)\n" + "\n\n".join(
            f"### step {e['step']} — {e['tool']} ({e['outcome']})\n"
            f"$ {e['command']}\n{e['raw_output']}"
            for e in payload["recent_evidence"]
        ),
        "\nReturn the JSON hint now.",
    ]
    return "\n".join(parts)


def estimate_tokens(payload) -> int:
    """Rough token estimate for the rendered prompt (~4 chars/token)."""
    return (len(SYSTEM_PROMPT) + len(render_prompt(payload))) // 4
