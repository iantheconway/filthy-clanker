"""
Per-step stuck signals, computed from a normalized trajectory.

Pure + stdlib-only. A "normalized action" is a dict produced by
``loader.normalize_actions`` with at least these fields::

    idx, agent, tool, cmd (normalized), raw_cmd, score (float), category

Every function here takes the full ``actions`` list and an index ``i`` and
returns a **raw magnitude** (roughly 0..1, some unbounded-but-capped). The
detector multiplies these by the configured weights. Keeping magnitudes and
weights separate makes the report readable ("error_rate=0.8 contributed 0.24").
"""
from __future__ import annotations

import difflib
import re

_DIGITS = re.compile(r"\d+")
_WS = re.compile(r"\s+")
# a long absolute challenge path, e.g. /root/.nyuctf/.../crypto/stfu/ — strip so
# "cat stfu.c" and "cat /root/.../stfu.c" look the same for repeat detection.
_PATHY = re.compile(r"(/[\w.\-]+){2,}/?")


def normalize_cmd(raw: str) -> str:
    """Canonicalize a command/arg string for near-duplicate comparison."""
    s = (raw or "").lower()
    s = _PATHY.sub("<path>", s)
    s = _DIGITS.sub("<n>", s)
    s = _WS.sub(" ", s).strip()
    return s


def similar(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


# ---- individual signals ---------------------------------------------------

def kb_stall(actions, i, cap=4) -> float:
    """Consecutive steps up to and including i with score <= 0.05 (no finding).

    Returns count/cap, clamped to 1.0.
    """
    run = 0
    j = i
    while j >= 0 and (actions[j]["score"] is not None) and actions[j]["score"] <= 0.05:
        run += 1
        j -= 1
    return min(run, cap) / cap


def repeat(actions, i, window=6, sim_threshold=0.90) -> float:
    """1.0 if the current command near-duplicates an earlier in-window command.

    Scales with how many of the in-window predecessors it matches (a tight loop
    of the same command scores higher than a single accidental repeat).
    """
    cur = actions[i]["cmd"]
    if not cur:
        return 0.0
    lo = max(0, i - window)
    hits = 0
    denom = 0
    for j in range(lo, i):
        denom += 1
        if similar(cur, actions[j]["cmd"]) >= sim_threshold:
            hits += 1
    if denom == 0:
        return 0.0
    # any hit is meaningful; multiple hits (real loop) pushes toward 1.0
    return min(1.0, 0.6 + 0.4 * (hits - 1)) if hits else 0.0


def error_rate(actions, i, window=6) -> float:
    """Fraction of in-window steps (incl. i) that hard-errored (score == 0.0)."""
    lo = max(0, i - window + 1)
    seg = actions[lo:i + 1]
    if not seg:
        return 0.0
    errs = sum(1 for a in seg if a["score"] == 0.0)
    return errs / len(seg)


def stale_progress(actions, i, cap=8) -> float:
    """Steps since the last 'real finding' (score >= 0.55), as count/cap."""
    steps = 0
    j = i
    while j >= 0 and (actions[j]["score"] is None or actions[j]["score"] < 0.55):
        steps += 1
        j -= 1
    return min(steps, cap) / cap


def oscillation(actions, i, window=6) -> float:
    """Agent bouncing between >=2 agents in-window with no score gain.

    1.0 when >=2 distinct agents appear in the window AND no in-window step made
    a real finding (score < 0.55 throughout); 0 otherwise.
    """
    lo = max(0, i - window + 1)
    seg = actions[lo:i + 1]
    if len(seg) < 3:
        return 0.0
    agents = {a["agent"] for a in seg if a["agent"]}
    made_progress = any((a["score"] or 0) >= 0.55 for a in seg)
    switches = sum(1 for k in range(lo + 1, i + 1)
                   if actions[k]["agent"] and actions[k]["agent"] != actions[k - 1]["agent"])
    if len(agents) >= 2 and switches >= 2 and not made_progress:
        return 1.0
    return 0.0


def step_budget(actions, i, budget=12) -> float:
    """How far over the soft per-category step budget we are, as a ramp 0..1.

    0 at/under budget, reaching 1.0 at 2x budget.
    """
    step = i + 1
    if step <= budget:
        return 0.0
    return min(1.0, (step - budget) / max(1, budget))
