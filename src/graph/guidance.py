"""
guidance_node — Phase 3 of the frontier-hint experiment, wired live (v2: executes).

When the team STALLS, we compact the ground-truth state and route it to a FRONTIER
model for a next-step plan. v1 injected that plan as advice — but the diagnosis on
csawpad was decisive: the local base model SEES the hint (routing LLM cites it) yet
never turns "read the file" into an actual tool call (turn after turn: reasoning,
tool_calls=[]). Advice alone can't move a model that won't execute.

v2 fix (the load-bearing change): the guidance node EXECUTES the frontier's pivotal
step itself via mcp_client.call_tool, captures the REAL output (and any grounded
flag → ends the run), and injects that ground truth so the local model continues
from a solved sub-step instead of being asked to act. Cost-asymmetric: the frontier
reasons + makes the one pivotal move at each stall; the local model does the rest.

Gated hard (per-session cap, cooldown, run-wide spend cap). Never raises into the
graph: any failure returns the reset dict (run continues).
"""
from __future__ import annotations

import logging
import os
import re
import sys

logger = logging.getLogger("filthy_clanker")

# Run-wide accumulator of FRONTIER-HINT USD spend (persists across challenges in the
# one eval process). Bounds guidance spend specifically — separate from the eval's
# total --max-cost, so the Haiku summariser's spend can't starve the hint budget.
_GUIDANCE_SPEND = 0.0

# Make the experiment package importable (compactor + frontier_hint live there).
_FH_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "experiments", "frontier_hint",
)
if _FH_DIR not in sys.path:
    sys.path.insert(0, _FH_DIR)


# ---- gating ---------------------------------------------------------------

def _gcfg(state) -> dict:
    settings = (state.get("config", {}) or {}).get("settings", {}) or {}
    g = dict(settings.get("guidance", {}) or {})
    g.setdefault("enabled", False)
    g.setdefault("model", "claude-sonnet-5")
    g.setdefault("max_hints_per_session", 4)
    g.setdefault("cooldown_msgs", 12)
    g.setdefault("step_budget_msgs", 28)
    g.setdefault("max_cost_usd", 3.0)
    g.setdefault("max_output", 16000)   # thinking + a full solver script needs a big output budget
    g.setdefault("max_call_cost_usd", 0.60)
    g.setdefault("execute_step", True)   # v2: run the frontier's pivotal step directly
    return g


# Hinter observability: the [Supervisor]/[guidance] logs don't reach the eval's captured
# stdout, so a run could no-op the hinter completely and look identical. These counters make
# it self-reporting (run_eval prints hint_summary() at the end, like the LLM error-summary):
# considered = times the step-budget trigger evaluated should_hint; reject = why it said no;
# reached = guidance_node actually ran; skipped_cost = per-call cost cap; fired = frontier called.
_HINT_STATS: dict = {"considered": 0, "reject": {}, "reached": 0, "skipped_cost": 0,
                     "fired": 0, "error": {}}


def hint_summary() -> dict:
    return {"considered": _HINT_STATS["considered"], "reached": _HINT_STATS["reached"],
            "skipped_cost": _HINT_STATS["skipped_cost"], "fired": _HINT_STATS["fired"],
            "reject": dict(_HINT_STATS["reject"]), "error": dict(_HINT_STATS["error"]),
            "spend_usd": round(_GUIDANCE_SPEND, 3)}


def should_hint(state) -> bool:
    """Enabled + under per-session cap + past cooldown + has key + under spend cap."""
    _HINT_STATS["considered"] += 1

    def _no(reason: str) -> bool:
        _HINT_STATS["reject"][reason] = _HINT_STATS["reject"].get(reason, 0) + 1
        return False

    g = _gcfg(state)
    if not g["enabled"]:
        return _no("disabled")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return _no("no_api_key")
    if int(state.get("hints_used", 0)) >= int(g["max_hints_per_session"]):
        return _no("session_cap")
    turn = len(state.get("messages", []) or [])
    if turn - int(state.get("last_hint_step", -10 ** 9)) < int(g["cooldown_msgs"]):
        return _no("cooldown")
    if _GUIDANCE_SPEND >= float(g["max_cost_usd"]):
        logger.info("[guidance] frontier-hint spend cap reached ($%.2f) — no more hints", _GUIDANCE_SPEND)
        return _no("spend_cap")
    return True


# ---- payload from live state ---------------------------------------------

def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for b in content:
            if isinstance(b, dict):
                out.append(b.get("text") or b.get("content") or
                           (b.get("output") if isinstance(b.get("output"), str) else "") or "")
            elif isinstance(b, str):
                out.append(b)
        return "\n".join(x for x in out if x)
    return str(content or "")


def _recent_evidence(raw_log, k=4, chars=1600):
    """The last k RAW (un-summarised) tool outputs — the ground truth the frontier needs.
    Sourced from raw_tool_log (captured pre-summarisation in the agent loop), NOT the
    message stream (which is summarised, laundering the local model's blind spot)."""
    recs = list(raw_log or [])
    n = len(recs)
    ev = []
    for j, r in enumerate(recs[-k:]):
        raw = str(r.get("raw", ""))
        raw = raw if len(raw) <= chars else raw[:chars] + f"\n…[+{len(raw) - chars} chars truncated]"
        low = raw[:80].lower()
        outcome = "ERROR" if ("error" in low or "unknown tool" in low or "traceback" in low) else "?"
        ev.append({"step": n - min(k, n) + j + 1, "tool": r.get("tool", "tool"),
                   "outcome": outcome, "command": str(r.get("cmd", ""))[:200], "raw_output": raw})
    return ev


def _tried_ledger(raw_log):
    """Deduped EXACT commands the agents ran (coarse outcome), from the raw records."""
    seen, led = {}, []
    for r in (raw_log or []):
        cmd = str(r.get("cmd", ""))[:160]
        if not cmd:
            continue
        if cmd in seen:
            led[seen[cmd]]["count"] += 1
            continue
        raw = str(r.get("raw", "")); low = raw[:80].lower()
        outcome = ("ERROR" if ("error" in low or "unknown tool" in low)
                   else "no-new-finding" if len(raw.strip()) < 40 else "data")
        seen[cmd] = len(led)
        led.append({"outcome": outcome, "count": 1, "tool": r.get("tool", "tool"), "cmd": cmd})
    return led[-24:]


def build_payload_from_state(state, reason: str):
    import compactor
    kb = dict(state.get("knowledge_base", {}) or {})
    kb_snap = {k: v for k, v in kb.items()
               if k not in ("scan_history", "exploit_history", "visited_urls",
                            "response_headers", "operator_plan", "operator_executed")}
    _clock = len(state.get("messages", []) or [])
    # v4: build tried-ledger + recent_evidence from the RAW tool log (real commands + raw
    # output), reproducing the offline compactor's fidelity instead of summarised messages.
    raw_log = state.get("raw_tool_log", []) or []
    tried = _tried_ledger(raw_log)
    evidence = _recent_evidence(raw_log)
    # v3: feed the operator's OWN prior executions back so the frontier PROGRESSES to the
    # next step (guidance-node executions aren't in raw_tool_log — that's agent calls only).
    op_exec = list(kb.get("operator_executed", []) or [])
    for e in op_exec:
        tried.insert(0, {"outcome": "operator-ran", "count": 1, "tool": "OPERATOR-ALREADY-RAN",
                         "cmd": str(e.get("cmd", ""))[:160]})
    for e in list(reversed(op_exec))[:2]:   # newest operator output first = highest priority
        evidence.insert(0, {"step": 0, "tool": "OPERATOR-EXECUTED", "outcome": "operator-ran",
                            "command": str(e.get("cmd", ""))[:200],
                            "raw_output": str(e.get("out", ""))[:1400]})
    return {
        "session_id": state.get("session_id", "?"),
        "category": state.get("challenge_category", "?"),
        "fired_at_step": _clock,
        "total_steps_in_run": _clock,
        "stuckness": 1.0,
        "stuck_symptom": {"fired_signals": [(reason, 1.0)], "raw_magnitudes": {reason: 1.0}},
        "objective": compactor._trim_task(state.get("task", ""), 900),
        "knowledge_base": kb_snap,
        "tried_ledger": tried,
        "recent_evidence": evidence,
    }


def _plan_steps(hint) -> list:
    """[{'command','why','done'}] from the frontier's ranked_actions (dicts or strings)."""
    if not isinstance(hint, dict):
        return []
    ra = hint.get("ranked_actions") or []
    ra = [ra] if isinstance(ra, (str, dict)) else list(ra)
    steps = []
    for a in ra[:4]:
        if isinstance(a, dict):
            cmd = a.get("command") or a.get("action") or a.get("cmd") or ""
            why = a.get("why") or a.get("reason") or ""
            tool = a.get("tool") or ""
        else:
            cmd, why, tool = str(a), "", ""
        if cmd:
            steps.append({"command": cmd, "why": why, "tool": tool, "done": False})
    return steps


def _format_hint(hint) -> str:
    """Turn the frontier's JSON hint into a directive. Tolerant of string items."""
    if not isinstance(hint, dict) or "_raw" in hint:
        return str(hint.get("_raw", hint) if isinstance(hint, dict) else hint)[:1600]
    lines = ["[FRONTIER HINT — a senior operator reviewed your stalled state]"]
    if hint.get("observation"):
        lines.append(f"You likely MISSED: {hint['observation']}")
    if hint.get("hypothesis"):
        lines.append(f"Most probable path to the flag: {hint['hypothesis']}")
    ra = hint.get("ranked_actions") or []
    ra = [ra] if isinstance(ra, (str, dict)) else list(ra)
    for i, a in enumerate(ra[:3], 1):
        if isinstance(a, dict):
            cmd = a.get("command") or a.get("action") or a.get("cmd") or ""
            why = a.get("why") or a.get("reason") or ""
        else:
            cmd, why = str(a), ""
        lines.append(f"NEXT ACTION {i}: {cmd}" + (f"  ({why})" if why else ""))
    ro = hint.get("rule_out") or []
    ro = [ro] if isinstance(ro, (str, dict)) else list(ro)
    for r in ro[:2]:
        lines.append(f"STOP trying: {r.get('hypothesis','')} — {r.get('why','')}"
                     if isinstance(r, dict) else f"STOP trying: {r}")
    return "\n".join(lines)


# ---- the node (v2: executes the pivotal step) ----------------------------

def make_guidance_node(mcp_client):
    """Guidance node bound to the live MCP client so it can EXECUTE the frontier's
    pivotal step directly (the local model won't turn a hint into a tool call)."""

    async def guidance_node(state) -> dict:
        global _GUIDANCE_SPEND
        g = _gcfg(state)
        reason = state.get("hint_reason", "stall")
        turn = len(state.get("messages", []) or [])
        base_reset = {
            "hint_reason": None, "unproductive_streak": 0, "exploit_attempts": 0,
            "completed_agents": [], "last_hint_step": turn,
            "hints_used": int(state.get("hints_used", 0)) + 1,
        }
        _HINT_STATS["reached"] += 1
        try:
            import compactor  # noqa: F401
            import frontier_hint
            payload = build_payload_from_state(state, reason)
            projected = frontier_hint.project_cost(payload, g["model"], g["max_output"])
            if projected > float(g["max_call_cost_usd"]):
                logger.warning("[guidance] projected cost $%.3f > cap $%.3f — skipping", projected, g["max_call_cost_usd"])
                _HINT_STATS["skipped_cost"] += 1
                return base_reset
            logger.info("[guidance] STALL (%s) at turn %d — frontier hint (%s, ~$%.3f)…",
                        reason, turn, g["model"], projected)
            result = frontier_hint.call_frontier(payload, g["model"], g["max_output"])
            _HINT_STATS["fired"] += 1
            hint = result.get("hint", {})
            usage = result.get("usage") or {}
            _GUIDANCE_SPEND += projected
            try:
                from llms import cost as _cost
                _cost.add_usage(g["model"], {"input_tokens": usage.get("input", 0),
                                             "output_tokens": usage.get("output", 0)})
            except Exception:
                pass
            hint_text = _format_hint(hint)
            logger.info("[guidance] hint (%d in/%d out): %s", usage.get("input", 0),
                        usage.get("output", 0), hint_text[:150].replace("\n", " | "))

            kb_update = dict(state.get("knowledge_base", {}) or {})
            steps = _plan_steps(hint)
            kb_update["operator_plan"] = steps   # persists across compaction
            msgs = [{"role": "user", "content": hint_text}]
            executed = None

            # --- v2: EXECUTE the pivotal step directly ---
            if g.get("execute_step", True) and steps and mcp_client is not None:
                cmd = steps[0]["command"]
                executed = cmd
                # Honor the frontier's tool choice: submit_flag (exact value, no wrapper
                # mangling — the clean path for the flag) vs execute_command (run a solver).
                _tool = (steps[0].get("tool") or "execute_command").strip().lower().replace("-", "_")
                if _tool in ("submit_flag", "submitflag"):
                    call_tool, call_args = "submit_flag", {"flag": cmd}
                else:
                    call_tool, call_args = "execute_command", {"command": cmd}
                logger.info("[guidance] EXECUTING operator step 1 → %s: %s", call_tool, cmd[:180])
                try:
                    raw = await mcp_client.call_tool(call_tool, call_args)
                except Exception as exc:
                    raw = f"(operator execution error: {exc})"
                    logger.warning("[guidance] %s failed: %s", call_tool, exc)
                raw = raw or ""
                # v3: record the execution so the NEXT payload shows it (frontier advances)
                prior_exec = list(state.get("knowledge_base", {}).get("operator_executed", []) or [])
                prior_exec.append({"cmd": cmd, "out": raw[:1400]})
                kb_update["operator_executed"] = prior_exec[-8:]
                try:
                    try:
                        from .agents import _extract_flags_from_raw
                    except ImportError:
                        from agents import _extract_flags_from_raw  # standalone/smoke context
                    gf = _extract_flags_from_raw(raw, {"command": cmd}, flag_format=state.get("flag_format", ""))
                except Exception:
                    gf = []
                if gf:
                    kb_update["flags"] = list(set(kb_update.get("flags", [])) | set(gf))
                    kb_update["grounded_flags"] = list(set(kb_update.get("grounded_flags", [])) | set(gf))
                    logger.info("[guidance] operator execution produced GROUNDED FLAG(s): %s", gf)
                # DETERMINISM: if the solver PRINTED the flag with a `FLAG:` marker, submit it right
                # now — one-shot solve, no separate submit round-trip. Handles plaintext + wrapped.
                if call_tool == "execute_command":
                    _fm = re.search(r'(?im)^[ \t]*FLAG:[ \t]*(\S.*?)[ \t]*$', raw)
                    if _fm:
                        _fl = _fm.group(1).strip()
                        if len(_fl) >= 4 and "not found" not in _fl.lower() and "<" not in _fl:
                            logger.info("[guidance] solver printed FLAG marker — auto-submitting: %s", _fl[:120])
                            try:
                                _sub = await mcp_client.call_tool("submit_flag", {"flag": _fl})
                                raw = raw + "\n[auto-submit] " + str(_sub)[:200]
                            except Exception as _e:
                                logger.warning("[guidance] auto-submit failed: %s", _e)
                steps[0]["done"] = True
                kb_update["operator_plan"] = steps
                msgs.append({"role": "user", "content":
                    "[OPERATOR EXECUTED the next step for you — REAL output below]\n"
                    f"$ {cmd}\n{raw[:2500]}\n"
                    "Use this actual output. Do NOT repeat analysis already done; continue from here."})

            triple = {"turn": turn, "reason": reason, "payload": payload, "hint": hint,
                      "executed": executed, "kb_at_hint": state.get("knowledge_base", {})}
            return {
                **base_reset,
                "messages": msgs,
                "knowledge_base": kb_update,
                "hint_log": list(state.get("hint_log", []) or []) + [triple],
            }
        except Exception as exc:
            _k = type(exc).__name__
            _HINT_STATS["error"][_k] = _HINT_STATS["error"].get(_k, 0) + 1
            logger.error("[guidance] hint failed (%s) — continuing unhinted", exc, exc_info=True)
            return base_reset

    return guidance_node
