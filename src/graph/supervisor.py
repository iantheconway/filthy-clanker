"""
Supervisor node — the orchestration brain of the multi-agent workflow.

Responsibilities:
  1. Examine current state and knowledge base.
  2. Call the supervisor LLM to determine the next agent to route to.
  3. Detect "exploit loops" (too many failures) and trigger HITL interrupt.
  4. Detect context limit proximity and route to the compaction node.
  5. Detect task completion (flag found) and route to END.

The supervisor uses LangGraph's `interrupt()` to pause execution and hand
control back to the human operator when manual intervention is needed.
"""
from __future__ import annotations

import json
import logging
import sys
import os
from typing import Any, Literal

from langgraph.types import interrupt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from llms import AnthropicClient, GeminiClient, OllamaClient

from .state import TeamState
from .summarizer import _is_placeholder_flag, _flag_matches_format

logger = logging.getLogger("filthy_clanker")


# Possible routing destinations
NEXT_OPTIONS = Literal["recon", "exploit", "privesc", "vulnsearch", "compaction", "__end__"]


_PROVIDER_DEFAULT_MODELS = {
    "anthropic": "claude-opus-4-6",
    "gemini": "gemini-2.5-flash",
    "ollama": "llama3.2",
}


def _resolve_llm(state: TeamState, agent_cfg: dict):
    """
    Determine (provider, model, llm_client) for the supervisor.
    Mirrors the same logic in agents.py — global override wins, else per-agent config.
    """
    override = state.get("provider")

    if override:
        provider = override
        if provider == "ollama":
            host = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
            model = os.getenv("OLLAMA_MODEL") or agent_cfg.get("model", "llama3.2")
            llm = OllamaClient(host=host, model=model)
        elif provider == "anthropic":
            agent_model = agent_cfg.get("model", "")
            model = agent_model if ":" not in agent_model else _PROVIDER_DEFAULT_MODELS["anthropic"]
            llm = AnthropicClient(api_key=os.getenv("ANTHROPIC_API_KEY", ""), model=model)
        elif provider == "gemini":
            agent_model = agent_cfg.get("model", "")
            model = agent_model if ":" not in agent_model else _PROVIDER_DEFAULT_MODELS["gemini"]
            llm = GeminiClient(api_key=os.getenv("GEMINI_API_KEY", ""), model=model)
        else:
            raise ValueError(f"Unknown provider override: {provider}")
    else:
        provider = agent_cfg.get("provider", "anthropic")
        model = agent_cfg.get("model", _PROVIDER_DEFAULT_MODELS.get(provider, ""))
        if provider == "anthropic":
            llm = AnthropicClient(api_key=os.getenv("ANTHROPIC_API_KEY", ""), model=model)
        elif provider == "gemini":
            llm = GeminiClient(api_key=os.getenv("GEMINI_API_KEY", ""), model=model)
        elif provider == "ollama":
            host = agent_cfg.get("host", os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434"))
            llm = OllamaClient(host=host, model=model)
        else:
            raise ValueError(f"Unknown provider in agents.yaml for supervisor: {provider}")

    return provider, model, llm


_HTTP_PORTS = {80, 443, 8080, 8443, 8000, 8008, 8888, 3000, 4000, 5000, 9000, 9090}


def _http_services(kb: dict) -> list[str]:
    """
    Return a list of 'ip:port' strings for any HTTP/HTTPS service found in the
    knowledge base — either by service name or by well-known port number.
    """
    found: list[str] = []
    for addr, svc in kb.get("services", {}).items():
        if any(s in svc.lower() for s in ("http", "https", "web", "www")):
            found.append(addr)
    for ip, ports in kb.get("open_ports", {}).items():
        for port in ports:
            if port in _HTTP_PORTS:
                addr = f"{ip}:{port}"
                if addr not in found:
                    found.append(addr)
    return found


async def supervisor_node(state: TeamState) -> dict:
    """
    LangGraph node: decide which agent runs next (or end the session).

    HITL interrupts are triggered when:
      - exploit_attempts >= max_exploit_attempts
      - The LLM requests manual credential input
      - hitl_reason is already set from a previous node
    """
    config = state.get("config", {})
    settings = config.get("settings", {})
    max_attempts: int = settings.get("max_exploit_attempts", 5)
    context_limit: int = settings.get("context_limit_threshold", 80000)
    max_unproductive: int = settings.get("max_unproductive_turns", 4)
    autonomous: bool = bool(settings.get("autonomous", False))

    kb = state.get("knowledge_base", {})
    exploit_attempts = state.get("exploit_attempts", 0)
    current_estimate = state.get("context_token_estimate", 0)
    current_agent = state.get("current_agent", "none")
    completed_agents_now: list[str] = list(state.get("completed_agents") or [])

    # -----------------------------------------------------------------------
    # 1. Check if flag was already captured
    #    Ignore placeholder/template strings (e.g. "flag{STFUj...}") AND strings
    #    of the wrong shape for this challenge's flag_format (e.g. a `key{...}`
    #    decoy or a bare hex blob when the flag is `flag{...}`). Ending on either
    #    hands back a wrong flag and cuts investigation short.
    # -----------------------------------------------------------------------
    _flag_format = state.get("flag_format", "")
    flags = [f for f in kb.get("flags", [])
             if not _is_placeholder_flag(f) and _flag_matches_format(f, _flag_format)]
    if flags:
        logger.info("[Supervisor] Flag captured: %s. Mission complete!", flags)
        return {"next": "__end__"}

    # -----------------------------------------------------------------------
    # 1a. Unproductive-streak circuit breaker. If agents have taken several turns
    #     in a row without executing a single tool (emitting text but never
    #     acting — the exploit-spin failure mode), the team is stuck and more
    #     time won't help. End the challenge instead of burning the full timeout.
    # -----------------------------------------------------------------------
    unproductive = state.get("unproductive_streak", 0)
    if unproductive >= max_unproductive and not flags:
        if autonomous:
            logger.warning(
                "[Supervisor] %d consecutive unproductive turns (no tool calls) — "
                "team is stuck; ending challenge.", unproductive,
            )
            return {"next": "__end__"}
        human_response = interrupt({
            "reason": "stuck_no_tool_calls",
            "message": (
                f"The team has taken {unproductive} turns without running any tool.\n"
                f"Knowledge Base:\n{json.dumps(kb, indent=2)}\n\n"
                "Provide a concrete next command, tool, or hint to act on."
            ),
        })
        return {
            "messages": [{"role": "user", "content": f"[Human Operator]: {human_response}"}],
            "unproductive_streak": 0,
            "completed_agents": [],
            "hitl_reason": None,
            "next": "recon",
        }

    # -----------------------------------------------------------------------
    # 1e. Autonomous win scan — scan recent message text for flag patterns that
    #     the KB extractor may have missed (e.g. inside a summarized block).
    #     Only in autonomous mode to avoid unnecessary parsing overhead in HITL.
    # -----------------------------------------------------------------------
    if autonomous:
        import re as _re
        _FLAG_SCAN = _re.compile(
            r'(?:flag|key|ctf|htb|thm|picoctf|csaw|crypto|web|pwn|misc|rev|forensics'
            r'|rtcp|ductf|darkctf|bucket|nite|jctf|cyber)\{[^}]{1,200}\}',
            _re.IGNORECASE,
        )
        _HEX_SCAN = _re.compile(r'\b([0-9a-f]{32})\b', _re.IGNORECASE)
        messages = state.get("messages", [])
        # Only scan the last few messages — earlier ones are already processed.
        _scan_msgs = messages[-6:] if len(messages) > 6 else messages
        _found_flags: list[str] = []
        for _msg in _scan_msgs:
            _content = _msg.get("content", "")
            if isinstance(_content, list):
                _content = " ".join(
                    b.get("text", "") or b.get("content", "")
                    for b in _content if isinstance(b, dict)
                )
            _content = str(_content)
            _found_flags += _FLAG_SCAN.findall(_content)
            # Hex flags only when a flag-file reference is nearby
            if _re.search(r'(?:user|root|flag)\.txt', _content, _re.IGNORECASE):
                _found_flags += _HEX_SCAN.findall(_content)

        _found_flags = [f for f in _found_flags
                        if not _is_placeholder_flag(f) and _flag_matches_format(f, _flag_format)]
        if _found_flags:
            _existing = set(kb.get("flags", []))
            _new = [f for f in _found_flags if f not in _existing]
            if _new:
                logger.info(
                    "[Supervisor] Autonomous win scan found flags in message text: %s", _new
                )
                updated_kb = dict(kb)
                updated_kb["flags"] = list(_existing | set(_new))
                return {
                    "knowledge_base": updated_kb,
                    "next": "__end__",
                }

    # -----------------------------------------------------------------------
    # 1b. Shell / foothold already obtained → skip straight to privesc
    # -----------------------------------------------------------------------
    shells = kb.get("shells", [])
    if shells and current_agent not in ("privesc", "refusal_specialist"):
        logger.info("[Supervisor] Foothold detected (%s) → privesc", shells)
        return {"next": "privesc"}

    # -----------------------------------------------------------------------
    # 1c. Confirmed exploit path in attack_surface → route to exploit immediately.
    #     Prevents recon/webexplorer from continuing to enumerate after a clear
    #     attack vector (RCE, backdoor, etc.) has already been identified.
    # -----------------------------------------------------------------------
    _hv_signals = (
        "exploit", "backdoor", "rce", "remote code exec",
        "command injection", "sql inject", "path travers",
        "lfi", "rfi", "ssrf", "xxe", "auth bypass",
        "privilege escal", "arbitrary code", "arbitrary command",
        "unauthenticated", "unauthor",
    )
    attack_surface_entries = kb.get("attack_surface", [])
    _has_confirmed_exploit = any(
        any(s in e.lower() for s in _hv_signals) for e in attack_surface_entries
    )
    if (_has_confirmed_exploit
            and not shells
            and current_agent in ("recon", "webexplorer", "vulnsearch", "refusal_specialist")):
        logger.info("[Supervisor] Confirmed exploit path found → exploit (skipping further recon)")
        return {"next": "exploit"}

    # -----------------------------------------------------------------------
    # 1d. Tech stack known + vuln research done + exploit not yet attempted
    #     The attack_surface may be empty because vulnsearch ran but found nothing
    #     structured to add — don't keep looping through research agents.
    #     Move directly to exploit so the agent can work from the full message
    #     history and its own searchsploit/web access.
    # -----------------------------------------------------------------------
    if (kb.get("tech_stack")
            and "vulnsearch" in completed_agents_now
            and "exploit" not in completed_agents_now
            and not shells):
        logger.info("[Supervisor] Tech stack known + vuln research done + exploit not yet attempted → exploit")
        return {"next": "exploit"}

    # -----------------------------------------------------------------------
    # 2. Context compaction check — route to summarizer before hitting limits
    # -----------------------------------------------------------------------
    if current_estimate > context_limit:
        logger.info("[Supervisor] Context limit approaching (~%s tokens). Triggering compaction.", f"{current_estimate:,}")
        return {"next": "compaction"}

    # -----------------------------------------------------------------------
    # 3. Exploit loop detection — HITL interrupt (skipped in autonomous mode)
    # -----------------------------------------------------------------------
    if exploit_attempts >= max_attempts:
        if autonomous:
            # In autonomous mode: reset the counter and route to recon for a
            # fresh angle rather than looping the exploit agent forever.
            logger.info(
                "[Supervisor] Autonomous mode — exploit loop (%d failures) detected. "
                "Resetting counter and routing to recon for a fresh angle.",
                exploit_attempts,
            )
            refreshed_completed = [a for a in completed_agents_now if a not in ("exploit", "recon")]
            return {
                "exploit_attempts": 0,
                "completed_agents": refreshed_completed,
                "hitl_reason": None,
                "next": "recon",
            }
        hitl_payload = {
            "reason": "exploit_loop",
            "message": (
                f"The exploit agent has failed {exploit_attempts} consecutive times. "
                f"Please provide hints, credentials, or a new attack vector.\n"
                f"Knowledge Base:\n{json.dumps(kb, indent=2)}"
            ),
        }
        logger.info("[Supervisor] Exploit loop detected (%s failures). Requesting human input...", exploit_attempts)
        human_response = interrupt(hitl_payload)
        # Resume after human provides input — inject as a user message, reset counter,
        # and clear exploit from completed_agents so it can run again with new direction.
        new_message = {
            "role": "user",
            "content": f"[Human Operator]: {human_response}",
        }
        refreshed_completed = [a for a in completed_agents_now if a != "exploit"]
        return {
            "messages": [new_message],
            "exploit_attempts": 0,
            "completed_agents": refreshed_completed,
            "hitl_reason": None,
            "next": "exploit",
        }

    # -----------------------------------------------------------------------
    # 4. HTTP auto-trigger — route to webexplorer the moment an HTTP port is
    #    discovered by recon, before asking the LLM to decide anything.
    #    Only fires immediately after recon so webexplorer isn't re-triggered
    #    on every supervisor cycle (including after refusal_specialist).
    # -----------------------------------------------------------------------
    http_svcs = _http_services(kb)
    # Only auto-trigger webexplorer immediately after recon — NOT after refusal_specialist,
    # which may have just corrected a webexplorer turn (causing an infinite loop).
    if http_svcs and current_agent == "recon":
        logger.info("[Supervisor] HTTP service(s) detected on %s → webexplorer", http_svcs)
        return {"next": "webexplorer"}

    # -----------------------------------------------------------------------
    # 4b. Tech-stack vuln research trigger — after webexplorer or recon
    #     populates tech_stack, route to vulnsearch for a CVE/searchsploit
    #     research pass before any active exploitation.
    #     Only fires once: skip if attack_surface already contains CVE or
    #     EDB references from a previous vulnsearch run.
    # -----------------------------------------------------------------------
    tech_stack = kb.get("tech_stack", {})
    attack_surface = kb.get("attack_surface", [])
    cve_researched = any(
        "cve" in entry.lower() or "edb-" in entry.lower() or "searchsploit" in entry.lower()
        for entry in attack_surface
    )
    if (tech_stack
            and not cve_researched
            and current_agent in ("webexplorer", "recon")
            and "vulnsearch" not in completed_agents_now):
        logger.info("[Supervisor] tech_stack populated (%d entries), no CVE research yet → vulnsearch",
                    len(tech_stack))
        return {"next": "vulnsearch"}

    # -----------------------------------------------------------------------
    # 4c. Completed-agent loop guard — if every work agent has completed and
    #     no flag exists, we're stuck; trigger HITL so the human can provide
    #     a new direction (or in autonomous mode, reset and retry from recon).
    # -----------------------------------------------------------------------
    all_work_agents = {"recon", "webexplorer", "vulnsearch", "exploit", "privesc"}
    if all_work_agents.issubset(set(completed_agents_now)) and not flags:
        if autonomous:
            logger.warning(
                "[Supervisor] Autonomous mode — all agents completed but no flag. "
                "Resetting completed_agents and retrying from recon."
            )
            return {
                "completed_agents": [],
                "exploit_attempts": 0,
                "hitl_reason": None,
                "next": "recon",
            }
        logger.warning("[Supervisor] All agents completed but no flag — requesting human input")
        human_response = interrupt({
            "reason": "all_agents_complete_no_flag",
            "message": (
                "All agents have signalled TASK COMPLETE but no flag was captured.\n"
                f"Knowledge Base:\n{json.dumps(kb, indent=2)}\n\n"
                "Please provide a new attack direction, credentials, or hints."
            ),
        })
        return {
            "messages": [{"role": "user", "content": f"[Human Operator]: {human_response}"}],
            "completed_agents": [],   # reset so agents can run again
            "hitl_reason": None,
            "next": "recon",
        }

    # -----------------------------------------------------------------------
    # 5. Propagate an existing hitl_reason (suppressed in autonomous mode)
    # -----------------------------------------------------------------------
    hitl_reason = state.get("hitl_reason")
    if hitl_reason:
        if autonomous:
            logger.info(
                "[Supervisor] Autonomous mode — ignoring hitl_reason and continuing: %s",
                str(hitl_reason)[:120],
            )
            return {
                "hitl_reason": None,
                "next": "exploit",
            }
        human_response = interrupt({
            "reason": "manual_intervention",
            "message": hitl_reason,
        })
        new_message = {
            "role": "user",
            "content": f"[Human Operator]: {human_response}",
        }
        # Clear exploit from completed so it can retry with the human's guidance.
        refreshed_completed = [a for a in completed_agents_now if a != "exploit"]
        return {
            "messages": [new_message],
            "completed_agents": refreshed_completed,
            "hitl_reason": None,
            "next": "exploit",
        }

    # -----------------------------------------------------------------------
    # 6. Ask the supervisor LLM to route
    # -----------------------------------------------------------------------
    sup_cfg = config.get("agents", {}).get("supervisor", {})
    system_prompt = sup_cfg.get("system_prompt", "Route the CTF team.")

    provider, model, llm = _resolve_llm(state, sup_cfg)
    logger.info("[supervisor] provider=%s model=%s", provider, model)

    # Build a concise status digest for the supervisor LLM
    messages = state.get("messages", [])
    recent_messages = messages[-6:] if len(messages) > 6 else messages

    recent_text_parts = []
    for m in recent_messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
        recent_text_parts.append(f"[{role.upper()}]: {str(content)[:500]}")
    recent_digest = "\n".join(recent_text_parts)

    # Agents that have already signalled a substantive TASK COMPLETE — read
    # directly from state (maintained by _run_agent_loop in agents.py).
    completed_hint = ""
    if completed_agents_now:
        done_list = ", ".join(completed_agents_now)
        completed_hint = (
            f"\nAgents that have already reported TASK COMPLETE this session: {done_list}. "
            f"Do NOT re-route to them unless the knowledge base has materially changed "
            f"in a way that specifically requires their skills again.\n"
            f"Available (not yet run): "
            f"{', '.join(a for a in ['recon','webexplorer','vulnsearch','exploit','privesc'] if a not in completed_agents_now) or 'none'}\n"
        )

    # Build a concise work-history digest from the new KB fields
    exploit_history = kb.get("exploit_history", [])
    visited_urls = kb.get("visited_urls", [])
    scan_history_summary = {}
    for ip_key, entries in (kb.get("scan_history") or {}).items():
        scan_history_summary[ip_key] = entries[-10:]  # last 10 per host to keep prompt short

    work_history_block = ""
    if shells:
        work_history_block += f"\nShells/foothold already obtained: {json.dumps(shells)}"
    if exploit_history:
        work_history_block += f"\nFailed exploit attempts: {json.dumps(exploit_history[-10:])}"
    if visited_urls:
        work_history_block += f"\nURLs already visited: {len(visited_urls)} (not listing all)"
    if scan_history_summary:
        work_history_block += f"\nRecent scan history: {json.dumps(scan_history_summary)}"

    routing_prompt = (
        f"Current Knowledge Base (summary):\n"
        f"  IPs: {kb.get('ips', [])}\n"
        f"  Open ports: {kb.get('open_ports', {})}\n"
        f"  Services: {kb.get('services', {})}\n"
        f"  Tech stack: {kb.get('tech_stack', {})}\n"
        f"  Attack surface ({len(kb.get('attack_surface', []))} entries): "
        f"{kb.get('attack_surface', [])[:10]}\n"
        f"  Credentials: {len(kb.get('credentials', []))} found\n"
        f"  Flags: {kb.get('flags', [])}\n"
        f"{work_history_block}\n\n"
        f"Recent Activity:\n{recent_digest}\n\n"
        f"Exploit attempts so far: {exploit_attempts}\n"
        f"Current agent: {state.get('current_agent', 'none')}\n"
        f"{completed_hint}"
        f"\nDecide the next action. Respond with ONLY a JSON object:\n"
        f'{{"next": "<recon|webexplorer|vulnsearch|exploit|privesc|FINISH>", "reasoning": "<brief reason>", "hitl_reason": null}}'
    )

    routing_messages = [{"role": "user", "content": routing_prompt}]
    try:
        response = await llm.generate_response(
            messages=routing_messages,
            tools=[],
            system_prompt=system_prompt,
        )
    except Exception as exc:
        # Routing LLM failed even after client retries — don't crash the challenge.
        # Fall back to a knowledge-base heuristic so the graph keeps moving.
        fallback = "exploit" if (kb.get("attack_surface") or kb.get("open_ports")) else "recon"
        logger.error("[Supervisor] routing LLM failed: %s — falling back to %s", exc, fallback)
        return {"next": fallback, "hitl_reason": None}

    # Parse the routing decision
    tool_calls = llm.parse_tool_calls(response)
    assistant_msg = llm.make_assistant_message(response)

    # Extract text from the response
    decision_text = ""
    if isinstance(assistant_msg.get("content"), list):
        for block in assistant_msg["content"]:
            if isinstance(block, dict) and block.get("type") == "text":
                decision_text += block.get("text", "")
    elif isinstance(assistant_msg.get("content"), str):
        decision_text = assistant_msg["content"]

    # Parse JSON from LLM response
    next_agent = "recon"  # safe default
    reasoning = ""
    new_hitl = None

    try:
        # Find JSON object in the response
        import re
        json_match = re.search(r'\{[^{}]+\}', decision_text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            raw_next = parsed.get("next", "recon").lower()
            reasoning = parsed.get("reasoning", "")
            new_hitl = parsed.get("hitl_reason")

            if raw_next == "finish":
                next_agent = "__end__"
            elif raw_next in ("recon", "webexplorer", "vulnsearch", "exploit", "privesc", "refusal_specialist"):
                next_agent = raw_next
            else:
                next_agent = "recon"
    except (json.JSONDecodeError, AttributeError):
        # Fallback: keyword search in response
        if "refusal_specialist" in decision_text.lower() or "refusal specialist" in decision_text.lower():
            next_agent = "refusal_specialist"
        elif "webexplorer" in decision_text.lower() or "web explorer" in decision_text.lower():
            next_agent = "webexplorer"
        elif "exploit" in decision_text.lower():
            next_agent = "exploit"
        elif "privesc" in decision_text.lower() or "privilege" in decision_text.lower():
            next_agent = "privesc"
        elif "finish" in decision_text.lower() or "complete" in decision_text.lower():
            next_agent = "__end__"
        else:
            next_agent = "recon"

    # No-flag guard: the LLM must not end the session without a captured flag.
    # If it tries, solicit human feedback instead — the team may have stalled.
    # Use the placeholder-filtered `flags`, not raw kb — a template flag is not a win.
    if next_agent == "__end__" and not flags:
        logger.warning(
            "[Supervisor] LLM requested FINISH but no flag captured — %s",
            "continuing autonomously" if autonomous else "triggering HITL",
        )
        if autonomous:
            refreshed_completed = [a for a in completed_agents_now if a != "exploit"]
            return {
                "completed_agents": refreshed_completed,
                "exploit_attempts": 0,
                "hitl_reason": None,
                "next": "exploit",
            }
        human_response = interrupt({
            "reason": "finish_without_flag",
            "message": (
                "The supervisor decided to finish, but no flag has been captured.\n"
                f"Knowledge Base:\n{json.dumps(kb, indent=2)}\n\n"
                "Please provide a new attack direction, missing credentials, or hints "
                "so the team can continue."
            ),
        })
        refreshed_completed = [a for a in completed_agents_now if a != "exploit"]
        return {
            "messages": [{"role": "user", "content": f"[Human Operator]: {human_response}"}],
            "completed_agents": refreshed_completed,
            "exploit_attempts": 0,
            "hitl_reason": None,
            "next": "exploit",
        }

    # Hard guard: if the LLM wants to route back to an already-completed agent,
    # find the highest-priority uncompleted work agent instead.  This is
    # code-enforced — the LLM is free to ignore prompt hints but not this guard.
    completed_agents_guard = completed_agents_now  # already computed above
    if next_agent in completed_agents_guard and next_agent != "__end__":
        original_next = next_agent
        # Walk priority order — pick the first agent that hasn't run yet.
        # If all are completed, step 4c (HITL) will catch it on the next
        # supervisor pass; routing to "__end__" here would be too drastic.
        _priority = ["exploit", "privesc", "vulnsearch", "webexplorer", "recon"]
        for _candidate in _priority:
            if _candidate not in completed_agents_guard:
                next_agent = _candidate
                break
        else:
            # Every work agent is marked complete — let the graph reach END
            # (the all-agents HITL at step 4c above should have caught this).
            next_agent = "__end__"
        logger.warning("[Supervisor] Hard guard: LLM wanted %s (already completed) → redirecting to %s",
                       original_next.upper(), next_agent.upper())

    # Loop guard: if the refusal_specialist just ran, the supervisor must not
    # route back to it again — the specialist already had its chance.  Pick the
    # most appropriate work agent from the knowledge base instead.
    if next_agent == "refusal_specialist" and state.get("current_agent") == "refusal_specialist":
        kb_now = state.get("knowledge_base", {})
        if [f for f in kb_now.get("flags", [])
                if not _is_placeholder_flag(f) and _flag_matches_format(f, _flag_format)]:
            next_agent = "__end__"
        elif kb_now.get("open_ports") or kb_now.get("attack_surface"):
            next_agent = "exploit"
        else:
            next_agent = "recon"
        logger.warning("[Supervisor] Loop guard: refusal_specialist→refusal_specialist prevented, "
                       "redirecting to %s", next_agent.upper())

    logger.info("[Supervisor] → %s%s", next_agent.upper(), f" | {reasoning[:120]}" if reasoning else "")

    return {
        "next": next_agent,
        "hitl_reason": new_hitl,
    }


def route_from_supervisor(state: TeamState) -> str:
    """
    LangGraph conditional edge function.
    Reads `state['next']` and returns the destination node name.
    """
    return state.get("next", "recon")
