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

logger = logging.getLogger("filthy_clanker")


# Possible routing destinations
NEXT_OPTIONS = Literal["recon", "exploit", "privesc", "compaction", "__end__"]


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
            host = os.getenv("OLLAMA_HOST", "http://10.0.2.2:11434")
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
            host = agent_cfg.get("host", os.getenv("OLLAMA_HOST", "http://10.0.2.2:11434"))
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

    kb = state.get("knowledge_base", {})
    exploit_attempts = state.get("exploit_attempts", 0)
    current_estimate = state.get("context_token_estimate", 0)

    # -----------------------------------------------------------------------
    # 1. Check if flag was already captured
    # -----------------------------------------------------------------------
    flags = kb.get("flags", [])
    if flags:
        logger.info("[Supervisor] Flag captured: %s. Mission complete!", flags)
        return {"next": "__end__"}

    # -----------------------------------------------------------------------
    # 2. Context compaction check — route to summarizer before hitting limits
    # -----------------------------------------------------------------------
    if current_estimate > context_limit:
        logger.info("[Supervisor] Context limit approaching (~%s tokens). Triggering compaction.", f"{current_estimate:,}")
        return {"next": "compaction"}

    # -----------------------------------------------------------------------
    # 3. Exploit loop detection — HITL interrupt
    # -----------------------------------------------------------------------
    if exploit_attempts >= max_attempts:
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
        # Resume after human provides input — inject as a user message and reset counter
        new_message = {
            "role": "user",
            "content": f"[Human Operator]: {human_response}",
        }
        return {
            "messages": [new_message],
            "exploit_attempts": 0,
            "hitl_reason": None,
            "next": "exploit",
        }

    # -----------------------------------------------------------------------
    # 4. HTTP auto-trigger — route to webexplorer the moment an HTTP port is
    #    discovered by recon, before asking the LLM to decide anything.
    #    Only fires when recon (or refusal_specialist correcting recon) just ran,
    #    so webexplorer isn't re-triggered on every supervisor cycle.
    # -----------------------------------------------------------------------
    current_agent = state.get("current_agent", "none")
    http_svcs = _http_services(kb)
    if http_svcs and current_agent in ("recon", "refusal_specialist"):
        logger.info("[Supervisor] HTTP service(s) detected on %s → webexplorer", http_svcs)
        return {"next": "webexplorer"}

    # -----------------------------------------------------------------------
    # 4b. Tech-stack vuln research trigger — after webexplorer (or recon)
    #     populates tech_stack, route to exploit for a CVE research pass before
    #     any active exploitation.  Only fires once: skip if attack_surface
    #     already contains CVE references from a previous research pass.
    # -----------------------------------------------------------------------
    tech_stack = kb.get("tech_stack", {})
    attack_surface = kb.get("attack_surface", [])
    cve_researched = any(
        "cve" in entry.lower() or "searchsploit" in entry.lower()
        for entry in attack_surface
    )
    if (tech_stack
            and not cve_researched
            and current_agent in ("webexplorer", "recon", "refusal_specialist")):
        logger.info("[Supervisor] tech_stack populated (%d entries), no CVE research yet → exploit",
                    len(tech_stack))
        return {"next": "exploit"}

    # -----------------------------------------------------------------------
    # 5. Propagate an existing hitl_reason
    # -----------------------------------------------------------------------
    hitl_reason = state.get("hitl_reason")
    if hitl_reason:
        human_response = interrupt({
            "reason": "manual_intervention",
            "message": hitl_reason,
        })
        new_message = {
            "role": "user",
            "content": f"[Human Operator]: {human_response}",
        }
        return {
            "messages": [new_message],
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

    routing_prompt = (
        f"Current Knowledge Base:\n{json.dumps(kb, indent=2)}\n\n"
        f"Recent Activity:\n{recent_digest}\n\n"
        f"Exploit attempts so far: {exploit_attempts}\n"
        f"Current agent: {state.get('current_agent', 'none')}\n\n"
        f"Decide the next action. Respond with ONLY a JSON object:\n"
        f'{{"next": "<recon|webexplorer|exploit|privesc|FINISH>", "reasoning": "<brief reason>", "hitl_reason": null}}'
    )

    routing_messages = [{"role": "user", "content": routing_prompt}]
    response = await llm.generate_response(
        messages=routing_messages,
        tools=[],
        system_prompt=system_prompt,
    )

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
            elif raw_next in ("recon", "webexplorer", "exploit", "privesc", "refusal_specialist"):
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

    # Loop guard: if the refusal_specialist just ran, the supervisor must not
    # route back to it again — the specialist already had its chance.  Pick the
    # most appropriate work agent from the knowledge base instead.
    if next_agent == "refusal_specialist" and state.get("current_agent") == "refusal_specialist":
        kb_now = state.get("knowledge_base", {})
        if kb_now.get("flags"):
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
