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
import sys
import os
from typing import Any, Literal

from langgraph.types import interrupt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from llms import AnthropicClient, GeminiClient, OllamaClient

from .state import TeamState


# Possible routing destinations
NEXT_OPTIONS = Literal["recon", "exploit", "privesc", "compaction", "__end__"]


def _build_llm_client(provider: str, agent_cfg: dict):
    model = agent_cfg.get("model")
    if provider == "anthropic":
        return AnthropicClient(model=model)
    elif provider == "gemini":
        return GeminiClient(model=model)
    elif provider == "ollama":
        host = agent_cfg.get("host", "http://10.0.2.2:11434")
        return OllamaClient(host=host, model=model)
    raise ValueError(f"Unknown provider: {provider}")


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
        print(f"\n[Supervisor] Flag captured: {flags}. Mission complete!")
        return {"next": "__end__"}

    # -----------------------------------------------------------------------
    # 2. Context compaction check — route to summarizer before hitting limits
    # -----------------------------------------------------------------------
    if current_estimate > context_limit:
        print(f"\n[Supervisor] Context limit approaching "
              f"(~{current_estimate:,} tokens). Triggering compaction.")
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
        print(f"\n[Supervisor] Exploit loop detected ({exploit_attempts} failures). "
              f"Requesting human input...")
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
    # 4. Propagate an existing hitl_reason
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
    # 5. Ask the supervisor LLM to route
    # -----------------------------------------------------------------------
    sup_cfg = config.get("agents", {}).get("supervisor", {})
    provider = sup_cfg.get("provider", state.get("provider", "anthropic"))
    system_prompt = sup_cfg.get("system_prompt", "Route the CTF team.")

    llm = _build_llm_client(provider, sup_cfg)

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
        f'{{"next": "<recon|exploit|privesc|FINISH>", "reasoning": "<brief reason>", "hitl_reason": null}}'
    )

    routing_messages = [{"role": "user", "content": routing_prompt}]
    response = await llm.generate_response(
        messages=routing_messages,
        tools=[],
        system=system_prompt,
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
            elif raw_next in ("recon", "exploit", "privesc"):
                next_agent = raw_next
            else:
                next_agent = "recon"
    except (json.JSONDecodeError, AttributeError):
        # Fallback: keyword search in response
        if "exploit" in decision_text.lower():
            next_agent = "exploit"
        elif "privesc" in decision_text.lower() or "privilege" in decision_text.lower():
            next_agent = "privesc"
        elif "finish" in decision_text.lower() or "complete" in decision_text.lower():
            next_agent = "__end__"
        else:
            next_agent = "recon"

    print(f"\n[Supervisor] → {next_agent.upper()}"
          + (f" | {reasoning[:120]}" if reasoning else ""))

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
