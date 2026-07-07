"""
Specialized agent nodes: Recon, Exploit, PrivEsc.

Each node receives TeamState, runs an internal tool-call loop using the
configured LLM and available MCP tools, then returns a state update including:
  - New messages (tool calls + results)
  - Updated knowledge_base
  - Updated context_token_estimate
  - Updated exploit_attempts (Exploit agent only)
"""
from __future__ import annotations

import json
import re
import sys
import os
from typing import Any, Dict, List, Optional, Tuple

from .state import TeamState, KnowledgeBase
from .summarizer import maybe_summarize

# Import existing LLM clients
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from llms import AnthropicClient, GeminiClient, OllamaClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_llm_client(provider: str, agent_cfg: dict):
    """Instantiate the correct LLM client for a given agent config."""
    model = agent_cfg.get("model")
    if provider == "anthropic":
        return AnthropicClient(model=model)
    elif provider == "gemini":
        return GeminiClient(model=model)
    elif provider == "ollama":
        host = agent_cfg.get("host", "http://10.0.2.2:11434")
        return OllamaClient(host=host, model=model)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def _estimate_tokens(messages: list) -> int:
    """Rough token estimate: total chars / 4."""
    return sum(len(str(m)) for m in messages) // 4


# Concrete signals that an exploit tool call clearly failed. Kept conservative:
# a false positive inflates the exploit failure counter and can trip a spurious
# HITL breakpoint, so we avoid bare substrings like "error" that appear in plenty
# of successful output.
_FAILURE_SIGNALS = (
    "connection refused", "connection timed out", "connection reset",
    "no route to host", "could not connect", "unable to connect",
    "authentication failed", "login failed", "permission denied",
    "access denied", "exploit failed", "exploit completed, but no session",
    "no session was created", "command not found",
    "traceback (most recent call last)",
)


def _looks_like_failure(result: str) -> bool:
    """Return True if the tool output contains a concrete failure signal."""
    head = result.lower()[:1000]
    return any(sig in head for sig in _FAILURE_SIGNALS)


def _cred_keys(kb: KnowledgeBase) -> set:
    """Stable set of credential fingerprints for progress comparison."""
    return {json.dumps(c, sort_keys=True) for c in kb.get("credentials", [])}


def _made_progress(before: KnowledgeBase, after: KnowledgeBase) -> bool:
    """Did this agent turn surface a genuinely new high-value finding?"""
    return bool(
        (set(after.get("flags", [])) - set(before.get("flags", [])))
        or (_cred_keys(after) - _cred_keys(before))
        or (set(after.get("attack_surface", [])) - set(before.get("attack_surface", [])))
    )


def _extract_kb_updates(tool_name: str, tool_result: str, kb: KnowledgeBase) -> KnowledgeBase:
    """
    Parse tool output for common security findings and update the knowledge base.

    Returns a new KnowledgeBase dict with any new discoveries merged in. Nested
    containers are copied rather than mutated in place — the input `kb` may be a
    checkpointed state object shared across nodes, and mutating it in place would
    corrupt the persisted graph state.
    """
    kb = dict(kb)  # shallow copy; nested containers copied on write below

    # Discover IPs
    ip_pattern = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
    found_ips = ip_pattern.findall(tool_result)
    if found_ips:
        existing = set(kb.get("ips", []))
        new_ips = [ip for ip in found_ips if ip not in existing
                   and not ip.startswith(("127.", "0.", "255."))]
        if new_ips:
            kb["ips"] = list(existing | set(new_ips))

    # Discover open ports (from nmap-style output: "PORT/tcp open")
    port_pattern = re.compile(r'(\d+)/(tcp|udp)\s+open\s+(\S+)?', re.IGNORECASE)
    for match in port_pattern.finditer(tool_result):
        port = int(match.group(1))
        service = (match.group(3) or "unknown").strip()
        # Associate with the first known IP or "target"
        target_ip = (kb.get("ips") or ["target"])[0]
        ports = dict(kb.get("open_ports", {}))
        ports_for_ip = ports.get(target_ip, [])
        if port not in ports_for_ip:
            ports[target_ip] = sorted(set(ports_for_ip + [port]))
            kb["open_ports"] = ports

        # Record service
        key = f"{target_ip}:{port}"
        services = dict(kb.get("services", {}))
        if key not in services:
            services[key] = service
            kb["services"] = services

    # Detect credentials (loose heuristic pairing users with passwords)
    users = re.findall(r'(?:username|user|login)\s*[=:]\s*(\S+)', tool_result, re.IGNORECASE)
    passwords = re.findall(r'(?:password|passwd|pwd)\s*[=:]\s*(\S+)', tool_result, re.IGNORECASE)
    if users and passwords:
        creds = list(kb.get("credentials", []))
        for u, p in zip(users, passwords):
            entry = {"user": u, "pass": p}
            if entry not in creds:
                creds.append(entry)
        kb["credentials"] = creds

    # Detect flags
    flag_pattern = re.compile(r'(?:HTB|FLAG|CTF)\{[^}]+\}', re.IGNORECASE)
    flags_found = flag_pattern.findall(tool_result)
    if flags_found:
        existing_flags = set(kb.get("flags", []))
        kb["flags"] = list(existing_flags | set(flags_found))

    # Interesting attack surface (directories, files from gobuster/ferox)
    dir_pattern = re.compile(r'(?:Status: 200|Found:)\s*(\/\S+)', re.IGNORECASE)
    found_paths = dir_pattern.findall(tool_result)
    if found_paths:
        surface = set(kb.get("attack_surface", []))
        kb["attack_surface"] = list(surface | set(found_paths[:20]))  # cap at 20

    return kb


async def _run_agent_loop(
    agent_name: str,
    state: TeamState,
    tools: list,
    mcp_client: Any,
) -> dict:
    """
    Core ReAct loop shared by all specialized agents.
    Runs the LLM with tool calls until it produces a final text response.
    Returns a partial state update dict.
    """
    config = state.get("config", {})
    agent_cfg = config.get("agents", {}).get(agent_name, {})
    provider = agent_cfg.get("provider", state.get("provider", "anthropic"))
    system_prompt = agent_cfg.get("system_prompt", "")

    # Build LLM client
    llm = _build_llm_client(provider, agent_cfg)

    # Fetch the raw MCP tool schemas. Each client's generate_response() calls
    # its own format_tools() internally, so we pass the raw schemas through
    # (pre-formatting here would double-format and strip the input schemas).
    raw_tools = await mcp_client.list_tools()

    # Inject knowledge base into system prompt
    kb = state.get("knowledge_base", {})
    kb_text = json.dumps(kb, indent=2)
    full_system = (
        f"{system_prompt}\n\n"
        f"=== KNOWLEDGE BASE (shared with team) ===\n{kb_text}\n"
        f"=== CURRENT TASK ===\n{state.get('task', 'Hack the target machine.')}"
    )

    # Start from current conversation history
    messages = list(state.get("messages", []))
    new_messages: list = []
    updated_kb = dict(kb)
    exploit_delta = 0

    # ReAct loop: call LLM → execute tools → repeat
    max_iterations = 20
    for iteration in range(max_iterations):
        response = await llm.generate_response(
            messages=messages + new_messages,
            tools=raw_tools,
            system=full_system,
        )

        tool_calls = llm.parse_tool_calls(response)
        assistant_msg = llm.make_assistant_message(response)
        new_messages.append(assistant_msg)

        if not tool_calls:
            # No more tool calls — agent is done with its sub-task
            break

        # Execute each tool call. parse_tool_calls() normalizes every provider
        # to {"id": ..., "name": ..., "arguments": ...}.
        tool_results: list = []
        for tc in tool_calls:
            tool_name = tc.get("name", "")
            raw_args = tc.get("arguments", {})
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    raw_args = {}

            print(f"  [{agent_name}] → {tool_name}({json.dumps(raw_args)[:120]})")

            raw_result = await mcp_client.call_tool(tool_name, raw_args)

            # Auto-summarize large outputs
            result = maybe_summarize(raw_result, config)

            if result != raw_result:
                print(f"  [Summarizer] Condensed {len(raw_result):,} → {len(result):,} chars")

            # Track exploit failures (used to trigger a HITL breakpoint when the
            # exploit agent is stuck).
            if agent_name == "exploit" and _looks_like_failure(result):
                exploit_delta += 1

            # Update knowledge base from tool output
            updated_kb = _extract_kb_updates(tool_name, result, updated_kb)
            tool_results.append((tc, result))

        # Build tool result messages (provider-specific).
        if provider == "anthropic":
            # Anthropic bundles every tool_result block into one user message.
            content_blocks = [
                llm.make_tool_result_message(tc.get("id"), result)
                for tc, result in tool_results
            ]
            new_messages.append({"role": "user", "content": content_blocks})
        else:
            # Gemini and Ollama: one message per result, keyed by tool name.
            for tc, result in tool_results:
                new_messages.append(
                    llm.make_tool_result_message(tc.get("name", ""), result)
                )

    # Compute new token estimate
    all_messages = (messages + new_messages)
    new_estimate = _estimate_tokens(all_messages)

    # Exploit-loop bookkeeping: accumulate consecutive failures, but reset the
    # streak whenever the team surfaces a genuinely new high-value finding — a
    # breakthrough means we're no longer stuck, so we shouldn't trip HITL.
    prev_attempts = state.get("exploit_attempts", 0)
    if _made_progress(kb, updated_kb):
        exploit_attempts = 0
    else:
        exploit_attempts = prev_attempts + exploit_delta

    return {
        "messages": new_messages,
        "knowledge_base": updated_kb,
        "current_agent": agent_name,
        "context_token_estimate": new_estimate,
        "exploit_attempts": exploit_attempts,
    }


# ---------------------------------------------------------------------------
# Agent node factories — called by graph.py when building the StateGraph
# ---------------------------------------------------------------------------

def make_recon_node(mcp_client: Any, tools: list):
    """Return an async node function for the Recon agent."""
    async def recon_node(state: TeamState) -> dict:
        print(f"\n[Recon Agent] Starting reconnaissance...")
        return await _run_agent_loop("recon", state, tools, mcp_client)
    return recon_node


def make_exploit_node(mcp_client: Any, tools: list):
    """Return an async node function for the Exploit agent."""
    async def exploit_node(state: TeamState) -> dict:
        print(f"\n[Exploit Agent] Attempting exploitation...")
        return await _run_agent_loop("exploit", state, tools, mcp_client)
    return exploit_node


def make_privesc_node(mcp_client: Any, tools: list):
    """Return an async node function for the PrivEsc agent."""
    async def privesc_node(state: TeamState) -> dict:
        print(f"\n[PrivEsc Agent] Escalating privileges...")
        return await _run_agent_loop("privesc", state, tools, mcp_client)
    return privesc_node
