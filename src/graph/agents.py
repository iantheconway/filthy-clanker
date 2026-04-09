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

import fnmatch
import json
import logging
import re
import sys
import os
from typing import Any, Dict, List, Optional, Tuple

from .state import TeamState, KnowledgeBase
from .summarizer import maybe_summarize

# Import existing LLM clients
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from llms import AnthropicClient, GeminiClient, OllamaClient

logger = logging.getLogger("filthy_clanker")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Fallback models used when a global provider override is active but the
# per-agent model name is incompatible (e.g. "gemma4:e2b" can't run on Anthropic).
_PROVIDER_DEFAULT_MODELS = {
    "anthropic": "claude-opus-4-6",
    "gemini": "gemini-2.5-flash",
    "ollama": "llama3.2",
}


def _resolve_llm(state: TeamState, agent_cfg: dict) -> tuple[str, str, Any]:
    """
    Determine (provider, model, llm_client) for an agent invocation.

    If state["provider"] is set it acts as a global override — useful for
    switching all agents to Ollama for offline testing.  Otherwise each agent
    uses its own provider / model / host from agents.yaml.
    """
    override = state.get("provider")  # None → use per-agent config

    if override:
        provider = override
        if provider == "ollama":
            host = os.getenv("OLLAMA_HOST", "http://10.0.2.2:11434")
            # OLLAMA_MODEL set at startup by the interactive model picker
            model = os.getenv("OLLAMA_MODEL") or agent_cfg.get("model", "llama3.2")
            llm = OllamaClient(host=host, model=model)
        elif provider == "anthropic":
            agent_model = agent_cfg.get("model", "")
            # Ollama-style model names (contain ":") can't be used with Anthropic
            model = agent_model if ":" not in agent_model else _PROVIDER_DEFAULT_MODELS["anthropic"]
            llm = AnthropicClient(api_key=os.getenv("ANTHROPIC_API_KEY", ""), model=model)
        elif provider == "gemini":
            agent_model = agent_cfg.get("model", "")
            model = agent_model if ":" not in agent_model else _PROVIDER_DEFAULT_MODELS["gemini"]
            llm = GeminiClient(api_key=os.getenv("GEMINI_API_KEY", ""), model=model)
        else:
            raise ValueError(f"Unknown provider override: {provider}")
    else:
        # Per-agent config — provider, model, and host all come from agents.yaml
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
            raise ValueError(f"Unknown provider in agents.yaml for this agent: {provider}")

    return provider, model, llm


def _estimate_tokens(messages: list) -> int:
    """Rough token estimate: total chars / 4."""
    return sum(len(str(m)) for m in messages) // 4


def _extract_kb_updates(tool_name: str, tool_result: str, kb: KnowledgeBase) -> KnowledgeBase:
    """
    Parse tool output for common security findings and update the knowledge base.
    Returns a new (shallow-copied) KnowledgeBase dict with any new discoveries merged in.
    """
    kb = dict(kb)  # shallow copy
    text = tool_result.lower()

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
        ports = kb.get("open_ports", {})
        ports_for_ip = ports.get(target_ip, [])
        if port not in ports_for_ip:
            ports_for_ip = sorted(set(ports_for_ip + [port]))
            ports[target_ip] = ports_for_ip
            kb["open_ports"] = ports

        # Record service
        key = f"{target_ip}:{port}"
        services = kb.get("services", {})
        if key not in services:
            services[key] = service
            kb["services"] = services

    # Detect credentials (loose heuristic)
    cred_patterns = [
        re.compile(r'(?:password|passwd|pwd)\s*[=:]\s*(\S+)', re.IGNORECASE),
        re.compile(r'(?:username|user|login)\s*[=:]\s*(\S+)', re.IGNORECASE),
    ]
    users = re.findall(r'(?:username|user|login)\s*[=:]\s*(\S+)', tool_result, re.IGNORECASE)
    passwords = re.findall(r'(?:password|passwd|pwd)\s*[=:]\s*(\S+)', tool_result, re.IGNORECASE)
    if users and passwords:
        creds = kb.get("credentials", [])
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
    system_prompt = agent_cfg.get("system_prompt", "")

    # Resolve provider/model — global override in state takes precedence, else per-agent config
    provider, model, llm = _resolve_llm(state, agent_cfg)
    logger.info("[%s] provider=%s model=%s", agent_name, provider, model)

    # Fetch raw MCP tools — each client's generate_response formats them internally
    all_tools = await mcp_client.list_tools()

    # Filter to the agent's allowed tool set (fnmatch patterns in agents.yaml).
    # If no 'tools' key is present, all tools are passed through.
    allowed_patterns: list[str] = agent_cfg.get("tools", [])
    if allowed_patterns:
        raw_tools = [
            t for t in all_tools
            if any(fnmatch.fnmatch(t["name"], pat) for pat in allowed_patterns)
        ]
        logger.info("[%s] Tool filter: %d/%d tools allowed",
                    agent_name, len(raw_tools), len(all_tools))
    else:
        raw_tools = all_tools

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
            system_prompt=full_system,
        )

        tool_calls = llm.parse_tool_calls(response)
        assistant_msg = llm.make_assistant_message(response)
        new_messages.append(assistant_msg)

        if not tool_calls:
            # No more tool calls — agent is done with its sub-task
            break

        # Execute each tool call
        # All clients return: {"id": str, "name": str, "arguments": dict}
        tool_results: list = []
        for tc in tool_calls:
            tool_name = tc.get("name", "")
            raw_args = tc.get("arguments") or tc.get("input", {})
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    raw_args = {}

            logger.info("[%s] → %s(%s)", agent_name, tool_name, json.dumps(raw_args)[:200])
            raw_result = await mcp_client.call_tool(tool_name, raw_args)

            # Auto-summarize large outputs
            result = maybe_summarize(raw_result, config)

            if result != raw_result:
                logger.info("[Summarizer] Condensed %s → %s chars", f"{len(raw_result):,}", f"{len(result):,}")

            # Track exploit failures
            if agent_name == "exploit" and (
                "failed" in result.lower() or
                "error" in result.lower()[:200] or
                "connection refused" in result.lower()
            ):
                exploit_delta += 1

            # Update knowledge base from tool output
            updated_kb = _extract_kb_updates(tool_name, result, updated_kb)
            tool_results.append((tc, result))

        # Build tool result messages (provider-specific)
        if provider == "anthropic":
            result_blocks = [
                llm.make_tool_result_message(tc.get("id", ""), result)
                for tc, result in tool_results
            ]
            # Anthropic bundles all results in one user message
            combined = {"role": "user", "content": []}
            for rb in result_blocks:
                combined["content"].extend(rb.get("content", []))
            new_messages.append(combined)
        else:
            # Gemini and Ollama: one message per result
            for tc, result in tool_results:
                new_messages.append(llm.make_tool_result_message(tc.get("name", ""), result))

    # Compute new token estimate
    all_messages = (messages + new_messages)
    new_estimate = _estimate_tokens(all_messages)

    return {
        "messages": new_messages,
        "knowledge_base": updated_kb,
        "current_agent": agent_name,
        "context_token_estimate": new_estimate,
        "exploit_attempts": state.get("exploit_attempts", 0) + exploit_delta,
    }


# ---------------------------------------------------------------------------
# Agent node factories — called by graph.py when building the StateGraph
# ---------------------------------------------------------------------------

def make_recon_node(mcp_client: Any, tools: list):
    """Return an async node function for the Recon agent."""
    async def recon_node(state: TeamState) -> dict:
        logger.info("[Recon Agent] Starting reconnaissance...")
        return await _run_agent_loop("recon", state, tools, mcp_client)
    return recon_node


def make_exploit_node(mcp_client: Any, tools: list):
    """Return an async node function for the Exploit agent."""
    async def exploit_node(state: TeamState) -> dict:
        logger.info("[Exploit Agent] Attempting exploitation...")
        return await _run_agent_loop("exploit", state, tools, mcp_client)
    return exploit_node


def make_privesc_node(mcp_client: Any, tools: list):
    """Return an async node function for the PrivEsc agent."""
    async def privesc_node(state: TeamState) -> dict:
        logger.info("[PrivEsc Agent] Escalating privileges...")
        return await _run_agent_loop("privesc", state, tools, mcp_client)
    return privesc_node


def make_webexplorer_node(mcp_client: Any, tools: list):
    """Return an async node function for the Web Explorer agent."""
    async def webexplorer_node(state: TeamState) -> dict:
        logger.info("[WebExplorer Agent] Browsing and mapping web content...")
        return await _run_agent_loop("webexplorer", state, tools, mcp_client)
    return webexplorer_node
