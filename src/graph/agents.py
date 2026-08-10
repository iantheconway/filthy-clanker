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
import hashlib
import json
import logging
import re
import sys
import os
from typing import Any, Dict, List, Optional, Tuple

from .state import TeamState, KnowledgeBase
from .summarizer import maybe_summarize, _FLAG_RE, _HEX_FLAG_RE, _is_placeholder_flag

# Import existing LLM clients
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from llms import AnthropicClient, GeminiClient, OllamaClient

logger = logging.getLogger("filthy_clanker")


# ---------------------------------------------------------------------------
# Raw-output flag extraction (runs BEFORE summarisation)
# ---------------------------------------------------------------------------

def _extract_flags_from_raw(raw: str, tool_args: dict | None = None) -> list[str]:
    """
    Scan the RAW (un-summarised) tool output for flag strings and return them.
    Called before maybe_summarize() so a lossy summary can never hide a flag.
    """
    flags: list[str] = []
    # Bracket-format flags: flag{...}, key{...}, HTB{...}, etc.
    flags += _FLAG_RE.findall(raw)
    # Broader single-word bracket flags that _FLAG_RE might miss
    _broad = re.findall(r'\b\w{2,20}\{[^}]{1,200}\}', raw)
    flags += [f for f in _broad if f not in flags]
    # 32-char hex strings when a flag-file context is present
    _args_str = json.dumps(tool_args) if tool_args else ""
    if re.search(r'(?:user|root|flag)\.txt', raw + _args_str, re.IGNORECASE):
        flags += _HEX_FLAG_RE.findall(raw)
    # Drop placeholder/template flags (e.g. "flag{...}") so a hedge never counts.
    flags = [f for f in flags if not _is_placeholder_flag(f)]
    return list(dict.fromkeys(flags))  # deduplicate preserving order


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
            host = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
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
            host = agent_cfg.get("host", os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434"))
            llm = OllamaClient(host=host, model=model)
        else:
            raise ValueError(f"Unknown provider in agents.yaml for this agent: {provider}")

    return provider, model, llm


def _estimate_tokens(messages: list) -> int:
    """Rough token estimate: total chars / 4."""
    return sum(len(str(m)) for m in messages) // 4


# Commands/tools that count as genuine disassembly / dynamic / symbolic analysis.
# The reversing agent's completion-gate requires at least one of these to have run
# before it may report — matched against the tool name AND the command arguments
# (execute_command carries the real work in its `command`).
_DEEP_RE_RE = re.compile(
    r'\b(objdump|radare2|rizin|r2\b|gdb|ltrace|strace|angr|ROPgadget|ropper'
    r'|one_gadget|checksec|readelf|\bnm\b|ghidra|decompil)', re.IGNORECASE,
)


def _is_deep_re_call(tool_name: str, raw_args: dict) -> bool:
    """True if a tool call is real binary analysis (disassembly/debug/symbolic)."""
    if _DEEP_RE_RE.search(tool_name or ""):
        return True
    try:
        blob = json.dumps(raw_args)
    except (TypeError, ValueError):
        blob = str(raw_args)
    return bool(_DEEP_RE_RE.search(blob))


# A genuine SOLVE attempt (compute/decrypt/exploit), as opposed to mere inspection
# (file/strings/cat). Used by the exploit agent's follow-through gate on file-based
# crypto/rev/pwn: it may not conclude until it has actually RUN a solve script or
# executed the target — forcing "attempt" over "guess".
_SOLVE_ATTEMPT_RE = re.compile(
    r'python3?\s+(?:-c|-m\s|\S*\.py)'   # run a python script/one-liner
    r'|openssl\s+[a-z]'                  # openssl crypto operation
    r'|RsaCtfTool|\bsage\b|factordb'     # crypto solvers
    r'|from\s+pwn|import\s+pwn'          # pwntools exploitation
    r'|import\s+angr|import\s+z3|claripy',  # symbolic execution
    re.IGNORECASE,
)
# Executing a local binary/exploit: `./x` at the START of the command (or after a
# shell separator) — NOT `./x` as an argument to an inspection tool (objdump -d ./x).
_EXEC_LOCAL_RE = re.compile(r'(?:^|;|&&|\|\|?|\n)\s*\./\S+')


def _is_solve_attempt(tool_name: str, raw_args: dict) -> bool:
    """True if a tool call is a real solve attempt (script/crypto/exploit/run)."""
    cmd = ""
    if isinstance(raw_args, dict):
        cmd = str(raw_args.get("command") or raw_args.get("cmd") or "")
    blob = cmd or (json.dumps(raw_args) if raw_args else "")
    return bool(_SOLVE_ATTEMPT_RE.search(blob)) or bool(_EXEC_LOCAL_RE.search(cmd))


def _trim_tool_descriptions(tools: list, max_chars: int) -> list:
    """
    Return a copy of ``tools`` with each ``description`` capped at ``max_chars``.

    Hexstrike exposes ~150 tools with verbose descriptions; the full schema is
    resent on every turn and bloats the prompt, which degrades smaller local
    models (SPEC: filthy-clanker-agent-solve-quality, exp. 3). A ``max_chars`` of
    0 disables trimming (full descriptions passed through unchanged).

    Trimming keeps the first sentence when it fits, otherwise cuts on a word
    boundary and appends an ellipsis. Tool names and inputSchema are untouched —
    only prose is shortened, so the model still knows exactly what to call.
    """
    if not max_chars or max_chars <= 0:
        return tools
    trimmed: list = []
    for t in tools:
        desc = t.get("description") or ""
        if len(desc) <= max_chars:
            trimmed.append(t)
            continue
        # Prefer cutting at the end of the first sentence if it lands within budget.
        dot = desc.find(". ")
        if 0 < dot + 1 <= max_chars:
            new_desc = desc[: dot + 1]
        else:
            cut = desc.rfind(" ", 0, max_chars)
            new_desc = (desc[:cut] if cut > 0 else desc[:max_chars]).rstrip() + " …"
        nt = dict(t)
        nt["description"] = new_desc
        trimmed.append(nt)
    return trimmed


def _log_prompt_size(agent_name: str, system_prompt: str, tools: list) -> None:
    """Log a rough per-turn prompt-size breakdown (SPEC exp. 3 measurement).

    Reports the token estimate of the system prompt and of the tool schema
    separately so the tool-description contribution is directly visible.
    """
    sys_tok = len(system_prompt) // 4
    tools_tok = len(json.dumps(tools)) // 4
    logger.info(
        "[%s] Prompt size (est. tokens): system=%d, tool_schema=%d (%d tools), total=%d",
        agent_name, sys_tok, tools_tok, len(tools), sys_tok + tools_tok,
    )


def _extract_kb_updates(tool_name: str, tool_result: str, kb: KnowledgeBase,
                        tool_args: dict | None = None) -> KnowledgeBase:
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
        # Copy nested containers before mutating — kb may be checkpointed shared
        # state, and a shallow dict(kb) still aliases these inner dicts/lists.
        ports = dict(kb.get("open_ports", {}))
        ports_for_ip = ports.get(target_ip, [])
        if port not in ports_for_ip:
            ports_for_ip = sorted(set(ports_for_ip + [port]))
            ports[target_ip] = ports_for_ip
            kb["open_ports"] = ports

        # Record service
        key = f"{target_ip}:{port}"
        services = dict(kb.get("services", {}))
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
        creds = list(kb.get("credentials", []))
        for u, p in zip(users, passwords):
            entry = {"user": u, "pass": p}
            if entry not in creds:
                creds.append(entry)
        kb["credentials"] = creds

    # Detect flags — bracket format (HTB{...}) and raw 32-char hex (user.txt/root.txt)
    flag_pattern = re.compile(r'(?:HTB|FLAG|CTF)\{[^}]+\}', re.IGNORECASE)
    hex_flag_pattern = re.compile(r'\b([0-9a-f]{32})\b', re.IGNORECASE)
    flags_found = flag_pattern.findall(tool_result)
    # Only capture bare hex strings when the context suggests a flag file
    # Check both the tool result and the tool arguments (e.g. the command string)
    _args_str = json.dumps(tool_args) if tool_args else ""
    _flag_file_ctx = re.search(r'(?:user|root|flag)\.txt', tool_result + _args_str, re.IGNORECASE)
    if _flag_file_ctx:
        flags_found += hex_flag_pattern.findall(tool_result)
    flags_found = [f for f in flags_found if not _is_placeholder_flag(f)]
    if flags_found:
        existing_flags = set(kb.get("flags", []))
        kb["flags"] = list(existing_flags | set(flags_found))

    # Interesting attack surface (directories, files from gobuster/ferox)
    dir_pattern = re.compile(r'(?:Status: 200|Found:)\s*(\/\S+)', re.IGNORECASE)
    found_paths = dir_pattern.findall(tool_result)
    if found_paths:
        surface = set(kb.get("attack_surface", []))
        kb["attack_surface"] = list(surface | set(found_paths[:20]))  # cap at 20

    # CVE and searchsploit findings → attack_surface
    # Searchsploit output lines: "Title  | EDB-ID | ..."
    edb_pattern = re.compile(r'(.+?)\s*\|\s*(\d{4,6})\s*\|', re.MULTILINE)
    cve_inline = re.compile(r'(CVE-\d{4}-\d{4,})', re.IGNORECASE)
    vuln_entries: list[str] = []
    for m in edb_pattern.finditer(tool_result):
        title = m.group(1).strip()
        edb_id = m.group(2).strip()
        if title and not title.startswith("-"):
            vuln_entries.append(f"EDB-{edb_id}: {title}")
    for cve in cve_inline.findall(tool_result):
        vuln_entries.append(cve.upper())

    # Web search results (brave_web_search etc.) — extract sentences from
    # Title/Description fields that contain general exploit-class vocabulary.
    # This is intentionally broad: the goal is to capture any confirmed finding
    # that an agent could act on, not to match specific CVEs or software names.
    _VULN_VOCAB = re.compile(
        r'\b(exploit(?:able|ed)?|backdoor|rce|remote\s+code\s+exec(?:ution)?'
        r'|command\s+injection|sql\s+inject(?:ion)?|path\s+travers(?:al)?'
        r'|lfi|rfi|ssrf|xss|xxe|deseri[a-z]+|auth(?:entication)?\s+bypass'
        r'|privilege\s+escal(?:ation)?|arbitrary\s+(?:code|command|file)'
        r'|unauthenticated|unauth(?:orized)?)\b',
        re.IGNORECASE,
    )
    # Match "Title: ..." and "Description: ..." lines from search result blocks
    search_field_pattern = re.compile(
        r'(?:Title|Description)\s*:\s*([^\n]{10,300})', re.IGNORECASE
    )
    for m in search_field_pattern.finditer(tool_result):
        sentence = m.group(1).strip()
        if _VULN_VOCAB.search(sentence):
            # Strip HTML-like strong tags that some search APIs return
            clean = re.sub(r'<[^>]+>', '', sentence).strip()
            if clean:
                vuln_entries.append(clean[:150])

    if vuln_entries:
        surface = set(kb.get("attack_surface", []))
        kb["attack_surface"] = list(surface | set(vuln_entries[:30]))

    # -----------------------------------------------------------------------
    # Tech stack and response header extraction
    # -----------------------------------------------------------------------
    target_ip = (kb.get("ips") or ["target"])[0]

    # Patterns that yield "Software/version" strings from various tools
    version_patterns = [
        # curl -I / HTTP header values:  "Server: Apache/2.4.49"
        re.compile(r'^(?:server|x-powered-by|x-generator|x-aspnet-version|x-runtime|via)\s*:\s*(.+)$',
                   re.IGNORECASE | re.MULTILINE),
        # nmap -sV banner:  "80/tcp open  http  Apache httpd 2.4.49"
        re.compile(r'\d+/tcp\s+open\s+\S+\s+(.+)', re.IGNORECASE),
        # nmap script output:  "| http-server-header: Apache/2.4.49"
        re.compile(r'\|\s*http-server-header:\s*(.+)', re.IGNORECASE),
        # nikto:  "+ Server: Apache/2.4.49 (Ubuntu)"
        re.compile(r'\+\s*Server:\s*(.+)', re.IGNORECASE),
        # whatweb:  Apache[2.4.49], PHP[8.1.0-dev]
        re.compile(r'(\w[\w.-]+)\[(\d[\d.a-z_-]+)\]', re.IGNORECASE),
    ]

    tech_entries: list[str] = []
    whatweb_pat = version_patterns[-1]  # the name[version] pattern
    for pat in version_patterns:
        for m in pat.finditer(tool_result):
            if pat is whatweb_pat:
                entry = f"{m.group(1)}/{m.group(2)}"
            else:
                entry = m.group(1).strip()
            if entry and len(entry) < 120:
                tech_entries.append(entry)

    if tech_entries:
        # Associate with the port referenced in the tool call (default 80 for web)
        port_match = re.search(r':(\d+)', tool_name) or re.search(r'\b(80|443|8080|8443)\b', tool_result)
        port = port_match.group(1) if port_match else "80"
        key = f"{target_ip}:{port}"
        tech_stack = dict(kb.get("tech_stack") or {})
        existing = set(tech_stack.get(key, []))
        tech_stack[key] = list(existing | set(tech_entries))
        kb["tech_stack"] = tech_stack

    # Response headers from curl -I / wget --server-response output
    # Match lines of the form "Header-Name: value" (HTTP header format)
    header_line_pat = re.compile(
        r'^([\w-]+):\s+(.+)$', re.MULTILINE
    )
    # Only extract if this looks like an HTTP response (has HTTP/ status line or
    # at least a Server header) to avoid false positives from other tool output.
    if re.search(r'HTTP/\d', tool_result) or re.search(r'^Server:', tool_result, re.MULTILINE | re.IGNORECASE):
        headers_found: dict[str, str] = {}
        for m in header_line_pat.finditer(tool_result):
            hname = m.group(1).strip()
            hval = m.group(2).strip()
            # Skip obviously non-header lines (JSON keys, nmap fields, etc.)
            if len(hname) <= 60 and len(hval) <= 256:
                headers_found[hname] = hval
        if headers_found:
            port_h = re.search(r':(\d+)', tool_name)
            port_str = port_h.group(1) if port_h else "80"
            key = f"{target_ip}:{port_str}"
            resp_headers = dict(kb.get("response_headers") or {})
            existing_h = dict(resp_headers.get(key, {}))
            existing_h.update(headers_found)
            resp_headers[key] = existing_h
            kb["response_headers"] = resp_headers

    return kb


# Vocabulary that indicates a confirmed, directly exploitable finding in attack_surface.
# These terms match what the general web-search extractor and CVE/EDB patterns produce.
# Kept intentionally broad — the same words used in _VULN_VOCAB extraction above.
_HIGH_VALUE_EXPLOIT_SIGNALS = (
    "exploit", "backdoor", "rce", "remote code exec",
    "command injection", "sql inject", "path travers",
    "lfi", "rfi", "ssrf", "xxe", "auth bypass",
    "privilege escal", "arbitrary code", "arbitrary command",
    "unauthenticated", "unauthor",
)


def _has_confirmed_exploit_path(kb: KnowledgeBase) -> bool:
    """Return True if attack_surface contains a confirmed, directly exploitable finding."""
    for entry in kb.get("attack_surface", []):
        el = entry.lower()
        if any(s in el for s in _HIGH_VALUE_EXPLOIT_SIGNALS):
            return True
    return False


def _scan_key_args(tool_name: str, raw_args: dict) -> str:
    """Return a short, human-readable summary of the key arguments for scan_history."""
    for key in ("target", "host", "url", "command", "ip", "address", "domain"):
        val = raw_args.get(key)
        if val:
            return str(val)[:80]
    # Fallback: first value in args
    if raw_args:
        first_val = next(iter(raw_args.values()))
        return str(first_val)[:80]
    return ""


def _extract_url_from_args(raw_args: dict) -> Optional[str]:
    """Extract the first HTTP/HTTPS URL from tool call arguments."""
    for key in ("url", "target", "command", "uri", "address"):
        val = raw_args.get(key, "")
        if val:
            m = re.search(r'https?://[^\s\'")\]]+', str(val))
            if m:
                return m.group(0).rstrip("'\")")
    return None


def _detect_shell_from_output(result: str, known_ips: List[str]) -> Optional[Dict[str, str]]:
    """
    Detect a successful shell / RCE from command output.
    Only returns a result for strong signals to avoid false positives.
    """
    # uid=0(root) or uid=1000(user) — very reliable indicator of code execution
    uid_match = re.search(r'\buid=(\d+)\((\w+)\)', result)
    if uid_match:
        return {
            "user": uid_match.group(2),
            "uid": uid_match.group(1),
            "host": known_ips[0] if known_ips else "unknown",
            "via": "RCE (uid output)",
        }
    # "root@hostname:~#" style prompts that are the LAST line of output
    last_lines = result.strip().splitlines()[-3:]
    for line in last_lines:
        m = re.match(r'^(\w[\w-]*)@([\w.-]+)[:#\$]\s*(?:~.*)?[#\$]\s*$', line.strip())
        if m:
            return {
                "user": m.group(1),
                "host": m.group(2),
                "via": "shell prompt",
            }
    return None


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
    # Special tools that must always be visible regardless of the allowlist:
    _ALWAYS_ALLOWED = {"submit_flag"}
    allowed_patterns: list[str] = agent_cfg.get("tools", [])
    if allowed_patterns:
        raw_tools = [
            t for t in all_tools
            if t["name"] in _ALWAYS_ALLOWED
            or any(fnmatch.fnmatch(t["name"], pat) for pat in allowed_patterns)
        ]
        logger.info("[%s] Tool filter: %d/%d tools allowed",
                    agent_name, len(raw_tools), len(all_tools))
    else:
        raw_tools = all_tools

    # Inject knowledge base and exact tool names into system prompt.
    # Listing the tool names directly prevents the model from guessing.
    kb = state.get("knowledge_base", {})
    kb_text = json.dumps(kb, indent=2)
    tool_names_list = ", ".join(t["name"] for t in raw_tools) if raw_tools else "(none)"

    # Add a submit_flag reminder if the tool is available (eval mode).
    _has_submit_flag = any(t["name"] == "submit_flag" for t in raw_tools)
    _submit_flag_note = (
        "\n\n=== FLAG SUBMISSION ===\n"
        "When you capture the flag, call submit_flag(flag='<full flag string>') IMMEDIATELY.\n"
        "This is required — do not just write the flag in text. submit_flag must be called."
        if _has_submit_flag else ""
    )

    full_system = (
        f"{system_prompt}\n\n"
        f"=== AVAILABLE TOOLS (exact names — use only these) ===\n{tool_names_list}\n\n"
        f"=== KNOWLEDGE BASE (shared with team) ===\n{kb_text}\n"
        f"=== CURRENT TASK ===\n{state.get('task', 'Hack the target machine.')}"
        f"{_submit_flag_note}"
    )

    # Trim verbose tool descriptions to keep the per-turn prompt small on local
    # models (SPEC exp. 3). 0 / missing => no trimming. Measure and log the
    # resulting prompt size so the tool-schema contribution is observable.
    _settings_pre = config.get("settings", {})
    _max_desc = _settings_pre.get("max_tool_description_chars", 0)
    raw_tools = _trim_tool_descriptions(raw_tools, _max_desc)
    _log_prompt_size(agent_name, full_system, raw_tools)

    # Start from current conversation history
    messages = list(state.get("messages", []))
    new_messages: list = []
    updated_kb = dict(kb)
    exploit_delta = 0

    # ReAct loop: call LLM → execute tools → repeat
    max_iterations = 20
    # Tracks the iteration at which a high-value exploit path was first confirmed.
    # Recon/webexplorer are given one more iteration to write up then forced to stop.
    _exploit_found_at: Optional[int] = None
    # Counts consecutive iterations where every tool call was an unknown-tool error.
    _consecutive_unknown_iters: int = 0
    # Whether this turn executed at least one REAL (non-unknown) tool call. A turn
    # with none is "idle" — it feeds the unproductive-streak circuit breaker and
    # marks the agent complete so the supervisor stops re-routing to it (kills the
    # observed exploit spin: 44 routes, 0 tool calls, never completing).
    _made_tool_call: bool = False
    # Reversing completion-gate: the reversing agent may not finish until it has
    # actually disassembled/traced the binary. _did_deep_re flips on the first real
    # RE tool run; _re_gate_used caps the forcing so a stubborn model can't loop.
    _did_deep_re: bool = False
    _re_gate_used: int = 0
    _RE_GATE_MAX: int = 2
    # Exploit follow-through gate: on a file-based crypto/rev/pwn challenge the
    # exploit agent may not conclude until it has actually RUN a solve script /
    # executed the target (forces "attempt" over "guess").
    _did_solve_attempt: bool = False
    _solve_gate_used: int = 0
    # Duplicate-output detection: tool_name → set of MD5 hashes of raw results.
    # If a tool returns the exact same bytes twice, we inject a strategy-change hint.
    _seen_output_hashes: dict[str, set] = {}

    for iteration in range(max_iterations):
        try:
            response = await llm.generate_response(
                messages=messages + new_messages,
                tools=raw_tools,
                system_prompt=full_system,
            )
        except Exception as exc:
            # LLM call failed even after the client's own retries (e.g. a hard
            # Ollama 400/5xx or the server being down). End this agent's turn
            # gracefully so the supervisor can reroute — never crash the whole
            # challenge on one bad call. Keep the note < 80 chars so the
            # lightweight evaluator routes it to the supervisor, not refusal.
            logger.error("[%s] LLM call failed after retries: %s — ending turn", agent_name, exc)
            new_messages.append({"role": "assistant", "content": "[LLM error — ending turn.]"})
            break

        tool_calls = llm.parse_tool_calls(response)
        assistant_msg = llm.make_assistant_message(response)
        new_messages.append(assistant_msg)

        if not tool_calls:
            # Reversing completion-gate: refuse to finish until the binary has
            # actually been disassembled/traced. Inject a hard corrective and loop
            # again, up to _RE_GATE_MAX times (then let it out to avoid an infinite
            # loop if the model simply won't comply).
            if (agent_name == "reversing" and not _did_deep_re
                    and _re_gate_used < _RE_GATE_MAX):
                _re_gate_used += 1
                logger.info("[reversing] Completion blocked — no disassembly yet "
                            "(gate %d/%d), forcing RE step", _re_gate_used, _RE_GATE_MAX)
                new_messages.append({
                    "role": "user",
                    "content": (
                        "STOP — you have not disassembled the binary yet, so you "
                        "cannot possibly know the answer. Before reporting anything, "
                        "call execute_command to run REAL analysis on the provided "
                        "file now, e.g.:\n"
                        "  objdump -d <path>            (disassembly)\n"
                        "  radare2 -A -q -c 'pdf @ main' <path>\n"
                        "  gdb -batch -ex 'disas main' <path>   or ltrace/strace <path>\n"
                        "For an input-derived flag, write a python3 script using angr "
                        "(symbolic execution) or z3. Read the output and reason about "
                        "the logic. Do NOT report until you have done this."
                    ),
                })
                continue
            # Exploit follow-through gate on file-based crypto/rev/pwn: force at least
            # one real solve attempt (script/crypto/exploit run) before concluding.
            _fb_cat = (state.get("challenge_category") or "").lower()
            if (agent_name == "exploit" and state.get("has_files")
                    and _fb_cat in ("crypto", "rev", "pwn")
                    and not _did_solve_attempt and _solve_gate_used < _RE_GATE_MAX):
                _solve_gate_used += 1
                logger.info("[exploit] Completion blocked — no solve attempt yet "
                            "(gate %d/%d), forcing a solve script", _solve_gate_used, _RE_GATE_MAX)
                new_messages.append({
                    "role": "user",
                    "content": (
                        "STOP — you have only inspected the file, not attempted a "
                        "solution. Write and RUN a real solve now via execute_command, "
                        "and read its output before concluding:\n"
                        "  crypto: python3 -c \"from Crypto... ; ...\"  (or sympy/gmpy2/"
                        "z3, openssl, RsaCtfTool) — actually compute/decrypt.\n"
                        "  rev/pwn: run the binary with crafted input, or a pwntools/"
                        "angr/z3 python3 script. Do NOT guess or hand-derive the flag."
                    ),
                })
                continue
            # No more tool calls — agent is done with its sub-task
            break

        # Execute each tool call
        # All clients return: {"id": str, "name": str, "arguments": dict}
        tool_results: list = []
        _all_unknown_this_iter = True  # will be cleared when any real tool succeeds
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

            # Detect unknown-tool errors and replace with a corrective message that
            # lists the actual available tools so the model stops guessing.
            if raw_result.startswith("Unknown tool:"):
                available_names = [t["name"] for t in raw_tools]
                raw_result = (
                    f"Error: tool '{tool_name}' does not exist.\n"
                    f"You MUST only call tools from this exact list "
                    f"(no others exist):\n  {', '.join(available_names)}\n"
                    f"Choose the correct tool from the list above and call it now."
                )
                logger.warning("[%s] Unknown tool '%s' called — injecting corrective feedback",
                               agent_name, tool_name)
            else:
                _all_unknown_this_iter = False
                _made_tool_call = True  # a real tool ran → this turn is productive
                if not _did_deep_re and _is_deep_re_call(tool_name, raw_args):
                    _did_deep_re = True
                if not _did_solve_attempt and _is_solve_attempt(tool_name, raw_args):
                    _did_solve_attempt = True

                # ---- Raw flag extraction (BEFORE summarization) ----
                # Run on the un-summarised output so a lossy summary can never
                # hide a flag.  Captured flags go directly into the KB.
                _raw_flags = _extract_flags_from_raw(raw_result, raw_args)
                if _raw_flags:
                    _existing_flags = set(updated_kb.get("flags", []))
                    _new_flags = [f for f in _raw_flags if f not in _existing_flags]
                    if _new_flags:
                        updated_kb = dict(updated_kb)
                        updated_kb["flags"] = list(_existing_flags | set(_raw_flags))
                        logger.info(
                            "[%s] Flags extracted from RAW output (pre-summarisation): %s",
                            agent_name, _new_flags,
                        )

                # ---- Duplicate output detection ----
                # If this tool returns the exact same bytes as a previous call,
                # replace the result with a strategy-change prompt so the agent
                # doesn't just loop with the same input.
                _out_hash = hashlib.md5(
                    raw_result.encode(errors="replace")
                ).hexdigest()
                _tool_hashes = _seen_output_hashes.setdefault(tool_name, set())
                if _out_hash in _tool_hashes:
                    logger.warning(
                        "[%s] Duplicate output detected from %s — injecting strategy-change prompt",
                        agent_name, tool_name,
                    )
                    raw_result = (
                        f"[DUPLICATE OUTPUT DETECTED]\n"
                        f"'{tool_name}' returned the EXACT same output as a previous call.\n"
                        f"Calling it again with the same arguments WILL NOT help.\n"
                        f"You MUST change your strategy: use different arguments, a different "
                        f"tool, or a completely different approach.\n\n"
                        f"Previous output (first 800 chars for reference):\n"
                        f"{raw_result[:800]}"
                    )
                else:
                    _tool_hashes.add(_out_hash)

            # Tool-output handling. The reversing agent must see RAW disassembly
            # (paraphrasing assembly is useless), so instead of summarising we pass
            # it verbatim up to a cap, then hard-TRUNCATE with a note steering it to
            # targeted disassembly — never hand a full 150k objdump to the model.
            # All other agents summarise as before.
            if agent_name == "reversing":
                _rev_thr = config.get("settings", {}).get("reversing_output_threshold", 16000)
                if len(raw_result) > _rev_thr:
                    result = (
                        raw_result[:_rev_thr]
                        + f"\n\n[... disassembly truncated at {_rev_thr:,} of "
                        f"{len(raw_result):,} chars. Do NOT dump the whole binary — run "
                        f"TARGETED disassembly of the function you care about, e.g. "
                        f"radare2 -q -c 'pdf @ main' <bin>  or  objdump -d --disassemble=<func> <bin>. ...]"
                    )
                else:
                    result = raw_result
            else:
                result = maybe_summarize(raw_result, config)

            if result != raw_result:
                logger.info("[Summarizer] Condensed %s → %s chars", f"{len(raw_result):,}", f"{len(result):,}")

            # Track exploit failures — only count hard failures, not data returns.
            # A non-zero exit code or stderr output signals genuine failure.
            _is_hard_fail = (
                bool(re.search(r'(?:exit\s*code|returncode|return\s*code)\s*[=:]\s*[1-9]', result, re.IGNORECASE))
                or bool(re.search(r'(?:stderr|STDERR)\s*:\s*\S', result))
                or any(kw in result.lower() for kw in (
                    "connection refused", "no route to host",
                    "connection timed out", "access denied",
                ))
            )
            if agent_name == "exploit" and _is_hard_fail:
                exploit_delta += 1

            # Update knowledge base from tool output
            updated_kb = _extract_kb_updates(tool_name, result, updated_kb, raw_args)

            # ---- scan_history: record every tool call so agents don't repeat work ----
            _sh_target = (updated_kb.get("ips") or ["target"])[0]
            _sh_args_summary = _scan_key_args(tool_name, raw_args)
            _sh_entry = f"[{agent_name}] {tool_name}: {_sh_args_summary}"
            _scan_hist = dict(updated_kb.get("scan_history") or {})
            _target_hist = list(_scan_hist.get(_sh_target, []))
            if _sh_entry not in _target_hist:
                _target_hist.append(_sh_entry)
                _scan_hist[_sh_target] = _target_hist[-60:]  # cap per-target
                updated_kb["scan_history"] = _scan_hist

            # ---- visited_urls: record HTTP fetches for webexplorer deduplication ----
            _url = _extract_url_from_args(raw_args)
            if _url:
                _visited = list(updated_kb.get("visited_urls") or [])
                if _url not in _visited:
                    _visited.append(_url)
                    updated_kb["visited_urls"] = _visited[-300:]  # cap

            # ---- exploit_history: record outcomes of exploit tool calls ----
            # A "failure" requires hard evidence: non-zero exit code, stderr output,
            # or an unambiguous connection error.  Commands that return data (even
            # partial or unexpected) are recorded as "discovery" so the agent can
            # act on the output rather than being told to skip it next time.
            if agent_name == "exploit":
                _exit_nonzero = bool(re.search(
                    r'(?:exit\s*code|returncode|return\s*code)\s*[=:]\s*[1-9]',
                    result, re.IGNORECASE,
                ))
                _has_stderr = bool(re.search(r'(?:stderr|STDERR)\s*:\s*\S', result))
                _conn_fail = any(kw in result.lower() for kw in (
                    "connection refused", "no route to host",
                    "connection timed out", "access denied",
                ))
                _exp_hist = list(updated_kb.get("exploit_history") or [])
                if _exit_nonzero or _has_stderr or _conn_fail:
                    _exp_entry = f"{tool_name}({_sh_args_summary[:60]}): failed"
                elif result.strip() and not result.startswith("[Summarizer"):
                    # Command returned data — record as a discovery, not a failure
                    _exp_entry = f"{tool_name}({_sh_args_summary[:60]}): discovery"
                else:
                    _exp_entry = None
                if _exp_entry and _exp_entry not in _exp_hist:
                    _exp_hist.append(_exp_entry)
                    updated_kb["exploit_history"] = _exp_hist[-40:]

            # ---- shells: auto-detect successful RCE / foothold ----
            _shell = _detect_shell_from_output(result, list(updated_kb.get("ips") or []))
            if _shell:
                _shells = list(updated_kb.get("shells") or [])
                # Avoid duplicate entries for the same user@host
                _shell_key = (_shell.get("user"), _shell.get("host"))
                if not any(
                    (s.get("user"), s.get("host")) == _shell_key for s in _shells
                ):
                    _shells.append(_shell)
                    updated_kb["shells"] = _shells
                    logger.info("[%s] Shell detected: %s@%s via %s",
                                agent_name, _shell.get("user"), _shell.get("host"), _shell.get("via"))

            tool_results.append((tc, result))

        # Build tool result messages (provider-specific)
        if provider == "anthropic":
            # Anthropic bundles all tool results in a single user message whose
            # `content` is a list of tool_result blocks (one per tool call).
            combined = {"role": "user", "content": []}
            for tc, result in tool_results:
                block = llm.make_tool_result_message(tc.get("id", ""), result)
                combined["content"].append(block)
            new_messages.append(combined)
        else:
            # Gemini and Ollama: one message per result
            for tc, result in tool_results:
                new_messages.append(llm.make_tool_result_message(tc.get("name", ""), result))

        # ---- Unknown-tool loop guard ----
        # If every tool call this iteration was an unknown-tool error, increment the
        # counter.  After 3 consecutive all-unknown iterations the model is stuck in a
        # hallucination loop — force TASK COMPLETE so the supervisor can reroute.
        if _all_unknown_this_iter:
            _consecutive_unknown_iters += 1
            logger.warning("[%s] All tool calls this iteration were unknown tools (%d consecutive)",
                           agent_name, _consecutive_unknown_iters)
            if _consecutive_unknown_iters >= 3:
                available_names = [t["name"] for t in raw_tools]
                forced_completion = (
                    "TASK COMPLETE\n\n"
                    f"Unable to proceed: the last {_consecutive_unknown_iters} attempts all called "
                    f"non-existent tools. Available tools are: {available_names}. "
                    "Handing back to supervisor."
                )
                new_messages.append({"role": "assistant", "content": forced_completion})
                logger.warning("[%s] Forcing exit after %d consecutive unknown-tool iterations",
                               agent_name, _consecutive_unknown_iters)
                break
        else:
            _consecutive_unknown_iters = 0

        # ---- Early exit: stop recon/webexplorer once a confirmed exploit path is found ----
        # Tool result messages are built first so the conversation stays well-formed.
        # We mark the discovery iteration, then force stop one iteration later so the
        # agent has had one chance to react to the finding.
        if agent_name in ("recon", "webexplorer") and _exploit_found_at is None:
            if _has_confirmed_exploit_path(updated_kb):
                _exploit_found_at = iteration
                logger.info(
                    "[%s] Confirmed exploit path detected at iteration %d — "
                    "one more iteration allowed, then forcing TASK COMPLETE",
                    agent_name, iteration,
                )
        if _exploit_found_at is not None and iteration >= _exploit_found_at + 1:
            surface_preview = updated_kb.get("attack_surface", [])[:5]
            forced_completion = (
                "TASK COMPLETE\n\n"
                "A confirmed exploitation path has been identified. Stopping recon and "
                "handing off to the exploit agent.\n\n"
                f"Key findings in attack_surface: {surface_preview}"
            )
            new_messages.append({"role": "assistant", "content": forced_completion})
            logger.info(
                "[%s] Early exit enforced (exploit path found at iteration %d, now at %d)",
                agent_name, _exploit_found_at, iteration,
            )
            break

    # -----------------------------------------------------------------------
    # KB pruning — keep scan_history from bloating the context estimate.
    # If the current message payload already exceeds the configured cap,
    # trim each host's scan_history down to the last N entries.
    # -----------------------------------------------------------------------
    _settings = config.get("settings", {})
    _kb_token_cap: int = _settings.get("kb_token_cap", 30_000)
    _max_scan_per_host: int = _settings.get("max_scan_history_per_host", 10)
    _rough_kb_tokens = len(json.dumps(updated_kb)) // 4
    if _rough_kb_tokens > _kb_token_cap:
        _scan_hist_raw = dict(updated_kb.get("scan_history") or {})
        for _host_key in _scan_hist_raw:
            _scan_hist_raw[_host_key] = _scan_hist_raw[_host_key][-_max_scan_per_host:]
        updated_kb = dict(updated_kb)
        updated_kb["scan_history"] = _scan_hist_raw
        logger.info(
            "[%s] KB token estimate %d > cap %d — pruned scan_history to last %d per host",
            agent_name, _rough_kb_tokens, _kb_token_cap, _max_scan_per_host,
        )

    # Compute new token estimate
    all_messages = (messages + new_messages)
    new_estimate = _estimate_tokens(all_messages)

    # Detect substantive TASK COMPLETE in the final assistant message so the
    # supervisor knows not to re-route to this agent unnecessarily.
    _final_text, _final_had_tool = _last_assistant_text_and_tools(new_messages)
    _body = _final_text[len("task complete"):].strip() if _final_text.lower().lstrip().startswith("task complete") else ""
    _is_substantive_completion = (
        _final_text.lower().lstrip().startswith("task complete")
        and (_final_had_tool or len(_body) >= 80)
    )

    # Also scan the agent's final text response for flags — agents sometimes
    # summarize flag values in their TASK COMPLETE message even when the tool
    # result was already processed (e.g. after summarization stripped context).
    if _final_text:
        _text_flags = re.findall(r'(?:HTB|FLAG|CTF)\{[^}]+\}', _final_text, re.IGNORECASE)
        # Hex flags in text: only if flagfile context nearby
        if re.search(r'(?:user|root|flag)\.txt', _final_text, re.IGNORECASE):
            _text_flags += re.findall(r'\b([0-9a-f]{32})\b', _final_text, re.IGNORECASE)
        # A flag mentioned in the agent's own prose is the least trustworthy source
        # (often a hedge/example like "flag{STFUj...}"); drop placeholders.
        _text_flags = [f for f in _text_flags if not _is_placeholder_flag(f)]
        if _text_flags:
            _existing = set(updated_kb.get("flags", []))
            updated_kb = dict(updated_kb)
            updated_kb["flags"] = list(_existing | set(_text_flags))
            logger.info("[%s] Flags extracted from agent text: %s", agent_name, _text_flags)

    # For the exploit agent: only mark complete if a flag was actually captured.
    # If exploit ran and failed (no flag in KB), do NOT add it to completed_agents —
    # the exploit_attempts counter + HITL handles the exhaustion case, and we want
    # the supervisor to be able to re-route to exploit after human guidance or after
    # other agents surface new information.
    # For all other agents: the substantive-completion check is sufficient.
    _flag_captured = bool(updated_kb.get("flags"))
    if agent_name == "exploit" and _is_substantive_completion and not _flag_captured:
        _is_substantive_completion = False
        logger.info(
            "[%s] TASK COMPLETE received but no flag captured — "
            "NOT marking as completed so supervisor can retry after new leads",
            agent_name,
        )

    # An idle turn (no real tool call) means the agent has nothing to act on right
    # now — mark it complete so the supervisor stops re-routing to it. This also
    # overrides the exploit special-case above: exploit is kept uncompleted only
    # when it actually TRIED (made tool calls) but found no flag; an idle exploit
    # turn is marked complete like any other, breaking the spin. New leads from
    # other agents still clear completed_agents elsewhere, allowing a real retry.
    _mark_complete = _is_substantive_completion or not _made_tool_call
    # The reversing agent is a ONE-SHOT deep pass: after it runs (its internal loop
    # + disassembly gate already gave it up to 20 iterations), mark it complete so
    # the supervisor hands off instead of re-invoking it (which caused a ping-pong
    # with idle network agents). New leads still clear completed_agents elsewhere.
    if agent_name == "reversing":
        _mark_complete = True
    existing_completed = list(state.get("completed_agents") or [])
    if _mark_complete and agent_name not in existing_completed:
        existing_completed = existing_completed + [agent_name]

    # Unproductive-streak circuit breaker: reset on any real tool call, else grow.
    _prev_streak = state.get("unproductive_streak", 0)
    _new_streak = 0 if _made_tool_call else _prev_streak + 1
    if not _made_tool_call:
        logger.info("[%s] Idle turn (no tool calls) — unproductive_streak=%d", agent_name, _new_streak)

    return {
        "messages": new_messages,
        "knowledge_base": updated_kb,
        "current_agent": agent_name,
        "context_token_estimate": new_estimate,
        "exploit_attempts": state.get("exploit_attempts", 0) + exploit_delta,
        "unproductive_streak": _new_streak,
        "completed_agents": existing_completed,
    }


# ---------------------------------------------------------------------------
# Refusal detection
# ---------------------------------------------------------------------------

REFUSAL_KEYWORDS: list[str] = [
    # Explicit capability denials
    "as an ai", "as a language model", "i cannot", "i can't", "i am unable",
    "i'm unable", "i am not able", "i'm not able", "i don't have the ability",
    "i cannot assist", "i cannot help", "i cannot perform", "i cannot execute",
    "i cannot run", "i cannot access", "i won't", "i will not", "i must decline",
    # Ethical/policy deflections
    "against my", "ethical guidelines", "it would be unethical",
    "against my programming", "designed to", "my purpose",
    # Hand-off / abdication patterns (like the example log)
    "handed over to the human", "hand off to", "handing over",
    "the process is concluded", "process is now handed",
    "i have provided the reconnaissance", "provided the necessary",
    "for the human", "human operator", "human team to proceed",
    "manual exploitation", "leaving this to",
]


def _last_assistant_text_and_tools(messages: list) -> tuple[str, bool]:
    """
    Scan backwards through messages to find the last assistant turn.
    Returns (text_content, had_tool_use).
    """
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        text = ""
        had_tool = False
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    text += block.get("text", "")
                elif block.get("type") == "tool_use":
                    had_tool = True
        return text, had_tool
    return "", False


def _has_genuine_findings(text: str) -> bool:
    """
    Return True if the response contains structured findings that indicate the
    agent did real work and is reporting completion — not refusing to start.
    Used to suppress false-positive refusal detection.
    """
    t = text.lower()
    # Numbered finding lists (e.g. "1.  **Port 22...")
    if re.search(r'^\s*\d+\.\s', text, re.MULTILINE):
        return True
    # Open port references
    if re.search(r'\bport\s+\d+\b|\d+/tcp|\d+/udp', t):
        return True
    # KB / JSON blocks in output
    if any(marker in t for marker in ('```json', '"ports_open"', '"services"', '"open_ports"')):
        return True
    # "Current Findings" / "Knowledge Base Update" headers typical of completion reports
    if any(marker in t for marker in ('current findings', 'knowledge base update', 'next steps')):
        return True
    # CVE / vulnerability research output
    if re.search(r'\bcve-\d{4}-\d+\b', t):
        return True
    if any(marker in t for marker in ('searchsploit', 'tech_stack', 'tech stack', 'x-powered-by', 'server:')):
        return True
    # Explicit completion signal
    if t.lstrip().startswith("task complete"):
        return True
    return False


def lightweight_evaluator(state: TeamState) -> str:
    """
    LangGraph conditional edge function — runs after every agent node.

    Checks the OPENING of the last assistant message (first 300 chars) for:
      (A) Refusal/abdication keywords.
      (B) Substantial reasoning text with zero tool calls.

    Scanning only the opening avoids false positives where an agent legitimately
    reports "I cannot do more" *after* delivering real findings.
    Responses that contain genuine findings are also exempted from (B).

    Returns "refusal_specialist" if either condition is met, else "supervisor".
    """
    messages = state.get("messages", [])
    text, had_tool = _last_assistant_text_and_tools(messages)

    if not text:
        return "supervisor"

    # Evaluate only the opening for keyword matching — a completion report that
    # mentions "I cannot do more" deep in the body is not a refusal.
    opening = text[:300].lower()

    # (A) keyword match in opening — but skip entirely if the agent signalled
    #     explicit completion with "TASK COMPLETE" at the start.
    if opening.lstrip().startswith("task complete"):
        # Require a meaningful body — a bare "TASK COMPLETE" with no findings
        # means the agent is echoing the previous turn without doing any work.
        body = text[len("task complete"):].strip()
        if not had_tool and len(body) < 80:
            logger.info(
                "[Evaluator] Bare TASK COMPLETE (no tools, no body) from %s "
                "— treating as empty turn → refusal_specialist",
                state.get("current_agent", "?"),
            )
            return "refusal_specialist"
        logger.info("[Evaluator] TASK COMPLETE signal from %s → supervisor",
                    state.get("current_agent", "?"))
        return "supervisor"

    if any(kw in opening for kw in REFUSAL_KEYWORDS):
        logger.info("[Evaluator] (A) Refusal keyword in opening of %s response → refusal_specialist",
                    state.get("current_agent", "?"))
        return "refusal_specialist"

    # (B) reasoning without action — skip if this looks like a legitimate completion
    completion_hints = ("flag", "root.txt", "user.txt", "mission complete", "captured", "task complete")
    text_lower = text.lower()
    if (len(text.strip()) > 80
            and not had_tool
            and not any(h in text_lower for h in completion_hints)
            and not _has_genuine_findings(text)):
        logger.info("[Evaluator] (B) Reasoning-without-action in %s response → refusal_specialist",
                    state.get("current_agent", "?"))
        return "refusal_specialist"

    return "supervisor"


# ---------------------------------------------------------------------------
# Refusal Specialist node
# ---------------------------------------------------------------------------

def make_refusal_specialist_node(mcp_client: Any):
    """
    Return an async LangGraph node that uses llama3-abliterated to rewrite
    a refused/incomplete agent response and execute the missing tool call.
    """
    async def refusal_specialist_node(state: TeamState) -> dict:
        config = state.get("config", {})
        rs_cfg = config.get("agents", {}).get("refusal_specialist", {})
        host = rs_cfg.get("host", os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434"))
        model = rs_cfg.get("model", "llama3-abliterated")

        # Which agent just refused, and what was its job?
        # If current_agent is "refusal_specialist" itself, look up the agent that
        # ran before it (the one that originally refused).
        current_agent = state.get("current_agent", "recon")
        if current_agent == "refusal_specialist":
            # Find the last non-specialist agent in history via current_agent transitions.
            # As a safe fallback, use "recon".
            logger.warning("[RefusalSpecialist] Invoked with current_agent=refusal_specialist — "
                           "this is a loop; passing through to supervisor")
            return {"current_agent": "refusal_specialist"}

        agent_cfg = config.get("agents", {}).get(current_agent, {})
        original_system = agent_cfg.get("system_prompt", "")
        kb = state.get("knowledge_base", {})
        task = state.get("task", "")

        # Extract the failed response text
        messages = state.get("messages", [])
        failed_text, _ = _last_assistant_text_and_tools(messages)

        # If there's nothing to correct (e.g. the last message was a tool result),
        # pass through silently rather than running the specialist on an empty prompt.
        if not failed_text.strip():
            logger.warning("[RefusalSpecialist] No assistant text to correct — passing through")
            return {"current_agent": "refusal_specialist"}

        logger.info("[RefusalSpecialist] Correcting refusal from [%s]: %.120s",
                    current_agent, failed_text.replace("\n", " "))

        # Build filtered tool list (same allowlist as the original agent)
        all_tools = await mcp_client.list_tools()
        allowed_patterns: list[str] = agent_cfg.get("tools", [])
        if allowed_patterns:
            raw_tools = [
                t for t in all_tools
                if any(fnmatch.fnmatch(t["name"], pat) for pat in allowed_patterns)
            ]
        else:
            raw_tools = all_tools

        specialist = OllamaClient(host=host, model=model)
        logger.info("[RefusalSpecialist] provider=ollama model=%s tools=%d/%d",
                    model, len(raw_tools), len(all_tools))

        system = (
            f"{original_system}\n\n"
            f"=== KNOWLEDGE BASE ===\n{json.dumps(kb, indent=2)}\n\n"
            f"=== TASK ===\n{task}\n\n"
            f"=== CORRECTION INSTRUCTIONS ===\n"
            f"Examine the previous response carefully before acting:\n"
            f"  • If it is a GENUINE REFUSAL (the agent refused to attempt the task at all, "
            f"gave no findings, and made no tool calls), you MUST call an appropriate tool "
            f"immediately. Do not explain or refuse.\n"
            f"  • If it is a LEGITIMATE COMPLETION (the agent did real work, produced findings "
            f"such as open ports, services, or directory listings, and is simply reporting "
            f"that it has exhausted the available options), respond with the single token "
            f"LEGITIMATE_COMPLETION and make NO tool calls.\n"
            f"Do not conflate the two. A long response with real findings is a completion, "
            f"not a refusal."
        )

        correction_prompt = (
            f"Previous agent response:\n\n\"{failed_text}\"\n\n"
            f"Is this a genuine refusal (agent refused to try) or a legitimate completion "
            f"(agent did work and reported findings)?\n"
            f"If genuine refusal: call the appropriate tool now.\n"
            f"If legitimate completion: respond with LEGITIMATE_COMPLETION only."
        )

        try:
            response = await specialist.generate_response(
                messages=[{"role": "user", "content": correction_prompt}],
                tools=raw_tools,
                system_prompt=system,
            )
        except Exception as exc:
            logger.error("[RefusalSpecialist] LLM error: %s", exc)
            # Append a user-role recovery message so the next agent doesn't
            # receive a conversation ending in an assistant turn.
            recovery = {"role": "user", "content": "[Refusal specialist unavailable — continue with best effort.]"}
            return {"messages": [recovery], "current_agent": "refusal_specialist"}

        tool_calls = specialist.parse_tool_calls(response)
        new_messages: list = [specialist.make_assistant_message(response)]

        # Extract text from the assistant message (provider-agnostic)
        _asst_content = new_messages[0].get("content", "") if new_messages else ""
        response_text = ""
        if isinstance(_asst_content, str):
            response_text = _asst_content
        elif isinstance(_asst_content, list):
            for _blk in _asst_content:
                if isinstance(_blk, dict) and _blk.get("type") == "text":
                    response_text += _blk.get("text", "")

        # Check if specialist determined this was a legitimate completion —
        # either as text OR as a fake "LEGITIMATE_COMPLETION" tool call (some
        # abliterated models treat it as a tool name instead of a token).
        _legit_tool_call = any(
            tc.get("name", "").replace(" ", "_").lower() in (
                "legitimate_completion", "legitimate_completion_signal"
            )
            for tc in tool_calls
        )
        if "legitimate_completion" in response_text.lower() or _legit_tool_call:
            logger.info("[RefusalSpecialist] Determined response was a legitimate completion — passing through")
            return {"messages": new_messages, "current_agent": "refusal_specialist"}

        # Strip any remaining fake LEGITIMATE_COMPLETION tool calls so they
        # don't reach mcp_client.call_tool() and produce "Unknown tool" errors.
        tool_calls = [
            tc for tc in tool_calls
            if tc.get("name", "").replace(" ", "_").lower() != "legitimate_completion"
        ]

        if not tool_calls:
            logger.warning("[RefusalSpecialist] Specialist produced no tool calls — routing to supervisor")
            return {"messages": new_messages, "current_agent": "refusal_specialist"}

        # Execute the tool call(s) and collect results
        updated_kb = dict(kb)
        for tc in tool_calls:
            tool_name = tc.get("name", "")
            raw_args = tc.get("arguments") or tc.get("input", {})
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    raw_args = {}

            result = await mcp_client.call_tool(tool_name, raw_args)
            updated_kb = _extract_kb_updates(tool_name, result, updated_kb, raw_args)
            new_messages.append(specialist.make_tool_result_message(tool_name, result))

        new_estimate = _estimate_tokens(list(messages) + new_messages)

        return {
            "messages": new_messages,
            "knowledge_base": updated_kb,
            "current_agent": "refusal_specialist",
            "context_token_estimate": new_estimate,
        }

    return refusal_specialist_node


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


def make_vulnsearch_node(mcp_client: Any, tools: list):
    """Return an async node function for the Vulnerability Search agent."""
    async def vulnsearch_node(state: TeamState) -> dict:
        logger.info("[VulnSearch Agent] Researching vulnerabilities in tech stack...")
        return await _run_agent_loop("vulnsearch", state, tools, mcp_client)
    return vulnsearch_node


def make_reversing_node(mcp_client: Any, tools: list):
    """Return an async node function for the Reversing / binary-analysis agent.

    Specialised for file-based rev/pwn challenges. Its power comes from the
    disassembly completion-gate in _run_agent_loop (agent_name == "reversing"):
    it may not finish until it has actually run a disassembler/debugger, which
    breaks the observed "file/strings then quit" pattern.
    """
    async def reversing_node(state: TeamState) -> dict:
        logger.info("[Reversing Agent] Disassembling and analysing the binary...")
        return await _run_agent_loop("reversing", state, tools, mcp_client)
    return reversing_node
