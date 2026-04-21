# Filthy-Clanker

An AI-powered CTF solver that connects LLMs to 130+ security tools via the Model Context Protocol (MCP). It uses a **LangGraph multi-agent architecture** where a Supervisor orchestrates six specialized agents, each with its own model, tool allowlist, and system prompt configured in `agents.yaml`.

## Architecture

```
User → main.py → LangGraph StateGraph (SqliteSaver checkpoint)
                        │
                    supervisor
                    ├── [deterministic checks, no LLM]
                    │     • flag captured?        → END
                    │     • context limit?        → compaction
                    │     • exploit loop (≥N)?    → HITL interrupt
                    │     • HTTP port in KB?      → webexplorer
                    │     • tech_stack + vulns?   → exploit
                    └── [LLM routing decision]
                          • recon
                          • webexplorer
                          • vulnsearch
                          • exploit
                          • privesc
                          • refusal_specialist
                          • FINISH → END

Each agent node:
  LLM (provider/model per agents.yaml)
    └── ReAct tool loop (MCP tools filtered by per-agent allowlist)
          └── lightweight_evaluator (post-run refusal/completion check)
                ├── no refusal   → supervisor
                └── refusal      → refusal_specialist → supervisor
```

### Agents

| Agent | Role | Default Model |
|-------|------|---------------|
| **supervisor** | Orchestrates routing; deterministic checks first, then LLM | claude-sonnet-4-6 |
| **recon** | Port scanning, service enumeration, DNS, SMB, web discovery | claude-haiku-4-5 |
| **webexplorer** | Browses web apps — reads pages, follows links, extracts JS secrets, maps forms | claude-haiku-4-5 |
| **vulnsearch** | Searches CVEs and exploits for discovered tech stack entries | claude-sonnet-4-6 |
| **exploit** | Web vulns (SQLi/SSTI/LFI), service exploitation, brute force, cracking | claude-opus-4-6 |
| **privesc** | SUID/sudo/cron/kernel escalation, credential dumping, flag capture | claude-sonnet-4-6 |
| **refusal_specialist** | Rewrites refused responses and executes the missing tool call | ollama (abliterated) |
| **compaction** | Ollama-based context summarisation when token limit approaches | llama3.2 |

### Supervisor Routing

The supervisor runs deterministic checks before making any LLM call, short-circuiting to the correct agent without spending tokens:

1. Flag in KB → `END`
2. Context estimate exceeds threshold → `compaction`
3. Exploit attempt counter hit → `HITL interrupt`
4. HTTP port found and webexplorer not yet run → `webexplorer`
5. Tech stack known + vulnsearch done + no shells → `exploit`
6. LLM routing decision for everything else

### Refusal Detection

After every agent run, the `lightweight_evaluator` inspects the opening 300 characters of the last assistant message for:

- **(A) Refusal keywords** — "I cannot perform", "I am unable to", "handed over to the human", etc. Agents are instructed to use these exact phrases when declining.
- **(B) Reasoning without action** — substantial text response with no tool calls and no genuine findings.

If either fires, the graph routes to `refusal_specialist` (running an abliterated local model), which distinguishes genuine refusals from legitimate completions and either executes the missing tool call or passes through.

### Provider Configuration

Each agent has its own `provider` and `model` in `agents.yaml`. Default config uses Anthropic Claude models sized to task complexity — Opus for exploit, Sonnet for routing/analysis, Haiku for fast enumeration tasks, and Ollama for the refusal specialist. At startup you can override all agents to a single provider without editing the YAML.

### Shared State (`TeamState`)

| Field | Description |
|-------|-------------|
| `messages` | Full conversation history, appended across all agents |
| `knowledge_base` | Structured facts: IPs, open ports, services, tech stack, credentials, flags, attack surface, scan history |
| `completed_agents` | Agents that have signalled substantive completion (exploit only added if flag captured) |
| `exploit_attempts` | Counter for consecutive exploit failures |
| `context_token_estimate` | Rolling token count for compaction trigger |
| `hitl_reason` | Set by any agent to pause for human input |

### Tool Allowlists

Each agent in `agents.yaml` has a `tools:` list of glob patterns matched against MCP tool names (e.g. `"nmap*"`, `"sqlmap*"`). Agents only see their relevant subset of the 130+ available tools, reducing prompt size and keeping models focused.

### Flag Capture

Flags are extracted from tool output and agent text automatically:

- **Bracket format**: `HTB{...}`, `FLAG{...}`, `CTF{...}`
- **Raw hex format**: 32-character hex strings (standard HTB `user.txt`/`root.txt` format), detected when the tool result or arguments reference a flag file path

Captured flags are written to `knowledge_base["flags"]`. The supervisor will not route to `END` unless at least one flag is present; if an agent signals completion without a flag, an HITL breakpoint fires.

## Prerequisites

- Python 3.13+
- [Hexstrike-AI](https://github.com/your-org/hexstrike-ai) cloned to `/home/kali/hexstrike-ai` (or set `HEXSTRIKE_DIR` in `.env`)
- At least one of: Anthropic API key, Gemini API key, or local Ollama instance
- Ollama required for the refusal specialist and compaction node regardless of main provider

## Installation

```bash
git clone <repo-url> filthy-clanker
cd filthy-clanker

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
# Edit .env with your keys
```

## Configuration

### `.env`

```env
ANTHROPIC_API_KEY=your-anthropic-key
GEMINI_API_KEY=your-gemini-key

# HackTheBox integration (optional)
HTB_APP_TOKEN=your-htb-app-token

# Hexstrike server location
HEXSTRIKE_DIR=/home/kali/hexstrike-ai
HEXSTRIKE_PORT=8888

# Brave Search MCP (optional — enables web search tools)
# BRAVE_API_KEY=your-brave-api-key

# LangSmith tracing (optional)
# LANGSMITH_TRACING=true
# LANGSMITH_API_KEY=your-langsmith-key
# LANGSMITH_PROJECT=filthy-clanker
```

### `agents.yaml`

All agent behaviour is configured here — no source code changes needed:

```yaml
agents:
  recon:
    provider: anthropic
    model: claude-haiku-4-5-20251001
    tools:            # glob patterns — agent only sees matching MCP tools
      - "nmap*"
      - "gobuster*"
      - ...
    system_prompt: |
      ...

  refusal_specialist:
    provider: ollama
    model: huihui_ai/qwen3.5-abliterated:9b
    host: "http://10.0.2.2:11434"

settings:
  tool_output_threshold: 4000    # chars before auto-summarisation
  context_limit_threshold: 80000 # tokens before compaction
  max_exploit_attempts: 5        # failures before HITL interrupt
  checkpoint_db: "sessions/checkpoint.db"
  training_data_dir: "data/training"
```

## Usage

```bash
source venv/bin/activate
python src/main.py
```

On startup:

1. Connects to HackTheBox (if `HTB_APP_TOKEN` is set) and detects any active machine
2. Starts the Hexstrike Flask server (or detects it if already running)
3. Prompts for LLM configuration — use per-agent config from `agents.yaml` or override all agents with one provider
4. Connects to the MCP server and lists available tools
5. Prompts to start a new session or resume a previous one
6. Enters the autonomous agent loop

Example startup:

```
[*] HTB client connected.
[*] Active machine: Knife (Linux, Easy) @ 10.10.11.100
[*] Hexstrike server is ready.

LLM provider configuration:
  1) Use per-agent config from agents.yaml
       supervisor:   anthropic / claude-sonnet-4-6
       recon:        anthropic / claude-haiku-4-5-20251001
       exploit:      anthropic / claude-opus-4-6
       ...
  2) Override all agents — Anthropic (Claude)
  3) Override all agents — Gemini
  4) Override all agents — Ollama (local)
Enter 1–4 [1]: 1

[*] MCP session initialized.
[*] 150 MCP tools available.
[*] Agent log: logs/session-2026-04-20-120000-abc12345.log
[*] Running autonomous agents. Press Ctrl+C to interrupt.

12:00:01  [recon] provider=anthropic model=claude-haiku-4-5-20251001
12:00:23  [MCP] CALL  nmap_scan({"target": "10.10.11.100", "arguments": "-sC -sV"})
12:00:45  [MCP] RESULT nmap_scan — 3821 chars: PORT   STATE SERVICE ...
12:00:45  [Supervisor] HTTP service(s) detected on ['10.10.11.100:80'] → webexplorer
```

## Session Management

Sessions are checkpointed to `sessions/checkpoint.db` (SQLite via LangGraph's `AsyncSqliteSaver`). Select **Resume** at startup and enter a session number to continue from the last checkpoint — the full message history and knowledge base are restored automatically.

When resuming, you can optionally inject a new task message to redirect the agents without losing prior findings.

## LangSmith Tracing

Set `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, and `LANGSMITH_PROJECT` in `.env` to stream traces to [smith.langchain.com](https://smith.langchain.com). Every LLM call and MCP tool invocation is traced as a child span under the LangGraph node that triggered it.

## Agent Logs

Every session writes a structured log to `logs/<session-id>.log`:

- Every MCP tool call: name, arguments, result (first 300 chars), error if any
- Every LLM API call: provider, model, message count, system prompt snippet, response text, tool calls selected
- Supervisor routing decisions and reasoning
- Refusal detections, specialist invocations, and loop guard triggers
- Context compaction events

```bash
tail -f logs/session-2026-04-20-120000-abc12345.log
```

## In-Session Controls

| Control | Action |
|---------|--------|
| `Ctrl+C` | Interrupt — choose to continue, inject a message, or exit |
| HITL breakpoints | Auto-pause when exploit loop threshold is hit, `hitl_reason` is set, or FINISH is called without a captured flag |

## HackTheBox Integration

When `HTB_APP_TOKEN` is set, machine management commands are available at the pre-run prompt:

| Command | Description |
|---------|-------------|
| `/machine` | Show the currently active machine |
| `/machine <name>` | Look up any machine |
| `/spawn <name>` | Spawn a machine and auto-set the task |
| `/stop` | Stop the active machine |
| `/reset` | Reset the active machine |
| `/flag <flag>` | Submit a flag |
| `/vpn` | Show VPN status |

## Training Data

Each session appends tool call trajectories to `data/training/` as JSONL. Each record captures:

- Agent name, tool called, arguments, result snippet
- Knowledge base state before and after the tool call
- `success_score` heuristic (0.0–1.0) based on what was discovered

| Score | Condition |
|-------|-----------|
| 1.0 | Flag captured |
| 0.85 | New credentials discovered |
| 0.70 | New exploitable attack surface |
| 0.55 | New open port/service |
| 0.20 | Directory/file discovery |
| 0.05 | Tool ran, no new findings |
| 0.0 | Tool error or explicit failure |

Records accumulate in `data/training/all_trajectories.jsonl` across all sessions.

## Project Structure

```
src/
├── main.py               # Entry point: server management, provider selection, session loop
├── config.py             # System prompt builder
├── htb_client.py         # HackTheBox API wrapper
├── data_capture.py       # Trajectory JSONL logging
├── studio.py             # LangSmith Studio graph entry point (stub MCP, no checkpointer)
├── llms/
│   ├── base.py           # Abstract LLM client interface
│   ├── anthropic_client.py  # Adaptive thinking, @traceable spans
│   ├── gemini_client.py
│   └── ollama_client.py  # Sync requests in executor, no read timeout
├── mcp_client/
│   └── client.py         # MCP stdio client pool — logs and traces every tool call
└── graph/
    ├── graph.py          # StateGraph assembly — nodes, edges, conditional routing
    ├── state.py          # TeamState and KnowledgeBase TypedDicts
    ├── supervisor.py     # Supervisor node: deterministic checks + LLM routing
    ├── agents.py         # All agent nodes + lightweight_evaluator + refusal_specialist
    ├── summarizer.py     # Ollama compaction node + inline tool output summariser
    └── tools.py          # LangChain MCP tool wrappers (for standalone use)
agents.yaml               # All agent config: models, providers, prompts, tool allowlists
langgraph.json            # LangSmith Studio graph discovery config
logs/                     # Per-session structured logs
sessions/                 # SQLite checkpoint database
data/training/            # Trajectory JSONL files
```
