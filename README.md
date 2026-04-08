# Filthy-Clanker

An AI-powered CTF solver that connects LLMs (Claude, Gemini, or Ollama) to 130+ security tools via the Model Context Protocol (MCP). It uses a **LangGraph multi-agent architecture** with a Supervisor orchestrating three specialized agents (Recon, Exploit, PrivEsc), plus an Ollama-powered compaction node for managing context windows.

## Architecture

```
User → main.py → LangGraph StateGraph (SqliteSaver checkpoint)
                       │
                   supervisor ──► recon    ─────────────────┐
                       ▲          exploit  ────────────────►│
                       │          privesc  ────────────────►│
                       │          compaction (Ollama) ──────►│
                       └──────────────────────────────────────┘
                              (loop until flag found or FINISH)

Each agent: LLM client → MCP tool loop → knowledge_base update → supervisor
```

The Supervisor calls the LLM to decide which agent runs next. Each specialized agent runs a ReAct-style tool loop internally, executing MCP tools until the LLM produces a final text response, then returns state updates back to the Supervisor.

**Key state**: `TeamState` has two tiers — `messages` (full task history) and `knowledge_base` (structured facts: IPs, ports, credentials, flags, attack surface). Both are shared across all agents.

## Prerequisites

- Python 3.13+
- [Hexstrike-AI](https://github.com/your-org/hexstrike-ai) cloned to `/home/kali/hexstrike-ai` (or set `HEXSTRIKE_DIR`)
- At least one LLM API key (Anthropic or Google Gemini), or a local Ollama instance

## Installation

```bash
git clone <repo-url> filthy-clanker
cd filthy-clanker

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Configuration

```bash
cp .env.example .env
# Edit .env with your keys
```

`.env` contents:

```env
# At least one API key is required
ANTHROPIC_API_KEY=your-anthropic-key
GEMINI_API_KEY=your-gemini-key

# HackTheBox integration (optional — enables /spawn, /flag, etc.)
HTB_APP_TOKEN=your-htb-app-token

# Hexstrike configuration (defaults shown)
HEXSTRIKE_DIR=/home/kali/hexstrike-ai
HEXSTRIKE_PORT=8888

# Ollama (optional — required if you select Ollama as provider)
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2

# Optional MCP overrides (derived from the above by default)
# MCP_COMMAND=/home/kali/hexstrike-ai/hexstrike-env/bin/python3
# MCP_ARGS=/home/kali/hexstrike-ai/hexstrike_mcp.py --server http://127.0.0.1:8888
```

Agent behaviour (models, system prompts, thresholds) is configured in **`agents.yaml`** — edit this without touching source code.

## Usage

```bash
source venv/bin/activate
python src/main.py
```

On startup:

1. Connects to HackTheBox (if `HTB_APP_TOKEN` is set) and detects any active machine
2. Starts the Hexstrike Flask server (or detects it if already running)
3. Prompts for LLM provider (Anthropic / Gemini / Ollama)
4. Connects to the MCP server and loads all available tools
5. Prompts to start a new session or resume a previous one
6. Enters the autonomous agent loop

Example startup:

```
[*] HTB client connected.
[*] Active machine: Headless (Linux, Easy) @ 10.10.11.8
[*] Hexstrike server is ready.

Select primary LLM provider:
  1) Anthropic (Claude)
  2) Gemini
  3) Ollama (local)
Enter 1, 2, or 3: 1

[*] MCP session initialized.
[*] 134 MCP tools available.
[*] Session ID: session-2026-04-08-120000-abc12345
[*] Agent log: logs/session-2026-04-08-120000-abc12345.log
[*] Running autonomous agents. Press Ctrl+C to interrupt.

[Recon Agent] Starting reconnaissance...
[recon] → nmap_scan({"target": "10.10.11.8", "arguments": "-sC -sV -p-"})
[Supervisor] → EXPLOIT | Open port 5000 HTTP found, attempting web exploitation
[Exploit Agent] Attempting exploitation...
```

## Session Management

Sessions are checkpointed to `sessions/checkpoint.db` (SQLite via LangGraph's `SqliteSaver`). Select **Resume** at startup and enter a session number to continue from the last checkpoint — the full message history and knowledge base are restored.

## Agent Logs

Every session writes a timestamped log to `logs/<session-id>.log`. This captures all agent activity (tool calls, supervisor routing decisions, summarization events, flag discoveries) even if terminal output is lost. Tail it in another terminal:

```bash
tail -f logs/session-2026-04-08-120000-abc12345.log
```

## In-Session Controls

| Control | Action |
|---------|--------|
| `Ctrl+C` | Interrupt — choose to continue, inject a message, or exit |
| HITL breakpoints | Auto-pause when exploit loop detected or credentials needed |

## HackTheBox Integration

When `HTB_APP_TOKEN` is set, machine management commands are available before the agent loop starts:

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

Each session logs tool call trajectories to `data/training/` as JSONL files. Each record captures the tool called, arguments, result snippet, knowledge base before/after, and a `success_score` heuristic (0.0–1.0). Records accumulate in `data/training/all_trajectories.jsonl` across all sessions.

## Project Structure

```
src/
├── main.py              # Entry point, server management, session loop
├── config.py            # System prompt builder
├── htb_client.py        # HackTheBox API wrapper
├── data_capture.py      # Trajectory logging
├── llms/
│   ├── base.py          # Abstract LLM client interface
│   ├── anthropic_client.py
│   ├── gemini_client.py
│   └── ollama_client.py
├── mcp_client/
│   └── client.py        # MCP stdio client
└── graph/
    ├── graph.py         # LangGraph StateGraph builder
    ├── state.py         # TeamState / KnowledgeBase types
    ├── supervisor.py    # Supervisor node + routing
    ├── agents.py        # Recon / Exploit / PrivEsc nodes
    ├── summarizer.py    # Ollama-based compaction
    └── tools.py         # LangChain MCP tool wrappers
agents.yaml              # Agent config (models, prompts, thresholds)
logs/                    # Per-session agent activity logs
sessions/                # SQLite checkpoint DB
data/training/           # Trajectory JSONL files
```
