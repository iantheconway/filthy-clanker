# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Filthy-Clanker is an AI-powered CTF solver that connects LLMs (Claude or Gemini) to 130+ security tools via the Model Context Protocol (MCP). It uses a **LangGraph multi-agent architecture** with a Supervisor, three specialized agents (Recon, Exploit, PrivEsc), and an Ollama-powered summarizer for log distillation.

## Running

```bash
source venv/bin/activate
python src/main.py
```

Requires a `.env` file (copy from `.env.example`) with at least one API key (`ANTHROPIC_API_KEY` or `GEMINI_API_KEY`). The Hexstrike-AI repo must be cloned to `/home/kali/hexstrike-ai` (or set `HEXSTRIKE_DIR`).

On startup: starts/detects the Hexstrike Flask server, prompts for LLM provider, builds the LangGraph StateGraph, enters autonomous agent loop with HITL support.

## Dependencies

```bash
pip install -r requirements.txt
```

Key packages: `anthropic`, `google-genai`, `mcp`, `langgraph`, `langchain-core`, `langgraph-checkpoint-sqlite`, `pyyaml`, `aiosqlite`.

Python 3.13+ required. No test suite or linter configured.

## Architecture

```
User → main.py → LangGraph StateGraph (SqliteSaver checkpoint)
                       │
                   supervisor ──► recon ──────────────────┐
                       ▲          exploit ───────────────►│
                       │          privesc ───────────────►│
                       │          compaction (Ollama) ───►│
                       └──────────────────────────────────┘
                                 (loop until flag found or FINISH)

Each agent: LLM client → MCP tool loop → knowledge_base update → supervisor
```

### StateGraph Flow (`src/graph/graph.py`)

1. `START → supervisor` — always begins at the supervisor
2. `supervisor → {recon, exploit, privesc, compaction, END}` — conditional routing via `route_from_supervisor()`
3. All agent nodes → `supervisor` — unconditional return after sub-task completion
4. `compaction → supervisor` — after compressing message history via Ollama

### Key Design Points

- **TeamState** (`src/graph/state.py`): Two-tier state separating `messages` (task history with append reducer) from `knowledge_base` (structured facts: IPs, ports, creds, flags, attack surface). Both are shared across all agents.

- **agents.yaml**: External config file for all agent definitions (model, provider, system_prompt) and global settings (thresholds, paths). Edit this to change agent behavior without touching source code.

- **Supervisor** (`src/graph/supervisor.py`): Calls the LLM to route between agents. Uses LangGraph `interrupt()` for HITL breakpoints when exploit attempts exceed `max_exploit_attempts` or when `hitl_reason` is set.

- **Specialized agents** (`src/graph/agents.py`): Each agent runs a ReAct-style tool loop internally until it has no more tool calls, then returns state updates. The Exploit agent tracks consecutive failures in `exploit_attempts`.

- **Summarizer** (`src/graph/summarizer.py`): Two roles:
  - `maybe_summarize(text, config)` — inline; condenses tool output exceeding `tool_output_threshold` chars via Ollama before passing to the main LLM.
  - `compaction_node(state)` — LangGraph node; compresses bulk message history when `context_token_estimate > context_limit_threshold`, replacing old messages with a summary and resetting the token counter.

- **LangChain tool wrappers** (`src/graph/tools.py`): `build_langchain_tools(mcp_client)` creates one `MCPTool(BaseTool)` instance per MCP tool, with dynamically-generated Pydantic args schemas from the MCP inputSchema. Usable standalone in LangChain agents.

- **SqliteSaver checkpointing**: Every node execution is checkpointed. Sessions are identified by `thread_id` (= `session_id` in state). Resume by passing the same `thread_id` to `graph.get_state()` or by selecting a previous session at startup.

- **LLM abstraction**: `src/llms/base.py` defines `BaseLLMClient` ABC. Three implementations: `AnthropicClient`, `GeminiClient`, `OllamaClient`. Agent nodes instantiate these from `agents.yaml` config per-agent, so different agents can use different providers.

- **Provider handover**: When `context_token_estimate > context_limit_threshold`, the supervisor routes to the `compaction` node before the context window fills. The `provider` field in TeamState can be updated to switch LLMs after compaction.

## Data / Training Directory

```
data/
└── training/
    ├── .gitkeep
    ├── session-YYYY-MM-DD-HHMMSS-<uuid>.jsonl   ← per-session trajectories
    └── all_trajectories.jsonl                   ← cumulative across all sessions
```

Each JSONL line is a trajectory record (`src/data_capture.py`):
```json
{
  "id": "<uuid>",
  "timestamp": "2026-04-02T...",
  "session_id": "session-...",
  "agent": "recon",
  "task": "Hack the machine at 10.10.11.100",
  "action": {"tool_name": "nmap", "arguments": {"target": "10.10.11.100"}},
  "result_snippet": "...",
  "result_length": 4821,
  "knowledge_base_before": {...},
  "knowledge_base_after": {...},
  "success_score": 0.55,
  "exploit_attempts": 0
}
```

**`success_score` heuristic (0.0–1.0):**
| Score | Condition |
|-------|-----------|
| 1.0   | Flag captured |
| 0.85  | New credentials discovered |
| 0.70  | New attack surface (exploitable path) |
| 0.55  | New open port/service |
| 0.40  | New IP discovered |
| 0.20  | Successful directory/file discovery |
| 0.05  | Tool ran, no new findings |
| 0.0   | Tool error or explicit failure |

## Sessions

Graph sessions are persisted in `sessions/checkpoint.db` (SQLite). Select "Resume" at startup and enter the session ID to continue from the last checkpoint. The session ID is printed at startup and shown in the session list.

## In-Session Controls

- `Ctrl+C` — interrupt agent loop, choose to continue / inject message / exit
- HITL breakpoints — automatic pause when exploit loop detected or credentials needed
- HTB commands (`/spawn`, `/stop`, `/reset`, `/flag`, `/vpn`) — available at startup prompt
