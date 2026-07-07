# Filthy-Clanker

An AI-powered CTF (Capture The Flag) challenge solver that connects LLMs to 130+ security tools via the Model Context Protocol (MCP). It uses a **LangGraph multi-agent architecture** — a Supervisor routes work between three specialized agents (Recon, Exploit, PrivEsc), an Ollama-powered summarizer distills large tool outputs, and every step is checkpointed to SQLite so sessions can be paused, resumed, and steered by a human operator.

## Architecture

```
User → main.py → LangGraph StateGraph (AsyncSqliteSaver checkpoint)
                       │
                   supervisor ──► recon ──────────────────┐
                       ▲          exploit ───────────────►│
                       │          privesc ───────────────►│
                       │          compaction (Ollama) ───►│
                       └──────────────────────────────────┘
                            (loop until flag found or FINISH)

Each agent: LLM client → MCP tool loop → knowledge_base update → supervisor
                                    │
                            Hexstrike MCP server → Flask server → nmap/gobuster/…
```

The Supervisor decides which agent runs next based on the shared **knowledge base**
(IPs, ports, services, credentials, flags, attack surface). It triggers a
human-in-the-loop (HITL) breakpoint when the exploit agent stalls, and routes to the
compaction node before the context window fills. See [CLAUDE.md](CLAUDE.md) for a full
architecture reference.

## Prerequisites

- Python 3.12+
- [Hexstrike-AI](https://github.com/your-org/hexstrike-ai) cloned to `/home/kali/hexstrike-ai` (or set `HEXSTRIKE_DIR`)
- An API key for at least one LLM provider:
  - [Anthropic](https://console.anthropic.com/) (Claude)
  - [Google AI Studio](https://aistudio.google.com/) (Gemini)
  - [Ollama](https://ollama.com/) running locally (used for the summarizer, and optionally as a primary provider)

## Installation

```bash
git clone <repo-url> filthy-clanker
cd filthy-clanker

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

# Optional — HackTheBox commands (/spawn, /flag, …). pyhackthebox pins an old
# `requests`, so install it without its dependency pins:
pip install --no-deps pyhackthebox
```

## Configuration

Copy the example env file and fill in your keys:

```bash
cp .env.example .env
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

# Optional MCP overrides (derived from the above by default)
# MCP_COMMAND=/home/kali/hexstrike-ai/hexstrike-env/bin/python3
# MCP_ARGS=/home/kali/hexstrike-ai/hexstrike_mcp.py --server http://127.0.0.1:8888
```

To get an HTB app token, go to your [HackTheBox profile settings](https://app.hackthebox.com/profile/settings) and create one under "App Tokens".

## Usage

```bash
source venv/bin/activate
python src/main.py
```

On startup the program will:

1. Connect to HackTheBox (if `HTB_APP_TOKEN` is set) and detect any active machine
2. Start the Hexstrike Flask server (or detect it if already running)
3. Prompt you to select a primary LLM provider (Anthropic, Gemini, or Ollama)
4. Connect to the Hexstrike MCP server and list available tools
5. Build the LangGraph StateGraph with SQLite checkpointing
6. Offer a **new session** or to **resume** a previous one from its last checkpoint
7. Run the autonomous agent loop, pausing for human input at HITL breakpoints

Example session:

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

[*] Session ID: session-2026-07-06-...
[*] Running autonomous agents. Press Ctrl+C to interrupt.

[Supervisor] → RECON | need to enumerate the target
  [recon] → nmap_scan({"target": "10.10.11.8"})
[Supervisor] → EXPLOIT | web service on 5000 looks promising
  [exploit] → ...
```

While running:

- **`Ctrl+C`** — interrupt the loop; choose to continue, inject a message, or exit
- **HITL breakpoints** — the run pauses automatically when the exploit agent stalls
  (too many consecutive failures) or credentials are needed; type a hint to resume
- On exit, state is checkpointed to `sessions/checkpoint.db` and the Hexstrike server
  is shut down. Re-run and pick **Resume** to continue from the last checkpoint.

## HackTheBox Integration

When `HTB_APP_TOKEN` is set, you get direct access to HTB machine management from within the chat. The system prompt automatically adapts to the active machine, so the AI knows the target name, OS, difficulty, and IP without you having to type it.

### HTB Commands

| Command | Description |
|---------|-------------|
| `/machine` | Show the currently active machine |
| `/machine <name>` | Look up details for any machine |
| `/spawn <name>` | Spawn a machine and update the AI's context |
| `/stop` | Stop the active machine |
| `/reset` | Reset the active machine (~1 min) |
| `/flag <flag>` | Submit a flag for the active machine |
| `/vpn` | Show current VPN server info |

### Example: Full workflow

```
You: /spawn Headless
[*] Spawning 'Headless'...
[*] Machine spawned at 10.10.11.8

You: let's go
Assistant: Understood — now targeting Headless at 10.10.11.8. I'll start with a quick
nmap scan to see what's open...

    ... hacking happens ...

You: /flag 8a3f5b2c1d4e6f7a8b9c0d1e2f3a4b5c
[*] Congratulations! Machine owned!

You: /stop
[*] Stopped machine 'Headless'.
```

### How it works

- On startup, the client checks for an already-running machine and auto-populates the system prompt.
- `/spawn` updates the system prompt and injects a context message into the conversation so the AI immediately knows the new target.
- `/stop` resets the prompt back to a generic one.
- Without `HTB_APP_TOKEN`, everything works as before — you just won't have the HTB commands, and the system prompt will be generic.

## Project Structure

```
agents.yaml              # Agent definitions (model/provider/prompt) + global settings
src/
├── main.py              # Entry point, server management, graph session loop, HITL
├── config.py            # System prompt helper
├── htb_client.py        # HackTheBox API wrapper
├── session.py           # Legacy JSON session save/resume helpers
├── data_capture.py      # Trajectory (State, Action, Result) logging for fine-tuning
├── graph/
│   ├── graph.py         # StateGraph assembly + AsyncSqliteSaver checkpointer
│   ├── state.py         # TeamState + KnowledgeBase types
│   ├── supervisor.py    # Routing brain + HITL interrupts
│   ├── agents.py        # Recon / Exploit / PrivEsc ReAct tool loops
│   ├── summarizer.py    # Ollama tool-output condensing + context compaction node
│   └── tools.py         # LangChain-compatible MCP tool wrappers
├── llms/
│   ├── base.py          # Abstract LLM client interface
│   ├── anthropic_client.py  # Claude integration
│   ├── gemini_client.py     # Gemini integration
│   └── ollama_client.py     # Ollama integration
└── mcp_client/
    └── client.py        # MCP protocol client
```

## Testing

```bash
pip install pytest pytest-asyncio
pytest
```

The suite covers the LLM client message-shaping, the session helpers, and the
multi-agent graph wiring (tool dispatch, knowledge-base extraction, exploit-loop
detection, supervisor routing, and HITL interrupt/resume).
